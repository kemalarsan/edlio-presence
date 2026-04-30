"""
renderer/audio_prep.py — Audio → AudioFeatures conversion utilities

Bridges raw audio (WAV files, PCM arrays) to the AudioFeatures format
that engine.py consumes.

Depends on:
  - librosa (CPU-safe)
  - transformers (for Whisper feature extractor)
  - torch (for tensor ops; falls back to numpy-only in stub mode)

In GPU mode, the Whisper encoder runs on the GPU.
In stub mode (CPU / no GPU), we return dummy zero-filled features so the
stub renderer has something to consume.

Track C interface:
  If Track C ships a phoneme extractor before this module evolves, plug it in here:
  1. Call your phoneme extractor on the same audio
  2. Populate AudioFeatures.phonemes with the timeline
  3. Pass the full AudioFeatures through — MuseTalk ignores phonemes for now,
     but Track G and future renderers will use them.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterator, Optional

import numpy as np

from renderer.engine import AudioFeatures

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Whisper feature shapes (MuseTalk v1.5 constants)
# ──────────────────────────────────────────────────────────────────────────────

SAMPLE_RATE = 16_000  # Hz — MuseTalk requires 16kHz mono
FPS = 25  # output video frames per second
WHISPER_DIM = 384  # whisper-tiny hidden-state dimension
WHISPER_FRAMES_PER_VIDEO_FRAME = 50  # (audio_padding_left + 1 + audio_padding_right) * 2 * 5


def load_wav(wav_path: str | Path) -> np.ndarray:
    """
    Load a WAV file as 16kHz mono float32 PCM.

    Returns
    -------
    np.ndarray, shape (N,), dtype float32
    """
    try:
        import librosa  # noqa: PLC0415
        audio, sr = librosa.load(str(wav_path), sr=SAMPLE_RATE, mono=True)
        logger.debug("Loaded %s: %d samples @ %d Hz", wav_path, len(audio), sr)
        return audio.astype(np.float32)
    except ImportError:
        # librosa not available — read raw PCM via scipy as fallback
        from scipy.io import wavfile  # noqa: PLC0415
        sr, audio = wavfile.read(str(wav_path))
        if sr != SAMPLE_RATE:
            raise ValueError(f"WAV must be 16kHz; got {sr}Hz. Resample first.")
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32) / 32768.0
        return audio


def pcm_to_features(
    pcm: np.ndarray,
    model_dir: Optional[str] = None,
    device: str = "cpu",
    use_stub: bool = False,
    fps: float = FPS,
) -> AudioFeatures:
    """
    Convert raw 16kHz mono PCM to AudioFeatures (Whisper embeddings).

    Parameters
    ----------
    pcm : np.ndarray, shape (N,), float32
        Raw audio PCM at 16kHz.
    model_dir : str | None
        Path to weights dir containing whisper/ subdirectory.
        Defaults to /models (Docker container path).
    device : str
        'cuda' for GPU inference, 'cpu' for CPU (slow).
    use_stub : bool
        If True, return dummy zero features (no Whisper needed).
    fps : float
        Target video FPS. Affects how many feature frames are produced.

    Returns
    -------
    AudioFeatures
        .whisper_features: (T, 50, 384) float32 numpy array
        .audio_pcm: the raw PCM (stored for debugging / phoneme extraction)
    """
    if use_stub:
        return _stub_features(pcm, fps=fps)

    try:
        import torch  # noqa: PLC0415
    except ImportError:
        logger.warning("torch not available — falling back to stub features")
        return _stub_features(pcm, fps=fps)

    from pathlib import Path as _Path  # noqa: PLC0415
    import sys  # noqa: PLC0415

    # Make musetalk importable
    vendor_dir = str(_Path(__file__).parent / "muse_talk_vendor")
    if vendor_dir not in sys.path:
        sys.path.insert(0, vendor_dir)

    from musetalk.utils.audio_processor import AudioProcessor  # noqa: PLC0415
    from transformers import WhisperModel  # noqa: PLC0415

    weights_dir = _Path(model_dir) if model_dir else _Path("/models")
    whisper_path = str(weights_dir / "whisper")

    processor = AudioProcessor(feature_extractor_path=whisper_path)
    whisper_model = WhisperModel.from_pretrained(whisper_path)
    whisper_model = whisper_model.to(device).eval()

    weight_dtype = torch.float16 if device == "cuda" else torch.float32
    whisper_model = whisper_model.to(weight_dtype)

    # Write PCM to a temp WAV so AudioProcessor can read it
    import tempfile  # noqa: PLC0415
    import soundfile as sf  # noqa: PLC0415

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = tmp.name
    sf.write(tmp_path, pcm, SAMPLE_RATE)

    mel_features, pcm_len = processor.get_audio_feature(tmp_path)
    whisper_chunks = processor.get_whisper_chunk(
        mel_features, device, weight_dtype, whisper_model, pcm_len, fps=fps
    )

    # whisper_chunks: (T, 50, 384) tensor
    features_np = whisper_chunks.cpu().float().numpy()

    import os  # noqa: PLC0415
    os.unlink(tmp_path)

    return AudioFeatures(
        whisper_features=features_np,
        audio_pcm=pcm,
        fps=fps,
    )


def wav_to_features(
    wav_path: str | Path,
    model_dir: Optional[str] = None,
    device: str = "cpu",
    use_stub: bool = False,
    fps: float = FPS,
) -> AudioFeatures:
    """
    Convenience: load WAV file and convert to AudioFeatures in one call.

    Parameters
    ----------
    wav_path : str or Path
        Path to WAV file (16kHz mono recommended; resampled automatically with librosa).
    model_dir : str | None
        Path to model weights directory. Default: /models.
    device : str
        'cuda' or 'cpu'.
    use_stub : bool
        If True, skip Whisper and return dummy features.
    fps : float
        Target video FPS.

    Returns
    -------
    AudioFeatures
    """
    pcm = load_wav(wav_path)
    return pcm_to_features(pcm, model_dir=model_dir, device=device, use_stub=use_stub, fps=fps)


def wav_to_feature_chunks(
    wav_path: str | Path,
    chunk_seconds: float = 2.0,
    model_dir: Optional[str] = None,
    device: str = "cpu",
    use_stub: bool = False,
    fps: float = FPS,
) -> Iterator[AudioFeatures]:
    """
    Stream a WAV file as AudioFeatures chunks (for large files / streaming simulation).

    Parameters
    ----------
    wav_path : str or Path
    chunk_seconds : float
        Duration of each chunk in seconds. At 25fps, 2s = 50 frames per chunk.
    model_dir / device / use_stub / fps
        Same as wav_to_features().

    Yields
    ------
    AudioFeatures
        Each with .whisper_features shape (chunk_frames, 50, 384).
    """
    pcm = load_wav(wav_path)
    chunk_samples = int(chunk_seconds * SAMPLE_RATE)

    for start in range(0, len(pcm), chunk_samples):
        chunk_pcm = pcm[start : start + chunk_samples]
        if len(chunk_pcm) == 0:
            break
        features = pcm_to_features(
            chunk_pcm,
            model_dir=model_dir,
            device=device,
            use_stub=use_stub,
            fps=fps,
        )
        yield features


# ──────────────────────────────────────────────────────────────────────────────
# Stub helpers
# ──────────────────────────────────────────────────────────────────────────────


def _stub_features(pcm: np.ndarray, fps: float = FPS) -> AudioFeatures:
    """
    Generate dummy zero-filled Whisper features for stub mode.
    Shape is (T, 50, 384) where T = ceil(len(pcm) / SAMPLE_RATE * fps).
    """
    duration_s = len(pcm) / SAMPLE_RATE if len(pcm) > 0 else 1.0
    n_frames = max(1, int(np.ceil(duration_s * fps)))
    features = np.zeros((n_frames, WHISPER_FRAMES_PER_VIDEO_FRAME, WHISPER_DIM), dtype=np.float32)
    logger.debug("Stub features: %d frames for %.2fs of audio", n_frames, duration_s)
    return AudioFeatures(whisper_features=features, audio_pcm=pcm, fps=fps)
