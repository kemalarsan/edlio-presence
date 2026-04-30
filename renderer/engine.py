"""
renderer/engine.py — MuseTalk engine wrapper for edlio-presence

Clean public interface around MuseTalk's inference pipeline.
This is the file Track G (SDK) and downstream consumers depend on.

Audio-input format (V1): Whisper embeddings
    - Shape: (T, 50, 384) float32 tensor  — T audio frames at 25fps
    - Produced by: musetalk.utils.audio_processor.AudioProcessor
    - See AudioFeatures dataclass below for the canonical schema

Audio-input format (V2, future): phoneme-augmented
    - Wraps Whisper embeddings + optional phoneme labels from Track C
    - MuseTalk uses only whisper_features for inference; phonemes are passed
      through for downstream consumers (mouth-shape analytics, timing metadata)

Frame-output format:
    - numpy RGB array, shape (256, 256, 3), dtype uint8
    - Coordinate order: (H, W, C) — height × width × channels
    - Color order: RGB (NOT BGR — callers don't need to convert)
    - This is what Track D (browser decode) and Track G (SDK) receive

Portrait:
    - Input: any image with a single clear face (≥256×256 recommended)
    - Internal: MuseTalk crops + resizes face region to 256×256 before processing
    - Output frames are 256×256 (the face region only, composited back if full-frame)

Usage example:
    engine = MuseTalkEngine("assets/tenedos-face-v1.png", device="cuda")
    async for frame in engine.render_stream(audio_chunk_iter):
        send_frame_to_encoder(frame)  # numpy RGB (256, 256, 3)
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import AsyncIterator, Iterator, Optional

import numpy as np

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Data contracts (stable interface for Track C and Track G)
# ──────────────────────────────────────────────────────────────────────────────


@dataclass
class AudioFeatures:
    """
    Lingua franca between Track C (audio/phoneme extractor) and Track A (renderer).

    V1 (current): whisper_features carries everything MuseTalk needs.
    V2 (planned): phonemes provides enriched timing metadata for future renderers.

    Fields
    ------
    whisper_features : np.ndarray, shape (T, 50, 384), dtype float32
        Whisper encoder hidden-state features, one vector per video frame at 25fps.
        Produced by musetalk.utils.audio_processor.AudioProcessor.get_whisper_chunk().

    phonemes : list[dict] | None
        Optional phoneme timeline from Track C. Format:
            [{"phoneme": "AH", "viseme": "AA", "start_ms": 0, "end_ms": 80}, ...]
        MuseTalk V1 ignores this field; keep None until Track C ships.

    audio_pcm : np.ndarray | None
        Raw 16kHz mono PCM float32, shape (N,).
        Included for fallback processors and debugging. Optional.

    fps : float
        Video frames per second this feature set was computed for. Default 25.
    """

    whisper_features: np.ndarray  # (T, 50, 384) float32
    phonemes: Optional[list] = None
    audio_pcm: Optional[np.ndarray] = None
    fps: float = 25.0

    def num_frames(self) -> int:
        """Number of video frames encoded in this feature batch."""
        return int(self.whisper_features.shape[0])


@dataclass
class RendererFrame:
    """
    A single rendered output frame.

    frame : np.ndarray, shape (256, 256, 3), dtype uint8, RGB
    frame_index : int  — 0-based position in the output sequence
    timestamp_ms : float — approximate wall-clock time of this frame
    is_stub : bool — True if engine ran in stub/CPU mode (GPU not available)
    """

    frame: np.ndarray
    frame_index: int = 0
    timestamp_ms: float = 0.0
    is_stub: bool = False


# ──────────────────────────────────────────────────────────────────────────────
# Stub renderer (CPU-safe, no CUDA required)
# ──────────────────────────────────────────────────────────────────────────────


class _StubRenderer:
    """
    Trivial stub that returns the portrait frame unchanged for every audio frame.

    Purpose: allows downstream tracks (D, G) to integrate and test their
    pipelines before GPU infra is live.  Swap this out by calling engine with
    use_stub=False once Track B provides a running GPU pod.

    Behaviour:
      - Loads the portrait image once at init
      - Each call to render_frame() returns the portrait resized to 256×256
      - No model weights are loaded; no CUDA required
    """

    def __init__(self, portrait_path: str):
        try:
            import cv2  # noqa: PLC0415
            img_bgr = cv2.imread(portrait_path)
            if img_bgr is None:
                raise FileNotFoundError(f"Portrait not found: {portrait_path}")
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            self.portrait = cv2.resize(img_rgb, (256, 256), interpolation=cv2.INTER_LANCZOS4)
        except ImportError:
            # cv2 not available — build a grey placeholder
            logger.warning("cv2 not available; using grey placeholder portrait")
            self.portrait = np.full((256, 256, 3), 128, dtype=np.uint8)
        logger.info("StubRenderer: loaded portrait %s → 256×256", portrait_path)

    def render_frame(self, audio_features: AudioFeatures, frame_index: int = 0) -> np.ndarray:
        """Return portrait unchanged — stub implementation."""
        return self.portrait.copy()

    def warm_up(self) -> None:
        """No-op for the stub."""
        logger.debug("StubRenderer: warm_up() called (no-op)")


# ──────────────────────────────────────────────────────────────────────────────
# Real MuseTalk renderer (GPU required)
# ──────────────────────────────────────────────────────────────────────────────

# Vendor path: renderer/muse_talk_vendor/ relative to this file
_VENDOR_DIR = Path(__file__).parent / "muse_talk_vendor"
_DEFAULT_MODEL_DIR = Path("/models")  # mounted volume in Docker container


class _MuseTalkRenderer:
    """
    Wraps MuseTalk inference.  GPU required.

    This class is intentionally private — callers use MuseTalkEngine.
    Import errors from missing CUDA / weights are caught here and surfaced
    with clear messages.
    """

    def __init__(
        self,
        portrait_path: str,
        device: str = "cuda",
        model_dir: Optional[str] = None,
        use_float16: bool = True,
        fps: float = 25.0,
    ):
        import torch  # noqa: PLC0415

        self.device = device
        self.fps = fps
        self.use_float16 = use_float16
        self.weight_dtype = torch.float16 if use_float16 else torch.float32
        self.model_dir = Path(model_dir) if model_dir else _DEFAULT_MODEL_DIR

        # Add vendor to path so MuseTalk imports resolve
        vendor_str = str(_VENDOR_DIR)
        if vendor_str not in sys.path:
            sys.path.insert(0, vendor_str)

        self._load_portrait(portrait_path)
        self._load_models()

    def _load_portrait(self, portrait_path: str) -> None:
        import cv2  # noqa: PLC0415

        img_bgr = cv2.imread(portrait_path)
        if img_bgr is None:
            raise FileNotFoundError(f"Portrait not found: {portrait_path}")
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        self.portrait_orig = img_rgb

        # Resize to 256×256 — MuseTalk's native face region size
        self.portrait_256 = cv2.resize(img_rgb, (256, 256), interpolation=cv2.INTER_LANCZOS4)
        logger.info("Portrait loaded: %s → shape %s", portrait_path, self.portrait_256.shape)

    def _load_models(self) -> None:
        """
        Load VAE, UNet, PositionalEncoding, and Whisper model.

        Model weight paths follow MuseTalk's expected layout under /models/:
          /models/musetalkV15/unet.pth
          /models/musetalkV15/musetalk.json
          /models/sd-vae/
          /models/whisper/
        """
        import torch  # noqa: PLC0415
        from musetalk.models.unet import PositionalEncoding, UNet  # noqa: PLC0415
        from musetalk.models.vae import VAE  # noqa: PLC0415
        from musetalk.utils.utils import load_all_model  # noqa: PLC0415
        from transformers import WhisperModel  # noqa: PLC0415

        unet_path = str(self.model_dir / "musetalkV15" / "unet.pth")
        unet_config = str(self.model_dir / "musetalkV15" / "musetalk.json")

        logger.info("Loading MuseTalk models from %s …", self.model_dir)
        self.vae, self.unet, self.pe = load_all_model(
            unet_model_path=unet_path,
            vae_type="sd-vae",
            unet_config=unet_config,
            device=self.device,
        )

        # Load Whisper encoder
        whisper_path = str(self.model_dir / "whisper")
        logger.info("Loading Whisper encoder from %s …", whisper_path)
        self.whisper = WhisperModel.from_pretrained(whisper_path)
        self.whisper.to(self.device).to(self.weight_dtype)
        self.whisper.eval()

        # Move MuseTalk models to device + dtype
        self.vae.vae.to(self.device, dtype=self.weight_dtype)
        self.unet.model.to(self.device, dtype=self.weight_dtype)
        # Positional encoding also needs to match weight dtype (contains learnable params)
        self.pe.to(self.device, dtype=self.weight_dtype)

        logger.info("MuseTalk models loaded on %s (fp16=%s)", self.device, self.use_float16)

        # Pre-encode the portrait face latents (done once, reused every frame)
        self._precompute_portrait_latents()

    def _precompute_portrait_latents(self) -> None:
        """
        Encode the portrait face region into VAE latents.
        This is a one-time cost at warm-up; latents are reused for every frame.
        """
        import torch  # noqa: PLC0415

        frame = self.portrait_256
        with torch.no_grad():
            latents = self.vae.get_latents_for_unet(frame)
            self.portrait_latents = latents.to(dtype=self.weight_dtype)
        logger.debug("Portrait latents precomputed: shape %s", self.portrait_latents.shape)

    def warm_up(self) -> None:
        """
        Run a dummy inference pass to warm up CUDA kernels.
        Recommended before real-time streaming begins.
        """
        import torch  # noqa: PLC0415

        logger.info("Warming up MuseTalk GPU kernels …")
        dummy_audio = torch.zeros(1, 50, 384, device=self.device, dtype=self.weight_dtype)
        dummy_features = AudioFeatures(whisper_features=dummy_audio.cpu().numpy())
        self.render_frame(dummy_features, frame_index=0)
        logger.info("Warm-up complete.")

    def render_frame(self, audio_features: AudioFeatures, frame_index: int = 0) -> np.ndarray:
        """
        Render a single face frame from audio features.

        Parameters
        ----------
        audio_features : AudioFeatures
            Whisper features for this frame window.
        frame_index : int
            Which time-step to extract from the whisper_features batch.

        Returns
        -------
        np.ndarray, shape (256, 256, 3), dtype uint8, RGB
        """
        import torch  # noqa: PLC0415

        # Extract one frame's worth of whisper embeddings: (1, 50, 384)
        whisper_slice = audio_features.whisper_features[frame_index : frame_index + 1]
        audio_tensor = torch.from_numpy(whisper_slice).to(self.device, dtype=self.weight_dtype)

        # Apply positional encoding
        audio_prompt = self.pe(audio_tensor)

        # UNet timestep (inference always uses t=0 for MuseTalk)
        timesteps = torch.zeros(1, device=self.device, dtype=torch.long)

        with torch.no_grad():
            pred_latents = self.unet.model(
                self.portrait_latents,
                timesteps,
                encoder_hidden_states=audio_prompt,
            ).sample
            frame_rgb = self.vae.decode_latents(pred_latents)

        # frame_rgb: (256, 256, 3) float32 in [0, 255]
        frame_uint8 = np.clip(frame_rgb[0] if hasattr(frame_rgb, "__len__") else frame_rgb, 0, 255).astype(np.uint8)
        return frame_uint8


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────


class MuseTalkEngine:
    """
    Public face renderer interface for edlio-presence.

    This is the file Track G (SDK) imports.  Keep breaking changes out.

    Parameters
    ----------
    portrait_path : str
        Path to the face portrait image (any common format).
        Recommended: 512×512 or larger with a single frontal face.
        MuseTalk crops + resizes the face region to 256×256 internally.

    device : str
        PyTorch device string. Use 'cuda' for GPU (required for real-time).
        Use 'cpu' for stub-mode testing only.

    use_stub : bool | None
        True  → always use stub renderer (fast, CPU-safe, no weights needed).
        False → always use real MuseTalk renderer (needs GPU + model weights).
        None  → auto: use stub if CUDA is not available (default, Toyota ethos).

    model_dir : str | None
        Path to MuseTalk model weights directory.
        Defaults to /models (the Docker volume mount; see infra/README.md).
        Override for local testing: e.g. model_dir='/tmp/musetalk-weights'.

    use_float16 : bool
        Run inference in FP16 (recommended for V100/L40S, saves VRAM).
        Set False only if you see NaN outputs on specific hardware.

    fps : float
        Target output frame rate. Affects audio feature chunking. Default 25.

    Examples
    --------
    # GPU mode (production)
    engine = MuseTalkEngine("assets/tenedos-face-v1.png", device="cuda")
    frame = engine.render_frame(audio_features)  # numpy RGB (256, 256, 3)

    # Stub mode (integration testing without GPU)
    engine = MuseTalkEngine("assets/tenedos-face-v1.png", use_stub=True)
    async for frame in engine.render_stream(audio_chunk_iter):
        send_to_encoder(frame)

    # Auto-detect (Toyota ethos — degrades gracefully)
    engine = MuseTalkEngine("assets/tenedos-face-v1.png")  # use_stub=None
    """

    def __init__(
        self,
        portrait_path: str,
        device: str = "cuda",
        use_stub: Optional[bool] = None,
        model_dir: Optional[str] = None,
        use_float16: bool = True,
        fps: float = 25.0,
    ):
        self.portrait_path = portrait_path
        self.fps = fps
        self._frame_counter = 0

        # Resolve stub mode
        if use_stub is None:
            use_stub = not self._cuda_available()
            if use_stub:
                logger.info(
                    "CUDA not available — falling back to stub renderer. "
                    "Set use_stub=False and provide a GPU to use real MuseTalk."
                )

        if use_stub:
            logger.info("MuseTalkEngine: stub mode (CPU-safe, portrait passthrough)")
            self._renderer = _StubRenderer(portrait_path)
            self.is_stub = True
        else:
            logger.info("MuseTalkEngine: GPU mode on device=%s", device)
            self._renderer = _MuseTalkRenderer(
                portrait_path=portrait_path,
                device=device,
                model_dir=model_dir,
                use_float16=use_float16,
                fps=fps,
            )
            self.is_stub = False

    @staticmethod
    def _cuda_available() -> bool:
        try:
            import torch  # noqa: PLC0415
            return torch.cuda.is_available()
        except ImportError:
            return False

    def warm_up(self) -> None:
        """
        Pre-warm GPU kernels before real-time streaming.
        Call once after init, before the first render_stream() or render_frame().
        No-op in stub mode.
        """
        self._renderer.warm_up()

    def render_frame(
        self,
        audio_features: AudioFeatures,
        frame_index: int = 0,
    ) -> RendererFrame:
        """
        Render a single face frame.

        Parameters
        ----------
        audio_features : AudioFeatures
            Audio features for this frame (Whisper embeddings, shape ≥ (1, 50, 384)).
        frame_index : int
            Time-step index within audio_features.whisper_features to render.

        Returns
        -------
        RendererFrame
            .frame     — numpy RGB (256, 256, 3) uint8
            .is_stub   — True if GPU wasn't available
        """
        raw = self._renderer.render_frame(audio_features, frame_index=frame_index)
        idx = self._frame_counter
        self._frame_counter += 1
        ts_ms = (idx / self.fps) * 1000.0
        return RendererFrame(frame=raw, frame_index=idx, timestamp_ms=ts_ms, is_stub=self.is_stub)

    def render_batch(
        self,
        audio_features: AudioFeatures,
    ) -> list[RendererFrame]:
        """
        Render all frames in an AudioFeatures batch.

        Returns
        -------
        list[RendererFrame]
            One RendererFrame per time-step in audio_features.whisper_features.
        """
        n = audio_features.num_frames()
        return [self.render_frame(audio_features, frame_index=i) for i in range(n)]

    async def render_stream(
        self,
        audio_chunks: AsyncIterator[AudioFeatures],
    ) -> AsyncIterator[RendererFrame]:
        """
        Async generator: consume an async iterator of AudioFeatures, yield RendererFrames.

        Designed for integration with Track G's WebRTC/WebSocket output pipeline.

        Parameters
        ----------
        audio_chunks : AsyncIterator[AudioFeatures]
            Stream of AudioFeatures batches (e.g. 80ms chunks at 25fps → 2 frames each).

        Yields
        ------
        RendererFrame
            One per video frame, in order. frame is numpy RGB (256, 256, 3) uint8.

        Example
        -------
        async for rendered in engine.render_stream(audio_source()):
            await ws.send_bytes(encode_mjpeg(rendered.frame))
        """
        async for chunk in audio_chunks:
            n = chunk.num_frames()
            for i in range(n):
                rendered = self.render_frame(chunk, frame_index=i)
                yield rendered
                # Yield control to event loop between frames
                await asyncio.sleep(0)

    def render_stream_sync(
        self,
        audio_chunks: Iterator[AudioFeatures],
    ) -> Iterator[RendererFrame]:
        """
        Synchronous generator variant for non-async callers (CLI, tests).

        Parameters
        ----------
        audio_chunks : Iterator[AudioFeatures]
            Iterator of AudioFeatures batches.

        Yields
        ------
        RendererFrame
        """
        for chunk in audio_chunks:
            n = chunk.num_frames()
            for i in range(n):
                yield self.render_frame(chunk, frame_index=i)

    def reset(self) -> None:
        """Reset frame counter (call between sessions/takes)."""
        self._frame_counter = 0

    def __repr__(self) -> str:
        mode = "stub" if self.is_stub else "gpu"
        return f"MuseTalkEngine(portrait={self.portrait_path!r}, mode={mode}, fps={self.fps})"
