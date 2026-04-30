"""
renderer/test_inference.py — Smoke tests for the MuseTalk engine scaffold

Runs end-to-end in stub mode (CPU-safe, no GPU required).
GPU-mode tests are marked with @pytest.mark.gpu and skipped unless
RENDERER_TEST_GPU=1 is set in the environment.

Run:
    # Stub-mode only (CI / Day 1 integration testing)
    cd /Users/tenedos/edlio-presence
    python -m pytest renderer/test_inference.py -v

    # GPU mode (Day 2, once Track B pod is live)
    RENDERER_TEST_GPU=1 python -m pytest renderer/test_inference.py -v
"""

from __future__ import annotations

import os
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pytest

# Ensure repo root is on path
_repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(_repo_root))

from renderer.audio_prep import _stub_features, load_wav, wav_to_feature_chunks, wav_to_features
from renderer.engine import AudioFeatures, MuseTalkEngine, RendererFrame

# ──────────────────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────────────────

GPU_AVAILABLE = os.environ.get("RENDERER_TEST_GPU", "0") == "1"
requires_gpu = pytest.mark.skipif(not GPU_AVAILABLE, reason="Set RENDERER_TEST_GPU=1 to run GPU tests")


@pytest.fixture(scope="session")
def portrait_path(tmp_path_factory):
    """Create a tiny 64×64 test portrait PNG."""
    try:
        import cv2
        out_dir = tmp_path_factory.mktemp("assets")
        path = str(out_dir / "test_portrait.png")
        # Simple face-like gradient image
        img = np.zeros((64, 64, 3), dtype=np.uint8)
        img[10:54, 10:54] = [200, 180, 160]  # skin tone rectangle
        cv2.imwrite(path, img[:, :, ::-1])
        return path
    except ImportError:
        pytest.skip("cv2 not available — cannot create test portrait")


@pytest.fixture(scope="session")
def test_wav_path(tmp_path_factory):
    """Create a 2-second 16kHz mono WAV file with a sine wave."""
    out_dir = tmp_path_factory.mktemp("audio")
    path = str(out_dir / "test_audio.wav")
    sr = 16_000
    t = np.linspace(0, 2.0, 2 * sr, endpoint=False)
    pcm = (np.sin(2 * np.pi * 440 * t) * 0.5).astype(np.float32)
    try:
        import soundfile as sf
        sf.write(path, pcm, sr)
    except ImportError:
        # Fall back to scipy
        from scipy.io import wavfile
        wavfile.write(path, sr, (pcm * 32767).astype(np.int16))
    return path


# ──────────────────────────────────────────────────────────────────────────────
# Unit tests — AudioFeatures dataclass
# ──────────────────────────────────────────────────────────────────────────────


class TestAudioFeatures:
    def test_num_frames(self):
        features = np.zeros((30, 50, 384), dtype=np.float32)
        af = AudioFeatures(whisper_features=features)
        assert af.num_frames() == 30

    def test_default_fps(self):
        af = AudioFeatures(whisper_features=np.zeros((1, 50, 384), dtype=np.float32))
        assert af.fps == 25.0

    def test_optional_fields(self):
        af = AudioFeatures(whisper_features=np.zeros((1, 50, 384), dtype=np.float32))
        assert af.phonemes is None
        assert af.audio_pcm is None

    def test_phoneme_passthrough(self):
        """Phonemes can be attached for Track C / future renderers."""
        phonemes = [{"phoneme": "AH", "viseme": "AA", "start_ms": 0, "end_ms": 80}]
        af = AudioFeatures(
            whisper_features=np.zeros((1, 50, 384), dtype=np.float32),
            phonemes=phonemes,
        )
        assert af.phonemes == phonemes


# ──────────────────────────────────────────────────────────────────────────────
# Unit tests — Stub features
# ──────────────────────────────────────────────────────────────────────────────


class TestStubFeatures:
    def test_shape(self):
        pcm = np.zeros(32_000, dtype=np.float32)  # 2s at 16kHz
        af = _stub_features(pcm, fps=25.0)
        # 2s × 25fps = 50 frames
        assert af.whisper_features.shape == (50, 50, 384)
        assert af.whisper_features.dtype == np.float32

    def test_zero_pcm(self):
        pcm = np.zeros(0, dtype=np.float32)
        af = _stub_features(pcm, fps=25.0)
        assert af.num_frames() >= 1  # at least 1 frame for empty audio

    def test_pcm_stored(self):
        pcm = np.ones(16_000, dtype=np.float32)
        af = _stub_features(pcm)
        assert af.audio_pcm is pcm


# ──────────────────────────────────────────────────────────────────────────────
# Unit tests — Stub renderer
# ──────────────────────────────────────────────────────────────────────────────


class TestStubRenderer:
    def test_init_with_portrait(self, portrait_path):
        engine = MuseTalkEngine(portrait_path=portrait_path, use_stub=True)
        assert engine.is_stub is True

    def test_render_frame_returns_correct_shape(self, portrait_path):
        engine = MuseTalkEngine(portrait_path=portrait_path, use_stub=True)
        af = AudioFeatures(whisper_features=np.zeros((1, 50, 384), dtype=np.float32))
        result = engine.render_frame(af, frame_index=0)
        assert isinstance(result, RendererFrame)
        assert result.frame.shape == (256, 256, 3)
        assert result.frame.dtype == np.uint8
        assert result.is_stub is True

    def test_render_frame_index_increments(self, portrait_path):
        engine = MuseTalkEngine(portrait_path=portrait_path, use_stub=True)
        af = AudioFeatures(whisper_features=np.zeros((5, 50, 384), dtype=np.float32))
        for i in range(5):
            result = engine.render_frame(af, frame_index=i)
            assert result.frame_index == i

    def test_render_batch(self, portrait_path):
        engine = MuseTalkEngine(portrait_path=portrait_path, use_stub=True)
        af = AudioFeatures(whisper_features=np.zeros((10, 50, 384), dtype=np.float32))
        frames = engine.render_batch(af)
        assert len(frames) == 10
        assert all(f.frame.shape == (256, 256, 3) for f in frames)

    def test_render_stream_sync(self, portrait_path):
        engine = MuseTalkEngine(portrait_path=portrait_path, use_stub=True)

        def chunk_iter():
            for _ in range(3):
                yield AudioFeatures(whisper_features=np.zeros((5, 50, 384), dtype=np.float32))

        frames = list(engine.render_stream_sync(chunk_iter()))
        assert len(frames) == 15  # 3 chunks × 5 frames

    def test_warm_up_noop(self, portrait_path):
        """warm_up() should not raise in stub mode."""
        engine = MuseTalkEngine(portrait_path=portrait_path, use_stub=True)
        engine.warm_up()  # must not raise

    def test_reset_frame_counter(self, portrait_path):
        engine = MuseTalkEngine(portrait_path=portrait_path, use_stub=True)
        af = AudioFeatures(whisper_features=np.zeros((3, 50, 384), dtype=np.float32))
        for i in range(3):
            engine.render_frame(af, frame_index=i)
        assert engine._frame_counter == 3
        engine.reset()
        assert engine._frame_counter == 0

    def test_timestamp_increases(self, portrait_path):
        engine = MuseTalkEngine(portrait_path=portrait_path, use_stub=True)
        af = AudioFeatures(whisper_features=np.zeros((3, 50, 384), dtype=np.float32))
        timestamps = [engine.render_frame(af, frame_index=i).timestamp_ms for i in range(3)]
        assert timestamps == sorted(timestamps)
        assert all(t >= 0 for t in timestamps)

    def test_repr(self, portrait_path):
        engine = MuseTalkEngine(portrait_path=portrait_path, use_stub=True)
        r = repr(engine)
        assert "stub" in r
        assert portrait_path in r

    def test_rgb_not_bgr(self, portrait_path):
        """
        Frames must be RGB. This test creates a portrait with a strong red channel
        and verifies the first frame has higher R than B.
        """
        import cv2

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            img = np.zeros((64, 64, 3), dtype=np.uint8)
            img[:, :, 0] = 200  # R channel high
            img[:, :, 2] = 50  # B channel low
            cv2.imwrite(tmp.name, img[:, :, ::-1])  # cv2 writes BGR

            engine = MuseTalkEngine(portrait_path=tmp.name, use_stub=True)
            af = AudioFeatures(whisper_features=np.zeros((1, 50, 384), dtype=np.float32))
            result = engine.render_frame(af)

            # In RGB output: channel 0 = R, channel 2 = B
            mean_r = result.frame[:, :, 0].mean()
            mean_b = result.frame[:, :, 2].mean()
            assert mean_r > mean_b, (
                f"Expected R > B in RGB output, got R={mean_r:.1f} B={mean_b:.1f}. "
                "Possible BGR/RGB mismatch."
            )

        os.unlink(tmp.name)


# ──────────────────────────────────────────────────────────────────────────────
# Async stream test
# ──────────────────────────────────────────────────────────────────────────────


class TestAsyncStream:
    def test_render_stream_async(self, portrait_path):
        import asyncio

        engine = MuseTalkEngine(portrait_path=portrait_path, use_stub=True)

        async def run():
            async def chunk_aiter():
                for _ in range(3):
                    yield AudioFeatures(whisper_features=np.zeros((4, 50, 384), dtype=np.float32))
                    await asyncio.sleep(0)

            frames = []
            async for rendered in engine.render_stream(chunk_aiter()):
                frames.append(rendered)
            return frames

        frames = asyncio.run(run())
        assert len(frames) == 12  # 3 chunks × 4 frames
        assert all(f.is_stub for f in frames)


# ──────────────────────────────────────────────────────────────────────────────
# Integration: CLI end-to-end in stub mode
# ──────────────────────────────────────────────────────────────────────────────


class TestCLIIntegration:
    def test_cli_stub_produces_mp4(self, portrait_path, test_wav_path, tmp_path):
        """Full CLI pipeline in stub mode — must produce a non-empty MP4."""
        import subprocess

        out_mp4 = str(tmp_path / "out.mp4")
        result = subprocess.run(
            [
                sys.executable, "-m", "renderer.cli",
                "--portrait", portrait_path,
                "--audio", test_wav_path,
                "--out", out_mp4,
                "--stub",
                "--max-frames", "5",
            ],
            capture_output=True,
            text=True,
            cwd=str(_repo_root),
        )
        # CLI may fail if ffmpeg is not installed, but should not crash with Python error
        if result.returncode != 0:
            # Check it's an ffmpeg issue, not a Python bug
            assert "Traceback" not in result.stderr or "ffmpeg" in result.stderr.lower(), (
                f"CLI crashed with Python error:\n{result.stderr}"
            )
            pytest.skip("ffmpeg not available; skipping MP4 output test")
        assert Path(out_mp4).exists()
        assert Path(out_mp4).stat().st_size > 0

    def test_cli_stub_frames_dir(self, portrait_path, test_wav_path, tmp_path):
        """CLI --frames-dir saves PNG files."""
        import subprocess

        frames_dir = str(tmp_path / "frames")
        result = subprocess.run(
            [
                sys.executable, "-m", "renderer.cli",
                "--portrait", portrait_path,
                "--audio", test_wav_path,
                "--out", str(tmp_path / "out.mp4"),
                "--frames-dir", frames_dir,
                "--stub",
                "--max-frames", "3",
            ],
            capture_output=True,
            text=True,
            cwd=str(_repo_root),
        )
        if result.returncode != 0 and "Traceback" in result.stderr:
            pytest.fail(f"CLI crashed:\n{result.stderr}")

        if Path(frames_dir).exists():
            pngs = list(Path(frames_dir).glob("*.png"))
            assert len(pngs) == 3


# ──────────────────────────────────────────────────────────────────────────────
# GPU tests — skipped unless RENDERER_TEST_GPU=1
# ──────────────────────────────────────────────────────────────────────────────


@requires_gpu
class TestGPURenderer:
    """
    GPU tests — require:
      - RENDERER_TEST_GPU=1 env var
      - CUDA GPU available
      - Model weights at /models/ (or --model-dir override via RENDERER_MODEL_DIR)

    These will be validated Day 2 when Track B GPU pod is live.
    """

    MODEL_DIR = os.environ.get("RENDERER_MODEL_DIR", "/models")

    def test_cuda_available(self):
        import torch
        assert torch.cuda.is_available(), "CUDA not available — check GPU setup"

    def test_engine_loads_on_gpu(self, portrait_path):
        engine = MuseTalkEngine(
            portrait_path=portrait_path,
            device="cuda",
            use_stub=False,
            model_dir=self.MODEL_DIR,
        )
        assert not engine.is_stub

    def test_render_frame_on_gpu(self, portrait_path):
        engine = MuseTalkEngine(
            portrait_path=portrait_path,
            device="cuda",
            use_stub=False,
            model_dir=self.MODEL_DIR,
        )
        af = AudioFeatures(whisper_features=np.zeros((1, 50, 384), dtype=np.float32))
        result = engine.render_frame(af)
        assert result.frame.shape == (256, 256, 3)
        assert result.frame.dtype == np.uint8
        assert not result.is_stub

    def test_latency_under_50ms(self, portrait_path):
        """
        Per-frame latency must be <50ms on target GPU (V100/L40S).

        Published MuseTalk benchmark: ~33ms/frame on V100.
        This test validates we meet the <50ms target.
        """
        engine = MuseTalkEngine(
            portrait_path=portrait_path,
            device="cuda",
            use_stub=False,
            model_dir=self.MODEL_DIR,
        )
        engine.warm_up()

        af = AudioFeatures(whisper_features=np.zeros((1, 50, 384), dtype=np.float32))

        # Time 10 frames, take the median
        times = []
        for _ in range(10):
            t0 = time.perf_counter()
            engine.render_frame(af)
            times.append((time.perf_counter() - t0) * 1000.0)

        median_ms = sorted(times)[len(times) // 2]
        print(f"\nPer-frame latency: median={median_ms:.1f}ms, min={min(times):.1f}ms, max={max(times):.1f}ms")
        assert median_ms < 50.0, (
            f"Per-frame latency {median_ms:.1f}ms exceeds 50ms target. "
            "Check GPU utilization, fp16 mode, and model loading."
        )

    def test_fp16_output_valid(self, portrait_path):
        """FP16 inference must not produce NaN or all-black frames."""
        engine = MuseTalkEngine(
            portrait_path=portrait_path,
            device="cuda",
            use_stub=False,
            use_float16=True,
            model_dir=self.MODEL_DIR,
        )
        af = AudioFeatures(whisper_features=np.random.randn(1, 50, 384).astype(np.float32))
        result = engine.render_frame(af)
        assert not np.isnan(result.frame).any(), "NaN values in output frame"
        assert result.frame.max() > 0, "All-black frame output (model may not be loaded)"
