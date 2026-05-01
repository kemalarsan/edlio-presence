"""
renderer/server.py — FastAPI wrapper for the MuseTalk pipeline.

Wire-up of renderer.engine.MuseTalkEngine over HTTP so tenedos-voice (and
future clients) can call /render with audio + portrait and get back an MP4.

Run on a GPU pod launched from ghcr.io/kemalarsan/edlio-presence:day2-snapshot:

    cd /workspace/edlio-presence
    uvicorn renderer.server:app --host 0.0.0.0 --port 8080

Endpoints:
    GET  /healthz            — liveness (used by tenedos-voice probe)
    POST /render             — synchronous render: { audio, portrait } → { videoUrl }
    GET  /renders/{filename} — static MP4 serving

Auth: optional bearer token via env RENDERER_AUTH_TOKEN.

Concurrency note: MuseTalk is GPU-bound; serialize renders with a single lock
for now. Upgrade to a queue+worker pattern when we have multi-GPU pods.
"""

from __future__ import annotations

import base64
import hashlib
import io
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request
from pathlib import Path
from typing import Optional

import numpy as np
from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

log = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s - %(message)s",
)

# Make `audio.extractor`, `visemes`, and `renderer.engine` importable
# regardless of how uvicorn is launched. audio/extractor.py does
# `from visemes import ...` which requires audio/ on sys.path too.
_ROOT = Path(__file__).parent.parent
for p in (
    _ROOT,
    _ROOT / "audio",
    _ROOT / "renderer" / "muse_talk_vendor",
):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
RENDERER_AUTH_TOKEN = os.environ.get("RENDERER_AUTH_TOKEN")
OUTPUT_DIR = Path(os.environ.get("RENDERER_OUTPUT_DIR", "/workspace/renders"))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR = os.environ.get(
    "RENDERER_MODEL_DIR", "/workspace/edlio-presence/models"
)
PUBLIC_BASE_URL = os.environ.get("RENDERER_PUBLIC_BASE_URL", "")  # e.g. "https://pod.runpod.app"

app = FastAPI(title="edlio-presence renderer", version="0.2.0")

_render_lock = threading.Lock()
_engine_cache: dict[str, object] = {}  # portraitCacheKey → MuseTalkEngine


# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------
def verify_token(authorization: Optional[str] = Header(default=None)) -> None:
    """Bearer auth. If RENDERER_AUTH_TOKEN isn't set, allow all (dev mode)."""
    if not RENDERER_AUTH_TOKEN:
        return
    expected = f"Bearer {RENDERER_AUTH_TOKEN}"
    if authorization != expected:
        raise HTTPException(status_code=401, detail="invalid or missing bearer token")


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
class RenderRequest(BaseModel):
    audio: str = Field(..., description="Base64 PCM16 mono 16kHz OR http(s) URL to wav/mp3")
    portrait: str = Field(..., description="http(s) URL to a portrait image (PNG/JPEG)")
    portraitCacheKey: Optional[str] = Field(
        default=None,
        description="Stable key so repeated renders of the same portrait reuse preprocessed latents",
    )
    fps: int = Field(default=25, ge=5, le=60, description="Target frames per second")


class RenderResponse(BaseModel):
    ok: bool
    videoUrl: str
    durationSec: float
    metrics: dict[str, float | str]


# ---------------------------------------------------------------------------
# /healthz
# ---------------------------------------------------------------------------
@app.get("/healthz")
def healthz() -> dict[str, object]:
    import torch  # inline so /healthz works even if CUDA import fails

    cuda_ok = False
    gpu_name = ""
    try:
        cuda_ok = torch.cuda.is_available()
        if cuda_ok:
            gpu_name = torch.cuda.get_device_name(0)
    except Exception as e:  # noqa: BLE001
        log.warning("CUDA probe failed: %s", e)

    return {
        "ok": True,
        "version": "0.2.0",
        "engine": "musetalk-v1.5",
        "cuda": cuda_ok,
        "gpu": gpu_name,
        "model_dir": MODEL_DIR,
        "model_dir_exists": Path(MODEL_DIR).exists(),
        "cached_engines": list(_engine_cache.keys()),
    }


# ---------------------------------------------------------------------------
# /render
# ---------------------------------------------------------------------------
@app.post("/render", response_model=RenderResponse, dependencies=[Depends(verify_token)])
def render(req: RenderRequest) -> RenderResponse:
    """
    Synchronous render. Blocks until the MP4 is written, then returns a URL
    pointing at our static /renders mount.

    Flow:
      1. Resolve audio → local wav file
      2. Resolve portrait → local image file (cached if portraitCacheKey reused)
      3. Get or build a MuseTalkEngine for this portrait (cached by portraitCacheKey)
      4. Extract Whisper features from the audio
      5. Render N frames
      6. Write frames + mux audio → /workspace/renders/<job_id>.mp4
      7. Return URL
    """
    start = time.time()
    job_id = hashlib.sha256(
        f"{req.portrait}|{req.portraitCacheKey}|{time.time()}".encode()
    ).hexdigest()[:16]
    log.info("render job=%s portrait=%s cacheKey=%s", job_id, req.portrait, req.portraitCacheKey)

    with tempfile.TemporaryDirectory(prefix=f"render-{job_id}-") as tmpdir:
        tmp = Path(tmpdir)

        # 1. Audio
        try:
            audio_path = _resolve_audio(req.audio, tmp / "input.wav")
        except Exception as e:  # noqa: BLE001
            raise HTTPException(400, detail={"error": "bad_audio", "message": str(e)})

        # 2. Portrait
        try:
            portrait_path = _resolve_portrait(req.portrait, tmp / "portrait.jpg", req.portraitCacheKey)
        except Exception as e:  # noqa: BLE001
            raise HTTPException(400, detail={"error": "portrait_fetch_failed", "message": str(e)})

        # 3–6 inside a lock because the GPU can only do one render at a time
        with _render_lock:
            try:
                engine = _get_or_create_engine(portrait_path, req.portraitCacheKey)
                features = _extract_features(audio_path, fps=float(req.fps))
                frames = _render_frames(engine, features)
                out_path = OUTPUT_DIR / f"{job_id}.mp4"
                _write_mp4(frames, audio_path, out_path, fps=req.fps)
            except Exception as e:  # noqa: BLE001
                log.exception("render failed")
                raise HTTPException(500, detail={"error": "render_failed", "message": str(e)})

    elapsed = time.time() - start
    duration_sec = len(frames) / float(req.fps)

    base = PUBLIC_BASE_URL.rstrip("/") if PUBLIC_BASE_URL else ""
    video_url = f"{base}/renders/{out_path.name}" if base else f"/renders/{out_path.name}"

    return RenderResponse(
        ok=True,
        videoUrl=video_url,
        durationSec=duration_sec,
        metrics={
            "job_id": job_id,
            "frames": len(frames),
            "render_time_sec": round(elapsed, 2),
            "fps_out": req.fps,
            "fps_achieved": round(len(frames) / elapsed, 2) if elapsed > 0 else 0,
        },
    )


# ---------------------------------------------------------------------------
# /renders/{filename}
# ---------------------------------------------------------------------------
@app.get("/renders/{filename}")
def get_render(filename: str) -> FileResponse:
    path = OUTPUT_DIR / filename
    if not path.exists() or path.suffix != ".mp4":
        raise HTTPException(status_code=404, detail="not found")
    return FileResponse(path, media_type="video/mp4")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _resolve_audio(audio: str, dst: Path) -> Path:
    """Turn the request's audio field into a local wav file."""
    if audio.startswith(("http://", "https://")):
        log.info("fetching audio URL: %s", audio)
        urllib.request.urlretrieve(audio, dst)
        return dst
    # Else: assume base64-encoded bytes. We accept either raw PCM16 or an
    # encoded wav/mp3; ffmpeg will normalize.
    raw = base64.b64decode(audio)
    raw_path = dst.with_suffix(".bin")
    raw_path.write_bytes(raw)
    # Re-encode to 16kHz mono wav for Whisper.
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(raw_path), "-ac", "1", "-ar", "16000", str(dst)],
        check=True,
        capture_output=True,
    )
    return dst


def _resolve_portrait(url: str, dst: Path, cache_key: Optional[str]) -> Path:
    """Fetch portrait to a local file. Cached on disk if cache_key provided."""
    if cache_key:
        cached = OUTPUT_DIR.parent / "portraits" / f"{cache_key}.jpg"
        cached.parent.mkdir(parents=True, exist_ok=True)
        if cached.exists():
            log.info("portrait cache hit: %s", cache_key)
            return cached
    if url.startswith(("http://", "https://")):
        log.info("fetching portrait URL: %s", url)
        urllib.request.urlretrieve(url, dst)
    else:
        # Allow absolute paths on the pod (dev only)
        if not Path(url).exists():
            raise FileNotFoundError(url)
        shutil.copy(url, dst)
    if cache_key:
        shutil.copy(dst, OUTPUT_DIR.parent / "portraits" / f"{cache_key}.jpg")
    return dst


def _get_or_create_engine(portrait_path: Path, cache_key: Optional[str]) -> object:
    """Cache one MuseTalkEngine per portraitCacheKey."""
    key = cache_key or str(portrait_path)
    if key in _engine_cache:
        log.info("engine cache hit: %s", key)
        return _engine_cache[key]

    from renderer.engine import MuseTalkEngine  # local import so import errors surface here

    log.info("building engine for portrait=%s cacheKey=%s", portrait_path, key)
    engine = MuseTalkEngine(
        portrait_path=str(portrait_path),
        device="cuda",
        use_stub=False,
        use_float16=True,
        model_dir=MODEL_DIR,
    )
    _engine_cache[key] = engine
    return engine


def _extract_features(audio_path: Path, fps: float):
    """Extract Whisper features from the audio file.

    Returns an AudioFeatures instance compatible with MuseTalkEngine.render_frame.
    """
    from audio.extractor import extract_musetalk_features
    from renderer.engine import AudioFeatures
    import torch

    mt_features = extract_musetalk_features(str(audio_path), fps=fps)
    audio_prompts = mt_features.audio_prompts
    if isinstance(audio_prompts, torch.Tensor):
        audio_prompts = audio_prompts.detach().cpu().numpy()
    T = audio_prompts.shape[0]
    # (T, 10, 5, 384) → (T, 50, 384)
    whisper_features = audio_prompts.reshape(T, -1, 384).astype(np.float32)
    return AudioFeatures(whisper_features=whisper_features, fps=fps)


def _render_frames(engine, features) -> list[np.ndarray]:
    """Render all frames, returning plain numpy RGB arrays."""
    T = features.num_frames()
    out: list[np.ndarray] = []
    log.info("rendering %d frames", T)
    for i in range(T):
        rf = engine.render_frame(features, frame_index=i)
        arr = rf.frame if hasattr(rf, "frame") else rf
        out.append(arr)
    return out


def _write_mp4(frames: list[np.ndarray], audio_path: Path, out_path: Path, fps: int) -> None:
    """Write frames → no-audio mp4, then mux audio in."""
    import cv2  # local import so CPU-only test envs can import the module

    h, w = frames[0].shape[:2]
    noaudio = out_path.with_suffix(".noaudio.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(noaudio), fourcc, float(fps), (w, h))
    for f in frames:
        writer.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
    writer.release()

    subprocess.run(
        [
            "ffmpeg", "-y",
            "-i", str(noaudio),
            "-i", str(audio_path),
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-c:a", "aac",
            "-shortest",
            str(out_path),
        ],
        check=True,
        capture_output=True,
    )
    noaudio.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Entry
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn

    uvicorn.run("renderer.server:app", host="0.0.0.0", port=8080, log_level="info")
