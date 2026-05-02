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
    portrait: str = Field(
        ..., description="http(s) URL to a portrait image (PNG/JPEG) — or a video (.mp4/.mov) when portraitIsVideo=true",
    )
    portraitIsVideo: bool = Field(
        default=False,
        description=(
            "When true, treat `portrait` as a video URL and use the video-reference "
            "renderer (samples N frames, banks their latents, cycles per output frame). "
            "Requires faceBox. Produces visibly more alive output at a higher warm-up cost."
        ),
    )
    portraitNumRefFrames: int = Field(
        default=60,
        ge=4,
        le=240,
        description=(
            "When portraitIsVideo=true, how many reference frames to sample from the clip. "
            "Default 60 (~2.4s of 25fps motion before the bank loops). Higher=more variety, "
            "more VRAM, slower warm-up."
        ),
    )
    portraitCacheKey: Optional[str] = Field(
        default=None,
        description="Stable key so repeated renders of the same portrait reuse preprocessed latents",
    )
    fps: int = Field(default=25, ge=5, le=60, description="Target frames per second")
    faceBox: Optional[list[int]] = Field(
        default=None,
        description=(
            "Optional face bounding box in portrait pixel coords [x1, y1, x2, y2]. "
            "When provided, the server renders a full-portrait composite with the "
            "predicted mouth region blended back in. When omitted, returns the raw "
            "256×256 face patch (legacy POC behaviour)."
        ),
    )
    extraMargin: int = Field(
        default=10,
        ge=0,
        le=200,
        description="Extra pixels added to bottom of faceBox to include chin. Default 10 matches MuseTalk app.py.",
    )
    parsingMode: str = Field(
        default="jaw",
        description="Face parsing mode passed through to blending.get_image(). 'jaw' is MuseTalk V1.5 default.",
    )
    gfpgan: bool = Field(
        default=False,
        description=(
            "When true, run a GFPGAN post-sharpening pass on MuseTalk's 256×256 face output "
            "before compositing. Requires faceBox (composite path) and the GFPGAN weights on "
            "the model volume. Weights auto-download on first use (~334 MB)."
        ),
    )
    gfpganWeight: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description=(
            "GFPGAN identity/quality knob. 0.0=no effect, 1.0=maximum restoration. "
            "Empirical sweet spot for talking-face stills is 0.2-0.3: enough to sharpen "
            "teeth and lip edges without the plasticky 'doll' look that appears >=0.5. "
            "Calibrated 2026-05-01 on tenedos-v1.png; revisit per portrait if needed."
        ),
    )


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

    latentsync_st = {}
    try:
        from renderer import latentsync_runner
        latentsync_st = latentsync_runner.status()
    except Exception as e:  # noqa: BLE001
        latentsync_st = {"error": str(e)}

    return {
        "ok": True,
        "version": "0.3.0",
        "engines": ["musetalk-v1.5", "latentsync-1.6"],
        "cuda": cuda_ok,
        "gpu": gpu_name,
        "model_dir": MODEL_DIR,
        "model_dir_exists": Path(MODEL_DIR).exists(),
        "cached_engines": list(_engine_cache.keys()),
        "latentsync": latentsync_st,
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

        # 2. Portrait — image OR video depending on req.portraitIsVideo
        portrait_local_ext = ".mp4" if req.portraitIsVideo else ".jpg"
        try:
            portrait_path = _resolve_portrait(
                req.portrait,
                tmp / f"portrait{portrait_local_ext}",
                req.portraitCacheKey,
                is_video=req.portraitIsVideo,
            )
        except Exception as e:  # noqa: BLE001
            raise HTTPException(400, detail={"error": "portrait_fetch_failed", "message": str(e)})

        # 3–6 inside a lock because the GPU can only do one render at a time
        with _render_lock:
            try:
                use_composite = req.faceBox is not None and len(req.faceBox) == 4
                if req.portraitIsVideo:
                    if not use_composite:
                        raise HTTPException(
                            400,
                            detail={
                                "error": "bad_request",
                                "message": "portraitIsVideo=true requires faceBox",
                            },
                        )
                    renderer_obj = _get_or_create_videoref_renderer(
                        video_path=portrait_path,
                        cache_key=req.portraitCacheKey,
                        face_box=list(req.faceBox or []),
                        extra_margin=req.extraMargin,
                        parsing_mode=req.parsingMode,
                        use_gfpgan=req.gfpgan,
                        gfpgan_weight=req.gfpganWeight,
                        num_ref_frames=req.portraitNumRefFrames,
                    )
                    log.info(
                        "render mode: videoref (bbox=%s, gfpgan=%s, weight=%.2f, num_ref_frames=%d)",
                        req.faceBox, req.gfpgan, req.gfpganWeight, req.portraitNumRefFrames,
                    )
                elif use_composite:
                    renderer_obj = _get_or_create_composite_renderer(
                        portrait_path=portrait_path,
                        cache_key=req.portraitCacheKey,
                        face_box=list(req.faceBox or []),
                        extra_margin=req.extraMargin,
                        parsing_mode=req.parsingMode,
                        use_gfpgan=req.gfpgan,
                        gfpgan_weight=req.gfpganWeight,
                    )
                    log.info(
                        "render mode: composite (full portrait, bbox=%s, gfpgan=%s, weight=%.2f)",
                        req.faceBox, req.gfpgan, req.gfpganWeight,
                    )
                else:
                    renderer_obj = _get_or_create_engine(portrait_path, req.portraitCacheKey)
                    log.info("render mode: legacy (256×256 face patch)")

                features = _extract_features(audio_path, fps=float(req.fps))
                frames = _render_frames(renderer_obj, features)
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


@app.get("/renders")
def list_renders(limit: int = 20) -> dict[str, object]:
    """List recent renders by mtime. Used when Cloudflare 524s on sync calls
    and we need to find the output file post-hoc."""
    items = []
    for p in sorted(OUTPUT_DIR.glob("*.mp4"), key=lambda x: x.stat().st_mtime, reverse=True)[:limit]:
        st = p.stat()
        items.append({
            "name": p.name,
            "size": st.st_size,
            "mtime": st.st_mtime,
            "age_sec": round(time.time() - st.st_mtime, 1),
        })
    return {"count": len(items), "items": items}


# ---------------------------------------------------------------------------
# /render_latentsync  —  alt renderer: ByteDance LatentSync-1.6 (512×512)
# ---------------------------------------------------------------------------
# Kept isolated from the MuseTalk path because LatentSync uses a different
# torch/diffusers/numpy stack in its own venv. We shell out to its inference
# CLI rather than bolting it into our compositor.

class LatentSyncRenderRequest(BaseModel):
    portraitVideo: str = Field(
        ...,
        description=(
            "Portrait video URL. LatentSync does its own face detection + "
            "512×512 crop + temporal lip-sync + compositing."
        ),
    )
    audio: str = Field(..., description="Audio URL or base64 (same contract as /render).")
    inferenceSteps: int = Field(default=20, ge=5, le=50)
    guidanceScale: float = Field(default=1.5, ge=1.0, le=3.0)
    seed: int = Field(default=1247, description="Set to -1 for random.")
    enableDeepcache: bool = Field(default=True)


@app.get("/latentsync/status")
def latentsync_status() -> dict[str, object]:
    """Expose install/setup progress so I can poll from my laptop."""
    from renderer import latentsync_runner
    st = latentsync_runner.status()
    log_tail = ""
    try:
        log_tail = Path("/tmp/latentsync-setup.log").read_text()[-2000:]
    except Exception:  # noqa: BLE001
        pass
    return {**st, "log_tail": log_tail}


@app.post("/latentsync/retry_setup", dependencies=[Depends(verify_token)])
def latentsync_retry_setup() -> dict[str, object]:
    """Clear .failed/.ready sentinels and kick the setup script in background.

    Needed because we can't SSH into the pod — this is the only way to recover
    from a setup failure without a full image rebuild + pod restart.
    """
    from renderer import latentsync_runner
    for sentinel in (latentsync_runner.LATENTSYNC_READY_SENTINEL, latentsync_runner.LATENTSYNC_FAILED_SENTINEL):
        try:
            sentinel.unlink(missing_ok=True)
        except Exception as e:  # noqa: BLE001
            log.warning("unlink %s: %s", sentinel, e)
    # Fire-and-forget the setup script. Log goes to /tmp/latentsync-setup.log.
    subprocess.Popen(
        [
            "bash", "-c",
            "bash /workspace/edlio-presence/infra/latentsync/setup.sh "
            "&& touch /workspace/LatentSync/.ready "
            "|| touch /workspace/LatentSync/.failed",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    return {"ok": True, "message": "setup kicked; poll /latentsync/status"}


# In-memory job registry. LatentSync takes > Cloudflare's 100s proxy timeout
# even on warm runs, so we can't return synchronously. Kick render in a
# background thread; clients poll /latentsync/jobs/{job_id} for status.
_latentsync_jobs: dict[str, dict] = {}
_latentsync_jobs_lock = threading.Lock()


def _do_latentsync_render(job_id: str, req: "LatentSyncRenderRequest") -> None:
    """Background thread: runs the actual render, updates registry."""
    from renderer import latentsync_runner
    t_start = time.time()

    def set_state(**kwargs) -> None:
        with _latentsync_jobs_lock:
            _latentsync_jobs[job_id].update(kwargs)

    try:
        with tempfile.TemporaryDirectory(prefix=f"ls-{job_id}-") as tmpdir:
            tmp = Path(tmpdir)
            set_state(state="fetching_audio")
            audio_path = _resolve_audio(req.audio, tmp / "input.wav")

            set_state(state="fetching_video")
            video_path = tmp / "portrait.mp4"
            urllib.request.urlretrieve(req.portraitVideo, video_path)

            out_path = OUTPUT_DIR / f"{job_id}.mp4"
            set_state(state="rendering")
            with _render_lock:
                result = latentsync_runner.render(
                    video_path=video_path,
                    audio_path=audio_path,
                    output_path=out_path,
                    inference_steps=req.inferenceSteps,
                    guidance_scale=req.guidanceScale,
                    seed=req.seed,
                    enable_deepcache=req.enableDeepcache,
                )

        total = time.time() - t_start
        base = PUBLIC_BASE_URL.rstrip("/")
        video_url = (
            f"{base}/renders/{out_path.name}" if base else f"/renders/{out_path.name}"
        )
        set_state(
            state="done",
            videoUrl=video_url,
            filename=out_path.name,
            metrics={
                "total_wall_sec": round(total, 2),
                "render_sec": round(float(result["elapsed_sec"]), 2),
                "engine": "latentsync-1.6",
                "inference_steps": req.inferenceSteps,
                "guidance_scale": req.guidanceScale,
            },
        )
    except latentsync_runner.LatentSyncNotReady as e:
        set_state(state="error", error="not_ready", detail=str(e))
    except latentsync_runner.LatentSyncSetupFailed as e:
        set_state(state="error", error="setup_failed", detail=str(e))
    except subprocess.CalledProcessError as e:
        set_state(
            state="error",
            error="latentsync_failed",
            returncode=e.returncode,
            stderr_tail=(e.stderr or "")[-2000:],
        )
    except Exception as e:  # noqa: BLE001
        set_state(state="error", error="unknown", detail=repr(e))


@app.post("/render_latentsync", dependencies=[Depends(verify_token)])
def render_latentsync(req: LatentSyncRenderRequest) -> dict[str, object]:
    """Kick a LatentSync render in the background. Returns job_id; poll /latentsync/jobs/{id}.

    We don't block because Cloudflare 524s at 100s and LatentSync always
    takes longer on the first render (model warm-up) and typically on warm
    runs too (3D UNet, 20 diffusion steps).
    """
    job_id = hashlib.sha256(
        f"latentsync|{req.portraitVideo}|{time.time()}".encode()
    ).hexdigest()[:16]
    log.info("latentsync render (async) job=%s", job_id)

    with _latentsync_jobs_lock:
        _latentsync_jobs[job_id] = {
            "job_id": job_id,
            "state": "queued",
            "submitted_at": time.time(),
            "request": req.model_dump(),
        }
    threading.Thread(
        target=_do_latentsync_render,
        args=(job_id, req),
        daemon=True,
        name=f"latentsync-{job_id}",
    ).start()

    return {"ok": True, "job_id": job_id, "state": "queued"}


@app.get("/latentsync/jobs/{job_id}")
def latentsync_job_status(job_id: str) -> dict[str, object]:
    with _latentsync_jobs_lock:
        job = _latentsync_jobs.get(job_id)
    if not job:
        raise HTTPException(404, detail={"error": "unknown_job", "job_id": job_id})
    return job


@app.get("/latentsync/jobs")
def latentsync_jobs_list() -> dict[str, object]:
    with _latentsync_jobs_lock:
        return {
            "count": len(_latentsync_jobs),
            "jobs": sorted(
                _latentsync_jobs.values(),
                key=lambda j: j.get("submitted_at", 0),
                reverse=True,
            )[:50],
        }


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


def _resolve_portrait(
    url: str,
    dst: Path,
    cache_key: Optional[str],
    *,
    is_video: bool = False,
) -> Path:
    """Fetch portrait to a local file. Cached on disk if cache_key provided.

    When ``is_video=True``, caches as .mp4 in portraits/ (separate namespace).
    """
    ext = ".mp4" if is_video else ".jpg"
    if cache_key:
        cached = OUTPUT_DIR.parent / "portraits" / f"{cache_key}{ext}"
        cached.parent.mkdir(parents=True, exist_ok=True)
        if cached.exists():
            log.info("portrait cache hit: %s (ext=%s)", cache_key, ext)
            return cached
    if url.startswith(("http://", "https://")):
        log.info("fetching portrait URL: %s (is_video=%s)", url, is_video)
        urllib.request.urlretrieve(url, dst)
    else:
        if not Path(url).exists():
            raise FileNotFoundError(url)
        shutil.copy(url, dst)
    if cache_key:
        shutil.copy(dst, OUTPUT_DIR.parent / "portraits" / f"{cache_key}{ext}")
    return dst


def _get_or_create_engine(portrait_path: Path, cache_key: Optional[str]) -> object:
    """Cache one MuseTalkEngine per portraitCacheKey (legacy 256×256 face patch path)."""
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


def _get_or_create_videoref_renderer(
    video_path: Path,
    cache_key: Optional[str],
    face_box: list[int],
    extra_margin: int,
    parsing_mode: str,
    use_gfpgan: bool = False,
    gfpgan_weight: float = 0.5,
    num_ref_frames: int = 60,
) -> object:
    """Cache one MuseTalkVideoRefRenderer per (cacheKey, bbox, params, num_ref).

    Uses the ``videoref::`` prefix so it never clashes with still-portrait or
    legacy entries in the shared ``_engine_cache``.
    """
    key = "videoref::" + "|".join([
        cache_key or str(video_path),
        ",".join(str(v) for v in face_box),
        str(extra_margin),
        parsing_mode,
        f"gfp={use_gfpgan}:{gfpgan_weight:.2f}",
        f"nref={num_ref_frames}",
    ])
    if key in _engine_cache:
        log.info("videoref renderer cache hit: %s", key)
        return _engine_cache[key]

    from renderer.composite_videoref import MuseTalkVideoRefRenderer  # local import

    log.info(
        "building videoref renderer for video=%s bbox=%s extra_margin=%d parsing_mode=%s gfpgan=%s num_ref=%d",
        video_path, face_box, extra_margin, parsing_mode, use_gfpgan, num_ref_frames,
    )
    renderer_ = MuseTalkVideoRefRenderer(
        video_path=str(video_path),
        face_box=face_box,
        device="cuda",
        model_dir=MODEL_DIR,
        use_float16=True,
        extra_margin=extra_margin,
        parsing_mode=parsing_mode,
        use_gfpgan=use_gfpgan,
        gfpgan_weight=gfpgan_weight,
        num_ref_frames=num_ref_frames,
    )
    _engine_cache[key] = renderer_
    return renderer_


def _get_or_create_composite_renderer(
    portrait_path: Path,
    cache_key: Optional[str],
    face_box: list[int],
    extra_margin: int,
    parsing_mode: str,
    use_gfpgan: bool = False,
    gfpgan_weight: float = 0.5,
) -> object:
    """Cache one MuseTalkCompositeRenderer per (portraitCacheKey, bbox, parsing_mode, gfpgan).

    Distinct cache namespace from the legacy engine so both render paths can
    coexist during rollout. GFPGAN-enabled and GFPGAN-disabled renderers are
    cached separately so toggling at request time just swaps cache entries.
    """
    key = "composite::" + "|".join([
        cache_key or str(portrait_path),
        ",".join(str(v) for v in face_box),
        str(extra_margin),
        parsing_mode,
        f"gfp={use_gfpgan}:{gfpgan_weight:.2f}",
    ])
    if key in _engine_cache:
        log.info("composite renderer cache hit: %s", key)
        return _engine_cache[key]

    from renderer.composite import MuseTalkCompositeRenderer  # local import

    log.info(
        "building composite renderer for portrait=%s bbox=%s extra_margin=%d parsing_mode=%s gfpgan=%s",
        portrait_path, face_box, extra_margin, parsing_mode, use_gfpgan,
    )
    renderer_ = MuseTalkCompositeRenderer(
        portrait_path=str(portrait_path),
        face_box=face_box,
        device="cuda",
        model_dir=MODEL_DIR,
        use_float16=True,
        extra_margin=extra_margin,
        parsing_mode=parsing_mode,
        use_gfpgan=use_gfpgan,
        gfpgan_weight=gfpgan_weight,
    )
    _engine_cache[key] = renderer_
    return renderer_


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
    """Render all frames, returning plain numpy RGB arrays.

    Accepts either the legacy MuseTalkEngine (returns RendererFrame wrapping
    a 256×256 face patch) or MuseTalkCompositeRenderer (returns a full-portrait
    np.ndarray directly).
    """
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
