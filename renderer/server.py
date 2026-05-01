"""
renderer/server.py — FastAPI wrapper for the MuseTalk pipeline.

Status: SCAFFOLD (2026-04-30). Not deployed yet.

Run locally (on a GPU pod launched from ghcr.io/kemalarsan/edlio-presence:day2-snapshot):

    cd /workspace/edlio-presence
    uvicorn renderer.server:app --host 0.0.0.0 --port 8080

The Vercel-side shim at /api/presence/render (in tenedos-voice) calls /render.

Endpoints:
    GET  /healthz         — liveness check (used by tenedos-voice probe)
    POST /render          — synchronous render: { audio, portrait } → { videoUrl }
    GET  /render/{job_id} — async render status (future; not implemented in v0)

Auth: optional bearer token via env RENDERER_AUTH_TOKEN.
"""

from __future__ import annotations

import base64
import hashlib
import logging
import os
import tempfile
import time
from pathlib import Path
from typing import Optional

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel, Field

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s - %(message)s")

app = FastAPI(title="edlio-presence renderer", version="0.1.0")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
RENDERER_AUTH_TOKEN = os.environ.get("RENDERER_AUTH_TOKEN")
OUTPUT_DIR = Path(os.environ.get("RENDERER_OUTPUT_DIR", "/workspace/renders"))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def verify_token(authorization: Optional[str] = Header(default=None)) -> None:
    """Bearer-token auth. If RENDERER_AUTH_TOKEN isn't set, allow all (dev mode)."""
    if not RENDERER_AUTH_TOKEN:
        return
    expected = f"Bearer {RENDERER_AUTH_TOKEN}"
    if authorization != expected:
        raise HTTPException(status_code=401, detail="invalid or missing bearer token")


# ---------------------------------------------------------------------------
# /healthz
# ---------------------------------------------------------------------------
@app.get("/healthz")
def healthz() -> dict[str, str | bool]:
    return {
        "ok": True,
        "version": "0.1.0",
        "engine": "musetalk-v1.5",
    }


# ---------------------------------------------------------------------------
# /render
# ---------------------------------------------------------------------------
class RenderRequest(BaseModel):
    audio: str = Field(..., description="Base64 PCM16 mono 16kHz, OR http(s) URL to wav/mp3")
    portrait: str = Field(..., description="http(s) URL to a portrait image (PNG/JPEG)")
    portraitCacheKey: Optional[str] = Field(
        default=None,
        description="Stable key so repeated renders of the same portrait reuse the preprocessed latents",
    )
    fps: int = Field(default=25, ge=5, le=60, description="Target frames per second")


class RenderResponse(BaseModel):
    ok: bool
    videoUrl: str
    durationSec: float
    metrics: dict[str, float | str]


@app.post("/render", response_model=RenderResponse, dependencies=[Depends(verify_token)])
def render(req: RenderRequest) -> RenderResponse:
    """
    Synchronous render. Blocks until the MP4 is written, then returns a URL
    pointing at our static /renders mount.

    v0 implementation notes:
      - audio can be either base64 PCM or a URL; resolve to a local wav first
      - portrait is always a URL (clients should upload once, pass the URL)
      - portraitCacheKey is hashed+saved so we can memoize preprocessing
      - the actual pipeline call is STUBBED with NotImplementedError — Day 3
        wires it to renderer.engine.MuseTalkEngine (already written and tested)
    """
    start = time.time()
    job_id = hashlib.sha256(
        f"{req.audio[:64]}|{req.portrait}|{req.portraitCacheKey}|{time.time()}".encode()
    ).hexdigest()[:16]
    out_path = OUTPUT_DIR / f"{job_id}.mp4"

    log.info("render job=%s portrait=%s cacheKey=%s", job_id, req.portrait, req.portraitCacheKey)

    # TODO(day3): implement the real pipeline below.
    #
    # 1. Resolve audio:
    #      if req.audio.startswith("http"):
    #          audio_path = _fetch_to_temp(req.audio, suffix=".wav")
    #      else:
    #          audio_path = _b64_to_wav(req.audio)
    #
    # 2. Resolve portrait:
    #      portrait_path = _fetch_to_temp(req.portrait, suffix=".jpg")
    #
    # 3. Run the engine:
    #      from renderer.engine import MuseTalkEngine
    #      engine = _get_or_create_engine(req.portraitCacheKey, portrait_path)
    #      engine.render(audio_path, out_path, fps=req.fps)
    #
    # 4. Return videoUrl pointing at /renders/{job_id}.mp4

    raise HTTPException(
        status_code=501,
        detail={
            "error": "not_implemented",
            "message": "renderer.server.render is scaffold; wire to renderer.engine on Day 3",
            "job_id": job_id,
            "out_path": str(out_path),
        },
    )


# ---------------------------------------------------------------------------
# Static file serving for /renders/*.mp4
# ---------------------------------------------------------------------------
@app.get("/renders/{filename}")
def get_render(filename: str) -> FileResponse:
    path = OUTPUT_DIR / filename
    if not path.exists() or not path.suffix == ".mp4":
        raise HTTPException(status_code=404, detail="not found")
    return FileResponse(path, media_type="video/mp4")


# ---------------------------------------------------------------------------
# Entrypoint helpers (for `python -m renderer.server` debugging)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn

    uvicorn.run("renderer.server:app", host="0.0.0.0", port=8080, log_level="info")
