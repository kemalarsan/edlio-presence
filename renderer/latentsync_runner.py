"""
renderer/latentsync_runner.py

Subprocess-based wrapper for ByteDance LatentSync-1.6.

LatentSync uses its own torch 2.5 / diffusers 0.32 / numpy 1.26 env that
conflicts with MuseTalk's. We keep it isolated in a venv at
/opt/latentsync-venv and shell out from our FastAPI server.

Contract (matches ByteDance's inference.py CLI):
    inputs:  video_path (portrait clip), audio_path
    output:  video_out_path (MP4)
    knobs:   inference_steps (int), guidance_scale (float), seed (int | -1)

We do NOT try to bolt LatentSync into our compositor. LatentSync already:
  - detects + tracks the face
  - crops to 512×512
  - does temporal lip-sync via its 3D UNet
  - composites back into the full video (their pipeline handles this)

Our job is just: give it video+audio, hand back the MP4.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import time
from pathlib import Path

log = logging.getLogger(__name__)

LATENTSYNC_REPO = Path("/workspace/LatentSync")
LATENTSYNC_VENV_PY = Path("/opt/latentsync-venv/bin/python")
LATENTSYNC_READY_SENTINEL = LATENTSYNC_REPO / ".ready"
LATENTSYNC_FAILED_SENTINEL = LATENTSYNC_REPO / ".failed"

# LatentSync 1.6 (512×512) is what we want for Ali's 538px face.
# 1.5 is 256×256 and would be another 2× downscale trap like MuseTalk.
DEFAULT_UNET_CONFIG = LATENTSYNC_REPO / "configs/unet/stage2_512.yaml"
DEFAULT_CKPT = LATENTSYNC_REPO / "checkpoints/latentsync_unet.pt"


class LatentSyncNotReady(RuntimeError):
    """Raised when someone calls render before setup.sh has finished."""


class LatentSyncSetupFailed(RuntimeError):
    """Raised when the background setup hit a fatal error."""


def status() -> dict[str, object]:
    """Return a small dict describing install status. Used by /healthz."""
    repo_exists = LATENTSYNC_REPO.exists()
    venv_exists = LATENTSYNC_VENV_PY.exists()
    ckpt_exists = DEFAULT_CKPT.is_file() and DEFAULT_CKPT.stat().st_size > 100_000_000
    ready = LATENTSYNC_READY_SENTINEL.exists()
    failed = LATENTSYNC_FAILED_SENTINEL.exists()
    return {
        "repo": repo_exists,
        "venv": venv_exists,
        "weights": ckpt_exists,
        "ready": ready,
        "failed": failed,
    }


def assert_ready() -> None:
    st = status()
    if st["failed"]:
        log_tail = ""
        try:
            log_tail = Path("/tmp/latentsync-setup.log").read_text()[-4000:]
        except Exception:  # noqa: BLE001
            pass
        raise LatentSyncSetupFailed(
            "LatentSync setup failed. Tail of /tmp/latentsync-setup.log:\n" + log_tail
        )
    if not st["ready"] or not st["weights"]:
        raise LatentSyncNotReady(
            f"LatentSync not ready yet. Status: {st}. "
            "Set PRESENCE_ENABLE_LATENTSYNC=1 on the pod env and wait for "
            "/workspace/LatentSync/.ready (first boot takes 3-5 min)."
        )


def render(
    *,
    video_path: Path,
    audio_path: Path,
    output_path: Path,
    inference_steps: int = 20,
    guidance_scale: float = 1.5,
    seed: int = 1247,
    enable_deepcache: bool = True,
    timeout_sec: int = 600,
) -> dict[str, object]:
    """
    Run LatentSync's inference CLI in its own venv as a subprocess.

    Returns dict with timing + stdout tail for observability.

    Raises:
        LatentSyncNotReady / LatentSyncSetupFailed / subprocess.CalledProcessError
    """
    assert_ready()

    for p in (video_path, audio_path):
        if not p.is_file():
            raise FileNotFoundError(str(p))
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Run from inside the LatentSync repo so `configs/`, `scripts/`, etc. resolve.
    cmd = [
        str(LATENTSYNC_VENV_PY),
        "-m", "scripts.inference",
        "--unet_config_path", str(DEFAULT_UNET_CONFIG),
        "--inference_ckpt_path", str(DEFAULT_CKPT),
        "--inference_steps", str(inference_steps),
        "--guidance_scale", f"{guidance_scale}",
        "--seed", str(seed),
        "--video_path", str(video_path),
        "--audio_path", str(audio_path),
        "--video_out_path", str(output_path),
    ]
    if enable_deepcache:
        cmd.append("--enable_deepcache")

    log.info("latentsync cmd: %s", " ".join(cmd))

    env = os.environ.copy()
    # Make sure we're hitting CUDA not CPU, and not importing from our main torch
    env["PYTHONPATH"] = str(LATENTSYNC_REPO)
    env.pop("PYTHONHOME", None)

    t0 = time.time()
    proc = subprocess.run(
        cmd,
        cwd=str(LATENTSYNC_REPO),
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout_sec,
    )
    elapsed = time.time() - t0

    if proc.returncode != 0:
        log.error(
            "latentsync failed code=%d elapsed=%.1fs\nstderr tail:\n%s",
            proc.returncode, elapsed, (proc.stderr or "")[-3000:],
        )
        raise subprocess.CalledProcessError(
            proc.returncode, cmd, output=proc.stdout, stderr=proc.stderr,
        )

    if not output_path.is_file():
        raise RuntimeError(
            f"latentsync succeeded (code 0) but didn't write output to {output_path}"
        )

    return {
        "elapsed_sec": elapsed,
        "stdout_tail": (proc.stdout or "")[-1500:],
        "stderr_tail": (proc.stderr or "")[-500:],
    }


def purge_cache() -> None:
    """Free disk — nukes /tmp inside the repo. Idempotent."""
    for sub in ("temp", "outputs"):
        p = LATENTSYNC_REPO / sub
        if p.exists():
            shutil.rmtree(p, ignore_errors=True)
