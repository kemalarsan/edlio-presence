"""
renderer/aniportrait_runner.py

Subprocess-based wrapper for Tencent/Zhiji AniPortrait (2024).

AniPortrait is image+audio→video (pure synthesis of head motion + lip). It
runs in its own venv at /opt/aniportrait-venv to avoid clobbering MuseTalk
and LatentSync.

Their CLI wants a YAML config that points to a reference image + audio, so
we generate one per-render into a tmpdir and point their script at it.

Contract:
    inputs:
        image_path     — reference portrait (square-ish, face 50-70% of frame)
        audio_path     — WAV, English, clean vocals
    output:
        output_path    — MP4 in workspace OUTPUT_DIR
    knobs:
        width, height  — output resolution (their default is 512×512)
        num_frames     — clip length in frames (25fps → seconds × 25)
        seed           — RNG
        fp16_accel     — use film_net_fp16 frame-interpolation (faster)

Note on output format: AniPortrait's audio2vid writes into
    {their output dir}/{ref stem}_{audio stem}_W{W}_H{H}_L{L}_S{seed}.mp4
We let them write to a tmpdir, then move to our canonical output_path.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

import yaml

log = logging.getLogger(__name__)

ANIPORTRAIT_REPO = Path("/workspace/AniPortrait")
ANIPORTRAIT_VENV_PY = Path("/opt/aniportrait-venv/bin/python")
ANIPORTRAIT_READY_SENTINEL = ANIPORTRAIT_REPO / ".ready"
ANIPORTRAIT_FAILED_SENTINEL = ANIPORTRAIT_REPO / ".failed"


class AniPortraitNotReady(RuntimeError):
    """Setup hasn't finished yet."""


class AniPortraitSetupFailed(RuntimeError):
    """Setup script errored — see /tmp/aniportrait-setup.log."""


def status() -> dict[str, object]:
    repo_exists = ANIPORTRAIT_REPO.exists()
    venv_exists = ANIPORTRAIT_VENV_PY.exists()
    unet_ckpt = ANIPORTRAIT_REPO / "pretrained_weights/denoising_unet.pth"
    weights_exist = unet_ckpt.is_file() and unet_ckpt.stat().st_size > 100_000_000
    ready = ANIPORTRAIT_READY_SENTINEL.exists()
    failed = ANIPORTRAIT_FAILED_SENTINEL.exists()
    return {
        "repo": repo_exists,
        "venv": venv_exists,
        "weights": weights_exist,
        "ready": ready,
        "failed": failed,
    }


def assert_ready() -> None:
    st = status()
    if st["failed"]:
        log_tail = ""
        try:
            log_tail = Path("/tmp/aniportrait-setup.log").read_text()[-4000:]
        except Exception:  # noqa: BLE001
            pass
        raise AniPortraitSetupFailed(
            "AniPortrait setup failed. Tail of /tmp/aniportrait-setup.log:\n" + log_tail
        )
    if not st["ready"] or not st["weights"]:
        raise AniPortraitNotReady(
            f"AniPortrait not ready yet. Status: {st}. "
            "Set PRESENCE_ENABLE_ANIPORTRAIT=1 and wait for "
            "/workspace/AniPortrait/.ready (first boot takes 5-10 min)."
        )


def _build_config(
    *,
    image_path: Path,
    audio_path: Path,
    seed: int,
    num_frames: int,
    fp16_accel: bool,
) -> dict:
    """
    AniPortrait's audio2vid config format (from their
    configs/prompts/animation_audio.yaml). We generate one per render.
    """
    base = ANIPORTRAIT_REPO / "pretrained_weights"
    return {
        "pretrained_base_model_path": str(base / "stable-diffusion-v1-5"),
        "pretrained_vae_path": str(base / "sd-vae-ft-mse"),
        "image_encoder_path": str(base / "image_encoder"),
        "denoising_unet_path": str(base / "denoising_unet.pth"),
        "reference_unet_path": str(base / "reference_unet.pth"),
        "pose_guider_path": str(base / "pose_guider.pth"),
        "motion_module_path": str(base / "motion_module.pth"),
        "audio_model_path": str(base / "wav2vec2-base-960h"),
        "audio2mesh_path": str(base / "audio2mesh.pt"),
        "audio2pose_path": str(base / "audio2pose.pt"),
        "inference_config": "./configs/inference/inference_v2.yaml",
        "weight_dtype": "fp16",
        "test_cases": {
            str(image_path): [str(audio_path)],
        },
        "L": num_frames,   # num_frames
        "seed": seed,
    }


def render(
    *,
    image_path: Path,
    audio_path: Path,
    output_path: Path,
    width: int = 512,
    height: int = 512,
    seed: int = 42,
    num_frames: int = 200,   # ~8s at 25fps
    fp16_accel: bool = True,
    timeout_sec: int = 1800,  # 30 min — AniPortrait is slow (diffusion)
) -> dict[str, object]:
    """Run AniPortrait's audio2vid.

    Returns dict with elapsed + stdout tail.

    Raises AniPortraitNotReady / AniPortraitSetupFailed / subprocess.CalledProcessError.
    """
    assert_ready()

    for p in (image_path, audio_path):
        if not p.is_file():
            raise FileNotFoundError(str(p))
    output_path.parent.mkdir(parents=True, exist_ok=True)

    config = _build_config(
        image_path=image_path,
        audio_path=audio_path,
        seed=seed,
        num_frames=num_frames,
        fp16_accel=fp16_accel,
    )

    # Write config + run audio2vid in a temp dir so we can find the output
    # (AniPortrait writes into ./output/<timestamp>/)
    with tempfile.TemporaryDirectory(prefix="ap-run-") as tmp:
        tmpdir = Path(tmp)
        cfg_path = tmpdir / "run.yaml"
        with open(cfg_path, "w") as f:
            yaml.safe_dump(config, f)

        cmd = [
            str(ANIPORTRAIT_VENV_PY),
            "-m", "scripts.audio2vid",
            "--config", str(cfg_path),
            "-W", str(width),
            "-H", str(height),
        ]
        if fp16_accel:
            cmd.append("-acc")

        log.info("aniportrait cmd: %s", " ".join(cmd))

        env = os.environ.copy()
        env["PYTHONPATH"] = str(ANIPORTRAIT_REPO)
        env.pop("PYTHONHOME", None)

        t0 = time.time()
        proc = subprocess.run(
            cmd,
            cwd=str(ANIPORTRAIT_REPO),
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout_sec,
        )
        elapsed = time.time() - t0

        if proc.returncode != 0:
            log.error(
                "aniportrait failed code=%d elapsed=%.1fs\nstderr tail:\n%s",
                proc.returncode, elapsed, (proc.stderr or "")[-3000:],
            )
            raise subprocess.CalledProcessError(
                proc.returncode, cmd, output=proc.stdout, stderr=proc.stderr,
            )

        # AniPortrait dumps outputs under ./output/<date>/ with a specific
        # filename. Find the newest mp4 that appeared during this run.
        candidates = sorted(
            (ANIPORTRAIT_REPO / "output").rglob("*.mp4"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if not candidates:
            raise RuntimeError(
                f"aniportrait exit 0 but no mp4 found under {ANIPORTRAIT_REPO/'output'}"
            )
        newest = candidates[0]
        # Safety: only take files modified during this call window
        if newest.stat().st_mtime < t0 - 5:
            raise RuntimeError(
                f"aniportrait newest mp4 ({newest}) is older than render start "
                f"({t0}); their script probably didn't write a new file."
            )
        shutil.move(str(newest), str(output_path))

    return {
        "elapsed_sec": elapsed,
        "stdout_tail": (proc.stdout or "")[-1500:],
        "stderr_tail": (proc.stderr or "")[-500:],
    }


def purge_cache() -> None:
    p = ANIPORTRAIT_REPO / "output"
    if p.exists():
        shutil.rmtree(p, ignore_errors=True)
