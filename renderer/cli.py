#!/usr/bin/env python3
"""
renderer/cli.py — Command-line test tool for the MuseTalk engine

Usage:
    # GPU mode (requires model weights at /models/)
    python renderer/cli.py --portrait assets/tenedos-face-v1.png \
                           --audio /tmp/test.wav \
                           --out /tmp/out.mp4

    # Stub mode (CPU-safe, no weights needed — for integration testing)
    python renderer/cli.py --portrait assets/tenedos-face-v1.png \
                           --audio /tmp/test.wav \
                           --out /tmp/out.mp4 \
                           --stub

    # Custom model dir (override /models/ default)
    python renderer/cli.py --portrait assets/tenedos-face-v1.png \
                           --audio /tmp/test.wav \
                           --out /tmp/out.mp4 \
                           --model-dir /tmp/musetalk-weights

    # Show first N frames as PNGs (debug)
    python renderer/cli.py --portrait assets/tenedos-face-v1.png \
                           --audio /tmp/test.wav \
                           --frames-dir /tmp/debug-frames \
                           --max-frames 5 \
                           --stub

Run from the repo root:
    cd /Users/tenedos/edlio-presence
    python -m renderer.cli --portrait assets/tenedos-face-v1.png --audio ... --stub
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path

# Ensure renderer package is importable when run directly
_repo_root = Path(__file__).parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from renderer.audio_prep import wav_to_feature_chunks
from renderer.engine import MuseTalkEngine

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("renderer.cli")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="edlio-presence renderer CLI — test MuseTalk frame generation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--portrait",
        required=True,
        metavar="PATH",
        help="Path to face portrait image (PNG/JPG). Any resolution; will be resized to 256×256.",
    )
    p.add_argument(
        "--audio",
        required=True,
        metavar="PATH",
        help="Path to WAV file (16kHz mono preferred; resampled automatically if not).",
    )
    p.add_argument(
        "--out",
        metavar="PATH",
        default="/tmp/renderer-out.mp4",
        help="Output MP4 path. Default: /tmp/renderer-out.mp4",
    )
    p.add_argument(
        "--frames-dir",
        metavar="DIR",
        default=None,
        help="If set, also save individual frames as PNG files in this directory.",
    )
    p.add_argument(
        "--max-frames",
        type=int,
        default=None,
        metavar="N",
        help="Stop after rendering N frames (useful for quick smoke tests).",
    )
    p.add_argument(
        "--fps",
        type=float,
        default=25.0,
        help="Output frame rate. Default: 25.",
    )
    p.add_argument(
        "--stub",
        action="store_true",
        help="Use stub renderer (no GPU / weights needed). For integration testing.",
    )
    p.add_argument(
        "--device",
        default="cuda",
        help="PyTorch device string. Default: cuda.",
    )
    p.add_argument(
        "--model-dir",
        default=None,
        metavar="DIR",
        help="Path to MuseTalk model weights dir. Default: /models (Docker mount).",
    )
    p.add_argument(
        "--chunk-seconds",
        type=float,
        default=2.0,
        help="Audio chunk duration in seconds per batch. Default: 2.0.",
    )
    p.add_argument(
        "--no-float16",
        action="store_true",
        help="Disable FP16 (use FP32). Useful if you see NaN outputs on specific GPUs.",
    )
    p.add_argument(
        "--verbose",
        action="store_true",
        help="Enable DEBUG logging.",
    )
    return p.parse_args()


def frames_to_mp4(frames: list, out_path: str, fps: float) -> None:
    """
    Encode a list of numpy RGB frames to MP4 using ffmpeg via subprocess.

    We avoid moviepy to keep dependencies minimal; ffmpeg is available in
    the Docker container (infra/Dockerfile installs it).
    """
    import subprocess  # noqa: PLC0415
    import tempfile  # noqa: PLC0415

    import cv2  # noqa: PLC0415

    # Write frames to a temp dir as PNG sequence
    with tempfile.TemporaryDirectory(prefix="renderer-frames-") as tmpdir:
        for i, frame in enumerate(frames):
            # Convert RGB → BGR for cv2 write
            frame_bgr = frame[:, :, ::-1]
            cv2.imwrite(os.path.join(tmpdir, f"frame_{i:06d}.png"), frame_bgr)

        cmd = [
            "ffmpeg", "-y",
            "-framerate", str(fps),
            "-i", os.path.join(tmpdir, "frame_%06d.png"),
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-preset", "fast",
            out_path,
        ]
        logger.info("Encoding %d frames → %s", len(frames), out_path)
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            logger.error("ffmpeg failed:\n%s", result.stderr)
            raise RuntimeError(f"ffmpeg error: {result.returncode}")
        logger.info("Saved: %s", out_path)


def save_frames_png(frames: list, out_dir: str) -> None:
    import cv2  # noqa: PLC0415

    os.makedirs(out_dir, exist_ok=True)
    for i, frame in enumerate(frames):
        cv2.imwrite(os.path.join(out_dir, f"frame_{i:06d}.png"), frame[:, :, ::-1])
    logger.info("Saved %d PNGs to %s", len(frames), out_dir)


def main() -> int:
    args = parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # ── Validate inputs ──────────────────────────────────────────────────────
    if not Path(args.portrait).exists():
        logger.error("Portrait not found: %s", args.portrait)
        return 1

    if not Path(args.audio).exists():
        logger.error("Audio file not found: %s", args.audio)
        return 1

    # ── Init engine ──────────────────────────────────────────────────────────
    logger.info(
        "Initializing MuseTalkEngine: portrait=%s, stub=%s, device=%s",
        args.portrait, args.stub, args.device,
    )
    engine = MuseTalkEngine(
        portrait_path=args.portrait,
        device=args.device,
        use_stub=args.stub if args.stub else None,  # None = auto-detect
        model_dir=args.model_dir,
        use_float16=not args.no_float16,
        fps=args.fps,
    )

    if not engine.is_stub:
        logger.info("Warming up GPU kernels …")
        engine.warm_up()

    logger.info("Engine ready: %s", engine)

    # ── Render frames ────────────────────────────────────────────────────────
    all_frames = []
    t0 = time.perf_counter()
    frame_count = 0

    audio_chunks = wav_to_feature_chunks(
        args.audio,
        chunk_seconds=args.chunk_seconds,
        device=args.device if not args.stub else "cpu",
        use_stub=args.stub,
        fps=args.fps,
    )

    logger.info("Rendering …")
    for chunk in audio_chunks:
        for i in range(chunk.num_frames()):
            rendered = engine.render_frame(chunk, frame_index=i)
            all_frames.append(rendered.frame)
            frame_count += 1

            if args.max_frames and frame_count >= args.max_frames:
                logger.info("Reached --max-frames %d, stopping early.", args.max_frames)
                break

        if args.max_frames and frame_count >= args.max_frames:
            break

    elapsed = time.perf_counter() - t0
    fps_achieved = frame_count / elapsed if elapsed > 0 else 0.0

    logger.info(
        "Rendered %d frames in %.2fs (%.1f fps%s)",
        frame_count,
        elapsed,
        fps_achieved,
        " [STUB]" if engine.is_stub else "",
    )

    if not all_frames:
        logger.error("No frames rendered — check audio file and portrait.")
        return 1

    # ── Save frames PNG (optional) ────────────────────────────────────────────
    if args.frames_dir:
        save_frames_png(all_frames, args.frames_dir)

    # ── Encode to MP4 ────────────────────────────────────────────────────────
    try:
        frames_to_mp4(all_frames, args.out, fps=args.fps)
        logger.info("✅  Done → %s  (%d frames, %.1f fps achieved)", args.out, frame_count, fps_achieved)
    except Exception as exc:
        logger.error("MP4 encoding failed: %s", exc)
        # Still save PNGs as fallback
        fallback_dir = args.out.replace(".mp4", "-frames")
        logger.info("Saving frames as PNGs to %s …", fallback_dir)
        save_frames_png(all_frames, fallback_dir)
        logger.info("Fallback frames saved to: %s", fallback_dir)

    return 0


if __name__ == "__main__":
    sys.exit(main())
