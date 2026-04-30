"""
First real end-to-end render:  audio.wav  →  whisper features  →  MuseTalk  →  frames  →  MP4

Usage:
    python3 scripts/render_demo.py <audio.wav> <portrait.png> <output.mp4>
"""

import sys
import os
import time
import subprocess
from pathlib import Path

import numpy as np
import cv2
import torch

# edlio-presence modules
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "audio"))
sys.path.insert(0, str(ROOT / "renderer" / "muse_talk_vendor"))

from audio.extractor import extract_musetalk_features
from renderer.engine import MuseTalkEngine, AudioFeatures


def main():
    audio_path = sys.argv[1]
    portrait_path = sys.argv[2]
    output_mp4 = sys.argv[3]

    model_dir = os.environ.get("RENDERER_MODEL_DIR", "/workspace/edlio-presence/models")

    print(f"[1/5] Extracting features from: {audio_path}")
    t0 = time.time()
    mt_features = extract_musetalk_features(audio_path, fps=25.0)
    # mt_features.audio_prompts shape: (N, 10, 5, 384) torch tensor
    # Track A's engine wants (T, 50, 384) np.ndarray
    # Collapse the 10*5 dim into 50 (the engine's expected sequence length)
    audio_prompts = mt_features.audio_prompts
    if isinstance(audio_prompts, torch.Tensor):
        audio_prompts = audio_prompts.detach().cpu().numpy()
    T = audio_prompts.shape[0]
    # Flatten middle dims: (T, 10, 5, 384) -> (T, 50, 384)
    whisper_features = audio_prompts.reshape(T, -1, 384).astype(np.float32)
    print(f"       {T} frames, whisper_features shape: {whisper_features.shape}  ({time.time() - t0:.1f}s)")

    audio_features = AudioFeatures(whisper_features=whisper_features, fps=25.0)

    print(f"[2/5] Loading MuseTalk engine on GPU ({model_dir})…")
    t0 = time.time()
    engine = MuseTalkEngine(
        portrait_path=portrait_path,
        device="cuda",
        use_stub=False,
        use_float16=True,
        model_dir=model_dir,
    )
    print(f"       Loaded in {time.time() - t0:.1f}s")

    print(f"[3/5] Rendering {T} frames…")
    t0 = time.time()
    frames = []
    for i in range(T):
        rf = engine.render_frame(audio_features, frame_index=i)
        # Engine returns RendererFrame wrapper; grab the ndarray
        arr = rf.frame if hasattr(rf, "frame") else rf
        frames.append(arr)
        if (i + 1) % 25 == 0:
            elapsed = time.time() - t0
            fps = (i + 1) / elapsed
            print(f"       Frame {i+1}/{T}  |  {fps:.1f} fps  |  {1000/fps:.1f} ms/frame")

    render_time = time.time() - t0
    print(f"       Total: {render_time:.1f}s  |  avg {T/render_time:.1f} fps  |  {1000*render_time/T:.1f} ms/frame")

    print(f"[4/5] Writing frames → video")
    h, w = frames[0].shape[:2]
    tmp_noaudio = output_mp4.replace(".mp4", "_noaudio.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(tmp_noaudio, fourcc, 25.0, (w, h))
    for f in frames:
        writer.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
    writer.release()

    print(f"[5/5] Muxing audio → {output_mp4}")
    subprocess.run(
        ["ffmpeg", "-y", "-i", tmp_noaudio, "-i", audio_path, "-c:v", "libx264",
         "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", output_mp4],
        check=True, capture_output=True,
    )
    os.remove(tmp_noaudio)

    size_mb = os.path.getsize(output_mp4) / 1e6
    print(f"\n✅ DONE  →  {output_mp4}  ({size_mb:.2f} MB)")


if __name__ == "__main__":
    main()
