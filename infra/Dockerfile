# edlio-presence: MuseTalk renderer container
#
# Requirements verified from TMElyralab/MuseTalk (April 2026):
#   - Python 3.10
#   - CUDA 11.8 (cu118)
#   - PyTorch 2.0.1
#
# Base: nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04
# Using -devel (not -runtime) so we can compile mmcv/mmpose native extensions.
# Swap to -runtime after baking wheels if image size matters.

FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

LABEL maintainer="edlio-presence"
LABEL description="MuseTalk lip-sync renderer — GPU inference node"

# ── System packages ──────────────────────────────────────────────────────────
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 python3.10-dev python3.10-distutils python3-pip \
    ffmpeg \
    libgl1-mesa-glx libglib2.0-0 libsm6 libxext6 libxrender-dev \
    libgomp1 \
    git wget curl ca-certificates \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Make python3.10 the default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 \
 && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1 \
 && python -m pip install --upgrade pip setuptools wheel

# ── PyTorch (cu118, matching MuseTalk requirements) ───────────────────────────
RUN pip install --no-cache-dir \
    torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 \
    --index-url https://download.pytorch.org/whl/cu118

# ── MuseTalk Python requirements ─────────────────────────────────────────────
# Pin versions exactly as declared in MuseTalk/requirements.txt
RUN pip install --no-cache-dir \
    diffusers==0.30.2 \
    accelerate==0.28.0 \
    "numpy==1.23.5" \
    tensorflow==2.12.0 \
    tensorboard==2.12.0 \
    "opencv-python==4.9.0.80" \
    soundfile==0.12.1 \
    transformers==4.39.2 \
    "huggingface_hub==0.30.2" \
    librosa==0.11.0 \
    einops==0.8.1 \
    gradio==5.24.0 \
    gdown \
    requests \
    "imageio[ffmpeg]" \
    omegaconf \
    ffmpeg-python \
    moviepy

# ── MMLab ecosystem (mmcv/mmdet/mmpose — required for DWPose) ─────────────────
RUN pip install --no-cache-dir -U openmim \
 && mim install mmengine \
 && mim install "mmcv==2.0.1" \
 && mim install "mmdet==3.1.0" \
 && mim install "mmpose==1.1.0"

# ── MediaPipe (for client-side gaze — included here for completeness) ─────────
RUN pip install --no-cache-dir mediapipe

# ── Working directory + healthcheck script ───────────────────────────────────
WORKDIR /workspace

COPY infra/hello-gpu.py /workspace/hello-gpu.py

# Healthcheck: runs the GPU verifier every 60s (passes if exit 0)
HEALTHCHECK --interval=60s --timeout=30s --start-period=120s --retries=2 \
    CMD python /workspace/hello-gpu.py --quiet || exit 1

# Default: drop into bash so operators can poke around
CMD ["bash"]
