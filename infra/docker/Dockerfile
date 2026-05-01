# syntax=docker/dockerfile:1.6
#
# edlio-presence — GPU renderer image
#
# Base: RunPod's PyTorch 2.4 + CUDA 12.4 image (the one we proved works in Day 2).
# This image installs all Python deps + MuseTalk system libs, but NOT the 8.6GB
# weights — those are fetched at first startup by scripts/fetch_weights.sh.
#
# Build:
#   docker build -f infra/docker/Dockerfile -t ghcr.io/kemalarsan/edlio-presence:day2 .
#
# Push:
#   docker push ghcr.io/kemalarsan/edlio-presence:day2
#
# Run (on a RunPod with GPU):
#   # the image entrypoint will fetch weights if they're not already present
#
# Tag conventions:
#   day2      — first working image (this one)
#   latest    — same as most recent dayN
#   slim      — no weights (current default, pulled at boot)
#   fat       — weights pre-baked (future: slower push, faster cold start)

FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

LABEL org.opencontainers.image.source="https://github.com/kemalarsan/edlio-presence"
LABEL org.opencontainers.image.description="edlio-presence GPU renderer — torch 2.4 + diffusers 0.30.2 + MuseTalk V15"
LABEL org.opencontainers.image.licenses="MIT"

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    RENDERER_MODEL_DIR=/workspace/edlio-presence/models \
    HF_HOME=/workspace/.cache/huggingface

# System deps — ffmpeg for audio/video, espeak-ng for test audio generation.
# build-essential + cmake for any pip packages that need to compile.
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
        espeak-ng \
        curl \
        git \
        build-essential \
        cmake \
    && rm -rf /var/lib/apt/lists/*

# Pin hf_hub + hf CLI BEFORE pip installs so the right versions are resolved.
# The critical triangle we learned in Day 2:
#   diffusers 0.30.2  ← needs hf_hub < 1.0
#   transformers 4.44.2  ← works with torch 2.4 (no torch.load safety check)
#   huggingface_hub 0.26.x  ← has `hf` CLI (new) AND `cached_download` (old diffusers path)
RUN pip install --upgrade pip setuptools wheel && \
    pip install "huggingface_hub[cli]==0.26.5"

# Several legacy packages (chumpy, mmcv) have setup.py files that assume
# `pip`/`pkg_resources` are importable in the build env. Modern pip uses
# isolated build envs where they aren't. Install them first with
# --no-build-isolation so they use the outer env's setuptools.
#
# Order matters: chumpy < mmcv because they're small and standalone.
RUN pip install --no-build-isolation "chumpy==0.70" && \
    pip install --no-build-isolation "mmcv==2.1.0" --no-deps

# Project-level python deps. Using the lockfile from a running Day-2 pod.
# Pass --no-build-isolation globally as belt-and-suspenders; the base image
# already has setuptools/wheel, so no isolation is needed.
COPY infra/docker/requirements-pod.txt /tmp/requirements-pod.txt
RUN pip install --no-build-isolation -r /tmp/requirements-pod.txt

# Workspace setup
WORKDIR /workspace/edlio-presence

# Copy application code (but NOT the weights — they're .gitignored anyway)
COPY . /workspace/edlio-presence

# Make sure models/ symlink exists (models → renderer/muse_talk_vendor/models)
# This is the path MuseTalk hardcodes internally.
RUN ln -sf renderer/muse_talk_vendor/models /workspace/edlio-presence/models

# Make helper scripts executable
RUN chmod +x /workspace/edlio-presence/infra/docker/entrypoint.sh \
             /workspace/edlio-presence/infra/docker/fetch_weights.sh || true

ENTRYPOINT ["/workspace/edlio-presence/infra/docker/entrypoint.sh"]
CMD ["bash"]
