#!/usr/bin/env bash
# infra/latentsync/setup.sh — One-shot setup for LatentSync-1.6 on the pod.
#
# We keep LatentSync in an isolated venv because its requirements clobber
# MuseTalk's (torch 2.5 vs 2.4, diffusers 0.32 vs 0.30, numpy 1.26 vs 1.23).
# Both renderers must coexist on the same pod so we can A/B them.
#
# Idempotent — safe to re-run.
#
# Run once on the pod (from /workspace/edlio-presence):
#   bash infra/latentsync/setup.sh
#
# Expected disk footprint:
#   ~3 GB   venv with torch 2.5 + deps
#   ~3 GB   latentsync_unet.pt
#   ~0.2 GB whisper small.pt
#   ~0.3 GB sd-vae-ft-mse (HuggingFace cache)
#   = ~6.5 GB total
set -euo pipefail

REPO_DIR="/workspace/LatentSync"
VENV_DIR="/opt/latentsync-venv"
CKPT_DIR="${REPO_DIR}/checkpoints"
LOG="/tmp/latentsync-setup.log"

log() {
    local msg="[latentsync-setup $(date +%H:%M:%S)] $*"
    echo "$msg" | tee -a "$LOG"
}

log "=== starting ==="

# ─── 1. clone repo ──────────────────────────────────────────────────────────
if [[ -d "$REPO_DIR/.git" ]]; then
    log "repo already cloned at $REPO_DIR; pulling latest"
    (cd "$REPO_DIR" && git pull --ff-only) >>"$LOG" 2>&1
else
    log "cloning LatentSync → $REPO_DIR"
    git clone --depth 1 https://github.com/bytedance/LatentSync.git "$REPO_DIR" >>"$LOG" 2>&1
fi

# ─── 2. create venv ─────────────────────────────────────────────────────────
if [[ ! -d "$VENV_DIR" ]]; then
    log "creating venv at $VENV_DIR (~30s)"
    python3 -m venv "$VENV_DIR" >>"$LOG" 2>&1
    "$VENV_DIR/bin/pip" install --upgrade pip setuptools wheel >>"$LOG" 2>&1
fi

VPIP="$VENV_DIR/bin/pip"
VPY="$VENV_DIR/bin/python"

# ─── 3. install torch with CUDA 12.1 (matches their index-url) ──────────────
# We check by importing — pip install --upgrade is safe but slow if already done.
if ! "$VPY" -c "import torch; assert torch.__version__.startswith('2.5')" 2>/dev/null; then
    log "installing torch 2.5.1 + CUDA 12.1 wheels (~1.5 GB, ~90s)"
    "$VPIP" install --no-cache-dir \
        torch==2.5.1 torchvision==0.20.1 \
        --extra-index-url https://download.pytorch.org/whl/cu121 >>"$LOG" 2>&1
fi

# ─── 4. install their deps ──────────────────────────────────────────────────
if ! "$VPY" -c "import diffusers; assert diffusers.__version__ == '0.32.2'" 2>/dev/null; then
    log "installing LatentSync python deps (~2 min)"
    # Use their requirements.txt but skip torch/torchvision (already done above)
    # and skip gradio (we're a headless server).
    "$VPIP" install --no-cache-dir \
        diffusers==0.32.2 transformers==4.48.0 \
        decord==0.6.0 accelerate==0.26.1 einops==0.7.0 \
        omegaconf==2.3.0 opencv-python==4.9.0.80 \
        mediapipe==0.10.11 python_speech_features==0.6 \
        librosa==0.10.1 scenedetect==0.6.1 \
        ffmpeg-python==0.2.0 imageio==2.31.1 imageio-ffmpeg==0.5.1 \
        lpips==0.1.4 face-alignment==1.4.1 \
        huggingface-hub==0.30.2 numpy==1.26.4 \
        kornia==0.8.0 insightface==0.7.3 \
        onnxruntime-gpu==1.21.0 DeepCache==0.1.1 >>"$LOG" 2>&1
fi

# ─── 5. download weights ────────────────────────────────────────────────────
mkdir -p "$CKPT_DIR" "$CKPT_DIR/whisper"

UNET_PT="$CKPT_DIR/latentsync_unet.pt"
if [[ ! -s "$UNET_PT" ]]; then
    log "downloading latentsync_unet.pt from ByteDance/LatentSync-1.6 (~3 GB, 2-4 min)"
    # HF repo layout: ByteDance/LatentSync-1.6/latentsync_unet.pt
    "$VPY" - <<'PY' >>"$LOG" 2>&1
from huggingface_hub import hf_hub_download
path = hf_hub_download(
    repo_id="ByteDance/LatentSync-1.6",
    filename="latentsync_unet.pt",
    local_dir="/workspace/LatentSync/checkpoints",
)
print(f"downloaded: {path}")
PY
fi

WHISPER_PT="$CKPT_DIR/whisper/small.pt"
if [[ ! -s "$WHISPER_PT" ]]; then
    # LatentSync-1.6 uses whisper *small* (cross_attention_dim=768).
    # 1.5 used tiny (384). We're going for 1.6 quality.
    log "downloading whisper small.pt (~1 GB)"
    "$VPY" - <<'PY' >>"$LOG" 2>&1
from huggingface_hub import hf_hub_download
path = hf_hub_download(
    repo_id="ByteDance/LatentSync-1.6",
    filename="whisper/small.pt",
    local_dir="/workspace/LatentSync/checkpoints",
)
print(f"downloaded: {path}")
PY
fi

# ─── 6. pre-fetch sd-vae-ft-mse so first render doesn't stall ───────────────
log "pre-fetching sd-vae-ft-mse"
"$VPY" - <<'PY' >>"$LOG" 2>&1 || true
from diffusers import AutoencoderKL
AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse")
print("vae cached")
PY

# ─── 7. sanity: can we import their pipeline? ───────────────────────────────
log "sanity check: importing latentsync.pipelines.lipsync_pipeline"
cd "$REPO_DIR"
"$VPY" -c "
import sys
sys.path.insert(0, '/workspace/LatentSync')
from latentsync.pipelines.lipsync_pipeline import LipsyncPipeline
from latentsync.models.unet import UNet3DConditionModel
print('OK')
" >>"$LOG" 2>&1

log "=== done ==="
log "repo:     $REPO_DIR"
log "venv:     $VENV_DIR"
log "unet:     $UNET_PT ($(du -h "$UNET_PT" 2>/dev/null | cut -f1))"
log "whisper:  $WHISPER_PT ($(du -h "$WHISPER_PT" 2>/dev/null | cut -f1))"
log "Now you can render with: python -m scripts.inference --unet_config_path configs/unet/stage2_512.yaml ..."
