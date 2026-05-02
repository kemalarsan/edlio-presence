#!/usr/bin/env bash
# infra/aniportrait/setup.sh — One-shot setup for AniPortrait on the pod.
#
# AniPortrait (Tencent, 2024) is image+audio→video. Unlike LatentSync (which
# re-lip-syncs an existing video), AniPortrait *synthesizes* head motion and
# lip movement from a still reference + audio. We want it for the A/B/C sweep
# to see if pure-synthesis head motion ever beats our preserved-footage
# approach.
#
# Isolated venv at /opt/aniportrait-venv — keeps deps separate from MuseTalk
# and LatentSync.
#
# Run once on the pod:
#   bash infra/aniportrait/setup.sh
#
# Expected disk footprint:
#   ~3 GB   venv with torch 2.5 + deps
#   ~5 GB   StableDiffusion v1.5 unet (denoising_unet.pth, reference_unet.pth)
#   ~2 GB   motion_module + pose_guider + audio2pose/mesh + film_net
#   ~0.5 GB wav2vec2-base-960h
#   ~0.3 GB sd-vae-ft-mse
#   ~0.3 GB image_encoder
#   = ~11 GB total

set -euo pipefail

REPO_DIR="/workspace/AniPortrait"
VENV_DIR="/opt/aniportrait-venv"
WEIGHTS_DIR="${REPO_DIR}/pretrained_weights"
LOG="/tmp/aniportrait-setup.log"

log() {
    local msg="[aniportrait-setup $(date +%H:%M:%S)] $*"
    echo "$msg" | tee -a "$LOG"
}

log "=== starting ==="

# ─── 1. clone repo ──────────────────────────────────────────────────────────
if [[ -d "$REPO_DIR/.git" ]]; then
    log "repo already cloned at $REPO_DIR; pulling latest"
    (cd "$REPO_DIR" && git pull --ff-only) >>"$LOG" 2>&1
else
    log "cloning AniPortrait → $REPO_DIR"
    git clone --depth 1 https://github.com/Zejun-Yang/AniPortrait.git "$REPO_DIR" >>"$LOG" 2>&1
fi

# ─── 2. create venv ─────────────────────────────────────────────────────────
if [[ ! -d "$VENV_DIR" ]]; then
    log "creating venv at $VENV_DIR (~30s)"
    python3 -m venv "$VENV_DIR" >>"$LOG" 2>&1
    "$VENV_DIR/bin/pip" install --upgrade pip setuptools wheel >>"$LOG" 2>&1
fi

VPIP="$VENV_DIR/bin/pip"
VPY="$VENV_DIR/bin/python"

# ─── 3. install torch CUDA 12.1 (4090-compatible) ───────────────────────────
if ! "$VPY" -c "import torch; assert torch.__version__.startswith('2.5')" 2>/dev/null; then
    log "installing torch 2.5.1 + CUDA 12.1 wheels (~1.5 GB, ~90s)"
    "$VPIP" install --no-cache-dir \
        torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
        --extra-index-url https://download.pytorch.org/whl/cu121 >>"$LOG" 2>&1
fi

# ─── 4. install AniPortrait deps ────────────────────────────────────────────
if ! "$VPY" -c "import diffusers, pkg_resources" 2>/dev/null; then
    log "installing AniPortrait python deps (~2 min)"
    # Their requirements.txt pins many old versions. We use modern versions
    # that are compatible with torch 2.5.
    # setuptools<81 still exposes pkg_resources, which librosa 0.10 imports.
    "$VPIP" install --no-cache-dir \
        'setuptools<81' \
        diffusers==0.24.0 transformers==4.36.2 \
        accelerate==0.25.0 einops==0.7.0 \
        omegaconf==2.3.0 opencv-python==4.9.0.80 \
        mediapipe==0.10.11 python_speech_features==0.6 \
        librosa==0.10.1 av==12.0.0 \
        ffmpeg-python==0.2.0 imageio==2.33.1 imageio-ffmpeg==0.4.9 \
        scikit-image==0.22.0 scikit-learn==1.3.2 \
        huggingface-hub==0.23.2 numpy==1.26.4 \
        xformers==0.0.28.post3 safetensors==0.4.2 \
        soundfile==0.12.1 >>"$LOG" 2>&1
fi

# ─── 5. download weights ────────────────────────────────────────────────────
mkdir -p "$WEIGHTS_DIR"

# AniPortrait trained weights (~5 GB total)
ANI_FILES=(
    denoising_unet.pth
    reference_unet.pth
    pose_guider.pth
    motion_module.pth
    audio2mesh.pt
    audio2pose.pt
    film_net_fp16.pt
)
NEED_ANI_DOWNLOAD=0
for f in "${ANI_FILES[@]}"; do
    if [[ ! -s "$WEIGHTS_DIR/$f" ]]; then
        NEED_ANI_DOWNLOAD=1
        break
    fi
done
if [[ "$NEED_ANI_DOWNLOAD" == "1" ]]; then
    log "downloading AniPortrait checkpoints from ZJYang/AniPortrait (~5 GB, 2-5 min)"
    "$VPY" - <<PY >>"$LOG" 2>&1
from huggingface_hub import hf_hub_download
files = ${ANI_FILES[@]@Q}  # unused in python, but kept for logging
files = ["denoising_unet.pth", "reference_unet.pth", "pose_guider.pth",
         "motion_module.pth", "audio2mesh.pt", "audio2pose.pt", "film_net_fp16.pt"]
for fn in files:
    path = hf_hub_download(
        repo_id="ZJYang/AniPortrait",
        filename=fn,
        local_dir="${WEIGHTS_DIR}",
    )
    print(f"downloaded: {path}")
PY
fi

# Base models (StableDiffusion V1.5, sd-vae-ft-mse, image_encoder, wav2vec)
log "downloading base models (StableDiffusion unet, VAE, image encoder, wav2vec) ~4 GB"
"$VPY" - <<PY >>"$LOG" 2>&1
from huggingface_hub import snapshot_download
import os

base = "${WEIGHTS_DIR}"

# StableDiffusion V1.5 — only need unet + model_index + feature_extractor
# (full repo is 40+ GB — we can't afford to grab it all)
# Actually their animation config only references these parts:
print("fetching stable-diffusion-v1-5 unet + model_index")
snapshot_download(
    repo_id="runwayml/stable-diffusion-v1-5",
    local_dir=os.path.join(base, "stable-diffusion-v1-5"),
    allow_patterns=[
        "model_index.json",
        "unet/config.json",
        "unet/diffusion_pytorch_model.bin",
        "feature_extractor/preprocessor_config.json",
        "v1-inference.yaml",
    ],
)

# sd-vae-ft-mse
print("fetching sd-vae-ft-mse")
snapshot_download(
    repo_id="stabilityai/sd-vae-ft-mse",
    local_dir=os.path.join(base, "sd-vae-ft-mse"),
    allow_patterns=[
        "config.json",
        "diffusion_pytorch_model.bin",
        "diffusion_pytorch_model.safetensors",
    ],
)

# image_encoder (from lambda sd-image-variations)
print("fetching image_encoder")
snapshot_download(
    repo_id="lambdalabs/sd-image-variations-diffusers",
    local_dir=os.path.join(base, "__image_enc_tmp"),
    allow_patterns=[
        "image_encoder/config.json",
        "image_encoder/pytorch_model.bin",
    ],
)
# Move image_encoder up a level
import shutil
src = os.path.join(base, "__image_enc_tmp", "image_encoder")
dst = os.path.join(base, "image_encoder")
if os.path.exists(src) and not os.path.exists(dst):
    shutil.move(src, dst)
shutil.rmtree(os.path.join(base, "__image_enc_tmp"), ignore_errors=True)

# wav2vec2
print("fetching wav2vec2-base-960h")
snapshot_download(
    repo_id="facebook/wav2vec2-base-960h",
    local_dir=os.path.join(base, "wav2vec2-base-960h"),
    allow_patterns=[
        "config.json",
        "feature_extractor_config.json",
        "preprocessor_config.json",
        "pytorch_model.bin",
        "special_tokens_map.json",
        "tokenizer_config.json",
        "vocab.json",
    ],
    ignore_patterns=["README.md", "*.md"],
)

print("all base models fetched")
PY

# ─── 6. symlink pretrained_model ─ pretrained_weights ─────────────────────
# Their code has two directory naming conventions mixed in:
#   - README says pretrained_weights/
#   - frame_interpolation.py hard-codes ./pretrained_model/film_net_fp16.pt
# We download to pretrained_weights/ (per README) and symlink pretrained_model
# to it so hard-coded paths resolve.
if [[ ! -e "$REPO_DIR/pretrained_model" ]]; then
    log "symlinking pretrained_model → pretrained_weights (their code has both names)"
    (cd "$REPO_DIR" && ln -s pretrained_weights pretrained_model) >>"$LOG" 2>&1
fi

# ─── 7. sanity: can we import? ──────────────────────────────────────────
log "sanity check: importing scripts.audio2vid module"
cd "$REPO_DIR"
"$VPY" -c "
import sys
sys.path.insert(0, '.')
# Just import the minimal set — full import tries to load weights
from src.models.unet_2d_condition import UNet2DConditionModel
from src.models.unet_3d import UNet3DConditionModel
print('imports OK')
" >>"$LOG" 2>&1 || log "WARN: sanity import failed, may need additional deps"

log "=== done ==="
log "repo:     $REPO_DIR"
log "venv:     $VENV_DIR"
log "weights:  $WEIGHTS_DIR"
log "disk:     $(du -sh "$REPO_DIR" 2>/dev/null | cut -f1)"
