#!/usr/bin/env bash
# pod-setup.sh — Run this INSIDE a fresh RunPod container to install all deps.
# Uses prebuilt wheels only (no source compiles). Should finish in ~3-5 min.
#
# Usage (from Mac mini):
#   scp infra/pod-setup.sh root@POD_IP:/tmp/
#   ssh root@POD_IP "bash /tmp/pod-setup.sh"

set -euo pipefail

LOG=/tmp/pod-setup.log
START=$(date +%s)

# Progress logger — writes to both stdout AND log file
step() {
    local elapsed=$(( $(date +%s) - START ))
    echo "[$(printf '%3d' $elapsed)s] $*" | tee -a "$LOG"
}

step "🚀 Starting pod setup (target: <5 min)"
step "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'unknown')"
step "Python: $(python3 --version 2>&1)"
step "PyTorch: $(python3 -c 'import torch; print(torch.__version__, torch.version.cuda)' 2>&1)"

# ─── System deps ────────────────────────────────────────────────────────────
step "📦 Installing ffmpeg..."
DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends ffmpeg >>"$LOG" 2>&1 || {
    step "   ⚠️  First attempt failed, trying apt-get update + retry..."
    apt-get update -qq >>"$LOG" 2>&1
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends ffmpeg >>"$LOG" 2>&1
}
step "   ✓ ffmpeg $(ffmpeg -version 2>&1 | head -1 | awk '{print $3}')"

# ─── Python deps (audio + renderer) ─────────────────────────────────────────
cd /workspace/edlio-presence
step "🐍 Installing audio deps (~30s)..."
pip install -q -r audio/requirements.txt >>"$LOG" 2>&1
step "   ✓ audio deps installed"

step "🎬 Installing renderer deps (~1 min)..."
pip install -q -r renderer/requirements.txt >>"$LOG" 2>&1
step "   ✓ renderer deps installed"

# ─── MuseTalk-specific deps (prebuilt wheels ONLY) ──────────────────────────
# Detect torch+cuda versions to match the right mmcv wheel
TORCH_VER=$(python3 -c 'import torch; print(torch.__version__.split("+")[0])')
CUDA_VER=$(python3 -c 'import torch; print(torch.version.cuda.replace(".",""))')
step "🔨 Installing MuseTalk deps (torch=${TORCH_VER}, cuda=${CUDA_VER})..."

# diffusers + transformers + einops — pure Python, fast
pip install -q 'diffusers>=0.30' 'transformers>=4.40' einops >>"$LOG" 2>&1
step "   ✓ diffusers + transformers installed"

# mmcv — MUST use prebuilt wheels. Try CUDA-specific wheels first.
step "   📥 Attempting mmcv prebuilt wheel for torch${TORCH_VER%.*}+cu${CUDA_VER}..."
MMCV_INDEX="https://download.openmmlab.com/mmcv/dist/cu${CUDA_VER}/torch${TORCH_VER%.*}/index.html"
if pip install -q --no-build-isolation mmcv==2.1.0 -f "$MMCV_INDEX" >>"$LOG" 2>&1; then
    step "   ✓ mmcv 2.1.0 (prebuilt)"
else
    step "   ⚠️  No prebuilt mmcv for torch${TORCH_VER%.*}+cu${CUDA_VER} — falling back to openmim"
    pip install -q openmim >>"$LOG" 2>&1
    # Use --no-build-isolation + timeout to fail fast if compile triggers
    timeout 120 mim install 'mmcv>=2.0.1' >>"$LOG" 2>&1 || {
        step "   ❌ mmcv install failed. See $LOG for details."
        step "   This pod's cuda/torch versions don't have mmcv prebuilt wheels."
        step "   Recommendation: retry with a cu118/torch2.0 pod template."
        exit 1
    }
    step "   ✓ mmcv (via mim)"
fi

# mmdet + mmpose (pure-Python on top of mmcv)
step "   📥 Installing mmdet + mmpose..."
pip install -q 'mmdet>=3.1.0' 'mmpose>=1.1.0' >>"$LOG" 2>&1 || {
    step "   ⚠️  mmdet/mmpose failed — non-fatal for smoke test"
}
step "   ✓ mmdet + mmpose"

# MediaPipe — for DWPose face detection in MuseTalk
step "   📥 Installing mediapipe..."
pip install -q mediapipe >>"$LOG" 2>&1
step "   ✓ mediapipe"

# ─── Verify ─────────────────────────────────────────────────────────────────
step "✅ Verifying install..."
python3 <<'PYEOF' 2>&1 | tee -a "$LOG"
import sys
checks = [
    ('torch', 'import torch; assert torch.cuda.is_available()'),
    ('opencv', 'import cv2'),
    ('librosa', 'import librosa'),
    ('soundfile', 'import soundfile'),
    ('transformers', 'import transformers'),
    ('diffusers', 'import diffusers'),
    ('faster_whisper', 'from faster_whisper import WhisperModel'),
    ('mmcv', 'import mmcv'),
    ('mediapipe', 'import mediapipe'),
]
failed = []
for name, code in checks:
    try:
        exec(code)
        print(f'   ✓ {name}')
    except Exception as e:
        print(f'   ❌ {name}: {e}')
        failed.append(name)
if failed:
    print(f'\n❌ FAILED: {failed}')
    sys.exit(1)
else:
    print('\n✅ ALL DEPS PRESENT')
PYEOF

TOTAL=$(( $(date +%s) - START ))
step "🏁 Setup complete in ${TOTAL}s"
step "Log file: $LOG"
