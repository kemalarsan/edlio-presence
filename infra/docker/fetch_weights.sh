#!/usr/bin/env bash
# Fetch all MuseTalk weights + their configs.
#
# Derived from the (broken) upstream download_weights.sh, rewritten to:
#   1. Use the modern `hf download` CLI (huggingface-cli is deprecated)
#   2. Explicitly fetch config.json files (hf download --include does NOT auto-grab configs)
#
# Called by entrypoint.sh on first boot, OR manually:
#   bash infra/docker/fetch_weights.sh
set -euo pipefail

MODELS_DIR="${RENDERER_MODEL_DIR:-/workspace/edlio-presence/renderer/muse_talk_vendor/models}"
mkdir -p "${MODELS_DIR}"
cd "${MODELS_DIR}"

echo "[weights] target: ${MODELS_DIR}"

# ─── MuseTalk V1.5 (the one we actually use) ──────────────────────────────────
echo "[weights] MuseTalk V1.5…"
hf download TMElyralab/MuseTalk --include "musetalkV15/*" --local-dir . 2>&1 | tail -5
# Configs that `--include` didn't grab
curl -sL https://huggingface.co/TMElyralab/MuseTalk/resolve/main/musetalkV15/musetalk.json \
     -o musetalkV15/musetalk.json

# ─── MuseTalk V1 (kept for compatibility) ─────────────────────────────────────
echo "[weights] MuseTalk V1…"
hf download TMElyralab/MuseTalk --include "musetalk/*" --local-dir . 2>&1 | tail -3
curl -sL https://huggingface.co/TMElyralab/MuseTalk/resolve/main/musetalk/musetalk.json \
     -o musetalk/musetalk.json

# ─── Stable Diffusion VAE ─────────────────────────────────────────────────────
echo "[weights] sd-vae…"
mkdir -p sd-vae
hf download stabilityai/sd-vae-ft-mse --include "diffusion_pytorch_model.bin" --local-dir sd-vae 2>&1 | tail -3
curl -sL https://huggingface.co/stabilityai/sd-vae-ft-mse/resolve/main/config.json \
     -o sd-vae/config.json

# ─── Whisper tiny (audio encoder) ─────────────────────────────────────────────
echo "[weights] whisper tiny…"
mkdir -p whisper
hf download openai/whisper-tiny --include "pytorch_model.bin" --local-dir whisper 2>&1 | tail -3
for f in config.json preprocessor_config.json tokenizer.json vocab.json merges.txt \
         special_tokens_map.json generation_config.json; do
    curl -sfL "https://huggingface.co/openai/whisper-tiny/resolve/main/${f}" -o "whisper/${f}" \
        || echo "[weights] warn: couldn't fetch whisper/${f}"
done

# ─── DWPose (face keypoint detection) ─────────────────────────────────────────
echo "[weights] dwpose…"
hf download yzd-v/DWPose --include "dw-ll_ucoco_384.pth" --local-dir dwpose 2>&1 | tail -3

# ─── SyncNet (audio-visual sync loss, used during training only) ──────────────
echo "[weights] syncnet…"
hf download ByteDance/LatentSync --include "latentsync_syncnet.pt" --local-dir syncnet 2>&1 | tail -3 || \
    echo "[weights] warn: syncnet fetch failed (optional — training only)"

# ─── Face parsing (BiSeNet for mouth masking) ─────────────────────────────────
echo "[weights] face-parse-bisent…"
mkdir -p face-parse-bisent
curl -sfL https://github.com/zllrunning/face-parsing.PyTorch/releases/download/v0.0.1/resnet18-5c106cde.pth \
     -o face-parse-bisent/resnet18-5c106cde.pth \
    || echo "[weights] warn: face-parse-bisent resnet fetch failed"
curl -sfL "https://drive.google.com/uc?id=154JgKpzCPW82qINcVieuPH3fZ2e0P812&export=download" \
     -o face-parse-bisent/79999_iter.pth \
    || echo "[weights] warn: face-parse-bisent 79999_iter.pth requires manual download from Google Drive"

echo "[weights] done:"
du -sh "${MODELS_DIR}"/*/
