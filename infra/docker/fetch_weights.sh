#!/usr/bin/env bash
# Fetch all MuseTalk weights + their configs.
#
# Derived from the (broken) upstream download_weights.sh, rewritten to:
#   1. Use huggingface_hub's CLI via `python -m …`, since the `hf` /
#      `huggingface-cli` console-script entry points don't always survive
#      a dist-packages snapshot (they live in /usr/local/bin/, which our
#      snapshot tarball doesn't include).
#   2. Explicitly fetch config.json files (download --include does NOT
#      auto-grab configs for most repos).
#   3. Use ManyOtherFunctions/face-parse-bisent on HF instead of the
#      upstream Google-Drive URL, which is headless-hostile.
#
# Called by entrypoint.sh on first boot, OR manually:
#   bash infra/docker/fetch_weights.sh
set -euo pipefail

# Portable replacement for `hf download …` that works without the console-script.
hf_download() {
    python -m huggingface_hub.commands.huggingface_cli download "$@"
}

MODELS_DIR="${RENDERER_MODEL_DIR:-/workspace/edlio-presence/renderer/muse_talk_vendor/models}"
mkdir -p "${MODELS_DIR}"
cd "${MODELS_DIR}"

echo "[weights] target: ${MODELS_DIR}"

# ─── MuseTalk V1.5 (the one we actually use) ──────────────────────────────────
echo "[weights] MuseTalk V1.5…"
hf_download TMElyralab/MuseTalk --include "musetalkV15/*" --local-dir . 2>&1 | tail -5
# Configs that `--include` didn't grab
curl -sL https://huggingface.co/TMElyralab/MuseTalk/resolve/main/musetalkV15/musetalk.json \
     -o musetalkV15/musetalk.json

# ─── MuseTalk V1 (kept for compatibility) ─────────────────────────────────────
echo "[weights] MuseTalk V1…"
hf_download TMElyralab/MuseTalk --include "musetalk/*" --local-dir . 2>&1 | tail -3
curl -sL https://huggingface.co/TMElyralab/MuseTalk/resolve/main/musetalk/musetalk.json \
     -o musetalk/musetalk.json

# ─── Stable Diffusion VAE ─────────────────────────────────────────────────────
echo "[weights] sd-vae…"
mkdir -p sd-vae
hf_download stabilityai/sd-vae-ft-mse --include "diffusion_pytorch_model.bin" --local-dir sd-vae 2>&1 | tail -3
curl -sL https://huggingface.co/stabilityai/sd-vae-ft-mse/resolve/main/config.json \
     -o sd-vae/config.json

# ─── Whisper tiny (audio encoder) ─────────────────────────────────────────────
echo "[weights] whisper tiny…"
mkdir -p whisper
hf_download openai/whisper-tiny --include "pytorch_model.bin" --local-dir whisper 2>&1 | tail -3
for f in config.json preprocessor_config.json tokenizer.json vocab.json merges.txt \
         special_tokens_map.json generation_config.json; do
    curl -sfL "https://huggingface.co/openai/whisper-tiny/resolve/main/${f}" -o "whisper/${f}" \
        || echo "[weights] warn: couldn't fetch whisper/${f}"
done

# ─── DWPose (face keypoint detection) ─────────────────────────────────────────
echo "[weights] dwpose…"
hf_download yzd-v/DWPose --include "dw-ll_ucoco_384.pth" --local-dir dwpose 2>&1 | tail -3

# ─── SyncNet (audio-visual sync loss, used during training only) ──────────────
echo "[weights] syncnet…"
hf_download ByteDance/LatentSync --include "latentsync_syncnet.pt" --local-dir syncnet 2>&1 | tail -3 || \
    echo "[weights] warn: syncnet fetch failed (optional — training only)"

# ─── Face parsing (BiSeNet for mouth masking) ─────────────────────────────────
echo "[weights] face-parse-bisent…"
mkdir -p face-parse-bisent
# Mirror on HF has both files — upstream's Google-Drive URL is headless-hostile.
# (The same mirror is used by the project's download_weights.bat for Windows.)
hf_download ManyOtherFunctions/face-parse-bisent \
    --include "resnet18-5c106cde.pth" "79999_iter.pth" \
    --local-dir face-parse-bisent 2>&1 | tail -3 \
    || echo "[weights] warn: face-parse-bisent fetch from HF mirror failed"

echo "[weights] done:"
du -sh "${MODELS_DIR}"/*/
