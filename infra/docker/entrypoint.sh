#!/usr/bin/env bash
# Entrypoint for the edlio-presence pod image.
#
# On first boot (or if weights are missing), fetch them.
# Then exec whatever CMD was passed.
set -euo pipefail

MODEL_DIR="${RENDERER_MODEL_DIR:-/workspace/edlio-presence/renderer/muse_talk_vendor/models}"
WEIGHT_SENTINEL="${MODEL_DIR}/musetalkV15/unet.pth"

# If a network volume is mounted at the model dir, it might have been populated
# by a previous boot. If not, or the sentinel is missing, fetch all weights.
if [[ ! -f "${WEIGHT_SENTINEL}" ]]; then
    echo "[edlio-presence] weights missing at ${MODEL_DIR}, fetching…"
    /workspace/edlio-presence/infra/docker/fetch_weights.sh
else
    echo "[edlio-presence] weights already present at ${MODEL_DIR} — skipping fetch"
fi

# MuseTalk's face_parsing loader hard-codes `./models/...`, so pytest/renderer
# CWD needs a `models/` symlink pointing at the actual weights dir.
WORK_DIR=/workspace/edlio-presence
if [[ ! -e "${WORK_DIR}/models" ]]; then
    ln -sf "${MODEL_DIR}" "${WORK_DIR}/models"
fi

cd "${WORK_DIR}"
echo "[edlio-presence] ready (cwd=$(pwd)). starting: $*"
exec "$@"
