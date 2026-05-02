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

# ─── LatentSync setup (optional, run in background) ──────────────────────────
# If PRESENCE_ENABLE_LATENTSYNC=1, kick off setup.sh in the background so the
# server can start immediately while LatentSync's 6.5 GB download runs.
# Status visible in /healthz as `latentsync_ready`.
if [[ "${PRESENCE_ENABLE_LATENTSYNC:-0}" == "1" ]]; then
    LS_SENTINEL="/workspace/LatentSync/checkpoints/latentsync_unet.pt"
    if [[ ! -s "${LS_SENTINEL}" ]]; then
        echo "[edlio-presence] PRESENCE_ENABLE_LATENTSYNC=1 and weights missing — running setup in background (see /tmp/latentsync-setup.log)"
        (
            bash /workspace/edlio-presence/infra/latentsync/setup.sh \
                && touch /workspace/LatentSync/.ready \
                || touch /workspace/LatentSync/.failed
        ) &
    else
        echo "[edlio-presence] LatentSync weights already present — marking ready"
        mkdir -p /workspace/LatentSync
        touch /workspace/LatentSync/.ready
    fi
fi

cd "${WORK_DIR}"
echo "[edlio-presence] ready (cwd=$(pwd)). starting: $*"
exec "$@"
