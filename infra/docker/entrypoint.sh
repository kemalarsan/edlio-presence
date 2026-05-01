#!/usr/bin/env bash
# Entrypoint for the edlio-presence pod image.
#
# On first boot (or if weights are missing), fetch them.
# Then exec whatever CMD was passed.
set -euo pipefail

WEIGHT_SENTINEL=/workspace/edlio-presence/renderer/muse_talk_vendor/models/musetalkV15/unet.pth

if [[ ! -f "${WEIGHT_SENTINEL}" ]]; then
    echo "[edlio-presence] weights missing, fetching…"
    /workspace/edlio-presence/infra/docker/fetch_weights.sh
else
    echo "[edlio-presence] weights already present — skipping fetch"
fi

echo "[edlio-presence] ready. starting: $*"
exec "$@"
