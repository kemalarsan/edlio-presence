#!/usr/bin/env bash
# runpod-launch.sh — Launch a RunPod persistent pod for edlio-presence
#
# Prerequisites:
#   export RUNPOD_API_KEY=<your key>
#   export IMAGE=<registry/image:tag>   # e.g. ghcr.io/kemalarsan/edlio-presence:latest
#   export GPU_TYPE="NVIDIA L40S"       # or "NVIDIA GeForce RTX 4090"
#
# Usage:
#   bash infra/runpod-launch.sh
#
# This script calls the RunPod GraphQL API to create a pod.
# It does NOT start automatically — review the printed pod ID, then use
# the RunPod dashboard or API to start it.

set -euo pipefail

: "${RUNPOD_API_KEY:?ERROR: RUNPOD_API_KEY is not set}"
: "${IMAGE:?ERROR: IMAGE is not set (e.g. ghcr.io/kemalarsan/edlio-presence:latest)}"
GPU_TYPE="${GPU_TYPE:-NVIDIA L40S}"
POD_NAME="${POD_NAME:-edlio-presence-dev}"
CONTAINER_DISK_GB="${CONTAINER_DISK_GB:-50}"
VOLUME_DISK_GB="${VOLUME_DISK_GB:-100}"
# Volume mount path inside container — models + output land here
VOLUME_MOUNT_PATH="${VOLUME_MOUNT_PATH:-/models}"

RUNPOD_API="https://api.runpod.io/graphql?api_key=${RUNPOD_API_KEY}"

echo "→ Launching RunPod pod"
echo "  GPU type:  ${GPU_TYPE}"
echo "  Image:     ${IMAGE}"
echo "  Pod name:  ${POD_NAME}"
echo ""

# GraphQL mutation to create a persistent pod
# Docs: https://docs.runpod.io/sdks/graphql/manage-pods
MUTATION=$(cat <<EOF
mutation {
  podFindAndDeployOnDemand(input: {
    name: "${POD_NAME}"
    imageName: "${IMAGE}"
    gpuTypeId: "${GPU_TYPE}"
    cloudType: SECURE
    containerDiskInGb: ${CONTAINER_DISK_GB}
    volumeInGb: ${VOLUME_DISK_GB}
    volumeMountPath: "${VOLUME_MOUNT_PATH}"
    startSsh: true
    env: [
      { key: "HF_TOKEN", value: "${HF_TOKEN:-}" }
    ]
    ports: "8765/http,22/tcp"
  }) {
    id
    name
    desiredStatus
    imageName
    machineId
    costPerHr
    runtime {
      uptimeInSeconds
      gpus { id gpuUtilPercent memoryUtilPercent }
    }
  }
}
EOF
)

RESPONSE=$(curl -s -X POST "${RUNPOD_API}" \
  -H "Content-Type: application/json" \
  -d "$(jq -n --arg q "$MUTATION" '{query: $q}')")

echo "$RESPONSE" | jq .

POD_ID=$(echo "$RESPONSE" | jq -r '.data.podFindAndDeployOnDemand.id // empty')
COST=$(echo "$RESPONSE"   | jq -r '.data.podFindAndDeployOnDemand.costPerHr // "unknown"')

if [[ -z "$POD_ID" ]]; then
  echo ""
  echo "❌  Pod creation failed — see response above"
  exit 1
fi

echo ""
echo "✅  Pod created: ${POD_ID}"
echo "    Cost/hr:     \$${COST}"
echo ""
echo "Next steps:"
echo "  1. Watch pod start: https://www.runpod.io/console/pods"
echo "  2. SSH in (once running):"
echo "       runpodctl exec ${POD_ID} -- python /workspace/hello-gpu.py"
echo "  3. Or via SSH tunnel — grab the SSH command from the RunPod dashboard"
echo ""
echo "  To stop the pod (avoid charges):"
echo "    curl -s -X POST '${RUNPOD_API}' \\"
echo "      -H 'Content-Type: application/json' \\"
echo "      -d '{\"query\": \"mutation { podStop(input: { podId: \\\"${POD_ID}\\\" }) { id desiredStatus } }\"}'"
