#!/usr/bin/env bash
# build-push.sh — Build the edlio-presence Docker image and push to GHCR
#
# Usage:
#   bash infra/build-push.sh [--tag <tag>] [--push] [--no-cache]
#
# Prerequisites:
#   docker login ghcr.io -u <github-username> -p <PAT with packages:write>
#
# Environment:
#   REGISTRY   (default: ghcr.io/kemalarsan)
#   IMAGE_NAME (default: edlio-presence)
#   TAG        (default: latest, override with --tag or TAG env var)

set -euo pipefail

REGISTRY="${REGISTRY:-ghcr.io/kemalarsan}"
IMAGE_NAME="${IMAGE_NAME:-edlio-presence}"
TAG="${TAG:-latest}"
PUSH=false
NO_CACHE=""

# ── Parse args ────────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
  case $1 in
    --tag)       TAG="$2"; shift 2 ;;
    --push)      PUSH=true; shift ;;
    --no-cache)  NO_CACHE="--no-cache"; shift ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

FULL_IMAGE="${REGISTRY}/${IMAGE_NAME}:${TAG}"

# ── Must run from repo root ────────────────────────────────────────────────────
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

echo "→ Building ${FULL_IMAGE}"
echo "  Context: $(pwd)"
echo ""

docker build \
  ${NO_CACHE} \
  -f infra/Dockerfile \
  -t "${FULL_IMAGE}" \
  -t "${REGISTRY}/${IMAGE_NAME}:dev" \
  --label "org.opencontainers.image.source=https://github.com/kemalarsan/edlio-presence" \
  --label "org.opencontainers.image.created=$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
  --label "org.opencontainers.image.revision=$(git rev-parse --short HEAD 2>/dev/null || echo unknown)" \
  .

echo ""
echo "✅  Build complete: ${FULL_IMAGE}"

if [[ "$PUSH" == "true" ]]; then
  echo ""
  echo "→ Pushing to registry..."
  docker push "${FULL_IMAGE}"
  docker push "${REGISTRY}/${IMAGE_NAME}:dev"
  echo "✅  Pushed: ${FULL_IMAGE}"
  echo ""
  echo "RunPod image URL to use in runpod-launch.sh:"
  echo "  export IMAGE=${FULL_IMAGE}"
else
  echo ""
  echo "Not pushing (pass --push to push to GHCR)."
  echo "To push manually:"
  echo "  docker push ${FULL_IMAGE}"
fi
