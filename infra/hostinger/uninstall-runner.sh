#!/usr/bin/env bash
# uninstall-runner.sh — Remove the self-hosted GitHub Actions runner from this VPS.
#
# Usage:
#   export GH_RUNNER_TOKEN="<removal token from GitHub Settings → Actions → Runners>"
#   bash uninstall-runner.sh
#
# Get the removal token from:
#   GitHub → repo → Settings → Actions → Runners → click the runner → Remove runner
#
# What this does NOT touch:
#   - Docker (leave installed — harmless, and you might re-register the runner later)
#   - /var/lib/edlio-presence/ (leave the tarball — expensive to re-upload)
#
# Requires: root or passwordless sudo.

set -euo pipefail

# ─── Colour helpers ───────────────────────────────────────────────────────────
RED='\033[0;31m'
GRN='\033[0;32m'
YLW='\033[1;33m'
NC='\033[0m'
info() { echo -e "${GRN}[uninstall-runner]${NC} $*"; }
warn() { echo -e "${YLW}[uninstall-runner] WARN:${NC} $*"; }
die()  { echo -e "${RED}[uninstall-runner] ERROR:${NC} $*" >&2; exit 1; }

# ─── Validate env vars ────────────────────────────────────────────────────────
[[ -z "${GH_RUNNER_TOKEN:-}" ]] && die "GH_RUNNER_TOKEN is not set. Get a removal token from the runner's page in GitHub Settings → Actions → Runners."

# ─── Must run as root ─────────────────────────────────────────────────────────
if [[ "${EUID}" -ne 0 ]]; then
    die "Please run as root (or with: sudo -E bash uninstall-runner.sh)"
fi

RUNNER_DIR="/opt/github-runner"
RUNNER_USER="github-runner"

# ─── 1. Stop + uninstall the systemd service ──────────────────────────────────
if [[ -f "${RUNNER_DIR}/svc.sh" ]]; then
    info "Stopping systemd service…"
    pushd "${RUNNER_DIR}" > /dev/null
    ./svc.sh stop  || warn "Service stop returned non-zero (may already be stopped)."
    ./svc.sh uninstall || warn "Service uninstall returned non-zero (may already be gone)."
    popd > /dev/null
else
    warn "${RUNNER_DIR}/svc.sh not found — skipping service removal."
fi

# ─── 2. Unregister the runner from GitHub ─────────────────────────────────────
if [[ -f "${RUNNER_DIR}/config.sh" ]]; then
    info "Unregistering runner from GitHub…"
    sudo -u "${RUNNER_USER}" "${RUNNER_DIR}/config.sh" remove \
        --token "${GH_RUNNER_TOKEN}" \
        || warn "Runner unregistration failed (token may be expired or runner already removed)."
else
    warn "${RUNNER_DIR}/config.sh not found — skipping GitHub unregistration."
fi

# ─── 3. Remove the runner directory ───────────────────────────────────────────
if [[ -d "${RUNNER_DIR}" ]]; then
    info "Removing ${RUNNER_DIR}…"
    rm -rf "${RUNNER_DIR}"
else
    warn "${RUNNER_DIR} does not exist. Nothing to remove."
fi

# ─── 4. Optionally remove the github-runner user ──────────────────────────────
if id "${RUNNER_USER}" &>/dev/null; then
    info "Removing user '${RUNNER_USER}'…"
    userdel -r "${RUNNER_USER}" 2>/dev/null || warn "Could not fully remove user (home dir may have already been deleted)."
else
    warn "User '${RUNNER_USER}' not found. Skipping."
fi

# ─── Done ─────────────────────────────────────────────────────────────────────
echo ""
echo -e "${GRN}✓ Runner uninstalled.${NC}"
echo ""
echo "  Kept: Docker CE (still installed)"
echo "  Kept: /var/lib/edlio-presence/ (tarball preserved)"
echo ""
echo "To re-register later, get a new token and re-run install-runner.sh."
echo ""
