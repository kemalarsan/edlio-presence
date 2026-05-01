#!/usr/bin/env bash
# install-runner.sh — Provision a Hostinger KVM 2 VPS as a self-hosted GitHub
# Actions runner for edlio-presence.
#
# Usage:
#   export GH_REPO="kemalarsan/edlio-presence"
#   export GH_RUNNER_TOKEN="<ephemeral token from GitHub Settings → Actions → Runners>"
#   bash install-runner.sh
#
# Idempotent: safe to re-run. Docker install is skipped if already present;
# the runner is re-registered with --replace if already configured.
#
# Requires: Ubuntu 24.04 LTS, run as root or a user with passwordless sudo.

set -euo pipefail

# ─── Colour helpers ───────────────────────────────────────────────────────────
RED='\033[0;31m'
GRN='\033[0;32m'
YLW='\033[1;33m'
NC='\033[0m'
info()  { echo -e "${GRN}[install-runner]${NC} $*"; }
warn()  { echo -e "${YLW}[install-runner] WARN:${NC} $*"; }
die()   { echo -e "${RED}[install-runner] ERROR:${NC} $*" >&2; exit 1; }

# ─── Validate required env vars ───────────────────────────────────────────────
[[ -z "${GH_REPO:-}"         ]] && die "GH_REPO is not set (e.g. kemalarsan/edlio-presence)"
[[ -z "${GH_RUNNER_TOKEN:-}" ]] && die "GH_RUNNER_TOKEN is not set (get one from https://github.com/${GH_REPO}/settings/actions/runners/new)"

# ─── Must run as root or via sudo ─────────────────────────────────────────────
if [[ "${EUID}" -ne 0 ]]; then
    die "Please run as root (or with: sudo -E bash install-runner.sh)"
fi

# ─── Detect Ubuntu 24.04 ──────────────────────────────────────────────────────
if [[ -f /etc/os-release ]]; then
    # shellcheck source=/dev/null
    source /etc/os-release
    if [[ "${ID:-}" != "ubuntu" || "${VERSION_ID:-}" != "24.04" ]]; then
        warn "Expected Ubuntu 24.04, got ${PRETTY_NAME:-unknown}. Proceeding anyway — YMMV."
    fi
else
    warn "/etc/os-release not found; skipping OS check."
fi

# ─── 1. Install Docker CE (official apt repo, not snap) ───────────────────────
if command -v docker &>/dev/null; then
    info "Docker already installed ($(docker --version)). Skipping."
else
    info "Installing Docker CE…"
    apt-get update -qq
    apt-get install -y --no-install-recommends \
        ca-certificates curl gnupg lsb-release

    install -m 0755 -d /etc/apt/keyrings
    curl -fsSL https://download.docker.com/linux/ubuntu/gpg \
        | gpg --dearmor -o /etc/apt/keyrings/docker.gpg
    chmod a+r /etc/apt/keyrings/docker.gpg

    echo \
        "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] \
https://download.docker.com/linux/ubuntu \
$(lsb_release -cs) stable" \
        | tee /etc/apt/sources.list.d/docker.list > /dev/null

    apt-get update -qq
    apt-get install -y \
        docker-ce docker-ce-cli containerd.io \
        docker-buildx-plugin docker-compose-plugin

    systemctl enable --now docker
    info "Docker CE installed: $(docker --version)"
fi

# ─── 2. Create github-runner user ─────────────────────────────────────────────
RUNNER_USER="github-runner"
if id "${RUNNER_USER}" &>/dev/null; then
    info "User '${RUNNER_USER}' already exists. Skipping creation."
else
    info "Creating user '${RUNNER_USER}'…"
    useradd --system --create-home --home-dir /opt/github-runner \
        --shell /bin/bash "${RUNNER_USER}"
fi

# Ensure the user is in the docker group
if ! id -nG "${RUNNER_USER}" | grep -qw docker; then
    info "Adding '${RUNNER_USER}' to docker group…"
    usermod -aG docker "${RUNNER_USER}"
fi

# ─── 3. Fetch latest GitHub Actions runner release ────────────────────────────
RUNNER_DIR="/opt/github-runner"
RUNNER_ARCH="linux-x64"

info "Resolving latest GitHub Actions runner release…"
LATEST_TAG=$(curl -fsSL \
    "https://api.github.com/repos/actions/runner/releases/latest" \
    | grep '"tag_name"' | sed -E 's/.*"v([^"]+)".*/\1/')
[[ -z "${LATEST_TAG}" ]] && die "Could not determine latest runner version from GitHub API."
info "Latest runner: v${LATEST_TAG}"

RUNNER_TARBALL="actions-runner-${RUNNER_ARCH}-${LATEST_TAG}.tar.gz"
RUNNER_URL="https://github.com/actions/runner/releases/download/v${LATEST_TAG}/${RUNNER_TARBALL}"

# Idempotent: skip download if same version already extracted
if [[ -f "${RUNNER_DIR}/config.sh" ]] && \
   grep -q "${LATEST_TAG}" "${RUNNER_DIR}/bin/Runner.Listener" 2>/dev/null; then
    info "Runner v${LATEST_TAG} already extracted. Skipping download."
else
    info "Downloading runner ${RUNNER_TARBALL}…"
    mkdir -p "${RUNNER_DIR}"
    curl -fsSL "${RUNNER_URL}" -o "/tmp/${RUNNER_TARBALL}"
    tar -xzf "/tmp/${RUNNER_TARBALL}" -C "${RUNNER_DIR}"
    rm -f "/tmp/${RUNNER_TARBALL}"
    chown -R "${RUNNER_USER}:${RUNNER_USER}" "${RUNNER_DIR}"
    info "Runner extracted to ${RUNNER_DIR}."
fi

# ─── 4. Configure the runner ──────────────────────────────────────────────────
info "Configuring runner (--replace handles re-registration)…"
sudo -u "${RUNNER_USER}" "${RUNNER_DIR}/config.sh" \
    --url "https://github.com/${GH_REPO}" \
    --token "${GH_RUNNER_TOKEN}" \
    --labels "self-hosted,linux,x64,edlio-presence" \
    --name "$(hostname)-edlio" \
    --unattended \
    --replace

# ─── 5. Install + start the systemd service ───────────────────────────────────
# svc.sh must be run as root; it reads the runner's .credentials to name the unit
info "Installing systemd service…"
pushd "${RUNNER_DIR}" > /dev/null
./svc.sh install "${RUNNER_USER}"
./svc.sh start
popd > /dev/null

info "Runner service started."

# ─── 6. Create artifact directory ─────────────────────────────────────────────
ARTIFACT_DIR="/var/lib/edlio-presence"
if [[ ! -d "${ARTIFACT_DIR}" ]]; then
    info "Creating ${ARTIFACT_DIR}…"
    mkdir -p "${ARTIFACT_DIR}"
fi
chmod 755 "${ARTIFACT_DIR}"
chown "${RUNNER_USER}:${RUNNER_USER}" "${ARTIFACT_DIR}"

# ─── Done ─────────────────────────────────────────────────────────────────────
echo ""
echo -e "${GRN}✓ GitHub Actions runner installed and started.${NC}"
echo ""
echo "  Runner name : $(hostname)-edlio"
echo "  Repository  : https://github.com/${GH_REPO}"
echo "  Labels      : self-hosted, linux, x64, edlio-presence"
echo "  Service     : $(systemctl list-units 'actions.runner.*' --no-legend | awk '{print $1}' | head -1)"
echo "  Artifact dir: ${ARTIFACT_DIR}"
echo ""
echo -e "${YLW}Next step:${NC} scp the dist-packages tarball from Mac mini to this VPS:"
echo ""
echo "  scp /Users/tenedos/edlio-presence-build/dist-packages.tar.gz \\"
echo "      root@\$(curl -s ifconfig.me):${ARTIFACT_DIR}/dist-packages.tar.gz"
echo ""
echo "Then trigger the workflow:"
echo "  gh workflow run build-snapshot.yml --repo ${GH_REPO}"
echo ""
