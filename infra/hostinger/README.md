# Self-Hosted GitHub Actions Runner on Hostinger

This guide walks you through setting up a **Hostinger KVM 2 VPS** as a self-hosted GitHub Actions runner for the `edlio-presence` repository. Once done, the `Build & publish snapshot image` workflow will run on this VPS instead of GitHub's shared runners — allowing it to access the 3.3 GB `dist-packages.tar.gz` tarball locally and push an 11 GB Docker image without hitting network timeouts.

---

## Recommended Hostinger Plan

| Spec | Value |
|------|-------|
| Plan | **KVM 2** |
| vCPU | 2 cores |
| RAM | 8 GB |
| NVMe | 100 GB |
| Price | ~$7/mo |
| OS | **Ubuntu 24.04 LTS** (select during provisioning) |

This plan comfortably fits Docker CE, the build toolchain, and the 3.3 GB tarball — with room to spare for the image layer cache.

---

## One-time Setup Overview

```
1. Provision VPS on Hostinger → SSH in
2. Set env vars + run install-runner.sh
3. scp dist-packages.tar.gz from Mac mini → VPS
4. Trigger the workflow (UI or gh CLI)
5. Confirm image at ghcr.io/kemalarsan/edlio-presence:day2-snapshot
```

---

## Step 1 — Provision the VPS

1. Log in to [hpanel.hostinger.com](https://hpanel.hostinger.com)
2. **Order → VPS → KVM 2**
3. Select OS: **Ubuntu 24.04 LTS**
4. Note the public IP address shown on the dashboard (called `$VPS_IP` below)
5. Set a root password or upload an SSH public key

SSH in to verify access:
```bash
ssh root@$VPS_IP
```

---

## Step 2 — Get a GitHub Runner Registration Token

1. Go to: https://github.com/kemalarsan/edlio-presence/settings/actions/runners/new
2. Select **Linux / x64**
3. Copy the token shown under "Configure" — it looks like `AABC...` (expires after ~1 hour)

> **Note:** This token is single-use and short-lived. If it expires, generate a new one from the same URL.

---

## Step 3 — Run the Installer

On the VPS, set the two required env vars and pipe the script in one shot:

```bash
export GH_REPO="kemalarsan/edlio-presence"
export GH_RUNNER_TOKEN="<paste token from step 2>"
bash <(curl -fsSL https://raw.githubusercontent.com/kemalarsan/edlio-presence/main/infra/hostinger/install-runner.sh)
```

Or if you've already cloned the repo:
```bash
export GH_REPO="kemalarsan/edlio-presence"
export GH_RUNNER_TOKEN="<paste token from step 2>"
bash infra/hostinger/install-runner.sh
```

The script will:
- Install **Docker CE** (official apt repo, not snap)
- Create a `github-runner` system user
- Download & configure the latest GitHub Actions runner
- Install it as a **systemd service** (`actions.runner.*`)
- Create `/var/lib/edlio-presence/` (where the tarball lives)

When it finishes you'll see:
```
✓ GitHub Actions runner installed and started.
Next: scp the dist-packages tarball to /var/lib/edlio-presence/dist-packages.tar.gz
```

---

## Step 4 — Copy the Tarball from Mac Mini (one-time, ~3.3 GB)

On your **Mac mini** (or wherever `dist-packages.tar.gz` lives):

```bash
scp /Users/tenedos/edlio-presence-build/dist-packages.tar.gz \
    root@$VPS_IP:/var/lib/edlio-presence/dist-packages.tar.gz
```

This is a one-time transfer (~15-30 min on a typical home connection). The file stays on the VPS permanently — it's the frozen Python environment from Day 2.

Verify it landed correctly on the VPS:
```bash
ssh root@$VPS_IP "ls -lh /var/lib/edlio-presence/"
# Should show: dist-packages.tar.gz  ~3.3G
```

---

## Step 5 — Trigger the Workflow

Via the GitHub Actions UI:
1. Go to: https://github.com/kemalarsan/edlio-presence/actions/workflows/build-snapshot.yml
2. Click **Run workflow → Run workflow**

Or via the `gh` CLI:
```bash
gh workflow run build-snapshot.yml --repo kemalarsan/edlio-presence
```

Watch the run — it should take 5–15 minutes (mostly the `docker buildx build` + push step).

---

## Step 6 — Verify the Image

```bash
# On any machine with Docker logged in to GHCR:
docker pull ghcr.io/kemalarsan/edlio-presence:day2-snapshot
```

Or check the Packages tab on GitHub:
https://github.com/kemalarsan/edlio-presence/pkgs/container/edlio-presence

---

## Runner Health

Check the service status at any time:
```bash
ssh root@$VPS_IP "systemctl status actions.runner.*.service"
```

View runner logs:
```bash
ssh root@$VPS_IP "journalctl -u 'actions.runner.*' -n 50 --no-pager"
```

The runner shows as **Idle** in GitHub when healthy:
https://github.com/kemalarsan/edlio-presence/settings/actions/runners

---

## Teardown

To remove the runner (e.g., to re-register or decommission the VPS):

```bash
export GH_RUNNER_TOKEN="<new removal token from Settings → Actions → Runners>"
bash infra/hostinger/uninstall-runner.sh
```

> A **different token** is required for removal — get it from the runner's ⋮ menu in GitHub Settings → Actions → Runners → click the runner → Remove runner.

---

## Security Notes

- This runner is registered to a **private repository** — it only picks up jobs from `kemalarsan/edlio-presence`
- **Never** enable `pull_request` triggers on the `build-snapshot.yml` workflow — fork PRs could exfiltrate the tarball
- Keep the VPS OS and runner binary updated (`apt upgrade` monthly; the installer fetches the latest runner binary automatically on re-run)
- The `github-runner` user has Docker access but no sudo — limit blast radius if the runner is ever compromised
- Rotate the VPS SSH key if you share access with others
