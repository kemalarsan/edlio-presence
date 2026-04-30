# Track B: Infrastructure

This track owns the GPU container definition and RunPod launch machinery for the edlio-presence MuseTalk renderer.

**Tracks A (renderer) and C (audio) depend on this track confirming a working GPU environment before starting their work.**

---

## What lives here

| File | Purpose |
|------|---------|
| `Dockerfile` | Reproducible GPU image (CUDA 11.8 + PyTorch 2.0.1 + MuseTalk deps) |
| `hello-gpu.py` | In-container smoke test — exits 0 if GPU is ready |
| `docker-compose.dev.yml` | Local dev (requires host CUDA GPU) |
| `build-push.sh` | Build image + push to GHCR |
| `runpod-launch.sh` | Launch a RunPod persistent pod via GraphQL API |

---

## Prerequisites

### To build locally
- Docker Desktop (or Docker Engine) with BuildKit
- No GPU required to _build_; GPU required to _run_

### To run locally
- NVIDIA GPU with CUDA 11.8+ driver (driver ≥ 520)
- NVIDIA Container Toolkit installed (`nvidia-ctk`)

### To push to GHCR
- GitHub PAT with `packages:write` scope
- `docker login ghcr.io -u <github-username> -p <PAT>`

### To launch on RunPod
1. Create a RunPod account at https://www.runpod.io
2. Go to Settings → API Keys → create a key
3. Export: `export RUNPOD_API_KEY=<your-key>`

> **⚠️ Ali action required:** No RunPod credentials were found in 1Password or token cache.
> Create an account at runpod.io, load it with credits (~$20 to start), and save the API key to:
> - 1Password vault "Tenedos" as "RunPod - Tenedos" → field: `credential`
> - OR cache it: `echo "<key>" > /tmp/.openclaw-tokens/runpod`

---

## MuseTalk requirements (verified April 2026)

Source: https://github.com/TMElyralab/MuseTalk

| Component | Version |
|-----------|---------|
| Python | 3.10 |
| CUDA | 11.8 (cu118) |
| PyTorch | 2.0.1 |
| torchvision | 0.15.2 |
| torchaudio | 2.0.2 |
| diffusers | 0.30.2 |
| transformers | 4.39.2 |
| mmcv | 2.0.1 |
| mmdet | 3.1.0 |
| mmpose | 1.1.0 |
| ffmpeg | system package (apt) |

The Dockerfile pins all of these exactly. Do not upgrade without testing — MuseTalk's mmpose/DWPose integration is sensitive to mmcv ABI.

---

## GPU target

| GPU | VRAM | Cost/hr | Notes |
|-----|------|---------|-------|
| L40S | 48 GB | ~$1.00 | **Recommended for MVP dev** — plenty of headroom |
| RTX 4090 | 24 GB | ~$0.45 | Good alt — MuseTalk inference fits easily in 24 GB |
| Tesla V100 | 16 GB | ~$0.30 | Minimum — MuseTalk 1.5 benchmarked at 30fps+ here |

**Dev strategy:** Persistent pod during active development (iterate fast). Switch to serverless in production (cold-start ~15–60s is acceptable when idle).

---

## How to build

Run from the repo root:

```bash
# Build only (no push)
bash infra/build-push.sh

# Build + push to GHCR
bash infra/build-push.sh --push

# Build with specific tag
bash infra/build-push.sh --tag v0.1.0 --push

# Force clean build
bash infra/build-push.sh --no-cache --push
```

The image is tagged as:
- `ghcr.io/kemalarsan/edlio-presence:<tag>`
- `ghcr.io/kemalarsan/edlio-presence:dev` (always points to latest build)

---

## How to push to GHCR

```bash
# One-time login
docker login ghcr.io -u kemalarsan -p <GitHub PAT with packages:write>

# Build + push
bash infra/build-push.sh --push
```

---

## How to launch on RunPod

```bash
# Set credentials + image
export RUNPOD_API_KEY=<your-runpod-api-key>
export IMAGE=ghcr.io/kemalarsan/edlio-presence:latest
export GPU_TYPE="NVIDIA L40S"    # or "NVIDIA GeForce RTX 4090"

# Launch persistent pod
bash infra/runpod-launch.sh
```

The script prints the pod ID and cost/hr. The pod starts in ~2–3 minutes.

**Alternatively** — use RunPod's UI:
1. https://www.runpod.io/console/pods → Deploy
2. Select GPU: L40S (or RTX 4090)
3. Container image: `ghcr.io/kemalarsan/edlio-presence:latest`
4. Container disk: 50 GB, Volume: 100 GB mounted at `/models`
5. Click Deploy

---

## How to verify the GPU environment

Once the pod is running, SSH in and run:

```bash
python /workspace/hello-gpu.py
```

Expected output:
```
✅  PyTorch import: torch 2.0.1
✅  CUDA available: CUDA 11.8
✅  GPU device: NVIDIA L40S (47.5 GB VRAM)
✅  GPU tensor op: matmul ok, sum=65536
✅  ffmpeg: ffmpeg version 4.4.x ...

──────────────────────────────────────────────────
✅  GPU READY — all checks passed
──────────────────────────────────────────────────
```

Exit code 0 = ready to hand off to Track A.

---

## How to stop the pod (avoid charges)

```bash
# Via RunPod dashboard → Pods → Stop
# Or via API:
curl -s -X POST "https://api.runpod.io/graphql?api_key=${RUNPOD_API_KEY}" \
  -H "Content-Type: application/json" \
  -d '{"query": "mutation { podStop(input: { podId: \"<POD_ID>\" }) { id desiredStatus } }"}'
```

**⚠️ Always stop the pod when not actively developing. L40S = ~$24/day when idle.**

---

## Local dev (if you have a CUDA GPU on your machine)

```bash
# Start dev container (runs hello-gpu.py and exits)
docker compose -f infra/docker-compose.dev.yml up --build

# Drop into bash for interactive work
docker compose -f infra/docker-compose.dev.yml run --rm renderer bash
```

Create a `.env` file in the repo root for local overrides:
```
HF_TOKEN=hf_xxxx
HOST_MODEL_DIR=/path/to/your/models
```

---

## Troubleshooting

### `CUDA error: no kernel image is available for execution on the device`
Driver/CUDA version mismatch. The container uses CUDA 11.8. Your host driver must be ≥ 520. Check: `nvidia-smi`.

### `torch.cuda.is_available() == False`
1. Verify NVIDIA Container Toolkit is installed: `nvidia-ctk --version`
2. Verify the container runtime is set: `docker run --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi`

### `mmcv` fails to build/install
The MMLab packages compile native extensions. This needs the `-devel` base image (included). If you switched to `-runtime`, switch back.

### Out-of-memory on 16 GB GPU
MuseTalk 1.5 in fp32 needs ~12 GB. Use `--use_float16` flag in inference. V100 (16 GB) is tight. RTX 4090 or L40S recommended.

### Model weights not found
MuseTalk weights (~3–4 GB) must be downloaded separately. They are NOT bundled in the Docker image (too large). Mount them via the `/models` volume:
```bash
bash models/download_weights.sh  # once on the host, then mount /models
```
See MuseTalk docs: https://github.com/TMElyralab/MuseTalk#download-weights

---

## Day 2 recommended launch sequence

```bash
# 1. Build and push image (do once, re-push after Dockerfile changes)
bash infra/build-push.sh --push

# 2. Launch RunPod pod
export RUNPOD_API_KEY=<key>
export IMAGE=ghcr.io/kemalarsan/edlio-presence:latest
bash infra/runpod-launch.sh

# 3. Wait ~3 minutes for pod to start, then SSH in and verify
python /workspace/hello-gpu.py

# 4. Hand off to Track A: renderer is ready
```

---

## Registry

Default registry: `ghcr.io/kemalarsan/edlio-presence`

This uses GitHub Container Registry (GHCR), which is free for public repos and $0 for storage up to the GitHub Free tier. RunPod can pull public GHCR images directly without auth.

To make the package public after first push:
- https://github.com/kemalarsan → Packages → edlio-presence → Package settings → Change visibility → Public
