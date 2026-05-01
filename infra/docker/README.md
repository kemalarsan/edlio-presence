# edlio-presence Docker image

This directory defines the GPU renderer image — the reproducible environment
that got `25/25 tests passing` and the first end-to-end render on Day 2
(2026-04-30).

## Why this exists

Day 2 involved an 18-minute dependency dance on a fresh pod:
- `huggingface-cli` was deprecated → use `hf` CLI
- `diffusers 0.27` breaks with newer `huggingface_hub` → pin to `0.30.2`
- `transformers 4.55` requires `torch 2.6` to load `.bin` files → pin to `4.44.2`
- MuseTalk's internal code uses **relative paths** (`models/sd-vae`) → symlink `models → renderer/muse_talk_vendor/models`
- `hf download --include` does **not** grab `config.json` files → fetch them explicitly

This image bakes all of that in, so future pods are ready in ~30 seconds instead
of 18 minutes.

## Files

| file | what |
|---|---|
| `Dockerfile` | torch 2.4 + CUDA 12.4 base + pinned deps |
| `requirements-pod.txt` | `pip freeze` from the working Day-2 pod (211 packages) |
| `entrypoint.sh` | runs `fetch_weights.sh` on first boot, then execs the CMD |
| `fetch_weights.sh` | downloads all 8.6GB of weights + their missing config.json files |

## Build locally (current path — no GitHub Actions yet)

```bash
cd /Users/tenedos/edlio-presence
docker buildx build \
    --platform linux/amd64 \
    -f infra/docker/Dockerfile \
    -t ghcr.io/kemalarsan/edlio-presence:day2 \
    -t ghcr.io/kemalarsan/edlio-presence:latest \
    --push \
    .
```

Note: `--platform linux/amd64` is required when building from Apple Silicon
(Mac mini is arm64, RunPod is x86_64).

Prerequisite: `gh auth login` + `echo $GH_TOKEN | docker login ghcr.io -u kemalarsan --password-stdin`

## Build via GitHub Actions (future, when workflow scope is granted)

Run:
```bash
gh auth refresh -h github.com -s workflow
```

Then copy the workflow YAML from this file's git history (was dropped in
commit `07f1eb1` because the token lacked scope), re-add, commit, push.
GitHub Actions auto-builds + pushes on every commit to `main`.

## Launch a RunPod from the image

1. RunPod dashboard → "Pods" → "Deploy"
2. Choose a GPU (RTX A5000 proven in Day 2 — ~$0.16/hr)
3. Under "Custom Image" paste: `ghcr.io/kemalarsan/edlio-presence:day2`
4. Container Start Command: leave default or `bash`
5. Deploy

First boot: ~2 min (image pull ~30s + weight fetch ~90s on datacenter network).
Subsequent boots on same pod: ~30s (weights cached).

## Verify the image works

```bash
# Inside the pod
cd /workspace/edlio-presence
RENDERER_TEST_GPU=1 RENDERER_MODEL_DIR=/workspace/edlio-presence/models \
    python3 -m pytest renderer/test_inference.py -v

# Should see: 25 passed
```

## Day-2 lessons encoded here

1. Never use `huggingface-cli` — it's deprecated. Use `hf download` (from huggingface_hub[cli] ≥0.26).
2. `hf download --include "dir/*"` does NOT always fetch `config.json`; fetch it explicitly via `curl`.
3. diffusers 0.30.2 is the sweet spot for torch 2.4 (newer diffusers need newer hf_hub; older need `cached_download`).
4. transformers 4.44.2 bypasses the torch.load safety check that 4.55+ requires torch≥2.6 for.
5. MuseTalk has hardcoded relative paths. Always symlink `models → renderer/muse_talk_vendor/models` or MuseTalk can't find its own weights.
6. The positional encoding layer (`engine.pe`) must be cast to fp16 along with the UNet and VAE, or inference fails with a dtype mismatch.
