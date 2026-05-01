# Day-2 Lessons — edlio-presence GPU renderer

Captured during the 18-minute dependency dance on a fresh RunPod A5000 that
eventually got 25/25 tests passing and rendered the first MuseTalk video.

Cost of re-learning these: about 2 hours. Cost of re-reading this doc: 5 minutes.

---

## 1. HuggingFace CLI changed names

| before | after |
|---|---|
| `huggingface-cli` | `hf` |

`huggingface-cli` is deprecated in `huggingface_hub >= 0.25`. Scripts and docs
that call it still "work" with a warning, but the binary path may differ.
Use `hf` in all new code.

```bash
# Old (deprecated)
huggingface-cli download TMElyralab/MuseTalk --include "musetalkV15/*"

# New
hf download TMElyralab/MuseTalk --include "musetalkV15/*" --local-dir .
```

## 2. `hf download --include` does NOT grab `config.json` files

Surprise: `hf download ... --include "musetalkV15/*"` fetches `.pth` but
silently skips `config.json`. Config files sit at the parent level on HF
and the glob doesn't match. Symptom: runtime errors like
`OSError: [path] does not appear to have a file named config.json`.

Fix: fetch configs explicitly via `curl`:

```bash
curl -sL https://huggingface.co/TMElyralab/MuseTalk/resolve/main/musetalkV15/musetalk.json \
     -o musetalkV15/musetalk.json
```

See `infra/docker/fetch_weights.sh` for the complete list of configs that
`hf download` misses.

## 3. The diffusers / transformers / torch version triangle

These three versions must be chosen together, not independently.

| package | pin | why |
|---|---|---|
| `torch` | `2.4.1+cu124` | What RunPod's base image ships |
| `diffusers` | `0.30.2` | Newer (≥0.31) needs `huggingface_hub >= 1.0`; older (≤0.27) needs `cached_download` which doesn't exist in new hf_hub |
| `transformers` | `4.44.2` | 4.55+ uses `torch.load(weights_only=True)` by default which requires torch ≥ 2.6 for `.bin` files |
| `huggingface_hub` | `0.26.x` | Has both the new `hf` CLI AND `cached_download` (needed by diffusers 0.30) |
| `mmcv` | `2.1.0` | Must compile against torch 2.4; any newer/older combo fails |
| `mmpose` | `1.3.2` | Needs mmcv 2.1 |

If you bump ONE of these, you likely need to bump all of them together.

## 4. MuseTalk hardcodes relative paths

MuseTalk's internal code does things like `load('models/sd-vae/...')` from
deep inside `renderer/muse_talk_vendor/musetalk/whisper/...`. This only
works if `models/` exists at the project root AND points to the weights.

Always symlink:

```bash
ln -sf renderer/muse_talk_vendor/models /workspace/edlio-presence/models
```

## 5. Positional encoding must also be fp16

When running inference in half precision, you must cast the UNet, VAE,
**and** the positional encoding layer `engine.pe` to `torch.float16`.
Forgetting `engine.pe` produces a confusing dtype mismatch error during
attention computation, not during model load.

See `renderer/engine.py` — the fix is one line but non-obvious.

## 6. Legacy setup.py packages break modern pip isolated builds

Packages with setup.py files older than ~2020 often fail in modern pip's
isolated build environment because they expect `pip` and `pkg_resources`
to be importable. Known offenders in our stack: `chumpy` (2014), `mmcv`.

Workarounds, in order of preference:
1. Install with `pip install --no-build-isolation <pkg>` — uses the outer
   env's pip/setuptools instead of a sandboxed one
2. Use a pre-built wheel from the project's distribution URL
   (e.g. OpenMMLab publishes `mmcv` wheels at
   `https://download.openmmlab.com/mmcv/dist/cu${VER}/torch${VER}/`)
3. Install the package into an already-set-up env (don't rebuild)

The Docker image in this repo uses strategy (1) for chumpy+mmcv and
strategy (3) for everything else (env tarball from a proven pod).

## 7. Docker build strategy: snapshot the pod, don't rebuild from scratch

Trying to re-run the 18-minute pip install sequence in GitHub Actions CI
is fragile — any transitive legacy package (see #6) breaks it. Instead:

1. Run the pod once, get the env working manually, test it
2. `tar -czf /tmp/dist-packages.tar.gz -C /usr/local/lib/python3.11 dist-packages`
3. SCP the tarball to the build host
4. Dockerfile COPYs it in and untars it into the matching base image path

Result: a reproducible image whose Python env is byte-for-byte identical to
the proven-working pod. No risk of dep-resolution regression.

Downside: tarball is ~3.3GB. Acceptable for one-time push to GHCR.

## 8. RunPod specifics

- Base image on pods: `runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04`
- Ubuntu 22.04.5, Python 3.11.10, CUDA 12.4.1
- No docker-in-docker inside pods — can't build images from the pod
- SSH-in is fine; pod I/O is fast; home-network download of pod tarballs runs 30-80MB/s
- A5000 costs about $0.16/hr on-demand; full MuseTalk V1 weights + envs fit in 20GB

---

## When in doubt

Check `git log` on this repo for the Day-2 commits (`d23b9a0`, `07f1eb1`,
`a3dcd50`, `3882b90`, `06a0e1f`) — the commit messages document the
exact failure modes we hit and the fixes that worked.
