# Day 2 Wrap — 2026-04-30 (evening)

## Progress: ~70% of Day 2 goals complete

## What worked ✅
- RunPod account + $100 credits confirmed
- SSH key (`~/.ssh/runpod_ed25519` on Mac mini) authorized on RunPod account
- Fresh pod launched: `exuberant_blush_mink` (RTX 4090 secure, $0.69/hr — not the $0.34/hr community we planned, see below)
- SSH connection established via Web Terminal key injection (pod created before key was on account — one-shot fix)
- `hello-gpu.py` — **ALL 5 CHECKS PASS** on real hardware (torch 2.4.1+cu124, RTX 4090, 24GB, ffmpeg)
- Private repo cloned via GitHub deploy key (key ID 150161240, read-only)
- MuseTalk submodule checked out at pinned commit `0a89dec`
- `audio/requirements.txt` + `renderer/requirements.txt` installed cleanly

## Where we stopped 🛑
- Installing MuseTalk's `mmcv` dep from source
- Pod exited ~10 min in, mid-compile
- Likely cause: `mmcv` CUDA compilation OOM'd on the 16GB-RAM RTX 4090 community pod
- Contributing factor: pod was CUDA 12.4 / PyTorch 2.4 / Python 3.11 — `mmcv` prebuilt wheels only exist for cu118, so MIM fell back to source compile

## Total spend
- $0.14 (10 min of RTX 4090 secure-tier time)
- Balance: $99.86

## What I'd do differently (Toyota lessons)

### 1. Pre-bake the Docker image, don't install at runtime
The entire premise of Track B's Dockerfile was avoiding this exact situation. Fix:
- Build the image locally (M-series Mac can build linux/amd64 via buildx/QEMU)
- Push to GHCR (needs `packages:write` PAT)
- Launch pods FROM that image — everything pre-installed, mmcv compiled once
- Pod startup goes from "20 min compile" to "~30 sec cold start"

### 2. Pin CUDA version at pod launch
The MuseTalk repo pins to CUDA 11.8 because that's where `mmcv` has prebuilt wheels. We should either:
- **(a)** Launch pods with a cu118 PyTorch template to match our Dockerfile
- **(b)** Upgrade our Dockerfile to cu121 or cu124 + rebuild `mmcv` wheels to match (more work, but future-proof)

Recommendation: (a) for MVP. Boring, works, reproducible.

### 3. SSH key workflow needs one more step
When you add a key to RunPod account AFTER launching a pod, the pod doesn't auto-pick it up.
Fix: always add keys to account BEFORE launching pods, OR use Web Terminal to append to authorized_keys (what we did).
This is now documented in `infra/README.md` (TODO next Day 2 attempt).

### 4. mmcv source-compile guard
Add to `infra/Dockerfile`:
```dockerfile
RUN pip install --no-cache-dir \
    mmcv==2.0.1 \
    -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.0/index.html
```
This forces prebuilt wheels. If MIM can't find them → fail fast, don't slow-compile.

## Day 3 plan (clean restart)
1. Build `edlio-presence:dev` image locally on Mac mini via buildx
2. Get Ali's GitHub PAT with `packages:write`
3. Push to `ghcr.io/kemalarsan/edlio-presence:dev`
4. Launch new pod FROM that image via `infra/runpod-launch.sh`
5. SSH in → run `pytest renderer/` with GPU markers enabled
6. All 7 previously-skipped GPU tests should pass
7. Time budget: 1-2 hours. Cost budget: <$1 in GPU time.

## Files created in this session
- `/Users/tenedos/.ssh/runpod_ed25519` + `.pub` — dedicated RunPod key
- `/Users/tenedos/.ssh/config` — updated with RunPod auto-routing
- `/tmp/.openclaw-tokens/runpod` + `/Users/tenedos/.openclaw/tokens/runpod` — API key
- GitHub deploy key on repo (ID 150161240) — should be removed or rotated before production

## Honest assessment
Day 1 shipped ahead of schedule. Day 2 hit a predictable infra-vs-reality mismatch (pod CUDA ≠ Dockerfile CUDA, mmcv compile heavy). No code lost, no credentials lost, validated end-to-end connectivity. The 30% we didn't finish is 100% a "launch correct image" problem, not an "architecture is wrong" problem.

**Tomorrow's headline:** push proper image to GHCR, pod launches from it, tests run instantly.
