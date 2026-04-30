# Day 2 — 2026-04-30 (evening) — GPU Validation

## Summary

Two pod sessions, one successful. Validated infrastructure end-to-end on real hardware. Identified the one remaining gap before real inference works.

## What we proved ✅

1. **RunPod account + credits operational** — $100 balance, confirmed via API
2. **Direct-TCP SSH works** — RunPod proxy requires TTY (incompatible with non-interactive automation), but direct-port SSH works perfectly
3. **Dedicated SSH key pair** (`~/.ssh/runpod_ed25519` + config in `~/.ssh/config`) on Mac mini
4. **Deploy-key flow for private repo** works: `gh api` adds key → pod SSHes with dedicated deploy key → repo clones cleanly with submodule at pinned commit
5. **`pod-setup.sh` validated:**
   - ffmpeg (apt)
   - audio/requirements.txt (faster-whisper, librosa, soundfile)
   - renderer/requirements.txt (opencv, transformers, einops)
   - diffusers + transformers (MuseTalk deps)
   - **mmcv from source** (when cu124/torch2.4 prebuilt wheel doesn't exist upstream)
   - mmdet + mmpose
   - mediapipe
   - **Total install time: 18m48s** (most of it mmcv compile)
6. **Track A's test suite runs on GPU:**
   - 20/25 stub tests PASS
   - 1/5 GPU tests PASS (`test_cuda_available`) — CUDA fully functional
   - 4/5 GPU tests FAIL — **known reason:** MuseTalk pretrained weights not downloaded

## The mmcv compile reality

MuseTalk's upstream requirements pin torch 2.0.1 / cu118 **specifically because prebuilt mmcv wheels exist there**. Using a newer pod template (torch 2.4 / cu124) forces source compile: ~18 min.

**Fix for Day 3+:** either
- (a) Launch pods with torch2.0/cu118 template — mmcv install drops to 30 sec via wheel
- (b) Bake our Docker image with mmcv compiled once — push to GHCR — subsequent pod launches are instant

## The MuseTalk weights gap

MuseTalk needs pretrained weights (stable-diffusion VAE, MuseTalk UNet, DWPose detector). These are NOT in the repo — they're downloaded via `download_weights.sh` from HuggingFace + Google Drive. ~2-3 GB total.

Track A's scaffold correctly wires up the loader but never downloads weights (that's a deployment step, not a code step).

**For Day 3:** add weight download to `pod-setup.sh` OR (better) bake into Docker image so pod starts ready to infer.

## Cost
- Pod 1 (Secure 4090, stopped mid-install): $0.14
- Pod 2 (Community A5000, successful): $0.15
- **Total today: ~$0.30**
- Balance: $99.70 remaining

## Files added today
- `/Users/tenedos/.ssh/runpod_ed25519` (+.pub) — dedicated key
- `/Users/tenedos/.ssh/config` — updated with RunPod auto-routing
- `/tmp/.openclaw-tokens/runpod` + `/Users/tenedos/.openclaw/tokens/runpod`
- `infra/pod-setup.sh` — validated end-to-end, handles apt-lock retry, prebuilt-wheel fallback
- GitHub deploy keys on repo (IDs 150161240 + 150162514) — rotate/remove before production

## Day 3 plan (clean)
1. **Build Dockerfile on Mac mini** using torch2.0.1 + cu118 base
2. **Include weight download step** in Dockerfile so image ships ready-to-infer
3. **Push to `ghcr.io/kemalarsan/edlio-presence:dev`** (needs PAT with packages:write)
4. **Launch pod FROM that image** via infra/runpod-launch.sh
5. **Run GPU tests** — should take <1 min total (no install, no weight download, no compile)
6. **Test end-to-end:** audio.wav → AudioFeatures → MuseTalkEngine.render_frame() → MP4 of Tenedos talking
7. Update Liv portrait if quality needs bumping (current is 720x480 — may want 1024+)

## Honest assessment
Day 2 delivered: real GPU validation, real deps working, real model framework initialized. The ONLY piece missing for actual "Tenedos talks" is model weights — a deployment concern, not a code concern. That's a completely different class of problem than "architecture is wrong."

Tomorrow: bake it all into an image → launches become instant → we can iterate on quality.
