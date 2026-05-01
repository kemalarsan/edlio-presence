# Build Pipeline Decision — 2026-05-01

**Context:** Overnight (2026-04-30 → 2026-05-01), we spent ~8.5 hours and
~30 GB of Comcast upload trying to push an 11 GB Docker image from the
Mac mini to GHCR. Both `docker push` (4 retries) and `skopeo copy` (with
per-blob retries) failed identically with `use of closed network connection`
errors mid-upload. The Mac mini cannot reliably publish large images from
behind residential Comcast, and this will bite us on every rebuild.

**Decision needed:** How do we publish the GPU renderer image going forward?

## Constraints

- Renderer image is ~11 GB (base `runpod/pytorch:2.4.0` is huge) — can't shrink much
- The snapshot approach (`edlio-presence-build/Dockerfile`) works cleanly because
  `dist-packages.tar.gz` (3.3 GB) already contains the proven-working env
  from Day 2. Rebuilding from scratch (`infra/docker/Dockerfile`) hits legacy
  chumpy/mmcv installation problems.
- GitHub Actions free tier: 2 CPU, 7 GB RAM, 14 GB free disk, 6-hour timeout.
- The `runpod/pytorch` base image is ~8 GB — runner disk is tight but doable.

## Options

### Option A — Rebuild from scratch on GH Actions (use existing workflow)
`.github/workflows/docker-image.yml` already exists and builds from
`infra/docker/Dockerfile` (the full `pip install` path).

- ✅ Clean, repeatable, no artifact hosting
- ❌ Chumpy/mmcv install is finicky in CI (took 18+ min of iteration on the pod)
- ❌ Any pip pin rot could break CI without warning
- ❌ Slow feedback loop when it breaks (20-min build cycles)

### Option B — Use snapshot Dockerfile on GH Actions, host `dist-packages.tar.gz` externally ⭐
Commit `edlio-presence-build/Dockerfile` approach into `infra/docker/Dockerfile.snapshot`,
point a new workflow at it, and have the runner download `dist-packages.tar.gz`
from a CDN-backed store at build time.

Storage options for the 3.3 GB tarball:
- **Hugging Face Hub** (free, simple, gated repo if needed)
- **Cloudflare R2** (cheap, egress-free, $0.015/GB storage)
- **GitHub Releases** (2 GB per-asset limit — would need to split, ugly)
- **S3 / Spaces** (pay for egress, but tiny scale)

Recommended: **Hugging Face Hub private repo** — we already use HF for MuseTalk
weights (`HF_HOME` is set up), so auth is wired.

- ✅ Proven working environment baked in, no pip install roulette
- ✅ GH runner pulls 3.3 GB (~1 min) vs Mac mini pushing 11 GB (~hours)
- ✅ Rebuild is fast (~10 min instead of 20+)
- ❌ Requires one-time upload of `dist-packages.tar.gz` to HF (3.3 GB from my Comcast — still painful, but one time)
- ❌ Tarball goes stale if we change Python deps — refresh workflow needed

### Option C — Build on RunPod itself (docker-in-docker on a GPU pod)
Spin up a RunPod with docker pre-installed, build there, push from the
data center on gigabit.

- ✅ No upload bottleneck
- ✅ Can actually test GPU at build time
- ❌ Requires docker-in-docker setup on a pod (non-trivial)
- ❌ Costs GPU time ($0.16-$0.40/hr) for CPU-only build work (waste)
- ❌ Not really "infrastructure" — a manual ritual

### Option D — Defer entirely
Keep running the RunPod pod we already have ($4/day). Iterate on
tenedos-voice wiring against the live `/render` endpoint. Ship a Docker
image later when network conditions change.

- ✅ Unblocks immediate product work
- ✅ Zero infra effort
- ❌ Single-point-of-failure (pod dies → no renderer)
- ❌ Burns ~$30/week indefinitely

## Recommendation

**Option B.** The snapshot approach proved itself on Day 2 (25/25 tests,
real-time render). We keep what works and just move the publish step off
the Mac mini.

**Concrete plan:**

1. **One-time upload**: `dist-packages.tar.gz` (3.3 GB) → Hugging Face
   private repo `edlio/presence-layer-build-artifacts`. Yes this is a
   painful upload, but it's once.
2. **Add `infra/docker/Dockerfile.snapshot`** (commit the Day-2 build
   recipe into the repo).
3. **New workflow `.github/workflows/build-snapshot.yml`** — triggers on
   push to main or on `workflow_dispatch`, pulls the tarball from HF,
   builds, pushes to GHCR.
4. **Keep existing workflow** for the from-scratch path, but mark it as
   "slow fallback." Don't trigger on every push.

**Fallback plan (Option D):** While the HF upload is in progress, the
running pod stays live. Tenedos-voice iteration continues against
`https://<pod-url>/render`. We're not blocked.

## Next steps (ordered)

1. ☐ Get Ali's sign-off on Option B
2. ☐ Create HF private repo `edlio/presence-layer-build-artifacts`
3. ☐ Start the 3.3 GB upload from Mac mini (background, will take hours,
   but HF supports resume via `huggingface_hub`)
4. ☐ Commit `Dockerfile.snapshot` and `build-snapshot.yml` to `edlio-presence`
5. ☐ Trigger workflow, watch GH Actions build, celebrate first GHCR push
6. ☐ Update `docs/OVERNIGHT_2026-04-30.md` and close the loop

## Timing

- HF upload (my side): 3-6 hours (same Comcast, 3.3 GB not 11 GB)
- GH Actions build + push: ~10-15 min once tarball is hosted
- Total wall-clock to working CI: half a day, mostly passive upload time
