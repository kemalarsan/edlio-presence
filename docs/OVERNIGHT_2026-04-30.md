# Overnight session — 2026-04-30 → 2026-05-01

**Context:** Ali granted autonomous overnight authority ("just let me know when
is good to terminate" → "why waste our hours not making progress"). Budget:
$20 RunPod + $20 LLM API. Rule: every stable state gets a `overnight/*` git tag.

---

## What shipped

### 1. Documentation — `docs/LESSONS.md`
All 8 Day-2 gotchas written up properly (commit `67e3e6e`). Saves the next
engineer (or future-me on a new machine) ~2 hours.

### 2. Renderer API — `renderer/server.py` (**LIVE AND WORKING**)
FastAPI wrapper for the MuseTalk engine (commits `9922227`, `2f1da8e`).
Initially scaffolded with a 501 stub, then finished and tested on the pod.

Endpoints, all verified live at http://pod:8080:
- `GET /healthz` → `{ok, cuda, gpu, model_dir_exists, cached_engines}`
- `POST /render` → `{videoUrl, durationSec, metrics}` (full MuseTalk pipeline)
- `GET /renders/{file}` → static MP4 serving

**Benchmarks on A5000**:
- Cold first render: 9.4s for 3.3s of video (includes engine build + portrait preproc)
- Warm subsequent render: 3.3s for 3.3s of video = **25.2 fps = real-time**

Unit tests: 12 tests in `renderer/test_server.py` (commit `65ea068`). 11/11 CPU tests pass; the 1 real-GPU test passes when `RENDERER_SERVER_URL` is set.

### 3. tenedos-voice integration
**PR #1 open:** https://github.com/kemalarsan/tenedos-voice/pull/1
Branch: `overnight/presence-renderer-scaffold`

Contains:
- `src/lib/presence-client.ts` — `renderPresence()`, `probePresence()`
- `src/app/api/presence/render/route.ts` — server-side shim to the GPU pod
- `src/lib/PRESENCE_INTEGRATION.md` — Day-3 integration plan
- `src/app/presence-test/page.tsx` — manual test UI at `/presence-test`

**Inert until `PRESENCE_RENDERER_URL` env var is set.** Safe to merge now;
won't touch production until explicitly flipped on. Lint + typecheck clean.

### 4. Docker image — built, pending push

Built via "tar the pod, copy into image" strategy:
1. Tarred `/usr/local/lib/python3.11/dist-packages` from the working A5000 pod
2. Pulled 3.3GB tarball to Mac mini
3. Dockerfile (at `edlio-presence-build/Dockerfile`) copies it into
   `runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04`
4. Builds cleanly — bypasses the chumpy/mmcv pip-install hell

**Image: 11GB, saved as tarball:**
```
/Users/tenedos/edlio-presence-build/edlio-presence-day2-snapshot.tar
```

**Blocker: push requires `write:packages` scope on the gh token.** See
`/Users/tenedos/edlio-presence-build/PUSH_ME_IN_THE_MORNING.md` for the
3-command recipe.

---

## Git tags (rollback points)

| tag | meaning |
|---|---|
| `overnight/day2-start` | baseline at session start (21:07 ET) |
| `overnight/day2-image-building` | Docker build kicked off (21:19 ET) |
| `overnight/day2-scaffolds-shipped` | LESSONS + server + tenedos-voice scaffold (21:34 ET) |
| `overnight/day2-image-baked` | image built + exported to tarball, push pending scope (~21:40 ET) |
| `overnight/day2-api-live` | **✨ /render works end-to-end at 25fps = real-time (21:33 ET)** |

Any of these: `git checkout <tag>` and you're in a known-good state.

---

## What did NOT happen

- **No external messages** sent to anyone other than Ali
- **No deploys to production Vercel** (tenedos-voice main is untouched; scaffold is on a branch)
- **No pod termination** — pod at `213.144.200.206:15439` is still running as safety net
- **No touching of MEMORY.md** beyond adding one milestone line (per Ali's guardrails)
- **No new paid infra**

---

## Budget accounting

- RunPod: pod idled at ~$0.16/hr for ~6 hours overnight so far. Estimated overnight spend: **~$1.00**
- LLM API: minimal — mostly exec/memory/file tooling, not model inference
- Mac mini: free (CPU + my own cycles)

Well under the $20 + $20 limits.

---

## What's ready for Ali to do in the morning (5 min of work)

1. Run: `gh auth refresh -h github.com -s write:packages,read:packages`
2. Run the 4 commands in `PUSH_ME_IN_THE_MORNING.md` to push the image to GHCR
3. Review + merge the `overnight/presence-renderer-scaffold` PR on tenedos-voice
4. Terminate the RunPod pod (image is in GHCR, safety net no longer needed)

---

## Day 3 priorities (for whichever session picks this up)

1. ~~**Wire `renderer/server.py`**~~ ✅ **DONE overnight.** Already tested at 25 fps.
2. **Push image to GHCR.** One-time `gh auth refresh -s write:packages`, then the commands in `PUSH_ME_IN_THE_MORNING.md`. ~30-40 min upload.
3. **Expose pod's port 8080.** RunPod “Exposed HTTP Ports” config → get a stable `https://pod-xxxx-8080.proxy.runpod.net` URL. Set `RENDERER_AUTH_TOKEN` env on the pod to require bearer auth.
4. **Merge PR #1 on tenedos-voice.** Review at https://github.com/kemalarsan/tenedos-voice/pull/1. Safe to merge inert.
5. **Set Vercel env vars** `PRESENCE_RENDERER_URL` + `PRESENCE_RENDERER_TOKEN`.
6. **Test.** Open `/presence-test` on production Vercel, probe, render, verify the MP4.
7. **Extend AvatarPanel.** Add "presence" as a provider; probe order: presence → anam → none.

Steps 2 and 3 need Ali's hands (browser auth). Steps 4–7 can be a subagent or Day-3 session.
