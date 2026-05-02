# Portrait assets

Source portrait clips and derived thumbnails for the Tenedos avatar.

## Current canonical identity — tenedos-v2-no-glasses

Captured 2026-05-02 on iPhone 15 Pro Max, back camera, 4K @ 30fps HEVC 10-bit.
Ali Arsan, no glasses, neutral expression, natural blinks, 25s duration.

### Files

| File | Location | Notes |
|---|---|---|
| `tenedos-v2-no-glasses.mov` | local only (gitignored) | 78 MB, raw HEVC 4K |
| `tenedos-v2-no-glasses-1080p.mp4` | Supabase `presence-media/portraits/` | 12 MB, H.264, used by renderer |
| `IMG_2666-frame.jpg` | in repo | mid-frame thumbnail |

Public URL: https://uqwscuobzsuequtcdnwk.supabase.co/storage/v1/object/public/presence-media/portraits/tenedos-v2-no-glasses-1080p.mp4

### Face bbox

Detected with OpenCV Haar cascade (`haarcascade_frontalface_default.xml`,
`minSize=150`, `scaleFactor=1.1`, `minNeighbors=5`) on frame at t=5s.

**Bbox in 1080p pixel coords (what the renderer uses):** `[711, 180, 1249, 718]`

Vision-model spot check: 5/5 (tight, face fully contained, no crop).

## Alternate identity — tenedos-v2-glasses

Same shoot, Ali *with* glasses, 34s duration. Kept as an alternate identity
for contexts where "real Ali with glasses" matches user expectations (e.g.,
school-facing materials where the human counterpart wears glasses on calls).

Not the default — see MEMORY.md.

Public URL: https://uqwscuobzsuequtcdnwk.supabase.co/storage/v1/object/public/presence-media/portraits/tenedos-v2-glasses-1080p.mp4

## Legacy — tenedos-v1 (retired 2026-05-02)

Still portrait used for the Day-1 → Day-2 work. Single PNG frame,
`https://raw.githubusercontent.com/kemalarsan/edlio-presence/main/assets/test/tenedos-v1.png`.
Bbox: `[328, 169, 709, 673]`.
