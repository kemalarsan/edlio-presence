# renderer — Track A: MuseTalk Face Renderer

**Owner:** Track A  
**Status:** Day 1 scaffold — stub mode working, GPU mode ready to activate  
**Depends on:** Track B (GPU container — `infra/`) must be live before GPU mode is testable  
**Interface to:** Track C (audio/phoneme extractor), Track D (browser decode), Track G (SDK)

---

## What this track owns

Everything between "audio features come in" and "face frames go out":

```
AudioFeatures (Whisper embeddings)
        │
        ▼
  MuseTalkEngine          ← renderer/engine.py   (public API)
  MuseTalkRenderer        ← engine.py             (GPU path, MuseTalk V1.5)
  StubRenderer            ← engine.py             (CPU fallback, portrait passthrough)
        │
        ▼
RendererFrame (numpy RGB 256×256)
```

The engine is a single clean class (`MuseTalkEngine`) that Track G (SDK) imports. Everything else is internal.

---

## Model choice: MuseTalk V1.5

See [`refs/presence-layer/technical-brief.md`](../../../workspace/refs/presence-layer/technical-brief.md) for full rationale.

**Why MuseTalk, not LivePortrait or Wav2Lip:**

| Criterion | MuseTalk V1.5 | LivePortrait | Wav2Lip |
|---|---|---|---|
| License | MIT ✅ | Apache 2.0 ✅ | Research ⚠️ |
| Min GPU | V100 (16 GB) | RTX 4090 (24 GB) | GTX 1080 (8 GB) |
| Latency @ V100 | ~33ms/frame | ~40ms/frame | ~20ms/frame |
| Lip-sync quality | Good | Higher | Lower |
| Production maturity | ✅ Widely deployed | ✅ 2024 | ✅ Battle-tested |
| Our GPU target | L40S (48 GB) ✅ | L40S ✅ | L40S ✅ |

**Verdict:** MuseTalk gives reliable lip-sync at 30fps on V100-class hardware with an MIT license. Lip-sync accuracy matters more than expressiveness for a talking-head conversation agent. Toyota ethos: pick the boring one that works.

LivePortrait is the planned upgrade for Beta (better expressiveness, needs RTX 4090+).

---

## Vendor: MuseTalk submodule

```
renderer/muse_talk_vendor/   ← git submodule → github.com/TMElyralab/MuseTalk
```

Pinned commit: **`0a89dec45a0192b824e3cf4daf96c239440c5ed8`**  
Reason for pinning: MuseTalk's mmpose/DWPose integration is sensitive to mmcv ABI — never auto-upgrade without testing. Treat model upgrades as a deploy, not a dependency bump.

To update the pin (future, after testing):
```bash
cd renderer/muse_talk_vendor
git fetch origin
git checkout <new-commit-sha>
cd ../..
git add renderer/muse_talk_vendor
git commit -m "renderer: upgrade MuseTalk to <sha>"
```

---

## Interface spec — for Track C and Track D

### Input: `AudioFeatures` (lingua franca with Track C)

```python
@dataclass
class AudioFeatures:
    whisper_features: np.ndarray  # shape (T, 50, 384), dtype float32
    phonemes: list | None = None  # optional — see V2 note below
    audio_pcm: np.ndarray | None = None  # raw 16kHz mono float32
    fps: float = 25.0
```

**V1 (current):** `whisper_features` is everything MuseTalk needs. `T` = number of video frames at `fps` (default 25fps).  
**V2 (planned):** `phonemes` will carry Track C's phoneme timeline:
```python
phonemes = [
    {"phoneme": "AH", "viseme": "AA", "start_ms": 0,  "end_ms": 80},
    {"phoneme": "P",  "viseme": "PP", "start_ms": 80, "end_ms": 140},
    ...
]
```
MuseTalk V1 ignores phonemes. Future renderers (LivePortrait, custom expression system) will use them for expression control. Keep the field; populate it from Track C when it ships.

**Track C, please implement:**
1. Accept raw PCM (16kHz mono float32)
2. Produce Whisper embeddings via `musetalk.utils.audio_processor.AudioProcessor` — shape `(T, 50, 384)`
3. Optionally add phoneme timeline to `AudioFeatures.phonemes`
4. Return an `AudioFeatures` object
5. See `renderer/audio_prep.py` for the reference implementation of the Whisper path

### Output: `RendererFrame`

```python
@dataclass
class RendererFrame:
    frame: np.ndarray    # shape (256, 256, 3), dtype uint8, RGB
    frame_index: int     # 0-based position in sequence
    timestamp_ms: float  # wall-clock time of this frame
    is_stub: bool        # True if stub renderer (GPU not available)
```

**Track D (browser decode), Track G (SDK):**
- Frame format: **numpy RGB (H=256, W=256, C=3), uint8, color order RGB**
- **NOT BGR** — callers do not need to flip channels
- One frame per call to `render_frame()` or per iteration of `render_stream()`
- At 25fps: frames arrive every 40ms in real-time streaming mode
- Encode to MJPEG bytes: `cv2.imencode('.jpg', frame[:,:,::-1], [cv2.IMWRITE_JPEG_QUALITY, 85])[1].tobytes()`
- Encode to H.264: pipe frames into ffmpeg (see `cli.py` for reference)

---

## How to run — stub mode (CPU, no GPU needed)

Stub mode is for Day 1 integration testing. The engine returns the portrait image unchanged for every audio frame. No model weights, no CUDA required.

```bash
cd /Users/tenedos/edlio-presence

# Quick smoke test (all CPU-safe)
python -m pytest renderer/test_inference.py -v -k "not gpu"

# CLI test
python -m renderer.cli \
    --portrait assets/tenedos-face-v1.png \
    --audio /tmp/test.wav \
    --out /tmp/out.mp4 \
    --stub

# Python API
from renderer.engine import MuseTalkEngine, AudioFeatures
import numpy as np

engine = MuseTalkEngine("assets/tenedos-face-v1.png", use_stub=True)
af = AudioFeatures(whisper_features=np.zeros((10, 50, 384), dtype=np.float32))
for rendered in engine.render_stream_sync([af]):
    print(f"Frame {rendered.frame_index}: {rendered.frame.shape} [stub={rendered.is_stub}]")
```

---

## How to run — GPU mode (Track B container)

Once Track B GPU pod is live (see `infra/README.md`):

```bash
# On RunPod pod, inside the container:

# 1. Download model weights (one-time)
bash renderer/muse_talk_vendor/download_weights.sh

# 2. Verify GPU + weights
python infra/hello-gpu.py
ls /models/musetalkV15/unet.pth       # must exist
ls /models/whisper/config.json        # must exist

# 3. Run CLI in GPU mode
python -m renderer.cli \
    --portrait /workspace/assets/tenedos-face-v1.png \
    --audio /tmp/test.wav \
    --out /tmp/out.mp4 \
    --device cuda

# 4. Run GPU tests
RENDERER_TEST_GPU=1 python -m pytest renderer/test_inference.py -v
```

Model weights layout (must match `_DEFAULT_MODEL_DIR = /models`):
```
/models/
  musetalkV15/
    unet.pth
    musetalk.json
  sd-vae/
    config.json
    ...
  whisper/
    config.json
    ...
  dwpose/
    dw-ll_ucoco_384.pth
```

---

## Python API reference

```python
from renderer.engine import MuseTalkEngine, AudioFeatures, RendererFrame

# Init (auto-detects GPU; falls back to stub if unavailable)
engine = MuseTalkEngine(
    portrait_path="assets/tenedos-face-v1.png",
    device="cuda",          # 'cuda' or 'cpu'
    use_stub=None,          # None = auto, True = force stub, False = force GPU
    model_dir="/models",    # path to model weights (default: /models)
    use_float16=True,       # FP16 on GPU (recommended)
    fps=25.0,               # output frame rate
)

# Warm up GPU kernels before streaming (recommended)
engine.warm_up()

# Single frame
af = AudioFeatures(whisper_features=features_np)  # (T, 50, 384)
result: RendererFrame = engine.render_frame(af, frame_index=0)
frame_rgb = result.frame  # numpy (256, 256, 3) uint8 RGB

# Batch
frames: list[RendererFrame] = engine.render_batch(af)

# Sync streaming (for CLI / testing)
for rendered in engine.render_stream_sync(audio_chunk_iterator):
    send(rendered.frame)

# Async streaming (for WebRTC/WebSocket production path)
async for rendered in engine.render_stream(async_audio_chunk_iterator):
    await ws.send_bytes(encode_mjpeg(rendered.frame))
```

---

## Latency budget

**Target:** <50ms per frame end-to-end on target GPU

| Stage | Budget | Notes |
|---|---|---|
| Audio feature extraction (Whisper) | ~20ms | Track C's job; we receive pre-computed features |
| MuseTalk UNet inference | ~33ms | Published benchmark: V100 @ 30fps |
| VAE decode | ~5ms | Included in 33ms figure |
| Frame post-processing | <2ms | Resize, color convert |
| **Total renderer** | **~35ms** | **Well under 50ms target** |
| H.264 encode (NVENC) | ~5ms | Track B's encoder, not our budget |
| Network/WebRTC | variable | Track G's problem |
| **End-to-end target** | **<200ms** | See technical-brief.md |

Benchmark source: [MuseTalk README](https://github.com/TMElyralab/MuseTalk#performance) — ~30fps on V100 (16 GB). L40S (48 GB) should do better.

---

## Known gaps — to validate Day 2 when GPU is live

1. **MuseTalk inference round-trip** — `_MuseTalkRenderer.render_frame()` is written against the API we read from MuseTalk's source, but NOT yet run against a real GPU. The UNet forward pass + VAE decode need to be verified on actual hardware.

2. **Portrait face detection** — In production, MuseTalk runs DWPose face detection on the portrait to find and crop the face region. The stub skips this. The GPU path needs `musetalk.utils.preprocessing.get_landmark_and_bbox()` wired in before it's production-ready.

3. **Blending / inpainting** — MuseTalk composes the generated 256×256 face region back into the original portrait frame. `_MuseTalkRenderer` currently returns the raw face region only. For full-portrait output, wire in `musetalk.utils.blending.get_image()`. Document if Track D needs full-portrait or face-crop-only.

4. **FP16 NaN stability** — Need to verify `use_float16=True` doesn't produce NaN on V100 with our specific portrait. If NaNs appear, set `use_float16=False`.

5. **Whisper model load time** — Loading Whisper encoder at init adds ~5–10s cold start. Confirm this is acceptable for RunPod pod start-up. If not, lazy-load on first inference.

6. **Multi-session latent caching** — `_precompute_portrait_latents()` precomputes for one portrait. Multi-identity sessions (different portrait per session) need a portrait-keyed cache. Not needed for MVP (one Tenedos portrait), but plan before Phase 2.

7. **Load test at 5+ concurrent sessions** — Technical brief estimates 5–8 sessions/L40S. Needs validation before any public pricing claim.

---

## File layout

```
renderer/
  __init__.py          — package exports
  engine.py            — MuseTalkEngine public API + stub + GPU renderer
  audio_prep.py        — WAV → AudioFeatures conversion (Whisper embeddings)
  cli.py               — command-line test tool
  test_inference.py    — pytest smoke tests (stub + GPU)
  requirements.txt     — Python dependencies
  README.md            — this file
  muse_talk_vendor/    — MuseTalk git submodule (pinned to 0a89dec)
```
