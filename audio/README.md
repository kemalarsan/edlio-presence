# audio/ — Phoneme/Viseme Extraction

**Track C** of the Edlio Presence Layer. Converts arbitrary audio (WAV or streaming PCM) into a time-aligned phoneme/viseme stream for driving mouth shapes in the face renderer.

---

## Decision: Which Pipeline?

**Chosen: faster-whisper (whisper-tiny, CPU int8) + CMU ARPAbet → viseme mapping**

### Why faster-whisper over the alternatives

| Option | Verdict | Reason |
|--------|---------|--------|
| **faster-whisper** ✅ | **Chosen** | CTranslate2 int8, 22× real-time on Apple M2 CPU. Word timestamps built-in. MIT license. Actively maintained. Well-understood failure modes. |
| WhisperX | Too heavy | Adds wav2vec2 phoneme alignment — great accuracy, but adds ~1.5GB dependencies (transformers + torchaudio + alignment models). Overkill for MVP. |
| Montreal Forced Aligner | Batch only | Designed for offline alignment. Requires audio + transcript pair. Not suitable for streaming or TTS-only contexts. |
| Phonemizer | Wrong direction | text→phoneme, not audio→phoneme. Useful for TTS normalization, not audio ingestion. |
| Allosaurus | Multilingual but slower | Universal phone recognition. Slower than whisper-tiny on CPU, less maintained. No word alignment natively. |
| Rhubarb Lip Sync | Considered | CLI tool, fast, viseme-native output. Good option but requires C++ binary and can't be imported as a Python module. Harder to integrate into streaming pipeline. Reserve for Day 4 if faster-whisper proves insufficient. |

### Toyota ethos rationale

faster-whisper is the boring-correct answer: it's in production at thousands of sites, int8 CPU mode is fast enough for our streaming chunk sizes, and word-level timestamps (±20ms) are sufficient to drive 25fps video without perceptible misalignment. We don't need phoneme-level accuracy from the audio model — we distribute phoneme timing evenly within each word span and the eye can't tell the difference at 25fps.

---

## ⚠️ Critical: MuseTalk Does NOT Use Phonemes

After reading MuseTalk's source (`musetalk/utils/audio_processor.py`), we found that **MuseTalk's UNet consumes raw Whisper encoder hidden states**, not phonemes or visemes.

The pipeline in MuseTalk:
1. Load audio → mel spectrogram (Whisper feature extractor, 30s window)  
2. Pass mel through `whisper.encoder()` → get all 12 `hidden_states`
3. Stack hidden states: shape `[1, T, 12, 384]` at 50 Hz
4. Slice into per-frame clips with padding → shape `[num_frames, 10, 5, 384]`
5. Rearrange → `[num_frames, 50, 384]` → feed to UNet as `audio_prompts`

**What this means for Track A:**
- Track A (MuseTalk renderer) must use `extract_musetalk_features()` from `extractor.py`
- The phoneme/viseme stream from `extract_from_wav()` is for **Wav2Lip fallback** and **renderer-agnostic use cases** (future non-MuseTalk backends)
- Do NOT feed phoneme events into MuseTalk directly — it won't work

---

## Files

```
audio/
├── extractor.py       — Core extraction: phoneme/viseme + MuseTalk feature extraction
├── visemes.py         — ARPAbet/IPA → viseme ID mapping (Preston Blair 15-class)
├── test_sample.py     — Smoke test + latency benchmark
├── requirements.txt   — Python dependencies
└── README.md          — This file
```

---

## How to Run

### Setup

```bash
cd /Users/tenedos/edlio-presence/audio

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install deps
pip install -r requirements.txt
```

### Run the smoke test

```bash
# With a WAV file you provide
python test_sample.py /path/to/audio.wav

# Self-test: generates audio via macOS 'say' and runs benchmark
python test_sample.py --self-test

# Use whisper-base for better accuracy (slower)
python test_sample.py /path/to/audio.wav --model base

# Show all events (not just first 50)
python test_sample.py /path/to/audio.wav --all
```

### Use in code

```python
from extractor import extract_from_wav, PhonemeEvent

events = extract_from_wav("/path/to/audio.wav")
for e in events:
    print(f"{e.timestamp_ms:.0f}ms  {e.phoneme:<6}  viseme={e.viseme_id} ({e.viseme_name})")
```

**Output format:** `list[PhonemeEvent]` where each event has:

| Field | Type | Description |
|-------|------|-------------|
| `timestamp_ms` | `float` | Start time in milliseconds from audio start |
| `phoneme` | `str` | ARPAbet symbol, stress-stripped (e.g. `"AH"`, `"T"`, `"SIL"`) |
| `duration_ms` | `float` | Duration in milliseconds |
| `viseme_id` | `int` | Preston Blair viseme ID, 0–15 |
| `viseme_name` | `str` | Human-readable viseme name (e.g. `"PP"`, `"aa"`, `"sil"`) |
| `word` | `str` | Source word (for debugging) |
| `confidence` | `float` | Whisper word-level confidence (0–1) |

---

## Latency Benchmark

**Hardware:** Apple M2 (arm64), macOS, CPU-only (no GPU)  
**Audio:** 7.61s TTS output (`say` command, 16kHz mono WAV)  
**Model:** `whisper-tiny` (int8 quantized via CTranslate2)

| Run | Wall time | ms per second of audio |
|-----|-----------|------------------------|
| Cold (first run, includes model load) | 4,956ms | 651ms/s |
| Warm run 1 | 332ms | 43.6ms/s |
| Warm run 2 | 333ms | 43.7ms/s |

**Warm throughput: 43.7ms/s audio = 22.9× faster than real-time** ✅

Target was <200ms/s (5× real-time). We're at 43.7ms/s — over 4× better than target. This gives substantial headroom for the streaming pipeline to buffer audio, extract, and still deliver visemes ahead of the renderer's need.

Cold start: ~5s (model load). For streaming deployment, load the model at service startup, not per-request.

---

## Streaming Support

`extract_streaming()` in `extractor.py` is implemented but Day 1 status is **scaffold only** — it buffers incoming PCM chunks into 1.5s windows and calls `extract_from_wav()` on each. Timestamps are adjusted by the buffer offset.

This is a validated approach but hasn't been load-tested under continuous WebRTC input. Day 3-4 goal: replace temp-file approach with in-memory numpy array pipeline and benchmark actual streaming latency on the GPU server.

For Day 1 integration: use `extract_from_wav()` for file-based testing. The streaming path will be wired by Track D (streaming infrastructure).

---

## Viseme Classes

15-class Preston Blair / Disney Animation standard. Industry-standard for real-time lip sync.

| ID | Name | Phonemes | Mouth shape |
|----|------|----------|-------------|
| 0 | sil | silence, HH | Closed/neutral |
| 1 | PP | P, B, M | Lips pressed together |
| 2 | FF | F, V | Upper teeth on lower lip |
| 3 | TH | TH, DH | Tongue between teeth |
| 4 | DD | T, D | Tongue tip up |
| 5 | kk | K, G, NG | Back of tongue raised |
| 6 | CH | CH, JH, SH, ZH | Lips slightly forward |
| 7 | SS | S, Z | Teeth close, narrow gap |
| 8 | nn | N, L | Tongue to ridge |
| 9 | RR | R, ER | Lips slightly pursed |
| 10 | aa | AA, AH, AX | Wide open mouth |
| 11 | E | EH, AE | Lips spread, mid-open |
| 12 | ih | IH, IY, Y | Lips slightly spread, near-close |
| 13 | oh | AO, AW, OY | Rounded, mid-open |
| 14 | ou | OW, UH, UW | Rounded, close |
| 15 | W | W | Strongly rounded, pursed |

---

## Known Limitations (MVP)

1. **English only.** Whisper-tiny auto-detects language but phoneme→viseme mapping is English ARPAbet. For other languages, the phoneme distribution will work passably but accuracy degrades. International support is a Phase 2 item.

2. **Phoneme timing is approximate.** We evenly distribute phoneme duration within each word boundary. Actual phoneme timing can vary by ±40ms. At 25fps (40ms/frame), this means off-by-one-frame errors are possible. For lip sync at conversation speed, this is imperceptible.

3. **MuseTalk incompatibility.** As noted above: phoneme events are for non-MuseTalk renderers. Use `extract_musetalk_features()` for MuseTalk.

4. **CMU dict coverage.** Proper nouns, technical terms, and neologisms fall back to character-level phoneme approximation. Install `pip install pronouncing` to significantly improve coverage for uncommon words.

5. **Streaming not production-ready.** `extract_streaming()` uses temp files per chunk. Production streaming will need in-memory numpy pipeline (Day 3-4).

6. **No Mandarin/Hindi phoneme support.** The IPA→viseme table covers Latin-script languages best. For CJK audio, Whisper still produces transcription but phoneme-level alignment will be less accurate.

---

## Integration Notes for Track A (MuseTalk Renderer)

```python
# For MuseTalk: use this function, NOT extract_from_wav
from extractor import extract_musetalk_features

features = extract_musetalk_features("audio.wav", fps=25.0)
# features.audio_prompts: torch.Tensor [num_frames, 50, 384]
# Directly compatible with MuseTalk's datagen() / UNet audio_prompts input
```

Requires `torch` + `transformers` + `einops` (not in base requirements.txt — listed as optional deps). The Whisper model (openai/whisper-tiny) auto-downloads to `~/.cache/huggingface/hub/` on first call.

---

## Future Improvements (Post-MVP)

- **Rhubarb Lip Sync integration** as an optional extraction backend (CLI wrapper). Rhubarb outputs visemes directly and is faster than Whisper for pure lip-sync use cases without transcription.
- **In-memory streaming** (no temp files) for `extract_streaming()`.
- **phoneme confidence per frame** — currently confidence is word-level; phoneme-level would let the renderer smooth between visemes more accurately.
- **Multi-language viseme tables** — CMU ARPAbet covers English; add SAMPA for European languages.
