# Edlio Presence Layer

> An open, pluggable avatar/face layer for AI agents.
> Provider-agnostic audio input. Toyota-grade reliability.

## What this is

A presence layer that turns an audio conversation into a synced video-of-a-face conversation. Provider-agnostic — works with OpenAI Realtime, ElevenLabs, Cartesia, or any audio source.

**What it is NOT:**
- Not an agent (no LLM built in)
- Not a bundled stack (unlike Anam/Tavus which bundle face+voice+brain)
- Not a demo toy — reliability is the point

## Why now

Open-source face-gen models (MuseTalk, LivePortrait, Wav2Lip) are production-ready. Every agent platform needs faces. Nobody is building the shared primitive. Anam/Tavus bundle; we don't.

## Core ethos: Toyota

- Reliability > flashiness
- Graceful degradation is a core feature
- "Boring correctness" > "novel research papers"
- Never breaks in front of a kid, teacher, parent, or CEO
- Observable — when it breaks, we know why within 30 seconds

## MVP differentiator

Lip-sync + **webcam gaze tracking**. The face appears to look at the user because we know where the user's eyes are (client-side MediaPipe Face Mesh → gaze params → renderer). This crosses the believability threshold nobody else hits.

## Build status

**Day 1 (2026-04-30):** Repo created, skeleton in progress.

See [`refs/presence-layer/`](https://github.com/kemalarsan/tenedos-workspace) in the Tenedos workspace for:
- `technical-brief.md` — product positioning, architecture, model selection, cost model
- `mvp-scope.md` — 14-day build plan with parallel subagent tracks
- `identity-design.md` — Tenedos V1 face and fleet identity strategy

## Track layout

```
edlio-presence/
├── renderer/       # Track A: face-gen inference (MuseTalk)
├── audio/          # Track C: phoneme/viseme extraction
├── gaze/           # Track E: client-side webcam → gaze vector
├── sdk/            # Track G: npm package @edlio/presence
├── infra/          # Track B: Dockerfiles, RunPod configs
├── observability/  # Track K: metrics, dashboards
├── assets/         # portraits, test audio, reference clips
└── docs/           # integration guides
```

## License

TBD — see Decision Point #1 in the technical brief.
