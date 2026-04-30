"""
test_sample.py — Smoke test and latency benchmark for the phoneme/viseme extractor.

Usage:
    cd audio && python test_sample.py /path/to/audio.wav
    cd audio && python test_sample.py /path/to/audio.wav --model base
    cd audio && python test_sample.py --self-test   # generate test audio via macOS 'say'

Output:
    - Full phoneme/viseme event stream printed to stdout
    - Latency benchmark (ms per second of audio)
    - Summary statistics (total phonemes, viseme distribution)

Exit codes:
    0 = pass (events extracted, latency within threshold)
    1 = failure (no events or error)
"""

import argparse
import os
import subprocess
import sys
import tempfile
import time

# Ensure we import from the local audio/ directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from extractor import extract_from_wav, PhonemeEvent
from visemes import VISEME_NAMES, NUM_VISEMES


# ─── Latency threshold (ms per second of audio) ────────────────────────────────
# At this rate, we can keep up with real-time streaming:
#   1000ms/s = exactly real-time
#    100ms/s = 10× faster than real-time (our target for MVP)
LATENCY_THRESHOLD_MS_PER_S = 200.0  # allow 200ms of processing per 1s of audio


def get_audio_duration_s(path: str) -> float:
    """Get duration of a WAV file in seconds using wave module."""
    import wave
    try:
        with wave.open(path, 'rb') as wf:
            frames = wf.getnframes()
            rate = wf.getframerate()
            return frames / float(rate)
    except Exception:
        return 0.0


def print_events(events: list[PhonemeEvent], max_events: int = 50) -> None:
    """Pretty-print phoneme events."""
    print(f"\n{'Time(ms)':>10}  {'Phoneme':<8}  {'Dur(ms)':>8}  {'Viseme':>7}  {'Name':<8}  Word")
    print("─" * 70)

    shown = 0
    for event in events:
        if shown >= max_events:
            remaining = len(events) - max_events
            print(f"  ... {remaining} more events (use --all to show all)")
            break
        print(
            f"{event.timestamp_ms:>10.1f}  {event.phoneme:<8}  "
            f"{event.duration_ms:>8.1f}  {event.viseme_id:>7}  "
            f"{event.viseme_name:<8}  {event.word}"
        )
        shown += 1


def print_viseme_distribution(events: list[PhonemeEvent]) -> None:
    """Print distribution of visemes across the audio."""
    from collections import Counter

    counts = Counter(e.viseme_id for e in events)
    total = len(events)

    print("\nViseme distribution:")
    print(f"{'ID':>4}  {'Name':<8}  {'Count':>6}  {'%':>6}  Bar")
    print("─" * 50)
    for vid in range(NUM_VISEMES):
        count = counts.get(vid, 0)
        if count == 0:
            continue
        pct = 100.0 * count / total
        bar = "█" * int(pct / 2)
        print(f"{vid:>4}  {VISEME_NAMES[vid]:<8}  {count:>6}  {pct:>5.1f}%  {bar}")


def benchmark(
    wav_path: str,
    model_size: str = "tiny",
    runs: int = 3,
) -> dict:
    """
    Run the extractor N times and return latency stats.
    First run includes model load time; subsequent runs are warm.
    """
    duration_s = get_audio_duration_s(wav_path)
    if duration_s <= 0:
        print(f"Warning: could not determine audio duration for {wav_path}")
        duration_s = 1.0

    cold_times = []
    warm_times = []

    for i in range(runs):
        t0 = time.perf_counter()
        events = extract_from_wav(wav_path, model_size=model_size)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        if i == 0:
            cold_times.append(elapsed_ms)
        else:
            warm_times.append(elapsed_ms)

    cold_total_ms = cold_times[0] if cold_times else 0
    warm_avg_ms = sum(warm_times) / len(warm_times) if warm_times else cold_total_ms

    return {
        "audio_duration_s": duration_s,
        "audio_duration_ms": duration_s * 1000.0,
        "cold_ms": cold_total_ms,
        "warm_avg_ms": warm_avg_ms,
        "cold_ms_per_s": cold_total_ms / duration_s,
        "warm_ms_per_s": warm_avg_ms / duration_s,
        "realtime_factor_warm": duration_s / (warm_avg_ms / 1000.0),
        "event_count": len(events),
        "events": events,
    }


def generate_test_audio(output_path: str) -> bool:
    """Generate test audio using macOS 'say' command."""
    aiff_path = output_path.replace(".wav", ".aiff")
    text = "Hello, my name is Tenedos. I am an AI assistant built for the Edlio presence layer. Let me demonstrate phoneme extraction."

    print(f"Generating test audio via macOS 'say'...")
    try:
        subprocess.run(
            ["say", text, "-o", aiff_path],
            check=True, capture_output=True
        )
        # Convert to 16kHz mono wav
        subprocess.run(
            ["ffmpeg", "-y", "-i", aiff_path, "-ar", "16000", "-ac", "1", output_path],
            check=True, capture_output=True
        )
        os.unlink(aiff_path)
        print(f"Test audio saved to: {output_path}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Failed to generate test audio: {e}")
        return False
    except FileNotFoundError as e:
        print(f"Required tool not found (say or ffmpeg): {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Phoneme/viseme extractor smoke test and benchmark"
    )
    parser.add_argument(
        "wav_path", nargs="?",
        help="Path to WAV file to process (16kHz mono recommended)"
    )
    parser.add_argument(
        "--model", default="tiny", choices=["tiny", "base", "small"],
        help="Whisper model size (default: tiny)"
    )
    parser.add_argument(
        "--runs", type=int, default=3,
        help="Number of benchmark runs (default: 3; first is cold)"
    )
    parser.add_argument(
        "--all", action="store_true",
        help="Show all events (default: first 50)"
    )
    parser.add_argument(
        "--self-test", action="store_true",
        help="Generate test audio via macOS 'say' and run benchmark"
    )
    parser.add_argument(
        "--no-bench", action="store_true",
        help="Skip benchmark runs (just show events once)"
    )

    args = parser.parse_args()

    # Resolve wav path
    wav_path = args.wav_path
    if args.self_test or wav_path is None:
        tmp_path = "/tmp/test_tenedos.wav"
        if not os.path.exists(tmp_path):
            if not generate_test_audio(tmp_path):
                print("ERROR: Could not generate test audio. Provide a wav file explicitly.")
                sys.exit(1)
        wav_path = tmp_path

    if not os.path.exists(wav_path):
        print(f"ERROR: File not found: {wav_path}")
        sys.exit(1)

    print(f"\n{'='*70}")
    print(f"Edlio Presence Layer — Phoneme/Viseme Extractor Test")
    print(f"{'='*70}")
    print(f"Audio file:  {wav_path}")
    print(f"Model:       whisper-{args.model} (CPU, int8)")
    print(f"Runs:        {args.runs} (1 cold + {args.runs-1} warm)")
    print(f"{'='*70}\n")

    # First pass — show events
    print("Extracting phoneme/viseme stream...")
    t0 = time.perf_counter()
    events = extract_from_wav(wav_path, model_size=args.model)
    first_run_ms = (time.perf_counter() - t0) * 1000.0
    duration_s = get_audio_duration_s(wav_path)

    if not events:
        print("ERROR: No events extracted. Check audio file and model.")
        sys.exit(1)

    print(f"✓ Extracted {len(events)} phoneme events from {duration_s:.2f}s of audio")
    print(f"  Cold run: {first_run_ms:.0f}ms ({first_run_ms/duration_s:.1f}ms/s audio)")

    max_events = None if args.all else 50
    print_events(events, max_events=max_events if max_events else len(events))
    print_viseme_distribution(events)

    # Benchmark runs
    if not args.no_bench and args.runs > 1:
        print(f"\n{'='*70}")
        print(f"Benchmarking ({args.runs - 1} warm runs)...")
        warm_times = []
        for i in range(args.runs - 1):
            t0 = time.perf_counter()
            _ = extract_from_wav(wav_path, model_size=args.model)
            warm_times.append((time.perf_counter() - t0) * 1000.0)
            print(f"  Run {i+2}: {warm_times[-1]:.0f}ms")

        warm_avg = sum(warm_times) / len(warm_times)
        warm_ms_per_s = warm_avg / duration_s
        realtime_factor = duration_s / (warm_avg / 1000.0)

        print(f"\n{'─'*50}")
        print(f"Latency Summary (whisper-{args.model}, CPU int8, arm64)")
        print(f"{'─'*50}")
        print(f"  Audio duration:     {duration_s:.2f}s")
        print(f"  Cold run:           {first_run_ms:.0f}ms  ({first_run_ms/duration_s:.1f}ms/s)")
        print(f"  Warm avg:           {warm_avg:.0f}ms  ({warm_ms_per_s:.1f}ms/s)")
        print(f"  Realtime factor:    {realtime_factor:.1f}× faster than real-time")
        print(f"  Threshold:          {LATENCY_THRESHOLD_MS_PER_S:.0f}ms/s (10× real-time)")

        status = "✓ PASS" if warm_ms_per_s <= LATENCY_THRESHOLD_MS_PER_S else "✗ FAIL"
        print(f"\n  Result: {status}")

        if warm_ms_per_s > LATENCY_THRESHOLD_MS_PER_S:
            print(f"  WARNING: {warm_ms_per_s:.1f}ms/s exceeds {LATENCY_THRESHOLD_MS_PER_S}ms/s threshold")
            print(f"  Consider using 'tiny' model or reducing chunk size")
            sys.exit(1)
    else:
        warm_ms_per_s = first_run_ms / duration_s
        realtime_factor = duration_s / (first_run_ms / 1000.0)
        print(f"\nSingle-run latency: {first_run_ms:.0f}ms ({warm_ms_per_s:.1f}ms/s, {realtime_factor:.1f}× realtime)")

    print(f"\n{'='*70}")
    print("Smoke test complete.")
    sys.exit(0)


if __name__ == "__main__":
    main()
