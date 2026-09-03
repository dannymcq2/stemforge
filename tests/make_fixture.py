"""Render a synthetic track with a known key and tempo, for testing."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import soundfile as sf

SR = 44100


def midi_to_hz(note: int) -> float:
    return 440.0 * 2 ** ((note - 69) / 12)


def tone(freq: float, duration: float, harmonics: int = 5, decay: float = 3.0) -> np.ndarray:
    t = np.linspace(0, duration, int(SR * duration), endpoint=False)
    out = np.zeros_like(t)
    for h in range(1, harmonics + 1):
        out += np.sin(2 * np.pi * freq * h * t) / (h ** 1.6)
    return out * np.exp(-decay * t)


def noise_hit(duration: float, decay: float, low: float = 0.0) -> np.ndarray:
    t = np.linspace(0, duration, int(SR * duration), endpoint=False)
    rng = np.random.default_rng(0)
    out = rng.standard_normal(t.size)
    if low:
        # crude one-pole high-pass so hats sit above the bass
        alpha = np.exp(-2 * np.pi * low / SR)
        filtered = np.zeros_like(out)
        prev = 0.0
        for i, x in enumerate(out):
            prev = alpha * prev + (1 - alpha) * x
            filtered[i] = x - prev
        out = filtered
    return out * np.exp(-decay * t)


def render(
    bpm: float = 128.0,
    bars: int = 8,
    key_root: int = 57,
    pitched: bool = True,
    drums: bool = True,
) -> np.ndarray:
    """A minor by default (MIDI 57 = A3), i-VI-III-VII.

    `pitched` and `drums` select which layers are rendered, so tests can ask
    for a drums-only signal that stands in for a separated drum stem.
    """
    beat = 60.0 / bpm
    total = int(SR * beat * 4 * bars) + SR
    mix = np.zeros(total)

    minor_triad = [0, 3, 7]
    major_triad = [0, 4, 7]
    progression = [(0, minor_triad), (8, major_triad), (3, major_triad), (10, major_triad)]

    for bar in range(bars):
        bar_start = bar * 4 * beat
        offset, intervals = progression[bar % len(progression)]

        # Chord pad, whole bar.
        for interval in intervals if pitched else ():
            note = key_root + 12 + offset + interval
            sig = tone(midi_to_hz(note), beat * 4, harmonics=6, decay=0.8) * 0.12
            start = int(bar_start * SR)
            mix[start:start + sig.size] += sig[: total - start]

        # Bass on every beat.
        for b in range(4) if pitched else ():
            note = key_root - 12 + offset
            sig = tone(midi_to_hz(note), beat * 0.9, harmonics=3, decay=4.0) * 0.35
            start = int((bar_start + b * beat) * SR)
            mix[start:start + sig.size] += sig[: total - start]

        if not drums:
            continue

        # Kick on 1 and 3, snare on 2 and 4, hats on eighths.
        for b in (0, 2):
            sig = tone(55.0, 0.25, harmonics=1, decay=22.0) * 0.6
            start = int((bar_start + b * beat) * SR)
            mix[start:start + sig.size] += sig[: total - start]
        for b in (1, 3):
            sig = noise_hit(0.18, 30.0) * 0.28
            start = int((bar_start + b * beat) * SR)
            mix[start:start + sig.size] += sig[: total - start]
        for b in range(8):
            sig = noise_hit(0.05, 90.0, low=6000.0) * 0.14
            start = int((bar_start + b * beat / 2) * SR)
            mix[start:start + sig.size] += sig[: total - start]

    return (mix / np.max(np.abs(mix)) * 0.89).astype(np.float32)


if __name__ == "__main__":
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "tests/fixture_am_128.wav")
    out.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out), render(), SR, subtype="PCM_16")
    print(out)
