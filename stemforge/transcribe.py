"""Polyphonic audio-to-MIDI transcription on top of Spotify's Basic Pitch."""

from __future__ import annotations

import contextlib
import io
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .analysis import PITCH_CLASSES, TempoResult
from .audio import Audio, write

# Per-stem transcription settings. Narrow frequency windows and tuned
# thresholds matter more than the model choice: a bass line transcribed with
# vocal settings picks up a fifth of harmonics that are not played notes.
@dataclass(frozen=True)
class Preset:
    label: str
    onset_threshold: float
    frame_threshold: float
    minimum_note_length: float  # milliseconds
    minimum_frequency: float | None
    maximum_frequency: float | None
    program: int  # General MIDI program number
    is_drum: bool = False
    pitch_bend: bool = False


PRESETS: dict[str, Preset] = {
    "bass": Preset(
        "Bass", onset_threshold=0.5, frame_threshold=0.3, minimum_note_length=90.0,
        minimum_frequency=30.0, maximum_frequency=400.0, program=33,
    ),
    "vocals": Preset(
        "Vocals", onset_threshold=0.6, frame_threshold=0.3, minimum_note_length=120.0,
        minimum_frequency=80.0, maximum_frequency=1400.0, program=53, pitch_bend=True,
    ),
    "piano": Preset(
        "Piano", onset_threshold=0.5, frame_threshold=0.3, minimum_note_length=60.0,
        minimum_frequency=27.5, maximum_frequency=4200.0, program=0,
    ),
    "guitar": Preset(
        "Guitar", onset_threshold=0.55, frame_threshold=0.3, minimum_note_length=70.0,
        minimum_frequency=70.0, maximum_frequency=2200.0, program=27,
    ),
    "other": Preset(
        "Other", onset_threshold=0.55, frame_threshold=0.33, minimum_note_length=80.0,
        minimum_frequency=55.0, maximum_frequency=3000.0, program=81,
    ),
    "mix": Preset(
        "Full mix", onset_threshold=0.6, frame_threshold=0.35, minimum_note_length=90.0,
        minimum_frequency=40.0, maximum_frequency=3000.0, program=0,
    ),
}

# Drums get onset detection rather than pitch tracking; these are the GM
# percussion notes each detected band maps onto.
DRUM_MAP = {"kick": 36, "snare": 38, "hihat": 42}

DEFAULT_PRESET = PRESETS["other"]

# Basic Pitch resamples internally; this is just what we hand it.
TRANSCRIBE_SAMPLE_RATE = 22050


@dataclass
class Note:
    start: float
    end: float
    pitch: int
    velocity: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "start": round(self.start, 4),
            "end": round(self.end, 4),
            "pitch": self.pitch,
            "name": f"{PITCH_CLASSES[self.pitch % 12]}{self.pitch // 12 - 1}",
            "velocity": self.velocity,
        }


@dataclass
class Transcription:
    stem: str
    notes: list[Note]
    program: int
    is_drum: bool

    def as_dict(self) -> dict[str, Any]:
        pitches = [n.pitch for n in self.notes]
        return {
            "stem": self.stem,
            "note_count": len(self.notes),
            "lowest": min(pitches) if pitches else None,
            "highest": max(pitches) if pitches else None,
            "program": self.program,
            "is_drum": self.is_drum,
        }


def _quantize_time(t: float, first_beat: float, seconds_per_step: float) -> float:
    steps = round((t - first_beat) / seconds_per_step)
    return first_beat + steps * seconds_per_step


def quantize(
    notes: list[Note],
    tempo: TempoResult,
    subdivision: int = 4,
    strength: float = 1.0,
    min_duration_steps: float = 0.5,
) -> list[Note]:
    """Snap note starts and ends toward a `subdivision`-per-beat grid.

    `strength` of 1.0 lands notes exactly on the grid; 0.5 moves them halfway,
    which keeps some of the original feel.
    """
    if subdivision <= 0 or tempo.bpm <= 0 or not notes:
        return notes

    step = 60.0 / tempo.bpm / subdivision
    first = tempo.first_beat
    floor = step * min_duration_steps

    out: list[Note] = []
    for note in notes:
        start = _quantize_time(note.start, first, step)
        end = _quantize_time(note.end, first, step)
        start = note.start + (start - note.start) * strength
        end = note.end + (end - note.end) * strength
        if end - start < floor:
            end = start + max(floor, note.end - note.start)
        out.append(Note(start, end, note.pitch, note.velocity))
    return out


def transcribe(
    audio: Audio,
    preset: Preset | str = DEFAULT_PRESET,
    onset_threshold: float | None = None,
    frame_threshold: float | None = None,
) -> Transcription:
    """Transcribe one stem to notes."""
    from basic_pitch.inference import predict

    if isinstance(preset, str):
        name = preset
        preset = PRESETS.get(preset, DEFAULT_PRESET)
    else:
        name = preset.label.lower()

    # Basic Pitch reads from disk, so stage a mono WAV in a temp directory.
    with tempfile.TemporaryDirectory() as tmp:
        staged = Path(tmp) / "stem.wav"
        mono = audio.resampled(TRANSCRIBE_SAMPLE_RATE).mono()
        write(staged, mono, TRANSCRIBE_SAMPLE_RATE, subtype="PCM_16")

        # Basic Pitch's CoreML path prints tensor shapes to stdout on every
        # call, which would litter the CLI output and the server log.
        with contextlib.redirect_stdout(io.StringIO()):
            _, _, note_events = predict(
                staged,
                onset_threshold=(
                    preset.onset_threshold if onset_threshold is None else onset_threshold
                ),
                frame_threshold=(
                    preset.frame_threshold if frame_threshold is None else frame_threshold
                ),
                minimum_note_length=preset.minimum_note_length,
                minimum_frequency=preset.minimum_frequency,
                maximum_frequency=preset.maximum_frequency,
                multiple_pitch_bends=preset.pitch_bend,
                melodia_trick=True,
            )

    notes = [
        Note(
            start=float(start),
            end=float(end),
            pitch=int(pitch),
            velocity=int(np.clip(round(amplitude * 127), 1, 127)),
        )
        for start, end, pitch, amplitude, *_ in note_events
    ]
    notes.sort(key=lambda n: (n.start, n.pitch))
    return Transcription(name, notes, preset.program, preset.is_drum)


# Frequency bands each percussion voice lives in, and the gate that decides
# whether that band really fired on a given onset.
_ONSET_MERGE_FRAMES = 2   # ~23 ms at the analysis hop size
_BAND_SHARE = 0.35        # a band this far below the loudest is spill, not a hit

_DRUM_BANDS = {
    "kick": (20.0, 150.0),
    "snare": (180.0, 1200.0),
    "hihat": (4500.0, 12000.0),
}


def transcribe_drums(audio: Audio, sensitivity: float = 1.0) -> Transcription:
    """Onset-based drum transcription into kick / snare / hi-hat.

    Basic Pitch models pitched instruments, so drums are handled separately.
    Onsets are found once across the whole stem, then each one is classified by
    which frequency bands actually spike at that moment. Classifying shared
    onsets — rather than running an independent detector per band — is what
    keeps a snare's low-frequency body from also being counted as a kick, while
    still allowing a kick and a hi-hat on the same beat to both be written.
    """
    import librosa

    sr = 22050
    hop = 256
    y = audio.resampled(sr).mono()
    if not np.any(y):
        return Transcription("drums", [], program=0, is_drum=True)

    stft = np.abs(librosa.stft(y, n_fft=2048, hop_length=hop))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)

    # Positive spectral flux per band: how much that band's energy jumped.
    band_flux: dict[str, np.ndarray] = {}
    for band, (low, high) in _DRUM_BANDS.items():
        mask = (freqs >= low) & (freqs <= high)
        if not mask.any():
            band_flux[band] = np.zeros(stft.shape[1])
            continue
        energy = stft[mask].sum(axis=0)
        flux = np.diff(energy, prepend=energy[0])
        band_flux[band] = np.maximum(flux, 0.0)

    # Each band proposes its own onsets. A quiet hi-hat under a loud kick would
    # never clear a threshold on a summed envelope, so the bands are peak-picked
    # independently and their proposals pooled.
    proposals: dict[str, set[int]] = {}
    for band, flux in band_flux.items():
        peak = flux.max() or 1.0
        found = librosa.onset.onset_detect(
            onset_envelope=flux / peak, sr=sr, hop_length=hop, units="frames",
            backtrack=False, delta=0.10 / max(sensitivity, 0.1),
        )
        proposals[band] = {int(f) for f in found}

    # Collapse near-simultaneous proposals so one hit yields one onset.
    onsets: list[int] = []
    for frame in sorted(set().union(*proposals.values()) if proposals else set()):
        if onsets and frame - onsets[-1] <= _ONSET_MERGE_FRAMES:
            continue
        onsets.append(frame)

    notes: list[Note] = []
    for frame in onsets:
        # Relative strength of each band at this onset, on its own scale, so a
        # loud kick does not drown out a hi-hat measured in different units.
        levels: dict[str, float] = {}
        for band, flux in band_flux.items():
            if not any(abs(p - frame) <= _ONSET_MERGE_FRAMES for p in proposals[band]):
                levels[band] = 0.0
                continue
            lo = max(frame - _ONSET_MERGE_FRAMES, 0)
            hi = min(frame + _ONSET_MERGE_FRAMES + 1, flux.size)
            levels[band] = float(flux[lo:hi].max()) / (float(flux.max()) or 1.0)

        loudest = max(levels.values(), default=0.0)
        if loudest <= 0.0:
            continue

        time = float(librosa.frames_to_time(frame, sr=sr, hop_length=hop))
        for band, level in levels.items():
            # Bands well below the strongest one are that hit's spectral
            # spill-over, not a separate drum being struck.
            if level < _BAND_SHARE * loudest:
                continue
            notes.append(
                Note(
                    start=time,
                    end=time + 0.08,
                    pitch=DRUM_MAP[band],
                    velocity=int(np.clip(round(55 + level * 72), 1, 127)),
                )
            )

    notes.sort(key=lambda n: (n.start, n.pitch))
    return Transcription("drums", notes, program=0, is_drum=True)


def write_midi(
    path: str | Path,
    transcriptions: list[Transcription],
    tempo: TempoResult | None = None,
    key_tonic: int | None = None,
    key_mode: str | None = None,
) -> Path:
    """Write one or more transcriptions to a single tempo-stamped MIDI file."""
    import pretty_midi

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    bpm = tempo.bpm if tempo and tempo.bpm > 0 else 120.0
    midi = pretty_midi.PrettyMIDI(initial_tempo=bpm)

    if key_tonic is not None:
        # pretty_midi numbers keys 0-11 major, 12-23 minor.
        number = key_tonic + (12 if key_mode == "minor" else 0)
        midi.key_signature_changes.append(pretty_midi.KeySignature(number, 0.0))

    for item in transcriptions:
        instrument = pretty_midi.Instrument(
            program=item.program, is_drum=item.is_drum, name=item.stem
        )
        for note in item.notes:
            instrument.notes.append(
                pretty_midi.Note(
                    velocity=note.velocity,
                    pitch=note.pitch,
                    start=note.start,
                    end=max(note.end, note.start + 0.01),
                )
            )
        midi.instruments.append(instrument)

    midi.write(str(path))
    return path
