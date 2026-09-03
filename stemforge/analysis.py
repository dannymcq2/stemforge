"""Key, tempo and downbeat estimation."""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any

import numpy as np

from .audio import Audio

PITCH_CLASSES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

# Enharmonic spellings that read naturally for each key, so the UI shows
# "Db major" rather than "C# major" and "Eb minor" rather than "D# minor".
_FLAT_MAJOR = {1: "Db", 3: "Eb", 6: "Gb", 8: "Ab", 10: "Bb"}
_FLAT_MINOR = {1: "C#", 3: "Eb", 6: "F#", 8: "G#", 10: "Bb"}

# Camelot wheel positions, indexed by pitch class.
_CAMELOT_MAJOR = {0: 8, 1: 3, 2: 10, 3: 5, 4: 12, 5: 7, 6: 2, 7: 9, 8: 4, 9: 11, 10: 6, 11: 1}
_CAMELOT_MINOR = {0: 5, 1: 12, 2: 7, 3: 2, 4: 9, 5: 4, 6: 11, 7: 6, 8: 1, 9: 8, 10: 3, 11: 10}

# Albrecht & Shanahan (2013) key profiles, fitted on a large corpus. They
# separate a key from its relative major/minor noticeably better than the
# classic Krumhansl-Kessler templates, which is the error that matters most
# here — A minor and C major share all seven notes.
_PROFILE_MAJOR = np.array(
    [0.238, 0.006, 0.111, 0.006, 0.137, 0.094, 0.016, 0.214, 0.009, 0.080, 0.008, 0.081]
)
_PROFILE_MINOR = np.array(
    [0.220, 0.006, 0.104, 0.123, 0.019, 0.103, 0.012, 0.214, 0.062, 0.022, 0.061, 0.052]
)

ANALYSIS_SAMPLE_RATE = 22050
_BASS_CUTOFF_HZ = 260.0



def key_name(tonic: int, mode: str) -> str:
    table = _FLAT_MAJOR if mode == "major" else _FLAT_MINOR
    return table.get(tonic, PITCH_CLASSES[tonic])


def camelot_code(tonic: int, mode: str) -> str:
    if mode == "major":
        return f"{_CAMELOT_MAJOR[tonic]}B"
    return f"{_CAMELOT_MINOR[tonic]}A"


@dataclass
class KeyCandidate:
    tonic: int
    mode: str
    score: float

    @property
    def name(self) -> str:
        return f"{key_name(self.tonic, self.mode)} {self.mode}"

    def as_dict(self) -> dict[str, Any]:
        return {
            "tonic": self.tonic,
            "mode": self.mode,
            "name": self.name,
            "camelot": camelot_code(self.tonic, self.mode),
            "score": round(self.score, 4),
        }


@dataclass
class KeyResult:
    tonic: int
    mode: str
    confidence: float
    alternates: list[KeyCandidate] = field(default_factory=list)

    @property
    def name(self) -> str:
        return f"{key_name(self.tonic, self.mode)} {self.mode}"

    @property
    def camelot(self) -> str:
        return camelot_code(self.tonic, self.mode)

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "tonic": self.tonic,
            "tonic_name": key_name(self.tonic, self.mode),
            "mode": self.mode,
            "camelot": self.camelot,
            "confidence": round(self.confidence, 4),
            "alternates": [c.as_dict() for c in self.alternates],
        }


@dataclass
class TempoResult:
    bpm: float
    confidence: float
    first_beat: float
    beat_times: list[float]
    downbeat_times: list[float]
    beats_per_bar: int
    candidates: list[float] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "bpm": round(self.bpm, 3),
            "confidence": round(self.confidence, 4),
            "first_beat": round(self.first_beat, 4),
            "beats_per_bar": self.beats_per_bar,
            "beat_count": len(self.beat_times),
            "candidates": [round(c, 2) for c in self.candidates],
            "beat_times": [round(t, 4) for t in self.beat_times],
            "downbeat_times": [round(t, 4) for t in self.downbeat_times],
        }


@dataclass
class AnalysisResult:
    key: KeyResult
    tempo: TempoResult
    duration: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "duration": round(self.duration, 3),
            "key": self.key.as_dict(),
            "tempo": self.tempo.as_dict(),
        }


def _weighted_chroma(y: np.ndarray, sr: int) -> np.ndarray:
    """Energy-weighted average CQT chroma of the harmonic component."""
    import librosa

    harmonic = librosa.effects.harmonic(y, margin=3.0)
    chroma = librosa.feature.chroma_cqt(
        y=harmonic, sr=sr, bins_per_octave=36, hop_length=512
    )
    # Frames with more harmonic energy carry more weight, so quiet intros and
    # tails do not drag the profile around.
    weights = np.linalg.norm(chroma, axis=0)
    if weights.sum() <= 0:
        return chroma.mean(axis=1)
    return (chroma * weights).sum(axis=1) / weights.sum()


def _bass_chroma(y: np.ndarray, sr: int) -> np.ndarray:
    """Chroma of the low end only — an approximation of the bass line.

    Root movement carries most of the tonal centre, and it is what separates a
    key from its relative: A minor and C major share a scale but not a root.
    """
    import librosa

    stft = np.abs(librosa.stft(y, n_fft=4096, hop_length=1024))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=4096)
    stft[freqs > _BASS_CUTOFF_HZ, :] = 0.0
    chroma = librosa.feature.chroma_stft(S=stft**2, sr=sr)
    weights = chroma.sum(axis=0)
    if weights.sum() <= 0:
        return chroma.mean(axis=1)
    return (chroma * weights).sum(axis=1) / weights.sum()


def _correlate(profile: np.ndarray, template: np.ndarray, tonic: int) -> float:
    rotated = np.roll(template - template.mean(), tonic)
    centered = profile - profile.mean()
    denom = np.linalg.norm(centered) * np.linalg.norm(rotated)
    return float(np.dot(centered, rotated) / denom) if denom else 0.0


def _normalize(values: np.ndarray) -> np.ndarray:
    total = values.sum()
    return values / total if total > 0 else values


def detect_key(audio: Audio) -> KeyResult:
    """Key detection from full-band chroma, edge chroma and bass root movement.

    Three pieces of evidence are combined: the profile correlation over the
    whole track, the same correlation restricted to the opening and closing
    sections (tracks tend to start and end on the tonic), and how strongly the
    bass sits on each candidate tonic.
    """
    y = audio.resampled(ANALYSIS_SAMPLE_RATE).mono()
    sr = ANALYSIS_SAMPLE_RATE
    if y.size < sr:
        y = np.pad(y, (0, sr - y.size))

    edge = max(int(y.size * 0.15), sr * 2)
    head, tail = y[:edge], y[-edge:]

    global_chroma = _weighted_chroma(y, sr)
    head_chroma = _weighted_chroma(head, sr)
    tail_chroma = _weighted_chroma(tail, sr)
    bass = _normalize(_bass_chroma(y, sr))
    bass_head = _normalize(_bass_chroma(head, sr))

    candidates: list[KeyCandidate] = []
    for mode, template in (("major", _PROFILE_MAJOR), ("minor", _PROFILE_MINOR)):
        for tonic in range(12):
            score = (
                1.00 * _correlate(global_chroma, template, tonic)
                + 0.30 * _correlate(head_chroma, template, tonic)
                + 0.20 * _correlate(tail_chroma, template, tonic)
                # Bass sitting on the tonic, relative to an even spread.
                + 2.50 * (bass[tonic] - 1.0 / 12.0)
                + 1.50 * (bass_head[tonic] - 1.0 / 12.0)
            )
            candidates.append(KeyCandidate(tonic, mode, float(score)))

    candidates.sort(key=lambda c: c.score, reverse=True)
    best, runner_up = candidates[0], candidates[1]
    spread = float(best.score - candidates[-1].score) or 1.0
    confidence = float(np.clip((best.score - runner_up.score) / spread * 3.0, 0.0, 1.0))

    return KeyResult(best.tonic, best.mode, confidence, alternates=candidates[1:4])


def _refine_bpm(beat_times: np.ndarray, coarse_bpm: float) -> float:
    """Sharpen a coarse tempo using the median inter-beat interval."""
    if beat_times.size < 4:
        return float(coarse_bpm)
    intervals = np.diff(beat_times)
    # Drop intervals that disagree with the bulk, which are dropped or doubled beats.
    median = float(np.median(intervals))
    if median <= 0:
        return float(coarse_bpm)
    keep = intervals[np.abs(intervals - median) < 0.25 * median]
    if keep.size < 3:
        return float(coarse_bpm)
    return float(60.0 / keep.mean())


def _fold_into_range(bpm: float, low: float, high: float) -> float:
    """Halve or double a tempo until it lands in the preferred range."""
    if bpm <= 0:
        return bpm
    while bpm < low:
        bpm *= 2.0
    while bpm > high:
        bpm /= 2.0
    return bpm


def _beat_phase(
    y: np.ndarray,
    sr: int,
    onset_env: np.ndarray,
    beat_frames: np.ndarray,
    beats_per_bar: int,
    hop_length: int,
) -> int:
    """Which beat of the bar the first tracked beat is — i.e. the bar phase.

    Two cues vote. Chord changes overwhelmingly land on bar lines, so chroma
    flux peaks there; and the low end (kick) is usually heaviest on beat one.
    """
    import librosa

    if beat_frames.size < beats_per_bar * 2:
        return 0

    chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=hop_length)
    flux = np.concatenate([[0.0], np.linalg.norm(np.diff(chroma, axis=1), axis=0)])

    stft = np.abs(librosa.stft(y, n_fft=2048, hop_length=hop_length))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)
    low = stft[freqs <= 150.0]
    low_env = librosa.onset.onset_strength(
        S=librosa.amplitude_to_db(low, ref=np.max), sr=sr, hop_length=hop_length
    )

    def at_beats(curve: np.ndarray) -> np.ndarray:
        idx = np.clip(beat_frames, 0, curve.size - 1)
        return curve[idx]

    scores = []
    for phase in range(beats_per_bar):
        parts = []
        for curve in (flux, low_env, onset_env):
            values = at_beats(curve)
            downbeats = values[phase::beats_per_bar]
            if downbeats.size == 0 or values.std() == 0:
                parts.append(0.0)
            else:
                parts.append(float((downbeats.mean() - values.mean()) / values.std()))
        # Chroma flux is the most reliable of the three, so it is weighted up.
        scores.append(2.0 * parts[0] + 1.5 * parts[1] + 0.5 * parts[2])

    return int(np.argmax(scores))


def detect_tempo(
    audio: Audio,
    beats_per_bar: int = 4,
    bpm_range: tuple[float, float] = (70.0, 180.0),
) -> TempoResult:
    """Beat-tracked tempo, refined to a fractional BPM, with beat and bar grids."""
    import librosa

    sr = ANALYSIS_SAMPLE_RATE
    y = audio.resampled(sr).mono()
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=512, aggregate=np.median)

    candidates = librosa.feature.tempo(
        onset_envelope=onset_env, sr=sr, hop_length=512, aggregate=None
    )
    candidates = np.atleast_1d(candidates)
    coarse = float(np.median(candidates))

    coarse_bpm, beat_frames = librosa.beat.beat_track(
        onset_envelope=onset_env, sr=sr, hop_length=512, start_bpm=coarse, units="frames"
    )
    coarse_bpm = float(np.atleast_1d(coarse_bpm)[0])
    beat_times = librosa.frames_to_time(beat_frames, sr=sr, hop_length=512)

    bpm = _fold_into_range(_refine_bpm(beat_times, coarse_bpm), *bpm_range)

    # Confidence: how tightly the beats agree with a constant-tempo grid.
    if beat_times.size >= 4:
        intervals = np.diff(beat_times)
        jitter = float(np.std(intervals) / max(np.mean(intervals), 1e-9))
        confidence = float(np.clip(1.0 - jitter * 4.0, 0.0, 1.0))
    else:
        confidence = 0.0

    phase = _beat_phase(
        y, sr, onset_env, np.asarray(beat_frames), beats_per_bar, hop_length=512
    )
    downbeats = beat_times[phase::beats_per_bar] if beat_times.size else np.array([])

    # Beat tracking often skips the opening beat or two. Walk the bar grid back
    # toward zero so the reported downbeat is the first one in the file, which
    # is what a DAW needs to line bar 1 up with the music.
    if downbeats.size:
        bar = 60.0 / bpm * beats_per_bar
        first_beat = float(downbeats[0])
        while first_beat - bar >= -0.02:
            first_beat -= bar
        first_beat = max(first_beat, 0.0)
        leading = np.arange(first_beat, downbeats[0] - bar * 0.5, bar)
        downbeats = np.concatenate([leading, downbeats])
    else:
        first_beat = float(beat_times[0]) if beat_times.size else 0.0

    unique = sorted({round(_fold_into_range(float(c), *bpm_range), 2) for c in candidates})

    return TempoResult(
        bpm=bpm,
        confidence=confidence,
        first_beat=first_beat,
        beat_times=[float(t) for t in beat_times],
        downbeat_times=[float(t) for t in downbeats],
        beats_per_bar=beats_per_bar,
        candidates=unique[:5],
    )


def analyze(audio: Audio, beats_per_bar: int = 4) -> AnalysisResult:
    """Full key + tempo analysis of one audio buffer."""
    return AnalysisResult(
        key=detect_key(audio),
        tempo=detect_tempo(audio, beats_per_bar=beats_per_bar),
        duration=audio.duration,
    )
