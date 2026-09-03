"""Audio loading helpers shared by the analysis and transcription stages."""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf

# Formats libsndfile reads directly. Anything else takes the ffmpeg detour.
_NATIVE_SUFFIXES = {".wav", ".aif", ".aiff", ".aifc", ".flac", ".ogg", ".oga", ".opus"}


@dataclass(frozen=True)
class Audio:
    """Decoded audio: float32 samples shaped (channels, frames)."""

    samples: np.ndarray
    sample_rate: int

    @property
    def channels(self) -> int:
        return int(self.samples.shape[0])

    @property
    def frames(self) -> int:
        return int(self.samples.shape[1])

    @property
    def duration(self) -> float:
        return self.frames / self.sample_rate

    def mono(self) -> np.ndarray:
        """Mono mixdown as a 1-D float32 array."""
        if self.channels == 1:
            return self.samples[0]
        return self.samples.mean(axis=0)

    def resampled(self, sample_rate: int) -> "Audio":
        if sample_rate == self.sample_rate:
            return self
        import librosa

        out = librosa.resample(
            self.samples, orig_sr=self.sample_rate, target_sr=sample_rate, axis=-1
        )
        return Audio(np.ascontiguousarray(out, dtype=np.float32), sample_rate)


def _decode_with_ffmpeg(path: Path, sample_rate: int | None) -> Audio:
    """Decode via ffmpeg to raw float32, for mp3/m4a/aac/etc."""
    rate_args = ["-ar", str(sample_rate)] if sample_rate else []
    probe = subprocess.run(
        [
            "ffprobe", "-v", "error",
            "-select_streams", "a:0",
            "-show_entries", "stream=sample_rate,channels",
            "-of", "csv=p=0",
            str(path),
        ],
        capture_output=True,
        text=True,
    )
    if probe.returncode != 0:
        raise RuntimeError(f"Could not read audio from {path.name}: {probe.stderr.strip()}")
    src_rate, src_channels = (int(v) for v in probe.stdout.strip().split(",")[:2])

    out_rate = sample_rate or src_rate
    proc = subprocess.run(
        [
            "ffmpeg", "-v", "error", "-i", str(path),
            "-f", "f32le", "-acodec", "pcm_f32le",
            *rate_args, "-",
        ],
        capture_output=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed on {path.name}: {proc.stderr.decode(errors='replace').strip()}")

    flat = np.frombuffer(proc.stdout, dtype=np.float32)
    usable = (flat.size // src_channels) * src_channels
    samples = flat[:usable].reshape(-1, src_channels).T
    return Audio(np.ascontiguousarray(samples, dtype=np.float32), out_rate)


def load(path: str | Path, sample_rate: int | None = None, mono: bool = False) -> Audio:
    """Load `path` as float32 audio, resampling to `sample_rate` when given."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    if path.suffix.lower() in _NATIVE_SUFFIXES:
        data, rate = sf.read(str(path), dtype="float32", always_2d=True)
        audio = Audio(np.ascontiguousarray(data.T), int(rate))
        if sample_rate:
            audio = audio.resampled(sample_rate)
    else:
        audio = _decode_with_ffmpeg(path, sample_rate)

    if mono and audio.channels > 1:
        audio = Audio(audio.mono()[None, :].copy(), audio.sample_rate)
    return audio


def write(path: str | Path, samples: np.ndarray, sample_rate: int, subtype: str = "PCM_24") -> Path:
    """Write (channels, frames) or (frames,) audio to `path`."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = np.asarray(samples, dtype=np.float32)
    if data.ndim == 1:
        data = data[None, :]
    sf.write(str(path), data.T, sample_rate, subtype=subtype)
    return path


def peak_normalize(samples: np.ndarray, headroom_db: float = -1.0) -> np.ndarray:
    """Scale so the loudest sample sits `headroom_db` below full scale."""
    peak = float(np.max(np.abs(samples))) if samples.size else 0.0
    if peak <= 0:
        return samples
    target = 10.0 ** (headroom_db / 20.0)
    return samples * (target / peak)
