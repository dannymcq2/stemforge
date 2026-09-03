"""Stem separation on top of Demucs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import numpy as np

from .audio import Audio

# The models worth exposing, ordered from "best default" downwards.
MODELS: dict[str, dict] = {
    "htdemucs": {
        "label": "Hybrid Transformer (4 stems)",
        "stems": ["drums", "bass", "other", "vocals"],
        "notes": "Fast, well-rounded default.",
    },
    "htdemucs_ft": {
        "label": "Hybrid Transformer fine-tuned (4 stems)",
        "stems": ["drums", "bass", "other", "vocals"],
        "notes": "Highest separation quality; roughly 4x slower than htdemucs.",
    },
    "htdemucs_6s": {
        "label": "Hybrid Transformer (6 stems)",
        "stems": ["drums", "bass", "other", "vocals", "guitar", "piano"],
        "notes": "Adds guitar and piano stems. Piano is the weakest of the six.",
    },
    "mdx_extra": {
        "label": "MDX Extra (4 stems)",
        "stems": ["drums", "bass", "other", "vocals"],
        "notes": "Alternative character; sometimes cleaner on dense mixes.",
    },
}

DEFAULT_MODEL = "htdemucs"


def default_device() -> str:
    """Pick the fastest torch backend available on this machine."""
    import torch

    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


@dataclass
class SeparationResult:
    """Separated stems, keyed by name, all at `sample_rate`."""

    stems: dict[str, np.ndarray]
    sample_rate: int
    model: str

    def as_audio(self, name: str) -> Audio:
        return Audio(self.stems[name], self.sample_rate)

    def residual(self, keep: Iterable[str]) -> np.ndarray:
        """Everything except the named stems, summed — the 'minus' mix."""
        keep = set(keep)
        rest = [s for name, s in self.stems.items() if name not in keep]
        if not rest:
            return np.zeros_like(next(iter(self.stems.values())))
        return np.sum(rest, axis=0)


def separate(
    audio: Audio,
    model: str = DEFAULT_MODEL,
    device: str | None = None,
    shifts: int = 1,
    overlap: float = 0.25,
    progress: Callable[[float, str], None] | None = None,
) -> SeparationResult:
    """Split `audio` into stems.

    `shifts` trades time for accuracy: each extra shift re-runs the model on a
    randomly offset copy and averages, which measurably reduces bleed. `overlap`
    controls how much consecutive analysis windows share.
    """
    import demucs.api
    import torch

    if model not in MODELS:
        raise ValueError(f"Unknown model {model!r}. Choose from: {', '.join(MODELS)}")

    device = device or default_device()

    def on_progress(data: dict) -> None:
        if progress is None:
            return
        total = data.get("audio_length", 0) * data.get("models", 1)
        done = data.get("segment_offset", 0) + data.get("model_idx_in_bag", 0) * data.get(
            "audio_length", 0
        )
        if total:
            progress(min(done / total, 1.0), f"Separating with {model}")

    separator = demucs.api.Separator(
        model=model,
        device=device,
        shifts=max(1, shifts),
        overlap=overlap,
        progress=False,
        callback=on_progress if progress else None,
    )

    wav = torch.from_numpy(np.ascontiguousarray(audio.samples))
    if wav.shape[0] == 1:
        wav = wav.repeat(2, 1)  # Demucs expects stereo in.

    _, stems = separator.separate_tensor(wav, sr=audio.sample_rate)
    out = {name: tensor.cpu().numpy().astype(np.float32) for name, tensor in stems.items()}
    return SeparationResult(stems=out, sample_rate=separator.samplerate, model=model)


def write_stems(
    result: SeparationResult,
    directory: str | Path,
    stems: Iterable[str] | None = None,
    subtype: str = "PCM_24",
    include_residual: bool = False,
) -> dict[str, Path]:
    """Write each stem to `directory` as a WAV; returns name -> path."""
    from .audio import write

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    wanted = list(stems) if stems else list(result.stems)

    paths: dict[str, Path] = {}
    for name in wanted:
        if name not in result.stems:
            continue
        paths[name] = write(
            directory / f"{name}.wav", result.stems[name], result.sample_rate, subtype
        )

    if include_residual and wanted:
        paths["minus"] = write(
            directory / ("minus_" + "_".join(wanted) + ".wav"),
            result.residual(wanted),
            result.sample_rate,
            subtype,
        )
    return paths
