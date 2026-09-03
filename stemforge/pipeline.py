"""End-to-end job: analyse, separate, transcribe, export."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

from . import audio as audio_io
from . import daw, separate, transcribe
from .analysis import AnalysisResult, analyze
from .separate import DEFAULT_MODEL, MODELS

ProgressFn = Callable[[float, str], None]

# Stems worth transcribing by default. `other` is usually a pad/synth soup that
# transcribes poorly, so it is opt-in.
DEFAULT_MIDI_STEMS = ("bass", "vocals", "piano", "guitar")


@dataclass
class JobOptions:
    model: str = DEFAULT_MODEL
    device: str | None = None
    shifts: int = 1
    overlap: float = 0.25
    stems: Sequence[str] | None = None          # None -> every stem the model makes
    midi_stems: Sequence[str] = DEFAULT_MIDI_STEMS
    transcribe_drums: bool = True
    quantize: int = 0                           # subdivisions per beat; 0 disables
    quantize_strength: float = 1.0
    beats_per_bar: int = 4
    wav_subtype: str = "PCM_24"
    include_residual: bool = False
    normalize_stems: bool = False
    daw_export: bool = True

    def as_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "device": self.device,
            "shifts": self.shifts,
            "overlap": self.overlap,
            "stems": list(self.stems) if self.stems else None,
            "midi_stems": list(self.midi_stems),
            "transcribe_drums": self.transcribe_drums,
            "quantize": self.quantize,
            "quantize_strength": self.quantize_strength,
            "beats_per_bar": self.beats_per_bar,
        }


@dataclass
class JobResult:
    source: Path
    output_dir: Path
    analysis: AnalysisResult
    stem_paths: dict[str, Path] = field(default_factory=dict)
    midi_paths: dict[str, Path] = field(default_factory=dict)
    transcriptions: list[transcribe.Transcription] = field(default_factory=list)
    options: JobOptions = field(default_factory=JobOptions)

    def as_dict(self) -> dict[str, Any]:
        return {
            "source": str(self.source),
            "source_name": self.source.name,
            "output_dir": str(self.output_dir),
            "analysis": self.analysis.as_dict(),
            "stems": {k: str(v) for k, v in self.stem_paths.items()},
            "midi": {k: str(v) for k, v in self.midi_paths.items()},
            "transcriptions": [t.as_dict() for t in self.transcriptions],
            "options": self.options.as_dict(),
        }


def _stage(progress: ProgressFn | None, fraction: float, message: str) -> None:
    if progress:
        progress(max(0.0, min(1.0, fraction)), message)


def run(
    source: str | Path,
    output_dir: str | Path,
    options: JobOptions | None = None,
    progress: ProgressFn | None = None,
) -> JobResult:
    """Process one file end to end and write everything under `output_dir`."""
    options = options or JobOptions()
    source = Path(source).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    _stage(progress, 0.02, f"Loading {source.name}")
    mix = audio_io.load(source)

    _stage(progress, 0.06, "Detecting key and tempo")
    analysis = analyze(mix, beats_per_bar=options.beats_per_bar)
    _stage(
        progress,
        0.12,
        f"{analysis.key.name} · {analysis.tempo.bpm:.1f} BPM",
    )

    def sep_progress(fraction: float, message: str) -> None:
        _stage(progress, 0.12 + fraction * 0.55, message)

    result = separate.separate(
        mix,
        model=options.model,
        device=options.device,
        shifts=options.shifts,
        overlap=options.overlap,
        progress=sep_progress,
    )

    if options.normalize_stems:
        result.stems = {
            name: audio_io.peak_normalize(data) for name, data in result.stems.items()
        }

    _stage(progress, 0.70, "Writing stems")
    stems_dir = output_dir / "stems"
    stem_paths = separate.write_stems(
        result,
        stems_dir,
        stems=options.stems,
        subtype=options.wav_subtype,
        include_residual=options.include_residual,
    )

    midi_dir = output_dir / "midi"
    transcriptions: list[transcribe.Transcription] = []
    midi_paths: dict[str, Path] = {}

    wanted_midi = [s for s in options.midi_stems if s in result.stems]
    total_midi = len(wanted_midi) + (1 if options.transcribe_drums and "drums" in result.stems else 0)
    done = 0

    for name in wanted_midi:
        _stage(progress, 0.70 + (done / max(total_midi, 1)) * 0.25, f"Transcribing {name}")
        item = transcribe.transcribe(result.as_audio(name), preset=name)
        if options.quantize:
            item.notes = transcribe.quantize(
                item.notes, analysis.tempo, options.quantize, options.quantize_strength
            )
        transcriptions.append(item)
        midi_paths[name] = transcribe.write_midi(
            midi_dir / f"{name}.mid", [item], analysis.tempo,
            analysis.key.tonic, analysis.key.mode,
        )
        done += 1

    if options.transcribe_drums and "drums" in result.stems:
        _stage(progress, 0.70 + (done / max(total_midi, 1)) * 0.25, "Transcribing drums")
        drums = transcribe.transcribe_drums(result.as_audio("drums"))
        if options.quantize:
            drums.notes = transcribe.quantize(
                drums.notes, analysis.tempo, options.quantize, options.quantize_strength
            )
        transcriptions.append(drums)
        midi_paths["drums"] = transcribe.write_midi(
            midi_dir / "drums.mid", [drums], analysis.tempo,
            analysis.key.tonic, analysis.key.mode,
        )

    if transcriptions:
        midi_paths["all"] = transcribe.write_midi(
            midi_dir / "all_stems.mid", transcriptions, analysis.tempo,
            analysis.key.tonic, analysis.key.mode,
        )

    job = JobResult(
        source=source,
        output_dir=output_dir,
        analysis=analysis,
        stem_paths=stem_paths,
        midi_paths=midi_paths,
        transcriptions=transcriptions,
        options=options,
    )

    _stage(progress, 0.97, "Writing session files")
    (output_dir / "analysis.json").write_text(json.dumps(job.as_dict(), indent=2, default=float))
    if options.daw_export:
        daw.write_session(job)

    _stage(progress, 1.0, "Done")
    return job


def run_many(
    sources: Iterable[str | Path],
    output_root: str | Path,
    options: JobOptions | None = None,
    progress: ProgressFn | None = None,
) -> list[JobResult]:
    """Process several files, each into its own subfolder of `output_root`."""
    sources = [Path(s) for s in sources]
    output_root = Path(output_root)
    results: list[JobResult] = []
    for index, source in enumerate(sources):
        def scoped(fraction: float, message: str, i=index) -> None:
            _stage(progress, (i + fraction) / len(sources), f"[{i + 1}/{len(sources)}] {message}")

        results.append(run(source, output_root / source.stem, options, scoped))
    return results
