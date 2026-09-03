"""Command line interface: `stemforge <command>`."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import __version__


def _progress_printer():
    last = {"message": ""}

    def report(fraction: float, message: str) -> None:
        if message == last["message"]:
            return
        last["message"] = message
        sys.stderr.write(f"\r\033[K[{fraction * 100:5.1f}%] {message}")
        sys.stderr.flush()
        if fraction >= 1.0:
            sys.stderr.write("\n")

    return report


def _cmd_analyze(args: argparse.Namespace) -> int:
    from . import audio as audio_io
    from .analysis import analyze

    result = analyze(audio_io.load(args.input), beats_per_bar=args.beats_per_bar)
    if args.json:
        print(json.dumps(result.as_dict(), indent=2))
    else:
        key, tempo = result.key, result.tempo
        print(f"Key    {key.name}  (Camelot {key.camelot}, {key.confidence:.0%} confident)")
        print(f"Tempo  {tempo.bpm:.2f} BPM  ({tempo.confidence:.0%} confident)")
        print(f"First downbeat  {tempo.first_beat:.3f} s")
        print(f"Duration        {result.duration:.1f} s")
        if key.alternates:
            others = ", ".join(c.name for c in key.alternates)
            print(f"Also considered  {others}")
    return 0


def _cmd_process(args: argparse.Namespace) -> int:
    from .pipeline import JobOptions, run, run_many

    options = JobOptions(
        model=args.model,
        device=args.device,
        shifts=args.shifts,
        overlap=args.overlap,
        stems=args.stems,
        midi_stems=args.midi_stems,
        transcribe_drums=not args.no_drum_midi,
        quantize=args.quantize,
        quantize_strength=args.quantize_strength,
        beats_per_bar=args.beats_per_bar,
        wav_subtype=args.bit_depth,
        include_residual=args.residual,
        normalize_stems=args.normalize,
    )

    progress = None if args.quiet else _progress_printer()
    inputs = [Path(p) for p in args.input]

    if len(inputs) == 1:
        results = [run(inputs[0], args.output or Path.cwd() / inputs[0].stem, options, progress)]
    else:
        results = run_many(inputs, args.output or Path.cwd(), options, progress)

    for job in results:
        key, tempo = job.analysis.key, job.analysis.tempo
        print(f"\n{job.source.name}")
        print(f"  {key.name} · {tempo.bpm:.2f} BPM · Camelot {key.camelot}")
        print(f"  stems: {', '.join(sorted(job.stem_paths))}")
        if job.midi_paths:
            print(f"  midi:  {', '.join(sorted(job.midi_paths))}")
        print(f"  -> {job.output_dir}")

    if args.open_in:
        from . import daw

        job = results[0]
        daw.open_in(args.open_in, sorted(job.stem_paths.values()))
    return 0


def _cmd_serve(args: argparse.Namespace) -> int:
    from .server import serve

    serve(host=args.host, port=args.port, open_browser=not args.no_browser)
    return 0


def _cmd_models(_: argparse.Namespace) -> int:
    from .separate import DEFAULT_MODEL, MODELS, default_device

    print(f"Device: {default_device()}\n")
    for name, info in MODELS.items():
        mark = " (default)" if name == DEFAULT_MODEL else ""
        print(f"{name}{mark}\n  {info['label']}\n  stems: {', '.join(info['stems'])}\n  {info['notes']}\n")
    return 0


def build_parser() -> argparse.ArgumentParser:
    from .separate import DEFAULT_MODEL, MODELS
    from .pipeline import DEFAULT_MIDI_STEMS

    parser = argparse.ArgumentParser(
        prog="stemforge", description="Stem separation, MIDI transcription, key and BPM."
    )
    parser.add_argument("--version", action="version", version=f"stemforge {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    analyze_p = sub.add_parser("analyze", help="Print key, tempo and downbeat only.")
    analyze_p.add_argument("input")
    analyze_p.add_argument("--json", action="store_true")
    analyze_p.add_argument("--beats-per-bar", type=int, default=4)
    analyze_p.set_defaults(func=_cmd_analyze)

    process_p = sub.add_parser("process", help="Separate stems, transcribe MIDI, analyse.")
    process_p.add_argument("input", nargs="+")
    process_p.add_argument("-o", "--output", type=Path, help="Output directory.")
    process_p.add_argument("-m", "--model", default=DEFAULT_MODEL, choices=list(MODELS))
    process_p.add_argument("--device", choices=["cpu", "mps", "cuda"])
    process_p.add_argument(
        "--shifts", type=int, default=1,
        help="Averaged random shifts. 2-5 reduces bleed at a proportional time cost.",
    )
    process_p.add_argument("--overlap", type=float, default=0.25)
    process_p.add_argument("--stems", nargs="*", help="Only write these stems.")
    process_p.add_argument(
        "--midi-stems", nargs="*", default=list(DEFAULT_MIDI_STEMS),
        help="Stems to transcribe to MIDI.",
    )
    process_p.add_argument("--no-drum-midi", action="store_true")
    process_p.add_argument(
        "--quantize", type=int, default=0, metavar="N",
        help="Snap notes to N subdivisions per beat (4 = sixteenths). 0 leaves timing free.",
    )
    process_p.add_argument("--quantize-strength", type=float, default=1.0)
    process_p.add_argument("--beats-per-bar", type=int, default=4)
    process_p.add_argument("--bit-depth", default="PCM_24", choices=["PCM_16", "PCM_24", "FLOAT"])
    process_p.add_argument(
        "--residual", action="store_true", help="Also write the everything-else mix."
    )
    process_p.add_argument("--normalize", action="store_true")
    process_p.add_argument("--open-in", choices=["logic", "garageband"])
    process_p.add_argument("-q", "--quiet", action="store_true")
    process_p.set_defaults(func=_cmd_process)

    serve_p = sub.add_parser("serve", help="Run the desktop app UI.")
    serve_p.add_argument("--host", default="127.0.0.1")
    serve_p.add_argument("--port", type=int, default=8787)
    serve_p.add_argument("--no-browser", action="store_true")
    serve_p.set_defaults(func=_cmd_serve)

    models_p = sub.add_parser("models", help="List separation models and the active device.")
    models_p.set_defaults(func=_cmd_models)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except KeyboardInterrupt:
        sys.stderr.write("\nInterrupted.\n")
        return 130
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        sys.stderr.write(f"\nError: {exc}\n")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
