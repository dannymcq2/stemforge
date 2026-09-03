"""Hand-off to Logic Pro and GarageBand.

Neither DAW has a documented project format we can write, so the reliable
route is a session folder they both import cleanly: stems that all start at
00:00:00, a tempo-map MIDI file that sets project tempo and key signature on
import, and per-stem MIDI regions.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .pipeline import JobResult

APPS = {
    "logic": "Logic Pro",
    "garageband": "GarageBand",
}


def app_path(app: str) -> Path | None:
    """Where an app is installed, or None if it is not on this machine."""
    name = APPS.get(app, app)
    candidate = Path("/Applications") / f"{name}.app"
    return candidate if candidate.exists() else None


def installed_apps() -> dict[str, bool]:
    return {key: app_path(key) is not None for key in APPS}


def write_tempo_map(job: "JobResult", path: Path) -> Path:
    """A MIDI file carrying only tempo and key signature.

    Dragging this into an empty Logic or GarageBand project (and accepting the
    tempo-import prompt) sets the grid before any audio goes in, so the stems
    line up with the bar ruler instead of floating.
    """
    from .transcribe import write_midi

    return write_midi(
        path, [], job.analysis.tempo, job.analysis.key.tonic, job.analysis.key.mode
    )


def _readme(job: "JobResult") -> str:
    key = job.analysis.key
    tempo = job.analysis.tempo
    stems = ", ".join(sorted(job.stem_paths)) or "none"
    midi = ", ".join(sorted(k for k in job.midi_paths if k != "all")) or "none"
    offset = tempo.first_beat

    lines = [
        f"{job.source.name}",
        "=" * len(job.source.name),
        "",
        f"Key            {key.name}  (Camelot {key.camelot}, confidence {key.confidence:.0%})",
        f"Tempo          {tempo.bpm:.2f} BPM  (confidence {tempo.confidence:.0%})",
        f"Time signature {tempo.beats_per_bar}/4",
        f"First downbeat {offset:.3f} s",
        f"Length         {job.analysis.duration:.1f} s",
        "",
        f"Stems          {stems}",
        f"MIDI           {midi}",
        "",
        "Logic Pro / GarageBand",
        "----------------------",
        "1. New empty project.",
        "2. Drag `tempo_map.mid` onto the arrange area and accept the tempo import.",
        f"   The project is now at {tempo.bpm:.2f} BPM in {key.name}.",
        "3. Drag the whole `stems` folder in. Choose 'Multiple tracks' and place",
        "   every region at bar 1 / position 1 1 1 1.",
        "4. Drag files from `midi` onto empty software-instrument tracks.",
        "",
    ]
    if offset > 0.02:
        lines += [
            f"Note: the first downbeat sits {offset:.3f} s into the file. To have bar 1",
            "      land on the downbeat, nudge every region left by that amount, or set",
            f"      the project start offset to -{offset:.3f} s.",
            "",
        ]
    lines += [
        "Any other DAW",
        "-------------",
        "The stems are 24-bit WAV at the source sample rate and all start at 00:00:00,",
        "so importing them at the timeline origin keeps them phase-aligned with each",
        "other and with the original mix.",
        "",
    ]
    return "\n".join(lines)


def write_session(job: "JobResult") -> Path:
    """Create the DAW hand-off folder inside the job's output directory."""
    session = job.output_dir / "daw"
    session.mkdir(parents=True, exist_ok=True)

    write_tempo_map(job, session / "tempo_map.mid")
    (session / "README.txt").write_text(_readme(job))
    return session


def open_in(app: str, paths: list[Path]) -> None:
    """Open files in Logic Pro or GarageBand via LaunchServices."""
    name = APPS.get(app, app)
    if app_path(app) is None:
        raise FileNotFoundError(f"{name} is not installed in /Applications.")
    if not shutil.which("open"):
        raise RuntimeError("`open` is only available on macOS.")
    subprocess.run(["open", "-a", name, *[str(p) for p in paths]], check=True)


def reveal(path: Path) -> None:
    """Show a file or folder in Finder."""
    if shutil.which("open"):
        subprocess.run(["open", "-R", str(path)], check=False)
