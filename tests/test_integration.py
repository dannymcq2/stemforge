"""End-to-end run through Demucs and Basic Pitch.

Slow, and downloads model weights on first use, so it is opt-in:

    pytest -m slow
"""

import json

import pretty_midi
import pytest

from stemforge.pipeline import JobOptions, run

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def job(fixture_path, tmp_path_factory):
    options = JobOptions(
        midi_stems=["bass"], transcribe_drums=True, quantize=4, shifts=1
    )
    return run(fixture_path, tmp_path_factory.mktemp("out"), options)


def test_writes_every_stem_the_model_produces(job):
    assert set(job.stem_paths) == {"drums", "bass", "other", "vocals"}
    assert all(path.is_file() and path.stat().st_size > 1000 for path in job.stem_paths.values())


def test_writes_per_stem_and_combined_midi(job):
    assert {"bass", "drums", "all"} <= set(job.midi_paths)
    assert all(path.is_file() for path in job.midi_paths.values())


def test_bass_midi_follows_the_written_chord_roots(job):
    midi = pretty_midi.PrettyMIDI(str(job.midi_paths["bass"]))
    classes = {note.pitch % 12 for note in midi.instruments[0].notes}
    # The fixture walks A, F, C, G.
    assert {9, 5, 0, 7} <= classes


def test_quantised_notes_land_on_the_sixteenth_grid(job):
    midi = pretty_midi.PrettyMIDI(str(job.midi_paths["bass"]))
    step = 60.0 / job.analysis.tempo.bpm / 4
    origin = job.analysis.tempo.first_beat
    for note in midi.instruments[0].notes:
        offset = (note.start - origin) % step
        assert min(offset, step - offset) < 1e-3


def test_analysis_json_is_written_and_complete(job):
    payload = json.loads((job.output_dir / "analysis.json").read_text())
    assert payload["analysis"]["key"]["name"] == "A minor"
    assert payload["analysis"]["tempo"]["bpm"] == pytest.approx(128.0, abs=1.0)
    assert payload["stems"] and payload["midi"]


def test_daw_session_folder_is_created(job):
    assert (job.output_dir / "daw" / "tempo_map.mid").is_file()
    assert (job.output_dir / "daw" / "README.txt").is_file()
