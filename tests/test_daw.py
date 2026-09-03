from pathlib import Path

import pretty_midi
import pytest

from stemforge import daw
from stemforge.analysis import AnalysisResult, KeyResult, TempoResult
from stemforge.pipeline import JobResult


@pytest.fixture
def job(tmp_path) -> JobResult:
    tempo = TempoResult(
        bpm=128.0, confidence=0.9, first_beat=0.25,
        beat_times=[0.25, 0.72], downbeat_times=[0.25], beats_per_bar=4,
    )
    key = KeyResult(tonic=9, mode="minor", confidence=0.8)
    return JobResult(
        source=Path("/music/Track.wav"),
        output_dir=tmp_path,
        analysis=AnalysisResult(key=key, tempo=tempo, duration=180.0),
        stem_paths={"bass": tmp_path / "stems" / "bass.wav"},
        midi_paths={"bass": tmp_path / "midi" / "bass.mid"},
    )


def test_session_folder_has_a_tempo_map_and_instructions(job):
    session = daw.write_session(job)
    assert (session / "tempo_map.mid").is_file()
    assert (session / "README.txt").is_file()


def test_tempo_map_carries_tempo_and_key_but_no_notes(job):
    midi = pretty_midi.PrettyMIDI(str(daw.write_tempo_map(job, job.output_dir / "t.mid")))
    _, tempi = midi.get_tempo_changes()
    assert tempi[0] == pytest.approx(128.0, rel=1e-3)
    assert midi.key_signature_changes[0].key_number == 21  # A minor
    assert midi.instruments == []


def test_readme_states_the_key_tempo_and_offset(job):
    text = (daw.write_session(job) / "README.txt").read_text()
    assert "A minor" in text
    assert "8A" in text          # Camelot
    assert "128.00 BPM" in text
    assert "0.250" in text       # first downbeat


def test_readme_warns_when_the_downbeat_is_late(job):
    text = (daw.write_session(job) / "README.txt").read_text()
    assert "nudge every region left" in text


def test_readme_omits_the_offset_warning_when_aligned(job):
    job.analysis.tempo.first_beat = 0.0
    text = (daw.write_session(job) / "README.txt").read_text()
    assert "nudge every region left" not in text


def test_installed_apps_reports_both_daws():
    apps = daw.installed_apps()
    assert set(apps) == {"logic", "garageband"}
    assert all(isinstance(v, bool) for v in apps.values())


def test_opening_a_missing_app_is_an_explicit_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="not installed"):
        daw.open_in("definitely-not-a-daw", [tmp_path])
