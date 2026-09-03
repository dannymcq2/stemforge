import numpy as np
import pytest

from conftest import FIXTURE_BPM, FIXTURE_KEY
from stemforge import audio as audio_io
from stemforge.analysis import (
    analyze,
    camelot_code,
    detect_key,
    detect_tempo,
    key_name,
    _fold_into_range,
    _refine_bpm,
)


@pytest.fixture(scope="module")
def result(fixture_path):
    return analyze(audio_io.load(fixture_path))


def test_detects_the_written_key(result):
    assert (result.key.tonic, result.key.mode) == FIXTURE_KEY
    assert result.key.name == "A minor"
    assert result.key.camelot == "8A"


def test_detects_the_written_tempo(result):
    assert result.tempo.bpm == pytest.approx(FIXTURE_BPM, abs=1.0)
    assert result.tempo.confidence > 0.5


def test_first_downbeat_is_at_the_start(result):
    # The fixture begins exactly on beat one of bar one.
    assert result.tempo.first_beat < 0.1


def test_beat_grid_is_evenly_spaced(result):
    intervals = np.diff(result.tempo.beat_times)
    assert np.std(intervals) < 0.02
    assert np.mean(intervals) == pytest.approx(60.0 / FIXTURE_BPM, abs=0.02)


def test_result_serialises_to_plain_json_types(result):
    import json

    payload = json.dumps(result.as_dict())
    assert "A minor" in payload
    assert json.loads(payload)["tempo"]["bpm"] == pytest.approx(FIXTURE_BPM, abs=1.0)


def test_silence_does_not_crash_detection():
    silence = audio_io.Audio(np.zeros((1, 44100 * 4), dtype=np.float32), 44100)
    out = analyze(silence)
    assert 0.0 <= out.key.confidence <= 1.0
    assert out.tempo.bpm >= 0.0


@pytest.mark.parametrize(
    "tonic,mode,expected",
    [(0, "major", "C"), (1, "major", "Db"), (3, "minor", "Eb"), (6, "major", "Gb")],
)
def test_key_names_use_readable_spellings(tonic, mode, expected):
    assert key_name(tonic, mode) == expected


def test_camelot_codes_of_relative_keys_share_a_number():
    # C major and A minor are the same wheel position, different letter.
    assert camelot_code(0, "major") == "8B"
    assert camelot_code(9, "minor") == "8A"


@pytest.mark.parametrize(
    "bpm,expected", [(35.0, 70.0), (200.0, 100.0), (128.0, 128.0), (70.0, 70.0)]
)
def test_tempo_octaves_fold_into_the_preferred_range(bpm, expected):
    assert _fold_into_range(bpm, 70.0, 180.0) == pytest.approx(expected)


def test_refine_bpm_ignores_dropped_beats():
    period = 60.0 / 120.0
    beats = np.arange(0, 20) * period
    beats = np.delete(beats, 7)  # a missed beat leaves a double-length gap
    assert _refine_bpm(beats, 118.0) == pytest.approx(120.0, abs=0.5)


def test_refine_bpm_falls_back_when_there_are_too_few_beats():
    assert _refine_bpm(np.array([0.0, 0.5]), 99.0) == pytest.approx(99.0)


def test_half_time_beats_per_bar_changes_the_bar_grid(fixture_path):
    audio = audio_io.load(fixture_path)
    four = detect_tempo(audio, beats_per_bar=4)
    three = detect_tempo(audio, beats_per_bar=3)
    assert three.beats_per_bar == 3
    assert len(three.downbeat_times) > len(four.downbeat_times)


def test_key_alternates_are_ranked_below_the_winner(fixture_path):
    key = detect_key(audio_io.load(fixture_path))
    assert len(key.alternates) == 3
    scores = [c.score for c in key.alternates]
    assert scores == sorted(scores, reverse=True)
