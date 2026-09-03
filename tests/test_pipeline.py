import json

import pytest

from stemforge.pipeline import DEFAULT_MIDI_STEMS, JobOptions
from stemforge.separate import DEFAULT_MODEL, MODELS, SeparationResult
import numpy as np


def test_default_options_are_serialisable():
    payload = json.dumps(JobOptions().as_dict())
    assert json.loads(payload)["model"] == DEFAULT_MODEL


def test_default_midi_stems_all_exist_in_some_model():
    every = {stem for info in MODELS.values() for stem in info["stems"]}
    assert set(DEFAULT_MIDI_STEMS) <= every


def test_drums_are_not_in_the_pitched_midi_defaults():
    # Drums go through the percussion path, not Basic Pitch.
    assert "drums" not in DEFAULT_MIDI_STEMS


def test_residual_sums_the_stems_that_were_not_kept():
    stems = {name: np.full((1, 10), value, dtype=np.float32)
             for name, value in (("a", 1.0), ("b", 2.0), ("c", 4.0))}
    result = SeparationResult(stems=stems, sample_rate=44100, model="test")
    assert np.allclose(result.residual(["a"]), 6.0)
    assert np.allclose(result.residual(["a", "b"]), 4.0)


def test_residual_of_everything_is_silence():
    stems = {"a": np.ones((1, 10), dtype=np.float32)}
    result = SeparationResult(stems=stems, sample_rate=44100, model="test")
    assert np.allclose(result.residual(["a"]), 0.0)


def test_write_stems_names_files_after_the_stems(tmp_path):
    from stemforge.separate import write_stems

    stems = {"bass": np.zeros((2, 4410), dtype=np.float32),
             "drums": np.zeros((2, 4410), dtype=np.float32)}
    result = SeparationResult(stems=stems, sample_rate=44100, model="test")
    paths = write_stems(result, tmp_path, include_residual=True)

    assert paths["bass"].name == "bass.wav"
    assert paths["drums"].name == "drums.wav"
    assert paths["minus"].name == "minus_bass_drums.wav"
    assert all(p.is_file() for p in paths.values())


def test_write_stems_honours_an_explicit_subset(tmp_path):
    from stemforge.separate import write_stems

    stems = {n: np.zeros((2, 4410), dtype=np.float32) for n in ("bass", "drums")}
    result = SeparationResult(stems=stems, sample_rate=44100, model="test")
    paths = write_stems(result, tmp_path, stems=["bass"])
    assert set(paths) == {"bass"}


def test_unknown_model_is_rejected_before_any_work():
    from stemforge.separate import separate
    from stemforge.audio import Audio

    silence = Audio(np.zeros((2, 100), dtype=np.float32), 44100)
    with pytest.raises(ValueError, match="Unknown model"):
        separate(silence, model="not-a-model")


def test_every_model_declares_stems_and_a_label():
    for name, info in MODELS.items():
        assert info["stems"], name
        assert info["label"] and info["notes"], name
