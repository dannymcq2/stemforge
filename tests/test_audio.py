import numpy as np
import pytest

from stemforge import audio as audio_io


def test_load_reports_shape_and_duration(fixture_path):
    audio = audio_io.load(fixture_path)
    assert audio.sample_rate == 44100
    assert audio.channels == 1
    assert audio.samples.dtype == np.float32
    assert audio.duration == pytest.approx(16.0, abs=0.6)


def test_mono_mixdown_of_stereo():
    stereo = np.stack([np.ones(100), np.full(100, 3.0)]).astype(np.float32)
    audio = audio_io.Audio(stereo, 44100)
    assert audio.channels == 2
    assert np.allclose(audio.mono(), 2.0)


def test_resample_changes_rate_and_length(fixture_path):
    audio = audio_io.load(fixture_path)
    out = audio.resampled(22050)
    assert out.sample_rate == 22050
    assert out.frames == pytest.approx(audio.frames / 2, rel=0.01)
    # Resampling is a no-op when the rate already matches.
    assert out.resampled(22050) is out


def test_round_trip_through_disk(tmp_path, fixture_path):
    audio = audio_io.load(fixture_path)
    target = tmp_path / "out.wav"
    audio_io.write(target, audio.samples, audio.sample_rate)
    back = audio_io.load(target)
    assert back.frames == audio.frames
    assert np.max(np.abs(back.mono() - audio.mono())) < 1e-3


def test_peak_normalize_hits_the_target_headroom():
    quiet = (np.random.default_rng(0).standard_normal(1000) * 0.01).astype(np.float32)
    loud = audio_io.peak_normalize(quiet, headroom_db=-1.0)
    assert np.max(np.abs(loud)) == pytest.approx(10 ** (-1 / 20), rel=1e-4)


def test_peak_normalize_leaves_silence_alone():
    silence = np.zeros(100, dtype=np.float32)
    assert np.array_equal(audio_io.peak_normalize(silence), silence)


def test_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        audio_io.load(tmp_path / "nope.wav")
