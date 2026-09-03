import numpy as np
import pretty_midi
import pytest

from stemforge.analysis import TempoResult
from stemforge.transcribe import (
    DRUM_MAP,
    PRESETS,
    Note,
    Transcription,
    quantize,
    transcribe_drums,
    write_midi,
)
from stemforge import audio as audio_io


def tempo(bpm=120.0, first_beat=0.0) -> TempoResult:
    return TempoResult(
        bpm=bpm, confidence=0.9, first_beat=first_beat,
        beat_times=[], downbeat_times=[], beats_per_bar=4,
    )


def test_quantize_snaps_notes_onto_the_grid():
    # 120 BPM sixteenths are 0.125 s apart.
    notes = [Note(0.13, 0.38, 60, 100), Note(0.26, 0.51, 62, 90)]
    out = quantize(notes, tempo(), subdivision=4)
    assert [round(n.start, 4) for n in out] == [0.125, 0.25]
    assert [round(n.end, 4) for n in out] == [0.375, 0.5]


def test_quantize_respects_the_bar_offset():
    out = quantize([Note(1.04, 1.3, 60, 100)], tempo(first_beat=0.04), subdivision=4)
    assert round(out[0].start, 4) == 1.04


def test_partial_strength_moves_notes_only_part_way():
    out = quantize([Note(0.2, 0.4, 60, 100)], tempo(), subdivision=4, strength=0.5)
    # The grid point is 0.25; half strength lands halfway between.
    assert out[0].start == pytest.approx(0.225)


def test_quantize_never_collapses_a_note_to_zero_length():
    out = quantize([Note(0.124, 0.126, 60, 100)], tempo(), subdivision=4)
    assert out[0].end > out[0].start


def test_quantize_is_a_no_op_when_disabled():
    notes = [Note(0.13, 0.38, 60, 100)]
    assert quantize(notes, tempo(), subdivision=0) is notes


def test_write_midi_records_tempo_and_key(tmp_path):
    item = Transcription("bass", [Note(0.0, 0.5, 45, 100)], program=33, is_drum=False)
    path = write_midi(tmp_path / "bass.mid", [item], tempo(140.0), key_tonic=9, key_mode="minor")

    midi = pretty_midi.PrettyMIDI(str(path))
    _, tempi = midi.get_tempo_changes()
    assert tempi[0] == pytest.approx(140.0, rel=1e-3)
    # pretty_midi numbers minor keys 12-23, so A minor is 9 + 12.
    assert midi.key_signature_changes[0].key_number == 21
    assert midi.instruments[0].program == 33
    assert midi.instruments[0].name == "bass"


def test_write_midi_marks_drum_tracks(tmp_path):
    drums = Transcription("drums", [Note(0.0, 0.1, 36, 100)], program=0, is_drum=True)
    midi = pretty_midi.PrettyMIDI(str(write_midi(tmp_path / "d.mid", [drums], tempo())))
    assert midi.instruments[0].is_drum


def test_write_midi_combines_several_tracks(tmp_path):
    tracks = [
        Transcription("bass", [Note(0.0, 0.5, 45, 100)], 33, False),
        Transcription("drums", [Note(0.0, 0.1, 36, 100)], 0, True),
    ]
    midi = pretty_midi.PrettyMIDI(str(write_midi(tmp_path / "all.mid", tracks, tempo())))
    assert [i.name for i in midi.instruments] == ["bass", "drums"]


def test_drum_transcription_finds_kick_snare_and_hats(drums_path):
    result = transcribe_drums(audio_io.load(drums_path))
    assert result.is_drum
    found = {n.pitch for n in result.notes}
    assert DRUM_MAP["kick"] in found
    assert DRUM_MAP["snare"] in found
    assert DRUM_MAP["hihat"] in found
    # The fixture writes 16 kicks, 16 snares and 64 eighth-note hi-hats over
    # 8 bars. Hats landing under a snare are the ones most easily missed, since
    # a snare covers the whole spectrum.
    counts = {
        name: len([n for n in result.notes if n.pitch == pitch])
        for name, pitch in DRUM_MAP.items()
    }
    assert 12 <= counts["kick"] <= 20
    assert 12 <= counts["snare"] <= 20
    assert 40 <= counts["hihat"] <= 72


def test_drum_notes_are_ordered_in_time(drums_path):
    notes = transcribe_drums(audio_io.load(drums_path)).notes
    assert [n.start for n in notes] == sorted(n.start for n in notes)


def test_presets_keep_each_instrument_in_its_own_register():
    assert PRESETS["bass"].maximum_frequency < PRESETS["vocals"].maximum_frequency
    assert PRESETS["bass"].minimum_frequency < PRESETS["vocals"].minimum_frequency
    assert all(0 <= p.program <= 127 for p in PRESETS.values())


def test_note_serialises_with_a_readable_name():
    assert Note(0.0, 1.0, 60, 100).as_dict()["name"] == "C4"
