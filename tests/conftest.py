import sys
from pathlib import Path

import pytest
import soundfile as sf

sys.path.insert(0, str(Path(__file__).parent))
from make_fixture import SR, render  # noqa: E402

FIXTURE_BPM = 128.0
FIXTURE_KEY = (9, "minor")  # A minor


@pytest.fixture(scope="session")
def fixture_path(tmp_path_factory) -> Path:
    path = tmp_path_factory.mktemp("audio") / "a_minor_128.wav"
    sf.write(str(path), render(bpm=FIXTURE_BPM), SR, subtype="PCM_16")
    return path


@pytest.fixture(scope="session")
def drums_path(tmp_path_factory) -> Path:
    """Drums on their own — what the drum transcriber sees in the pipeline."""
    path = tmp_path_factory.mktemp("audio") / "drums_128.wav"
    sf.write(str(path), render(bpm=FIXTURE_BPM, pitched=False), SR, subtype="PCM_16")
    return path
