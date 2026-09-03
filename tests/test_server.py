import pytest
from fastapi.testclient import TestClient

from stemforge.server import app

client = TestClient(app)


def test_config_lists_models_and_a_device():
    body = client.get("/api/config").json()
    assert "htdemucs" in body["models"]
    assert body["device"] in {"cpu", "mps", "cuda"}
    assert set(body["daw"]) == {"logic", "garageband"}
    assert any(info["default"] for info in body["models"].values())


def test_index_page_is_served():
    response = client.get("/")
    assert response.status_code == 200
    assert "StemForge" in response.text


def test_submitting_a_missing_file_is_rejected():
    response = client.post("/api/jobs", json={"paths": ["/no/such/file.wav"]})
    assert response.status_code == 400
    assert "not found" in response.json()["detail"].lower()


def test_submitting_nothing_is_rejected():
    assert client.post("/api/jobs", json={"paths": []}).status_code == 400


def test_unknown_job_is_a_404():
    assert client.get("/api/jobs/deadbeef").status_code == 404


def test_upload_stages_the_file_and_returns_its_path(tmp_path):
    source = tmp_path / "my track.wav"
    source.write_bytes(b"RIFF____WAVE")
    with source.open("rb") as handle:
        body = client.post("/api/upload", files={"files": ("my track.wav", handle)}).json()

    staged = body["paths"][0]
    assert staged.endswith("my track.wav")
    assert open(staged, "rb").read() == b"RIFF____WAVE"


def test_file_endpoint_refuses_paths_outside_job_folders():
    response = client.get("/api/file", params={"path": "/etc/hosts"})
    assert response.status_code == 403


@pytest.mark.parametrize("field,value", [("shifts", 3), ("quantize", 4), ("model", "htdemucs_6s")])
def test_job_request_accepts_the_tuning_fields(field, value, tmp_path):
    # A bad field name would be a 422 from validation, not a 400 about the file.
    payload = {"paths": [str(tmp_path / "absent.wav")], field: value}
    assert client.post("/api/jobs", json=payload).status_code == 400
