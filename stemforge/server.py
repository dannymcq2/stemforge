"""Local HTTP server backing the desktop UI.

Binds to the loopback interface only. The browser is the window; everything —
file pickers, processing, DAW hand-off — happens in this process on the
user's machine.
"""

from __future__ import annotations

import queue
import re
import shutil
import subprocess
import tempfile
import threading
import traceback
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import daw
from .pipeline import DEFAULT_MIDI_STEMS, JobOptions, run
from .separate import DEFAULT_MODEL, MODELS, default_device

STATIC_DIR = Path(__file__).parent / "static"
DEFAULT_OUTPUT_ROOT = Path.home() / "Music" / "StemForge"
# Dropped files arrive as uploads, because browsers do not reveal local paths.
UPLOAD_DIR = Path(tempfile.gettempdir()) / "stemforge-uploads"


@dataclass
class Job:
    id: str
    source: Path
    output_dir: Path
    options: JobOptions
    status: str = "queued"
    fraction: float = 0.0
    message: str = "Queued"
    result: dict[str, Any] | None = None
    error: str | None = None
    log: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "source": str(self.source),
            "source_name": self.source.name,
            "output_dir": str(self.output_dir),
            "status": self.status,
            "fraction": round(self.fraction, 4),
            "message": self.message,
            "result": self.result,
            "error": self.error,
            "log": self.log[-40:],
        }


class JobStore:
    """Jobs run one at a time — the models want the whole GPU."""

    def __init__(self) -> None:
        self._jobs: dict[str, Job] = {}
        self._queue: queue.Queue[str] = queue.Queue()
        self._lock = threading.Lock()
        self._worker = threading.Thread(target=self._loop, daemon=True)
        self._worker.start()

    def submit(self, job: Job) -> Job:
        with self._lock:
            self._jobs[job.id] = job
        self._queue.put(job.id)
        return job

    def get(self, job_id: str) -> Job:
        with self._lock:
            job = self._jobs.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="No such job")
        return job

    def all(self) -> list[Job]:
        with self._lock:
            return list(self._jobs.values())

    def _loop(self) -> None:
        while True:
            job_id = self._queue.get()
            with self._lock:
                job = self._jobs.get(job_id)
            if job is None:
                continue
            self._process(job)

    def _process(self, job: Job) -> None:
        job.status = "running"

        def progress(fraction: float, message: str) -> None:
            job.fraction = fraction
            if message != job.message:
                job.log.append(message)
            job.message = message

        try:
            result = run(job.source, job.output_dir, job.options, progress)
            job.result = result.as_dict()
            job.status = "done"
            job.fraction = 1.0
            job.message = "Done"
        except Exception as exc:  # noqa: BLE001 - surfaced to the UI
            job.status = "error"
            job.error = str(exc) or exc.__class__.__name__
            job.message = "Failed"
            job.log.append(traceback.format_exc(limit=3))


store = JobStore()
app = FastAPI(title="StemForge")


class JobRequest(BaseModel):
    paths: list[str]
    output_dir: str | None = None
    model: str = DEFAULT_MODEL
    device: str | None = None
    shifts: int = 1
    overlap: float = 0.25
    stems: list[str] | None = None
    midi_stems: list[str] = list(DEFAULT_MIDI_STEMS)
    transcribe_drums: bool = True
    quantize: int = 0
    quantize_strength: float = 1.0
    beats_per_bar: int = 4
    bit_depth: str = "PCM_24"
    include_residual: bool = False
    normalize_stems: bool = False


class OpenRequest(BaseModel):
    target: str  # "logic" | "garageband" | "finder"
    what: str = "stems"  # "stems" | "midi" | "folder"


def _osascript(script: str) -> str:
    if not shutil.which("osascript"):
        raise HTTPException(status_code=501, detail="Native pickers need macOS.")
    proc = subprocess.run(["osascript", "-e", script], capture_output=True, text=True)
    if proc.returncode != 0:
        # A user cancelling the dialog is a normal outcome, not an error.
        if "User canceled" in proc.stderr or "-128" in proc.stderr:
            return ""
        raise HTTPException(status_code=500, detail=proc.stderr.strip())
    return proc.stdout.strip()


@app.get("/api/config")
def get_config() -> dict[str, Any]:
    return {
        "models": {
            name: {**info, "default": name == DEFAULT_MODEL} for name, info in MODELS.items()
        },
        "device": default_device(),
        "default_output": str(DEFAULT_OUTPUT_ROOT),
        "default_midi_stems": list(DEFAULT_MIDI_STEMS),
        "daw": daw.installed_apps(),
        "native_pickers": bool(shutil.which("osascript")),
    }


@app.post("/api/pick/files")
def pick_files() -> dict[str, list[str]]:
    """Open the macOS file picker and return the chosen audio files."""
    out = _osascript(
        'set chosen to choose file with prompt "Choose audio files" '
        "with multiple selections allowed\n"
        "set out to \"\"\n"
        "repeat with f in chosen\n"
        "  set out to out & POSIX path of f & linefeed\n"
        "end repeat\n"
        "return out"
    )
    return {"paths": [line for line in out.splitlines() if line.strip()]}


@app.post("/api/pick/folder")
def pick_folder() -> dict[str, str | None]:
    out = _osascript(
        'POSIX path of (choose folder with prompt "Choose an output folder")'
    )
    return {"path": out or None}


@app.post("/api/upload")
async def upload(files: list[UploadFile] = File(...)) -> dict[str, list[str]]:
    """Stage dropped files on disk and hand back their paths.

    A browser will not tell a page where a dropped file lives, so the bytes
    come over the wire instead. Everything stays on this machine.
    """
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    saved: list[str] = []
    for upload_file in files:
        safe = re.sub(r"[^\w.\- ]", "_", Path(upload_file.filename or "audio").name)
        target = UPLOAD_DIR / f"{uuid.uuid4().hex[:8]}_{safe}"
        with target.open("wb") as handle:
            while chunk := await upload_file.read(1 << 20):
                handle.write(chunk)
        saved.append(str(target))
    return {"paths": saved}


@app.post("/api/jobs")
def create_jobs(request: JobRequest) -> dict[str, Any]:
    if not request.paths:
        raise HTTPException(status_code=400, detail="No files given.")

    root = Path(request.output_dir or DEFAULT_OUTPUT_ROOT).expanduser()
    options = JobOptions(
        model=request.model,
        device=request.device,
        shifts=request.shifts,
        overlap=request.overlap,
        stems=request.stems,
        midi_stems=request.midi_stems,
        transcribe_drums=request.transcribe_drums,
        quantize=request.quantize,
        quantize_strength=request.quantize_strength,
        beats_per_bar=request.beats_per_bar,
        wav_subtype=request.bit_depth,
        include_residual=request.include_residual,
        normalize_stems=request.normalize_stems,
    )

    created = []
    for raw in request.paths:
        source = Path(raw).expanduser()
        if not source.exists():
            raise HTTPException(status_code=400, detail=f"File not found: {raw}")
        job = Job(
            id=uuid.uuid4().hex[:12],
            source=source,
            output_dir=root / source.stem,
            options=options,
        )
        created.append(store.submit(job).as_dict())
    return {"jobs": created}


@app.get("/api/jobs")
def list_jobs() -> dict[str, Any]:
    return {"jobs": [job.as_dict() for job in store.all()]}


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str) -> dict[str, Any]:
    return store.get(job_id).as_dict()


@app.post("/api/jobs/{job_id}/open")
def open_job(job_id: str, request: OpenRequest) -> dict[str, Any]:
    job = store.get(job_id)
    if job.result is None:
        raise HTTPException(status_code=409, detail="Job has not finished.")

    if request.what == "stems":
        paths = [Path(p) for p in job.result["stems"].values()]
    elif request.what == "midi":
        paths = [Path(p) for p in job.result["midi"].values()]
    else:
        paths = [job.output_dir]

    if not paths:
        raise HTTPException(status_code=404, detail="Nothing to open.")

    try:
        if request.target == "finder":
            subprocess.run(["open", str(job.output_dir)], check=True)
        else:
            daw.open_in(request.target, paths)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return {"opened": [str(p) for p in paths]}


@app.get("/api/file")
def get_file(path: str) -> FileResponse:
    """Serve a produced file for in-page playback or download."""
    resolved = Path(path).expanduser().resolve()
    allowed = {job.output_dir.resolve() for job in store.all()}
    if not any(resolved.is_relative_to(root) for root in allowed):
        raise HTTPException(status_code=403, detail="Path is outside any job folder.")
    if not resolved.is_file():
        raise HTTPException(status_code=404, detail="No such file.")
    return FileResponse(resolved, filename=resolved.name)


@app.exception_handler(Exception)
def unhandled(_, exc: Exception) -> JSONResponse:  # pragma: no cover
    return JSONResponse(status_code=500, content={"detail": str(exc)})


app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")


def serve(host: str = "127.0.0.1", port: int = 8787, open_browser: bool = True) -> None:
    import uvicorn

    if open_browser:
        url = f"http://{host}:{port}/"
        threading.Timer(1.2, lambda: subprocess.run(["open", url], check=False)).start()
    uvicorn.run(app, host=host, port=port, log_level="warning")
