from __future__ import annotations

import json
import os
import re
import shutil
import threading
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, RedirectResponse, Response

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = Path(os.getenv("PLANETREAD_DATA_DIR", PROJECT_ROOT / "backend_data"))
if not DATA_ROOT.is_absolute():
    DATA_ROOT = PROJECT_ROOT / DATA_ROOT
JOBS_ROOT = DATA_ROOT / "jobs"
MAX_UPLOAD_BYTES = int(os.getenv("PLANETREAD_MAX_UPLOAD_BYTES", str(2 * 1024**3)))
CHUNK_SIZE = 8 * 1024**2
ALLOWED_EXTENSIONS = {".mp4", ".mov", ".webm", ".mkv"}
JOBS_ROOT.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="PlanetRead Processing API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[origin.strip() for origin in os.getenv(
        "PLANETREAD_ALLOWED_ORIGINS",
        "http://localhost:5173,http://127.0.0.1:5173",
    ).split(",")],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="planetread-pipeline")
state_lock = threading.Lock()


@app.get("/", include_in_schema=False)
def frontend_redirect():
    return RedirectResponse(os.getenv("PLANETREAD_FRONTEND_URL", "http://127.0.0.1:5173"))


@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    return Response(status_code=204)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def safe_filename(filename: str) -> str:
    name = Path(filename or "video.mp4").name
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", Path(name).stem).strip("._") or "video"
    return f"{stem}{Path(name).suffix.lower()}"


def job_dir(job_id: str) -> Path:
    if not re.fullmatch(r"[a-f0-9]{32}", job_id):
        raise HTTPException(status_code=404, detail="Job not found")
    directory = JOBS_ROOT / job_id
    if not directory.exists():
        raise HTTPException(status_code=404, detail="Job not found")
    return directory


def read_job(job_id: str) -> dict:
    metadata = job_dir(job_id) / "job.json"
    if not metadata.exists():
        raise HTTPException(status_code=404, detail="Job not found")
    return json.loads(metadata.read_text(encoding="utf-8"))


def write_job(directory: Path, **changes) -> dict:
    with state_lock:
        path = directory / "job.json"
        data = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
        data.update(changes, updated_at=now())
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(data, indent=2), encoding="utf-8")
        temporary.replace(path)
        return data


def public_job(data: dict) -> dict:
    job_id = data["id"]
    response = {key: data.get(key) for key in (
        "id", "filename", "status", "stage", "progress", "error", "created_at", "updated_at",
    )}
    response["status_url"] = f"/api/jobs/{job_id}"
    if data.get("status") == "completed":
        response["result_url"] = f"/api/jobs/{job_id}/result"
    return response


def refresh_progress(directory: Path, data: dict) -> dict:
    if data.get("status") != "processing":
        return data
    output = directory / "output"
    updates = {}
    if (output / "florence_log.jsonl").exists():
        updates = {"stage": "Understanding visual scenes", "progress": 30}
    logs = list(output.glob("*.log"))
    log_text = logs[0].read_text(encoding="utf-8", errors="ignore")[-12000:] if logs else ""
    for marker, stage, value in [
        ("Extracting audio", "Analyzing audio", 42),
        ("Raw events:", "Detecting non-speech moments", 58),
        ("After dedup:", "Grouping detected sounds", 67),
        ("Timeline:", "Generating captions", 76),
        ("SRT:", "Creating annotated frames", 86),
        ("Generating video", "Rendering captioned video", 92),
    ]:
        if marker in log_text:
            updates = {"stage": stage, "progress": value}
    if updates and updates.get("progress", 0) > data.get("progress", 0):
        return write_job(directory, **updates)
    return data


def run_pipeline(job_id: str) -> None:
    directory = JOBS_ROOT / job_id
    upload_path = next((directory / "input").iterdir())
    output_dir = directory / "output"
    output_dir.mkdir(exist_ok=True)
    try:
        write_job(directory, status="processing", stage="Loading ML models", progress=8, started_at=now())
        os.environ.setdefault("MPLCONFIGDIR", str(DATA_ROOT / ".matplotlib"))
        (DATA_ROOT / ".matplotlib").mkdir(exist_ok=True)
        from panns_pipeline import process_video

        write_job(directory, stage="Analyzing video, sound and scenes", progress=15)
        process_video(str(upload_path), output_dir)
        required = [output_dir / "results.json", output_dir / "captions.srt", output_dir / "final_output.mp4"]
        if not all(path.exists() for path in required):
            missing = ", ".join(path.name for path in required if not path.exists())
            raise RuntimeError(f"Pipeline finished without required output: {missing}")
        result_data = json.loads((output_dir / "results.json").read_text(encoding="utf-8"))
        write_job(
            directory, status="completed", stage="Complete", progress=100,
            caption_segments=result_data.get("caption_segments", len(result_data.get("timeline", []))),
            completed_at=now(), error=None,
        )
    except Exception as exc:
        (directory / "backend-error.log").write_text(traceback.format_exc(), encoding="utf-8")
        write_job(directory, status="failed", stage="Processing failed", error=str(exc), completed_at=now())


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok", "pipeline": "panns_pipeline.py", "worker_capacity": 1}


@app.post("/api/jobs", status_code=202)
def create_job(video: UploadFile) -> dict:
    extension = Path(video.filename or "").suffix.lower()
    if extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=415, detail="Upload an MP4, MOV, WEBM or MKV video")
    job_id = uuid.uuid4().hex
    directory = JOBS_ROOT / job_id
    input_dir = directory / "input"
    input_dir.mkdir(parents=True)
    destination = input_dir / safe_filename(video.filename or "video.mp4")
    size = 0
    try:
        with destination.open("wb") as output:
            while chunk := video.file.read(CHUNK_SIZE):
                size += len(chunk)
                if size > MAX_UPLOAD_BYTES:
                    raise HTTPException(status_code=413, detail="Video exceeds the 2 GB upload limit")
                output.write(chunk)
    except Exception:
        shutil.rmtree(directory, ignore_errors=True)
        raise
    finally:
        video.file.close()
    metadata = write_job(
        directory, id=job_id, filename=video.filename, stored_filename=destination.name,
        size_bytes=size, status="queued", stage="Waiting for pipeline", progress=3,
        error=None, created_at=now(),
    )
    executor.submit(run_pipeline, job_id)
    return public_job(metadata)


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str) -> dict:
    directory = job_dir(job_id)
    return public_job(refresh_progress(directory, read_job(job_id)))


@app.get("/api/jobs/{job_id}/result")
def get_result(job_id: str) -> dict:
    data = read_job(job_id)
    if data.get("status") != "completed":
        raise HTTPException(status_code=409, detail="Result is not ready")
    output_dir = job_dir(job_id) / "output"
    result = json.loads((output_dir / "results.json").read_text(encoding="utf-8"))
    timeline = [{**item, "id": i, "frame_url": f"/api/jobs/{job_id}/frames/{i}"} for i, item in enumerate(result.get("timeline", []), 1)]
    return {
        "matched": True, "job_id": job_id,
        "caption_segments": result.get("caption_segments", len(timeline)),
        "duration_sec": max((item["end_sec"] for item in timeline), default=0),
        "video_url": f"/api/jobs/{job_id}/video",
        "srt_url": f"/api/jobs/{job_id}/captions", "timeline": timeline,
    }


@app.get("/api/jobs/{job_id}/video")
def get_video(job_id: str):
    path = job_dir(job_id) / "output" / "final_output.mp4"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Video output not found")
    return FileResponse(path, media_type="video/mp4", filename=f"planetread-{job_id}.mp4", content_disposition_type="inline")


@app.get("/api/jobs/{job_id}/captions")
def get_captions(job_id: str):
    path = job_dir(job_id) / "output" / "captions.srt"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Captions not found")
    return FileResponse(path, media_type="application/x-subrip", filename=f"planetread-{job_id}.srt")


@app.get("/api/jobs/{job_id}/frames/{caption_id}")
def get_frame(job_id: str, caption_id: int):
    output_dir = job_dir(job_id) / "output"
    result_path = output_dir / "results.json"
    if not result_path.exists():
        raise HTTPException(status_code=404, detail="Frame not found")
    timeline = json.loads(result_path.read_text(encoding="utf-8")).get("timeline", [])
    if caption_id < 1 or caption_id > len(timeline):
        raise HTTPException(status_code=404, detail="Frame not found")
    timestamp = timeline[caption_id - 1]["start_sec"]
    frames = list((output_dir / "annotated_frames").glob("frame_*.png"))
    if not frames:
        raise HTTPException(status_code=404, detail="Frame not found")

    def frame_time(path: Path) -> float:
        match = re.search(r"frame_([\d.]+)s", path.name)
        return float(match.group(1)) if match else 0

    return FileResponse(min(frames, key=lambda p: abs(frame_time(p) - timestamp)), media_type="image/png")
