"""FastAPI service: status, background prediction jobs persisted on disk, saved reports, dashboard."""
from __future__ import annotations

import json
import os
import tempfile
import uuid
from pathlib import Path
from typing import Annotated

from fastapi import BackgroundTasks, FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = Path(os.getenv("SOUNDTRACE_JOB_DIR", PROJECT_ROOT / "artifacts" / "jobs"))
MAX_BYTES = int(os.getenv("SOUNDTRACE_MAX_UPLOAD_MB", "200")) * 1024 * 1024
VIDEO_TYPES = {".mp4", ".mov", ".avi", ".mkv", ".webm"}

app = FastAPI(title="SoundTrace", version="0.2.0")
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
                   allow_methods=["*"], allow_headers=["*"])


def project_path(env: str, default: str) -> Path:
    path = Path(os.getenv(env) or default)
    return path if path.is_absolute() else PROJECT_ROOT / path


def reports_dir() -> Path:
    return PROJECT_ROOT / "artifacts" / "reports"


def job_path(job_id: str) -> Path:
    if not job_id.isalnum():  # ids never reach the filesystem unless alphanumeric
        raise HTTPException(404, "Prediction not found")
    return DATA_DIR / f"{job_id}.json"


def write_job(job_id: str, value: dict) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    tmp = job_path(job_id).with_suffix(".tmp")
    tmp.write_text(json.dumps(value), encoding="utf-8")
    tmp.replace(job_path(job_id))


def read_job(job_id: str) -> dict:
    if not job_path(job_id).is_file():
        raise HTTPException(404, "Prediction not found")
    return json.loads(job_path(job_id).read_text(encoding="utf-8"))


def read_report(report_id: str) -> dict:
    path = reports_dir() / f"{report_id}.json"
    if not report_id.isalnum() or not path.is_file():
        raise HTTPException(404, "Evaluation not found")
    return json.loads(path.read_text(encoding="utf-8"))


@app.get("/api/v1/health")
def health():
    return {"status": "ok", "service": "soundtrace"}


@app.get("/api/v1/dataset/status")
def dataset_status():
    from .data import read_manifest, validate_manifest
    root = project_path("SOUNDTRACE_DATA_DIR", "data/raw")
    manifest = project_path("SOUNDTRACE_MANIFEST", "data/manifests/avatar.jsonl")
    rows = read_manifest(manifest)
    valid = bool(rows) and validate_manifest(rows, manifest)["valid"]
    names = {str(r["dataset"]) for r in rows if r.get("dataset")}
    videos = len({str(r.get("video_id")) for r in rows})
    message = ("Validated manifest ready for experiments." if valid else
               "Dataset files detected; build and validate a manifest." if root.exists() else
               "Dataset not configured. See docs/RESEARCH.md.")
    return {"available": root.exists(), "manifest": manifest.name if rows else None, "manifest_valid": valid,
            "manifest_records": len(rows), "manifest_video_count": videos, "video_count": videos if valid else 0,
            "dataset": names.pop() if len(names) == 1 else None, "message": message}


@app.get("/api/v1/model/status")
def model_status():
    path = project_path("SOUNDTRACE_CHECKPOINT", "artifacts/checkpoints/best.pt")
    if not path.is_file():
        return {"available": False, "checkpoint": None, "message": "No trained checkpoint configured."}
    info = {"available": True, "checkpoint": path.name, "message": "Trained checkpoint ready."}
    try:
        import torch
        state = torch.load(path, map_location="cpu", weights_only=True)
        cfg = state.get("config", {})
        info |= {"variant": "gated" if cfg.get("gated") else "baseline", "val_iou": state.get("val_iou"),
                 "epoch": state.get("epoch"), "image_size": cfg.get("image_size"),
                 "pretrained_visual": bool(cfg.get("pretrained"))}
    except Exception:  # noqa: BLE001 - status stays available even if metadata can't be read
        pass
    return info


def process_upload(job_id: str, path: Path) -> None:
    try:
        from .model import load_checkpoint, predict_video
        checkpoint = project_path("SOUNDTRACE_CHECKPOINT", "artifacts/checkpoints/best.pt")
        if not checkpoint.is_file():
            raise ValueError("No trained checkpoint configured; refusing random-weight prediction.")
        model, state = load_checkpoint(checkpoint)
        progress = lambda p: write_job(job_id, {"id": job_id, "status": "processing", "progress": p})
        result = predict_video(model, state["config"], path, float(os.getenv("SOUNDTRACE_INFERENCE_FPS", "2")), progress)
        write_job(job_id, {"id": job_id, "status": "completed", "trained": True, "checkpoint": checkpoint.name, **result})
    except Exception as exc:  # noqa: BLE001 - failures are persisted for the client
        write_job(job_id, {"id": job_id, "status": "failed", "error": str(exc)})
    finally:
        path.unlink(missing_ok=True)  # uploads are never retained


@app.post("/api/v1/predictions", status_code=202)
async def create_prediction(background: BackgroundTasks, file: Annotated[UploadFile, File()]):
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in VIDEO_TYPES:
        raise HTTPException(415, "Unsupported video format")
    job_id = uuid.uuid4().hex
    path = Path(tempfile.gettempdir()) / f"soundtrace-{job_id}{suffix}"
    size = 0
    try:
        with path.open("wb") as out:
            while chunk := await file.read(1 << 20):
                size += len(chunk)
                if size > MAX_BYTES:
                    raise HTTPException(413, "Upload exceeds configured size limit")
                out.write(chunk)
    except Exception:
        path.unlink(missing_ok=True)
        raise
    write_job(job_id, {"id": job_id, "status": "queued"})
    background.add_task(process_upload, job_id, path)
    return {"id": job_id, "status": "queued"}


@app.get("/api/v1/predictions/{prediction_id}")
def get_prediction(prediction_id: str):
    job = read_job(prediction_id)
    meta = {k: v for k, v in job.items() if k != "frames"}
    return meta | ({"frame_count": len(job["frames"])} if "frames" in job else {})


@app.get("/api/v1/predictions/{prediction_id}/masks")
def get_masks(prediction_id: str):
    job = read_job(prediction_id)
    if "frames" not in job:
        raise HTTPException(409, "Prediction masks are not ready")
    return {k: job[k] for k in ("frames", "size", "fps", "duration")}


@app.get("/api/v1/predictions/{prediction_id}/metadata")
def get_metadata(prediction_id: str):
    job = read_job(prediction_id)
    if "frames" in job:
        job["frames"] = [{k: v for k, v in f.items() if k != "mask"} for f in job["frames"]]
    return job


@app.get("/api/v1/evaluations")
@app.get("/api/v1/experiments")
def evaluations():
    """Report summaries, newest first (large per-example lists are left out)."""
    files = sorted(reports_dir().glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True) if reports_dir().exists() else []
    drop = {"results", "test_examples", "history"}
    return [{k: v for k, v in json.loads(p.read_text(encoding="utf-8")).items() if k not in drop} for p in files]


@app.get("/api/v1/evaluations/{evaluation_id}")
@app.get("/api/v1/experiments/{evaluation_id}")
def evaluation(evaluation_id: str):
    return read_report(evaluation_id)


@app.post("/api/v1/experiments/{experiment_id}/compare")
def compare_experiment(experiment_id: str):
    """Find the other variant trained on the same manifest, seed and source-video split."""
    one = read_report(experiment_id)
    keys = ("kind", "manifest_sha256", "seed", "split_fingerprint")
    for other in evaluations():
        if (one.get("kind") == "real_data_experiment" and other.get("variant") not in (None, one.get("variant"))
                and all(other.get(k) == one.get(k) for k in keys)):
            return {"experiment": one, "comparison": other, "comparison_basis": "same manifest, seed, and source-video split",
                    "test_mean_iou_delta": one["test_mean_iou"] - other["test_mean_iou"]}
    raise HTTPException(409, "A completed baseline and gated experiment are both required for comparison")


if (PROJECT_ROOT / "frontend" / "dist").is_dir():  # serve the built dashboard on the same origin
    app.mount("/", StaticFiles(directory=PROJECT_ROOT / "frontend" / "dist", html=True), name="dashboard")
