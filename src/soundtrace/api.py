"""Local FastAPI endpoints with disk-backed prediction job state."""
from __future__ import annotations

import json
import os
import tempfile
import uuid
from pathlib import Path
from threading import Lock
from typing import Annotated

from fastapi import BackgroundTasks, FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="SoundTrace", version="0.1.0")
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"], allow_methods=["*"], allow_headers=["*"])
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = Path(os.getenv("SOUNDTRACE_JOB_DIR", PROJECT_ROOT / "artifacts" / "jobs"))
DATA_DIR.mkdir(parents=True, exist_ok=True)
lock = Lock()
MAX_BYTES = int(os.getenv("SOUNDTRACE_MAX_UPLOAD_MB", "200")) * 1024 * 1024


def configured_path(value: str, default: str) -> Path:
    path = Path(value or default)
    return path if path.is_absolute() else PROJECT_ROOT / path


def configured_checkpoint() -> Path:
    return configured_path(os.getenv("SOUNDTRACE_CHECKPOINT", ""), "artifacts/checkpoints/best.pt")


def job_path(job_id: str) -> Path:
    if not job_id.isalnum():
        raise HTTPException(404, "Prediction not found")
    return DATA_DIR / f"{job_id}.json"


def write_job(job_id: str, value: dict) -> None:
    target = job_path(job_id)
    temp = target.with_suffix(".tmp")
    temp.write_text(json.dumps(value), encoding="utf-8")
    temp.replace(target)


def read_job(job_id: str) -> dict:
    target = job_path(job_id)
    if not target.is_file():
        raise HTTPException(404, "Prediction not found")
    return json.loads(target.read_text(encoding="utf-8"))


@app.get("/api/v1/health")
def health():
    return {"status": "ok", "service": "soundtrace"}


@app.get("/api/v1/dataset/status")
def dataset_status():
    root = configured_path(os.getenv("SOUNDTRACE_DATA_DIR", ""), "data/raw")
    manifest = configured_path(os.getenv("SOUNDTRACE_MANIFEST", ""), "data/manifests/avatar.jsonl")
    manifest_valid = False
    manifest_records = 0
    manifest_video_count = 0
    dataset_name = None
    manifest_errors = 0
    if manifest.is_file():
        try:
            from .data import read_manifest, validate_manifest

            rows = read_manifest(manifest)
            validation = validate_manifest(rows, manifest)
            manifest_valid = validation["valid"]
            manifest_records = len(rows)
            manifest_video_count = len({str(row["video_id"]) for row in rows if "video_id" in row})
            dataset_names = {str(row["dataset"]) for row in rows if row.get("dataset")}
            dataset_name = next(iter(dataset_names)) if len(dataset_names) == 1 else None
            manifest_errors = len(validation["errors"])
        except (OSError, ValueError, KeyError, TypeError):
            manifest_errors = 1
    if manifest_valid:
        message = f"Validated {dataset_name or 'dataset'} manifest ready for experiments."
    elif root.exists():
        message = "Dataset files detected; review and validate a manifest before training."
    else:
        message = "Dataset not configured. Follow docs/DATASET_TERMS.md."
    count = manifest_video_count if manifest_valid else 0
    return {"available": root.exists(), "video_count": count,
            "manifest": manifest.name if manifest.is_file() else None,
            "manifest_valid": manifest_valid, "manifest_records": manifest_records,
            "manifest_video_count": manifest_video_count, "manifest_errors": manifest_errors,
            "dataset": dataset_name, "message": message}


@app.get("/api/v1/model/status")
def model_status():
    path = configured_checkpoint()
    ready = path.is_file()
    return {"available": ready, "checkpoint": path.name if ready else None,
            "message": "Trained checkpoint ready." if ready else "No trained checkpoint configured."}


def process_upload(job_id: str, path: Path) -> None:
    try:
        import cv2
        import numpy as np
        import torch
        from PIL import Image

        from .data import audio_window, decode_audio
        from .model import build_model

        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise ValueError("Video could not be decoded.")
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if fps <= 0 or frame_count <= 0:
            cap.release()
            raise ValueError("Video duration or frame rate could not be read.")
        checkpoint = configured_checkpoint()
        if not checkpoint.is_file():
            cap.release()
            raise ValueError("No trained checkpoint configured; refusing random-weight prediction.")
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        config = state["config"]
        model = build_model(config)
        model.load_state_dict(state["model"])
        model.eval()
        sample_rate = int(config.get("audio_sample_rate", 16000))
        waveform = decode_audio(path, sample_rate)
        segment_seconds = float(config.get("segment_seconds", 1.0))
        size = int(config.get("image_size", 128))
        inference_fps = float(os.getenv("SOUNDTRACE_INFERENCE_FPS", "2"))
        if inference_fps <= 0:
            raise ValueError("SOUNDTRACE_INFERENCE_FPS must be greater than zero")
        stride = max(1, round(fps / inference_fps))
        frame_ids = list(range(0, frame_count, stride))
        if frame_ids[-1] != frame_count - 1:
            frame_ids.append(frame_count - 1)
        masks = []
        with torch.no_grad():
            for index, frame_id in enumerate(frame_ids):
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
                ok, frame = cap.read()
                if not ok:
                    continue
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = Image.fromarray(rgb).resize((size, size))
                tensor = torch.from_numpy(np.asarray(image).copy()).permute(2, 0, 1).float().div(255)[None]
                timestamp = frame_id / fps
                audio = audio_window(waveform, timestamp, sample_rate, segment_seconds)[None]
                output = model(tensor, audio)
                mask = (output["logits"].sigmoid()[0, 0] > 0.5).to(torch.uint8).cpu().numpy()
                masks.append({"frame": frame_id, "timestamp": timestamp, "mask": mask.tolist()})
                if index % 10 == 0:
                    write_job(job_id, {"id": job_id, "status": "processing", "progress": (index + 1) / len(frame_ids), "frames_done": index + 1, "frames_total": len(frame_ids)})
        cap.release()
        if not masks:
            raise ValueError("No decodable video frames were found.")
        write_job(job_id, {"id": job_id, "status": "completed", "trained": True,
                           "fps": fps, "duration": frame_count / fps, "sampled_fps": fps / stride,
                           "size": [size, size], "frames": masks})
    except Exception as exc:  # noqa: BLE001 - background failures are persisted for the client.
        write_job(job_id, {"id": job_id, "status": "failed", "error": str(exc)})
    finally:
        path.unlink(missing_ok=True)


@app.post("/api/v1/predictions", status_code=202)
async def predict(background: BackgroundTasks, file: Annotated[UploadFile, File()]):
    suffix = Path(file.filename or "upload.mp4").suffix.lower()
    if suffix not in {".mp4", ".mov", ".avi", ".mkv", ".webm"}:
        raise HTTPException(415, "Unsupported video format")
    job_id = uuid.uuid4().hex
    path = Path(tempfile.gettempdir()) / f"soundtrace-{job_id}{suffix}"
    size = 0
    try:
        with path.open("wb") as out:
            while chunk := await file.read(1024 * 1024):
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
    result = read_job(prediction_id)
    if "frames" in result:
        return {key: value for key, value in result.items() if key != "frames"} | {"frame_count": len(result["frames"])}
    return result


@app.get("/api/v1/predictions/{prediction_id}/masks")
def get_masks(prediction_id: str):
    result = read_job(prediction_id)
    if "frames" not in result:
        raise HTTPException(409, "Prediction masks are not ready")
    return {"frames": result["frames"], "size": result["size"], "fps": result["fps"], "duration": result["duration"]}


@app.get("/api/v1/predictions/{prediction_id}/metadata")
def get_metadata(prediction_id: str):
    result = read_job(prediction_id)
    return {key: value for key, value in result.items() if key != "frames"}


@app.get("/api/v1/evaluations")
def evaluations():
    folder = PROJECT_ROOT / "artifacts" / "reports"
    if not folder.exists():
        return []
    files = sorted(folder.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    return [json.loads(p.read_text(encoding="utf-8")) for p in files]


@app.get("/api/v1/experiments")
def experiments():
    return evaluations()


@app.get("/api/v1/evaluations/{evaluation_id}")
def evaluation(evaluation_id: str):
    report = PROJECT_ROOT / "artifacts" / "reports" / f"{evaluation_id}.json"
    if not report.is_file():
        raise HTTPException(404, "Evaluation not found")
    return json.loads(report.read_text(encoding="utf-8"))


@app.get("/api/v1/experiments/{experiment_id}")
def experiment(experiment_id: str):
    return evaluation(experiment_id)


@app.post("/api/v1/experiments/{experiment_id}/compare")
def compare_experiment(experiment_id: str):
    folder = PROJECT_ROOT / "artifacts" / "reports"
    current = folder / f"{experiment_id}.json"
    if not current.is_file():
        raise HTTPException(404, "Experiment not found")
    one = json.loads(current.read_text(encoding="utf-8"))
    candidates = [json.loads(p.read_text(encoding="utf-8")) for p in folder.glob("*.json") if p != current]
    other = next((x for x in candidates
                  if x.get("kind") == "real_data_experiment" and one.get("kind") == "real_data_experiment"
                  and x.get("variant") and x.get("variant") != one.get("variant")
                  and x.get("manifest_sha256") == one.get("manifest_sha256") and x.get("seed") == one.get("seed")
                  and x.get("split_fingerprint") == one.get("split_fingerprint")
                  and x.get("split_video_counts") == one.get("split_video_counts")
                  and x.get("test_mean_iou") is not None and one.get("test_mean_iou") is not None), None)
    if other is None:
        raise HTTPException(409, "A completed baseline and gated experiment are both required for comparison")
    return {"experiment": one, "comparison": other,
            "test_mean_iou_delta": one["test_mean_iou"] - other["test_mean_iou"],
            "comparison_basis": "same manifest, seed, and source-video split"}
