"""Local FastAPI endpoints; in-process background tasks are not durable across restarts."""
from __future__ import annotations
import os
import tempfile
import uuid
from pathlib import Path
from threading import Lock
from fastapi import BackgroundTasks, FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="SoundTrace", version="0.1.0")
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"], allow_methods=["*"], allow_headers=["*"])
jobs: dict[str, dict] = {}
lock = Lock()
MAX_BYTES = int(os.getenv("SOUNDTRACE_MAX_UPLOAD_MB", "200")) * 1024 * 1024
PROJECT_ROOT = Path(__file__).resolve().parents[2]

def configured_path(value: str, default: str) -> Path:
    path = Path(value or default)
    return path if path.is_absolute() else PROJECT_ROOT / path

@app.get("/api/v1/health")
def health(): return {"status": "ok", "service": "soundtrace"}

@app.get("/api/v1/dataset/status")
def dataset_status():
    root = configured_path(os.getenv("SOUNDTRACE_DATA_DIR", ""), "data/raw")
    count = sum(1 for p in root.rglob("*") if p.suffix.lower() in {".mp4", ".mov", ".avi", ".mkv"}) if root.exists() else 0
    return {"available": root.exists(), "video_count": count,
            "message": "Dataset path configured; validate annotations before training." if root.exists() else "Dataset not configured. Follow docs/DATASET_TERMS.md."}

@app.get("/api/v1/model/status")
def model_status():
    raw_path = os.getenv("SOUNDTRACE_CHECKPOINT", "")
    path = configured_path(raw_path, "") if raw_path else None
    ready = bool(path and path.is_file())
    return {"available": ready, "checkpoint": path.name if ready else None,
            "message": "Trained checkpoint ready." if ready else "No trained checkpoint configured."}

def process_upload(job_id: str, path: Path):
    try:
        import cv2
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened(): raise ValueError("Video could not be decoded.")
        frames, fps = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), cap.get(cv2.CAP_PROP_FPS) or 0
        timestamp = (frames // 2) / fps if fps > 0 else 0
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, frames // 2))
        ok, frame = cap.read(); cap.release()
        if not ok: raise ValueError("No decodable video frame was found.")
        checkpoint = configured_path(os.getenv("SOUNDTRACE_CHECKPOINT", ""), "") if os.getenv("SOUNDTRACE_CHECKPOINT") else None
        if not checkpoint or not checkpoint.is_file():
            raise ValueError("No trained checkpoint configured; refusing random-weight prediction.")
        import torch
        from PIL import Image
        import numpy as np
        import os as _os
        from .model import build_model
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        model = build_model(state["config"]); model.load_state_dict(state["model"]); model.eval()
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image_size = int(state["config"].get("image_size", 128))
        image = Image.fromarray(rgb).resize((image_size, image_size))
        tensor = torch.from_numpy(np.asarray(image).copy()).permute(2,0,1).float().div(255)[None]
        from .data import decode_audio, audio_window
        sample_rate = int(state["config"].get("audio_sample_rate", 16000))
        waveform = decode_audio(path, sample_rate)
        duration = float(state["config"].get("segment_seconds", 1.0))
        audio = audio_window(waveform, timestamp, sample_rate, duration)[None]
        with torch.no_grad(): pred = model(tensor, audio)
        mask = (pred["logits"].sigmoid()[0,0] > .5).to(torch.uint8).cpu().numpy().tolist()
        with lock: jobs[job_id] = {"id": job_id, "status": "completed", "trained": True, "fps": fps, "duration": frames / fps if fps else None, "timestamp": timestamp, "mask": mask, "size": [image_size,image_size], "gate_mean": float(pred["gate"].mean())}
    except Exception as exc:
        with lock: jobs[job_id] = {"id": job_id, "status": "failed", "error": str(exc)}
    finally:
        path.unlink(missing_ok=True)

@app.post("/api/v1/predictions", status_code=202)
async def predict(background: BackgroundTasks, file: UploadFile = File(...)):
    suffix = Path(file.filename or "upload.mp4").suffix.lower()
    if suffix not in {".mp4", ".mov", ".avi", ".mkv", ".webm"}: raise HTTPException(415, "Unsupported video format")
    job_id = uuid.uuid4().hex
    path = Path(tempfile.gettempdir()) / f"soundtrace-{job_id}{suffix}"
    size = 0
    try:
        with path.open("wb") as out:
            while chunk := await file.read(1024 * 1024):
                size += len(chunk)
                if size > MAX_BYTES: raise HTTPException(413, "Upload exceeds configured size limit")
                out.write(chunk)
    except Exception:
        path.unlink(missing_ok=True); raise
    with lock: jobs[job_id] = {"id": job_id, "status": "queued"}
    background.add_task(process_upload, job_id, path)
    return {"id": job_id, "status": "queued"}

@app.get("/api/v1/predictions/{prediction_id}")
def get_prediction(prediction_id: str):
    with lock: result = jobs.get(prediction_id)
    if not result: raise HTTPException(404, "Prediction not found")
    return result

@app.get("/api/v1/predictions/{prediction_id}/masks")
def get_masks(prediction_id: str):
    result = get_prediction(prediction_id)
    if "mask" not in result: raise HTTPException(409, "Prediction mask is not ready")
    return {"mask": result["mask"], "size": result["size"]}

@app.get("/api/v1/predictions/{prediction_id}/metadata")
def get_metadata(prediction_id: str):
    result = get_prediction(prediction_id)
    return {k:v for k,v in result.items() if k != "mask"}

@app.get("/api/v1/evaluations")
def evaluations():
    folder = Path(__file__).resolve().parents[2] / "artifacts" / "reports"
    if not folder.exists(): return []
    import json
    files = sorted(folder.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    return [json.loads(p.read_text(encoding="utf-8")) for p in files]

@app.get("/api/v1/experiments")
def experiments(): return evaluations()

@app.get("/api/v1/evaluations/{evaluation_id}")
def evaluation(evaluation_id: str):
    report = Path(__file__).resolve().parents[2] / "artifacts" / "reports" / f"{evaluation_id}.json"
    if not report.is_file(): raise HTTPException(404, "Evaluation not found")
    import json
    return json.loads(report.read_text(encoding="utf-8"))

@app.get("/api/v1/experiments/{experiment_id}")
def experiment(experiment_id: str):
    report = Path(__file__).resolve().parents[2] / "artifacts" / "reports" / f"{experiment_id}.json"
    if not report.is_file(): raise HTTPException(404, "Experiment not found")
    import json
    return json.loads(report.read_text(encoding="utf-8"))

@app.post("/api/v1/experiments/{experiment_id}/compare")
def compare_experiment(experiment_id: str):
    import json
    folder = Path(__file__).resolve().parents[2] / "artifacts" / "reports"
    current = folder / f"{experiment_id}.json"
    if not current.is_file(): raise HTTPException(404, "Experiment not found")
    one = json.loads(current.read_text(encoding="utf-8"))
    candidates = [json.loads(p.read_text(encoding="utf-8")) for p in folder.glob("*.json") if p != current]
    other = next((x for x in candidates if x.get("variant") and x.get("variant") != one.get("variant") and x.get("test_mean_iou") is not None), None)
    if other is None: raise HTTPException(409, "A completed baseline and gated experiment are both required for comparison")
    return {"experiment": one, "comparison": other, "test_mean_iou_delta": one["test_mean_iou"] - other["test_mean_iou"]}
