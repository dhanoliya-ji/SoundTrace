"""Manifest, decoding, alignment and synthetic fixture helpers."""
from __future__ import annotations

import csv
import json
import pickle
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
from PIL import Image


def discover(root: Path) -> dict:
    videos = sorted(p for p in root.rglob("*") if p.suffix.lower() in {".mp4", ".mov", ".avi", ".mkv"}) if root.exists() else []
    masks = sorted(p for p in root.rglob("*") if p.suffix.lower() in {".png", ".jpg", ".jpeg"}) if root.exists() else []
    return {"root": str(root), "videos": videos, "masks": masks}


def build_manifest(root: Path, output: Path) -> dict:
    found = discover(root)
    mask_by_stem = {p.stem: p for p in found["masks"]}
    rows, missing = [], []
    for video in found["videos"]:
        mask = mask_by_stem.get(video.stem)
        if mask:
            rows.append({"video": str(video.resolve()), "mask": str(mask.resolve()), "video_id": video.stem,
                         "timestamp": 0.0, "review_required": True})
        else:
            missing.append(video.name)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return {"videos": len(found["videos"]), "masks": len(found["masks"]), "matched": len(rows), "missing_masks": missing, "manifest": str(output)}


def build_avsbench_manifest(root: Path, output: Path, task: str) -> dict:
    """Read official object-task metadata and S4/MS3 frame, mask, and audio paths."""
    if task not in {"s4", "ms3"}:
        raise ValueError("AVSBench object task must be s4 or ms3")
    stem = "s4" if task == "s4" else "ms3"
    subset = "Single-source" if task == "s4" else "Multi-sources"
    csv_name = f"{stem}_meta_data.csv"
    csv_candidates = list(root.rglob(csv_name))
    if len(csv_candidates) != 1:
        raise ValueError(f"Expected one {csv_name} below dataset root; found {len(csv_candidates)}")
    meta = csv_candidates[0]
    rows, errors = [], []
    with meta.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError(f"Metadata CSV has no header: {meta}")
        fields = {name.strip().lower(): name for name in reader.fieldnames}
        aliases = {"video_name": ("video_name", "video_id", "video", "name", "id"),
                   "category": ("category", "class"), "split": ("split", "set")}
        resolved = {}
        for target, choices in aliases.items():
            resolved[target] = next((fields[c] for c in choices if c in fields), None)
        required_columns = ("video_name", "split") if task == "ms3" else ("video_name", "category", "split")
        if any(not resolved[name] for name in required_columns):
            raise ValueError(f"Unexpected {csv_name} columns: {reader.fieldnames}")
        for line_no, row in enumerate(reader, 2):
            video_id = str(row[resolved["video_name"]]).strip()
            category = str(row[resolved["category"]]).strip() if resolved["category"] else ""
            split = str(row[resolved["split"]]).strip().lower()
            split = "val" if split in {"valid", "validation", "val"} else split
            if not video_id or split not in {"train", "val", "test"}:
                errors.append({"line": line_no, "video_id": video_id, "error": f"invalid id or split {split!r}"})
                continue
            if task == "s4":
                base = root / subset / "s4_data"
                frame_dir = base / "visual_frames" / split / category / video_id
                mask_dir = base / "gt_masks" / split / category / video_id
                feature_path = base / "audio_log_mel" / split / category / f"{video_id}.pkl"
            else:
                base = root / subset / "ms3_data"
                frame_dir = base / "visual_frames" / video_id
                mask_dir = base / "gt_masks" / split / video_id
                feature_path = base / "audio_log_mel" / split / f"{video_id}.pkl"
            if not feature_path.is_file():
                errors.append({"line": line_no, "video_id": video_id,
                               "error": f"missing audio feature file: {feature_path.relative_to(root)}"})
                continue
            frame_ids = [1] if task == "s4" and split == "train" else [1, 2, 3, 4, 5]
            for frame_id in frame_ids:
                frame_name = f"{video_id}_{frame_id}.png" if task == "s4" else f"{video_id}.mp4_{frame_id}.png"
                image_path = frame_dir / frame_name
                mask_name = f"{video_id}_{frame_id}.png"
                mask_path = mask_dir / mask_name
                if not image_path.is_file() or not mask_path.is_file():
                    errors.append({"line": line_no, "video_id": video_id, "frame_id": frame_id,
                                   "error": f"expected one visual frame and mask; found {int(image_path.is_file())} frames and {int(mask_path.is_file())} masks"})
                    continue
                rows.append({"video_id": video_id, "frame": str(image_path.resolve()),
                             "mask": str(mask_path.resolve()), "audio_features": str(feature_path.resolve()),
                             "audio_index": frame_id - 1, "timestamp": frame_id - 0.5,
                             "split": split, "task": task, "category": category,
                             "annotation_frame": frame_id})
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return {"task": task, "metadata_csv": str(meta), "records": len(rows), "errors": errors,
            "manifest": str(output), "valid": bool(rows) and not errors}


def read_manifest(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def decode_audio(path: str | Path, sample_rate: int = 16000) -> torch.Tensor:
    """Decode/resample a video's audio with OpenCV-independent FFmpeg subprocess output."""
    import subprocess
    command = ["ffmpeg", "-v", "error", "-i", str(path), "-vn", "-ac", "1", "-ar", str(sample_rate),
               "-f", "f32le", "pipe:1"]
    try:
        result = subprocess.run(command, check=True, capture_output=True)
    except FileNotFoundError as exc:
        raise RuntimeError("FFmpeg is required for audio decoding. Install FFmpeg and add it to PATH.") from exc
    except subprocess.CalledProcessError as exc:
        message = exc.stderr.decode("utf-8", errors="replace")[-500:]
        raise ValueError(f"Audio decode failed for {Path(path).name}: {message}") from exc
    if not result.stdout:
        raise ValueError(f"No decodable audio track found in {Path(path).name}")
    return torch.from_numpy(np.frombuffer(result.stdout, dtype="<f4").copy())


def resolve_path(value: str, manifest_path: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (manifest_path.parent / path).resolve()


def validate_manifest(rows: list[dict], manifest_path: Path) -> dict:
    required = {"mask", "video_id", "timestamp"}
    errors, seen = [], {}
    if not rows:
        errors.append({"line": 0, "error": "manifest contains no records"})
    split_presence = ["split" in row for row in rows]
    if any(split_presence) and not all(split_presence):
        errors.append({"line": 0, "error": "split must be present on every row or on none of them"})
    for index, row in enumerate(rows, 1):
        if row.get("review_required"):
            errors.append({"line": index, "error": "auto-discovered pairing must be manually reviewed before training"})
        absent = required - row.keys()
        if absent:
            errors.append({"line": index, "error": f"missing fields: {sorted(absent)}"})
            continue
        mask = resolve_path(row["mask"], manifest_path)
        if not mask.is_file(): errors.append({"line": index, "error": f"mask missing: {mask.name}"})
        if row.get("video"):
            video = resolve_path(row["video"], manifest_path)
            if not video.is_file(): errors.append({"line": index, "error": f"video missing: {video.name}"})
        if row.get("frame") and not resolve_path(row["frame"], manifest_path).is_file():
            errors.append({"line": index, "error": f"frame missing: {Path(row['frame']).name}"})
        if row.get("audio_features") and not resolve_path(row["audio_features"], manifest_path).is_file():
            errors.append({"line": index, "error": f"audio feature file missing: {Path(row['audio_features']).name}"})
        if not row.get("video") and not (row.get("frame") and row.get("audio_features")):
            errors.append({"line": index, "error": "provide a video path, or both frame and audio_features paths"})
        group = str(row["video_id"])
        split = row.get("split")
        if split:
            if split not in {"train", "val", "test"}:
                errors.append({"line": index, "error": f"invalid split {split!r}; expected train, val, or test"})
            if group in seen and seen[group] != split:
                errors.append({"line": index, "error": f"video leakage: {group} appears in {seen[group]} and {split}"})
            seen[group] = split
    return {"records": len(rows), "valid": not errors, "errors": errors}


def load_record(row: dict, manifest_path: Path, size: int, sample_rate: int, duration: float):
    import cv2
    mask_path = resolve_path(row["mask"], manifest_path)
    if row.get("frame"):
        video = load_image(str(resolve_path(row["frame"], manifest_path)), size)
    else:
        video_path = resolve_path(row["video"], manifest_path)
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened(): raise ValueError(f"Cannot decode video: {video_path.name}")
        cap.set(cv2.CAP_PROP_POS_MSEC, max(0, float(row["timestamp"])) * 1000)
        ok, frame = cap.read(); cap.release()
        if not ok: raise ValueError(f"Cannot decode frame at {row['timestamp']}s in {video_path.name}")
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(frame).resize((size, size))
        arr = np.asarray(image, dtype=np.float32).copy() / 255.0
        video = torch.from_numpy(arr).permute(2, 0, 1)
    mask = load_mask(str(mask_path), size)
    if row.get("audio_features"):
        feature_path = resolve_path(row["audio_features"], manifest_path)
        features = cached_audio_features(str(feature_path))
        feature_index = int(row.get("audio_index", round(float(row["timestamp"]) + 0.5) - 1))
        if features.ndim == 4 and features.shape[0] == 5:
            if not 0 <= feature_index < 5:
                raise ValueError(f"Audio feature index must be between 0 and 4: {feature_index}")
            audio = features[feature_index]
        elif features.ndim == 3:
            audio = features
        else:
            raise ValueError(f"Unexpected AVSBench audio feature shape: {list(features.shape)}")
        if not torch.isfinite(audio).all():
            raise ValueError(f"Non-finite audio features in {feature_path.name}")
    else:
        video_path = resolve_path(row["video"], manifest_path)
        waveform = cached_audio(str(video_path), sample_rate)
        audio = audio_window(waveform, float(row["timestamp"]), sample_rate, duration)
    return video, audio, mask


@lru_cache(maxsize=16)
def cached_audio(path: str, sample_rate: int) -> torch.Tensor:
    return decode_audio(path, sample_rate)


@lru_cache(maxsize=32)
def cached_audio_features(path: str) -> torch.Tensor:
    """Load audio-feature pickles produced by the official AVSBench scripts."""
    with Path(path).open("rb") as handle:
        value = pickle.load(handle)
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    features = torch.as_tensor(value, dtype=torch.float32)
    if features.ndim == 4 and features.shape[0] == 5:
        return features.contiguous()
    if features.ndim == 3:
        return features.contiguous()
    raise ValueError(f"Unsupported audio feature shape in {Path(path).name}: {list(features.shape)}")


def audio_window(waveform: torch.Tensor, timestamp: float, sample_rate: int, duration: float = 1.0) -> torch.Tensor:
    """Centered fixed-size window with zero padding at boundaries."""
    if waveform.ndim == 2:
        waveform = waveform.mean(0)
    n = max(1, round(sample_rate * duration))
    center = round(timestamp * sample_rate)
    start = center - n // 2
    if timestamp <= 0:
        start = 0
    out = waveform.new_zeros(n)
    src_a, src_b = max(0, start), min(waveform.numel(), start + n)
    if src_b > src_a:
        out[src_a - start:src_b - start] = waveform[src_a:src_b]
    return out


def shift_log_mel(feature: torch.Tensor, seconds: float, clip_duration: float = 1.0) -> torch.Tensor:
    """Shift a [channels, time, mel] clip with zero padding, never circular wrap."""
    if feature.ndim != 3:
        raise ValueError(f"Expected log-mel feature [channels, time, mel], got {list(feature.shape)}")
    offset = round(abs(seconds) / clip_duration * feature.shape[-2])
    if offset == 0:
        return feature
    shifted = torch.zeros_like(feature)
    if offset >= feature.shape[-2]:
        return shifted
    if seconds > 0:
        shifted[:, :-offset, :] = feature[:, offset:, :]
    else:
        shifted[:, offset:, :] = feature[:, :-offset, :]
    return shifted


def load_image(path: str, size: int = 128) -> torch.Tensor:
    image = Image.open(path).convert("RGB").resize((size, size))
    arr = np.asarray(image, dtype=np.float32).copy() / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1)


def load_mask(path: str, size: int = 128) -> torch.Tensor:
    image = Image.open(path).convert("L").resize((size, size), Image.Resampling.NEAREST)
    return torch.from_numpy((np.asarray(image).copy() > 0).astype(np.float32))[None]


def synthetic_batch(batch: int = 2, size: int = 64, samples: int = 2560):
    video = torch.rand(batch, 3, size, size)
    mask = torch.zeros(batch, 1, size, size)
    for i in range(batch):
        y, x = size // 3, (i + 1) * size // (batch + 1)
        mask[i, 0, y:y + size // 4, x:x + size // 4] = 1
        video[i] = video[i] * (1 - mask[i]) + mask[i] * torch.tensor([0.9, 0.2, 0.1])[:, None, None]
    audio = torch.randn(batch, samples) * 0.05
    return video, audio, mask
