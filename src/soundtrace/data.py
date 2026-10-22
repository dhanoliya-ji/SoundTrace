"""Manifest, decoding, alignment and synthetic fixture helpers."""
from __future__ import annotations

import json
import csv
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
    """Read the official object-task metadata and emitted frame/mask filenames.

    This creates sample records from the official metadata CSV and processed
    AVSBench image/mask directories. It never guesses a pair from a shared
    video/image stem; absent or ambiguous files are reported as errors.
    """
    if task not in {"s4", "ms3"}:
        raise ValueError("AVSBench object task must be s4 or ms3")
    subset, stem = ("Single-source", "s4") if task == "s4" else ("Multi-sources", "ms3")
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
        if not all(resolved.values()):
            raise ValueError(f"Unexpected {csv_name} columns: {reader.fieldnames}")
        for line_no, row in enumerate(reader, 2):
            video_id = str(row[resolved["video_name"]]).strip()
            category = str(row[resolved["category"]]).strip()
            split = str(row[resolved["split"]]).strip().lower()
            split = "val" if split in {"valid", "validation", "val"} else split
            if not video_id or split not in {"train", "val", "test"}:
                errors.append({"line": line_no, "video_id": video_id, "error": f"invalid id or split {split!r}"})
                continue
            video_matches = [p for p in root.rglob(f"{video_id}.mp4") if "raw_videos" in p.parts]
            if len(video_matches) != 1:
                errors.append({"line": line_no, "video_id": video_id, "error": f"expected one raw video, found {len(video_matches)}"})
                continue
            frame_ids = [1] if task == "s4" and split == "train" else [1, 2, 3, 4, 5]
            for frame_id in frame_ids:
                image_names = {f"{video_id}.mp4_{frame_id}.png", f"{video_id}_{frame_id}.png"}
                image_matches = [p for p in root.rglob("*.png") if "visual_frames" in p.parts and p.name in image_names]
                mask_name = f"{video_id}_{frame_id}.png"
                mask_matches = [p for p in root.rglob(mask_name) if "gt_masks" in p.parts]
                if len(image_matches) != 1 or len(mask_matches) != 1:
                    errors.append({"line": line_no, "video_id": video_id, "frame_id": frame_id,
                                   "error": f"expected one visual frame and mask; found {len(image_matches)} frames and {len(mask_matches)} masks"})
                    continue
                rows.append({"video_id": video_id, "video": str(video_matches[0].resolve()),
                             "frame": str(image_matches[0].resolve()), "mask": str(mask_matches[0].resolve()),
                             "timestamp": frame_id - 0.5, "split": split, "task": task,
                             "category": category, "annotation_frame": frame_id})
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
    required = {"video", "mask", "video_id", "timestamp"}
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
        video, mask = resolve_path(row["video"], manifest_path), resolve_path(row["mask"], manifest_path)
        if not video.is_file(): errors.append({"line": index, "error": f"video missing: {video.name}"})
        if not mask.is_file(): errors.append({"line": index, "error": f"mask missing: {mask.name}"})
        if row.get("frame") and not resolve_path(row["frame"], manifest_path).is_file():
            errors.append({"line": index, "error": f"frame missing: {Path(row['frame']).name}"})
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
    video_path = resolve_path(row["video"], manifest_path)
    mask_path = resolve_path(row["mask"], manifest_path)
    if row.get("frame"):
        video = load_image(str(resolve_path(row["frame"], manifest_path)), size)
    else:
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
    waveform = cached_audio(str(video_path), sample_rate)
    audio = audio_window(waveform, float(row["timestamp"]), sample_rate, duration)
    return video, audio, mask


@lru_cache(maxsize=16)
def cached_audio(path: str, sample_rate: int) -> torch.Tensor:
    return decode_audio(path, sample_rate)


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
