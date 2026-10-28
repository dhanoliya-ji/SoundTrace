"""Dataset manifests (AVSBench S4/MS3, AVATAR), decoding, alignment and fixtures."""
from __future__ import annotations

import csv
import hashlib
import json
import pickle
import shutil
import subprocess
import zipfile
from functools import cache, lru_cache
from pathlib import Path, PurePosixPath

import numpy as np
import torch
from PIL import Image, ImageDraw

SPLITS = ("train", "val", "test")


def read_manifest(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_manifest(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def resolve(value: str, manifest: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (manifest.parent / path).resolve()


def build_avsbench_manifest(root: Path, output: Path, task: str) -> dict:
    """Read official S4/MS3 metadata CSVs and pair frames, masks and log-mel features."""
    subset, base = ("Single-source", "s4_data") if task == "s4" else ("Multi-sources", "ms3_data")
    metas = list(root.rglob(f"{task}_meta_data.csv"))
    if len(metas) != 1:
        raise ValueError(f"Expected one {task}_meta_data.csv below dataset root; found {len(metas)}")
    base = root / subset / base
    rows, errors = [], []
    with metas[0].open(newline="", encoding="utf-8-sig") as handle:
        for line, raw in enumerate(csv.DictReader(handle), 2):
            row = {k.strip().lower(): str(v).strip() for k, v in raw.items()}
            vid = row.get("video_name") or row.get("video_id") or ""
            cat = row.get("category", "")
            split = {"valid": "val", "validation": "val"}.get(row.get("split", "").lower(), row.get("split", "").lower())
            if not vid or split not in SPLITS:
                errors.append({"line": line, "error": f"invalid id or split {split!r}"})
                continue
            # S4 nests by category; MS3 frames are not split into folders.
            sub = f"{split}/{cat}" if task == "s4" else split
            frames = base / "visual_frames" / (f"{sub}/{vid}" if task == "s4" else vid)
            feats = base / "audio_log_mel" / sub / f"{vid}.pkl"
            if not feats.is_file():
                errors.append({"line": line, "error": f"missing audio features for {vid}"})
                continue
            # Official S4 training is one-shot: only the first frame is labelled.
            for i in [1] if task == "s4" and split == "train" else range(1, 6):
                frame = frames / (f"{vid}_{i}.png" if task == "s4" else f"{vid}.mp4_{i}.png")
                mask = base / "gt_masks" / sub / vid / f"{vid}_{i}.png"
                if not frame.is_file() or not mask.is_file():
                    errors.append({"line": line, "error": f"missing frame or mask {i} for {vid}"})
                    continue
                rows.append({"video_id": vid, "frame": str(frame.resolve()), "mask": str(mask.resolve()),
                             "audio_features": str(feats.resolve()), "audio_index": i - 1,
                             "timestamp": i - 0.5, "split": split, "task": task, "category": cat,
                             "annotation_frame": i})
    write_manifest(rows, output)
    return {"task": task, "records": len(rows), "errors": errors, "manifest": str(output),
            "valid": bool(rows) and not errors}


def safe_extract(archive: Path, dest: Path, members: set[str] | None = None) -> None:
    """Extract a ZIP, rejecting any member path that could escape `dest`."""
    with zipfile.ZipFile(archive) as bundle:
        for name in bundle.namelist():
            parts = PurePosixPath(name).parts
            if "\\" in name or name.startswith("/") or ".." in parts or (parts and ":" in parts[0]):
                raise ValueError(f"Unsafe path in dataset archive {archive.name}: {name}")
        bundle.extractall(dest, members=None if members is None else sorted(members))


def canonical_id(value: str) -> str:
    """`clip_00012` and `clip_12` name the same AVATAR video."""
    head, sep, tail = value.rpartition("_")
    return f"{head}_{int(tail)}" if sep and tail.isdigit() else value


def build_avatar_manifest(root: Path, output: Path, max_videos: int | None = None) -> dict:
    """Rasterize AVATAR polygon annotations into masks and pair frames with source clips."""
    root = root.resolve()
    meta_dir, video_dir, dl = root / "metadata", root / "video", root / "downloads"
    if not meta_dir.exists() and (dl / "metadata.zip").is_file():
        safe_extract(dl / "metadata.zip", meta_dir)
    if not meta_dir.is_dir():
        raise ValueError("AVATAR requires downloads/metadata.zip or an extracted metadata/ folder")
    annotations = sorted(meta_dir.rglob("*.json"))
    ids = sorted({canonical_id(p.parent.name) for p in annotations})
    if max_videos is not None:  # deterministic subset, stable as max_videos grows
        ids = sorted(ids, key=lambda v: hashlib.sha256(v.encode()).digest())[:max_videos]
    ids = set(ids)
    if (dl / "video.zip").is_file():
        with zipfile.ZipFile(dl / "video.zip") as bundle:
            wanted = {n for n in bundle.namelist() if n.lower().endswith(".mp4")
                      and canonical_id(Path(n).stem) in ids and not (root / n).is_file()}
        if wanted:
            safe_extract(dl / "video.zip", root, wanted)
    videos = {canonical_id(p.stem): p for p in video_dir.glob("*.mp4")} if video_dir.is_dir() else {}
    rows, errors = [], []
    for path in annotations:
        if canonical_id(path.parent.name) not in ids:
            continue
        try:
            rec = json.loads(path.read_text(encoding="utf-8"))
            vid, frame_no, fps = canonical_id(str(rec["video_id"])), int(rec["frame_number"]), float(rec["fps"])
        except (OSError, ValueError, KeyError, TypeError) as exc:
            errors.append({"annotation": path.name, "error": f"could not read annotation: {exc}"})
            continue
        image, video = path.with_suffix(".jpg"), videos.get(vid)
        duration, ts = float(rec.get("duration", 0)), frame_no / fps if fps > 0 else -1
        if not image.is_file() or video is None or ts < 0 or (duration and ts > duration):
            errors.append({"video_id": vid, "error": f"missing frame/video or bad timing for {path.name}"})
            continue
        mask = Image.new("L", (int(rec.get("original_width", 640)), int(rec.get("original_height", 360))))
        draw = ImageDraw.Draw(mask)
        for inst in rec.get("annotations", []):
            polys = inst.get("segmentation") or []
            for poly in [polys] if polys and isinstance(polys[0], (int, float)) else polys:
                if len(poly) >= 6:
                    draw.polygon(list(zip(poly[0::2], poly[1::2])), fill=255)
        mask_path = root / "soundtrace_masks" / vid / f"{frame_no:05d}.png"
        mask_path.parent.mkdir(parents=True, exist_ok=True)
        mask.save(mask_path)
        anns = rec.get("annotations", [])
        rows.append({"video_id": vid, "frame": str(image), "video": str(video), "mask": str(mask_path),
                     "timestamp": ts, "frame_number": frame_no, "fps": fps, "duration": duration,
                     "category": sorted({str(a.get("audio_visual_category", "")) for a in anns}),
                     "scenario": sorted({str(a.get("task", "")) for a in anns}), "dataset": "AVATAR"})
    write_manifest(rows, output)
    return {"task": "avatar", "records": len(rows), "videos": len({r["video_id"] for r in rows}),
            "errors": errors, "manifest": str(output), "valid": bool(rows) and not errors,
            "note": "AVATAR has no official split; SoundTrace assigns deterministic source-video splits."}


def validate_manifest(rows: list[dict], manifest: Path) -> dict:
    errors, seen = [], {}
    if not rows:
        errors.append({"line": 0, "error": "manifest contains no records"})
    if 0 < sum("split" in r for r in rows) < len(rows):
        errors.append({"line": 0, "error": "split must be present on every row or on none of them"})
    for i, row in enumerate(rows, 1):
        missing = {"mask", "video_id", "timestamp"} - row.keys()
        if missing:
            errors.append({"line": i, "error": f"missing fields: {sorted(missing)}"})
            continue
        for key in ("mask", "video", "frame", "audio_features"):
            if row.get(key) and not resolve(row[key], manifest).is_file():
                errors.append({"line": i, "error": f"{key} missing: {Path(row[key]).name}"})
        if not row.get("video") and not (row.get("frame") and row.get("audio_features")):
            errors.append({"line": i, "error": "provide a video path, or both frame and audio_features paths"})
        if split := row.get("split"):
            vid = str(row["video_id"])
            if split not in SPLITS:
                errors.append({"line": i, "error": f"invalid split {split!r}"})
            if seen.setdefault(vid, split) != split:
                errors.append({"line": i, "error": f"video leakage: {vid} appears in {seen[vid]} and {split}"})
    return {"records": len(rows), "valid": not errors, "errors": errors}


def ffmpeg_exe() -> str:
    if exe := shutil.which("ffmpeg"):
        return exe
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError as exc:
        raise RuntimeError("FFmpeg is required for audio decoding. Install FFmpeg or imageio-ffmpeg.") from exc


def decode_audio(path: str | Path, sample_rate: int = 16000) -> torch.Tensor:
    """Decode a media file's audio to a mono float waveform."""
    cmd = [ffmpeg_exe(), "-v", "error", "-i", str(path), "-vn", "-ac", "1", "-ar", str(sample_rate), "-f", "f32le", "pipe:1"]
    try:
        out = subprocess.run(cmd, check=True, capture_output=True).stdout
    except subprocess.CalledProcessError as exc:
        raise ValueError(f"Audio decode failed for {Path(path).name}: {exc.stderr.decode(errors='replace')[-300:]}") from exc
    if not out:
        raise ValueError(f"No decodable audio track found in {Path(path).name}")
    return torch.from_numpy(np.frombuffer(out, dtype="<f4").copy())


@cache
def cached_audio(path: str, sample_rate: int) -> torch.Tensor:
    return decode_audio(path, sample_rate)


def audio_window(wave: torch.Tensor, timestamp: float, sample_rate: int, seconds: float = 1.0) -> torch.Tensor:
    """Fixed window centred on `timestamp` (starting at 0 for t<=0), zero padded at the edges."""
    n = max(1, round(sample_rate * seconds))
    start = 0 if timestamp <= 0 else round(timestamp * sample_rate) - n // 2
    out = wave.new_zeros(n)
    a, b = max(0, start), min(wave.numel(), start + n)
    if b > a:
        out[a - start:b - start] = wave[a:b]
    return out


def shift_log_mel(feature: torch.Tensor, seconds: float, clip_seconds: float = 1.0) -> torch.Tensor:
    """Shift a [channels, time, mel] clip in time with zero padding (never circular)."""
    k = round(abs(seconds) / clip_seconds * feature.shape[-2])
    if k == 0:
        return feature
    out = torch.zeros_like(feature)
    if k < feature.shape[-2]:
        if seconds > 0:
            out[:, :-k] = feature[:, k:]
        else:
            out[:, k:] = feature[:, :-k]
    return out


@lru_cache(maxsize=64)
def avsbench_features(path: str) -> torch.Tensor:
    """Official AVSBench log-mel pickles: [5, 1, 96, 64] (one clip per annotated second)."""
    with open(path, "rb") as handle:
        value = pickle.load(handle)
    feats = torch.as_tensor(value.detach().cpu() if hasattr(value, "detach") else value, dtype=torch.float32)
    if feats.ndim != 4 or feats.shape[0] != 5:
        raise ValueError(f"Unsupported audio feature shape in {Path(path).name}: {list(feats.shape)}")
    return feats


def load_image(path: str | Path, size: int) -> torch.Tensor:
    return to_tensor(Image.open(path).convert("RGB"), size)


def to_tensor(image: Image.Image, size: int) -> torch.Tensor:
    return torch.from_numpy(np.asarray(image.resize((size, size)), dtype=np.float32) / 255).permute(2, 0, 1)


def load_mask(path: str | Path, size: int) -> torch.Tensor:
    image = Image.open(path).convert("L").resize((size, size), Image.Resampling.NEAREST)
    return torch.from_numpy(np.asarray(image) > 0).float()[None]


def load_record(row: dict, manifest: Path, cfg: dict, shift: float = 0.0):
    """Return (image [3,S,S], audio waveform [N] or log-mel [1,T,M], mask [1,S,S]).

    `shift` moves the audio window relative to the frame (seconds) for misalignment tests.
    """
    size, sr, seconds = int(cfg.get("image_size", 224)), int(cfg.get("audio_sample_rate", 16000)), float(cfg.get("segment_seconds", 1.0))
    if row.get("frame"):
        image = load_image(resolve(row["frame"], manifest), size)
    else:
        import cv2
        cap = cv2.VideoCapture(str(resolve(row["video"], manifest)))
        cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, float(row["timestamp"])) * 1000)
        ok, frame = cap.read()
        cap.release()
        if not ok:
            raise ValueError(f"Cannot decode frame at {row['timestamp']}s in {Path(row['video']).name}")
        image = to_tensor(Image.fromarray(np.ascontiguousarray(frame[:, :, ::-1])), size)
    if row.get("audio_features"):
        audio = shift_log_mel(avsbench_features(str(resolve(row["audio_features"], manifest)))[int(row["audio_index"])], shift, seconds)
    else:
        audio = audio_window(cached_audio(str(resolve(row["video"], manifest)), sr), float(row["timestamp"]) + shift, sr, seconds)
    return image, audio, load_mask(resolve(row["mask"], manifest), size)


def synthetic_batch(batch: int = 2, size: int = 64, samples: int = 2560):
    """Coloured squares + noise audio. Proves the pipeline runs, nothing about accuracy."""
    video, mask = torch.rand(batch, 3, size, size), torch.zeros(batch, 1, size, size)
    for i in range(batch):
        x = (i + 1) * size // (batch + 1)
        mask[i, 0, size // 3:size // 3 + size // 4, x:x + size // 4] = 1
        video[i] = video[i] * (1 - mask[i]) + mask[i] * torch.tensor([0.9, 0.2, 0.1])[:, None, None]
    return video, torch.randn(batch, samples) * 0.05, mask
