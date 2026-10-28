import csv
import json
import pickle
import wave
import zipfile

import pytest
import torch
from PIL import Image

from soundtrace import api
from soundtrace.cli import avsbench_f_measure, corrupt, f_measure_curves, metric, split_rows
from soundtrace.data import (
    audio_window,
    build_avatar_manifest,
    build_avsbench_manifest,
    decode_audio,
    load_record,
    read_manifest,
    safe_extract,
    shift_log_mel,
    synthetic_batch,
    validate_manifest,
)
from soundtrace.model import SoundTraceNet


def test_audio_window_center_and_padding():
    wave_ = torch.arange(10, dtype=torch.float32)
    assert audio_window(wave_, 5, 1, 4).tolist() == [3, 4, 5, 6]
    assert audio_window(wave_, 1, 1, 4).tolist() == [0, 0, 1, 2]


def test_log_mel_time_shift_zero_pads_without_wrapping():
    feature = torch.arange(12, dtype=torch.float32).reshape(1, 6, 2)
    assert shift_log_mel(feature, 1 / 3).tolist() == [[[4, 5], [6, 7], [8, 9], [10, 11], [0, 0], [0, 0]]]
    assert shift_log_mel(feature, -1 / 3).tolist() == [[[0, 0], [0, 0], [0, 1], [2, 3], [4, 5], [6, 7]]]


@pytest.mark.parametrize("gated", [False, True])
def test_model_shapes_gate_gradients_and_log_mel_input(gated):
    video, audio, mask = synthetic_batch(2, 32, 4000)
    model = SoundTraceNet(16, gated)
    out = model(video, audio)
    assert out["logits"].shape == mask.shape and out["gate"].shape == (2, 16)
    torch.nn.functional.binary_cross_entropy_with_logits(out["logits"], mask).backward()
    assert (model.gate[-2].weight.grad is not None) == gated  # the gate is wired in only when gated
    assert model(video, torch.rand(2, 1, 96, 64))["logits"].shape == (2, 1, 32, 32)


def test_metrics_follow_avsbench_conventions():
    target = torch.zeros(1, 1, 8, 8)
    target[..., 2:5, 2:5] = 1
    assert metric(target * 20 - 10, target) == 1
    empty = torch.zeros(1, 1, 2, 2)
    assert metric(torch.full_like(empty, -20), empty) == 1
    assert metric(torch.tensor([[[[20.0, -20], [-20, -20]]]]), empty) == 0.75
    t = torch.zeros(2, 1, 4, 4)
    t[0, 0, 1:3, 1:3] = 1
    curves = f_measure_curves(torch.where(t > 0, 20.0, -20.0), t)
    assert curves[1] is None and avsbench_f_measure({"a": [curves[0]], "b": [curves[1]]}) > 0.999


def test_corruptions_are_deterministic_and_bounded():
    x, a = torch.ones(1, 3, 8, 8), torch.ones(1, 100)
    c = {"audio_noise_std": 0.1, "audio_dropout_probability": 0.5, "visual_occlusion_fraction": 0.25}
    x1, a1 = corrupt(x, a, c, torch.Generator().manual_seed(1))
    x2, a2 = corrupt(x, a, c, torch.Generator().manual_seed(1))
    assert torch.equal(a1, a2) and torch.equal(x1, x2) and (x1 == 0).sum() == 3 * 16
    assert corrupt(x, a, {"audio_attenuation": 0}, torch.Generator())[1].abs().sum() == 0


def test_video_level_split_has_no_leakage():
    rows = [{"video_id": f"v{i % 10}", "timestamp": i} for i in range(40)]
    splits = split_rows(rows, {"seed": 3})
    ids = [{r["video_id"] for r in s} for s in splits.values()]
    assert not (ids[0] & ids[1] or ids[0] & ids[2] or ids[1] & ids[2]) and all(ids)


def test_manifest_validation_catches_leakage_missing_files_and_empty(tmp_path):
    rows = [{"video": "v.mp4", "mask": "m.png", "video_id": "same", "timestamp": 0, "split": "train"},
            {"video": "v.mp4", "mask": "m.png", "video_id": "same", "timestamp": 1, "split": "test"}]
    errors = [e["error"] for e in validate_manifest(rows, tmp_path / "m.jsonl")["errors"]]
    assert any("leakage" in e for e in errors) and any("video missing" in e for e in errors)
    assert not validate_manifest([], tmp_path / "m.jsonl")["valid"]


def write_features(folder, name):
    folder.mkdir(parents=True)
    with (folder / f"{name}.pkl").open("wb") as handle:
        pickle.dump(torch.zeros(5, 1, 96, 64), handle)


def test_avsbench_s4_manifest_is_one_shot_for_train(tmp_path):
    data = tmp_path / "Single-source" / "s4_data"
    data.parent.mkdir()
    with (data.parent / "s4_meta_data.csv").open("w", newline="") as f:
        csv.writer(f).writerows([["video_id", "category", "split"], ["clipA", "guitar", "train"]])
    for kind in ("visual_frames", "gt_masks"):
        (data / kind / "train/guitar/clipA").mkdir(parents=True)
    for i in range(1, 6):
        Image.new("RGB", (8, 8)).save(data / f"visual_frames/train/guitar/clipA/clipA_{i}.png")
        Image.new("L", (8, 8), 255).save(data / f"gt_masks/train/guitar/clipA/clipA_{i}.png")
    write_features(data / "audio_log_mel/train/guitar", "clipA")
    out = tmp_path / "s4.jsonl"
    assert build_avsbench_manifest(tmp_path, out, "s4")["valid"]
    rows = read_manifest(out)
    assert len(rows) == 1 and rows[0]["timestamp"] == 0.5 and validate_manifest(rows, out)["valid"]
    image, audio, mask = load_record(rows[0], out, {"image_size": 16})
    assert image.shape == (3, 16, 16) and audio.shape == (1, 96, 64) and mask.shape == (1, 16, 16)


def test_avsbench_ms3_manifest_uses_video_frame_names(tmp_path):
    data = tmp_path / "Multi-sources" / "ms3_data"
    data.parent.mkdir()
    with (data.parent / "ms3_meta_data.csv").open("w", newline="") as f:
        csv.writer(f).writerows([["video_name", "split"], ["clipB", "train"]])
    (data / "visual_frames/clipB").mkdir(parents=True)
    (data / "gt_masks/train/clipB").mkdir(parents=True)
    for i in range(1, 6):
        Image.new("RGB", (8, 8)).save(data / f"visual_frames/clipB/clipB.mp4_{i}.png")
        Image.new("L", (8, 8), 255).save(data / f"gt_masks/train/clipB/clipB_{i}.png")
    write_features(data / "audio_log_mel/train", "clipB")
    out = tmp_path / "ms3.jsonl"
    assert build_avsbench_manifest(tmp_path, out, "ms3")["valid"]
    rows = read_manifest(out)
    assert [r["audio_index"] for r in rows] == [0, 1, 2, 3, 4]


def test_avatar_conversion_rasterizes_polygons_and_pairs_padded_ids(tmp_path):
    root = tmp_path / "avatar"
    (root / "downloads").mkdir(parents=True)
    ann = {"video_id": "sample_12", "frame_number": 5, "original_width": 16, "original_height": 16,
           "fps": 10.0, "duration": 2.0, "annotations": [{"segmentation": [[2, 2, 12, 2, 12, 12, 2, 12]],
                                                           "task": "Single-Sound", "audio_visual_category": "bell"}]}
    Image.new("RGB", (16, 16)).save(tmp_path / "f.jpg")
    with zipfile.ZipFile(root / "downloads/metadata.zip", "w") as z:
        z.writestr("sample_00012/00005.json", json.dumps(ann))
        z.write(tmp_path / "f.jpg", "sample_00012/00005.jpg")
    with zipfile.ZipFile(root / "downloads/video.zip", "w") as z:
        z.writestr("video/sample_00012.mp4", b"x")
        z.writestr("video/unrelated_00001.mp4", b"x")
    out = tmp_path / "avatar.jsonl"
    result = build_avatar_manifest(root, out, max_videos=1)
    rows = read_manifest(out)
    assert result["valid"] and rows[0]["video_id"] == "sample_12" and rows[0]["timestamp"] == 0.5
    assert validate_manifest(rows, out)["valid"] and not (root / "video/unrelated_00001.mp4").exists()
    mask = Image.open(rows[0]["mask"])
    assert mask.getpixel((5, 5)) == 255 and mask.getpixel((0, 0)) == 0


def test_archive_extraction_rejects_parent_traversal(tmp_path):
    with zipfile.ZipFile(tmp_path / "bad.zip", "w") as z:
        z.writestr("../outside.txt", "no")
    with pytest.raises(ValueError, match="Unsafe path"):
        safe_extract(tmp_path / "bad.zip", tmp_path / "out")
    assert not (tmp_path / "outside.txt").exists()


def test_decode_audio(tmp_path):
    with pytest.raises((RuntimeError, ValueError)):
        decode_audio(tmp_path / "missing.mp4")
    with wave.open(str(tmp_path / "a.wav"), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(bytes(3200))
    assert decode_audio(tmp_path / "a.wav").shape == (1600,)


def test_prediction_jobs_persist_and_reject_bad_ids(tmp_path, monkeypatch):
    monkeypatch.setattr(api, "DATA_DIR", tmp_path)
    api.write_job("abc123", {"id": "abc123", "status": "completed", "fps": 2.0, "duration": 1.0, "size": [2, 2],
                             "frames": [{"frame": 0, "timestamp": 0.0, "mask": [[0, 1], [1, 0]]}]})
    assert api.get_prediction("abc123")["frame_count"] == 1
    assert api.get_masks("abc123")["frames"][0]["mask"] == [[0, 1], [1, 0]]
    assert "mask" not in api.get_metadata("abc123")["frames"][0]
    with pytest.raises(api.HTTPException):
        api.get_prediction("../etc")


def test_status_endpoints(tmp_path, monkeypatch):
    for env in ("SOUNDTRACE_CHECKPOINT", "SOUNDTRACE_DATA_DIR", "SOUNDTRACE_MANIFEST"):
        monkeypatch.delenv(env, raising=False)
    monkeypatch.setattr(api, "PROJECT_ROOT", tmp_path)
    assert not api.model_status()["available"] and not api.dataset_status()["manifest_valid"]
    ckpt = tmp_path / "artifacts/checkpoints/best.pt"
    ckpt.parent.mkdir(parents=True)
    torch.save({"config": {"gated": True}, "model": {}, "val_iou": 0.5}, ckpt)
    assert api.model_status()["variant"] == "gated"
    for name in ("f.jpg", "m.png", "v.mp4"):
        (tmp_path / name).write_bytes(b"x")
    manifest = tmp_path / "data/manifests/avatar.jsonl"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(json.dumps({"video_id": "c", "timestamp": 0, "frame": str(tmp_path / "f.jpg"), "dataset": "AVATAR",
                                    "mask": str(tmp_path / "m.png"), "video": str(tmp_path / "v.mp4")}) + "\n")
    status = api.dataset_status()
    assert status["manifest_valid"] and status["video_count"] == 1 and status["dataset"] == "AVATAR"
