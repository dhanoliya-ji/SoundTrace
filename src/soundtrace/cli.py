"""SoundTrace CLI: dataset checks, training, paired studies, evaluation, robustness, prediction."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import subprocess
import sys
import time
import uuid
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.nn import functional as F

from .data import build_avatar_manifest, build_avsbench_manifest, load_record, read_manifest, synthetic_batch, validate_manifest
from .model import build_model, load_checkpoint, predict_video

ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "artifacts" / "reports"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------------- metrics
def iou_scores(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Per-image IoU at 0.5; an empty target scores the correctly-empty fraction (AVSBench convention)."""
    pred, tgt = logits.sigmoid() > 0.5, target > 0.5
    inter, union = (pred & tgt).sum((1, 2, 3)).float(), (pred | tgt).sum((1, 2, 3)).float()
    empty_score = (~pred & ~tgt).sum((1, 2, 3)).float() / tgt[0].numel()
    return torch.where(tgt.sum((1, 2, 3)) == 0, empty_score, inter / union.clamp_min(1))


def f1_scores(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred, tgt = logits.sigmoid() > 0.5, target > 0.5
    denom = (pred.sum((1, 2, 3)) + tgt.sum((1, 2, 3))).float()
    return torch.where(denom > 0, 2 * (pred & tgt).sum((1, 2, 3)).float() / denom.clamp_min(1), torch.ones_like(denom))


def metric(logits, target) -> float:
    return iou_scores(logits, target).mean().item()


def f_measure_curves(logits, target, thresholds: int = 255):
    """Per-image precision/recall over thresholds for the AVSBench F-measure (None if mask empty)."""
    prob, tgt = logits.sigmoid().flatten(1), (target > 0.5).flatten(1)
    pred = prob[..., None] >= torch.linspace(0, 1 - 1e-10, thresholds, device=logits.device)
    tp = (pred & tgt[..., None]).sum(1).float()
    pos = tgt.sum(1, keepdim=True).float()
    curves = torch.stack((tp / (pred.sum(1) + 1e-20), tp / (pos + 1e-20)), 1)
    return [c if p > 0 else None for c, p in zip(curves, pos[:, 0])]


def avsbench_f_measure(curves_by_video: dict) -> float | None:
    """Mean over videos of max_t F_beta (beta^2=0.3) of the video-averaged P/R curve."""
    scores = []
    for curves in curves_by_video.values():
        if curves := [c for c in curves if c is not None]:
            p, r = torch.stack(curves).mean(0)
            scores.append(float((1.3 * p * r / (0.3 * p + r + 1e-20)).max()))
    return float(np.mean(scores)) if scores else None


# ---------------------------------------------------------------- corruptions
CONDITION_KEYS = {"name", "audio_video_shift_seconds", "audio_noise_std", "audio_attenuation",
                  "audio_dropout_probability", "visual_blur_sigma", "visual_downsample_factor",
                  "visual_occlusion_fraction"}


def corrupt(x: torch.Tensor, a: torch.Tensor, c: dict, rng: torch.Generator):
    """Deterministic audio/visual corruption. Masks are never modified. Time shifts happen at load time."""
    a = a * float(c.get("audio_attenuation", 1.0))
    if noise := float(c.get("audio_noise_std", 0)):
        a = a + torch.randn(a.shape, generator=rng) * noise
    if drop := float(c.get("audio_dropout_probability", 0)):
        a = a * (torch.rand(a.shape, generator=rng) >= drop)
    if blur := float(c.get("visual_blur_sigma", 0)):
        r = max(1, round(blur))
        x = F.avg_pool2d(F.pad(x, (r,) * 4, mode="replicate"), 2 * r + 1, stride=1)
    if (k := int(c.get("visual_downsample_factor", 1))) > 1:
        x = F.interpolate(F.interpolate(x, scale_factor=1 / k, mode="bilinear"), size=x.shape[-2:], mode="bilinear")
    if occ := float(c.get("visual_occlusion_fraction", 0)):
        h, w = x.shape[-2:]
        s = max(1, round((h * w * occ) ** 0.5))
        x = x.clone()
        x[..., (h - s) // 2:(h - s) // 2 + s, (w - s) // 2:(w - s) // 2 + s] = 0
    return x, a


# ---------------------------------------------------------------- helpers
def load_config(path) -> dict:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}


def manifest_path(cfg: dict) -> Path:
    path = Path(cfg.get("manifest", "data/manifests/avatar.jsonl"))
    return path if path.is_absolute() else ROOT / path


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def split_rows(rows: list[dict], cfg: dict) -> dict[str, list[dict]]:
    """Use manifest splits when present, else a seeded 70/15/15 split of *source videos*."""
    if rows and all("split" in r for r in rows):
        return {k: [r for r in rows if r["split"] == k] for k in ("train", "val", "test")}
    ids = sorted({str(r["video_id"]) for r in rows})
    if len(ids) < 3:
        raise ValueError("Need at least 3 distinct video_id values for video-level train/val/test splits.")
    random.Random(int(cfg.get("seed", 7))).shuffle(ids)
    n = max(1, round(len(ids) * 0.15))
    which = {v: "test" for v in ids[:n]} | {v: "val" for v in ids[n:2 * n]}
    out = {"train": [], "val": [], "test": []}
    for r in rows:
        out[which.get(str(r["video_id"]), "train")].append(r)
    return out


def checked_splits(cfg: dict):
    manifest = manifest_path(cfg)
    rows = read_manifest(manifest)
    status = validate_manifest(rows, manifest)
    if not status["valid"]:
        raise SystemExit(json.dumps(status["errors"][:10], indent=2))
    splits = split_rows(rows, cfg)
    if not all(splits.values()):
        raise SystemExit("Train, val and test source-video partitions must all contain examples.")
    return manifest, rows, splits


def provenance(manifest: Path, rows: list[dict], splits: dict, cfg: dict) -> dict:
    def git(*args):
        try:
            return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            return None
    ids = {k: sorted({str(r["video_id"]) for r in v}) for k, v in splits.items()}
    names = {str(r.get("dataset") or r.get("task") or "custom") for r in rows}
    return {"source_revision": git("rev-parse", "HEAD"), "source_tree_dirty": bool(git("status", "--porcelain")),
            "python": sys.version.split()[0], "torch": torch.__version__, "device": DEVICE,
            "seed": int(cfg.get("seed", 7)), "dataset": names.pop() if len(names) == 1 else "mixed",
            "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
            "split_fingerprint": hashlib.sha256(json.dumps(ids, sort_keys=True).encode()).hexdigest(),
            "split_video_counts": {k: len(v) for k, v in ids.items()}}


def save_report(report: dict) -> dict:
    REPORTS.mkdir(parents=True, exist_ok=True)
    rid = report["evaluation_id"]
    (REPORTS / f"{rid}.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    brief = {k: v for k, v in report.items() if not isinstance(v, (list, dict)) or k in ("condition_summary", "split_video_counts")}
    md = "# SoundTrace report\n\n" + "\n".join(f"- **{k}:** `{json.dumps(v)}`" for k, v in brief.items()) + "\n"
    (REPORTS / f"{rid}.md").write_text(md, encoding="utf-8")
    print(json.dumps(brief, indent=2))
    return report


def preload(rows: list[dict], manifest: Path, cfg: dict):
    """Load a split into memory once: images as uint8 to keep RAM bounded."""
    x, a, y = zip(*(load_record(r, manifest, cfg) for r in rows))
    return (torch.stack(x) * 255).round().byte(), torch.stack(a), torch.stack(y).bool()


# ---------------------------------------------------------------- train / evaluate
@torch.no_grad()
def evaluate(model, rows, manifest, cfg, conditions=({},)) -> tuple[list[dict], dict]:
    """Score held-out rows under each condition; returns per-example results and per-condition summary."""
    model.eval()
    rng = torch.Generator().manual_seed(int(cfg.get("seed", 7)))
    results, curves = [], {}
    for row in rows:
        clean = load_record(row, manifest, cfg)
        for c in conditions:
            name = str(c.get("name", "clean"))
            shift = float(c.get("audio_video_shift_seconds", 0))
            x, a, y = load_record(row, manifest, cfg, shift) if shift else clean
            x, a = corrupt(x[None], a[None], c, rng)
            start = time.perf_counter()
            logits = model(x.to(DEVICE), a.to(DEVICE))["logits"].float().cpu()
            latency = time.perf_counter() - start
            y = y[None]
            curves.setdefault(name, {}).setdefault(str(row["video_id"]), []).append(f_measure_curves(logits, y)[0])
            results.append({"video_id": row["video_id"], "timestamp": row["timestamp"], "condition": name,
                            "iou": iou_scores(logits, y).item(), "f1": f1_scores(logits, y).item(),
                            "latency_seconds": latency})
    summary = {}
    for name in curves:
        mine = [r for r in results if r["condition"] == name]
        summary[name] = {"mean_iou": float(np.mean([r["iou"] for r in mine])),
                         "mean_f1": float(np.mean([r["f1"] for r in mine])),
                         "f_measure": avsbench_f_measure(curves[name]), "examples": len(mine)}
    return results, summary


def run_smoke(gated: bool = False) -> dict:
    seed_all(7)
    model = build_model({"width": 16, "gated": gated}, pretrained=False)
    video, audio, mask = synthetic_batch(2, 64, 2560)
    out = model(video, audio)
    loss = F.binary_cross_entropy_with_logits(out["logits"], mask)
    loss.backward()
    result = {"kind": "synthetic engineering smoke test", "trained_on_real_data": False,
              "variant": "gated" if gated else "baseline", "shape": list(out["logits"].shape),
              "loss": float(loss), "synthetic_iou": metric(out["logits"].detach(), mask)}
    print(json.dumps(result, indent=2))
    return result


def train(cfg: dict) -> dict:
    if cfg.get("smoke"):
        return run_smoke(bool(cfg.get("gated")))
    manifest, rows, splits = checked_splits(cfg)
    seed_all(int(cfg.get("seed", 7)))
    ckpt = ROOT / cfg.get("checkpoint", "artifacts/checkpoints/best.pt")
    model = build_model(cfg).to(DEVICE)
    params = [p for p in model.parameters() if p.requires_grad]
    epochs, bs, patience = int(cfg.get("epochs", 10)), int(cfg.get("batch_size", 8)), int(cfg.get("patience", 5))
    opt = torch.optim.AdamW(params, lr=float(cfg.get("learning_rate", 3e-4)), weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    scaler = torch.amp.GradScaler(enabled=DEVICE == "cuda")
    print(f"loading {len(splits['train'])} train / {len(splits['val'])} val frames ...", flush=True)
    tx, ta, ty = preload(splits["train"], manifest, cfg)
    vx, va, vy = preload(splits["val"], manifest, cfg)
    best, stale, history, start = -1.0, 0, [], time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        for idx in torch.randperm(len(tx)).split(bs):
            x, a, y = tx[idx].float().div(255).to(DEVICE), ta[idx].to(DEVICE), ty[idx].float().to(DEVICE)
            if random.random() < 0.5:  # horizontal flip keeps audio-visual correspondence intact
                x, y = x.flip(-1), y.flip(-1)
            with torch.autocast(DEVICE, enabled=DEVICE == "cuda"):
                logits = model(x, a)["logits"].float()
            p = logits.sigmoid()
            loss = F.binary_cross_entropy_with_logits(logits, y) + 1 - (2 * (p * y).sum() + 1) / (p.sum() + y.sum() + 1)
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite training loss")
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            scaler.step(opt)
            scaler.update()
            losses.append(loss.item())
        sched.step()
        model.eval()
        with torch.no_grad():
            val = float(torch.cat([iou_scores(model(vx[i].float().div(255).to(DEVICE), va[i].to(DEVICE))["logits"].float(),
                                              vy[i].float().to(DEVICE)).cpu() for i in torch.arange(len(vx)).split(bs)]).mean())
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)), "val_iou": val})
        print(json.dumps(history[-1]), flush=True)
        if val > best:  # model selection uses validation only
            best, stale = val, 0
            ckpt.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"config": cfg, "model": model.state_dict(), "epoch": epoch, "val_iou": val}, ckpt)
        elif (stale := stale + 1) >= patience:
            break
    model, _ = load_checkpoint(ckpt, DEVICE)
    results, summary = evaluate(model, splits["test"], manifest, cfg)
    return save_report({"evaluation_id": uuid.uuid4().hex[:12], "kind": "real_data_experiment",
                        "variant": "gated" if model.gated else "baseline", **provenance(manifest, rows, splits, cfg),
                        "pretrained_visual": bool(cfg.get("pretrained")), "best_validation_iou": best,
                        "test_mean_iou": summary["clean"]["mean_iou"], "test_f1_at_0_5": summary["clean"]["mean_f1"],
                        "test_f_measure": summary["clean"]["f_measure"], "history": history,
                        "duration_seconds": time.time() - start, "checkpoint": ckpt.name,
                        "parameters": sum(p.numel() for p in model.parameters()), "test_examples": results})


def run_study(cfg: dict, seeds: list[int]) -> dict:
    """Train baseline and gated on identical manifests/splits per seed and report paired deltas."""
    sid, pairs = uuid.uuid4().hex[:12], []
    for seed in seeds:
        runs = {v: train({**cfg, "seed": seed, "gated": v == "gated",
                          "checkpoint": f"artifacts/checkpoints/{sid}-{v}-seed{seed}.pt"}) for v in ("baseline", "gated")}
        b, g = runs["baseline"], runs["gated"]
        assert b["split_fingerprint"] == g["split_fingerprint"], "paired runs must share a split"
        pairs.append({"seed": seed, "baseline_test_iou": b["test_mean_iou"], "gated_test_iou": g["test_mean_iou"],
                      "test_iou_delta": g["test_mean_iou"] - b["test_mean_iou"],
                      "baseline_checkpoint": b["checkpoint"], "gated_checkpoint": g["checkpoint"]})
    deltas = [p["test_iou_delta"] for p in pairs]
    return save_report({"evaluation_id": sid, "kind": "paired_study", "dataset": b["dataset"], "seeds": seeds,
                        "manifest_sha256": b["manifest_sha256"], "pairs": pairs,
                        "mean_test_iou_delta": float(np.mean(deltas)),
                        "std_test_iou_delta": float(np.std(deltas, ddof=1)) if len(deltas) > 1 else None,
                        "interpretation": "Positive favors gated fusion on this paired test split; not a significance test."})


def run_evaluation(kind: str, file_cfg: dict, checkpoint: str) -> dict:
    model, state = load_checkpoint(checkpoint, DEVICE)
    cfg = {**file_cfg, **state["config"]}  # model, preprocessing and seed come from training
    conditions = file_cfg.get("conditions", [{"name": "clean"}]) if kind == "robustness" else [{"name": "clean"}]
    if unknown := {k for c in conditions for k in c} - CONDITION_KEYS:
        raise SystemExit(f"Unknown robustness condition keys: {sorted(unknown)}")
    manifest, rows, splits = checked_splits(cfg)
    results, summary = evaluate(model, splits["test"], manifest, cfg, conditions)
    clean = summary.get("clean", {}).get("mean_iou")
    for s in summary.values():
        s["iou_change_vs_clean"] = None if clean is None else s["mean_iou"] - clean
    return save_report({"evaluation_id": uuid.uuid4().hex[:12], "kind": kind, "evaluated_split": "test",
                        "variant": "gated" if model.gated else "baseline", **provenance(manifest, rows, splits, cfg),
                        "checkpoint": Path(checkpoint).name, "mean_iou": float(np.mean([r["iou"] for r in results])),
                        "mean_latency_seconds": float(np.mean([r["latency_seconds"] for r in results])),
                        "condition_summary": summary, "results": results})


# ---------------------------------------------------------------- entry point
def main() -> None:
    parser = argparse.ArgumentParser(prog="soundtrace")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("doctor", "dataset-check", "smoke"):
        sub.add_parser(name)
    for name in ("build-manifest", "preprocess", "train", "study", "evaluate", "robustness"):
        p = sub.add_parser(name)
        p.add_argument("--config", required=True)
        p.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="override config values")
    bm = sub.choices["build-manifest"]
    bm.add_argument("--task", choices=("s4", "ms3", "avatar"), default="avatar")
    bm.add_argument("--data-root")
    bm.add_argument("--output")
    bm.add_argument("--max-videos", type=int)
    sub.choices["study"].add_argument("--seeds", default="7,17,27")
    for name in ("evaluate", "robustness"):
        sub.choices[name].add_argument("--checkpoint", required=True)
    p = sub.add_parser("predict")
    p.add_argument("--video", required=True)
    p.add_argument("--checkpoint", default=os.environ.get("SOUNDTRACE_CHECKPOINT", str(ROOT / "artifacts/checkpoints/best.pt")))
    sub.add_parser("report").add_argument("--evaluation-id", required=True)
    args = parser.parse_args()

    data_root = Path(os.environ.get("SOUNDTRACE_DATA_DIR", ROOT / "data" / "raw"))
    cfg = load_config(args.config) if hasattr(args, "config") else {}
    for item in getattr(args, "set", []):
        key, _, value = item.partition("=")
        cfg[key] = yaml.safe_load(value)

    if args.command == "doctor":
        from .data import ffmpeg_exe
        try:
            ffmpeg = ffmpeg_exe()
        except RuntimeError:
            ffmpeg = None
        print(json.dumps({"python": sys.version.split()[0], "torch": torch.__version__, "cuda": torch.cuda.is_available(),
                          "ffmpeg": ffmpeg, "dataset_root_exists": data_root.exists()}, indent=2))
    elif args.command == "dataset-check":
        files = [p.suffix.lower() for p in data_root.rglob("*")] if data_root.exists() else []
        print(json.dumps({"root": str(data_root), "available": data_root.exists(),
                          "video_count": sum(s in {".mp4", ".mov", ".avi", ".mkv"} for s in files),
                          "image_count": sum(s in {".png", ".jpg", ".jpeg"} for s in files)}, indent=2))
    elif args.command == "smoke":
        run_smoke()
    elif args.command == "build-manifest":
        root = Path(args.data_root or cfg.get("data_root") or data_root / ("AVATAR" if args.task == "avatar" else ""))
        out = ROOT / (args.output or f"data/manifests/{args.task}.jsonl")
        result = build_avatar_manifest(root, out, args.max_videos) if args.task == "avatar" else build_avsbench_manifest(root, out, args.task)
        print(json.dumps({**result, "errors": result["errors"][:20], "error_count": len(result["errors"])}, indent=2))
        if not result["valid"]:
            raise SystemExit(2)
    elif args.command == "preprocess":
        path = manifest_path(cfg)
        status = validate_manifest(read_manifest(path), path)
        print(json.dumps(status, indent=2))
        if not status["valid"]:
            raise SystemExit(2)
    elif args.command == "train":
        train(cfg)
    elif args.command == "study":
        run_study(cfg, [int(s) for s in args.seeds.split(",") if s.strip()])
    elif args.command in ("evaluate", "robustness"):
        run_evaluation(args.command, cfg, args.checkpoint)
    elif args.command == "predict":
        if not Path(args.checkpoint).is_file():
            raise SystemExit("A trained --checkpoint is required.")
        model, state = load_checkpoint(args.checkpoint, DEVICE)
        result = predict_video(model, state["config"], Path(args.video))
        from PIL import Image
        folder = ROOT / "artifacts" / "predictions" / Path(args.video).stem
        folder.mkdir(parents=True, exist_ok=True)
        for f in result["frames"]:
            Image.fromarray(np.array(f.pop("mask"), dtype=np.uint8) * 255).save(folder / f"{f['frame']:06d}.png")
        (folder / "metadata.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(json.dumps({"frames": len(result["frames"]), "output": str(folder)}, indent=2))
    elif args.command == "report":
        path = REPORTS / f"{args.evaluation_id}.json"
        if not path.is_file():
            raise SystemExit(f"Evaluation report not found: {args.evaluation_id}")
        print(path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
