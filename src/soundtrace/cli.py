"""Dataset, training, inference and evaluation CLI."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import time
import uuid

import numpy as np
import torch
from torch.nn import functional as F
import yaml

from .data import build_avsbench_manifest, build_manifest, load_record, read_manifest, synthetic_batch, validate_manifest
from .model import build_model

ROOT = Path(__file__).resolve().parents[2]


def metric(logits, target):
    pred, target = logits.sigmoid() > 0.5, target > 0.5
    inter = (pred & target).sum((1, 2, 3)).float()
    union = (pred | target).sum((1, 2, 3)).float()
    return torch.where(union > 0, inter / union, torch.ones_like(union)).mean().item()


def f1_metric(logits, target):
    pred, target = logits.sigmoid() > 0.5, target > 0.5
    true_positive = (pred & target).sum((1, 2, 3)).float()
    denominator = pred.sum((1, 2, 3)).float() + target.sum((1, 2, 3)).float()
    return torch.where(denominator > 0, 2 * true_positive / denominator,
                       torch.ones_like(denominator)).mean().item()


def load_config(path):
    return yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}


def manifest_path(cfg):
    path = Path(cfg.get("manifest", "data/manifests/train.jsonl"))
    return path if path.is_absolute() else ROOT / path


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)


def run_smoke(gated: bool = False):
    seed_all(7)
    model = build_model({"smoke": True, "width": 8, "gated": gated})
    video, audio, mask = synthetic_batch(batch=2, size=64, samples=2560)
    pred = model(video, audio)
    loss = F.binary_cross_entropy_with_logits(pred["logits"], mask)
    loss.backward()
    out = {"kind": "synthetic engineering smoke test", "trained_on_real_data": False,
           "variant": "gated" if gated else "baseline", "shape": list(pred["logits"].shape),
           "loss": float(loss.detach()), "synthetic_iou": metric(pred["logits"].detach(), mask),
           "gate_mean": float(pred["gate"].detach().mean())}
    print(json.dumps(out, indent=2)); return out


def batch_records(rows, manifest, cfg):
    records = [load_record(row, manifest, int(cfg.get("image_size", 128)),
                           int(cfg.get("audio_sample_rate", 16000)), float(cfg.get("segment_seconds", 1.0))) for row in rows]
    return tuple(torch.stack(x) for x in zip(*records))


def split_rows(rows, cfg):
    if rows and all("split" in row for row in rows):
        return {key: [row for row in rows if row.get("split") == key] for key in ("train", "val", "test")}
    ids = sorted({str(row["video_id"]) for row in rows})
    if len(ids) < 3: raise ValueError("Need at least 3 distinct video_id values for video-level train/val/test splits.")
    rng = random.Random(int(cfg.get("seed", 7))); rng.shuffle(ids)
    ntest, nval = max(1, round(len(ids)*.15)), max(1, round(len(ids)*.15))
    test, val, train = set(ids[:ntest]), set(ids[ntest:ntest+nval]), set(ids[ntest+nval:])
    result = {"train": [], "val": [], "test": []}
    for row in rows:
        key = "test" if str(row["video_id"]) in test else "val" if str(row["video_id"]) in val else "train"
        result[key].append(row)
    return result


def save_report(run_id, report):
    folder = ROOT / "artifacts" / "reports"; folder.mkdir(parents=True, exist_ok=True)
    (folder / f"{run_id}.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    md = "# SoundTrace evaluation report\n\n" + "\n".join(f"- **{k}:** `{v}`" for k, v in report.items()) + "\n"
    (folder / f"{run_id}.md").write_text(md, encoding="utf-8")


def run_provenance(manifest: Path, config_path: str, seed: int) -> dict:
    import subprocess
    try:
        revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
                                  capture_output=True, text=True).stdout.strip()
        dirty = bool(subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, check=True,
                                    capture_output=True, text=True).stdout.strip())
    except (OSError, subprocess.CalledProcessError):
        revision, dirty = None, None
    return {"source_revision": revision, "python": __import__("sys").version.split()[0],
            "torch": torch.__version__, "numpy": np.__version__,
            "device": "cpu", "seed": seed,
            "source_tree_dirty": dirty, "config": Path(config_path).name,
            "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest()}


def split_fingerprint(splits: dict[str, list[dict]]) -> str:
    identities = {name: sorted({str(row["video_id"]) for row in members})
                  for name, members in splits.items()}
    payload = json.dumps(identities, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def train(cfg, config_path):
    if cfg.get("smoke"):
        run_smoke(bool(cfg.get("gated"))); return
    manifest = manifest_path(cfg); rows = read_manifest(manifest)
    status = validate_manifest(rows, manifest)
    if not status["valid"]: raise ValueError(json.dumps(status["errors"][:10]))
    splits = split_rows(rows, cfg)
    if not splits["train"] or not splits["val"] or not splits["test"]:
        raise ValueError("Train, val and test source-video partitions must all contain examples.")
    seed_all(int(cfg.get("seed", 7)))
    size = int(cfg.get("image_size", 128)); model_cfg = {**cfg, "width": int(cfg.get("width", 32))}
    model = build_model(model_cfg); optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg.get("learning_rate", 3e-4)))
    epochs = int(cfg.get("epochs", 10)); batch_size = int(cfg.get("batch_size", 2)); best = -1; history=[]; start=time.time()
    for epoch in range(epochs):
        model.train(); order = list(splits["train"]); random.shuffle(order); train_loss=[]
        for at in range(0, len(order), batch_size):
            x, a, y = batch_records(order[at:at+batch_size], manifest, cfg)
            optimizer.zero_grad(set_to_none=True); logits=model(x,a)["logits"]
            loss=F.binary_cross_entropy_with_logits(logits,y) + (1 - (2*(logits.sigmoid()*y).sum()+1)/(logits.sigmoid().sum()+y.sum()+1))
            if not torch.isfinite(loss): raise FloatingPointError("Non-finite training loss")
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step(); train_loss.append(float(loss.detach()))
        model.eval(); vals=[]
        with torch.no_grad():
            for at in range(0,len(splits["val"]),batch_size):
                x,a,y=batch_records(splits["val"][at:at+batch_size],manifest,cfg); out=model(x,a)["logits"]
                vals.append(metric(out,y))
        score=float(np.mean(vals)); history.append({"epoch":epoch+1,"train_loss":float(np.mean(train_loss)),"val_iou":score})
        print(json.dumps(history[-1]))
        if score > best:
            best=score; folder=ROOT/"artifacts"/"checkpoints"; folder.mkdir(parents=True,exist_ok=True)
            torch.save({"config":model_cfg,"model":model.state_dict(),"epoch":epoch+1,"val_iou":score},folder/"best.pt")
    ckpt=ROOT/"artifacts"/"checkpoints"/"best.pt"; state=torch.load(ckpt,map_location="cpu",weights_only=True)
    model.load_state_dict(state["model"]); model.eval(); scores=[]
    with torch.no_grad():
        for row in splits["test"]:
            x,a,y=batch_records([row],manifest,cfg); logits=model(x,a)["logits"]
            scores.append({"video_id":row["video_id"],"iou":metric(logits,y),"f1":f1_metric(logits,y)})
    run_id=uuid.uuid4().hex[:12]
    report={"evaluation_id":run_id,"kind":"real_data_experiment",**run_provenance(manifest,str(config_path),int(cfg.get("seed",7))),"variant":"gated" if model.gated else "baseline","split_fingerprint":split_fingerprint(splits),"split_video_counts":{k:len({str(r['video_id']) for r in v}) for k,v in splits.items()},"best_validation_iou":best,"test_mean_iou":float(np.mean([x["iou"] for x in scores])),"test_examples":scores,"history":history,"duration_seconds":time.time()-start,"checkpoint":str(ckpt.relative_to(ROOT))}
    save_report(run_id,report); print(json.dumps(report,indent=2))


def load_checkpoint(path):
    state=torch.load(path,map_location="cpu",weights_only=True); model=build_model(state["config"]); model.load_state_dict(state["model"]); model.eval(); return model,state


def main():
    parser=argparse.ArgumentParser(prog="soundtrace"); sub=parser.add_subparsers(dest="command",required=True)
    sub.add_parser("doctor"); sub.add_parser("dataset-check"); sub.add_parser("smoke")
    for name in ("build-manifest","preprocess","train","evaluate","robustness"):
        p=sub.add_parser(name); p.add_argument("--config",required=True)
        if name=="build-manifest":
            p.add_argument("--task",choices=("inventory","s4","ms3"),default="inventory")
            p.add_argument("--data-root")
            p.add_argument("--output")
        if name in {"evaluate","robustness"}: p.add_argument("--checkpoint",required=True)
    p=sub.add_parser("predict"); p.add_argument("--video",required=True); p.add_argument("--checkpoint")
    p=sub.add_parser("report"); p.add_argument("--evaluation-id",required=True)
    args=parser.parse_args(); data_root=Path(os.environ.get("SOUNDTRACE_DATA_DIR",ROOT/"data"/"raw")); artifacts=ROOT/"artifacts"
    if args.command=="doctor":
        import importlib.util
        print(json.dumps({"python":__import__("sys").version.split()[0],"torch":torch.__version__,"cuda":torch.cuda.is_available(),"ffmpeg":__import__("shutil").which("ffmpeg"),"dataset_root_exists":data_root.exists(),"optional":{x:bool(importlib.util.find_spec(x)) for x in ["fastapi","cv2","yaml"]}},indent=2))
    elif args.command=="dataset-check":
        from .data import discover
        d=discover(data_root); print(json.dumps({"root":str(data_root),"available":data_root.exists(),"video_count":len(d["videos"]),"image_count":len(d["masks"]),"note":"Image files are candidate annotations and require release-specific review."},indent=2))
    elif args.command=="smoke": run_smoke()
    elif args.command=="build-manifest":
        cfg=load_config(args.config); root=Path(args.data_root or cfg.get("data_root") or data_root)
        out=Path(args.output or cfg.get("output_manifest",f"data/manifests/{args.task}.jsonl")); out=out if out.is_absolute() else ROOT/out
        result=build_avsbench_manifest(root,out,args.task) if args.task in {"s4","ms3"} else build_manifest(root,out)
        print(json.dumps(result,indent=2))
        if result.get("valid") is False: raise SystemExit(2)
    elif args.command=="preprocess":
        cfg=load_config(args.config); path=manifest_path(cfg); rows=read_manifest(path); status=validate_manifest(rows,path); print(json.dumps(status,indent=2))
        if not status["valid"]: raise SystemExit(2)
    elif args.command=="train": train(load_config(args.config),args.config)
    elif args.command in {"evaluate","robustness"}:
        cfg=load_config(args.config); path=manifest_path(cfg); rows=read_manifest(path); status=validate_manifest(rows,path)
        if not status["valid"]: raise SystemExit(json.dumps(status,indent=2))
        splits=split_rows(rows,cfg)
        if not splits["test"]: raise SystemExit("The manifest has no held-out test examples; evaluation never scores training or validation rows.")
        test_rows=splits["test"]
        model,_=load_checkpoint(args.checkpoint); model.eval(); results=[]
        if args.command == "robustness":
            conditions = cfg.get("conditions")
            if conditions is None:
                conditions = [{"audio_noise_std": noise, "audio_video_shift_seconds": shift,
                               "visual_blur_sigma": blur}
                              for noise in cfg.get("audio_noise_std", [0.0])
                              for shift in cfg.get("audio_video_shift_seconds", [0.0])
                              for blur in cfg.get("visual_blur_sigma", [0.0])]
        else:
            conditions = [{}]
        if not conditions or any(not isinstance(item, dict) for item in conditions):
            raise SystemExit("Robustness config must contain at least one condition mapping")
        rng=torch.Generator().manual_seed(int(cfg.get("seed",7)))
        with torch.no_grad():
            for row in test_rows:
                x,a,y=batch_records([row],path,cfg)
                for condition in conditions:
                    shift=float(condition.get("audio_video_shift_seconds", 0.0))
                    noise=float(condition.get("audio_noise_std", 0.0))
                    attenuation=float(condition.get("audio_attenuation", 1.0))
                    dropout=float(condition.get("audio_dropout_probability", 0.0))
                    blur=float(condition.get("visual_blur_sigma", 0.0))
                    downsample=int(condition.get("visual_downsample_factor", 1))
                    occlusion=float(condition.get("visual_occlusion_fraction", 0.0))
                    if noise < 0 or attenuation < 0 or not 0 <= dropout <= 1:
                        raise ValueError("Audio noise/attenuation must be nonnegative and dropout must be in [0, 1]")
                    if blur < 0 or downsample < 1 or not 0 <= occlusion <= 1:
                        raise ValueError("Visual blur must be nonnegative, downsample factor at least 1, and occlusion in [0, 1]")
                    shifted=a
                    if shift:
                        from .data import audio_window, cached_audio, resolve_path
                        sample_rate=int(cfg.get("audio_sample_rate",16000)); waveform=cached_audio(str(resolve_path(row["video"],path)),sample_rate)
                        shifted=audio_window(waveform,float(row["timestamp"])+shift,sample_rate,float(cfg.get("segment_seconds",1)))[None]
                    audio=shifted*attenuation
                    if noise:
                        audio=audio+torch.randn(audio.shape,generator=rng)*noise
                    if dropout:
                        keep=(torch.rand(audio.shape,generator=rng)>=dropout).to(audio.dtype)
                        audio=audio*keep
                    visual=x
                    if blur>0:
                        radius=max(1,round(blur)); kernel=radius*2+1
                        visual=F.avg_pool2d(F.pad(visual,(radius,)*4,mode="replicate"),kernel,stride=1)
                    if downsample>1:
                        height,width=visual.shape[-2:]
                        visual=F.interpolate(F.interpolate(visual,size=(max(1,height//downsample),max(1,width//downsample)),mode="bilinear",align_corners=False),size=(height,width),mode="bilinear",align_corners=False)
                    if occlusion:
                        height,width=visual.shape[-2:]; side=max(1,round((height*width*occlusion)**0.5))
                        top=max(0,(height-side)//2); left=max(0,(width-side)//2)
                        visual=visual.clone(); visual[:,:,top:top+side,left:left+side]=0
                    started=time.perf_counter(); logits=model(visual,audio)["logits"]; elapsed=time.perf_counter()-started
                    results.append({"video_id":row["video_id"],**condition,"iou":metric(logits,y),"f1":f1_metric(logits,y),"latency_seconds":elapsed})
        run_id=uuid.uuid4().hex[:12]; report={"evaluation_id":run_id,"kind":args.command,**run_provenance(path,args.config,int(cfg.get("seed",7))),"checkpoint":Path(args.checkpoint).name,"variant":"gated" if model.gated else "baseline","split_fingerprint":split_fingerprint(splits),"split_video_counts":{k:len({str(r['video_id']) for r in v}) for k,v in splits.items()},"evaluated_split":"test","results":results,"mean_iou":float(np.mean([r["iou"] for r in results])),"mean_f1":float(np.mean([r["f1"] for r in results])),"mean_latency_seconds":float(np.mean([r["latency_seconds"] for r in results]))}; save_report(run_id,report); print(json.dumps(report,indent=2))
    elif args.command=="predict":
        from .data import decode_audio,audio_window
        import cv2
        path=Path(args.video); ckpt=args.checkpoint or os.environ.get("SOUNDTRACE_CHECKPOINT")
        if not ckpt or not Path(ckpt).is_file(): raise SystemExit("A trained --checkpoint is required.")
        cap=cv2.VideoCapture(str(path)); ok,frame=cap.read(); fps=cap.get(cv2.CAP_PROP_FPS) or 1; frames=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); cap.set(cv2.CAP_PROP_POS_FRAMES,max(0,frames//2)); ok,frame=cap.read(); cap.release()
        if not ok: raise SystemExit("Could not decode video frame.")
        state=torch.load(ckpt,map_location="cpu",weights_only=True); cfg=state["config"]; audio=audio_window(decode_audio(path,int(cfg.get("audio_sample_rate",16000))),frames/(2*fps),int(cfg.get("audio_sample_rate",16000)),float(cfg.get("segment_seconds",1)))
        rgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB); from PIL import Image
        arr=np.asarray(Image.fromarray(rgb).resize((int(cfg.get("image_size",128)),)*2),dtype=np.float32).copy()/255
        video=torch.from_numpy(arr).permute(2,0,1)[None]; model,_=load_checkpoint(ckpt)
        with torch.no_grad(): out=model(video,audio[None]); mask=(out["logits"].sigmoid()[0,0]>.5).byte().numpy()
        folder=ROOT/"artifacts"/"predictions"; folder.mkdir(parents=True,exist_ok=True); target=folder/f"{path.stem}_mask.png"; Image.fromarray(mask*255).save(target)
        print(json.dumps({"trained":True,"timestamp":frames/(2*fps),"mask":str(target),"gate_mean":float(out["gate"].mean())},indent=2))
    elif args.command=="report":
        path=artifacts/"reports"/f"{args.evaluation_id}.json"
        if not path.exists(): raise SystemExit(f"Evaluation report not found: {path}")
        print(path.read_text(encoding="utf-8"))

if __name__=="__main__": main()
