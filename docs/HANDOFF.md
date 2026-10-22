# SoundTrace project status

## Built

- Compact audiovisual segmentation baseline and learned gated-fusion variant.
- Local FastAPI service and React dashboard for model status, video uploads, prediction overlays, and saved experiment reports.
- Reviewed-manifest training with source-video splits, validation checkpoint selection, and separate held-out test reporting.
- AVSBench S4/MS3 manifest adapter for the documented directory layouts and prepared five-clip log-mel features.
- AVATAR adapter that rasterizes per-frame polygon annotations, pairs annotated frames with source clips, and extracts only a deterministic pilot subset when `--max-videos` is supplied.
- Named robustness conditions for audio noise, attenuation/dropout, temporal shifts, visual blur/downsampling/occlusion, and combined corruptions.
- Paired multi-seed study runner that retains each variant checkpoint and summarizes test-IoU deltas.
- Report provenance for source revision, environment versions, seed, manifest hash, and source-video split fingerprint.

## Current evidence

- 17 Python tests pass and Ruff passes. The frontend production build and API imports were verified earlier; the API's health/status endpoints and checkpoint loading were also checked after training.
- AVATAR's two public archives are downloaded under `data/raw/AVATAR/downloads/`; both ZIP integrity checks passed. The local dataset card declares CC BY 4.0. Check underlying source-media terms and attribute the dataset authors before sharing or using the media beyond the intended research use.
- A deterministic 100-video AVATAR pilot manifest contains 483 annotated frames, has no pairing errors, validates successfully, and decodes the source clip's audio. The source-video split is 70 train, 15 validation, and 15 test videos. Manifest: `data/manifests/avatar.jsonl` (ignored by Git).
- One paired CPU training run completed for seed 7, three epochs per variant. Baseline held-out mean IoU was 0.2811; gated fusion was 0.2787 (delta -0.0024). This one-seed, 15-test-video result is only a pipeline pilot; it does not show a reliable model advantage or benchmark performance. Reports: `artifacts/reports/52a6d5d501dd.json`, `artifacts/reports/9c4a0ec2959a.json`, and `artifacts/reports/992420a666d2.json` (local, ignored files).
- The gated checkpoint from that run is installed at `artifacts/checkpoints/best.pt` for local API/dashboard inference. A real AVATAR clip upload completed through the prediction API and produced masks for 21 sampled frames. It is a small pilot checkpoint, not a trained production model.
- The AVSBench adapter was exercised against generated fixtures and checked against the official public S4/MS3 loader paths, but not against an acquired AVSBench release. No AVSBench dataset or official benchmark result is present.
- SoundTrace uses compact waveform/log-mel encoders and simplified fusion, not the official AVSBench VGGish/TPAVI pipeline. AVATAR's custom metrics and split are not AVSBench metrics or an official AVATAR protocol.
- The system FFmpeg binary is absent; `imageio-ffmpeg` provides the bundled decoder fallback. Training ran on CPU (`torch 2.9.1+cpu`); CUDA was unavailable.

## Reproduce the local AVATAR pilot

On Windows from the project root:

```powershell
.venv\Scripts\python.exe -m soundtrace.cli build-manifest --config configs/avatar.yaml --task avatar --data-root data/raw/AVATAR --max-videos 100
.venv\Scripts\python.exe -m soundtrace.cli study --config configs/avatar.yaml --seeds 7
```

The subset is selected deterministically from video IDs. AVATAR does not provide an official train/validation/test split in this release, so SoundTrace makes source-video-level splits.

## Next work

1. Obtain authorized AVSBench-object access if official AVSBench evaluation is needed, then build and inspect its S4/MS3 manifest and run the matching protocol.
2. Treat the AVATAR pilot as an engineering baseline. Improve the training recipe, test with multiple seeds and a larger subset, and keep all custom metrics labeled with their split/protocol.
3. Run robustness evaluations only after reviewing each condition and checkpoint; do not make robustness claims from unrun configurations.

## Important limits

Prediction jobs and masks persist under `artifacts/jobs`; the API processes a configurable sampled-frame timeline (2 FPS by default). It uses in-process background execution, so a process restart during active inference does not resume that work. Dataset archives, extracted media, masks, manifests, reports, and checkpoints are ignored by Git; do not add them to commits without reviewing their licenses and distribution rights.
