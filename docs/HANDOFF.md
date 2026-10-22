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

- At the previous implementation checkpoint, 18 Python tests passed, Ruff passed, and the frontend production build passed. In-process API health/status checks, checkpoint loading, and a real-clip prediction were verified. On 2026-10-22, the currently running API endpoint still reported `No trained checkpoint configured`; restart the API process to load current code and `artifacts/checkpoints/best.pt`.
- AVATAR's two public archives are downloaded under `data/raw/AVATAR/downloads/`; both ZIP integrity checks passed. The local dataset card declares CC BY 4.0. Check underlying source-media terms and attribute the dataset authors before sharing or using the media beyond the intended research use.
- A deterministic 100-video AVATAR pilot manifest contains 483 annotated frames, has no pairing errors, validates successfully, and decodes the source clip's audio. The source-video split is 70 train, 15 validation, and 15 test videos. Manifest: `data/manifests/avatar.jsonl` (ignored by Git).
- A paired CPU study completed for seeds 7, 17, and 27, with three epochs per model and the same 100-video manifest. Paired test-IoU deltas (gated minus baseline) were -0.0024, -0.0497, and +0.0200; mean -0.0107, sample standard deviation 0.0356. This small custom study shows no reliable gated-fusion advantage. Study report: `artifacts/reports/aacf3509d792.json` (local, ignored file).
- The matched robustness suite completed for baseline and gated checkpoints across all three seeds and 18 named conditions. Descriptively, mean gated-minus-baseline IoU delta averaged -0.0083 over ten audio conditions, +0.0320 over six visual-only conditions, and +0.0540 for the combined condition; variability was substantial across only 15 test videos per split, so these are exploratory results and do not establish robustness. Reports are local under `artifacts/reports/` with IDs `e4f9bc89b84e`, `7a099cbde9c1`, `5f09dd421aea`, `9e64de6c52dd`, `3f855c04dd14`, and `62c4056c35b2`. They record `source_tree_dirty: true` because the checkpoint-seed selection and dashboard/API status updates were in progress during this run.
- The seed-7 gated checkpoint, selected among the gated pilot runs by validation IoU, is installed at `artifacts/checkpoints/best.pt` for local API/dashboard inference. A real AVATAR clip upload completed through the prediction API and produced masks for 21 sampled frames. This remains a small pilot checkpoint, not a production model.
- The AVSBench adapter was exercised against generated fixtures and checked against the official public S4/MS3 loader paths, but not against an acquired AVSBench release. No AVSBench dataset or official benchmark result is present.
- SoundTrace uses compact waveform/log-mel encoders and simplified fusion, not the official AVSBench VGGish/TPAVI pipeline. AVATAR's custom metrics and split are not AVSBench metrics or an official AVATAR protocol.
- The system FFmpeg binary is absent; `imageio-ffmpeg` provides the bundled decoder fallback. Training ran on CPU (`torch 2.9.1+cpu`); CUDA was unavailable.

## Reproduce the local AVATAR pilot

On Windows from the project root:

```powershell
.venv\Scripts\python.exe -m soundtrace.cli build-manifest --config configs/avatar.yaml --task avatar --data-root data/raw/AVATAR --max-videos 100
.venv\Scripts\python.exe -m soundtrace.cli study --config configs/avatar.yaml --seeds 7
```

Run the paired pilot study with all three recorded seeds by changing the last option to `--seeds 7,17,27`. Run the matched corruption suite over the resulting pilot checkpoints with:

```powershell
Get-ChildItem artifacts\checkpoints -Filter "aacf3509d792-*.pt" | ForEach-Object {
  .venv\Scripts\python.exe -m soundtrace.cli robustness --config configs/avatar_corruption_tests.yaml --checkpoint $_.FullName
}
```

The subset is selected deterministically from video IDs. AVATAR does not provide an official train/validation/test split in this release, so SoundTrace makes source-video-level splits.

## Remaining work

- **Local runtime:** restart the API server so the dashboard can load the installed pilot checkpoint. The API process currently listening on port 8000 reports no checkpoint.
- **AVSBench benchmark:** no release has been acquired or validated in this workspace. The official project page links CSV IDs and annotated masks and describes processed video/audio access by email. After authorized data is available, validate its layout, inspect the generated S4/MS3 manifest, and run the official split and metric protocol. Adapter fixture tests alone are not a benchmark run.
- **Model evidence:** a larger, compute-appropriate study with a predeclared clean-performance tolerance and more seeds is needed before making a claim about the reliability gate. Existing AVATAR results are exploratory custom results and must not be described as official AVATAR or AVSBench scores.

## Important limits

Prediction jobs and masks persist under `artifacts/jobs`; the API processes a configurable sampled-frame timeline (2 FPS by default). It uses in-process background execution, so a process restart during active inference does not resume that work. Dataset archives, extracted media, masks, manifests, reports, and checkpoints are ignored by Git; do not add them to commits without reviewing their licenses and distribution rights.
