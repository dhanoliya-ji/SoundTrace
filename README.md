# SoundTrace

**Reliable Audio-Visual Source Segmentation Under Modality Corruption**

SoundTrace is a local research prototype for predicting image regions associated with sound using aligned audio and video. It provides a compact PyTorch baseline, a reliability-gated fusion variant, dataset utilities, experiments CLI, FastAPI service, and a React dashboard. It is an engineering implementation, not evidence of real-world accuracy.

## Current status

- The AVSBench dataset and pretrained weights are not included. No real-data training or benchmark result is claimed.
- The PyTorch network is a trainable, spatial convolutional audiovisual segmentation model. Its image/audio encoders initialize randomly unless an explicitly configured model provides weights; it does not silently download anything.
- The gate is an adaptive learned fusion path, not a calibrated confidence score.
- The synthetic smoke path checks that data, model, metrics and checkpoint code execute. It does not validate source localization.
- Research references and dataset access links are recorded in `docs/RELATED_WORK.md` and `docs/DATASET_TERMS.md`. Confirm current papers, code and terms from their primary sources before running a study.

## Quick start (PowerShell)

```powershell
cd D:\SoundTrace
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev,api]"
python -m soundtrace.cli doctor
python -m soundtrace.cli smoke
python -m soundtrace.cli dataset-check
python -m soundtrace.cli build-manifest --config configs/smoke_test.yaml
```

To run the backend in PowerShell (from the repository root): `uvicorn soundtrace.api:app --app-dir src --reload`. To run the UI, open another PowerShell window, `cd frontend`, run `npm install`, then `npm run dev`. Set `VITE_API_BASE_URL=http://127.0.0.1:8000` in `frontend/.env.local` if needed. The dashboard is served by Vite at its displayed local URL. With no checkpoint it reports unavailability and will not pretend random weights are trained. FFmpeg must be installed and on `PATH` for raw-video audio inference/training; AVSBench manifests use prepared log-mel features.

## Dataset

Request AVSBench-object access using the official project instructions; review the dataset license, source media terms, and attribution requirements first. Do not commit acquired data or checkpoints. Set `SOUNDTRACE_DATA_DIR` to the acquired root, then build an S4 or MS3 manifest using its matching config. The adapter follows the official loader paths and emits processed-frame/log-mel records. Inspect the generated manifest and reconcile it with your release before training.

See `docs/DATASET.md`, `docs/DATASET_TERMS.md`, and `docs/RELATED_WORK.md`.

## Train and evaluate

```powershell
python -m soundtrace.cli train --config configs/baseline.yaml
python -m soundtrace.cli train --config configs/gated_fusion.yaml
python -m soundtrace.cli evaluate --checkpoint artifacts/checkpoints/best.pt --config configs/baseline.yaml
python -m soundtrace.cli robustness --checkpoint artifacts/checkpoints/best.pt --config configs/corruption_tests.yaml
python -m soundtrace.cli study --config configs/s4_baseline.yaml --seeds 7,17,27
python -m soundtrace.cli predict --video .\example.mp4 --checkpoint artifacts/checkpoints/best.pt
```

`study` trains baseline and gated variants with matched settings and source-video splits for each seed, saves a separate checkpoint per run, and reports paired test-IoU deltas with their across-seed standard deviation. It requires a valid reviewed manifest and can take substantial time.

Training expects a reviewed JSONL manifest containing `mask`, `video_id`, and `timestamp`, plus either a `video` path or both `frame` and `audio_features` paths. An optional `split` field may provide `train`, `val`, and `test`; otherwise deterministic source-video splits are assigned and at least three distinct videos are required. Use `build-manifest` with `--task s4` or `--task ms3` for official folder layouts; inventory mode is not train-ready. On CPU use the smoke config; full-resolution or full-data work may exceed laptop resources. Evaluation writes observed metrics to `artifacts/`. Checkpoints are selected on validation only.

## API

`GET /api/v1/health`, `/dataset/status`, `/model/status`; `POST /predictions` accepts a video upload and queues background inference; `GET /predictions/{id}` returns persisted job status, while `GET /predictions/{id}/masks` returns sampled masks across the clip. Jobs persist on disk under `artifacts/jobs` (configurable with `SOUNDTRACE_JOB_DIR`). Uploads are size-limited and removed after processing. Inference samples frames at 2 FPS by default; set `SOUNDTRACE_INFERENCE_FPS` to change the sampling rate. Configure a trained checkpoint with `SOUNDTRACE_CHECKPOINT`.

## Design and research

The falsifiable hypothesis is that learned modality reliability gating retains clean-input quality while reducing degradation under controlled audio/video corruption or misalignment. Compare with identical source-video splits, seeds and compute; report metrics per condition and latency. Failure to improve robustness without meaningful clean regression falsifies the proposal. The gated model is compared against a concatenation baseline; no novelty claim is made.

Documents: `docs/ARCHITECTURE.md`, `docs/RESEARCH_PLAN.md`, `docs/EVALUATION_PROTOCOL.md`, `docs/EXPERIMENTS.md`, `docs/MODEL_CARD.md`, `docs/RESPONSIBLE_USE.md`, `docs/REPRODUCIBILITY.md`, `docs/PAPER_TEMPLATE.md`.

For current implementation status, verification limits, and the next steps requiring external data, see `docs/HANDOFF.md`.

## Limitations

The current data interface is a generic reviewed JSONL manifest, not a verified parser for every official AVSBench release. Both compact encoders initialize randomly; meaningful performance requires training on correctly aligned licensed examples. Prediction returns a sampled mask timeline across the clip and stores API job state on disk locally. Frame sampling can miss brief motion; increase `SOUNDTRACE_INFERENCE_FPS` when finer temporal coverage is needed. User media should be treated as sensitive.

## Attribution

AVSBench and Generalizable Audio-Visual Segmentation are external research works and repositories. See linked primary resources in related-work and terms docs. SoundTrace does not bundle their data, code or weights.
