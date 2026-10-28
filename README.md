# SoundTrace

**Reliable audio-visual source segmentation under modality corruption.** Given a video with sound, SoundTrace predicts
which pixels belong to the object making the sound. It compares a concatenation-fusion baseline with a
*reliability-gated* fusion model, both on clean inputs and under audio, visual and timing corruptions.

This is a research prototype: results come from a custom split of one dataset and are not official benchmark scores.

```
src/soundtrace/
  data.py    manifests (AVATAR, AVSBench S4/MS3), audio/video decoding, alignment, validation
  model.py   ResNet18 + log-mel CNN, baseline / gated fusion, checkpoint loading, video inference
  cli.py     metrics, corruptions, train / study / evaluate / robustness / predict
  api.py     FastAPI: status, background prediction jobs, reports; also serves the dashboard
frontend/    React + Vite dashboard (upload, mask overlay, timeline, experiment comparison)
configs/     train.yaml, corruptions.yaml, smoke.yaml
docs/RESEARCH.md   hypothesis, related work, data terms, protocol, model card, responsible use
```

## Setup (Windows PowerShell)

```powershell
python -m venv .venv; .\.venv\Scripts\Activate.ps1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126   # or /whl/cpu
pip install -e ".[dev]"
python -m soundtrace.cli doctor          # python / torch / CUDA / ffmpeg check
pytest -q; ruff check src tests          # no dataset, GPU or network needed
```

## Data

Download `video.zip` and `metadata.zip` from [AVATAR](https://huggingface.co/datasets/mipal/AVATAR) (CC BY 4.0) into
`data/raw/AVATAR/downloads/`, then:

```powershell
python -m soundtrace.cli build-manifest --config configs/train.yaml --task avatar --max-videos 600
```

For AVSBench (access by request), use `--task s4` or `--task ms3` with `--data-root <release>`, then train with
`--set manifest=data/manifests/s4.jsonl`.

## Train and evaluate

```powershell
python -m soundtrace.cli train --config configs/train.yaml                     # baseline
python -m soundtrace.cli train --config configs/train.yaml --set gated=true    # gated fusion
python -m soundtrace.cli study --config configs/train.yaml --seeds 7,17,27     # paired baseline vs gated
python -m soundtrace.cli evaluate   --config configs/train.yaml       --checkpoint artifacts/checkpoints/best.pt
python -m soundtrace.cli robustness --config configs/corruptions.yaml --checkpoint artifacts/checkpoints/best.pt
python -m soundtrace.cli predict --video clip.mp4                               # masks -> artifacts/predictions/
python -m soundtrace.cli report --evaluation-id <id>
python -m soundtrace.cli smoke                                                  # synthetic pipeline check only
```

Splits are made by source video. Checkpoints are selected on validation, and test metrics are reported once. Every
report records the git revision, manifest hash, split fingerprint, seed and device.

## Run the app

```powershell
cd frontend; npm install; npm run build; cd ..
uvicorn soundtrace.api:app --port 8000      # open http://127.0.0.1:8000
```

For UI development, run `npm run dev` in `frontend/`; it proxies `/api` to port 8000. The API uses
`artifacts/checkpoints/best.pt` (override with `SOUNDTRACE_CHECKPOINT`) and refuses to predict with random weights.
Uploads are size-limited and deleted after inference. The endpoints are: `GET /api/v1/health`, `/dataset/status`,
`/model/status`, `POST /predictions`, `GET /predictions/{id}[/masks|/metadata]`, `GET /evaluations[/{id}]`,
`GET /experiments[/{id}]` and `POST /experiments/{id}/compare`.

## Results

RESULTS_PLACEHOLDER

## Limitations

Binary masks only (no AVSS semantics). Single dataset, custom split. AVSBench support has only been tested on generated
fixtures. Gate values are not probabilities. See [docs/RESEARCH.md](docs/RESEARCH.md) for the hypothesis, data terms
and responsible-use notes.
