# SoundTrace project status

## Built

- Compact audiovisual segmentation baseline and learned gated-fusion variant.
- Local FastAPI service and React dashboard for model status, video uploads, prediction overlays, and saved experiment reports.
- Reviewed-manifest training with source-video splits, validation checkpoint selection, and separate held-out test reporting.
- AVSBench S4/MS3 manifest adapter that follows the official directory layouts and loads the prepared five-clip log-mel features.
- AVATAR alternative dataset adapter that rasterizes its public per-frame polygons and pairs annotated frames with the video/audio clips using a deterministic video-level split.
- Named robustness conditions for audio noise, attenuation/dropout, temporal shifts, visual blur/downsampling/occlusion, and combined corruptions.
- Paired multi-seed study runner that retains each variant checkpoint and summarizes test-IoU deltas.
- Report provenance for source revision, environment versions, seed, manifest hash, and source-video split fingerprint.

## Current evidence

- 14 Python tests pass, Ruff passes, the frontend production build passes, the FastAPI app imports, and the CLI exposes the paired-study command.
- A one-seed end-to-end paired training run succeeded on generated synthetic files. Its metrics are not AVSBench evidence; generated reports and checkpoints were removed afterward.
- The AVSBench adapter was exercised against generated fixtures and checked against the official public S4/MS3 loader paths, but not against an acquired release.
- AVATAR's published schema and archive layout were inspected; the local downloads are in progress. AVATAR is not an AVSBench benchmark substitute and has no official training split in the current archive.
- No AVSBench dataset, trained real-data checkpoint, or benchmark result is present. Synthetic smoke output is only an engineering check.
- The system FFmpeg binary is absent; SoundTrace now depends on `imageio-ffmpeg` and uses its bundled decoder as a fallback for raw-video audio.

## Next work

1. Request access to AVSBench-object from its maintainers and obtain the authorized release. The official request requires applicant details and is not something this workspace can complete on the user's behalf.
2. Build and inspect the S4/MS3 manifest against that exact release and annotations.
3. Run the paired study command with the reviewed manifest and multiple seeds.
4. Run the robustness configuration on each retained checkpoint and review the corruption results before making any robustness claim.

## Important limits

SoundTrace uses compact waveform/log-mel encoders and simplified fusion, not the official AVSBench VGGish/TPAVI pipeline. Its outputs are not official benchmark results. Prediction jobs and masks persist under `artifacts/jobs`; the API processes a configurable sampled-frame timeline (2 FPS by default). It still uses in-process background execution, so a process restart during active inference does not resume that work.
