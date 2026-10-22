# SoundTrace project status

## Built

- Compact audiovisual segmentation baseline and learned gated-fusion variant.
- Local FastAPI service and React dashboard for model status, video uploads, prediction overlays, and saved experiment reports.
- Reviewed-manifest training with source-video splits, validation checkpoint selection, and separate held-out test reporting.
- AVSBench S4/MS3 manifest adapter that follows the official directory layouts and loads the prepared five-clip log-mel features.
- Named robustness conditions for audio noise, attenuation/dropout, temporal shifts, visual blur/downsampling/occlusion, and combined corruptions.
- Paired multi-seed study runner that retains each variant checkpoint and summarizes test-IoU deltas.
- Report provenance for source revision, environment versions, seed, manifest hash, and source-video split fingerprint.

## Current evidence

- 12 Python tests pass, Ruff passes, the frontend production build passes, the FastAPI app imports, and the CLI exposes the paired-study command.
- A one-seed end-to-end paired training run succeeded on generated synthetic files. Its metrics are not AVSBench evidence; generated reports and checkpoints were removed afterward.
- The AVSBench adapter was exercised against generated fixtures and checked against the official public S4/MS3 loader paths, but not against an acquired release.
- No AVSBench dataset, trained real-data checkpoint, or benchmark result is present. Synthetic smoke output is only an engineering check.
- FFmpeg is not installed in this workspace; raw-video audio decoding is unavailable until installed. The official AVSBench log-mel manifest path does not need FFmpeg.

## Next work

1. Request access to AVSBench-object from its maintainers and obtain the authorized release. The official request requires applicant details and is not something this workspace can complete on the user's behalf.
2. Build and inspect the S4/MS3 manifest against that exact release and annotations.
3. Run the paired study command with the reviewed manifest and multiple seeds.
4. Run the robustness configuration on each retained checkpoint and review the corruption results before making any robustness claim.

## Important limits

SoundTrace uses compact waveform/log-mel encoders and simplified fusion, not the official AVSBench VGGish/TPAVI pipeline. Its outputs are not official benchmark results. The API is local and stores jobs in process memory; it returns a sampled-frame mask rather than a complete video timeline.
