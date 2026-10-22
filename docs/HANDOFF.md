# SoundTrace project status

## Built

- Compact audiovisual segmentation baseline and learned gated-fusion variant.
- Local FastAPI service and React dashboard for model status, video uploads, prediction overlays, and saved experiment reports.
- Reviewed-manifest training with source-video splits, validation checkpoint selection, and separate held-out test reporting.
- Optional AVSBench S4/MS3 manifest adapter that uses release metadata instead of guessing annotations from matching filenames.
- Named robustness conditions for audio noise, attenuation/dropout, temporal shifts, visual blur/downsampling/occlusion, and combined corruptions.
- Report provenance for source revision, environment versions, seed, manifest hash, and source-video split fingerprint.

## Current evidence

- Before the latest report/history and evaluation changes, the workspace recorded 7 passing Python tests and a successful frontend production build.
- The latest changes have not been run through the test suite or frontend build yet.
- No AVSBench dataset, trained real-data checkpoint, or benchmark result is present. Synthetic smoke output is only an engineering check.
- FFmpeg must be installed for video-audio decoding; CPU training may be slow.

## Next work

1. Install FFmpeg and obtain authorized AVSBench data by following its official terms.
2. Validate the S4/MS3 manifest output against the exact downloaded release and annotations.
3. Run tests and frontend build for the latest changes, then fix any failures.
4. Run baseline and gated models with the same manifest, split fingerprint, seed, and optimization settings.
5. Review clean and corruption results across multiple seeds before making any robustness claim.

## Important limits

SoundTrace uses a compact waveform encoder and simplified fusion, not the official AVSBench VGGish/TPAVI pipeline. Its outputs are not official benchmark results. The API is local and stores jobs in process memory; it returns a sampled-frame mask rather than a complete video timeline.
