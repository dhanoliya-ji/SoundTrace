# SoundTrace research notes

One document for related work, hypothesis, data terms, protocol, model card and responsible use.

## Question and hypothesis

*Can reliability-gated audio-visual fusion improve sounding-object segmentation when audio or video is degraded or
misaligned, without hurting clean-input quality?*

**Hypothesis.** On fixed source-video splits, the gated model beats the concatenation baseline under the predeclared
corruptions in `configs/corruptions.yaml` while staying within 1 IoU point of it on clean input.
**Falsified if** the gated model fails to improve robustness, loses more than that on clean input, or the effect flips sign
across seeds. No novelty is claimed: adaptive fusion and modality reliability are established research themes.

## Related work

- **AVSBench** ([paper](https://arxiv.org/abs/2207.05042), [code](https://github.com/OpenNLPLab/AVSBench)) defines pixel-level
  sounding-object segmentation: S4 (single source, one labelled training frame per video), MS3 (multi-source) and AVSS
  (semantic). Its baseline uses TPAVI fusion and VGGish audio. SoundTrace implements its S4/MS3 folder layout, IoU
  empty-mask convention and max-F-measure (β²=0.3), but **not** TPAVI/VGGish, so it is not a reproduction.
- **AVATAR** ([dataset](https://huggingface.co/datasets/mipal/AVATAR), ICCV 2025) gives video-centric instance polygons for
  single-sound, mixed, multi-entity and off-screen scenes. SoundTrace merges visible-source polygons into binary masks.
  Its official CIoU/AUC evaluator is not reproduced.
- **Generalizable AVS** ([code](https://github.com/GeWu-Lab/Generalizable-Audio-Visual-Segmentation), AAAI 2024) adapts SAM with
  audio prompts for unseen-class and cross-dataset settings. This shows that generalization and cross-modal adapters are
  already well studied.

## Data and terms

| Dataset | Access | Licence | Used here |
|---|---|---|---|
| AVATAR | public download (`video.zip` 3.8 GB, `metadata.zip` 1.6 GB) | CC BY 4.0; source-clip rights unverified | yes, 600-video subset |
| AVSBench S4/MS3 | request by email to maintainers | CC BY-NC 4.0 (research only) | adapter only, not acquired |

Put AVATAR's two zips in `data/raw/AVATAR/downloads/`, then run `build-manifest --task avatar --max-videos N`. The subset is
chosen deterministically by hashing the video ID, so it stays stable as N grows. AVATAR has no official split: SoundTrace
splits **source videos** 70/15/15 with a seed, so no frame or audio window of a video crosses splits. Media, masks,
manifests and checkpoints are git-ignored and must not be redistributed. Cite and attribute the AVATAR authors.

**Alignment.** For each annotated frame at time *t* = frame/fps, the audio is the 1 s window centred on *t* (zero
padded at clip edges; it starts at 0 when *t* ≤ 0). AVSBench uses its official per-second log-mel clip `audio_index`.

## Model

- Visual: ResNet18 up to `layer2` (stride 8, ImageNet weights from torchvision, stem and `layer1` frozen), 224×224 input.
- Audio: 64-bin log-mel (25 ms window, 10 ms hop at 16 kHz), followed by a 3-block CNN and global pooling.
- Baseline: concatenate the tiled audio vector with the visual map, then conv fusion and a 2× upsampling decoder.
- Gated: a per-channel sigmoid gate on the audio vector, computed from pooled visual features, audio features and
  their cosine agreement. The gate is a diagnostic, **not a calibrated confidence**.
- Loss: BCE plus soft Dice. AdamW with a cosine schedule, gradient clipping, AMP on CUDA, horizontal-flip augmentation,
  early stopping. Checkpoints are selected on validation IoU only.

## Evaluation protocol

All metrics are computed on the held-out test split at threshold 0.5: per-frame IoU, F1 and the AVSBench-style F-measure.
Robustness applies deterministic, seeded corruptions to the test inputs only (masks are never changed). The suite
covers audio noise, attenuation, muting, dropout, ±0.5 s and +2 s time shifts, blur, downsampling, occlusion and a
combined condition. Each report stores the git revision, manifest SHA-256, split fingerprint, seed, device and
per-example results under `artifacts/reports/`.

## Results

See the README's *Results* section; the numbers there are copied from the saved reports.

## Limitations and responsible use

- These are custom-split exploratory results on one dataset, not official AVATAR or AVSBench scores.
- Binary segmentation only (no AVSS semantics). Off-screen sources, reverberation and co-occurring sounds remain hard.
- Uploaded media may contain people and private speech. The API deletes uploads after inference and keeps only masks.
  Do not use the model to identify speakers, for surveillance or for consequential decisions.
