# Related work

## AVSBench

AVSBench establishes pixel-level sounding-object segmentation, with semi-supervised single-source S4 and fully supervised multi-source MS3; the official implementation also includes semantic AVSS. Its baseline uses an encoder-decoder, temporal pixel-wise audio-visual interaction (TPAVI), and an audio-visual correlation regularizer. The official object metadata and dataloader define five one-second samples: S4 train has one labeled frame per video, while MS3 and held-out S4 use five. SoundTrace now has an opt-in CSV/path adapter for S4/MS3, but its small waveform encoder and simple fusion are not a reproduction of TPAVI/VGGish. SoundTrace results must not be compared as official benchmark results without aligning preprocessing and evaluation exactly.

Primary sources: [AVSBench repository](https://github.com/OpenNLPLab/AVSBench), [official dataloader](https://github.com/OpenNLPLab/AVSBench/blob/main/avs_scripts/avs_ms3/dataloader.py), [ECCV 2022 paper](https://arxiv.org/abs/2207.05042), [project page](https://opennlplab.github.io/AVSBench/).

## AVATAR video-centric localization

AVATAR adds temporal, instance-level annotations for single-sound, mixed-sound, multi-entity, and off-screen scenes. Its publicly hosted release contains video clips and polygon instance masks, and its dataset card declares CC BY 4.0. SoundTrace can merge visible-source instance polygons into a binary mask and trains on a custom source-video split. This adaptation is useful for an auxiliary local experiment but does not reproduce AVATAR's CIoU/AUC benchmark evaluator or its official test protocol.

Primary sources: [AVATAR dataset card and files](https://huggingface.co/datasets/mipal/AVATAR), [project page](https://hahyeon610.github.io/Video-centric_Audio_Visual_Localization/), [ICCV 2025 paper](https://openaccess.thecvf.com/content/ICCV2025/papers/Choi_Whats_Making_That_Sound_Right_Now_Video-centric_Audio_Visual_Localization_ICCV_2025_paper.pdf).

## Generalizable audio-visual segmentation

The AAAI 2024 Generalizable Audio-Visual Segmentation work introduces semantic-aware audio prompts and a correlation adapter around a visual foundation model (SAM), and reports unseen-class and cross-dataset settings. Its V3 metadata creates seen/unseen and few-shot splits from AVSBench versions. This establishes that generalization and cross-modal adaptation are already studied; SoundTrace's gate is therefore not claimed as novel. SoundTrace does not reproduce GAVS.

Primary source: [GeWu-Lab repository](https://github.com/GeWu-Lab/Generalizable-Audio-Visual-Segmentation).

## Proposed experiment

Reliability-gated fusion is implemented as a learned per-channel audio modulation of spatial visual features. The baseline uses concatenation through the same decoder. This is an experimental comparison, not a novelty claim: adaptive fusion and modality reliability are established research themes. A meaningful contribution requires a literature review and controlled experiments demonstrating a useful robustness/clean-performance tradeoff.

## Hypothesis and falsification

Hypothesis: on fixed source-video partitions, learned gating will improve segmentation under predeclared audio/video corruption and temporal shifts while keeping clean-input performance within a predeclared tolerance of the concatenation baseline. Falsify if gated performance does not improve robustness, loses more than the tolerance on clean input, or fails to report useful behavior across seeds. No such measurements have yet been made.
