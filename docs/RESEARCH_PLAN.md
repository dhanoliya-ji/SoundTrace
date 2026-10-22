# Research plan

Question: can reliability-aware fusion improve audiovisual source segmentation under modality corruption or temporal misalignment without substantially reducing clean-input quality?

Compare a concatenation baseline and gated fusion using identical source-video train/validation/test splits, optimization budget, seeds, preprocessing, and checkpoint-selection rule. Declare the clean-performance tolerance and primary robustness metric before test evaluation. Test audio additive noise/attenuation/dropout, visual blur/downsampling/occlusion, and symmetric temporal shifts. Include per-condition IoU/F-score, clean scores, uncertainty/gate diagnostics, latency, and failure examples. Select checkpoints on validation only. Run multiple seeds and report variability. Stop or revise the hypothesis if gating hurts clean quality or fails under held-out corruption.

The AVSBench dataset is not currently present. No real-data results, novelty claim, or successful benchmark result is asserted.

## Executable corruption protocol

`configs/corruption_tests.yaml` defines named held-out test conditions for clean input, audio noise, attenuation/dropout, positive and negative temporal shifts, visual blur, downsampling, occlusion, and a combined perturbation. The robustness command records each example-condition pair with IoU, F1, and inference latency. These transformations are SoundTrace stress tests and are not claimed to reproduce an official AVSBench evaluation protocol. Review and freeze the condition list before inspecting test results.
