# Architecture

Decoded RGB frames enter a spatial convolutional visual encoder. A timestamp-centered waveform window decoded by FFmpeg enters a compact 1D audio encoder. The audio feature is broadcast over the visual feature map. Baseline fusion concatenates them; gated fusion predicts per-channel audio contribution and uses it in the segmentation path. A convolutional decoder returns full-resolution binary logits. The gate is an internal learned coefficient and is not calibrated reliability.

The current frontend API returns one middle-frame mask. API job state is in process memory and disappears on restart. These are deliberate prototype boundaries; durable persistence, complete video timelines, and verified AVSBench release parsing remain follow-up engineering work.
