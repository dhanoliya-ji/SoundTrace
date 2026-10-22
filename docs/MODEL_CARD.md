# Model card

**Intended use:** research prototyping of binary audio-visual sounding-region segmentation.

**Architecture:** compact spatial convolutional visual and waveform audio encoders, concatenation or learned channel-gated fusion, binary mask decoder.

**Training data:** a 100-video subset of AVATAR was used for a local pilot; the media, manifest, reports, and checkpoints are not bundled. The selected local pilot checkpoint is `artifacts/checkpoints/best.pt`; without that checkpoint, weights initialize randomly. The custom three-seed held-out pilot did not show a reliable gated-fusion advantage (mean gated-minus-baseline IoU -0.0107, sample standard deviation 0.0356). No official AVATAR or AVSBench benchmark result is available.

**Limitations and risks:** source localization is not source separation or identity recognition. Off-screen sources, occlusion, reverberation, ambiguous co-occurring sound, domain shift, privacy and annotation errors can cause failures. Gate values are not probabilities. Do not use for surveillance, attribution of speech, safety decisions, or claims about people. Do not treat synthetic smoke metrics as model quality.
