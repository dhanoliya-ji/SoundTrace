# Model card

**Intended use:** research prototyping of binary audio-visual sounding-region segmentation.

**Architecture:** compact spatial convolutional visual and waveform audio encoders, concatenation or learned channel-gated fusion, binary mask decoder.

**Training data:** none bundled. Weight initialization is random unless a locally trained checkpoint is supplied. No benchmark evaluation is available.

**Limitations and risks:** source localization is not source separation or identity recognition. Off-screen sources, occlusion, reverberation, ambiguous co-occurring sound, domain shift, privacy and annotation errors can cause failures. Gate values are not probabilities. Do not use for surveillance, attribution of speech, safety decisions, or claims about people. Do not treat synthetic smoke metrics as model quality.
