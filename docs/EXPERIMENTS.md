# Experiment tracking

The CLI writes config path, seed, manifest hash, source revision, runtime details, split fingerprint, variant, checkpoint, duration, validation score, per-example test IoU, and aggregate metrics to JSON/Markdown reports. Training selects checkpoints using validation examples. Standalone evaluation and robustness commands score only the held-out test split and stop if it is empty; they never fall back to scoring all manifest rows. Manifest validation rejects mixed split metadata and source videos assigned to multiple splits.

The dashboard lists saved reports. The comparison endpoint pairs baseline and gated training runs only when manifest hash, seed, and exact source-video split fingerprint match. This does not establish equivalence of optimization settings or prove statistical significance; inspect each report before drawing conclusions. Keep artifacts outside Git. No real-data experiment has been run in this workspace.
