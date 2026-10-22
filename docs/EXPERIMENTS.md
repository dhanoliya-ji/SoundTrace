# Experiment tracking

The CLI writes config path, seed, manifest hash, variant, checkpoint, duration, validation score, per-example test IoU, and aggregate metrics to JSON/Markdown reports. Checkpoint selection uses validation data; the held-out test split is only used after selection. Evaluation and robustness reports include their observed per-example corruption conditions. Save the source revision and runtime details alongside future runs before publication. Keep artifacts outside Git. No real-data experiment has been run in this workspace.
