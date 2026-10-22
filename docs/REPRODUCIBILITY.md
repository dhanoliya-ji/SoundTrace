# Reproducibility

Record Python/PyTorch versions, device, seed, config, source revision, dataset provenance, split manifest hash, transforms, checkpoint provenance, run duration, and actual outputs. Fix seeds for Python, NumPy, and Torch; GPU kernels may remain nondeterministic unless deterministic mode is enabled. Use the same source-video partitions and validation-only selection for model comparisons. Smoke checks are CPU-friendly engineering checks, not quality evidence.
