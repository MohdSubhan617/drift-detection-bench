# Architecture: drift-detection-bench

`bench.py` has two pure-style layers: `mean_shift_score` calculates a standardized difference for two input samples; `run_trials` owns seeded random generation, applies a fixed threshold to each generated score, and aggregates detections into `TrialResult`. There is no model, API, persistence layer, or statistical test implementation.

The score assumes independent numeric samples and summarizes mean separation relative to pooled sample spread. It is not a p-value or calibrated decision rule. See the README for the synthetic scenario and unsupported interpretations.
