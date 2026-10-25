# Drift Detection Bench

A small, reproducible experiment for measuring a standardized mean-shift score on synthetic normal samples.

## What it does

`mean_shift_score(reference, current)` computes the absolute difference in sample means divided by the pooled sample standard deviation. `run_trials` draws independent reference and shifted normal samples from a seeded NumPy generator, classifies a trial as detected when its score is at least `threshold`, and returns the detection count and rate.

```python
import numpy as np
from drift_detection_bench.bench import mean_shift_score, run_trials

score = mean_shift_score(np.array([0.0, 1.0, 2.0]), np.array([1.0, 2.0, 3.0]))
result = run_trials(seed=42, trials=100, n=200, shift=0.5, threshold=0.5)
```

The score is an effect-size-like statistic, not a hypothesis test. The default threshold is not calibrated, the experiment does not report confidence intervals, and synthetic normal results do not establish production drift performance.

## Setup and checks

Requires Python 3.13 or newer.

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
python -m pip check
python -m pytest -q
python -m ruff check .
python -m ruff format --check .
python -m pip_audit --local --skip-editable --progress-spinner off --timeout 60
```

The package uses NumPy; no external data or service is required. Inputs to the score must be finite one-dimensional samples with at least two values. Trial count, sample size, seed, shift, and threshold are validated.
