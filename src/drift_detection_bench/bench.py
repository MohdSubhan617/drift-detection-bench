from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Real

import numpy as np


@dataclass(frozen=True)
class TrialResult:
    trials: int
    detections: int
    rate: float


def mean_shift_score(reference: np.ndarray, current: np.ndarray) -> float:
    reference = np.asarray(reference, dtype=float)
    current = np.asarray(current, dtype=float)
    if reference.ndim != 1 or current.ndim != 1:
        raise ValueError("samples must be one-dimensional")
    if reference.size < 2 or current.size < 2:
        raise ValueError("each sample needs at least two observations")
    if not np.isfinite(reference).all() or not np.isfinite(current).all():
        raise ValueError("samples must contain only finite values")
    with np.errstate(over="ignore", invalid="ignore"):
        pooled = math.sqrt((float(reference.var(ddof=1)) + float(current.var(ddof=1))) / 2.0)
    if not math.isfinite(pooled):
        raise ValueError("sample spread must be finite")
    if pooled == 0:
        return 0.0 if float(reference.mean()) == float(current.mean()) else math.inf
    return abs(float(reference.mean()) - float(current.mean())) / pooled


def run_trials(
    *, seed: int, trials: int, n: int, shift: float, threshold: float = 0.5
) -> TrialResult:
    if (
        isinstance(seed, bool)
        or not isinstance(seed, int)
        or seed < 0
        or isinstance(trials, bool)
        or not isinstance(trials, int)
        or trials <= 0
        or isinstance(n, bool)
        or not isinstance(n, int)
        or n < 2
        or isinstance(shift, bool)
        or not isinstance(shift, Real)
        or not math.isfinite(shift)
        or isinstance(threshold, bool)
        or not isinstance(threshold, Real)
        or not math.isfinite(threshold)
        or threshold < 0
    ):
        raise ValueError("invalid trial configuration")
    rng = np.random.default_rng(seed)
    detections = 0
    for _ in range(trials):
        ref = rng.normal(0, 1, n)
        cur = rng.normal(shift, 1, n)
        detections += mean_shift_score(ref, cur) >= threshold
    return TrialResult(trials, detections, detections / trials)
