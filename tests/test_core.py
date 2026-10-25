import numpy as np
import pytest

from drift_detection_bench.bench import mean_shift_score, run_trials


def test_equal_samples_have_zero_score():
    assert mean_shift_score(np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 3.0])) == 0


def test_shift_increases_score():
    ref = np.array([0.0, 0.0, 1.0, 1.0])
    cur = np.array([5.0, 5.0, 6.0, 6.0])
    assert mean_shift_score(ref, cur) > 5


def test_trial_results_are_reproducible():
    assert run_trials(seed=7, trials=20, n=50, shift=0.5) == run_trials(
        seed=7, trials=20, n=50, shift=0.5
    )


def test_invalid_configuration_rejected():
    with pytest.raises(ValueError):
        run_trials(seed=1, trials=0, n=10, shift=0)


def test_non_finite_and_multidimensional_samples_rejected():
    with pytest.raises(ValueError, match="finite"):
        mean_shift_score(np.array([0.0, np.nan]), np.array([0.0, 1.0]))
    with pytest.raises(ValueError, match="one-dimensional"):
        mean_shift_score(np.ones((2, 2)), np.ones(4))


@pytest.mark.parametrize(
    "shift, threshold", [(float("nan"), 0.5), (0.5, float("inf")), (True, 0.5)]
)
def test_non_finite_trial_configuration_rejected(shift, threshold):
    with pytest.raises(ValueError, match="invalid trial configuration"):
        run_trials(seed=1, trials=2, n=10, shift=shift, threshold=threshold)


def test_overflowed_sample_spread_is_rejected():
    with (
        np.errstate(over="ignore", invalid="ignore"),
        pytest.raises(ValueError, match="sample spread must be finite"),
    ):
        mean_shift_score(np.array([-1e308, 1e308]), np.array([-1e308, 1e308]))
