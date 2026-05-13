"""Tests for miatt.landmarks — error metrics and aggregation."""

import numpy as np
import pytest

from miatt.landmarks import mean_euclidean_error, per_landmark_error, aggregate_landmarks


def test_mean_euclidean_error_identical() -> None:
    lm = {"AC": np.array([1.0, 2.0, 3.0]), "PC": np.array([4.0, 5.0, 6.0])}
    assert mean_euclidean_error(lm, lm) == pytest.approx(0.0)


def test_mean_euclidean_error_known() -> None:
    pred = {"AC": np.array([1.0, 0.0, 0.0])}
    gt = {"AC": np.array([0.0, 0.0, 0.0])}
    assert mean_euclidean_error(pred, gt) == pytest.approx(1.0)


def test_mean_euclidean_error_no_common_labels() -> None:
    pred = {"AC": np.array([0.0, 0.0, 0.0])}
    gt = {"PC": np.array([0.0, 0.0, 0.0])}
    with pytest.raises(ValueError):
        mean_euclidean_error(pred, gt)


def test_per_landmark_error() -> None:
    pred = {"AC": np.array([3.0, 0.0, 0.0]), "PC": np.array([0.0, 4.0, 0.0])}
    gt = {"AC": np.array([0.0, 0.0, 0.0]), "PC": np.array([0.0, 0.0, 0.0])}
    errors = per_landmark_error(pred, gt)
    assert errors["AC"] == pytest.approx(3.0)
    assert errors["PC"] == pytest.approx(4.0)


def test_aggregate_landmarks() -> None:
    sets = [
        {"AC": np.array([0.0, 0.0, 0.0])},
        {"AC": np.array([2.0, 2.0, 2.0])},
    ]
    result = aggregate_landmarks(sets)
    np.testing.assert_allclose(result["AC"], [1.0, 1.0, 1.0])
