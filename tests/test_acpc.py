"""Tests for miatt.acpc — ACPC alignment transform."""

import numpy as np
import pytest

from miatt.acpc import compute_acpc_transform, apply_transform, transform_landmarks


@pytest.fixture()
def standard_landmarks() -> dict[str, np.ndarray]:
    """Landmarks already in ACPC space — transform should be near-identity."""
    return {
        "AC": np.array([0.0, 0.0, 0.0]),
        "PC": np.array([0.0, -25.0, 0.0]),
        "LE": np.array([-30.0, 10.0, -35.0]),
        "RE": np.array([30.0, 10.0, -35.0]),
    }


def test_ac_lands_at_origin(standard_landmarks: dict[str, np.ndarray]) -> None:
    lm = standard_landmarks
    T = compute_acpc_transform(lm["AC"], lm["PC"], lm["LE"], lm["RE"])
    ac_aligned = apply_transform(T, lm["AC"]).squeeze()
    np.testing.assert_allclose(ac_aligned, [0.0, 0.0, 0.0], atol=1e-6)


def test_ac_pc_same_si_lr(standard_landmarks: dict[str, np.ndarray]) -> None:
    lm = standard_landmarks
    T = compute_acpc_transform(lm["AC"], lm["PC"], lm["LE"], lm["RE"])
    aligned = transform_landmarks(T, lm)
    # LR (index 0) and SI (index 2) must match between AC and PC
    np.testing.assert_allclose(aligned["AC"][0], aligned["PC"][0], atol=1e-5)
    np.testing.assert_allclose(aligned["AC"][2], aligned["PC"][2], atol=1e-5)


def test_eyes_on_common_si_plane(standard_landmarks: dict[str, np.ndarray]) -> None:
    lm = standard_landmarks
    T = compute_acpc_transform(lm["AC"], lm["PC"], lm["LE"], lm["RE"])
    aligned = transform_landmarks(T, lm)
    # SI coordinate (index 2) must match between LE and RE
    np.testing.assert_allclose(aligned["LE"][2], aligned["RE"][2], atol=1e-5)


def test_transform_landmarks_preserves_keys(standard_landmarks: dict[str, np.ndarray]) -> None:
    lm = standard_landmarks
    T = compute_acpc_transform(lm["AC"], lm["PC"], lm["LE"], lm["RE"])
    result = transform_landmarks(T, lm)
    assert set(result.keys()) == set(lm.keys())
