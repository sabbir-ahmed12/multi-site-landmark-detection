"""Tests for miatt.pipeline — Approach 1 mean baseline."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

from miatt.acpc import compute_acpc_transform, apply_transform
from miatt.pipeline import _landmarks_to_acpc, EvalResult


@pytest.fixture()
def acpc_landmarks() -> dict[str, np.ndarray]:
    """Landmarks already in ACPC space (AC at origin)."""
    return {
        "AC": np.array([0.0, 0.0, 0.0]),
        "PC": np.array([0.0, -27.0, 0.0]),
        "LE": np.array([-32.0, 18.0, -38.0]),
        "RE": np.array([32.0, 18.0, -38.0]),
        "genu": np.array([0.0, 20.0, 10.0]),
    }


@pytest.fixture()
def scanner_landmarks(acpc_landmarks: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """The same anatomy shifted to an arbitrary scanner origin."""
    offset = np.array([15.3, -8.7, 42.1])
    return {k: v + offset for k, v in acpc_landmarks.items()}


def test_landmarks_to_acpc_identity(acpc_landmarks: dict[str, np.ndarray]) -> None:
    """Landmarks already in ACPC space should be unchanged after transform."""
    result = _landmarks_to_acpc(acpc_landmarks)
    for label in acpc_landmarks:
        np.testing.assert_allclose(result[label], acpc_landmarks[label], atol=1e-5)


def test_landmarks_to_acpc_removes_translation(
    scanner_landmarks: dict[str, np.ndarray],
    acpc_landmarks: dict[str, np.ndarray],
) -> None:
    """Scanner-space landmarks should map back to the same ACPC coords."""
    result = _landmarks_to_acpc(scanner_landmarks)
    for label in acpc_landmarks:
        np.testing.assert_allclose(result[label], acpc_landmarks[label], atol=1e-5)


def test_landmarks_to_acpc_missing_key() -> None:
    lm = {"AC": np.zeros(3), "PC": np.array([0, -27, 0])}  # missing LE, RE
    with pytest.raises(ValueError, match="missing landmarks"):
        _landmarks_to_acpc(lm)


def test_landmarks_to_acpc_ac_at_origin(scanner_landmarks: dict[str, np.ndarray]) -> None:
    result = _landmarks_to_acpc(scanner_landmarks)
    np.testing.assert_allclose(result["AC"], [0.0, 0.0, 0.0], atol=1e-5)


def test_eval_result_str() -> None:
    r = EvalResult(site="siteA", n_train=80, n_eval=20, mean_error_mm=3.5,
                   per_subject_errors=[3.0, 4.0])
    s = str(r)
    assert "siteA" in s
    assert "3.50" in s
