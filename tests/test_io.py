"""Tests for miatt.io — fcsv round-trip and subject iteration."""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from miatt.io import load_fcsv, save_fcsv


@pytest.fixture()
def sample_landmarks() -> dict[str, np.ndarray]:
    return {
        "AC": np.array([0.0, 0.0, 0.0]),
        "PC": np.array([0.0, -25.0, 0.0]),
        "LE": np.array([-30.0, 20.0, -40.0]),
        "RE": np.array([30.0, 20.0, -40.0]),
    }


def test_fcsv_round_trip(sample_landmarks: dict[str, np.ndarray]) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        fcsv_path = Path(tmp) / "landmarks.fcsv"
        save_fcsv(sample_landmarks, fcsv_path)
        loaded = load_fcsv(fcsv_path)

    assert set(loaded.keys()) == set(sample_landmarks.keys())
    for label, xyz in sample_landmarks.items():
        np.testing.assert_allclose(loaded[label], xyz, atol=1e-5)


def test_save_fcsv_creates_parent_dirs(sample_landmarks: dict[str, np.ndarray]) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        nested = Path(tmp) / "a" / "b" / "landmarks.fcsv"
        save_fcsv(sample_landmarks, nested)
        assert nested.exists()
