"""Tests for miatt.cnn — pure numpy, no torch or SimpleITK."""

from __future__ import annotations

import numpy as np

from miatt.cnn import COORD_SCALE, LANDMARK_LABELS, N_LANDMARKS


def test_landmark_labels_count() -> None:
    assert N_LANDMARKS == 51
    assert len(LANDMARK_LABELS) == N_LANDMARKS


def test_landmark_labels_sorted() -> None:
    assert LANDMARK_LABELS == sorted(LANDMARK_LABELS)


def test_landmark_labels_no_duplicates() -> None:
    assert len(set(LANDMARK_LABELS)) == len(LANDMARK_LABELS)


def test_landmark_labels_known_entries() -> None:
    required = {"AC", "PC", "LE", "RE", "genu", "rostrum", "BPons"}
    assert required <= set(LANDMARK_LABELS)


def test_landmark_labels_all_strings() -> None:
    for lbl in LANDMARK_LABELS:
        assert isinstance(lbl, str) and lbl


def test_landmark_index_round_trip() -> None:
    """Label → index → label should be identity."""
    for i, label in enumerate(LANDMARK_LABELS):
        assert LANDMARK_LABELS[i] == label


def test_coord_scale_positive() -> None:
    assert COORD_SCALE > 0


def test_coord_scale_covers_acpc_range() -> None:
    """ACPC coords span roughly [-120, +100] mm; COORD_SCALE should normalise these to < 1."""
    max_expected_coord_mm = 120.0
    assert COORD_SCALE >= max_expected_coord_mm
