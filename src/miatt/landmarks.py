"""Landmark representation and scoring utilities."""

from __future__ import annotations

import numpy as np


CANONICAL_LANDMARKS = ["AC", "PC", "LE", "RE"]


def mean_euclidean_error(
    predicted: dict[str, np.ndarray],
    ground_truth: dict[str, np.ndarray],
    labels: list[str] | None = None,
) -> float:
    """Return mean Euclidean error (mm) over *labels* between two landmark sets."""
    labels = labels or sorted(set(predicted) & set(ground_truth))
    if not labels:
        raise ValueError("No common landmarks to compare.")
    errors = [np.linalg.norm(predicted[l] - ground_truth[l]) for l in labels]
    return float(np.mean(errors))


def per_landmark_error(
    predicted: dict[str, np.ndarray],
    ground_truth: dict[str, np.ndarray],
) -> dict[str, float]:
    """Return per-landmark Euclidean error (mm)."""
    common = sorted(set(predicted) & set(ground_truth))
    return {l: float(np.linalg.norm(predicted[l] - ground_truth[l])) for l in common}


def aggregate_landmarks(landmark_sets: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    """Compute per-label mean across a list of landmark dicts."""
    all_labels: set[str] = set()
    for ls in landmark_sets:
        all_labels.update(ls.keys())
    result: dict[str, np.ndarray] = {}
    for label in all_labels:
        coords = [ls[label] for ls in landmark_sets if label in ls]
        if coords:
            result[label] = np.mean(coords, axis=0)
    return result
