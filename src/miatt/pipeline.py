"""Top-level pipeline interface — selects and runs a detection approach."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np

from miatt.io import iter_subjects, load_fcsv, save_fcsv
from miatt.landmarks import aggregate_landmarks
from miatt.acpc import compute_acpc_transform, transform_landmarks


ApproachName = Literal["mean", "registration", "heuristic", "cnn"]


def run_mean_baseline(
    data_root: str | Path,
    site: str,
    output_root: str | Path,
) -> None:
    """Approach 1: predict landmarks using per-site mean of training landmarks.

    Expected to fail for sites B-F (different scanner orientations).
    """
    data_root = Path(data_root)
    output_root = Path(output_root)

    # Gather training landmarks
    training: list[dict[str, np.ndarray]] = []
    for subject_dir, fcsv_path in iter_subjects(data_root, site, labeled=True):
        if fcsv_path is not None:
            training.append(load_fcsv(fcsv_path))

    if not training:
        raise RuntimeError(f"No labeled subjects found for {site}")

    mean_landmarks = aggregate_landmarks(training)

    # Write predictions for unlabeled subjects
    for subject_dir, _ in iter_subjects(data_root, site, labeled=False):
        out_path = output_root / f"{site}_unlabeled" / subject_dir.name / "BCD_ACPC_Landmarks.fcsv"
        save_fcsv(mean_landmarks, out_path)
