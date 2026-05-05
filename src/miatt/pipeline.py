"""Top-level pipeline interface — selects and runs a detection approach."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np

from miatt.acpc import compute_acpc_transform, transform_landmarks
from miatt.io import iter_subjects, load_fcsv, save_fcsv
from miatt.landmarks import aggregate_landmarks, mean_euclidean_error, per_landmark_error


ApproachName = Literal["mean", "registration", "heuristic", "cnn"]

_ACPC_REQUIRED = frozenset({"AC", "PC", "LE", "RE"})


@dataclass
class EvalResult:
    site: str
    n_train: int
    n_eval: int
    mean_error_mm: float
    per_landmark_mean_mm: dict[str, float] = field(default_factory=dict)
    per_subject_errors: list[float] = field(default_factory=list)

    def __str__(self) -> str:
        return (
            f"{self.site}: train={self.n_train} eval={self.n_eval} "
            f"mean_err={self.mean_error_mm:.2f}mm "
            f"(std={np.std(self.per_subject_errors):.2f}mm)"
        )


def _landmarks_to_acpc(landmarks: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Return a copy of *landmarks* transformed to ACPC space.

    Uses the subject's own AC, PC, LE, RE to compute the rigid ACPC transform.
    Raises ValueError if any required key is missing.
    """
    missing = _ACPC_REQUIRED - landmarks.keys()
    if missing:
        raise ValueError(f"Cannot compute ACPC transform — missing landmarks: {missing}")
    T = compute_acpc_transform(
        landmarks["AC"], landmarks["PC"], landmarks["LE"], landmarks["RE"]
    )
    return transform_landmarks(T, landmarks)


def run_mean_baseline(
    data_root: str | Path,
    site: str,
    output_root: str | Path,
    eval_fraction: float = 0.2,
) -> EvalResult:
    """Approach 1: per-site mean landmark position in ACPC space.

    For each site:
      1. Transform every labeled subject's landmarks into ACPC space (using
         that subject's own AC/PC/LE/RE).
      2. Compute the element-wise mean across training subjects.
      3. Predict that mean set for every unlabeled subject.

    The mean is always in ACPC space (AC at origin), satisfying FR-3 regardless
    of the original scanner coordinate system.

    Why this fails for sites B–F: each subject's brain is at a different
    physical position, so the mean ACPC anatomy may not match the actual anatomy
    of any individual subject. However, the *ACPC-space* error is purely
    anatomical variation — not a coordinate-system artefact.

    Args:
        data_root: Root of MIATTFINALEXAMDATA.
        site: Site identifier, e.g. "siteA".
        output_root: Directory where predictions/site*_unlabeled/ will be written.
        eval_fraction: Fraction of labeled subjects held out for evaluation.

    Returns:
        EvalResult with per-subject and per-landmark error statistics.
    """
    data_root = Path(data_root)
    output_root = Path(output_root)

    # --- 1. Load and transform all labeled landmarks to ACPC space ----------
    acpc_by_subject: list[dict[str, np.ndarray]] = []
    for subject_dir, fcsv_path in iter_subjects(data_root, site, labeled=True):
        if fcsv_path is None:
            continue
        lm = load_fcsv(fcsv_path)
        try:
            acpc_by_subject.append(_landmarks_to_acpc(lm))
        except ValueError:
            continue  # skip subjects missing ACPC landmarks

    if not acpc_by_subject:
        raise RuntimeError(f"No usable labeled subjects found for {site}")

    # --- 2. Train / eval split (deterministic: first eval_fraction are eval) -
    n_eval = max(1, int(len(acpc_by_subject) * eval_fraction)) if eval_fraction > 0 else 0
    eval_set = acpc_by_subject[:n_eval]
    train_set = acpc_by_subject[n_eval:]

    # --- 3. Compute per-site mean from training subjects --------------------
    mean_lm = aggregate_landmarks(train_set)

    # --- 4. Evaluate on held-out subjects ------------------------------------
    per_subject: list[float] = []
    per_lm_errors: list[dict[str, float]] = []
    for lm in eval_set:
        per_subject.append(mean_euclidean_error(mean_lm, lm))
        per_lm_errors.append(per_landmark_error(mean_lm, lm))

    # Aggregate per-landmark errors
    all_labels = sorted(mean_lm.keys())
    per_lm_mean: dict[str, float] = {}
    for label in all_labels:
        vals = [e[label] for e in per_lm_errors if label in e]
        if vals:
            per_lm_mean[label] = float(np.mean(vals))

    result = EvalResult(
        site=site,
        n_train=len(train_set),
        n_eval=n_eval,
        mean_error_mm=float(np.mean(per_subject)) if per_subject else float("nan"),
        per_landmark_mean_mm=per_lm_mean,
        per_subject_errors=per_subject,
    )

    # --- 5. Write predictions for unlabeled subjects --------------------------
    for subject_dir, _ in iter_subjects(data_root, site, labeled=False):
        out_path = (
            output_root / f"{site}_unlabeled" / subject_dir.name / "BCD_ACPC_Landmarks.fcsv"
        )
        save_fcsv(mean_lm, out_path)

    return result
