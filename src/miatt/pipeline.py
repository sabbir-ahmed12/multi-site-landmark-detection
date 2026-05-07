"""Top-level pipeline interface — selects and runs a detection approach."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np

from miatt.acpc import compute_acpc_transform, transform_landmarks
from miatt.io import iter_subjects, load_fcsv, save_fcsv
from miatt.landmarks import aggregate_landmarks, mean_euclidean_error, per_landmark_error

if TYPE_CHECKING:
    import SimpleITK as sitk


def _sitk():
    import SimpleITK  # noqa: PLC0415
    return SimpleITK


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


def run_heuristic_baseline(
    data_root: str | Path,
    site: str,
    output_root: str | Path,
    eval_fraction: float = 0.2,
) -> EvalResult:
    """Approach 3: posterior-guided local refinement around the population mean.

    For each subject the population mean ACPC landmark is refined for a subset
    of landmarks (WM corpus callosum, globus pallidus) by searching for the
    local posterior-weighted centroid in the subject's tissue posterior image.
    All other landmarks retain the population mean.

    The ACPC transform is always derived from the subject's own labeled
    AC/PC/LE/RE (training and evaluation subjects).  On unlabeled subjects
    there are no labels, so we fall back to the registration-derived transform
    from Approach 2 (computed internally).

    Args:
        data_root: Root of MIATTFINALEXAMDATA.
        site: Site identifier, e.g. "siteA".
        output_root: Directory for prediction fcsv files.
        eval_fraction: Fraction of labeled subjects held out for evaluation.

    Returns:
        EvalResult with per-subject and per-landmark ACPC-space error statistics.
    """
    from miatt.heuristic import predict_landmarks_heuristic  # noqa: PLC0415

    data_root = Path(data_root)
    output_root = Path(output_root)

    all_pairs: list[tuple[Path, Path]] = [
        (subject_dir, fcsv_path)
        for subject_dir, fcsv_path in iter_subjects(data_root, site, labeled=True)
        if fcsv_path is not None
    ]
    if not all_pairs:
        raise RuntimeError(f"No labeled subjects found for {site}")

    n_eval = max(1, int(len(all_pairs) * eval_fraction)) if eval_fraction > 0 else 0
    eval_pairs = all_pairs[:n_eval]
    train_pairs = all_pairs[n_eval:]

    # --- 1. Compute population mean in ACPC space (same as Approach 1) ------
    acpc_train: list[dict[str, np.ndarray]] = []
    for subject_dir, fcsv_path in train_pairs:
        lm = load_fcsv(fcsv_path)
        try:
            acpc_train.append(_landmarks_to_acpc(lm))
        except ValueError:
            continue

    if not acpc_train:
        raise RuntimeError(f"No usable training subjects for {site}")

    mean_acpc_lm = aggregate_landmarks(acpc_train)

    # --- 2. Evaluate on held-out subjects ------------------------------------
    per_subject: list[float] = []
    per_lm_errors: list[dict[str, float]] = []

    for subject_dir, fcsv_path in eval_pairs:
        true_scanner_lm = load_fcsv(fcsv_path)
        try:
            T = compute_acpc_transform(
                true_scanner_lm["AC"],
                true_scanner_lm["PC"],
                true_scanner_lm["LE"],
                true_scanner_lm["RE"],
            )
            true_acpc = transform_landmarks(T, true_scanner_lm)
        except (KeyError, ValueError):
            continue

        predicted_acpc = predict_landmarks_heuristic(
            subject_dir, site, T, mean_acpc_lm
        )
        per_subject.append(mean_euclidean_error(predicted_acpc, true_acpc))
        per_lm_errors.append(per_landmark_error(predicted_acpc, true_acpc))

    all_labels = sorted(mean_acpc_lm.keys())
    per_lm_mean: dict[str, float] = {}
    for label in all_labels:
        vals = [e[label] for e in per_lm_errors if label in e]
        if vals:
            per_lm_mean[label] = float(np.mean(vals))

    result = EvalResult(
        site=site,
        n_train=len(train_pairs),
        n_eval=n_eval,
        mean_error_mm=float(np.mean(per_subject)) if per_subject else float("nan"),
        per_landmark_mean_mm=per_lm_mean,
        per_subject_errors=per_subject,
    )

    # --- 3. Write predictions for unlabeled subjects -------------------------
    # For unlabeled subjects: derive ACPC transform via Approach 2 registration
    from miatt.registration import build_acpc_template, propagate_landmarks, register_to_template  # noqa: PLC0415

    sitk = _sitk()
    cache_dir = output_root.parent / "cache"
    template, _ = build_acpc_template(train_pairs, site, cache_dir)

    for subject_dir, _ in iter_subjects(data_root, site, labeled=False):
        t1_path = subject_dir / f"t1_{site}.nii.gz"
        if t1_path.exists():
            t1 = sitk.ReadImage(str(t1_path))
            tx_sitk = register_to_template(t1, template)
            # Derive a 4×4 approximation from the registration transform
            # by mapping the predicted AC/PC/LE/RE back to scanner space
            pred_scanner = propagate_landmarks(tx_sitk, mean_acpc_lm)
            try:
                T_reg = compute_acpc_transform(
                    pred_scanner["AC"],
                    pred_scanner["PC"],
                    pred_scanner["LE"],
                    pred_scanner["RE"],
                )
            except (KeyError, ValueError):
                T_reg = None
        else:
            T_reg = None

        if T_reg is not None:
            predicted_acpc = predict_landmarks_heuristic(
                subject_dir, site, T_reg, mean_acpc_lm
            )
        else:
            predicted_acpc = mean_acpc_lm

        out_path = (
            output_root / f"{site}_unlabeled" / subject_dir.name / "BCD_ACPC_Landmarks.fcsv"
        )
        save_fcsv(predicted_acpc, out_path)

    return result


def run_registration_baseline(
    data_root: str | Path,
    site: str,
    output_root: str | Path,
    cache_dir: str | Path = Path("cache"),
    eval_fraction: float = 0.2,
) -> EvalResult:
    """Approach 2: rigid registration to within-site ACPC template.

    For each site:
      1. Build a mean ACPC T1 template from training subjects (cached to disk).
      2. For each eval/unlabeled subject, register the ACPC template (fixed) to
         the subject T1 (moving) using rigid Euler3DTransform + Mattes MI.
      3. Apply the registration transform (ACPC → scanner) to mean ACPC landmarks
         → predicted scanner-space coordinates.
      4. ACPC-align the predicted scanner landmarks using the predicted
         AC/PC/LE/RE → predicted ACPC coordinates.
      5. Evaluate against true ACPC landmarks (same metric as Approach 1).

    Expected behaviour:
      The registration recovers each subject's ACPC ↔ scanner rigid mapping from
      image content alone, without requiring labeled landmarks.  In ACPC space the
      error is expected to be similar to Approach 1 (anatomical variation dominates)
      because a rigid transform cannot resolve within-ACPC shape differences.  The
      practical gain is a principled per-subject ACPC transform for unlabeled
      subjects.

    Args:
        data_root: Root of MIATTFINALEXAMDATA.
        site: Site identifier, e.g. "siteA".
        output_root: Directory for prediction fcsv files.
        cache_dir: Directory for caching the ACPC template image (large NIfTI).
        eval_fraction: Fraction of labeled subjects held out for evaluation.

    Returns:
        EvalResult with per-subject and per-landmark ACPC-space error statistics.
    """
    from miatt.registration import (  # noqa: PLC0415
        build_acpc_template,
        propagate_landmarks,
        register_to_template,
    )

    data_root = Path(data_root)
    output_root = Path(output_root)
    cache_dir = Path(cache_dir)
    sitk = _sitk()

    # --- 1. Collect all labeled subject paths ---------------------------------
    all_pairs: list[tuple[Path, Path]] = [
        (subject_dir, fcsv_path)
        for subject_dir, fcsv_path in iter_subjects(data_root, site, labeled=True)
        if fcsv_path is not None
    ]
    if not all_pairs:
        raise RuntimeError(f"No labeled subjects found for {site}")

    # --- 2. Train / eval split (deterministic: first eval_fraction are eval) --
    n_eval = max(1, int(len(all_pairs) * eval_fraction)) if eval_fraction > 0 else 0
    eval_pairs = all_pairs[:n_eval]
    train_pairs = all_pairs[n_eval:]

    # --- 3. Build ACPC template from training subjects ------------------------
    template, mean_acpc_lm = build_acpc_template(train_pairs, site, cache_dir)

    # --- 4. Evaluate on held-out subjects -------------------------------------
    per_subject: list[float] = []
    per_lm_errors: list[dict[str, float]] = []

    for subject_dir, fcsv_path in eval_pairs:
        true_scanner_lm = load_fcsv(fcsv_path)
        try:
            true_acpc = _landmarks_to_acpc(true_scanner_lm)
        except ValueError:
            continue

        t1_path = subject_dir / f"t1_{site}.nii.gz"
        if not t1_path.exists():
            continue

        t1 = sitk.ReadImage(str(t1_path))
        tx = register_to_template(t1, template)
        predicted_scanner = propagate_landmarks(tx, mean_acpc_lm)

        try:
            predicted_acpc = _landmarks_to_acpc(predicted_scanner)
        except ValueError:
            continue

        per_subject.append(mean_euclidean_error(predicted_acpc, true_acpc))
        per_lm_errors.append(per_landmark_error(predicted_acpc, true_acpc))

    all_labels = sorted(mean_acpc_lm.keys())
    per_lm_mean: dict[str, float] = {}
    for label in all_labels:
        vals = [e[label] for e in per_lm_errors if label in e]
        if vals:
            per_lm_mean[label] = float(np.mean(vals))

    result = EvalResult(
        site=site,
        n_train=len(train_pairs),
        n_eval=n_eval,
        mean_error_mm=float(np.mean(per_subject)) if per_subject else float("nan"),
        per_landmark_mean_mm=per_lm_mean,
        per_subject_errors=per_subject,
    )

    # --- 5. Write predictions for unlabeled subjects --------------------------
    for subject_dir, _ in iter_subjects(data_root, site, labeled=False):
        t1_path = subject_dir / f"t1_{site}.nii.gz"
        if not t1_path.exists():
            predicted_acpc = mean_acpc_lm
        else:
            t1 = sitk.ReadImage(str(t1_path))
            tx = register_to_template(t1, template)
            predicted_scanner = propagate_landmarks(tx, mean_acpc_lm)
            try:
                predicted_acpc = _landmarks_to_acpc(predicted_scanner)
            except ValueError:
                predicted_acpc = mean_acpc_lm  # fallback: population mean

        out_path = (
            output_root / f"{site}_unlabeled" / subject_dir.name / "BCD_ACPC_Landmarks.fcsv"
        )
        save_fcsv(predicted_acpc, out_path)

    return result
