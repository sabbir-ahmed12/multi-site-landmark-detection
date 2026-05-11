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


ApproachName = Literal["mean", "registration", "heuristic", "cnn", "atlas"]

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


_ALL_SITES = ("siteA", "siteB", "siteC", "siteD", "siteE", "siteF")


def run_cnn_baseline(
    data_root: str | Path,
    site: str,
    output_root: str | Path,
    cache_dir: str | Path = Path("cache"),
    eval_fraction: float = 0.2,
    n_epochs: int = 100,
    batch_size: int = 8,
    device: str = "cuda",
) -> EvalResult:
    """Approach 4: 3D CNN direct coordinate regression.

    Trains a single cross-site model on all 6 sites combined (minus per-site
    eval holdouts), then evaluates on the requested site's holdout.  The
    model is cached as 'cache_dir/cnn_model.pt'; subsequent calls reuse it.

    Architecture: 5-block 3D CNN encoder + adaptive avg pool + FC head,
    ~5 M parameters.  Input: ACPC-resampled 2 mm T1 (1×96×101×101).
    Output: 51×3 ACPC-space landmark coordinates (mm).
    Loss: smooth L1.  Augmentation: intensity jitter + LR flip.

    Args:
        data_root: Root of MIATTFINALEXAMDATA.
        site: Site identifier to evaluate and write predictions for.
        output_root: Directory for prediction fcsv files.
        cache_dir: Directory for CNN model checkpoint and pre-processed volumes.
        eval_fraction: Fraction of labeled subjects held out per site.
        n_epochs: Training epochs (ignored if saved model exists).
        batch_size: Training batch size.
        device: Torch device string ('cuda' or 'cpu').

    Returns:
        EvalResult with per-subject and per-landmark ACPC-space error statistics
        for *site*.
    """
    from miatt.cnn import (  # noqa: PLC0415
        LANDMARK_LABELS,
        build_cnn_cache,
        predict_cnn,
        train_cnn,
    )

    data_root = Path(data_root)
    output_root = Path(output_root)
    cache_dir = Path(cache_dir)
    cnn_cache_dir = cache_dir / "cnn_volumes"
    model_path = cache_dir / "cnn_model.pt"
    sitk_mod = _sitk()

    # --- 1. Pre-compute ACPC T1 + landmark arrays for all sites ---------------
    all_train_files: list[Path] = []
    all_val_files: list[Path] = []
    site_splits: dict[str, dict[str, list[Path]]] = {}

    for s in _ALL_SITES:
        files = build_cnn_cache(data_root, s, cnn_cache_dir, labeled=True)
        n_val = max(1, int(len(files) * eval_fraction)) if eval_fraction > 0 else 0
        val_f = files[:n_val]
        train_f = files[n_val:]
        all_train_files.extend(train_f)
        all_val_files.extend(val_f)
        site_splits[s] = {"train": train_f, "val": val_f}

    # --- 2. Train (or load from checkpoint) -----------------------------------
    if not model_path.exists():
        print(f"  [CNN] training on {len(all_train_files)} subjects for {n_epochs} epochs …")
        train_cnn(
            all_train_files,
            all_val_files,
            model_path,
            n_epochs=n_epochs,
            batch_size=batch_size,
            device=device,
        )
    else:
        print(f"  [CNN] loaded cached model from {model_path}")

    # --- 3. Evaluate on the requested site's holdout --------------------------
    per_subject: list[float] = []
    per_lm_errors: list[dict[str, float]] = []
    n_train = len(site_splits[site]["train"])

    for npz_path in site_splits[site]["val"]:
        data = np.load(npz_path)
        vol = data["volume"]
        true_coords = data["landmarks"]  # (51, 3) already in ACPC

        true_acpc = {
            label: true_coords[i].astype(float)
            for i, label in enumerate(LANDMARK_LABELS)
        }
        pred_acpc = predict_cnn(model_path, vol, device=device)

        per_subject.append(mean_euclidean_error(pred_acpc, true_acpc))
        per_lm_errors.append(per_landmark_error(pred_acpc, true_acpc))

    all_labels = LANDMARK_LABELS
    per_lm_mean: dict[str, float] = {}
    for label in all_labels:
        vals = [e[label] for e in per_lm_errors if label in e]
        if vals:
            per_lm_mean[label] = float(np.mean(vals))

    result = EvalResult(
        site=site,
        n_train=n_train,
        n_eval=len(site_splits[site]["val"]),
        mean_error_mm=float(np.mean(per_subject)) if per_subject else float("nan"),
        per_landmark_mean_mm=per_lm_mean,
        per_subject_errors=per_subject,
    )

    # --- 4. Write predictions for unlabeled subjects --------------------------
    # Register each unlabeled T1 to the ACPC template, resample to ACPC space,
    # then run CNN inference.
    from miatt.preprocessing import normalize_intensity  # noqa: PLC0415
    from miatt.registration import (  # noqa: PLC0415
        _make_template_reference,
        build_acpc_template,
        register_to_template,
    )

    train_pairs_all: list[tuple[Path, Path]] = [
        (subject_dir, fcsv_path)
        for subject_dir, fcsv_path in iter_subjects(data_root, site, labeled=True)
        if fcsv_path is not None
    ]
    n_val_skip = max(1, int(len(train_pairs_all) * eval_fraction)) if eval_fraction > 0 else 0
    template, _ = build_acpc_template(train_pairs_all[n_val_skip:], site, cache_dir)

    for subject_dir, _ in iter_subjects(data_root, site, labeled=False):
        t1_path = subject_dir / f"t1_{site}.nii.gz"
        if not t1_path.exists():
            continue

        t1 = sitk_mod.ReadImage(str(t1_path))
        tx_sitk = register_to_template(t1, template)

        t1_norm = normalize_intensity(sitk_mod.Cast(t1, sitk_mod.sitkFloat32))
        resampler = sitk_mod.ResampleImageFilter()
        resampler.SetReferenceImage(_make_template_reference())
        resampler.SetTransform(tx_sitk)
        resampler.SetInterpolator(sitk_mod.sitkLinear)
        resampler.SetDefaultPixelValue(0.0)
        acpc_vol = resampler.Execute(t1_norm)
        vol_arr = sitk_mod.GetArrayFromImage(acpc_vol).astype(np.float32)

        pred_acpc = predict_cnn(model_path, vol_arr, device=device)

        out_path = (
            output_root / f"{site}_unlabeled" / subject_dir.name / "BCD_ACPC_Landmarks.fcsv"
        )
        save_fcsv(pred_acpc, out_path)

    return result


def run_atlas_baseline(
    data_root: str | Path,
    site: str,
    output_root: str | Path,
    eval_fraction: float = 0.2,
    n_atlases: int = 5,
    n_iterations: int = 200,
) -> EvalResult:
    """Approach 5: multi-atlas affine registration landmark detection.

    For each subject:
      1. Preprocess subject T1: float32, intensity normalise, 1 mm isotropic.
      2. Register each of N preprocessed siteA atlases (fixed) to the subject
         (moving) with affine + Mattes MI.  The transform maps atlas physical
         coords → subject physical coords.
      3. Apply each transform to atlas landmarks → N sets of predicted coords.
      4. Fuse by taking the per-landmark, per-axis median.
      5. ACPC-align predicted scanner landmarks via predicted AC/PC/LE/RE.
      6. Evaluate against true ACPC landmarks (same metric as other approaches).

    SiteA atlases are always drawn from the training partition of siteA to
    avoid data leakage when evaluating on siteA.

    Args:
        data_root:     Root of MIATTFINALEXAMDATA.
        site:          Site identifier, e.g. "siteA".
        output_root:   Directory where predictions/site*_unlabeled/ are written.
        eval_fraction: Fraction of labeled subjects held out for evaluation.
        n_atlases:     Number of siteA atlases to use.
        n_iterations:  Gradient-descent iterations per resolution level.

    Returns:
        EvalResult with per-subject and per-landmark ACPC-space error statistics.
    """
    from miatt.atlas import (  # noqa: PLC0415
        prep_for_registration,
        predict_landmarks_atlas,
        select_atlases,
    )

    data_root = Path(data_root)
    output_root = Path(output_root)
    sitk = _sitk()

    # --- 1. Determine eval split size (needed to choose atlas candidates) -----
    all_pairs: list[tuple[Path, Path]] = [
        (sd, fp)
        for sd, fp in iter_subjects(data_root, site, labeled=True)
        if fp is not None
    ]
    if not all_pairs:
        raise RuntimeError(f"No labeled subjects found for {site}")

    n_eval = max(1, int(len(all_pairs) * eval_fraction)) if eval_fraction > 0 else 0
    eval_pairs  = all_pairs[:n_eval]
    train_pairs = all_pairs[n_eval:]

    # siteA eval subjects are the first n_eval of siteA labeled; skip them
    # when selecting atlases so they cannot leak into their own evaluation.
    n_skip = n_eval if site == "siteA" else 0
    atlas_pairs = select_atlases(data_root, n=n_atlases, skip_first_n=n_skip)

    # --- 2. Load and preprocess atlas images + landmarks (cached in memory) ---
    print(f"  [atlas] loading {len(atlas_pairs)} atlas(es) …", flush=True)
    atlas_images:    list["sitk.Image"] = []
    atlas_landmarks: list[dict[str, np.ndarray]] = []

    for atlas_dir, atlas_fcsv in atlas_pairs:
        t1_path = atlas_dir / f"t1_siteA.nii.gz"
        if not t1_path.exists():
            continue
        img = sitk.ReadImage(str(t1_path))
        atlas_images.append(prep_for_registration(img))
        atlas_landmarks.append(load_fcsv(atlas_fcsv))

    if not atlas_images:
        raise RuntimeError("No atlas images could be loaded")

    # --- 3. Evaluate on labeled holdout subjects ------------------------------
    per_subject:   list[float] = []
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

        subj_img  = sitk.ReadImage(str(t1_path))
        subj_prep = prep_for_registration(subj_img)

        predicted_scanner = predict_landmarks_atlas(
            subj_prep, atlas_images, atlas_landmarks,
            n_iterations=n_iterations,
        )

        try:
            predicted_acpc = _landmarks_to_acpc(predicted_scanner)
        except ValueError:
            continue

        from miatt.landmarks import mean_euclidean_error, per_landmark_error  # noqa: PLC0415
        per_subject.append(mean_euclidean_error(predicted_acpc, true_acpc))
        per_lm_errors.append(per_landmark_error(predicted_acpc, true_acpc))

    from miatt.landmarks import mean_euclidean_error, per_landmark_error  # noqa: PLC0415
    all_labels = sorted(atlas_landmarks[0].keys()) if atlas_landmarks else []
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

    # --- 4. Write predictions for unlabeled subjects --------------------------
    for subject_dir, _ in iter_subjects(data_root, site, labeled=False):
        t1_path = subject_dir / f"t1_{site}.nii.gz"
        if not t1_path.exists():
            continue

        subj_img  = sitk.ReadImage(str(t1_path))
        subj_prep = prep_for_registration(subj_img)

        predicted_scanner = predict_landmarks_atlas(
            subj_prep, atlas_images, atlas_landmarks,
            n_iterations=n_iterations,
        )

        try:
            predicted_acpc = _landmarks_to_acpc(predicted_scanner)
        except ValueError:
            predicted_acpc = predicted_scanner

        out_path = (
            output_root / f"{site}_unlabeled" / subject_dir.name
            / "BCD_ACPC_Landmarks.fcsv"
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
