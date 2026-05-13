"""Approach 2 — rigid registration to within-site ACPC template.

Strategy
--------
Training phase (once per site, results cached to disk):
  1. For each labeled training subject, compute the ACPC transform from their
     own AC/PC/LE/RE landmarks and resample their T1 onto a fixed ACPC template
     grid.
  2. Average the resampled T1s → mean ACPC anatomical template.
  3. Compute mean landmark positions in ACPC space (same data as Approach 1).

Prediction phase (per eval/unlabeled subject):
  1. Rigidly register the ACPC template (fixed) to the subject T1 (moving).
     SimpleITK convention: Execute(fixed, moving) returns a transform that maps
     fixed-space points to moving-space points — here ACPC → scanner.
  2. Apply that transform to the mean ACPC landmarks → predicted scanner
     coordinates.
  3. ACPC-align the predicted scanner landmarks using predicted AC/PC/LE/RE.

Why this approach differs from Approach 1
------------------------------------------
Approach 1 always outputs the population mean in ACPC space — there is no
per-subject adaptation.  Registration recovers each subject's ACPC → scanner
rigid mapping from image content alone, without needing labeled AC/PC/LE/RE.
For unlabeled subjects this is the only available option short of learning-based
methods.  The expected ACPC-space error equals anatomical variation around the
mean (same as Approach 1) because the registration merely re-estimates the ACPC
transform — it cannot resolve within-ACPC anatomical variation.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from miatt.acpc import compute_acpc_transform, transform_landmarks
from miatt.io import load_fcsv, save_fcsv
from miatt.landmarks import aggregate_landmarks
from miatt.preprocessing import normalize_intensity

if TYPE_CHECKING:
    import SimpleITK as sitk

# ---------------------------------------------------------------------------
# Fixed ACPC template grid  (2 mm isotropic, RAS identity direction)
#   x: −100 → +100 mm  (LR, 101 voxels)
#   y: −120 →  +80 mm  (AP, PC-side negative, 101 voxels)
#   z:  −90 → +100 mm  (SI, 96 voxels)
# ---------------------------------------------------------------------------
_TEMPLATE_SPACING: tuple[float, float, float] = (2.0, 2.0, 2.0)
_TEMPLATE_SIZE: tuple[int, int, int] = (101, 101, 96)
_TEMPLATE_ORIGIN: tuple[float, float, float] = (-100.0, -120.0, -90.0)


def _sitk():
    import SimpleITK  # noqa: PLC0415
    return SimpleITK


def _make_template_reference() -> "sitk.Image":
    """Return a zero-filled SimpleITK image on the fixed ACPC template grid."""
    sitk = _sitk()
    ref = sitk.Image(_TEMPLATE_SIZE, sitk.sitkFloat32)
    ref.SetSpacing(_TEMPLATE_SPACING)
    ref.SetOrigin(_TEMPLATE_ORIGIN)
    ref.SetDirection((1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0))
    return ref


def resample_to_acpc_space(image: "sitk.Image", T_4x4: np.ndarray) -> "sitk.Image":
    """Resample a scanner-space image onto the fixed ACPC template grid.

    ResampleImageFilter uses backward mapping (output voxel → input voxel).
    Output is in ACPC space; input is in scanner space.  The backward mapping
    transform is therefore the inverse of T_4x4 (i.e., ACPC → scanner).

    Args:
        image: T1 in scanner (physical RAS) space.
        T_4x4: 4×4 rigid transform mapping scanner → ACPC (from
               compute_acpc_transform).

    Returns:
        Image resampled onto the fixed ACPC template grid.
    """
    sitk = _sitk()
    T_inv = np.linalg.inv(T_4x4)
    affine = sitk.AffineTransform(3)
    affine.SetMatrix(T_inv[:3, :3].flatten().tolist())
    affine.SetTranslation(T_inv[:3, 3].tolist())

    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(_make_template_reference())
    resampler.SetTransform(affine)
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetDefaultPixelValue(0.0)
    return resampler.Execute(image)


def build_acpc_template(
    training_pairs: list[tuple[Path, Path]],
    site: str,
    cache_dir: Path,
) -> tuple["sitk.Image", dict[str, np.ndarray]]:
    """Build (or load from disk) the mean ACPC T1 template and mean ACPC landmarks.

    For each training subject the normalized T1 is resampled into ACPC space
    using that subject's own labeled ACPC transform; the results are averaged.
    Both the template image and the mean landmark file are cached under
    *cache_dir* so the expensive resampling loop runs only once per site.

    Args:
        training_pairs: (subject_dir, fcsv_path) for every training subject.
        site: Site identifier; used to find ``t1_{site}.nii.gz`` and name cache.
        cache_dir: Directory where cached files are stored.

    Returns:
        (template_image, mean_acpc_landmarks) where the image is on the fixed
        ACPC grid and the landmark dict has AC at (0, 0, 0).
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    template_path = cache_dir / f"{site}_acpc_template.nii.gz"
    mean_lm_path = cache_dir / f"{site}_mean_acpc_lm.fcsv"

    if template_path.exists() and mean_lm_path.exists():
        return _sitk().ReadImage(str(template_path)), load_fcsv(mean_lm_path)

    sitk = _sitk()
    acpc_arrays: list[np.ndarray] = []
    acpc_landmarks: list[dict[str, np.ndarray]] = []

    for subject_dir, fcsv_path in training_pairs:
        lm = load_fcsv(fcsv_path)
        try:
            T = compute_acpc_transform(lm["AC"], lm["PC"], lm["LE"], lm["RE"])
        except (KeyError, ValueError):
            continue

        t1_path = Path(subject_dir) / f"t1_{site}.nii.gz"
        if not t1_path.exists():
            continue

        image = sitk.ReadImage(str(t1_path))
        image_norm = normalize_intensity(sitk.Cast(image, sitk.sitkFloat32))
        acpc_vol = resample_to_acpc_space(image_norm, T)
        acpc_arrays.append(sitk.GetArrayFromImage(acpc_vol))
        acpc_landmarks.append(transform_landmarks(T, lm))

    if not acpc_arrays:
        raise RuntimeError(f"No usable training subjects to build template for {site}")

    mean_arr = np.mean(acpc_arrays, axis=0).astype(np.float32)
    template = sitk.GetImageFromArray(mean_arr)
    template.SetSpacing(_TEMPLATE_SPACING)
    template.SetOrigin(_TEMPLATE_ORIGIN)
    template.SetDirection((1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0))

    mean_lm = aggregate_landmarks(acpc_landmarks)

    sitk.WriteImage(template, str(template_path))
    save_fcsv(mean_lm, mean_lm_path)

    return template, mean_lm


def register_to_template(
    subject_t1: "sitk.Image",
    template: "sitk.Image",
) -> "sitk.Transform":
    """Rigidly register the ACPC template (fixed) to a subject T1 (moving).

    Uses Euler3DTransform (6 DOF rigid body), Mattes Mutual Information, and a
    3-level multi-resolution pyramid (4×/2×/1×).  The geometric-centre
    initialiser aligns the image bounding boxes before optimisation.

    The returned transform maps ACPC-space points to scanner-space points
    (fixed → moving direction, per SimpleITK's Execute convention).

    Args:
        subject_t1: Subject T1 in scanner space (moving image).
        template: Mean ACPC template from build_acpc_template (fixed image).

    Returns:
        Rigid SimpleITK transform: ACPC space → scanner space.
    """
    sitk = _sitk()

    fixed = normalize_intensity(sitk.Cast(template, sitk.sitkFloat32))
    moving = normalize_intensity(sitk.Cast(subject_t1, sitk.sitkFloat32))

    init = sitk.CenteredTransformInitializer(
        fixed,
        moving,
        sitk.Euler3DTransform(),
        sitk.CenteredTransformInitializerFilter.GEOMETRY,
    )

    reg = sitk.ImageRegistrationMethod()
    reg.SetMetricAsMattesMutualInformation(numberOfHistogramBins=32)
    reg.SetMetricSamplingStrategy(reg.RANDOM)
    reg.SetMetricSamplingPercentage(0.05)
    reg.SetInitialTransform(init, inPlace=False)
    reg.SetOptimizerAsRegularStepGradientDescent(
        learningRate=1.0,
        minStep=0.001,
        numberOfIterations=200,
        gradientMagnitudeTolerance=1e-8,
    )
    reg.SetOptimizerScalesFromPhysicalShift()
    reg.SetShrinkFactorsPerLevel([4, 2, 1])
    reg.SetSmoothingSigmasPerLevel([4.0, 2.0, 1.0])
    reg.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

    return reg.Execute(fixed, moving)


def propagate_landmarks(
    transform: "sitk.Transform",
    acpc_landmarks: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """Map ACPC-space landmarks to scanner space via the registration transform.

    SimpleITK physical space is LPS, so RAS coords must be converted before
    calling TransformPoint and the LPS output converted back to RAS:
        x_LPS = -x_RAS,  y_LPS = -y_RAS,  z_LPS = z_RAS

    Args:
        transform: SimpleITK transform mapping ACPC → scanner
                   (returned by register_to_template).
        acpc_landmarks: Landmark positions in ACPC RAS space.

    Returns:
        Predicted landmark positions in scanner RAS space.
    """
    result = {}
    for label, ras in acpc_landmarks.items():
        lps_in = (-float(ras[0]), -float(ras[1]), float(ras[2]))
        lps_out = transform.TransformPoint(lps_in)
        result[label] = np.array([-lps_out[0], -lps_out[1], lps_out[2]])
    return result
