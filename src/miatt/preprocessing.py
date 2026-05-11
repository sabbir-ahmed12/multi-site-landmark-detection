"""Site-agnostic preprocessing for brain MRI volumes."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import SimpleITK as sitk


def _sitk():
    import SimpleITK  # noqa: PLC0415
    return SimpleITK


TARGET_SPACING = (1.0, 1.0, 1.0)   # mm isotropic
TARGET_ORIENTATION = "RAS"          # neuroimaging standard


# ---------------------------------------------------------------------------
# Orientation
# ---------------------------------------------------------------------------

def orientation_code(image: "sitk.Image") -> str:
    """Return the 3-letter DICOM orientation code of *image* (e.g. 'LPS', 'RAS')."""
    sitk = _sitk()
    filt = sitk.DICOMOrientImageFilter()
    return filt.GetOrientationFromDirectionCosines(image.GetDirection())


def reorient_to_ras(image: "sitk.Image") -> "sitk.Image":
    """Reorient *image* to RAS+ if it is not already.

    Uses DICOMOrientImageFilter which only flips/transposes axes — no
    interpolation, no intensity change.  Idempotent: an RAS image is
    returned unchanged.  Handles any arbitrary site orientation.
    """
    sitk = _sitk()
    filt = sitk.DICOMOrientImageFilter()
    filt.SetDesiredCoordinateOrientation(TARGET_ORIENTATION)
    return filt.Execute(image)


# ---------------------------------------------------------------------------
# Resampling
# ---------------------------------------------------------------------------

def resample_to_isotropic(
    image: "sitk.Image",
    target_spacing: tuple[float, float, float] = TARGET_SPACING,
    interpolator: int | None = None,
) -> "sitk.Image":
    """Resample *image* to *target_spacing* mm isotropic.

    Operates entirely in physical space so it is site-agnostic.  The image
    direction and origin are preserved.
    """
    sitk = _sitk()
    if interpolator is None:
        interpolator = sitk.sitkLinear

    original_spacing = image.GetSpacing()
    original_size = image.GetSize()

    new_size = [
        int(round(osz * osp / tsp))
        for osz, osp, tsp in zip(original_size, original_spacing, target_spacing)
    ]

    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(target_spacing)
    resampler.SetSize(new_size)
    resampler.SetOutputDirection(image.GetDirection())
    resampler.SetOutputOrigin(image.GetOrigin())
    resampler.SetInterpolator(interpolator)
    resampler.SetDefaultPixelValue(0)
    return resampler.Execute(image)


# ---------------------------------------------------------------------------
# Intensity normalisation
# ---------------------------------------------------------------------------

def zscore_normalize(image: "sitk.Image") -> "sitk.Image":
    """Z-score normalise *image* using foreground-voxel statistics.

    Foreground is defined as voxels above the 15th percentile of the
    intensity-shifted array.  Shifting by the array minimum before
    thresholding makes the mask robust to images with negative baselines
    (e.g. siteF range [-1000, 1000]) and non-zero backgrounds (e.g. siteD
    range [1500, 13200]).

    Returns a float32 image with the same spatial metadata as the input.
    """
    sitk = _sitk()
    arr = sitk.GetArrayFromImage(image).astype(np.float32)

    # Shift to [0, …] so percentile threshold is scale-independent
    arr_shifted = arr - arr.min()
    threshold = float(np.percentile(arr_shifted, 15))
    mask = arr_shifted > threshold

    if mask.sum() < 100:          # degenerate image guard
        mask = np.ones_like(arr, dtype=bool)

    fg = arr[mask]
    mean = float(fg.mean())
    std = float(fg.std())
    if std < 1e-6:
        std = 1.0

    normalized = (arr - mean) / std

    out = sitk.GetImageFromArray(normalized)
    out.CopyInformation(image)
    return out


def normalize_intensity(image: "sitk.Image") -> "sitk.Image":
    """Clip to [0.5%, 99.5%] percentile then rescale to [0, 1].

    Kept for backward compatibility.  Prefer ``zscore_normalize`` for
    site-agnostic pipelines that must handle different intensity ranges.
    """
    sitk = _sitk()
    arr = sitk.GetArrayFromImage(image).astype(np.float32)
    lo, hi = np.percentile(arr, [0.5, 99.5])

    clamp = sitk.ClampImageFilter()
    clamp.SetLowerBound(float(lo))
    clamp.SetUpperBound(float(hi))
    clamped = clamp.Execute(sitk.Cast(image, sitk.sitkFloat32))

    rescaler = sitk.RescaleIntensityImageFilter()
    rescaler.SetOutputMinimum(0.0)
    rescaler.SetOutputMaximum(1.0)
    return rescaler.Execute(clamped)


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

def preprocess(
    image: "sitk.Image",
    target_spacing: tuple[float, float, float] = TARGET_SPACING,
    interpolator: int | None = None,
) -> "sitk.Image":
    """Full standardisation pipeline: reorient → resample → z-score normalise.

    Each step is site-agnostic:
      1. ``reorient_to_ras``   — corrects arbitrary scanner orientation to RAS+
      2. ``resample_to_isotropic`` — makes voxel spacing uniform
      3. ``zscore_normalize``  — brings intensity to zero-mean / unit-std regardless
                                  of scanner gain, baseline offset, or bit depth

    The output is always RAS-oriented, isotropic at *target_spacing*, float32.
    """
    image = reorient_to_ras(image)
    image = resample_to_isotropic(image, target_spacing=target_spacing, interpolator=interpolator)
    image = zscore_normalize(image)
    return image


# ---------------------------------------------------------------------------
# Convenience loader
# ---------------------------------------------------------------------------

def load_t1(subject_dir: str | Path, site: str) -> "sitk.Image":
    """Load the T1 image for a subject given its directory and site name."""
    subject_dir = Path(subject_dir)
    t1_path = subject_dir / f"t1_{site}.nii.gz"
    return _sitk().ReadImage(str(t1_path))
