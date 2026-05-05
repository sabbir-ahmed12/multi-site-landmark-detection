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


TARGET_SPACING = (1.0, 1.0, 1.0)  # mm isotropic


def resample_to_isotropic(
    image: "sitk.Image",
    target_spacing: tuple[float, float, float] = TARGET_SPACING,
    interpolator: int | None = None,
) -> "sitk.Image":
    """Resample *image* to *target_spacing* mm isotropic.

    Uses the image's own direction and origin so the physical extent is preserved.
    This is site-agnostic: it operates entirely in physical space.
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


def normalize_intensity(image: "sitk.Image") -> "sitk.Image":
    """Clip to [0.5%, 99.5%] percentile then rescale to [0, 1]."""
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


def load_t1(subject_dir: str | Path, site: str) -> "sitk.Image":
    """Load the T1 image for a subject, given its directory and site name."""
    subject_dir = Path(subject_dir)
    t1_path = subject_dir / f"t1_{site}.nii.gz"
    return _sitk().ReadImage(str(t1_path))
