"""Approach 3 — posterior-based local refinement of landmark predictions.

Strategy
--------
Approach 1 predicts the population mean ACPC position for every subject.
For landmarks inside the corpus callosum (WM tissue), the WM posterior gives
a per-subject signal that we can use as a local centroid anchor.

For each subject:
  1. Start from the population mean ACPC positions (Approach 1 baseline).
  2. For each *refinable* landmark, map the mean ACPC position to scanner
     space using the per-subject ACPC transform T.
  3. Load the WM posterior in its native scanner space.
  4. Find the posterior-weighted centroid within an 8 mm search sphere.
  5. Map that centroid back to ACPC space → per-subject prediction.
  6. Fall back to the population mean if the posterior is absent or flat.

Experimental outcome (siteA, 20% holdout)
------------------------------------------
The WM centroid search produced a negative result:
  mean baseline (Approach 1): 4.59 mm
  heuristic (this approach):  5.13 mm  (+0.54 mm regression)

Root cause: the WM posterior is high throughout all white matter, not just at
the target CC sub-structure.  Within an 8 mm radius of any CC landmark, the
centroid captures the CC body, internal capsule, and corona radiata fibers.
The centroid therefore converges to the bulk WM centre of mass in that region
rather than the anatomically specific landmark, adding noise instead of signal.
The worst regressions were genu (+4.4 mm) and rostrum (+4.6 mm) — both
midline CC landmarks surrounded by dense, structureless WM.

The GLOBUS-based search (lat_left / lat_right) was trialled first and showed
catastrophic regression (>11 mm) because those landmarks sit on the GP capsule
wall, not inside the tissue mass.  That search was removed.

Conclusion: posterior centroid search is insufficient for this dataset.
A landmark-specific local search would require (a) a tissue whose centroid
coincides with the landmark (e.g., a thalamic nucleus centre), or (b) a
supervised regressor that maps posterior features to landmark offsets from the
mean (addressed by Approach 4: CNN).
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from miatt.acpc import apply_transform, compute_acpc_transform, transform_landmarks

if TYPE_CHECKING:
    import SimpleITK as sitk

# ---------------------------------------------------------------------------
# Landmark → tissue posterior assignments
# ---------------------------------------------------------------------------
_WM_LANDMARKS: frozenset[str] = frozenset({
    "genu",
    "rostrum",
    "l_inner_corpus",
    "r_inner_corpus",
    "callosum_left",
    "callosum_right",
    "m_ax_sup",
    "m_ax_inf",
})

# GLOBUS landmarks excluded: lat_left/lat_right are on the boundary of the
# globus pallidus capsule, not inside the tissue mass.  The centroid search
# finds the GP centre and pulls predictions away from the true wall landmark.
_GLOBUS_LANDMARKS: frozenset[str] = frozenset()

# Map from label → posterior file stem (within ACCUMULATED_POSTERIORS/)
_TISSUE_FILE: dict[str, str] = {
    label: "POSTERIOR_WM_TOTAL.nii.gz" for label in _WM_LANDMARKS
}

# Search radius (mm): tight enough to stay within the local CC sub-structure.
# At 8 mm the search captures ~16 voxels in each direction at 1 mm spacing,
# which covers the CC genu / body without reaching adjacent WM tracts.
_SEARCH_RADIUS: dict[str, float] = {
    "POSTERIOR_WM_TOTAL.nii.gz": 8.0,
}
_MIN_POSTERIOR = 0.05  # ignore voxels below this probability


def _sitk():
    import SimpleITK  # noqa: PLC0415
    return SimpleITK


def _load_posterior(subject_dir: Path, posterior_file: str) -> "sitk.Image":
    """Load a tissue posterior from subject_dir/ACCUMULATED_POSTERIORS/."""
    path = subject_dir / "ACCUMULATED_POSTERIORS" / posterior_file
    return _sitk().ReadImage(str(path))


def _posterior_weighted_centroid(
    posterior: "sitk.Image",
    center_physical: np.ndarray,
    radius_mm: float,
) -> np.ndarray | None:
    """Return posterior-weighted centroid near *center_physical* (RAS mm).

    Works in index space: converts the physical centre to a voxel index, then
    restricts to a box of ±radius_mm (rounded to whole voxels).  Returns None
    when the posterior is flat or absent in the search region.

    Args:
        posterior: Tissue posterior in scanner (physical RAS) space.
        center_physical: Search-centre in RAS mm.
        radius_mm: Half-width of the search box.

    Returns:
        Posterior-weighted centroid in RAS mm, or None if search fails.
    """
    sitk = _sitk()
    arr = sitk.GetArrayFromImage(posterior).astype(np.float32)  # (z, y, x)
    spacing = np.array(posterior.GetSpacing())  # (sx, sy, sz)
    size = np.array(posterior.GetSize())  # (nx, ny, nz)

    # Physical centre → continuous index (x, y, z) order
    ci = np.array(
        posterior.TransformPhysicalPointToContinuousIndex(center_physical.tolist())
    )

    # Voxel-space radii, clamped to ≥ 1
    radii = np.maximum(1, np.round(radius_mm / spacing)).astype(int)

    lo = np.maximum(0, np.floor(ci - radii).astype(int))
    hi = np.minimum(size - 1, np.ceil(ci + radii).astype(int))

    # Extract sub-volume; arr is (z,y,x) while lo/hi are (x,y,z)
    sub = arr[lo[2]:hi[2]+1, lo[1]:hi[1]+1, lo[0]:hi[0]+1].copy()
    sub[sub < _MIN_POSTERIOR] = 0.0

    total = sub.sum()
    if total < 1e-6:
        return None

    sub /= total
    # Build coordinate grids in (z,y,x) → (x,y,z) mapping
    iz = np.arange(lo[2], hi[2] + 1, dtype=float)
    iy = np.arange(lo[1], hi[1] + 1, dtype=float)
    ix = np.arange(lo[0], hi[0] + 1, dtype=float)

    # Marginal weighted sums
    cx = float((sub.sum(axis=(0, 1)) * ix).sum())
    cy = float((sub.sum(axis=(0, 2)) * iy).sum())
    cz = float((sub.sum(axis=(1, 2)) * iz).sum())

    return np.array(
        posterior.TransformContinuousIndexToPhysicalPoint([cx, cy, cz])
    )


def predict_landmarks_heuristic(
    subject_dir: Path,
    site: str,
    T_scanner_to_acpc: np.ndarray,
    mean_acpc_lm: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """Predict ACPC-space landmarks using posterior-guided local refinement.

    For landmarks in *_TISSUE_FILE*, a local posterior-weighted centroid search
    in scanner space replaces the population mean; all others keep the mean.

    Args:
        subject_dir: Subject directory containing ACCUMULATED_POSTERIORS/.
        site: Site identifier (unused here, kept for API consistency).
        T_scanner_to_acpc: 4×4 rigid matrix: scanner → ACPC.
        mean_acpc_lm: Population mean landmark positions in ACPC space.

    Returns:
        Predicted landmark positions in ACPC space.
    """
    T_inv = np.linalg.inv(T_scanner_to_acpc)  # ACPC → scanner
    predictions: dict[str, np.ndarray] = {}
    cached_posteriors: dict[str, "sitk.Image | None"] = {}

    for label, mean_acpc in mean_acpc_lm.items():
        if label not in _TISSUE_FILE:
            predictions[label] = mean_acpc.copy()
            continue

        post_file = _TISSUE_FILE[label]
        if post_file not in cached_posteriors:
            post_path = subject_dir / "ACCUMULATED_POSTERIORS" / post_file
            if post_path.exists():
                cached_posteriors[post_file] = _sitk().ReadImage(str(post_path))
            else:
                cached_posteriors[post_file] = None

        posterior = cached_posteriors[post_file]
        if posterior is None:
            predictions[label] = mean_acpc.copy()
            continue

        # Map mean ACPC position to scanner space
        mean_scanner = apply_transform(T_inv, mean_acpc).squeeze()

        radius = _SEARCH_RADIUS[post_file]
        centroid_scanner = _posterior_weighted_centroid(posterior, mean_scanner, radius)

        if centroid_scanner is None:
            predictions[label] = mean_acpc.copy()
        else:
            # Map centroid back to ACPC space
            predictions[label] = apply_transform(
                T_scanner_to_acpc, centroid_scanner
            ).squeeze()

    return predictions
