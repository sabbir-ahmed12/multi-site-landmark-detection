"""Tests for miatt.heuristic — pure numpy, no SimpleITK."""

from __future__ import annotations

import numpy as np

from miatt.heuristic import _TISSUE_FILE, _WM_LANDMARKS, _GLOBUS_LANDMARKS


def test_tissue_file_covers_all_landmarks() -> None:
    """Every landmark in the tissue sets must have a corresponding posterior file."""
    for label in _WM_LANDMARKS | _GLOBUS_LANDMARKS:
        assert label in _TISSUE_FILE, f"{label!r} missing from _TISSUE_FILE"


def test_wm_and_globus_sets_disjoint() -> None:
    assert _WM_LANDMARKS.isdisjoint(_GLOBUS_LANDMARKS)


def test_wm_landmarks_known_cc() -> None:
    """Confirm the core corpus-callosum landmarks are in the WM set."""
    required = {"genu", "rostrum", "l_inner_corpus", "r_inner_corpus"}
    assert required <= _WM_LANDMARKS


def test_globus_landmarks_empty() -> None:
    """GLOBUS set is intentionally empty: boundary landmarks cannot be localised
    by a centroid search inside the tissue mass."""
    assert len(_GLOBUS_LANDMARKS) == 0


def test_tissue_file_wm_value() -> None:
    for label in _WM_LANDMARKS:
        assert _TISSUE_FILE[label] == "POSTERIOR_WM_TOTAL.nii.gz"


def test_tissue_file_no_globus_keys() -> None:
    for v in _TISSUE_FILE.values():
        assert v != "POSTERIOR_GLOBUS_TOTAL.nii.gz"


# ---------------------------------------------------------------------------
# Synthetic centroid logic test (no SimpleITK — pure numpy geometry)
# ---------------------------------------------------------------------------

def _mock_weighted_centroid(
    arr: np.ndarray,
    spacing: np.ndarray,
    origin: np.ndarray,
    center_physical: np.ndarray,
    radius_mm: float,
    min_posterior: float = 0.05,
) -> np.ndarray | None:
    """Pure-numpy reimplementation matching the heuristic logic for unit tests."""
    size = np.array(arr.shape[::-1], dtype=float)  # (nx, ny, nz) from (nz, ny, nx)
    ci = (center_physical - origin) / spacing  # continuous index (x, y, z)
    radii = np.maximum(1, np.round(radius_mm / spacing)).astype(int)

    lo = np.maximum(0, np.floor(ci - radii).astype(int))
    hi = np.minimum(size.astype(int) - 1, np.ceil(ci + radii).astype(int))

    sub = arr[lo[2]:hi[2]+1, lo[1]:hi[1]+1, lo[0]:hi[0]+1].copy().astype(float)
    sub[sub < min_posterior] = 0.0
    total = sub.sum()
    if total < 1e-6:
        return None

    sub /= total
    iz = np.arange(lo[2], hi[2] + 1, dtype=float)
    iy = np.arange(lo[1], hi[1] + 1, dtype=float)
    ix = np.arange(lo[0], hi[0] + 1, dtype=float)

    cx = float((sub.sum(axis=(0, 1)) * ix).sum())
    cy = float((sub.sum(axis=(0, 2)) * iy).sum())
    cz = float((sub.sum(axis=(1, 2)) * iz).sum())

    return origin + np.array([cx, cy, cz]) * spacing


def test_centroid_finds_blob_centre() -> None:
    """A Gaussian blob should return a centroid near its centre."""
    spacing = np.array([1.0, 1.0, 1.0])
    origin = np.array([-25.0, -25.0, -25.0])
    shape = (50, 50, 50)  # (z, y, x)
    arr = np.zeros(shape, dtype=np.float32)

    # Place a blob at physical (0, 0, 0)
    blob_idx = np.round((np.array([0.0, 0.0, 0.0]) - origin) / spacing).astype(int)
    ix, iy, iz = blob_idx
    arr[iz, iy, ix] = 1.0
    # 3×3×3 neighbourhood
    for dz in range(-2, 3):
        for dy in range(-2, 3):
            for dx in range(-2, 3):
                zi, yi, xi = iz + dz, iy + dy, ix + dx
                if 0 <= xi < 50 and 0 <= yi < 50 and 0 <= zi < 50:
                    dist = np.sqrt(dx**2 + dy**2 + dz**2)
                    arr[zi, yi, xi] = max(arr[zi, yi, xi], np.exp(-0.5 * dist**2))

    center = np.array([0.0, 0.0, 0.0])
    result = _mock_weighted_centroid(arr, spacing, origin, center, radius_mm=10.0)
    assert result is not None
    np.testing.assert_allclose(result, center, atol=0.5)


def test_centroid_returns_none_when_flat() -> None:
    """All-zero posterior should return None (fall back to mean)."""
    spacing = np.array([1.0, 1.0, 1.0])
    origin = np.zeros(3)
    arr = np.zeros((20, 20, 20), dtype=np.float32)
    result = _mock_weighted_centroid(arr, spacing, origin, np.array([10.0, 10.0, 10.0]), 5.0)
    assert result is None


def test_centroid_ignores_out_of_sphere_voxels() -> None:
    """Voxels outside the search radius should not pull the centroid away."""
    spacing = np.array([1.0, 1.0, 1.0])
    origin = np.array([-15.0, -15.0, -15.0])
    arr = np.zeros((30, 30, 30), dtype=np.float32)

    # Strong blob at (-10, -10, -10) = index (5, 5, 5)
    arr[5, 5, 5] = 1.0
    # Distant distractor at (10, 10, 10) = index (25, 25, 25)
    arr[25, 25, 25] = 10.0

    # Search near (-10, -10, -10) with radius 6 mm — distractor is 35mm away
    result = _mock_weighted_centroid(
        arr, spacing, origin, np.array([-10.0, -10.0, -10.0]), radius_mm=6.0
    )
    assert result is not None
    np.testing.assert_allclose(result, [-10.0, -10.0, -10.0], atol=0.1)
