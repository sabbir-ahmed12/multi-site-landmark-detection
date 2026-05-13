"""Approach 6 — BCD-inspired Linear Landmark Localisation (LLS).

Reimplements the core step of BRAINSConstellationDetector's LLS model in
pure Python using the project's own labeled data as the training set.

Algorithm
---------
BCD works by:
  1. Detecting a few robust anchor landmarks (AC, PC, eye centres) from the
     raw image using Hough voting and template matching.
  2. Using a Linear Least Squares model trained from labeled examples to
     predict the remaining landmarks from image-feature observations at the
     expected landmark locations.

This module replaces step 1 with the per-site mean ACPC transform (same as
Approach 1) and implements step 2 as ridge regression from per-subject tissue
posterior values sampled at the mean landmark positions in scanner space.

Feature design
--------------
For each subject the 6 tissue posteriors (WM, GM, CSF, VB, Globus, BG) are
sampled at the 51 mean ACPC-space landmark positions projected back into the
subject's scanner space.  This yields a 51 × 6 = 306-dimensional feature
vector.  Ridge regression maps this vector to the 51 × 3 = 153 ACPC-space
landmark coordinates.

Train / inference consistency
------------------------------
Training:  ACPC transform from ground-truth AC/PC/LE/RE (T_true).
Inference: ACPC transform from the per-site mean AC/PC/LE/RE (T_approx).
Both use the SAME reference set (mean ACPC landmarks) as sampling anchor,
so the train/inference feature distributions remain close.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
from scipy.ndimage import map_coordinates

from miatt.acpc import apply_transform


POSTERIOR_NAMES: tuple[str, ...] = (
    "POSTERIOR_WM_TOTAL.nii.gz",
    "POSTERIOR_GM_TOTAL.nii.gz",
    "POSTERIOR_CSF_TOTAL.nii.gz",
    "POSTERIOR_VB_TOTAL.nii.gz",
    "POSTERIOR_GLOBUS_TOTAL.nii.gz",
    "POSTERIOR_BACKGROUND_TOTAL.nii.gz",
)
N_TISSUES: int = len(POSTERIOR_NAMES)


def _sitk():
    import SimpleITK  # noqa: PLC0415
    return SimpleITK


def _load_posterior_array(
    subject_dir: Path, name: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (array_ZYX, origin_xyz, spacing_xyz) for one posterior image."""
    sitk = _sitk()
    path = subject_dir / "ACCUMULATED_POSTERIORS" / name
    img = sitk.ReadImage(str(path))
    arr = sitk.GetArrayFromImage(img).astype(np.float32)   # (Z, Y, X)
    origin  = np.array(img.GetOrigin(),  dtype=np.float64)  # (x, y, z)
    spacing = np.array(img.GetSpacing(), dtype=np.float64)  # (x, y, z)
    return arr, origin, spacing


def _sample_at_physical(
    arr: np.ndarray,
    origin: np.ndarray,
    spacing: np.ndarray,
    pts: np.ndarray,           # (N, 3) RAS mm (x, y, z)
) -> np.ndarray:
    """Trilinearly sample *arr* at physical points *pts*.

    All images in this dataset have identity direction cosines so the
    physical→voxel conversion is a simple translate+scale:
        voxel_xyz = (physical_xyz − origin_xyz) / spacing_xyz
    The numpy array is in (Z, Y, X) order, so the map_coordinates
    coordinate axes are [z_idx, y_idx, x_idx].
    """
    vox = (pts - origin[np.newaxis, :]) / spacing[np.newaxis, :]  # (N, 3) x/y/z
    coords = np.stack([vox[:, 2], vox[:, 1], vox[:, 0]], axis=0)  # (3, N)
    return map_coordinates(arr, coords, order=1, mode="constant", cval=0.0).astype(np.float32)


def extract_features(
    subject_dir: Path,
    scanner_pts: np.ndarray,   # (N_LANDMARKS, 3) RAS mm
) -> np.ndarray:
    """Return (N_LANDMARKS * N_TISSUES,) posterior feature vector.

    For each of N_TISSUES posteriors the value at each scanner_pts location
    is sampled.  Missing posterior files are left as zeros.
    """
    n = scanner_pts.shape[0]
    feat = np.zeros((n, N_TISSUES), dtype=np.float32)
    for t, name in enumerate(POSTERIOR_NAMES):
        path = subject_dir / "ACCUMULATED_POSTERIORS" / name
        if not path.exists():
            continue
        arr, origin, spacing = _load_posterior_array(subject_dir, name)
        feat[:, t] = _sample_at_physical(arr, origin, spacing, scanner_pts)
    return feat.ravel()


class LLSModel:
    """Ridge regression model: features (N_LANDMARKS*N_TISSUES,) → ACPC coords (N_LANDMARKS*3,)."""

    def __init__(self) -> None:
        self.W: np.ndarray | None = None   # (n_out, n_feat)
        self.b: np.ndarray | None = None   # (n_out,)

    def fit(
        self,
        X: np.ndarray,     # (n_samples, n_feat)
        Y: np.ndarray,     # (n_samples, n_out)
        alpha: float = 10.0,
    ) -> "LLSModel":
        """Fit ridge regression: minimise ||Y − XW.T − b||² + α||W||²."""
        n, d = X.shape
        # Augment with bias column so we solve for W and b together.
        X64 = X.astype(np.float64)
        Y64 = Y.astype(np.float64)
        X_aug = np.hstack([X64, np.ones((n, 1))])  # (n, d+1)
        reg = np.eye(d + 1) * alpha
        reg[-1, -1] = 0.0                           # no regularisation on bias
        A = X_aug.T @ X_aug + reg                   # (d+1, d+1)
        B = X_aug.T @ Y64                           # (d+1, n_out)
        W_aug = np.linalg.solve(A, B)               # (d+1, n_out)
        self.W = W_aug[:-1, :].T.astype(np.float32)  # (n_out, d)
        self.b = W_aug[-1, :].astype(np.float32)     # (n_out,)
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Predict from feature vector(s).

        Args:
            x: (n_feat,) or (n_samples, n_feat).
        Returns:
            (n_out,) or (n_samples, n_out).
        """
        assert self.W is not None, "model not fitted"
        scalar = x.ndim == 1
        x2 = np.atleast_2d(x).astype(np.float32)
        out = x2 @ self.W.T + self.b[np.newaxis, :]
        return out.squeeze(0) if scalar else out

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as fh:
            pickle.dump({"W": self.W, "b": self.b}, fh)

    @classmethod
    def load(cls, path: Path) -> "LLSModel":
        with open(path, "rb") as fh:
            d = pickle.load(fh)
        m = cls()
        m.W = d["W"]
        m.b = d["b"]
        return m


def _inv4(T: np.ndarray) -> np.ndarray:
    """Invert a 4×4 rigid homogeneous transform."""
    R = T[:3, :3]
    t = T[:3, 3]
    T_inv = np.eye(4)
    T_inv[:3, :3] = R.T
    T_inv[:3, 3] = -R.T @ t
    return T_inv


def build_feature_matrix(
    labeled_pairs: list[tuple[Path, "dict[str, np.ndarray]"]],
    acpc_transforms: list[np.ndarray],      # T_true per subject
    mean_acpc_coords: np.ndarray,            # (N, 3) mean ACPC landmark positions
    label_order: list[str],
    site_names: list[str],                   # parallel to labeled_pairs
) -> tuple[np.ndarray, np.ndarray]:
    """Build feature matrix X and target matrix Y for LLS training.

    For each subject:
      - Project mean ACPC positions → scanner space via T_true^{-1}
      - Sample 6 posteriors at those scanner positions → 306-dim feature
      - Target = subject's true ACPC coordinates (153-dim)

    Returns X (n_subjects, 306) and Y (n_subjects, 153).
    """
    n = len(labeled_pairs)
    n_lm = len(label_order)
    n_feat = n_lm * N_TISSUES
    X = np.zeros((n, n_feat), dtype=np.float32)
    Y = np.zeros((n, n_lm * 3), dtype=np.float32)

    for i, ((subject_dir, lm_dict), T_true, site) in enumerate(
        zip(labeled_pairs, acpc_transforms, site_names)
    ):
        T_inv = _inv4(T_true)
        scanner_pts = apply_transform(T_inv, mean_acpc_coords)   # (N, 3)
        X[i] = extract_features(subject_dir, scanner_pts)

        # True ACPC coords in label order
        acpc_lm = {label: apply_transform(T_true, xyz).squeeze()
                   for label, xyz in lm_dict.items()}
        for j, label in enumerate(label_order):
            if label in acpc_lm:
                Y[i, j * 3: j * 3 + 3] = acpc_lm[label].astype(np.float32)

    return X, Y


def predict_landmarks_lls(
    subject_dir: Path,
    site: str,
    T_approx: np.ndarray,         # 4×4 approximate ACPC transform
    mean_acpc_coords: np.ndarray,  # (N, 3)
    model: LLSModel,
    label_order: list[str],
) -> "dict[str, np.ndarray]":
    """Predict ACPC-space landmarks for one subject using the LLS model."""
    T_inv = _inv4(T_approx)
    scanner_pts = apply_transform(T_inv, mean_acpc_coords)   # (N, 3)
    feat = extract_features(subject_dir, scanner_pts)        # (N * N_TISSUES,)
    pred_flat = model.predict(feat)                          # (N * 3,)
    return {
        label: pred_flat[j * 3: j * 3 + 3].astype(float)
        for j, label in enumerate(label_order)
    }
