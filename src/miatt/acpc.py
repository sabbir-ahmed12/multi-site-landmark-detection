"""ACPC alignment computation.

Computes the rigid transform that places:
  - AC at physical origin (0, 0, 0) mm
  - AC and PC at identical SI and LR coordinates (only AP differs)
  - LE and RE on a common SI plane
"""

from __future__ import annotations

import numpy as np


def compute_acpc_transform(
    ac: np.ndarray,
    pc: np.ndarray,
    le: np.ndarray,
    re: np.ndarray,
) -> np.ndarray:
    """Return a 4×4 rigid transform (RAS) that ACPC-aligns the given landmarks.

    The returned matrix maps original physical coordinates to ACPC space.

    Args:
        ac: Anterior commissure in RAS mm.
        pc: Posterior commissure in RAS mm.
        le: Left eye centre in RAS mm.
        re: Right eye centre in RAS mm.

    Returns:
        4×4 homogeneous rigid transform matrix.
    """
    # AP axis: AC→PC direction
    ap = pc - ac
    ap = ap / np.linalg.norm(ap)

    # LR axis derived from eye midplane; orthogonalise against AP
    inter_eye = re - le
    lr = inter_eye - np.dot(inter_eye, ap) * ap
    lr = lr / np.linalg.norm(lr)

    # SI axis: right-hand cross product
    si = np.cross(ap, lr)
    si = si / np.linalg.norm(si)

    # Rotation matrix (rows = new axes expressed in original frame)
    R = np.stack([lr, ap, si], axis=0)  # 3×3

    # Translation: after rotation, AC must land at origin
    t = -R @ ac  # 3-vector

    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def apply_transform(T: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Apply a 4×4 homogeneous transform to an (N, 3) array of points."""
    points = np.atleast_2d(points)
    ones = np.ones((points.shape[0], 1))
    homogeneous = np.hstack([points, ones])
    return (T @ homogeneous.T).T[:, :3]


def transform_landmarks(
    T: np.ndarray, landmarks: dict[str, np.ndarray]
) -> dict[str, np.ndarray]:
    """Apply *T* to every landmark in the dict and return a new dict."""
    return {label: apply_transform(T, xyz).squeeze() for label, xyz in landmarks.items()}
