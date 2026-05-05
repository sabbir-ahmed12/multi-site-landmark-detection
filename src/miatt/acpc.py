"""ACPC alignment computation.

Computes the rigid transform that places:
  - AC at physical origin (0, 0, 0) mm
  - AC and PC at identical SI (z) and LR (x) coordinates (only AP/y differs)
  - LE and RE on a common SI plane

Coordinate convention (RAS, right-handed):
  x = left→right (LR)
  y = posterior→anterior (AP), so PC has y < 0, AC has y = 0
  z = inferior→superior (SI)
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

    The returned matrix maps original physical coordinates to ACPC space:
      - AC → (0, 0, 0)
      - PC → (0, y_pc, 0) with y_pc < 0 (posterior)
      - LE, RE → same SI coordinate

    Args:
        ac: Anterior commissure in RAS mm.
        pc: Posterior commissure in RAS mm.
        le: Left eye centre in RAS mm.
        re: Right eye centre in RAS mm.

    Returns:
        4×4 homogeneous rigid transform matrix.
    """
    # AP row: anterior direction (PC → AC); positive y = anterior in RAS
    ap = ac - pc
    ap = ap / np.linalg.norm(ap)

    # LR row: right direction derived from eye midplane; orthogonalise against AP
    inter_eye = re - le  # points left→right
    lr = inter_eye - np.dot(inter_eye, ap) * ap
    lr = lr / np.linalg.norm(lr)

    # SI row: superior direction — right-hand rule: lr × ap = si
    # Verified: (1,0,0) × (0,1,0) = (0,0,1) = superior in RAS ✓
    si = np.cross(lr, ap)
    si = si / np.linalg.norm(si)

    # Rotation matrix: rows are the new axes expressed in original frame
    R = np.stack([lr, ap, si], axis=0)  # 3×3, det = +1

    # Translation: after rotation, AC must land at origin
    t = -R @ ac

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
