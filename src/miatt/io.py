"""I/O utilities for NIfTI volumes and Slicer .fcsv landmark files."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import SimpleITK as sitk  # only for type annotations, not runtime


def _sitk():
    """Lazy accessor for SimpleITK — avoids crash when pytest collection loads this module."""
    import SimpleITK  # noqa: PLC0415
    return SimpleITK


def load_image(path: str | Path) -> "sitk.Image":
    """Load a NIfTI volume from *path*."""
    return _sitk().ReadImage(str(path))


def load_fcsv(path: str | Path) -> dict[str, np.ndarray]:
    """Parse a Slicer .fcsv file and return {label: xyz_mm} mapping.

    Coordinates are returned in RAS physical space (millimetres).
    """
    landmarks: dict[str, np.ndarray] = {}
    with open(path, newline="") as fh:
        for row in csv.reader(fh):
            if not row or row[0].startswith("#"):
                continue
            # columns: id, x, y, z, ..., label
            label = row[11].strip() if len(row) > 11 else row[0].strip()
            xyz = np.array([float(row[1]), float(row[2]), float(row[3])])
            landmarks[label] = xyz
    return landmarks


def save_fcsv(
    landmarks: dict[str, np.ndarray],
    path: str | Path,
) -> None:
    """Write a Slicer 4.6-format .fcsv file from *landmarks* {label: xyz_mm}.

    Output format matches the dataset files (version 4.6, CoordinateSystem = 0 = RAS).
    Landmark order is preserved from the dict iteration order.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as fh:
        fh.write("# Markups fiducial file version = 4.6\n")
        fh.write("# CoordinateSystem = 0\n")
        fh.write(
            "# columns = id,x,y,z,ow,ox,oy,oz,vis,sel,lock,label,desc,associatedNodeID\n"
        )
        for idx, (label, xyz) in enumerate(landmarks.items()):
            fh.write(
                f"vtkMRMLMarkupsFiducialNode_{idx},"
                f"{xyz[0]:.6f},{xyz[1]:.6f},{xyz[2]:.6f},"
                f"0,0,0,1,1,1,0,{label},,\n"
            )


def iter_subjects(data_root: str | Path, site: str, labeled: bool = True):
    """Yield (subject_dir, fcsv_path_or_None) for every subject at *site*.

    Args:
        data_root: Root of MIATTFINALEXAMDATA.
        site: Site identifier, e.g. "siteA".
        labeled: If True, use the labeled split; False for unlabeled.
    """
    data_root = Path(data_root)
    split = site if labeled else f"{site}_unlabeled"
    site_dir = data_root / split
    for subject_dir in sorted(site_dir.iterdir()):
        if not subject_dir.is_dir() or subject_dir.name.startswith("."):
            continue
        fcsv = subject_dir / "BCD_ACPC_Landmarks.fcsv"
        yield subject_dir, (fcsv if fcsv.exists() else None)
