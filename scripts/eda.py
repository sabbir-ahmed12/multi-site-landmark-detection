"""Exploratory Data Analysis — generates summary plots and tables by site."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import SimpleITK as sitk
from tqdm import tqdm

from miatt.io import iter_subjects, load_fcsv

DATA_ROOT = Path("/nfs/s-l028/scratch/opt/ece5490/MIATTFINALEXAMDATA")
SITES = ["siteA", "siteB", "siteC", "siteD", "siteE", "siteF"]
OUT_DIR = Path("notebooks/eda_outputs")


def collect_header_stats() -> pd.DataFrame:
    """Read header metadata (spacing, size, origin) for every labeled subject."""
    rows = []
    for site in SITES:
        for subject_dir, fcsv_path in tqdm(
            iter_subjects(DATA_ROOT, site, labeled=True), desc=site
        ):
            t1_path = subject_dir / f"t1_{site}.nii.gz"
            if not t1_path.exists():
                continue
            img = sitk.ReadImage(str(t1_path))
            sp = img.GetSpacing()
            sz = img.GetSize()
            origin = img.GetOrigin()
            rows.append(
                {
                    "site": site,
                    "subject": subject_dir.name,
                    "sx": sp[0],
                    "sy": sp[1],
                    "sz": sp[2],
                    "nx": sz[0],
                    "ny": sz[1],
                    "nz": sz[2],
                    "ox": origin[0],
                    "oy": origin[1],
                    "oz": origin[2],
                }
            )
    return pd.DataFrame(rows)


def collect_landmark_stats() -> pd.DataFrame:
    """Read AC coordinates from labeled subjects to characterise site offset."""
    rows = []
    for site in SITES:
        for subject_dir, fcsv_path in iter_subjects(DATA_ROOT, site, labeled=True):
            if fcsv_path is None:
                continue
            try:
                lm = load_fcsv(fcsv_path)
            except Exception:
                continue
            if "AC" in lm:
                ac = lm["AC"]
                rows.append(
                    {"site": site, "subject": subject_dir.name, "AC_x": ac[0], "AC_y": ac[1], "AC_z": ac[2]}
                )
    return pd.DataFrame(rows)


def plot_spacing_by_site(df: pd.DataFrame, out_dir: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, dim in zip(axes, ["sx", "sy", "sz"]):
        df.boxplot(column=dim, by="site", ax=ax)
        ax.set_title(f"Spacing {dim[-1].upper()} (mm)")
        ax.set_xlabel("Site")
    fig.suptitle("Voxel spacing distribution by site")
    plt.tight_layout()
    fig.savefig(out_dir / "spacing_by_site.png", dpi=150)
    plt.close(fig)


def plot_ac_scatter(df: pd.DataFrame, out_dir: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    pairs = [("AC_x", "AC_y"), ("AC_x", "AC_z"), ("AC_y", "AC_z")]
    for ax, (xc, yc) in zip(axes, pairs):
        for site, grp in df.groupby("site"):
            ax.scatter(grp[xc], grp[yc], label=site, alpha=0.6, s=10)
        ax.set_xlabel(xc)
        ax.set_ylabel(yc)
        ax.legend(fontsize=7)
    fig.suptitle("AC physical coordinate scatter by site")
    plt.tight_layout()
    fig.savefig(out_dir / "ac_scatter_by_site.png", dpi=150)
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Collecting header statistics …")
    header_df = collect_header_stats()
    header_df.to_csv(OUT_DIR / "header_stats.csv", index=False)
    print(header_df.groupby("site")[["sx", "sy", "sz", "nx", "ny", "nz"]].describe())
    plot_spacing_by_site(header_df, OUT_DIR)

    print("\nCollecting landmark statistics …")
    lm_df = collect_landmark_stats()
    lm_df.to_csv(OUT_DIR / "landmark_stats.csv", index=False)
    print(lm_df.groupby("site")[["AC_x", "AC_y", "AC_z"]].describe())
    plot_ac_scatter(lm_df, OUT_DIR)

    print(f"\nOutputs written to {OUT_DIR}/")


if __name__ == "__main__":
    main()
