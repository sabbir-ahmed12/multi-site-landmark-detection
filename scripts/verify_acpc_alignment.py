"""
Verify that the labeled .fcsv files satisfy the three ACPC-alignment
constraints stated in the README (Pipeline section):

  C1. AC is at physical coordinate (0.0, 0.0, 0.0) mm.
  C2. AC and PC share the same X (left–right) and Z (superior–inferior);
      only Y (anterior–posterior) differs.
  C3. LE and RE share the same Z (superior–inferior) — common plane.

All coordinates are read directly from the .fcsv files, which store RAS mm
(CoordinateSystem=0 in Slicer 4.6 = RAS).

If every constraint has near-zero residuals across all subjects the T1
images are already in ACPC space and can be used as ground truth.

Outputs:
  - printed per-constraint statistics (mean, std, max absolute residual)
  - per-site breakdown
  - lists subjects that fail each constraint beyond a tolerance (1 mm)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from miatt.io import iter_subjects, load_fcsv

DATA_ROOT = Path("/nfs/s-l028/scratch/opt/ece5490/MIATTFINALEXAMDATA")
SITES     = ["siteA", "siteB", "siteC", "siteD", "siteE", "siteF"]
TOL_MM    = 1.0   # residual threshold for flagging a subject as failing


def check_subject(site: str, subj_dir: Path, fcsv_path: Path) -> dict | None:
    try:
        lm = load_fcsv(fcsv_path)
    except Exception as e:
        print(f"  [WARN] Could not parse {fcsv_path}: {e}")
        return None

    required = {"AC", "PC", "LE", "RE"}
    if not required.issubset(lm):
        missing = required - lm.keys()
        print(f"  [WARN] {site}/{subj_dir.name}: missing landmarks {missing}")
        return None

    ac = lm["AC"]
    pc = lm["PC"]
    le = lm["LE"]
    re = lm["RE"]

    # C1: AC deviation from origin
    c1 = float(np.linalg.norm(ac))

    # C2: AC–PC share X and Z  (only Y should differ)
    c2_x = float(abs(pc[0] - ac[0]))   # left–right residual
    c2_z = float(abs(pc[2] - ac[2]))   # superior–inferior residual

    # C3: LE–RE share Z
    c3 = float(abs(le[2] - re[2]))

    # Bonus: AC–PC distance (anatomical sanity, expected 23–30 mm)
    acpc_dist = float(np.linalg.norm(pc - ac))

    return {
        "site":     site,
        "subject":  subj_dir.name,
        "c1_ac_origin_mm":  c1,
        "c2_acpc_dx_mm":    c2_x,
        "c2_acpc_dz_mm":    c2_z,
        "c3_eyes_dz_mm":    c3,
        "acpc_dist_mm":     acpc_dist,
        "ac_x": ac[0], "ac_y": ac[1], "ac_z": ac[2],
        "pc_x": pc[0], "pc_y": pc[1], "pc_z": pc[2],
        "le_z": le[2], "re_z": re[2],
    }


def main() -> None:
    rows = []
    for site in SITES:
        n = 0
        for subj_dir, fcsv_path in iter_subjects(DATA_ROOT, site, labeled=True):
            if fcsv_path is None:
                continue
            rec = check_subject(site, subj_dir, fcsv_path)
            if rec:
                rows.append(rec)
                n += 1
        print(f"{site}: {n} subjects checked")

    df = pd.DataFrame(rows)
    total = len(df)
    print(f"\nTotal subjects checked: {total}\n")

    # ------------------------------------------------------------------
    # Global statistics
    # ------------------------------------------------------------------
    metrics = {
        "C1 — AC distance from origin (mm)":    "c1_ac_origin_mm",
        "C2 — AC–PC ΔX left–right (mm)":        "c2_acpc_dx_mm",
        "C2 — AC–PC ΔZ sup–inf (mm)":           "c2_acpc_dz_mm",
        "C3 — LE–RE ΔZ sup–inf (mm)":           "c3_eyes_dz_mm",
        "AC–PC distance (anatomical, mm)":       "acpc_dist_mm",
    }

    print("=" * 65)
    print("CONSTRAINT RESIDUALS — all labeled subjects")
    print("=" * 65)
    print(f"{'Metric':<42} {'mean':>7} {'std':>7} {'max':>7}")
    print("-" * 65)
    for label, col in metrics.items():
        s = df[col]
        print(f"{label:<42} {s.mean():>7.3f} {s.std():>7.3f} {s.max():>7.3f}")
    print("=" * 65)

    # ------------------------------------------------------------------
    # Per-site breakdown
    # ------------------------------------------------------------------
    print("\nPer-site mean residuals:")
    site_cols = ["c1_ac_origin_mm", "c2_acpc_dx_mm", "c2_acpc_dz_mm",
                 "c3_eyes_dz_mm", "acpc_dist_mm"]
    site_summary = df.groupby("site")[site_cols].mean().round(3)
    site_summary.columns = ["C1_AC", "C2_dx", "C2_dz", "C3_dz", "ACPC_dist"]
    print(site_summary.to_string())

    # ------------------------------------------------------------------
    # Flag subjects that fail any constraint beyond TOL_MM
    # ------------------------------------------------------------------
    print(f"\nSubjects with any constraint residual > {TOL_MM} mm:")
    constraint_cols = ["c1_ac_origin_mm", "c2_acpc_dx_mm",
                       "c2_acpc_dz_mm", "c3_eyes_dz_mm"]
    failed = df[df[constraint_cols].max(axis=1) > TOL_MM]
    if failed.empty:
        print(f"  None — all {total} subjects pass within {TOL_MM} mm.")
    else:
        print(f"  {len(failed)} subject(s) flagged:")
        for _, r in failed.iterrows():
            print(f"  {r['site']}/{r['subject']}  "
                  f"C1={r['c1_ac_origin_mm']:.2f}  "
                  f"C2_dx={r['c2_acpc_dx_mm']:.2f}  "
                  f"C2_dz={r['c2_acpc_dz_mm']:.2f}  "
                  f"C3_dz={r['c3_eyes_dz_mm']:.2f}")

    # ------------------------------------------------------------------
    # Verdict
    # ------------------------------------------------------------------
    max_c1 = df["c1_ac_origin_mm"].max()
    max_c2 = df[["c2_acpc_dx_mm", "c2_acpc_dz_mm"]].max().max()
    max_c3 = df["c3_eyes_dz_mm"].max()
    acpc_mean = df["acpc_dist_mm"].mean()

    print("\n" + "=" * 65)
    print("VERDICT")
    print("=" * 65)
    verdict_ok = max_c1 < TOL_MM and max_c2 < TOL_MM and max_c3 < TOL_MM
    if verdict_ok:
        print(f"  PASS  All three constraints satisfied within {TOL_MM} mm.")
        print(f"        The labeled T1 images ARE in ACPC space and can be")
        print(f"        used as ground truth for the full pipeline.")
    else:
        issues = []
        if max_c1 >= TOL_MM:
            issues.append(f"C1 max={max_c1:.2f} mm (AC not at origin)")
        if max_c2 >= TOL_MM:
            issues.append(f"C2 max={max_c2:.2f} mm (AC–PC X or Z offset)")
        if max_c3 >= TOL_MM:
            issues.append(f"C3 max={max_c3:.2f} mm (LE–RE Z offset)")
        print(f"  FAIL  Issues: {'; '.join(issues)}")
    print(f"        Mean AC–PC distance: {acpc_mean:.2f} mm "
          f"(expected 23–30 mm in adults)")
    print("=" * 65)


if __name__ == "__main__":
    main()
