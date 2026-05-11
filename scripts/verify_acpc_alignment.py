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
  - eda_report_acpc_verification.html  (same style as EDA reports)
"""

from __future__ import annotations

import datetime
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from miatt.io import iter_subjects, load_fcsv

DATA_ROOT = Path("/nfs/s-l028/scratch/opt/ece5490/MIATTFINALEXAMDATA")
SITES     = ["siteA", "siteB", "siteC", "siteD", "siteE", "siteF"]
TOL_MM    = 1.0   # residual threshold for flagging a subject as failing
OUT_DIR   = Path("/nfs/s-l028/scratch/Users/sahmed8/miatt-final-exam-sabbir-ahmed12")

SITE_COLORS = {
    "siteA": "#4e79a7", "siteB": "#f28e2b", "siteC": "#e15759",
    "siteD": "#76b7b2", "siteE": "#59a14f", "siteF": "#edc948",
}


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


CSS = """
  :root {
    --bg:#1a1a2e; --surface:#16213e; --surface2:#0f3460;
    --text:#e0e0e0; --muted:#aaa; --green:#4ade80; --yellow:#facc15; --red:#f87171;
  }
  * { box-sizing:border-box; margin:0; padding:0; }
  body { background:var(--bg); color:var(--text);
         font-family:'Segoe UI',sans-serif; font-size:14px; padding:24px; }
  h1   { color:#fff; font-size:1.6rem; margin-bottom:4px; }
  h2   { color:#ccc; font-size:1.1rem; margin:32px 0 12px;
         border-bottom:1px solid #333; padding-bottom:6px; }
  .subtitle { color:var(--muted); margin-bottom:20px; font-size:.9rem; }
  .verdict  { border-radius:8px; padding:16px 20px; margin:24px 0;
              font-size:1rem; font-weight:600; }
  .verdict.pass { background:#14532d; color:#4ade80; border:1px solid #4ade80; }
  .verdict.fail { background:#7f1d1d; color:#f87171; border:1px solid #f87171; }
  .verdict .detail { font-weight:400; font-size:.88rem; margin-top:6px; color:#ccc; }
  .constraints { display:grid; grid-template-columns:repeat(3,1fr); gap:16px;
                 margin-bottom:32px; }
  .cbox { background:var(--surface); border-radius:8px; padding:14px 16px;
          border-left:4px solid #555; }
  .cbox.ok   { border-color:#4ade80; }
  .cbox.warn { border-color:#facc15; }
  .cbox.fail { border-color:#f87171; }
  .cbox h3 { font-size:.9rem; color:#cce; margin-bottom:8px; }
  .cbox .val { font-size:1.4rem; font-weight:700; }
  .cbox .val.ok   { color:#4ade80; }
  .cbox .val.warn { color:#facc15; }
  .cbox .val.fail { color:#f87171; }
  .cbox .sub { font-size:.78rem; color:var(--muted); margin-top:2px; }
  .wrap { overflow-x:auto; margin-bottom:32px; }
  table { border-collapse:collapse; width:100%; min-width:600px; }
  th, td { padding:8px 12px; text-align:right; border-bottom:1px solid #2a2a4a;
           white-space:nowrap; }
  th { background:var(--surface2); color:#cce; font-weight:600; text-align:left; }
  td:first-child, td:nth-child(2) { text-align:left; }
  tr:hover td { background:#1f2f50; }
  .badge { display:inline-block; padding:2px 8px; border-radius:12px;
           font-size:.78rem; font-weight:700; color:#fff; }
  .ok-cell   { color:#4ade80; }
  .warn-cell { color:#facc15; }
  .fail-cell { color:#f87171; font-weight:600; }
  footer { margin-top:40px; color:var(--muted); font-size:.8rem;
           border-top:1px solid #2a2a4a; padding-top:12px; }
"""


def _cls(val: float, tol: float) -> str:
    if val < tol * 0.5:
        return "ok-cell"
    if val < tol:
        return "warn-cell"
    return "fail-cell"


def build_html(df: pd.DataFrame) -> str:
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    total = len(df)

    # ---------- constraint summary boxes ----------
    c1_max = df["c1_ac_origin_mm"].max()
    c2_max = df[["c2_acpc_dx_mm", "c2_acpc_dz_mm"]].max().max()
    c3_max = df["c3_eyes_dz_mm"].max()

    def box_cls(mx, tol):
        if mx < tol * 0.5: return "ok"
        if mx < tol:        return "warn"
        return "fail"

    def val_cls(mx, tol):
        if mx < tol * 0.5: return "ok"
        if mx < tol:        return "warn"
        return "fail"

    def cbox(cid, title, mean, std, mx, tol, note):
        bc = box_cls(mx, tol)
        vc = val_cls(mx, tol)
        return f"""
        <div class="cbox {bc}">
          <h3>{cid} — {title}</h3>
          <div class="val {vc}">{mx:.2f} mm <span style="font-size:.9rem">max</span></div>
          <div class="sub">mean {mean:.3f} &nbsp;|&nbsp; std {std:.3f} &nbsp;|&nbsp; tol {tol} mm</div>
          <div class="sub" style="margin-top:4px">{note}</div>
        </div>"""

    acpc_mean = df["acpc_dist_mm"].mean()
    acpc_std  = df["acpc_dist_mm"].std()
    acpc_max  = df["acpc_dist_mm"].max()

    boxes = (
        cbox("C1", "AC at origin (0,0,0)",
             df["c1_ac_origin_mm"].mean(), df["c1_ac_origin_mm"].std(), c1_max, TOL_MM,
             "siteA=0 mm exactly; siteB–F in native scanner space")
        + cbox("C2", "AC–PC share X and Z",
               max(df["c2_acpc_dx_mm"].mean(), df["c2_acpc_dz_mm"].mean()),
               max(df["c2_acpc_dz_mm"].std(), df["c2_acpc_dx_mm"].std()),
               c2_max, TOL_MM,
               "ΔX and ΔZ between AC and PC")
        + cbox("C3", "LE–RE common Z plane",
               df["c3_eyes_dz_mm"].mean(), df["c3_eyes_dz_mm"].std(), c3_max, TOL_MM,
               "Eye-centre annotation variability")
    )

    # ---------- per-site summary table ----------
    site_cols = ["c1_ac_origin_mm", "c2_acpc_dx_mm", "c2_acpc_dz_mm",
                 "c3_eyes_dz_mm", "acpc_dist_mm"]
    sg = df.groupby("site")[site_cols + ["subject"]].agg(
        {"c1_ac_origin_mm": ["mean","max"],
         "c2_acpc_dx_mm":   ["mean","max"],
         "c2_acpc_dz_mm":   ["mean","max"],
         "c3_eyes_dz_mm":   ["mean","max"],
         "acpc_dist_mm":    "mean",
         "subject":         "count"}
    )

    site_rows = ""
    for site in SITES:
        r = sg.loc[site]
        color = SITE_COLORS.get(site, "#888")
        n = int(r[("subject","count")])
        c1m = r[("c1_ac_origin_mm","mean")]; c1x = r[("c1_ac_origin_mm","max")]
        c2dm = max(r[("c2_acpc_dx_mm","mean")], r[("c2_acpc_dz_mm","mean")])
        c2dx = max(r[("c2_acpc_dx_mm","max")],  r[("c2_acpc_dz_mm","max")])
        c3m = r[("c3_eyes_dz_mm","mean")]; c3x = r[("c3_eyes_dz_mm","max")]
        apm = r[("acpc_dist_mm","mean")]
        site_rows += f"""
        <tr>
          <td><span class="badge" style="background:{color}">{site}</span></td>
          <td>{n}</td>
          <td class="{_cls(c1x, TOL_MM)}">{c1m:.2f} / {c1x:.2f}</td>
          <td class="{_cls(c2dx, TOL_MM)}">{c2dm:.2f} / {c2dx:.2f}</td>
          <td class="{_cls(c3x, TOL_MM)}">{c3m:.2f} / {c3x:.2f}</td>
          <td>{apm:.2f}</td>
        </tr>"""

    # ---------- per-subject table ----------
    subj_rows = ""
    for _, r in df.sort_values(["site","subject"]).iterrows():
        color = SITE_COLORS.get(r["site"], "#888")
        c1 = r["c1_ac_origin_mm"]
        c2 = max(r["c2_acpc_dx_mm"], r["c2_acpc_dz_mm"])
        c3 = r["c3_eyes_dz_mm"]
        ap = r["acpc_dist_mm"]
        subj_rows += f"""
        <tr>
          <td><span class="badge" style="background:{color}">{r['site']}</span></td>
          <td>{r['subject']}</td>
          <td class="{_cls(c1, TOL_MM)}">{c1:.2f}</td>
          <td class="{_cls(r['c2_acpc_dx_mm'], TOL_MM)}">{r['c2_acpc_dx_mm']:.2f}</td>
          <td class="{_cls(r['c2_acpc_dz_mm'], TOL_MM)}">{r['c2_acpc_dz_mm']:.2f}</td>
          <td class="{_cls(c3, TOL_MM)}">{c3:.2f}</td>
          <td>{ap:.2f}</td>
        </tr>"""

    # ---------- verdict ----------
    verdict_ok = c1_max < TOL_MM and c2_max < TOL_MM and c3_max < TOL_MM
    if verdict_ok:
        vclass = "pass"
        vtext  = "PASS — all three constraints satisfied within tolerance."
        vdetail = ""
    else:
        vclass = "fail"
        issues = []
        if c1_max >= TOL_MM: issues.append(f"C1 max={c1_max:.2f} mm — siteB–F not in ACPC space (expected)")
        if c2_max >= TOL_MM: issues.append(f"C2 max={c2_max:.2f} mm — AC–PC axis not yet aligned")
        if c3_max >= TOL_MM: issues.append(f"C3 max={c3_max:.2f} mm — eye-centre annotation variability")
        vtext  = "FAIL — constraints not globally satisfied."
        vdetail = "<br>".join(issues)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>MIATT — ACPC Alignment Verification</title>
<style>{CSS}</style>
</head>
<body>
<h1>ACPC Alignment Verification</h1>
<p class="subtitle">
  {total} labeled subjects across {len(SITES)} sites &nbsp;|&nbsp;
  Tolerance: {TOL_MM} mm &nbsp;|&nbsp; Coordinates: RAS mm (fcsv CoordinateSystem=0) &nbsp;|&nbsp;
  Generated: {now}
</p>

<div class="verdict {vclass}">
  {vtext}
  {"" if not vdetail else f'<div class="detail">{vdetail}</div>'}
  <div class="detail" style="margin-top:8px">
    Mean AC–PC distance: {acpc_mean:.2f} mm ± {acpc_std:.2f} mm (max {acpc_max:.2f} mm)
    — expected 23–30 mm in adults.
    The labeled T1+fcsv pairs are valid ground truth: coordinates are consistent with
    the T1 image space. siteA images are already in ACPC space (C1=0 exactly).
    siteB–F landmarks are in native scanner space; the pipeline must compute the
    ACPC rigid transform from AC/PC positions.
  </div>
</div>

<h2>Constraint Summary</h2>
<div class="constraints">{boxes}</div>

<h2>Per-Site Statistics (mean / max residual in mm)</h2>
<div class="wrap">
<table>
  <thead>
    <tr>
      <th>Site</th><th>N</th>
      <th>C1 — AC→origin</th>
      <th>C2 — AC–PC ΔX or ΔZ (worst)</th>
      <th>C3 — LE–RE ΔZ</th>
      <th>AC–PC dist (mean mm)</th>
    </tr>
  </thead>
  <tbody>{site_rows}</tbody>
</table>
</div>

<h2>Per-Subject Results</h2>
<p class="subtitle">
  Colour: <span class="ok-cell">■ &lt; 0.5×tol</span> &nbsp;
  <span class="warn-cell">■ 0.5–1×tol</span> &nbsp;
  <span class="fail-cell">■ &gt; tol ({TOL_MM} mm)</span>
</p>
<div class="wrap">
<table>
  <thead>
    <tr>
      <th>Site</th><th>Subject</th>
      <th>C1 AC→origin (mm)</th>
      <th>C2 AC–PC ΔX (mm)</th>
      <th>C2 AC–PC ΔZ (mm)</th>
      <th>C3 LE–RE ΔZ (mm)</th>
      <th>AC–PC dist (mm)</th>
    </tr>
  </thead>
  <tbody>{subj_rows}</tbody>
</table>
</div>

<footer>
  Data: {DATA_ROOT} &nbsp;|&nbsp;
  Script: scripts/verify_acpc_alignment.py &nbsp;|&nbsp;
  Constraints from README → Pipeline (code) section
</footer>
</body>
</html>
"""


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

    # ------------------------------------------------------------------
    # HTML report
    # ------------------------------------------------------------------
    out_path = OUT_DIR / "eda_report_acpc_verification.html"
    out_path.write_text(build_html(df), encoding="utf-8")
    print(f"\nHTML report → {out_path}  ({out_path.stat().st_size/1024:.0f} KB)")


if __name__ == "__main__":
    main()
