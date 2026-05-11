"""
Dataset exploration script.
Samples 20 subjects per site (siteA–siteF, 120 total), captures axial/coronal/sagittal
slices for T1 and T2 separately, records voxel spacing, orientation, intensity range.
Outputs: eda_report_T1.html and eda_report_T2.html (self-contained, base64 images).
"""

import base64
import datetime
import io
import random
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk

DATA_ROOT = Path("/nfs/s-l028/scratch/opt/ece5490/MIATTFINALEXAMDATA")
SITES = ["siteA", "siteB", "siteC", "siteD", "siteE", "siteF"]
N_PER_SITE = 20
SEED = 42
OUT_DIR = Path("/nfs/s-l028/scratch/Users/sahmed8/miatt-final-exam-sabbir-ahmed12")

SITE_COLORS = {
    "siteA": "#4e79a7",
    "siteB": "#f28e2b",
    "siteC": "#e15759",
    "siteD": "#76b7b2",
    "siteE": "#59a14f",
    "siteF": "#edc948",
}

# --------------------------------------------------------------------------- #
# Sample 20 subjects per site (same subjects used for both T1 and T2)
# --------------------------------------------------------------------------- #
random.seed(SEED)

selected = []   # list of (site, subject_id, subject_dir)
for site in SITES:
    site_dir = DATA_ROOT / site
    subj_dirs = sorted([
        d for d in site_dir.iterdir()
        if d.is_dir() and not d.name.startswith(".")
    ])
    chosen = random.sample(subj_dirs, min(N_PER_SITE, len(subj_dirs)))
    chosen.sort()
    for subj_dir in chosen:
        selected.append((site, subj_dir.name, subj_dir))

print(f"Sampled {len(selected)} subjects ({N_PER_SITE} per site):")
for site, sid, _ in selected:
    print(f"  {site}/{sid}")


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def fig_to_b64(fig: plt.Figure) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight", facecolor="#111")
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


def orientation_str(img: sitk.Image) -> str:
    direction = img.GetDirection()
    dir_mat = np.array(direction).reshape(3, 3)
    labels = []
    for col in range(3):
        vec = dir_mat[:, col]
        dom = int(np.argmax(np.abs(vec)))
        sign = "+" if vec[dom] > 0 else "-"
        labels.append(f"{sign}{'RAS'[dom]}")
    return f"X={labels[0]}, Y={labels[1]}, Z={labels[2]}"


def process_subjects(selected, modality: str) -> list[dict]:
    """Load images for the given modality and return a list of record dicts."""
    records = []
    for idx, (site, subj_id, subj_dir) in enumerate(selected, 1):
        # find the modality file: t1_siteX.nii.gz or t2_siteX.nii.gz
        candidates = list(subj_dir.glob(f"{modality}_*.nii.gz"))
        if not candidates:
            print(f"  [WARN] No {modality} file in {subj_dir}, skipping")
            continue
        img_path = candidates[0]
        print(f"[{idx}/{len(selected)}] {site}/{subj_id} — {img_path.name}")

        img = sitk.ReadImage(str(img_path))
        arr = sitk.GetArrayFromImage(img)   # (Z, Y, X)

        spacing = img.GetSpacing()          # (x, y, z) mm
        origin  = img.GetOrigin()
        size_xyz = img.GetSize()            # (X, Y, Z)

        int_min = float(arr.min())
        int_max = float(arr.max())

        cz = arr.shape[0] // 2
        cy = arr.shape[1] // 2
        cx = arr.shape[2] // 2

        axial_sl    = arr[cz, :, :]
        coronal_sl  = arr[:, cy, :]
        sagittal_sl = arr[:, :, cx]

        fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), facecolor="#111")
        fig.suptitle(
            f"{site} / subject {subj_id}   |   {img_path.name}",
            color="white", fontsize=11, y=1.01
        )
        for ax, (sl, title) in zip(axes, [
            (axial_sl,    f"Axial  (z={cz})"),
            (coronal_sl,  f"Coronal (y={cy})"),
            (sagittal_sl, f"Sagittal (x={cx})"),
        ]):
            ax.imshow(sl, cmap="gray", origin="lower",
                      vmin=np.percentile(sl, 1), vmax=np.percentile(sl, 99))
            ax.set_title(title, color="white", fontsize=9)
            ax.axis("off")

        plt.tight_layout()
        img_b64 = fig_to_b64(fig)
        plt.close(fig)

        records.append({
            "site":        site,
            "subject":     subj_id,
            "file":        img_path.name,
            "size_vox":    size_xyz,
            "spacing_mm":  spacing,
            "orientation": orientation_str(img),
            "origin_mm":   origin,
            "int_min":     int_min,
            "int_max":     int_max,
            "img_b64":     img_b64,
        })
    return records


# --------------------------------------------------------------------------- #
# HTML builder
# --------------------------------------------------------------------------- #
def make_card(rec: dict, card_idx: int) -> str:
    site  = rec["site"]
    color = SITE_COLORS.get(site, "#888")
    sx, sy, sz = rec["spacing_mm"]
    vx, vy, vz = rec["size_vox"]
    ox, oy, oz = rec["origin_mm"]
    return f"""
    <div class="card" id="card-{card_idx}">
      <div class="card-header" style="border-left:5px solid {color};">
        <span class="badge" style="background:{color};">{site}</span>
        <span class="subj">Subject {rec['subject']}</span>
        <span class="fname">{rec['file']}</span>
      </div>
      <img src="data:image/png;base64,{rec['img_b64']}" alt="MRI views" />
      <table class="meta">
        <tr><th>Voxel spacing (mm)</th>
            <td>x={sx:.4f} &nbsp; y={sy:.4f} &nbsp; z={sz:.4f}</td></tr>
        <tr><th>Image size (voxels)</th>
            <td>X={vx} &nbsp; Y={vy} &nbsp; Z={vz}</td></tr>
        <tr><th>Orientation</th>
            <td>{rec['orientation']}</td></tr>
        <tr><th>Origin (mm)</th>
            <td>({ox:.2f}, {oy:.2f}, {oz:.2f})</td></tr>
        <tr><th>Intensity range</th>
            <td>min = {rec['int_min']:.1f} &nbsp;&nbsp; max = {rec['int_max']:.1f}</td></tr>
      </table>
    </div>"""


CSS = """
  :root {
    --bg:#1a1a2e; --surface:#16213e; --surface2:#0f3460;
    --text:#e0e0e0; --muted:#aaa;
  }
  * { box-sizing:border-box; margin:0; padding:0; }
  body { background:var(--bg); color:var(--text);
         font-family:'Segoe UI',sans-serif; font-size:14px; padding:24px; }
  h1   { color:#fff; font-size:1.6rem; margin-bottom:4px; }
  .subtitle { color:var(--muted); margin-bottom:28px; font-size:.9rem; }
  h2   { color:#ccc; font-size:1.1rem; margin:32px 0 12px;
         border-bottom:1px solid #333; padding-bottom:6px; }
  .summary-wrap { overflow-x:auto; margin-bottom:40px; }
  table.summary { border-collapse:collapse; width:100%; min-width:750px; }
  table.summary th, table.summary td {
    padding:8px 12px; text-align:left;
    border-bottom:1px solid #2a2a4a; white-space:nowrap; }
  table.summary th { background:var(--surface2); color:#cce; font-weight:600; }
  table.summary tr:hover td { background:#1f2f50; }
  .card { background:var(--surface); border-radius:8px; margin-bottom:28px;
          overflow:hidden; box-shadow:0 2px 8px rgba(0,0,0,.4); }
  .card-header { padding:10px 14px; background:var(--surface2);
                 display:flex; align-items:center; gap:12px; flex-wrap:wrap; }
  .subj { font-weight:700; font-size:1rem; }
  .fname { color:var(--muted); font-size:.82rem; font-family:monospace; }
  .card img { width:100%; display:block; }
  table.meta { width:100%; border-collapse:collapse; }
  table.meta th, table.meta td {
    padding:7px 14px; border-bottom:1px solid #1e2d50; text-align:left; }
  table.meta th { width:210px; color:var(--muted); font-weight:500;
                  white-space:nowrap; }
  table.meta tr:last-child td, table.meta tr:last-child th { border-bottom:none; }
  .badge { display:inline-block; padding:2px 8px; border-radius:12px;
           font-size:.78rem; font-weight:700; color:#fff; }
  footer { margin-top:40px; color:var(--muted); font-size:.8rem;
           border-top:1px solid #2a2a4a; padding-top:12px; }
"""


def build_html(records: list[dict], modality: str) -> str:
    summary_rows = ""
    for i, r in enumerate(records, 1):
        sx, sy, sz = r["spacing_mm"]
        vx, vy, vz = r["size_vox"]
        color = SITE_COLORS.get(r["site"], "#888")
        summary_rows += f"""
      <tr>
        <td>{i}</td>
        <td><span class="badge" style="background:{color};">{r['site']}</span></td>
        <td>{r['subject']}</td>
        <td>{sx:.4f} × {sy:.4f} × {sz:.4f}</td>
        <td>{vx} × {vy} × {vz}</td>
        <td>{r['orientation']}</td>
        <td>{r['int_min']:.0f} – {r['int_max']:.0f}</td>
      </tr>"""

    cards_html = "\n".join(make_card(r, i + 1) for i, r in enumerate(records))
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>MIATT Dataset Exploration — {modality.upper()} — 20 subjects/site</title>
<style>{CSS}</style>
</head>
<body>
<h1>MIATT Dataset Exploration — {modality.upper()}</h1>
<p class="subtitle">
  {N_PER_SITE} randomly sampled subjects per site (seed={SEED}) &nbsp;|&nbsp;
  {len(SITES)} sites × {N_PER_SITE} = {len(SITES)*N_PER_SITE} subjects total &nbsp;|&nbsp;
  Modality: <strong>{modality.upper()}</strong> &nbsp;|&nbsp;
  Centre slices shown &nbsp;|&nbsp; Generated: {now}
</p>

<h2>Summary Table</h2>
<div class="summary-wrap">
<table class="summary">
  <thead>
    <tr>
      <th>#</th><th>Site</th><th>Subject</th>
      <th>Spacing x×y×z (mm)</th><th>Size X×Y×Z (vox)</th>
      <th>Orientation</th><th>Intensity range</th>
    </tr>
  </thead>
  <tbody>{summary_rows}</tbody>
</table>
</div>

<h2>Per-Subject Views (Axial · Coronal · Sagittal)</h2>
{cards_html}

<footer>
  Data: {DATA_ROOT} &nbsp;|&nbsp;
  Script: scripts/explore_dataset.py &nbsp;|&nbsp;
  Slices at centre voxel of each axis
</footer>
</body>
</html>
"""


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
for modality in ("t1", "t2"):
    print(f"\n{'='*60}")
    print(f"Processing {modality.upper()} images …")
    print('='*60)
    records = process_subjects(selected, modality)
    out_path = OUT_DIR / f"eda_report_{modality.upper()}.html"
    out_path.write_text(build_html(records, modality), encoding="utf-8")
    print(f"\nReport written → {out_path}  ({out_path.stat().st_size/1024/1024:.1f} MB)")
