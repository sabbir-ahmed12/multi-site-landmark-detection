"""
Standardisation pipeline + EDA report.

For each sampled subject:
  1. Load raw image, record header stats (orientation, spacing, intensity range).
  2. Apply preprocessing.preprocess():  reorient → resample → z-score normalise.
  3. Record post-processing stats.
  4. Render orthogonal slices of the standardised image.

Outputs:
  eda_report_T1_preprocessed.html
  eda_report_T2_preprocessed.html

These are self-contained HTML files (base64-embedded images) in the same
visual style as eda_report_T1.html / eda_report_T2.html.
"""

from __future__ import annotations

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

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from miatt.preprocessing import orientation_code, preprocess, TARGET_SPACING, TARGET_ORIENTATION

DATA_ROOT = Path("/nfs/s-l028/scratch/opt/ece5490/MIATTFINALEXAMDATA")
SITES     = ["siteA", "siteB", "siteC", "siteD", "siteE", "siteF"]
N_PER_SITE = 20
SEED       = 42          # same seed → same subjects as explore_dataset.py
OUT_DIR    = Path("/nfs/s-l028/scratch/Users/sahmed8/miatt-final-exam-sabbir-ahmed12")

SITE_COLORS = {
    "siteA": "#4e79a7", "siteB": "#f28e2b", "siteC": "#e15759",
    "siteD": "#76b7b2", "siteE": "#59a14f", "siteF": "#edc948",
}

# ---------------------------------------------------------------------------
# Sample subjects (identical selection to explore_dataset.py)
# ---------------------------------------------------------------------------
random.seed(SEED)

selected = []
for site in SITES:
    site_dir = DATA_ROOT / site
    subj_dirs = sorted([d for d in site_dir.iterdir()
                        if d.is_dir() and not d.name.startswith(".")])
    chosen = sorted(random.sample(subj_dirs, min(N_PER_SITE, len(subj_dirs))))
    for subj_dir in chosen:
        selected.append((site, subj_dir.name, subj_dir))

print(f"Sampled {len(selected)} subjects ({N_PER_SITE} per site).")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def fig_to_b64(fig: plt.Figure) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight", facecolor="#111")
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


def process_subjects(selected: list, modality: str) -> list[dict]:
    records = []
    for idx, (site, subj_id, subj_dir) in enumerate(selected, 1):
        candidates = list(subj_dir.glob(f"{modality}_*.nii.gz"))
        if not candidates:
            print(f"  [WARN] No {modality} in {subj_dir}, skipping")
            continue
        img_path = candidates[0]
        print(f"[{idx}/{len(selected)}] {site}/{subj_id} — {img_path.name}", flush=True)

        raw = sitk.ReadImage(str(img_path))
        raw_arr = sitk.GetArrayFromImage(raw)

        raw_orient  = orientation_code(raw)
        raw_spacing = raw.GetSpacing()          # (x, y, z) mm
        raw_size    = raw.GetSize()             # (X, Y, Z)
        raw_min     = float(raw_arr.min())
        raw_max     = float(raw_arr.max())

        # --- preprocessing pipeline ---
        std_img = preprocess(raw)
        std_arr = sitk.GetArrayFromImage(std_img).astype(np.float32)  # (Z, Y, X)

        std_orient  = orientation_code(std_img)
        std_spacing = std_img.GetSpacing()
        std_size    = std_img.GetSize()
        std_mean    = float(std_arr.mean())
        std_std     = float(std_arr.std())
        # clip for display: 3-sigma window covers most brain tissue
        display = np.clip(std_arr, -3.0, 3.0)

        cz = display.shape[0] // 2
        cy = display.shape[1] // 2
        cx = display.shape[2] // 2

        fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), facecolor="#111")
        fig.suptitle(
            f"{site} / {subj_id}  |  {img_path.name}  →  standardised",
            color="white", fontsize=10, y=1.01,
        )
        for ax, (sl, title) in zip(axes, [
            (display[cz, :, :],  f"Axial (z={cz})"),
            (display[:, cy, :],  f"Coronal (y={cy})"),
            (display[:, :, cx],  f"Sagittal (x={cx})"),
        ]):
            ax.imshow(sl, cmap="gray", origin="lower",
                      vmin=np.percentile(sl, 1), vmax=np.percentile(sl, 99))
            ax.set_title(title, color="white", fontsize=9)
            ax.axis("off")

        plt.tight_layout()
        img_b64 = fig_to_b64(fig)
        plt.close(fig)

        records.append({
            "site":         site,
            "subject":      subj_id,
            "file":         img_path.name,
            # raw
            "raw_spacing":  raw_spacing,
            "raw_size":     raw_size,
            "raw_orient":   raw_orient,
            "raw_min":      raw_min,
            "raw_max":      raw_max,
            # standardised
            "std_spacing":  std_spacing,
            "std_size":     std_size,
            "std_orient":   std_orient,
            "std_mean":     std_mean,
            "std_std":      std_std,
            "img_b64":      img_b64,
        })
    return records


# ---------------------------------------------------------------------------
# HTML
# ---------------------------------------------------------------------------

CSS = """
  :root {
    --bg:#1a1a2e; --surface:#16213e; --surface2:#0f3460;
    --text:#e0e0e0; --muted:#aaa; --green:#4ade80; --yellow:#facc15;
  }
  * { box-sizing:border-box; margin:0; padding:0; }
  body { background:var(--bg); color:var(--text);
         font-family:'Segoe UI',sans-serif; font-size:14px; padding:24px; }
  h1   { color:#fff; font-size:1.6rem; margin-bottom:4px; }
  .subtitle { color:var(--muted); margin-bottom:12px; font-size:.9rem; }
  .pipeline { display:flex; gap:0; align-items:center; margin-bottom:28px; flex-wrap:wrap; }
  .step { background:var(--surface2); border-radius:6px; padding:6px 14px;
          font-size:.82rem; font-weight:600; color:#cce; }
  .arrow { color:var(--muted); padding:0 8px; font-size:1.1rem; }
  h2   { color:#ccc; font-size:1.1rem; margin:32px 0 12px;
         border-bottom:1px solid #333; padding-bottom:6px; }
  .summary-wrap { overflow-x:auto; margin-bottom:40px; }
  table.summary { border-collapse:collapse; width:100%; min-width:900px; }
  table.summary th, table.summary td {
    padding:8px 12px; text-align:left;
    border-bottom:1px solid #2a2a4a; white-space:nowrap; }
  table.summary th { background:var(--surface2); color:#cce; font-weight:600; }
  table.summary tr:hover td { background:#1f2f50; }
  .changed { color:var(--yellow); font-weight:600; }
  .ok      { color:var(--green); }
  .card { background:var(--surface); border-radius:8px; margin-bottom:28px;
          overflow:hidden; box-shadow:0 2px 8px rgba(0,0,0,.4); }
  .card-header { padding:10px 14px; background:var(--surface2);
                 display:flex; align-items:center; gap:12px; flex-wrap:wrap; }
  .subj  { font-weight:700; font-size:1rem; }
  .fname { color:var(--muted); font-size:.82rem; font-family:monospace; }
  .card img { width:100%; display:block; }
  table.meta { width:100%; border-collapse:collapse; }
  table.meta th, table.meta td {
    padding:7px 14px; border-bottom:1px solid #1e2d50; text-align:left; }
  table.meta th { width:220px; color:var(--muted); font-weight:500; white-space:nowrap; }
  table.meta .raw { color:#aaa; }
  table.meta .std { color:var(--green); font-weight:600; }
  table.meta tr:last-child td, table.meta tr:last-child th { border-bottom:none; }
  .badge { display:inline-block; padding:2px 8px; border-radius:12px;
           font-size:.78rem; font-weight:700; color:#fff; }
  .std-badge { background:#166534; color:#4ade80; border:1px solid #4ade80;
               padding:2px 8px; border-radius:12px; font-size:.75rem; font-weight:700; }
  footer { margin-top:40px; color:var(--muted); font-size:.8rem;
           border-top:1px solid #2a2a4a; padding-top:12px; }
"""


def make_card(rec: dict, card_idx: int) -> str:
    site  = rec["site"]
    color = SITE_COLORS.get(site, "#888")
    rsx, rsy, rsz = rec["raw_spacing"]
    rvx, rvy, rvz = rec["raw_size"]
    ssx, ssy, ssz = rec["std_spacing"]
    svx, svy, svz = rec["std_size"]

    orient_changed = rec["raw_orient"] != rec["std_orient"]
    orient_raw_cls = "changed" if orient_changed else "ok"

    return f"""
    <div class="card" id="card-{card_idx}">
      <div class="card-header" style="border-left:5px solid {color};">
        <span class="badge" style="background:{color};">{site}</span>
        <span class="subj">Subject {rec['subject']}</span>
        <span class="fname">{rec['file']}</span>
        <span class="std-badge">standardised</span>
      </div>
      <img src="data:image/png;base64,{rec['img_b64']}" alt="MRI views (standardised)" />
      <table class="meta">
        <tr>
          <th>Spacing — raw (mm)</th>
          <td class="raw">x={rsx:.4f} &nbsp; y={rsy:.4f} &nbsp; z={rsz:.4f}</td>
        </tr>
        <tr>
          <th>Spacing — standardised</th>
          <td class="std">x={ssx:.4f} &nbsp; y={ssy:.4f} &nbsp; z={ssz:.4f}</td>
        </tr>
        <tr>
          <th>Size (vox) — raw</th>
          <td class="raw">X={rvx} &nbsp; Y={rvy} &nbsp; Z={rvz}</td>
        </tr>
        <tr>
          <th>Size (vox) — standardised</th>
          <td class="std">X={svx} &nbsp; Y={svy} &nbsp; Z={svz}</td>
        </tr>
        <tr>
          <th>Orientation — raw</th>
          <td class="{orient_raw_cls}">{rec['raw_orient']}</td>
        </tr>
        <tr>
          <th>Orientation — standardised</th>
          <td class="std">{rec['std_orient']}</td>
        </tr>
        <tr>
          <th>Intensity — raw range</th>
          <td class="raw">min = {rec['raw_min']:.1f} &nbsp;&nbsp; max = {rec['raw_max']:.1f}</td>
        </tr>
        <tr>
          <th>Intensity — z-score (μ / σ)</th>
          <td class="std">mean = {rec['std_mean']:.3f} &nbsp;&nbsp; std = {rec['std_std']:.3f}</td>
        </tr>
      </table>
    </div>"""


def build_html(records: list[dict], modality: str) -> str:
    tsp = TARGET_SPACING
    summary_rows = ""
    for i, r in enumerate(records, 1):
        rsx, rsy, rsz = r["raw_spacing"]
        ssx, ssy, ssz = r["std_spacing"]
        color = SITE_COLORS.get(r["site"], "#888")
        orient_changed = r["raw_orient"] != r["std_orient"]
        orient_cls = "changed" if orient_changed else "ok"
        summary_rows += f"""
      <tr>
        <td>{i}</td>
        <td><span class="badge" style="background:{color};">{r['site']}</span></td>
        <td>{r['subject']}</td>
        <td class="{orient_cls}">{r['raw_orient']} → {r['std_orient']}</td>
        <td>{rsx:.3f}×{rsy:.3f}×{rsz:.3f} → <span class="ok">{ssx:.1f}×{ssy:.1f}×{ssz:.1f}</span></td>
        <td>[{r['raw_min']:.0f}, {r['raw_max']:.0f}] → <span class="ok">μ={r['std_mean']:.2f} σ={r['std_std']:.2f}</span></td>
      </tr>"""

    cards_html = "\n".join(make_card(r, i + 1) for i, r in enumerate(records))
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>MIATT Preprocessed — {modality.upper()} — {N_PER_SITE} subjects/site</title>
<style>{CSS}</style>
</head>
<body>
<h1>MIATT Standardisation Report — {modality.upper()}</h1>
<p class="subtitle">
  {N_PER_SITE} randomly sampled subjects per site (seed={SEED}) &nbsp;|&nbsp;
  {len(SITES)} sites × {N_PER_SITE} = {len(SITES)*N_PER_SITE} subjects total &nbsp;|&nbsp;
  Modality: <strong>{modality.upper()}</strong> &nbsp;|&nbsp; Generated: {now}
</p>
<div class="pipeline">
  <span class="step">① Reorient → {TARGET_ORIENTATION}</span>
  <span class="arrow">→</span>
  <span class="step">② Resample → {tsp[0]:.1f}×{tsp[1]:.1f}×{tsp[2]:.1f} mm isotropic</span>
  <span class="arrow">→</span>
  <span class="step">③ Z-score normalise (foreground-masked, 15th-pct threshold)</span>
</div>

<h2>Summary — Before → After</h2>
<div class="summary-wrap">
<table class="summary">
  <thead>
    <tr>
      <th>#</th><th>Site</th><th>Subject</th>
      <th>Orientation</th>
      <th>Spacing (mm)</th>
      <th>Intensity</th>
    </tr>
  </thead>
  <tbody>{summary_rows}</tbody>
</table>
</div>

<h2>Per-Subject Views — Standardised (Axial · Coronal · Sagittal)</h2>
<p class="subtitle" style="margin-bottom:16px;">
  Slices at centre voxel &nbsp;|&nbsp; Display clipped to ±3σ &nbsp;|&nbsp;
  Green = standardised values &nbsp;|&nbsp; Yellow = orientation changed
</p>
{cards_html}

<footer>
  Data: {DATA_ROOT} &nbsp;|&nbsp;
  Script: scripts/preprocess_and_report.py &nbsp;|&nbsp;
  Pipeline: {TARGET_ORIENTATION} orientation, {tsp[0]:.1f} mm isotropic, z-score (fg-masked)
</footer>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
for modality in ("t1", "t2"):
    print(f"\n{'='*60}")
    print(f"Processing {modality.upper()} images …")
    print("="*60)
    records = process_subjects(selected, modality)
    out_path = OUT_DIR / f"eda_report_{modality.upper()}_preprocessed.html"
    out_path.write_text(build_html(records, modality), encoding="utf-8")
    print(f"\nReport written → {out_path}  ({out_path.stat().st_size/1024/1024:.1f} MB)")
