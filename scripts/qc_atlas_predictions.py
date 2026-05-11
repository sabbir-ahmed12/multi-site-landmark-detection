"""
QC report for atlas-based predictions.

Runs atlas prediction on one labeled subject per site (using ground-truth
landmarks for comparison), renders axial/coronal/sagittal slices at the
predicted AC with predicted landmarks (red circles) and true landmarks
(green crosses) overlaid, and writes a self-contained HTML report.

Usage:
    .pixi/envs/default/bin/python scripts/qc_atlas_predictions.py

Output:
    qc_atlas_predictions.html
"""

from __future__ import annotations

import base64
import datetime
import io
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from miatt.atlas import (
    prep_for_registration,
    predict_landmarks_atlas,
    select_atlases,
    visualize_predictions,
)
from miatt.io import iter_subjects, load_fcsv
from miatt.landmarks import mean_euclidean_error

DATA_ROOT  = Path("/nfs/s-l028/scratch/opt/ece5490/MIATTFINALEXAMDATA")
SITES      = ["siteA", "siteB", "siteC", "siteD", "siteE", "siteF"]
OUT_PATH   = Path("/nfs/s-l028/scratch/Users/sahmed8/miatt-final-exam-sabbir-ahmed12/qc_atlas_predictions.html")
N_ATLASES  = 5
N_ITER     = 200
SHOW_LM    = ["AC", "PC", "LE", "RE", "optic_chiasm", "genu", "rostrum"]

SITE_COLORS = {
    "siteA": "#4e79a7", "siteB": "#f28e2b", "siteC": "#e15759",
    "siteD": "#76b7b2", "siteE": "#59a14f", "siteF": "#edc948",
}


def fig_to_b64(fig: plt.Figure) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight", facecolor="#111")
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


def main() -> None:
    # --- load atlases once ---------------------------------------------------
    print(f"Loading {N_ATLASES} atlas(es) from siteA …", flush=True)
    atlas_pairs = select_atlases(DATA_ROOT, n=N_ATLASES, skip_first_n=0)
    atlas_images: list[sitk.Image] = []
    atlas_landmarks: list[dict] = []
    for atlas_dir, atlas_fcsv in atlas_pairs:
        t1_path = atlas_dir / "t1_siteA.nii.gz"
        if not t1_path.exists():
            continue
        img = sitk.ReadImage(str(t1_path))
        atlas_images.append(prep_for_registration(img))
        atlas_landmarks.append(load_fcsv(atlas_fcsv))
    print(f"  {len(atlas_images)} atlas(es) ready.\n")

    records = []

    for site in SITES:
        # pick the first labeled subject that has all ACPC landmarks
        subject_dir = fcsv_path = None
        for sd, fp in iter_subjects(DATA_ROOT, site, labeled=True):
            if fp is None:
                continue
            lm = load_fcsv(fp)
            if {"AC", "PC", "LE", "RE"}.issubset(lm):
                subject_dir, fcsv_path = sd, fp
                break

        if subject_dir is None:
            print(f"[{site}] no suitable subject found, skipping")
            continue

        t1_path = subject_dir / f"t1_{site}.nii.gz"
        print(f"[{site}] {subject_dir.name} — predicting …", flush=True)

        raw_img   = sitk.ReadImage(str(t1_path))
        subj_prep = prep_for_registration(raw_img)
        true_lm   = load_fcsv(fcsv_path)

        pred_lm = predict_landmarks_atlas(
            subj_prep, atlas_images, atlas_landmarks,
            n_iterations=N_ITER, verbose=True,
        )

        err = mean_euclidean_error(pred_lm, true_lm)
        print(f"  mean landmark error (scanner space): {err:.2f} mm")

        fig = visualize_predictions(raw_img, pred_lm, true_lm, landmarks_to_show=SHOW_LM)
        img_b64 = fig_to_b64(fig)
        plt.close(fig)

        # per-landmark table rows (ACPC subset only)
        lm_rows = ""
        for name in SHOW_LM:
            if name not in pred_lm or name not in true_lm:
                continue
            d = float(np.linalg.norm(pred_lm[name] - true_lm[name]))
            lm_rows += f"<tr><td>{name}</td><td>{d:.2f}</td></tr>"

        records.append({
            "site": site, "subject": subject_dir.name,
            "err": err, "img_b64": img_b64, "lm_rows": lm_rows,
        })

    # --- build HTML ----------------------------------------------------------
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    cards = ""
    for r in records:
        color = SITE_COLORS.get(r["site"], "#888")
        cards += f"""
        <div class="card">
          <div class="card-header" style="border-left:5px solid {color};">
            <span class="badge" style="background:{color};">{r['site']}</span>
            <span class="subj">Subject {r['subject']}</span>
            <span class="err">mean err = {r['err']:.2f} mm (scanner space)</span>
          </div>
          <img src="data:image/png;base64,{r['img_b64']}" />
          <table class="lm-table">
            <thead><tr><th>Landmark</th><th>Error (mm)</th></tr></thead>
            <tbody>{r['lm_rows']}</tbody>
          </table>
        </div>"""

    html = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="UTF-8">
<title>QC — Atlas Predictions</title>
<style>
  :root {{--bg:#1a1a2e;--surface:#16213e;--surface2:#0f3460;--text:#e0e0e0;--muted:#aaa;}}
  *{{box-sizing:border-box;margin:0;padding:0;}}
  body{{background:var(--bg);color:var(--text);font-family:'Segoe UI',sans-serif;
        font-size:14px;padding:24px;}}
  h1{{color:#fff;font-size:1.5rem;margin-bottom:4px;}}
  .subtitle{{color:var(--muted);margin-bottom:24px;font-size:.9rem;}}
  .card{{background:var(--surface);border-radius:8px;margin-bottom:28px;
         overflow:hidden;box-shadow:0 2px 8px rgba(0,0,0,.4);}}
  .card-header{{padding:10px 14px;background:var(--surface2);
                display:flex;align-items:center;gap:12px;flex-wrap:wrap;}}
  .subj{{font-weight:700;}}
  .err{{color:#facc15;font-size:.85rem;}}
  .card img{{width:100%;display:block;}}
  .badge{{display:inline-block;padding:2px 8px;border-radius:12px;
          font-size:.78rem;font-weight:700;color:#fff;}}
  table.lm-table{{width:100%;border-collapse:collapse;}}
  table.lm-table th,table.lm-table td{{padding:6px 14px;border-bottom:1px solid #1e2d50;}}
  table.lm-table th{{background:var(--surface2);color:#cce;}}
  footer{{margin-top:40px;color:var(--muted);font-size:.8rem;
          border-top:1px solid #2a2a4a;padding-top:12px;}}
</style></head><body>
<h1>QC — Atlas-Based Landmark Predictions</h1>
<p class="subtitle">
  {N_ATLASES} siteA atlases &nbsp;|&nbsp; Affine registration, {N_ITER} iter/level &nbsp;|&nbsp;
  Red circles = predicted &nbsp;|&nbsp; Green crosses = ground truth &nbsp;|&nbsp;
  Slices at predicted AC &nbsp;|&nbsp; Generated: {now}
</p>
{cards}
<footer>Script: scripts/qc_atlas_predictions.py &nbsp;|&nbsp; Data: {DATA_ROOT}</footer>
</body></html>"""

    OUT_PATH.write_text(html, encoding="utf-8")
    print(f"\nQC report written → {OUT_PATH}  ({OUT_PATH.stat().st_size/1024:.0f} KB)")


if __name__ == "__main__":
    main()
