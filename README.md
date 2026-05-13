# Multi-Site Brain MRI Landmark Detection and ACPC Alignment

Robust multi-site anatomical landmark detection and ACPC alignment pipeline for heterogeneous brain MRI datasets. This project focuses on reliable landmark localisation under strong cross-site domain shift across six simulated MRI acquisition sites.

![A representative T1 image of brain MRI with landmarks](artifacts/t1_with_landmarks.png)

## Overview

This repository contains a fully reproducible pipeline that:

- Detects 51 anatomical landmarks in brain T1 MRI volumes
- Performs rigid ACPC alignment (AC at origin, AC–PC on the AP axis, LE–RE coplanar)
- Generalises across six heterogeneous acquisition sites without site-specific configuration
- Exports predictions in 3D Slicer-compatible `.fcsv` format
- Evaluates accuracy via mean Euclidean error on a deterministic 20% per-site holdout

The six sites differ in intensity scale, voxel spacing, field of view, and physical origin — mimicking real multi-site clinical studies.

## Site Characteristics

| Site | Subjects | Spacing (mm) | Intensity range | AC position |
|------|:--------:|:------------:|:---------------:|:-----------:|
| siteA | 108 + 27 | 1.00 isotropic | 0 – 4096 (12-bit) | Always (0,0,0) — pre-ACPC-aligned |
| siteB | 106 + 27 | ~1.02 isotropic | 0 – 255 (8-bit) | Arbitrary (σ ≈ 17 mm/axis) |
| siteC | 107 + 27 | ~1.01 isotropic | 0 – 6000 | Arbitrary (σ ≈ 17 mm/axis) |
| siteD | 107 + 27 | ~1.03 isotropic | 1500 – 13200 (non-zero floor) | Arbitrary (σ ≈ 18 mm/axis) |
| siteE | 107 + 27 | ~0.98 isotropic | 0 – 1 (pre-normalised) | Arbitrary (σ ≈ 17 mm/axis) |
| siteF | 107 + 27 | ~1.03 isotropic | −1000 – 1000 (HU-like) | Arbitrary (σ ≈ 17 mm/axis) |

All images have identity direction cosines in SimpleITK's LPS convention (axis-aligned, no rotation). Cross-site generalisation requires handling translational offsets and intensity scale variation, not arbitrary orientations.

## Detection Approaches

Six approaches were implemented and evaluated. All use the same deterministic 80/20 per-site split (~21 eval subjects per site).

| # | Approach | Overall error | Notes |
|---|----------|:-------------:|-------|
| 1 | Per-site mean in ACPC space | 4.59 mm | Baseline; correct formulation uses ACPC space to handle translational scatter |
| 2 | Rigid registration to ACPC template | 4.59 mm | Null result — algebraically equivalent to Approach 1 |
| 3 | Posterior-guided local refinement | 5.13 mm | Negative result — WM centroid adds noise for sub-structure boundaries |
| 4 | **3D CNN coordinate regression** | **4.54 mm** | **Best; submitted** |
| 5 | Multi-atlas affine registration | 5.51 mm | Destabilised by translational scatter in scanner space |
| 6 | BCD-inspired LLS from tissue posteriors | 4.97 mm | Matches CNN on siteA; degrades for B–F due to approximate ACPC transform |

### Approach 4 — 3D CNN (Submitted)

A 5-block stride-2 3D encoder CNN (channels 1→24→48→96→192→384) with adaptive average pooling and two FC layers (384→256→153, Tanh output) predicts all 51 landmarks from the ACPC-resampled T1 volume (96×101×101 voxels at 2 mm isotropic). Trained on all six sites combined (~516 subjects), 100 epochs, AdamW + cosine LR schedule, smooth-L1 loss. Coordinate targets are normalised by 120 mm to [−1, +1] — critical for stable training when landmarks span ±100 mm from the origin.

## Preprocessing

Every volume passes through a site-agnostic preprocessing pipeline before any detection:

1. **Reorientation** — convert from LPS (SimpleITK convention) to RAS
2. **Resampling** — resample to 1 mm isotropic using linear interpolation
3. **Z-score normalisation** — subtract mean, divide by std (computed over brain voxels)

Without intensity normalisation, a model trained on siteA (range 0–4096) would fail on siteF (range −1000–1000). For ACPC-space approaches, a fixed 2 mm isotropic template grid (101×101×96 voxels) is used as the resampling target.

## Landmark Prediction

All 51 BCD landmark labels are predicted for every unlabeled subject. Predictions are in scanner space (native T1 physical coordinates), matching the format of ground-truth `.fcsv` files. For subjects in sites B–F, AC is at a non-zero scanner position (e.g. siteB AC ≈ (−33, −6, −5) mm).

For downstream non-linear registration, we recommend using the 26 landmarks with mean CNN error ≤ 4 mm (AC, PC, corpus callosum family, caudate heads, lateral ventricles, optic chiasm, BPons, SMV, CM) as hard correspondence constraints, and excluding orbital and cortical-pole landmarks (LE 11.2 mm, RE 12.6 mm, cortical poles 6–9 mm).

## Repository Structure

```text
.
├── README.md
├── PRD.md                          # Full project design document
├── miatt_sahmed8.pdf               # Final report (9 pages)
├── pixi.toml                       # Environment specification
├── pixi.lock                       # Pinned lockfile — always committed
├── pyproject.toml
├── conftest.py
├── artifacts/
│   ├── t1_with_landmarks.png       # Representative T1 with landmark overlay
│   ├── miatt_sahmed8.tex           # LaTeX source for the report
│   ├── eda_report_T1.html          # EDA report — raw T1 images
│   ├── eda_report_T2.html          # EDA report — raw T2 images
│   ├── eda_report_T1_preprocessed.html   # EDA report — preprocessed T1
│   ├── eda_report_T2_preprocessed.html   # EDA report — preprocessed T2
│   ├── eda_report_acpc_verification.html # ACPC alignment verification report
│   └── qc_atlas_predictions.html  # QC report for multi-atlas predictions
├── src/miatt/
│   ├── acpc.py                     # ACPC rigid transform computation
│   ├── io.py                       # NIfTI load, .fcsv read/write
│   ├── landmarks.py                # Error metrics, landmark aggregation
│   ├── preprocessing.py            # Reorientation, resampling, z-score
│   ├── registration.py             # Approach 2 — ACPC template registration
│   ├── heuristic.py                # Approach 3 — posterior-weighted centroid
│   ├── cnn.py                      # Approach 4 — 3D CNN regression
│   ├── atlas.py                    # Approach 5 — multi-atlas affine registration
│   ├── lls.py                      # Approach 6 — BCD-inspired LLS
│   └── pipeline.py                 # Approach dispatcher and eval harness
├── scripts/
│   ├── run_pipeline.py             # Main CLI — runs any approach on all sites
│   ├── eda.py                      # EDA report generator
│   └── verify_acpc_alignment.py    # Geometric ACPC constraint checker
├── tests/                          # 40 pytest unit tests (all passing)
├── results/                        # Per-approach JSON + Markdown summaries
└── predictions/                    # 162 predicted .fcsv files (committed)
    └── site{A-F}_unlabeled/<subj>/BCD_ACPC_Landmarks.fcsv
```

## Environment Setup

This project uses [pixi](https://pixi.sh) for fully reproducible environment management (conda + PyPI lockfile).

**Prerequisites:** Linux workstation with NFS scratch space; NVIDIA GPU with CUDA 12.x+ for CNN training/inference.

```bash
# 1. Clone the repository
git clone https://github.com/sabbir-ahmed12/multi-site-landmark-detection.git
cd multi-site-landmark-detection

# 2. Install pixi (if not already present)
curl -fsSL https://pixi.sh/install.sh | bash

# 3. Install all dependencies (reads pixi.lock — fully pinned)
pixi install

# 4. Verify the environment
pixi run python -c "import SimpleITK, torch, monai; print('OK')"

# 5. Run the test suite
pixi run test
```

**Key dependencies:** Python 3.12, SimpleITK 2.5, PyTorch 2.3 (cu128), MONAI 1.3, NumPy, SciPy, Matplotlib, Seaborn, pandas.

> **Note on SimpleITK:** A lazy import pattern (`_sitk()` accessor) is used in all source modules to avoid a C-extension segfault triggered by pytest's assertion rewriter. Do not move `import SimpleITK` to module level in any file imported during test collection.

## Running the Pipeline

The data must be accessible at the path configured in `scripts/run_pipeline.py` (default: `/nfs/s-l028/scratch/opt/ece5490/MIATTFINALEXAMDATA`).

```bash
# Run a specific approach on all sites
pixi run pipeline --approach cnn --output predictions

# Available approaches: mean | registration | heuristic | cnn | atlas | lls
pixi run pipeline --approach mean --output predictions_mean
pixi run pipeline --approach lls --lls-alpha 10.0 --output predictions_lls

# Run EDA (generates HTML reports)
pixi run eda
```

Pre-generated EDA reports are available in `artifacts/` — open `eda_report_T1.html` or `eda_report_T2.html` in a browser to explore per-site intensity distributions, spacing statistics, and AC position scatter before running the pipeline.

Predictions are written to `predictions/site{X}_unlabeled/<subject>/BCD_ACPC_Landmarks.fcsv`. On the first run the CNN model checkpoint, ACPC templates, and per-subject ACPC volume cache are written to `cache/` and reused on subsequent runs.

## Evaluation

Internal evaluation runs automatically as part of `run_pipeline.py` using the deterministic 20% holdout. Results are saved to `results/<approach>.json` and `results/<approach>.md`.

```bash
# Re-run evaluation for any approach
pixi run pipeline --approach cnn --output predictions

# Run the test suite (unit tests for numerical routines)
pixi run test
```

## Results

Best result: **4.54 mm** mean Euclidean error (3D CNN, 20% per-site holdout, all 51 landmarks).

| Site | CNN error (mm) |
|------|:--------------:|
| siteA | 4.54 |
| siteB | 4.38 |
| siteC | 4.64 |
| siteD | 5.04 |
| siteE | 4.13 |
| siteF | 4.53 |
| **Overall** | **4.54** |

**Best landmarks (CNN):** AC 0.06 mm, PC 0.93 mm, rostrum 1.55 mm, mid_lat 1.63 mm, mid_basel 2.01 mm.

**Worst landmarks (CNN):** RE 12.56 mm, LE 11.23 mm — orbital landmarks with low T1 contrast and high inter-subject shape variation. Adding T2 as a second input channel is the highest-leverage future improvement.

See `results/` for full per-site, per-landmark breakdowns and `miatt_sahmed8.pdf` for the complete project report.

## Future Extensions

- T2 as a second input channel (highest-leverage improvement for LE/RE)
- Landmark-specific loss weighting to redistribute error budget
- Nonlinear registration initialisation using the 26 reliable landmarks
- Domain adaptation for unseen sites

## Acknowledgements

Developed as part of the Multi-Dimensional Image Analysis Tools (MIATT) coursework at the University of Iowa.

- Dr. Hans Johnson (course instructor)
- ITK, SimpleITK, and MONAI communities
- 3D Slicer developers

