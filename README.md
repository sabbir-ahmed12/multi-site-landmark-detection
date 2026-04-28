[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/yYFqmZrA)
# MIATT Final Exam — Multi-Site Landmark Detection

## What this dataset is

804 brain MRI subjects split across **six simulated sites** (`siteA`
… `siteF`). Each site has its own quirks to mimic
the heterogeneity of a real multi-site neuroimaging study.

Your goal is to write code that **detects anatomical landmarks and
ACPC-aligns** every subject regardless of which site it came from,
then reports how well your method works on a held-out test set whose
ground-truth landmarks have been withheld.

## Data location and access policy

```
/nfs/s-l028/scratch/opt/ece5490/MIATTFINALEXAMDATA/
```

> **Data must stay on the engineering computers.** Copying any file in
> this directory to a personal laptop, cloud storage, USB device, or
> off-site VM is an academic misconduct violation. This rule simulates
> working inside a real data enclave or restricted cloud
> environment — the conditions you will face on most clinical
> projects after graduation.

## Directory layout

```
MIATTFINALEXAMDATA/
├── siteA/<subject>/                    ← labeled (training) set — 108 subjects
│   ├── t1_siteA.nii.gz
│   ├── t2_siteA.nii.gz
│   ├── BCD_ACPC_Landmarks.fcsv         ← ground-truth landmarks
│   └── ACCUMULATED_POSTERIORS/         ← 6 tissue probability maps
├── siteA_unlabeled/<subject>/          ← held-out (test) set — 27 subjects
│   ├── t1_siteA.nii.gz
│   ├── t2_siteA.nii.gz
│   └── ACCUMULATED_POSTERIORS/         ← NO .fcsv — that is the exam
├── siteB/                siteB_unlabeled/
├── siteC/                siteC_unlabeled/
├── siteD/                siteD_unlabeled/
├── siteE/                siteE_unlabeled/
└── siteF/                siteF_unlabeled/
```

For each subject in `site*_unlabeled/` you must produce a predicted
`BCD_ACPC_Landmarks.fcsv` of the same format as the labeled subjects.
The instructor will score your predictions against the withheld
ground truth.

## Site characteristics

The six sites differ along several data hetrogeneity orthogonal axes — students should
discover and document these from the data, not memorize them. (For
your own sanity: site identity is encoded in the file name, e.g.
`t1_siteC.nii.gz`.)

## Environment

This dataset assumes the standard MIATT lab environment: the
engineering Linux workstation with NFS scratch space and
cache redirects already configured (see the project-root
`README.md` and `miatt_environment_validator.py`).

Recommended Python toolchain (you may choose otherwise — justify
your choice in the report):

| Need | Suggested |
|---|---|
| I/O for NIfTI volumes | `itk` (the native Python wrapping) |
| Numerical work | `numpy`, `scipy` |
| Visualization | `matplotlib`, `napari`, `3D Slicer` |
| AI | MONAI |

One possible starting point for an isolated environment per project:

```bash
mkdir -p ${NFSSCRATCH}/miatt_final_exam && cd ${NFSSCRATCH}/miatt_final_exam
uv init
uv add itk numpy scipy matplotlib
```

Do **not** install dependencies into the system Python. Do **not**
work inside this `MIATTFINALEXAMDATA/` directory itself; treat it as
read-only input.

## The exam — what you will deliver

The final exam is a **20-minute oral defense** of a project
in which you build, test, and report on a multi-site landmark
detection pipeline.

### 1. Pipeline (code)

A reproducible pipeline that:

1. Reads any intensity volume from any site.
2. Detects some anatomical landmarks robustly across the six sites.
3. Computes a rigid ACPC alignment such that:
   - The anterior commissure (AC) is at physical coordinate
     `(0.0, 0.0, 0.0) mm`.
   - AC and PC have **identical superior–inferior** and **identical
     left–right** physical coordinates (only anterior–posterior
     differs).
   - The two eye centers (LE, RE) lie on a **common
     superior–inferior** plane.
4. Writes a Slicer-format `.fcsv` of the predicted landmarks for
   every subject in every `site*_unlabeled/` directory.

### 2. Additional landmarks beyond AC/PC/LE/RE

Any *other* fiducial points you choose to predict must be accompanied
by a written argument for **why** that point is robust enough to
trust. Not every landmark in the source data is equally reliable —
selection is part of the exam. The next phase of this project (July
2026) is projected to use these landmarks as control points for non-linear registration, so reliability under cross-site variation is what we care about.

### 3. Report (PDF, 2–3 pages of text; you may have additional appendices of tables and figures.)

Document:

1. The site characteristics you inferred from the data and the
   evidence for each.
2. Your design decisions — what you tried, what you rejected, and
   *why* (one sentence each is fine).
3. Your verification strategy — how you know the pipeline works on
   unseen data, not just the labeled set.
4. Reproducibility scaffolding — one-paragraph "stranger rebuilds this from scratch in July 2026" recipe.
5. How your landmark choices prepare for the July 2026 non-linear-registration phase.

### 4. Kaggle-style scoring

Your predicted landmarks for the unlabeled set will be scored against
the withheld ground truth. The metric (mean Euclidean error over the
canonical landmark subset) will be reported back to you, but the
ground truth itself will not.

### 5. Oral defense (20 minutes)

A discussion of the report and the code. Be ready to defend each
non-obvious choice and to explain how you would extend the pipeline
to handle a hypothetical seventh site you have never seen.

## Suggested starting checklist

1. Pick **two subjects from each site** (12 total) and visualize them
   side-by-side. Document what you see.
2. Interrogate the data to know what you have, generate intermediate reports, tabulate summary findings by site. Confirm with the data.
3. Build a *site-agnostic* preprocessing function. Justify the
   choice for both this dataset and future data sets.
4. Iterate: improve, evaluate on labeled holdouts you create from the
   labeled set itself, only run on `site*_unlabeled/` once at the end.

## What to submit

- A public GitHub repository with the pipeline code, lockfile,
  README, and the report PDF named `miatt_${hawkid}.pdf` (substitute
  your HawkID).
- A subdirectory in that repository containing your predicted
  `.fcsv` files, organized as
  `predictions/site<X>_unlabeled/<subject>/BCD_ACPC_Landmarks.fcsv`.
  These predicted landmark files **may** be shared publicly on
  GitHub. Any other file from `MIATTFINALEXAMDATA/` (T1, T2,
  posteriors, the provided labeled `.fcsv` files, etc.) **may not**
  leave the engineering data enclave.
- The 20-minute oral defense slot, scheduled during exam week.

Good luck.