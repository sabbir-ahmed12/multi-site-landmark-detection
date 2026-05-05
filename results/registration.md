# Results — Approach 2 — Rigid Registration to ACPC Template

**Date:** 2026-05-05  
**Evaluation:** 20% labeled holdout per site (deterministic split)  
**Metric:** Mean Euclidean error (mm) over all 51 landmarks

## Per-Site Summary

| Site | Train | Eval | Mean err (mm) | Std | Max |
|------|------:|-----:|--------------:|----:|----:|
| siteA | 87 | 21 | 4.86 | 1.67 | 10.79 |
| siteB | 85 | 21 | 4.34 | 1.18 | 7.22 |
| siteC | 86 | 21 | 4.61 | 1.77 | 9.48 |
| siteD | 86 | 21 | 5.03 | 2.31 | 11.17 |
| siteE | 86 | 21 | 4.13 | 1.40 | 7.50 |
| siteF | 86 | 21 | 4.55 | 1.58 | 8.95 |
| **Overall** | | | **4.59** | 1.71 | 11.17 |

## Best Landmarks (mean across sites)

| Landmark | Error (mm) |
|----------|-----------:|
| AC | 0.00 |
| PC | 1.03 |
| rostrum | 1.53 |
| mid_lat | 1.61 |
| mid_basel | 2.03 |
| genu | 2.25 |
| RP | 2.26 |
| lat_right | 2.39 |
| RP_front | 2.53 |
| lat_left | 2.54 |

## Worst Landmarks (mean across sites)

| Landmark | Error (mm) |
|----------|-----------:|
| RE | 12.37 |
| LE | 11.39 |
| r_sup_ext | 8.32 |
| top_left | 8.28 |
| l_sup_ext | 8.18 |
| top_right | 8.18 |
| left_lateral_inner_ear | 7.86 |
| r_front_pole | 7.75 |
| r_occ_pole | 7.10 |
| l_front_pole | 6.73 |

## Interpretation

**Hypothesis:** Rigid registration of each subject's T1 to a within-site ACPC
template would automatically recover the subject-specific ACPC transform without
needing labeled AC/PC/LE/RE, and thereby improve over Approach 1 for unlabeled
subjects.

**What was actually implemented:** A mean ACPC T1 template is built from
training subjects (T1s resampled to a fixed 2 mm, 101×101×96-voxel ACPC grid
using each subject's labeled ACPC transform).  For each eval/unlabeled subject,
the template is registered to the subject T1 via `Euler3DTransform` + Mattes
Mutual Information (3-level pyramid, 4×/2×/1× shrink, ~0.2 s per subject).
The registration transform (ACPC → scanner) is applied to the mean ACPC
landmarks → predicted scanner coordinates, which are then ACPC-aligned using
the predicted AC/PC/LE/RE.

**Outcome:** Overall mean error 4.59 mm — identical to Approach 1 in every
per-site and per-landmark detail.  AC is again 0.00 mm because ACPC-aligning
the predictions cancels any registration offset.

**Why Approach 2 matches Approach 1 in ACPC-space evaluation:**
The evaluation pipeline applies the true ACPC transform to both ground truth and
predictions.  For predictions: (i) registration finds T_reg ≈ T_acpc_inv; (ii)
propagating mean landmarks gives predicted_scanner = T_reg(M); (iii) ACPC-aligning
predicted_scanner with its own predicted AC/PC/LE/RE yields M again (the
population mean).  The ACPC-space evaluation metric is therefore insensitive to
registration quality; it reflects only anatomical variation around the mean — the
same quantity Approach 1 measures directly.

**Where Approach 2 adds value:**
Registration provides a principled per-subject ACPC transform estimate for
unlabeled subjects, replacing the manual AC/PC/LE/RE detection step.  The template
build and per-subject registration runtime is negligible (~0.2 s per subject after
the one-time cache step).  For the held-out unlabeled subjects, predictions are now
subject-specific in scanner space rather than a site-wide constant.

**Remaining error:** Identical error sources to Approach 1 — pure inter-subject
anatomical variation in ACPC space.  Rigid registration cannot resolve within-ACPC
shape differences.  Landmark-specific prediction beyond the population mean requires
Approach 4 (CNN regression).
