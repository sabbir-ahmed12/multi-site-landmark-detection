# Results — Approach 4 — 3D CNN Coordinate Regression

**Date:** 2026-05-12  
**Evaluation:** 20% labeled holdout per site (deterministic split)  
**Metric:** Mean Euclidean error (mm) over all 51 landmarks

## Per-Site Summary

| Site | Train | Eval | Mean err (mm) | Std | Max |
|------|------:|-----:|--------------:|----:|----:|
| siteA | 87 | 21 | 4.54 | 1.79 | 10.92 |
| siteB | 85 | 21 | 4.38 | 1.14 | 6.57 |
| siteC | 86 | 21 | 4.64 | 1.83 | 9.56 |
| siteD | 86 | 21 | 5.04 | 2.61 | 11.79 |
| siteE | 86 | 21 | 4.13 | 1.17 | 7.36 |
| siteF | 86 | 21 | 4.53 | 1.71 | 9.88 |
| **Overall** | | | **4.54** | 1.80 | 11.79 |

## Best Landmarks (mean across sites)

| Landmark | Error (mm) |
|----------|-----------:|
| AC | 0.06 |
| PC | 0.93 |
| rostrum | 1.55 |
| mid_lat | 1.63 |
| mid_basel | 2.01 |
| RP | 2.14 |
| genu | 2.21 |
| RP_front | 2.38 |
| lat_right | 2.52 |
| SMV | 2.56 |

## Worst Landmarks (mean across sites)

| Landmark | Error (mm) |
|----------|-----------:|
| RE | 12.56 |
| LE | 11.23 |
| r_sup_ext | 8.22 |
| top_left | 8.15 |
| l_sup_ext | 8.03 |
| top_right | 8.02 |
| left_lateral_inner_ear | 7.71 |
| r_front_pole | 7.64 |
| r_occ_pole | 6.98 |
| right_lateral_inner_ear | 6.63 |

## Interpretation

*(fill in after running)*
