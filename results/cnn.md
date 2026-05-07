# Results — Approach 4 — 3D CNN Coordinate Regression

**Date:** 2026-05-07  
**Evaluation:** 20% labeled holdout per site (deterministic split)  
**Metric:** Mean Euclidean error (mm) over all 51 landmarks

## Per-Site Summary

| Site | Train | Eval | Mean err (mm) | Std | Max |
|------|------:|-----:|--------------:|----:|----:|
| siteB | 85 | 21 | 4.38 | 1.14 | 6.57 |
| siteC | 86 | 21 | 4.64 | 1.83 | 9.56 |
| siteD | 86 | 21 | 5.04 | 2.61 | 11.79 |
| siteE | 86 | 21 | 4.13 | 1.17 | 7.36 |
| siteF | 86 | 21 | 4.53 | 1.71 | 9.88 |
| **Overall** | | | **4.54** | 1.80 | 11.79 |

## Best Landmarks (mean across sites)

| Landmark | Error (mm) |
|----------|-----------:|
| AC | 0.05 |
| PC | 0.92 |
| rostrum | 1.53 |
| mid_lat | 1.64 |
| mid_basel | 2.01 |
| RP | 2.17 |
| genu | 2.23 |
| RP_front | 2.40 |
| lat_right | 2.50 |
| SMV | 2.54 |

## Worst Landmarks (mean across sites)

| Landmark | Error (mm) |
|----------|-----------:|
| RE | 12.87 |
| LE | 10.96 |
| top_left | 8.18 |
| top_right | 8.17 |
| r_sup_ext | 8.16 |
| l_sup_ext | 8.11 |
| left_lateral_inner_ear | 7.93 |
| r_front_pole | 7.63 |
| r_occ_pole | 7.06 |
| right_lateral_inner_ear | 6.60 |

## Interpretation

*(fill in after running)*
