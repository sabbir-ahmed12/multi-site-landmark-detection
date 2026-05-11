# Results — Approach 5 — Multi-Atlas Affine Registration

**Date:** 2026-05-11  
**Evaluation:** 20% labeled holdout per site (deterministic split)  
**Metric:** Mean Euclidean error (mm) over all 51 landmarks

## Per-Site Summary

| Site | Train | Eval | Mean err (mm) | Std | Max |
|------|------:|-----:|--------------:|----:|----:|
| siteA | 87 | 21 | 5.91 | 2.34 | 13.19 |
| siteB | 85 | 21 | 5.47 | 1.27 | 8.36 |
| siteC | 86 | 21 | 5.56 | 2.19 | 11.02 |
| siteD | 86 | 21 | 5.77 | 2.44 | 12.23 |
| siteE | 86 | 21 | 5.09 | 1.73 | 8.83 |
| siteF | 86 | 21 | 5.27 | 1.81 | 10.77 |
| **Overall** | | | **5.51** | 2.02 | 13.19 |

## Best Landmarks (mean across sites)

| Landmark | Error (mm) |
|----------|-----------:|
| AC | 0.00 |
| PC | 1.73 |
| rostrum | 2.00 |
| mid_lat | 2.12 |
| mid_basel | 2.52 |
| genu | 2.90 |
| RP | 3.04 |
| lat_right | 3.18 |
| lat_left | 3.31 |
| optic_chiasm | 3.55 |

## Worst Landmarks (mean across sites)

| Landmark | Error (mm) |
|----------|-----------:|
| RE | 12.03 |
| LE | 11.21 |
| l_sup_ext | 9.83 |
| r_sup_ext | 9.69 |
| top_right | 9.16 |
| top_left | 8.90 |
| left_lateral_inner_ear | 8.52 |
| r_occ_pole | 8.36 |
| r_front_pole | 8.35 |
| l_lat_ext | 7.85 |

## Interpretation

*(fill in after running)*
