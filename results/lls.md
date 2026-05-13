# Results — Approach 6 — BCD-Inspired LLS from Tissue Posteriors

**Date:** 2026-05-12  
**Evaluation:** 20% labeled holdout per site (deterministic split)  
**Metric:** Mean Euclidean error (mm) over all 51 landmarks

## Per-Site Summary

| Site | Train | Eval | Mean err (mm) | Std | Max |
|------|------:|-----:|--------------:|----:|----:|
| siteA | 87 | 21 | 4.38 | 1.27 | 7.30 |
| siteB | 85 | 21 | 4.96 | 1.30 | 7.77 |
| siteC | 86 | 21 | 5.08 | 1.98 | 10.20 |
| siteD | 86 | 21 | 5.68 | 2.40 | 12.64 |
| siteE | 86 | 21 | 4.93 | 1.36 | 7.60 |
| siteF | 86 | 21 | 4.82 | 1.69 | 9.45 |
| **Overall** | | | **4.97** | 1.76 | 12.64 |

## Best Landmarks (mean across sites)

| Landmark | Error (mm) |
|----------|-----------:|
| AC | 0.00 |
| PC | 0.97 |
| rostrum | 1.66 |
| mid_lat | 1.72 |
| mid_basel | 2.21 |
| genu | 2.26 |
| RP | 2.42 |
| lat_right | 2.56 |
| lat_left | 2.70 |
| RP_front | 2.74 |

## Worst Landmarks (mean across sites)

| Landmark | Error (mm) |
|----------|-----------:|
| RE | 13.14 |
| LE | 12.69 |
| r_sup_ext | 9.37 |
| l_sup_ext | 9.32 |
| top_left | 8.72 |
| top_right | 8.68 |
| left_lateral_inner_ear | 8.66 |
| r_front_pole | 8.42 |
| r_occ_pole | 7.79 |
| right_lateral_inner_ear | 7.31 |

## Interpretation

*(fill in after running)*
