# Results — Approach 3 — Posterior-Guided Local Refinement

**Date:** 2026-05-07  
**Evaluation:** 20% labeled holdout per site (deterministic split)  
**Metric:** Mean Euclidean error (mm) over all 51 landmarks

## Per-Site Summary

| Site | Train | Eval | Mean err (mm) | Std | Max |
|------|------:|-----:|--------------:|----:|----:|
| siteA | 87 | 21 | 5.13 | 1.66 | 11.07 |
| **Overall** | | | **5.13** | 1.66 | 11.07 |

## Best Landmarks (mean across sites)

| Landmark | Error (mm) |
|----------|-----------:|
| AC | 0.00 |
| PC | 1.21 |
| mid_lat | 1.48 |
| mid_basel | 2.04 |
| RP | 2.22 |
| lat_left | 2.52 |
| RP_front | 2.54 |
| lat_right | 2.62 |
| optic_chiasm | 2.90 |
| SMV | 2.96 |

## Worst Landmarks (mean across sites)

| Landmark | Error (mm) |
|----------|-----------:|
| LE | 13.75 |
| RE | 11.45 |
| r_sup_ext | 8.82 |
| r_front_pole | 8.59 |
| top_left | 8.51 |
| l_sup_ext | 8.03 |
| l_front_pole | 7.69 |
| top_right | 7.62 |
| l_occ_pole | 7.53 |
| right_lateral_inner_ear | 7.42 |

## Interpretation

**Evaluation note:** Only siteA was evaluated (siteA is pre-ACPC-aligned, so the ACPC transform is the identity; this is the cleanest test site for the heuristic).

**Outcome: negative result.** The heuristic scored 5.13 mm overall vs 4.59 mm from Approach 1, a regression of +0.54 mm.  Every WM corpus callosum landmark got worse, with the largest regressions on `genu` (+4.4 mm, from 2.25 → 6.67 mm) and `rostrum` (+4.6 mm, from 1.59 → 6.21 mm).

**Root cause:** The WM posterior is high throughout all white matter, not specifically at the target corpus callosum sub-structure.  A weighted centroid within 8 mm of the mean genu position captures the CC genu, the internal capsule, and corona radiata fibres — all dense WM.  The resulting centroid is pulled toward the bulk WM centre of mass in the region, not toward the anatomically specific landmark, adding noise rather than signal.

**What would be needed for this to work:**
- A tissue whose spatial centroid coincides with the target landmark (e.g., a distinct nucleus with clear spatial boundaries).
- Or a supervised regressor (per landmark) that maps local posterior statistics to a correction vector.  This motivates Approach 4 (CNN), which learns such mappings end-to-end from all 642 labeled subjects.
