"""Run the landmark detection pipeline for all sites and report evaluation."""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

import numpy as np

from miatt.pipeline import ApproachName, EvalResult, run_mean_baseline

DATA_ROOT = Path("/nfs/s-l028/scratch/opt/ece5490/MIATTFINALEXAMDATA")
SITES = ["siteA", "siteB", "siteC", "siteD", "siteE", "siteF"]
RESULTS_DIR = Path("results")

APPROACH_LABELS = {
    "mean": "Approach 1 — Per-Site Mean (ACPC space)",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MIATT landmark detection pipeline")
    parser.add_argument(
        "--approach",
        choices=["mean", "registration", "heuristic", "cnn"],
        default="mean",
    )
    parser.add_argument("--sites", nargs="+", default=SITES)
    parser.add_argument("--output", type=Path, default=Path("predictions"))
    parser.add_argument(
        "--eval-fraction",
        type=float,
        default=0.2,
        help="Fraction of labeled subjects held out for evaluation (default: 0.2)",
    )
    return parser.parse_args()


def _lm_mean_across_sites(results: list[EvalResult]) -> dict[str, float]:
    all_lm: dict[str, list[float]] = {}
    for r in results:
        for lm, err in r.per_landmark_mean_mm.items():
            all_lm.setdefault(lm, []).append(err)
    return {lm: float(np.mean(v)) for lm, v in all_lm.items()}


def save_results(approach: str, results: list[EvalResult], eval_fraction: float) -> Path:
    """Persist evaluation results as JSON under results/."""
    RESULTS_DIR.mkdir(exist_ok=True)
    all_errors = [e for r in results for e in r.per_subject_errors]
    lm_means = _lm_mean_across_sites(results)

    payload = {
        "approach": approach,
        "label": APPROACH_LABELS.get(approach, approach),
        "date": str(date.today()),
        "eval_fraction": eval_fraction,
        "sites": {
            r.site: {
                "n_train": r.n_train,
                "n_eval": r.n_eval,
                "mean_error_mm": round(r.mean_error_mm, 4),
                "std_error_mm": round(float(np.std(r.per_subject_errors)), 4),
                "max_error_mm": round(float(np.max(r.per_subject_errors)), 4),
                "per_subject_errors": [round(e, 4) for e in r.per_subject_errors],
                "per_landmark_mean_mm": {
                    k: round(v, 4) for k, v in sorted(r.per_landmark_mean_mm.items())
                },
            }
            for r in results
        },
        "overall": {
            "mean_error_mm": round(float(np.mean(all_errors)), 4),
            "std_error_mm": round(float(np.std(all_errors)), 4),
            "max_error_mm": round(float(np.max(all_errors)), 4),
        },
        "landmark_summary": {
            "best_5": {
                lm: round(lm_means[lm], 4)
                for lm in sorted(lm_means, key=lm_means.get)[:5]  # type: ignore[arg-type]
            },
            "worst_10": {
                lm: round(lm_means[lm], 4)
                for lm in sorted(lm_means, key=lm_means.get, reverse=True)[:10]  # type: ignore[arg-type]
            },
        },
    }

    out = RESULTS_DIR / f"{approach}.json"
    out.write_text(json.dumps(payload, indent=2))
    return out


def save_markdown(approach: str, results: list[EvalResult]) -> Path:
    """Write a human-readable Markdown summary for the report appendix."""
    all_errors = [e for r in results for e in r.per_subject_errors]
    lm_means = _lm_mean_across_sites(results)
    label = APPROACH_LABELS.get(approach, approach)

    lines: list[str] = [
        f"# Results — {label}",
        f"\n**Date:** {date.today()}  ",
        f"**Evaluation:** 20% labeled holdout per site (deterministic split)  ",
        f"**Metric:** Mean Euclidean error (mm) over all 51 landmarks\n",
        "## Per-Site Summary\n",
        "| Site | Train | Eval | Mean err (mm) | Std | Max |",
        "|------|------:|-----:|--------------:|----:|----:|",
    ]
    for r in results:
        errs = r.per_subject_errors
        lines.append(
            f"| {r.site} | {r.n_train} | {r.n_eval} "
            f"| {r.mean_error_mm:.2f} | {np.std(errs):.2f} | {np.max(errs):.2f} |"
        )
    lines += [
        f"| **Overall** | | | **{np.mean(all_errors):.2f}** "
        f"| {np.std(all_errors):.2f} | {np.max(all_errors):.2f} |",
        "\n## Best Landmarks (mean across sites)\n",
        "| Landmark | Error (mm) |",
        "|----------|-----------:|",
    ]
    for lm in sorted(lm_means, key=lm_means.get)[:10]:  # type: ignore[arg-type]
        lines.append(f"| {lm} | {lm_means[lm]:.2f} |")
    lines += [
        "\n## Worst Landmarks (mean across sites)\n",
        "| Landmark | Error (mm) |",
        "|----------|-----------:|",
    ]
    for lm in sorted(lm_means, key=lm_means.get, reverse=True)[:10]:  # type: ignore[arg-type]
        lines.append(f"| {lm} | {lm_means[lm]:.2f} |")

    lines += [
        "\n## Interpretation\n",
        "*(fill in after running)*",
    ]

    out = RESULTS_DIR / f"{approach}.md"
    out.write_text("\n".join(lines) + "\n")
    return out


def print_summary(results: list[EvalResult], approach: str) -> None:
    label = APPROACH_LABELS.get(approach, approach)
    all_errors = [e for r in results for e in r.per_subject_errors]
    lm_means = _lm_mean_across_sites(results)

    print("\n" + "=" * 65)
    print(f"{label:^65}")
    print("=" * 65)
    print(f"{'Site':<8} {'Train':>6} {'Eval':>6} {'Mean err (mm)':>14} {'Std':>8} {'Max':>8}")
    print("-" * 65)
    for r in results:
        errs = r.per_subject_errors
        print(
            f"{r.site:<8} {r.n_train:>6} {r.n_eval:>6} "
            f"{r.mean_error_mm:>14.2f} {np.std(errs):>8.2f} {np.max(errs):>8.2f}"
        )
    print("-" * 65)
    print(
        f"{'Overall':<8} {'':>6} {'':>6} "
        f"{np.mean(all_errors):>14.2f} {np.std(all_errors):>8.2f} {np.max(all_errors):>8.2f}"
    )
    print("=" * 65)

    worst = sorted(lm_means, key=lm_means.get, reverse=True)[:10]  # type: ignore[arg-type]
    best = sorted(lm_means, key=lm_means.get)[:5]                  # type: ignore[arg-type]
    print("\nTop-5 best landmarks:  ", "  ".join(f"{lm} {lm_means[lm]:.1f}mm" for lm in best))
    print("Top-10 worst landmarks:")
    for lm in worst:
        print(f"  {lm:<30} {lm_means[lm]:.2f} mm")


def main() -> None:
    args = parse_args()
    results: list[EvalResult] = []

    for site in args.sites:
        print(f"[{site}] running approach={args.approach} …", end=" ", flush=True)
        if args.approach == "mean":
            r = run_mean_baseline(DATA_ROOT, site, args.output, args.eval_fraction)
        else:
            raise NotImplementedError(f"Approach '{args.approach}' not yet implemented.")
        results.append(r)
        print(f"done — eval mean error: {r.mean_error_mm:.2f} mm")

    print_summary(results, args.approach)

    json_path = save_results(args.approach, results, args.eval_fraction)
    md_path = save_markdown(args.approach, results)
    print(f"\nResults saved to: {json_path}  {md_path}")
    print(f"Predictions written to: {args.output}/")


if __name__ == "__main__":
    main()
