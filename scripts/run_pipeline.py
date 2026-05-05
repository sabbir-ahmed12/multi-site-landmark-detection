"""Run the landmark detection pipeline for all sites and report evaluation."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from miatt.pipeline import ApproachName, EvalResult, run_mean_baseline

DATA_ROOT = Path("/nfs/s-l028/scratch/opt/ece5490/MIATTFINALEXAMDATA")
SITES = ["siteA", "siteB", "siteC", "siteD", "siteE", "siteF"]


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


def print_summary(results: list[EvalResult]) -> None:
    print("\n" + "=" * 65)
    print(f"{'Approach 1 — Per-Site Mean Baseline':^65}")
    print("=" * 65)
    print(f"{'Site':<8} {'Train':>6} {'Eval':>6} {'Mean err (mm)':>14} {'Std':>8} {'Max':>8}")
    print("-" * 65)
    all_errors: list[float] = []
    for r in results:
        errs = r.per_subject_errors
        std = float(np.std(errs)) if errs else float("nan")
        mx = float(np.max(errs)) if errs else float("nan")
        print(
            f"{r.site:<8} {r.n_train:>6} {r.n_eval:>6} "
            f"{r.mean_error_mm:>14.2f} {std:>8.2f} {mx:>8.2f}"
        )
        all_errors.extend(errs)
    print("-" * 65)
    if all_errors:
        print(
            f"{'Overall':<8} {'':>6} {'':>6} "
            f"{np.mean(all_errors):>14.2f} {np.std(all_errors):>8.2f} {np.max(all_errors):>8.2f}"
        )
    print("=" * 65)

    # Per-landmark breakdown (top 10 worst)
    all_lm_errors: dict[str, list[float]] = {}
    for r in results:
        for lm, err in r.per_landmark_mean_mm.items():
            all_lm_errors.setdefault(lm, []).append(err)
    lm_means = {lm: float(np.mean(v)) for lm, v in all_lm_errors.items()}
    worst = sorted(lm_means, key=lm_means.get, reverse=True)[:10]  # type: ignore[arg-type]
    best = sorted(lm_means, key=lm_means.get)[:5]                  # type: ignore[arg-type]
    print("\nTop-10 worst landmarks (mean error across sites):")
    for lm in worst:
        print(f"  {lm:<30} {lm_means[lm]:.2f} mm")
    print("\nTop-5 best landmarks:")
    for lm in best:
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

    print_summary(results)
    print(f"\nPredictions written to: {args.output}/")


if __name__ == "__main__":
    main()
