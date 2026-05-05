"""Run the landmark detection pipeline for all sites."""

from __future__ import annotations

import argparse
from pathlib import Path

from miatt.pipeline import ApproachName, run_mean_baseline

DATA_ROOT = Path("/nfs/s-l028/scratch/opt/ece5490/MIATTFINALEXAMDATA")
SITES = ["siteA", "siteB", "siteC", "siteD", "siteE", "siteF"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MIATT landmark detection pipeline")
    parser.add_argument(
        "--approach",
        choices=["mean", "registration", "heuristic", "cnn"],
        default="mean",
        help="Detection approach to use (default: mean baseline)",
    )
    parser.add_argument(
        "--sites",
        nargs="+",
        default=SITES,
        help="Sites to process (default: all six)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("predictions"),
        help="Output directory for predicted .fcsv files",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for site in args.sites:
        print(f"Processing {site} with approach={args.approach} …")
        if args.approach == "mean":
            run_mean_baseline(DATA_ROOT, site, args.output)
        else:
            raise NotImplementedError(f"Approach '{args.approach}' not yet implemented.")
    print("Done.")


if __name__ == "__main__":
    main()
