"""Command-line entrypoint for advanced-model cross-validation."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


NEW_ATTEMPT_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = NEW_ATTEMPT_DIR.parent
for path in (PROJECT_ROOT, NEW_ATTEMPT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from cross_validation.model_validation import CrossValidationConfig, run_validation  # noqa: E402


def parse_args() -> argparse.Namespace:
    """Parse reproducible validation settings."""
    parser = argparse.ArgumentParser(
        description="Nested patient-grouped CV for XGBoost and CatBoost."
    )
    parser.add_argument("--folds", type=int, default=5, help="Outer folds (default: 5).")
    parser.add_argument("--inner-folds", type=int, default=3, help="Inner tuning folds (default: 3).")
    parser.add_argument("--n-iter", type=int, default=8, help="Random search candidates per outer fold.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42).")
    parser.add_argument("--n-jobs", type=int, default=-1, help="Parallel search jobs (default: all cores).")
    return parser.parse_args()


def main() -> None:
    """Run all four validation experiments."""
    args = parse_args()
    if args.folds < 2 or args.inner_folds < 2 or args.n_iter < 1:
        raise ValueError("folds and inner-folds must be >= 2; n-iter must be >= 1.")
    config = CrossValidationConfig(
        n_splits=args.folds,
        inner_splits=args.inner_folds,
        n_iter_search=args.n_iter,
        random_state=args.seed,
        n_jobs=args.n_jobs,
    )
    run_validation(config)


if __name__ == "__main__":
    main()
