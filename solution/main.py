"""Command line entry to run the scheduler on provided datasets."""
from __future__ import annotations

import argparse
from pathlib import Path

from .config import settings
from .scheduler import solve_files


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run memory-aware scheduling on datasets.")
    parser.add_argument(
        "--datasets",
        type=Path,
        nargs="*",
        help="Optional list of dataset json files. Defaults to bundled datasets.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=settings.outputs_dir / "Problem1",
        help="Directory to save schedule outputs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.datasets:
        dataset_paths = args.datasets
    else:
        dataset_paths = sorted(settings.datasets_dir.glob("*.json"))

    args.output.mkdir(parents=True, exist_ok=True)
    results = solve_files(dataset_paths)

    for result in results:
        schedule_path = args.output / f"{result.task_name}_schedule.txt"
        with schedule_path.open("w", encoding="utf-8") as fp:
            fp.write("\n".join(str(nid) for nid in result.order))
        print(f"{result.task_name:>20}: peak_L1_UB = {result.peak_memory}")


if __name__ == "__main__":
    main()
