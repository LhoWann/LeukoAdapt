"""Prepare every processed dataset: the C-NMC source sets, then the ALL-IDB target cells, splits and background banks."""

import argparse

from src.data.extract_all_idb import extract_all_idb
from src.data.sample_cnmc import sample_cnmc_dataset


def main(argv: list[str] | None = None) -> int:
    """Run the data preparation.

    Args:
        argv: Argument list; defaults to sys.argv[1:].

    Returns:
        Process exit code.
    """
    parser = argparse.ArgumentParser(
        prog="python main.py prepare-data",
        description="Sample C-NMC and extract ALL-IDB (the first run searches overlapping slides for about 10 minutes).",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed of the sampling and of both splits")
    args = parser.parse_args(argv)
    sample_cnmc_dataset(seed=args.seed)
    extract_all_idb(seed=args.seed)
    return 0
