"""LeukoAdapt command-line entry point.

Usage:
    python main.py run-all [options]        # the whole study, start to end
    python main.py prepare-data [options]   # C-NMC sampling, ALL-IDB extraction, splits, background banks
    python main.py train [options]          # one stage (gan, translate, classifier, all) or one baseline
    python main.py evaluate [options]       # final comparison of every finished run

Run `python main.py <command> --help` for the options of a command.
"""

import importlib
import sys

COMMANDS = {
    "run-all": "src.pipeline.run_all",
    "prepare-data": "src.pipeline.prepare_data",
    "train": "src.pipeline.train",
    "evaluate": "src.pipeline.evaluate",
}


def main(argv: list[str] | None = None) -> int:
    """Dispatch to the module of the requested command.

    Args:
        argv: Argument list; defaults to sys.argv[1:].

    Returns:
        Process exit code.
    """
    argv = sys.argv[1:] if argv is None else argv
    if not argv or argv[0] in ("-h", "--help"):
        print(__doc__)
        return 0
    command, rest = argv[0], argv[1:]
    if command not in COMMANDS:
        print(f"Unknown command '{command}'. Choose one of: {', '.join(COMMANDS)}.", file=sys.stderr)
        return 2
    return importlib.import_module(COMMANDS[command]).main(rest) or 0


if __name__ == "__main__":
    sys.exit(main())
