"""Run the whole study end to end (`python main.py run-all`).

Steps: data preparation, unit tests, the full pipeline and baselines for each scenario, and the final evaluation.
Every step is a `python main.py ...` subprocess started from the repo root with the current interpreter; the run stops
at the first failing step. Everything this module prints is also written to logs/run_all_<YYYYmmdd_HHMMSS>.log.

Example:
    python main.py run-all --dry_run
    python main.py run-all --scenarios slide_level --skip_data --epochs_gan 1 --epochs_clf 1
"""

import argparse
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import TextIO

REPO_ROOT = Path(__file__).resolve().parents[2]
SCENARIOS = ("cell_level", "slide_level")
BASELINES = ("source_only", "composite", "reinhard", "target_supervised")
SUMMARY_REPORT = Path("reports") / "summary.md"

Step = tuple[str, list[str]]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command line arguments.

    Args:
        argv: Argument list; defaults to sys.argv[1:].

    Returns:
        The parsed arguments.
    """
    parser = argparse.ArgumentParser(
        prog="python main.py run-all",
        description="Run the full study: data preparation, tests, training per scenario, baselines and evaluation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--scenarios",
        nargs="+",
        choices=SCENARIOS,
        default=list(SCENARIOS),
        help="Scenarios to run, in the given order",
    )
    parser.add_argument("--skip_data", action="store_true", help="Skip C-NMC sampling and ALL-IDB extraction")
    parser.add_argument("--skip_tests", action="store_true", help="Skip the unit tests")
    parser.add_argument("--skip_baselines", action="store_true", help="Skip the baseline classifiers")
    parser.add_argument("--epochs_gan", type=int, default=None, help="Forwarded to every training call (smoke runs)")
    parser.add_argument("--epochs_clf", type=int, default=None, help="Forwarded to every training call (smoke runs)")
    parser.add_argument("--device", type=str, default=None, help="Forwarded to every training call ('cuda' or 'cpu')")
    parser.add_argument("--dry_run", action="store_true", help="Print the numbered command list and exit")
    return parser.parse_args(argv)


def _train_options(args: argparse.Namespace, scenario: str) -> list[str]:
    """Build the options shared by every training call of a scenario.

    Args:
        args: Parsed run_all arguments.
        scenario: Scenario name.

    Returns:
        The option list to append after the stage-specific options.
    """
    options = ["--scenario", scenario]
    if scenario == "slide_level":
        options.append("--source_val")
    if args.epochs_gan is not None:
        options += ["--epochs_gan", str(args.epochs_gan)]
    if args.epochs_clf is not None:
        options += ["--epochs_clf", str(args.epochs_clf)]
    if args.device is not None:
        options += ["--device", args.device]
    return options


def build_steps(args: argparse.Namespace, python: str = sys.executable) -> list[Step]:
    """Build the ordered list of study steps without running anything.

    Args:
        args: Parsed run_all arguments.
        python: Interpreter used for every step.

    Returns:
        A list of (description, command) pairs in execution order.
    """
    steps: list[Step] = []
    main_py = [python, "main.py"]
    if not args.skip_data:
        steps.append(("Data preparation (first run ~10 min)", [*main_py, "prepare-data"]))
    if not args.skip_tests:
        steps.append(("Unit tests", [python, "-m", "pytest", "-q", "tests"]))
    for scenario in args.scenarios:
        options = _train_options(args, scenario)
        steps.append((f"[{scenario}] GAN + translation + classifier", [*main_py, "train", "--stage", "all", *options]))
        if args.skip_baselines:
            continue
        for baseline in BASELINES:
            command = [*main_py, "train", "--stage", "baseline", "--baseline", baseline, *options]
            steps.append((f"[{scenario}] Baseline {baseline}", command))
    steps.append(("Final evaluation", [*main_py, "evaluate"]))
    return steps


def format_command(command: list[str]) -> str:
    """Render a command for display.

    Args:
        command: Argument list.

    Returns:
        The command as one shell-like string.
    """
    return subprocess.list2cmdline(command) if sys.platform == "win32" else " ".join(command)


def format_duration(seconds: float) -> str:
    """Format a duration as H:MM:SS.

    Args:
        seconds: Elapsed seconds.

    Returns:
        The formatted duration.
    """
    minutes, secs = divmod(int(round(seconds)), 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours}:{minutes:02d}:{secs:02d}"


class Tee:
    """Print to the console and, when a log file is open, append the same text to it."""

    def __init__(self, log_file: TextIO | None = None) -> None:
        """Initialize the tee.

        Args:
            log_file: Open text file receiving a copy of every message, or None for console only.
        """
        self.log_file = log_file

    def __call__(self, message: str = "") -> None:
        """Print a message and copy it to the log.

        Args:
            message: Text to print.
        """
        print(message, flush=True)
        if self.log_file is not None:
            self.log_file.write(message + "\n")
            self.log_file.flush()


def print_summary(log: Tee, results: list[tuple[str, str, float | None]]) -> None:
    """Print the final table of steps with status and duration.

    Args:
        log: Output sink.
        results: (description, status, seconds or None) for every planned step.
    """
    width = max(len(name) for name, _, _ in results)
    log("")
    log("=" * 100)
    log("SUMMARY")
    log("=" * 100)
    log(f"{'#':>3}  {'Step':<{width}}  {'Status':<8}  Duration")
    for index, (name, status, seconds) in enumerate(results, start=1):
        duration = format_duration(seconds) if seconds is not None else "-"
        log(f"{index:>3}  {name:<{width}}  {status:<8}  {duration}")
    total = sum(seconds for _, _, seconds in results if seconds is not None)
    log(f"Total time: {format_duration(total)}")


def run_steps(steps: list[Step], log: Tee) -> int:
    """Run the steps in order, stopping at the first failure.

    Args:
        steps: (description, command) pairs from build_steps.
        log: Output sink.

    Returns:
        0 on success, otherwise the exit code of the failing step.
    """
    results: list[tuple[str, str, float | None]] = [(name, "PENDING", None) for name, _ in steps]
    exit_code = 0
    for index, (name, command) in enumerate(steps, start=1):
        log("")
        log("=" * 100)
        log(f"STEP {index}/{len(steps)}: {name}")
        log(f"Command: {format_command(command)}")
        log(f"Started: {datetime.now():%Y-%m-%d %H:%M:%S}")
        log("=" * 100)
        start = time.perf_counter()
        try:
            returncode = subprocess.run(command, cwd=REPO_ROOT, check=False).returncode
        except KeyboardInterrupt:
            returncode = 130
        elapsed = time.perf_counter() - start
        status = "OK" if returncode == 0 else "FAILED"
        results[index - 1] = (name, status, elapsed)
        log(f"Step {index}/{len(steps)} {status} in {format_duration(elapsed)}")
        if returncode != 0:
            log("")
            log(f"ERROR: step {index}/{len(steps)} '{name}' failed with exit code {returncode}.")
            log(f"Failed command: {format_command(command)}")
            log("Stopping; later steps were not run.")
            exit_code = returncode
            break
    print_summary(log, results)
    return exit_code


def main(argv: list[str] | None = None) -> int:
    """Run the study, or print the plan with --dry_run.

    Args:
        argv: Argument list; defaults to sys.argv[1:].

    Returns:
        Process exit code.
    """
    args = parse_args(argv)
    steps = build_steps(args)

    if args.dry_run:
        log = Tee()
        log(f"Dry run: {len(steps)} steps would be executed from {REPO_ROOT}")
        for index, (name, command) in enumerate(steps, start=1):
            log(f"{index:>3}. {name}")
            log(f"     {format_command(command)}")
        return 0

    log_dir = REPO_ROOT / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"run_all_{datetime.now():%Y%m%d_%H%M%S}.log"
    with log_path.open("w", encoding="utf-8") as log_file:
        log = Tee(log_file)
        log(f"run_all started {datetime.now():%Y-%m-%d %H:%M:%S}; {len(steps)} steps; log: {log_path}")
        log(f"Arguments: {vars(args)}")
        exit_code = run_steps(steps, log)
        log("")
        if exit_code == 0:
            log(f"Study finished. Reports: {REPO_ROOT / SUMMARY_REPORT}")
        else:
            log(f"Study stopped with exit code {exit_code}. Partial reports (if any): {REPO_ROOT / SUMMARY_REPORT}")
        log(f"Log: {log_path}")
    return exit_code
