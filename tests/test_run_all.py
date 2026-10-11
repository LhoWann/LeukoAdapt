"""Unit tests for the run-all step list and the main.py dispatcher (no step is executed)."""

import unittest

import main
from src.pipeline.run_all import BASELINES, build_steps, parse_args

PY = "python"
MAIN = [PY, "main.py"]


def commands(argv: list[str]) -> list[list[str]]:
    """Build the command list for the given run_all arguments.

    Args:
        argv: run_all command line arguments.

    Returns:
        The commands in execution order.
    """
    return [command for _, command in build_steps(parse_args(argv), python=PY)]


def train_calls(argv: list[str]) -> list[list[str]]:
    """Return only the training commands.

    Args:
        argv: run_all command line arguments.

    Returns:
        The `main.py train` commands in execution order.
    """
    return [command for command in commands(argv) if command[1:3] == ["main.py", "train"]]


class TestBuildSteps(unittest.TestCase):
    """The step list must follow the study protocol for every flag combination."""

    def test_default_step_list(self) -> None:
        """Verify the exact default sequence: data, tests, both scenarios with baselines, evaluation."""
        expected = [[*MAIN, "prepare-data"], [PY, "-m", "pytest", "-q", "tests"]]
        for scenario, extra in (("cell_level", []), ("slide_level", ["--source_val"])):
            expected.append([*MAIN, "train", "--stage", "all", "--scenario", scenario, *extra])
            for baseline in BASELINES:
                expected.append(
                    [*MAIN, "train", "--stage", "baseline", "--baseline", baseline, "--scenario", scenario, *extra]
                )
        expected.append([*MAIN, "evaluate"])
        actual = commands([])
        if actual != expected:
            raise AssertionError(f"default steps differ:\n{actual}\n!=\n{expected}")

    def test_slide_level_uses_source_val(self) -> None:
        """Verify every slide_level training call carries --source_val and no cell_level call does."""
        calls = train_calls([])
        expected_calls = 2 * (1 + len(BASELINES))
        if len(calls) != expected_calls:
            raise AssertionError(f"expected {expected_calls} training calls, got {len(calls)}")
        for command in calls:
            scenario = command[command.index("--scenario") + 1]
            if ("--source_val" in command) != (scenario == "slide_level"):
                raise AssertionError(f"wrong --source_val usage: {command}")

    def test_scenario_order_and_selection(self) -> None:
        """Verify --scenarios restricts and orders the scenarios."""
        calls = train_calls(["--scenarios", "slide_level", "cell_level", "--skip_baselines"])
        scenarios = [command[command.index("--scenario") + 1] for command in calls]
        if scenarios != ["slide_level", "cell_level"]:
            raise AssertionError(f"unexpected scenario order: {scenarios}")

    def test_skip_flags_remove_steps(self) -> None:
        """Verify each --skip_* flag removes exactly its steps."""
        if any("prepare-data" in c for c in commands(["--skip_data"])):
            raise AssertionError("--skip_data left a data step")
        if any("pytest" in c for c in commands(["--skip_tests"])):
            raise AssertionError("--skip_tests left the test step")
        if any("baseline" in c for c in commands(["--skip_baselines"])):
            raise AssertionError("--skip_baselines left a baseline step")
        minimal = commands(["--skip_data", "--skip_tests", "--skip_baselines", "--scenarios", "cell_level"])
        expected = [[*MAIN, "train", "--stage", "all", "--scenario", "cell_level"], [*MAIN, "evaluate"]]
        if minimal != expected:
            raise AssertionError(f"minimal step list differs: {minimal}")

    def test_epochs_and_device_forwarded(self) -> None:
        """Verify --epochs_gan, --epochs_clf and --device reach every training call and nothing else."""
        argv = ["--epochs_gan", "1", "--epochs_clf", "2", "--device", "cpu"]
        for command in commands(argv):
            is_train = command[1:3] == ["main.py", "train"]
            for option, value in (("--epochs_gan", "1"), ("--epochs_clf", "2"), ("--device", "cpu")):
                present = option in command and command[command.index(option) + 1] == value
                if present != is_train:
                    raise AssertionError(f"{option} forwarding wrong for {command}")

    def test_epochs_not_forwarded_by_default(self) -> None:
        """Verify training keeps its own defaults when no epochs are given."""
        for command in train_calls([]):
            if "--epochs_gan" in command or "--epochs_clf" in command or "--device" in command:
                raise AssertionError(f"unexpected override in {command}")


class TestMainDispatcher(unittest.TestCase):
    """main.py must route every command to an importable module and reject unknown ones."""

    def test_every_command_has_a_main(self) -> None:
        """Each command module exposes main(argv)."""
        for command, module_name in main.COMMANDS.items():
            module = __import__(module_name, fromlist=["main"])
            if not callable(getattr(module, "main", None)):
                raise AssertionError(f"{command}: {module_name} has no main()")

    def test_unknown_command(self) -> None:
        """An unknown command exits with code 2."""
        if main.main(["no-such-command"]) != 2:
            raise AssertionError("unknown command must return exit code 2")

    def test_run_all_dry_run(self) -> None:
        """The run-all dry run succeeds through the dispatcher."""
        if main.main(["run-all", "--dry_run"]) != 0:
            raise AssertionError("run-all --dry_run must return 0")


if __name__ == "__main__":
    unittest.main()
