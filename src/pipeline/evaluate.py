"""Aggregate every finished classifier run and compare methods statistically.

Reads ``results.json`` written by ``python main.py train`` for the attention-guided CycleGAN classifier and the baselines in both
evaluation scenarios, prints a metrics table per scenario, runs pairwise McNemar tests (Eq. 9 of Baydilli, 2025) on
the matched test predictions, and writes ``summary.md`` and ``summary.json`` to the output directory.

Usage:
    python main.py evaluate [--checkpoints_dir checkpoints] [--output_dir reports]
"""

import argparse
import json
from datetime import datetime
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np

from src.utils.metrics import mcnemar_test, wilson_interval

SCENARIOS: dict[str, dict[str, Any]] = {
    "cell_level": {
        "title": "Scenario 1 (cell_level): paper protocol",
        "note": "Test set of 100 ALL + 100 Normal cells, as in Baydilli (2025). Balanced, so Accuracy is meaningful.",
        "primary_metrics": ["Accuracy", "Balanced_Accuracy", "F_Score"],
    },
    "slide_level": {
        "title": "Scenario 2 (slide_level): leak-free split",
        "note": (
            "Test set of 113 ALL + 25 Normal cells from held-out slides. The classes are imbalanced, so methods should "
            "be judged on Balanced Accuracy and AUROC; plain Accuracy is dominated by the ALL class."
        ),
        "primary_metrics": ["Balanced_Accuracy", "AUROC"],
    },
}

METHOD_DIRS: dict[str, str] = {
    "attention_gan": "classifier_{scenario}",
    "source_only": "baseline_source_only_{scenario}",
    "composite": "baseline_composite_{scenario}",
    "reinhard": "baseline_reinhard_{scenario}",
    "target_supervised": "baseline_target_supervised_{scenario}",
}

PAPER_REFERENCE_NAME = "Baydilli (2025) Table 5"
PAPER_REFERENCE: dict[str, Any] = {
    "num_test": 200,
    "selection_rule": "-",
    "test_metrics": {
        "TN": 91.0,
        "TP": 87.0,
        "FP": 9.0,
        "FN": 13.0,
        "Accuracy": 0.8900,
        "Balanced_Accuracy": 0.8900,
        "Precision": 0.9063,
        "Recall": 0.8700,
        "Specificity": 0.9100,
        "NPV": 0.8750,
        "F_Score": 0.8878,
    },
}
PAPER_REFERENCE_FOOTNOTE = (
    "Reference row: the proposed model of Baydilli (2025) Table 5 (TN 91, TP 87 on 100 ALL + 100 Normal). The "
    "paper's Precision, Recall and Specificity columns are mislabelled; the values shown are the paper's numbers "
    "re-assigned to the correct columns (its 'Precision' 0.8700 is Recall, its 'Recall' 0.9063 is Precision, its "
    "'Specificity' 0.8750 is NPV; Specificity = 91/100). Balanced Accuracy and the Wilson CI are derived from the "
    "same confusion matrix. No per-image predictions are published, so this row is excluded from the McNemar tests."
)

TABLE_COLUMNS: list[tuple[str, str]] = [
    ("Balanced_Accuracy", "Bal. Acc."),
    ("AUROC", "AUROC"),
    ("Precision", "Precision"),
    ("Recall", "Recall"),
    ("Specificity", "Specificity"),
    ("NPV", "NPV"),
    ("F_Score", "F-Score"),
]
EXACT_TEST_THRESHOLD = 25


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command line arguments.

    Args:
        argv: Argument list; defaults to sys.argv[1:].

    Returns:
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        prog="python main.py evaluate", description="Aggregate finished runs and compare methods with McNemar's test."
    )
    parser.add_argument("--checkpoints_dir", type=str, default="checkpoints", help="Directory holding the run folders.")
    parser.add_argument("--output_dir", type=str, default="reports", help="Directory for summary.md and summary.json.")
    return parser.parse_args(argv)


def load_run(path: Path) -> tuple[dict[str, Any] | None, str]:
    """Load one results.json file.

    Args:
        path: Path to the results.json file.

    Returns:
        Tuple of (parsed results or None, status), where status is "ok", "not run" or a short error description.
    """
    if not path.is_file():
        return None, "not run"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return None, f"unreadable ({type(exc).__name__})"
    if not isinstance(data, dict) or not isinstance(data.get("test_metrics"), dict):
        return None, "unreadable (no test_metrics)"
    return data, "ok"


def format_metric(value: Any) -> str:
    """Format a metric value with four decimals, or "-" when absent.

    Args:
        value: Metric value or None.

    Returns:
        Formatted string.
    """
    if value is None:
        return "-"
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return "-"


def format_count(value: Any) -> str:
    """Format an integer-valued count, or "-" when absent.

    Args:
        value: Count value or None.

    Returns:
        Formatted string.
    """
    if value is None:
        return "-"
    try:
        return str(int(round(float(value))))
    except (TypeError, ValueError):
        return "-"


def accuracy_with_ci(metrics: dict[str, Any], num_test: int | None) -> tuple[float | None, float | None, float | None]:
    """Return accuracy and its Wilson 95% interval, recomputing the interval from counts when it is missing.

    Args:
        metrics: Test metrics of a run.
        num_test: Number of test samples, used when the confusion counts are incomplete.

    Returns:
        Tuple of (accuracy, ci_low, ci_high); entries are None when they cannot be determined.
    """
    accuracy = metrics.get("Accuracy")
    low, high = metrics.get("Accuracy_CI95_Low"), metrics.get("Accuracy_CI95_High")
    if low is not None and high is not None:
        return accuracy, low, high
    counts = [metrics.get(key) for key in ("TP", "TN", "FP", "FN")]
    if all(count is not None for count in counts):
        total = int(sum(counts))
        correct = int(counts[0] + counts[1])
    elif accuracy is not None and num_test:
        total = int(num_test)
        correct = int(round(float(accuracy) * total))
    else:
        return accuracy, None, None
    low, high = wilson_interval(correct, total)
    return accuracy, low, high


def build_metric_row(method: str, run: dict[str, Any]) -> dict[str, Any]:
    """Collect the table fields of one run.

    Args:
        method: Method name shown in the table.
        run: Parsed results.json content.

    Returns:
        Dictionary with the method name, test size, accuracy with CI, the remaining metrics and the selection rule.
    """
    metrics = run["test_metrics"]
    num_test = run.get("num_test")
    accuracy, low, high = accuracy_with_ci(metrics, num_test)
    selection = str(run.get("selection_rule", "-"))
    if run.get("selected_epoch") is not None and "epoch" not in selection:
        selection = f"{selection} (epoch {run['selected_epoch']})"
    return {
        "method": method,
        "n_test": num_test,
        "TN": metrics.get("TN"),
        "TP": metrics.get("TP"),
        "Accuracy": accuracy,
        "Accuracy_CI95_Low": low,
        "Accuracy_CI95_High": high,
        **{key: metrics.get(key) for key, _ in TABLE_COLUMNS},
        "selection_rule": selection,
    }


def predictions_by_file(run: dict[str, Any]) -> tuple[dict[str, tuple[int, int]] | None, str]:
    """Index the test predictions of a run by file name.

    Args:
        run: Parsed results.json content.

    Returns:
        Tuple of (mapping file -> (label, pred) or None, reason when the mapping cannot be built).
    """
    records = run.get("test_predictions")
    if not isinstance(records, list) or not records:
        return None, "no test_predictions"
    mapping: dict[str, tuple[int, int]] = {}
    for record in records:
        try:
            name, label, pred = str(record["file"]), int(record["label"]), int(record["pred"])
        except (KeyError, TypeError, ValueError):
            return None, "malformed test_predictions"
        if name in mapping:
            return None, f"duplicate test file {name}"
        mapping[name] = (label, pred)
    return mapping, ""


def compare_pair(name_a: str, run_a: dict[str, Any], name_b: str, run_b: dict[str, Any]) -> dict[str, Any]:
    """Run McNemar's test between two runs on their shared test files.

    Args:
        name_a: Method name of model A.
        run_a: Parsed results of model A.
        name_b: Method name of model B.
        run_b: Parsed results of model B.

    Returns:
        Dictionary with the method names, status ("ok" or "not comparable"), and either the test statistics or the
        reason the pair could not be compared.
    """
    result: dict[str, Any] = {"method_a": name_a, "method_b": name_b}
    preds_a, reason_a = predictions_by_file(run_a)
    preds_b, reason_b = predictions_by_file(run_b)
    if preds_a is None or preds_b is None:
        reasons = [f"{name}: {reason}" for name, reason in ((name_a, reason_a), (name_b, reason_b)) if reason]
        return {**result, "status": "not comparable", "reason": "; ".join(reasons)}

    files_a, files_b = set(preds_a), set(preds_b)
    if files_a != files_b:
        reason = (
            f"test files differ ({len(files_a - files_b)} only in {name_a}, {len(files_b - files_a)} only in {name_b})"
        )
        return {**result, "status": "not comparable", "reason": reason}

    files = sorted(files_a)
    if any(preds_a[f][0] != preds_b[f][0] for f in files):
        return {**result, "status": "not comparable", "reason": "labels disagree for the same test file"}

    y_true = np.array([preds_a[f][0] for f in files])
    test = mcnemar_test(y_true, np.array([preds_a[f][1] for f in files]), np.array([preds_b[f][1] for f in files]))
    return {
        **result,
        "status": "ok",
        "n": len(files),
        **test,
        "exact_recommended": test["b"] + test["c"] < EXACT_TEST_THRESHOLD,
    }


def evaluate_scenario(checkpoints_dir: Path, scenario: str) -> dict[str, Any]:
    """Collect runs, metric rows and pairwise McNemar tests of one scenario.

    Args:
        checkpoints_dir: Directory holding the run folders.
        scenario: Scenario name, "cell_level" or "slide_level".

    Returns:
        Dictionary with the scenario description, metric rows, run statuses and McNemar results.
    """
    runs: dict[str, dict[str, Any]] = {}
    statuses: dict[str, dict[str, str]] = {}
    for method, pattern in METHOD_DIRS.items():
        path = checkpoints_dir / pattern.format(scenario=scenario) / "results.json"
        run, status = load_run(path)
        statuses[method] = {"status": status, "path": path.as_posix()}
        if run is not None:
            runs[method] = run

    rows = [build_metric_row(method, run) for method, run in runs.items()]
    reference = None
    if scenario == "cell_level":
        reference = build_metric_row(PAPER_REFERENCE_NAME, PAPER_REFERENCE)
        reference["selection_rule"] = "-"
    comparisons = [compare_pair(a, runs[a], b, runs[b]) for a, b in combinations(runs, 2)]

    return {
        **SCENARIOS[scenario],
        "runs": statuses,
        "metrics": rows,
        "reference": reference,
        "reference_footnote": PAPER_REFERENCE_FOOTNOTE if reference else None,
        "run_details": {
            method: {
                key: run.get(key) for key in ("selection_rule", "selected_epoch", "num_train", "num_val", "num_test")
            }
            | {"test_metrics": run["test_metrics"]}
            for method, run in runs.items()
        },
        "mcnemar": comparisons,
    }


def markdown_table(header: list[str], rows: list[list[str]]) -> list[str]:
    """Render a markdown table.

    Args:
        header: Column titles.
        rows: Table body, one list of cell strings per row.

    Returns:
        Lines of the table.
    """
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return lines


def metrics_table(rows: list[dict[str, Any]]) -> list[str]:
    """Render the per-method metrics table.

    Args:
        rows: Metric rows from build_metric_row.

    Returns:
        Lines of the markdown table.
    """
    header = ["Method", "n_test", "TN", "TP", "Accuracy [Wilson 95% CI]"] + [title for _, title in TABLE_COLUMNS]
    header.append("Selection rule")
    body = []
    for row in rows:
        accuracy = format_metric(row["Accuracy"])
        if row["Accuracy_CI95_Low"] is not None and row["Accuracy_CI95_High"] is not None:
            accuracy += f" [{row['Accuracy_CI95_Low']:.4f}, {row['Accuracy_CI95_High']:.4f}]"
        cells = [row["method"], format_count(row["n_test"]), format_count(row["TN"]), format_count(row["TP"]), accuracy]
        cells += [format_metric(row[key]) for key, _ in TABLE_COLUMNS]
        cells.append(row["selection_rule"])
        body.append(cells)
    return markdown_table(header, body)


def mcnemar_table(comparisons: list[dict[str, Any]]) -> list[str]:
    """Render the pairwise McNemar table.

    Args:
        comparisons: Results of compare_pair.

    Returns:
        Lines of the markdown table.
    """
    header = ["Method A", "Method B", "n", "b", "c", "chi2", "p (chi2)", "p (exact)", "Significant", "Note"]
    body = []
    for comp in comparisons:
        if comp["status"] != "ok":
            body.append([comp["method_a"], comp["method_b"]] + ["-"] * 7 + [f"not comparable: {comp['reason']}"])
            continue
        note = "b + c < 25, use exact p" if comp["exact_recommended"] else ""
        body.append(
            [
                comp["method_a"],
                comp["method_b"],
                str(comp["n"]),
                format_count(comp["b"]),
                format_count(comp["c"]),
                f"{comp['chi2']:.4f}",
                f"{comp['p_value_chi2']:.4g}",
                f"{comp['p_value_exact']:.4g}",
                "yes" if comp["significant"] else "no",
                note,
            ]
        )
    return markdown_table(header, body)


def render_markdown(summary: dict[str, Any]) -> str:
    """Render the full summary as markdown.

    Args:
        summary: Output of build_summary.

    Returns:
        Markdown document.
    """
    lines = [
        "# Final evaluation summary",
        "",
        f"Generated {summary['generated_at']} from `{summary['checkpoints_dir']}`.",
        "",
        "Scenario 2 (slide_level) has an imbalanced test set (113 ALL + 25 Normal); judge it on Balanced Accuracy and "
        "AUROC rather than Accuracy.",
    ]
    for scenario in summary["scenarios"].values():
        lines += ["", f"## {scenario['title']}", "", scenario["note"], ""]
        rows = scenario["metrics"] + ([scenario["reference"]] if scenario["reference"] else [])
        if rows:
            lines += metrics_table(rows)
        else:
            lines.append("No finished runs.")
        if scenario["reference_footnote"]:
            lines += ["", f"Note: {scenario['reference_footnote']}"]

        not_run = [method for method, info in scenario["runs"].items() if info["status"] == "not run"]
        broken = [
            f"{method} ({info['status']})"
            for method, info in scenario["runs"].items()
            if info["status"] not in ("ok", "not run")
        ]
        if not_run:
            lines += ["", "Not run: " + ", ".join(not_run) + "."]
        if broken:
            lines += ["", "Skipped: " + ", ".join(broken) + "."]

        lines += ["", "### Pairwise McNemar tests", ""]
        if scenario["mcnemar"]:
            lines += mcnemar_table(scenario["mcnemar"])
            lines += [
                "",
                "b = test images model A classifies correctly and model B wrongly, c = the reverse. "
                "chi2 = (|b - c| - 1)^2 / (b + c) (Eq. 9, continuity corrected, 1 dof); significant means chi2 > 3.841 "
                "(alpha = 0.05). The exact binomial p-value is preferable when b + c < 25.",
            ]
        else:
            lines.append("Fewer than two finished runs; no comparison possible.")
    return "\n".join(lines) + "\n"


def build_summary(checkpoints_dir: Path) -> dict[str, Any]:
    """Evaluate every scenario.

    Args:
        checkpoints_dir: Directory holding the run folders.

    Returns:
        Machine-readable summary of all scenarios.
    """
    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "checkpoints_dir": checkpoints_dir.as_posix(),
        "scenarios": {scenario: evaluate_scenario(checkpoints_dir, scenario) for scenario in SCENARIOS},
    }


def main(argv: list[str] | None = None) -> int:
    """Run the final evaluation.

    Args:
        argv: Argument list; defaults to sys.argv[1:].

    Returns:
        Process exit code.
    """
    args = parse_args(argv)
    checkpoints_dir = Path(args.checkpoints_dir)
    summary = build_summary(checkpoints_dir)

    if not any(scenario["metrics"] for scenario in summary["scenarios"].values()):
        print(f"No finished runs found under '{checkpoints_dir}' (expected <run>/results.json). Nothing to evaluate.")
        return 0

    markdown = render_markdown(summary)
    print(markdown)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.md").write_text(markdown, encoding="utf-8")
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote {output_dir / 'summary.md'} and {output_dir / 'summary.json'}")
    return 0
