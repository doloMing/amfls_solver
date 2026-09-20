from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import statistics


FORMAL_METHODS = (
    "amfls",
    "fixed_rbgk",
    "lsqr",
    "lsmr",
    "lsmb",
    "lsrn",
    "sparse_embedding_lsqr",
    "spir",
    "fossils",
)

CSR_NATIVE_METHODS = {
    "sparse_embedding_lsqr",
    "spir",
    "fossils",
}

STATUS_NAMES = {
    0: "success",
    1: "work limit",
    2: "basis limit",
    3: "precision limit",
    4: "numerical breakdown",
}

STOP_REASON_NAMES = {
    0: "certified optimality",
    1: "exact stationarity",
    2: "exhausted search space",
    3: "maximum depth",
    4: "maximum epochs",
    5: "maximum basis",
    6: "precision limit",
    7: "numerical breakdown",
    8: "compatible backward error",
    9: "least-squares backward error",
    10: "relative energy error",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Write the large sparse application comparison as Markdown."
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        action="append",
        help="Raw CSV shard; repeat once per registered matrix.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Frozen sparse application configuration used by the runner.",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def finite_float(row: dict[str, str], field: str) -> float | None:
    text = row.get(field, "").strip()
    if not text:
        return None
    value = float(text)
    return value if math.isfinite(value) else None


def integer(row: dict[str, str], field: str) -> int:
    text = row.get(field, "").strip()
    return int(text) if text else 0


def passed(row: dict[str, str]) -> bool:
    return integer(row, "success") == 1 and integer(row, "external_success") == 1


def format_number(value: float | None) -> str:
    if value is None:
        return "--"
    if value == 0.0:
        return "0"
    if abs(value) >= 1000.0 or abs(value) < 1.0e-3:
        return f"{value:.3e}"
    return f"{value:.4g}"


def median_field(rows: list[dict[str, str]], field: str) -> float | None:
    values = [finite_float(row, field) for row in rows]
    finite = [value for value in values if value is not None]
    return statistics.median(finite) if finite else None


def load_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {
            "suite",
            "case",
            "instance_seed",
            "method",
            "m",
            "n",
            "nnz",
            "success",
            "external_success",
            "status",
            "stop_reason",
            "wall_seconds",
            "kernel_seconds",
            "total_vector_work",
            "total_block_calls",
            "search_vector_work",
            "residual_norm",
            "normal_residual_norm",
            "compatible_backward_error_external",
            "least_squares_backward_error_external",
            "backward_error_external",
            "search_block_calls",
            "validation_block_calls",
            "auxiliary_width",
            "depth",
            "error",
        }
        missing = required - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"missing result columns: {sorted(missing)}")
        rows = [dict(row) for row in reader]
    if not rows:
        raise ValueError("no sparse application result rows")
    return rows


def expected_grid(config_path: Path) -> set[tuple[str, str, int, str]]:
    if not config_path.is_file():
        raise FileNotFoundError(config_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    methods = tuple(str(value) for value in config["methods"])
    if methods != FORMAL_METHODS:
        raise ValueError("sparse formal config has an unexpected method list")
    suite = str(config["suite"])
    seeds = [int(value) for value in config["instance_seeds"]]
    expected: set[tuple[str, str, int, str]] = set()
    for case in config["cases"]:
        case_name = str(case["name"])
        case_seeds = [int(value) for value in case.get("instance_seeds", seeds)]
        for seed in case_seeds:
            for method in methods:
                key = (suite, case_name, seed, method)
                if key in expected:
                    raise ValueError(f"duplicate row required by formal config: {key}")
                expected.add(key)
    if not expected:
        raise ValueError("sparse formal config defines an empty result grid")
    return expected


def validate_grid(rows: list[dict[str, str]], config_path: Path) -> None:
    observed: set[tuple[str, str, int, str]] = set()
    for row in rows:
        method = row["method"]
        if method not in FORMAL_METHODS:
            raise ValueError(f"noncanonical method in formal results: {method}")
        key = (
            row["suite"],
            row["case"],
            int(row["instance_seed"]),
            method,
        )
        if key in observed:
            raise ValueError(f"duplicate result row: {key}")
        observed.add(key)
    expected = expected_grid(config_path)
    missing = sorted(expected - observed)
    extra = sorted(observed - expected)
    if missing or extra:
        details: list[str] = []
        if missing:
            details.append(f"missing {len(missing)} rows; first: {missing[:5]}")
        if extra:
            details.append(f"extra {len(extra)} rows; first: {extra[:5]}")
        raise ValueError("formal result grid mismatch: " + "; ".join(details))


def dataset_table(rows: list[dict[str, str]]) -> list[str]:
    lines = [
        "| case | rows | columns | nonzeros | right-hand side |",
        "|---|---:|---:|---:|---|",
    ]
    for case in sorted({row["case"] for row in rows}):
        selected = [row for row in rows if row["case"] == case]
        first = selected[0]
        lines.append(
            f"| {case} | {integer(first, 'm')} | {integer(first, 'n')} | "
            f"{integer(first, 'nnz')} | {first['rhs']} |"
        )
    return lines


def terminal_outcome(row: dict[str, str]) -> str:
    if row.get("error", "").strip():
        return "exception"
    status = integer(row, "status")
    stop_reason = integer(row, "stop_reason")
    status_name = STATUS_NAMES.get(status, f"status {status}")
    stop_name = STOP_REASON_NAMES.get(
        stop_reason, f"stop reason {stop_reason}"
    )
    if status == 0:
        return (
            "success"
            if passed(row)
            else "success / external failure"
        )
    return f"{status_name} / {stop_name}"


def outcome_summary(rows: list[dict[str, str]]) -> str:
    counts: dict[str, int] = {}
    for row in rows:
        outcome = terminal_outcome(row)
        counts[outcome] = counts.get(outcome, 0) + 1
    return "; ".join(
        f"{outcome}: {counts[outcome]}" for outcome in sorted(counts)
    )


def reliability_table(rows: list[dict[str, str]]) -> list[str]:
    lines = [
        "Attempt medians use all attempts. An exception contributes its recorded "
        "wall time but has no solver work counters. Recorded work is the callback "
        "ledger; for the CSR-native sparse-embedding methods it omits their direct "
        "nonzero scans and is therefore not used for cross-method ratios.",
        "",
        "| method | formal pass | exceptions | terminal outcomes | false accepts | "
        "median attempt wall (s) | median recorded attempt vectors | "
        "worst accepted backward error |",
        "|---|---:|---:|---|---:|---:|---:|---:|",
    ]
    for method in FORMAL_METHODS:
        selected = [row for row in rows if row["method"] == method]
        accepted = [row for row in selected if passed(row)]
        errors = sum(bool(row.get("error", "").strip()) for row in selected)
        false_accepts = sum(
            integer(row, "success") == 1 and integer(row, "external_success") != 1
            for row in selected
        )
        backward = [
            finite_float(row, "backward_error_external") for row in accepted
        ]
        finite_backward = [value for value in backward if value is not None]
        worst = max(finite_backward) if finite_backward else None
        lines.append(
            f"| {method} | {len(accepted)}/{len(selected)} | {errors} | "
            f"{outcome_summary(selected)} | {false_accepts} | "
            f"{format_number(median_field(selected, 'wall_seconds'))} | "
            f"{format_number(median_field(selected, 'total_vector_work'))} | "
            f"{format_number(worst)} |"
        )
    return lines


def paired_ratios(
    rows: list[dict[str, str]],
    baseline: str,
    field: str,
) -> tuple[list[float], int]:
    by_instance: dict[tuple[str, str], dict[str, dict[str, str]]] = {}
    for row in rows:
        key = (row["case"], row["instance_seed"])
        by_instance.setdefault(key, {})[row["method"]] = row
    case_medians: list[float] = []
    paired = 0
    for case in sorted({key[0] for key in by_instance}):
        instances = [
            methods for (name, _), methods in by_instance.items() if name == case
        ]
        values: list[float] = []
        complete = bool(instances)
        for methods in instances:
            amfls = methods["amfls"]
            reference = methods[baseline]
            numerator = finite_float(amfls, field)
            denominator = finite_float(reference, field)
            if (
                not passed(amfls)
                or not passed(reference)
                or numerator is None
                or denominator is None
                or denominator <= 0.0
            ):
                complete = False
                break
            values.append(numerator / denominator)
        if complete:
            case_medians.append(statistics.median(values))
            paired += len(values)
    return case_medians, paired


def comparison_table(rows: list[dict[str, str]]) -> list[str]:
    lines = [
        "Ratios are `AMFLS/baseline`; values below one favor AMFLS. A matrix is "
        "included only when every seed passes the formal accuracy test for both methods. "
        "The CSR-native sparse-embedding methods perform direct nonzero scans that "
        "are included in wall time but not in the callback ledgers, so their vector, "
        "search-vector, and block-call ratios are shown as `--`.",
        "",
        "| baseline | complete matrices | paired runs | wall ratio | "
        "vector-work ratio | search-vector ratio | block-call ratio |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for baseline in FORMAL_METHODS[1:]:
        wall, paired = paired_ratios(rows, baseline, "wall_seconds")
        if baseline in CSR_NATIVE_METHODS:
            vector_ratio = None
            search_vector_ratio = None
            block_ratio = None
        else:
            vectors, _ = paired_ratios(rows, baseline, "total_vector_work")
            search_vectors, _ = paired_ratios(
                rows, baseline, "search_vector_work"
            )
            blocks, _ = paired_ratios(rows, baseline, "total_block_calls")
            vector_ratio = statistics.median(vectors) if vectors else None
            search_vector_ratio = (
                statistics.median(search_vectors) if search_vectors else None
            )
            block_ratio = statistics.median(blocks) if blocks else None
        lines.append(
            f"| {baseline} | {len(wall)} | {paired} | "
            f"{format_number(statistics.median(wall) if wall else None)} | "
            f"{format_number(vector_ratio)} | "
            f"{format_number(search_vector_ratio)} | "
            f"{format_number(block_ratio)} |"
        )
    return lines


def case_table(rows: list[dict[str, str]]) -> list[str]:
    lines = [
        "Attempt wall and work medians use all attempts with a recorded value; "
        "accuracy medians use formally accepted attempts only.",
        "",
        "| case | method | formal pass | terminal outcomes | attempt wall (s) | "
        "attempt kernel (s) | recorded attempt vectors | recorded attempt search "
        "vectors | recorded attempt block calls | accepted residual | accepted "
        "normal residual | accepted backward error | attempt width | attempt depth |",
        "|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for case in sorted({row["case"] for row in rows}):
        for method in FORMAL_METHODS:
            selected = [
                row
                for row in rows
                if row["case"] == case and row["method"] == method
            ]
            accepted = [row for row in selected if passed(row)]
            lines.append(
                f"| {case} | {method} | {len(accepted)}/{len(selected)} | "
                f"{outcome_summary(selected)} | "
                f"{format_number(median_field(selected, 'wall_seconds'))} | "
                f"{format_number(median_field(selected, 'kernel_seconds'))} | "
                f"{format_number(median_field(selected, 'total_vector_work'))} | "
                f"{format_number(median_field(selected, 'search_vector_work'))} | "
                f"{format_number(median_field(selected, 'total_block_calls'))} | "
                f"{format_number(median_field(accepted, 'residual_norm'))} | "
                f"{format_number(median_field(accepted, 'normal_residual_norm'))} | "
                f"{format_number(median_field(accepted, 'backward_error_external'))} | "
                f"{format_number(median_field(selected, 'auxiliary_width'))} | "
                f"{format_number(median_field(selected, 'depth'))} |"
            )
    return lines


def render(rows: list[dict[str, str]], input_paths: list[Path]) -> str:
    suite_names = sorted({row["suite"] for row in rows})
    lines = [
        "# Large sparse application study",
        "",
        "Inputs: " + ", ".join(path.name for path in input_paths) + ".",
        "",
        "Suites: " + ", ".join(suite_names) + ".",
        "",
        "## Matrices",
        "",
        *dataset_table(rows),
        "",
        "## Reliability and accuracy",
        "",
        *reliability_table(rows),
        "",
        "## AMFLS baseline comparisons",
        "",
        *comparison_table(rows),
        "",
        "## Per-matrix results",
        "",
        *case_table(rows),
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    rows = [row for path in args.input for row in load_rows(path)]
    validate_grid(rows, args.config)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(render(rows, args.input), encoding="utf-8")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
