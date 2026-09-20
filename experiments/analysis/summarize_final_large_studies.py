from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import statistics


CANONICAL_METHODS = (
    "amfls",
    "fixed_rbgk",
    "lsqr",
    "lsmr",
    "lsmb",
    "lsrn",
    "sparse_embedding_lsqr",
    "spir",
    "fossils",
    "rbcg",
    "aplicur",
)

PRECISION_STATUS_METHODS = frozenset({"amfls", "fixed_rbgk"})

RATIO_METRICS = (
    ("wall", "wall_seconds"),
    ("vectors", "total_vector_work"),
    ("search vectors", "search_vector_work"),
    ("block calls", "total_block_calls"),
)

CASE_NUMERIC_PARAMETER_FIELDS = (
    "outlier_count",
    "outlier_ratio",
    "smallest_outlier_ratio",
    "alpha",
    "endpoint",
    "two_mode_small_singular_value",
    "noise_ratio",
)

METHOD_SEED_OFFSETS = {
    "fixed_rbgk": 1_000_003,
    "lsrn": 2_000_003,
    "sparse_embedding_lsqr": 3_000_007,
    "spir": 4_000_009,
    "fossils": 5_000_011,
    "aplicur": 6_000_013,
    "rbcg": 7_000_027,
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
        description="Summarize the frozen formal dense large-matrix studies."
    )
    parser.add_argument(
        "--input",
        type=Path,
        action="append",
        required=True,
        help="Result CSV. Repeat this option to combine formal suites.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        action="append",
        required=True,
        help="Frozen formal config. Repeat once for every summarized suite.",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def finite_float(row: dict[str, str], field: str) -> float | None:
    value = row.get(field, "").strip()
    if not value:
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def integer(row: dict[str, str], field: str) -> int:
    value = row.get(field, "").strip()
    return int(value) if value else 0


def passed(row: dict[str, str]) -> bool:
    return integer(row, "success") == 1 and integer(row, "external_success") == 1


def is_precision_status_case(row: dict[str, str]) -> bool:
    return integer(row, "precision_status_case") == 1


def precision_status_summary(rows: list[dict[str, str]]) -> str:
    selected = [
        row
        for row in rows
        if is_precision_status_case(row)
        and row.get("precision_status_correct", "").strip()
    ]
    if not selected:
        return "--"
    correct = sum(integer(row, "precision_status_correct") == 1 for row in selected)
    return f"{correct}/{len(selected)}"


def primary_accuracy(row: dict[str, str]) -> float | None:
    regularization_ratio = finite_float(row, "regularization_ratio") or 0.0
    field = (
        "relative_energy_error_external"
        if regularization_ratio > 0.0
        else "backward_error_external"
    )
    return finite_float(row, field)


def median_field(rows: list[dict[str, str]], field: str) -> float | None:
    values = [finite_float(row, field) for row in rows]
    finite = [value for value in values if value is not None]
    return statistics.median(finite) if finite else None


def median_primary_accuracy(rows: list[dict[str, str]]) -> float | None:
    values = [primary_accuracy(row) for row in rows]
    finite = [value for value in values if value is not None]
    return statistics.median(finite) if finite else None


def interquartile_range(values: list[float]) -> float | None:
    finite = [value for value in values if math.isfinite(value)]
    if not finite:
        return None
    if len(finite) == 1:
        return 0.0
    quartiles = statistics.quantiles(finite, n=4, method="inclusive")
    return quartiles[2] - quartiles[0]


def iqr_field(rows: list[dict[str, str]], field: str) -> float | None:
    values = [finite_float(row, field) for row in rows]
    return interquartile_range(
        [value for value in values if value is not None]
    )


def median_case_wall_iqr(rows: list[dict[str, str]]) -> float | None:
    case_iqrs = [
        iqr_field(
            [row for row in rows if row["case"] == case],
            "wall_seconds",
        )
        for case in sorted({row["case"] for row in rows})
    ]
    finite = [value for value in case_iqrs if value is not None]
    return statistics.median(finite) if finite else None


def format_number(value: float | None) -> str:
    if value is None:
        return "--"
    if value == 0.0:
        return "0"
    if abs(value) >= 1000.0 or abs(value) < 1.0e-3:
        return f"{value:.3e}"
    return f"{value:.4g}"


def load_rows(paths: list[Path]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
        with path.open(encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            required = {
                "suite",
                "case",
                "instance_seed",
                "algorithm_seed",
                "method",
                "m",
                "n",
                "rank",
                "coherence",
                "spectrum",
                "rhs",
                "success",
                "external_success",
                "precision_status_case",
                "precision_status_correct",
                "status",
                "stop_reason",
                "regularization",
                "regularization_ratio",
                "tolerance",
                "backward_error_external",
                "relative_energy_error_external",
                "wall_seconds",
                "total_vector_work",
                "search_vector_work",
                "total_block_calls",
                "search_block_calls",
                "validation_block_calls",
                "relative_solution_error",
                "relative_null_component",
                "auxiliary_width",
                "depth",
                "error",
            }
            missing = required - set(reader.fieldnames or ())
            if missing:
                raise ValueError(f"{path} is missing columns: {sorted(missing)}")
            rows.extend(dict(row) for row in reader)
    if not rows:
        raise ValueError("no result rows")
    return rows


def load_configs(paths: list[Path]) -> dict[str, dict[str, object]]:
    configs: dict[str, dict[str, object]] = {}
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
        config = json.loads(path.read_text(encoding="utf-8"))
        suite = str(config["suite"])
        if suite in configs:
            raise ValueError(f"duplicate formal suite config: {suite}")
        configs[suite] = config
    return configs


def expected_grid(
    config_paths: list[Path],
) -> dict[tuple[str, str, int, str], dict[str, int | float]]:
    expected: dict[
        tuple[str, str, int, str], dict[str, int | float]
    ] = {}
    for config in load_configs(config_paths).values():
        suite = str(config["suite"])
        methods = [str(value) for value in config["methods"]]
        seeds = [int(value) for value in config["instance_seeds"]]
        tolerance = float(config["tolerance"])
        algorithm_seed_offset = int(config["algorithm_seed_offset"])
        for case in config["cases"]:
            case_name = str(case["name"])
            case_seeds = [
                int(value) for value in case.get("instance_seeds", seeds)
            ]
            case_tolerance = float(case.get("tolerance", tolerance))
            regularization_ratio = float(case.get("regularization_ratio", 0.0))
            for seed in case_seeds:
                for method in methods:
                    key = (suite, case_name, seed, method)
                    if key in expected:
                        raise ValueError(
                            f"duplicate row required by formal configs: {key}"
                        )
                    expected[key] = {
                        "m": int(case["m"]),
                        "n": int(case["n"]),
                        "rank": int(case["rank"]),
                        "tolerance": case_tolerance,
                        "regularization_ratio": regularization_ratio,
                        "precision_status_case": int(
                            bool(case.get("precision_status_case", False))
                        ),
                        "algorithm_seed": seed
                        + algorithm_seed_offset
                        + METHOD_SEED_OFFSETS.get(method, 0),
                    }
    if not expected:
        raise ValueError("formal configs define an empty result grid")
    return expected


def require_integer(
    row: dict[str, str], field: str, key: tuple[object, ...]
) -> int:
    value = row.get(field, "").strip()
    if not value:
        raise ValueError(f"result row {key} has an empty {field}")
    try:
        return int(value)
    except ValueError as error:
        raise ValueError(f"result row {key} has invalid {field}: {value!r}") from error


def require_finite_float(
    row: dict[str, str], field: str, key: tuple[object, ...]
) -> float:
    value = finite_float(row, field)
    if value is None:
        raise ValueError(f"result row {key} has invalid or empty {field}")
    return value


def floats_match(observed: float, expected: float) -> bool:
    return math.isclose(observed, expected, rel_tol=1.0e-12, abs_tol=0.0)


def validate_grid(rows: list[dict[str, str]], config_paths: list[Path]) -> None:
    allowed = set(CANONICAL_METHODS)
    observed: dict[tuple[str, str, int, str], dict[str, str]] = {}
    for row in rows:
        method = row["method"]
        if method not in allowed:
            raise ValueError(f"noncanonical method in final results: {method}")
        key = (
            row["suite"],
            row["case"],
            int(row["instance_seed"]),
            method,
        )
        if key in observed:
            raise ValueError(f"duplicate result row: {key}")
        observed[key] = row
    expected = expected_grid(config_paths)
    missing = sorted(set(expected) - set(observed))
    extra = sorted(set(observed) - set(expected))
    if missing or extra:
        details: list[str] = []
        if missing:
            details.append(f"missing {len(missing)} rows; first: {missing[:5]}")
        if extra:
            details.append(f"extra {len(extra)} rows; first: {extra[:5]}")
        raise ValueError("formal result grid mismatch: " + "; ".join(details))

    for key, metadata in expected.items():
        row = observed[key]
        for field in (
            "m",
            "n",
            "rank",
            "algorithm_seed",
            "precision_status_case",
        ):
            value = require_integer(row, field, key)
            if value != metadata[field]:
                raise ValueError(
                    f"result row {key} has {field}={value}, expected "
                    f"{metadata[field]} from the formal config"
                )
        precision_correct_text = row.get("precision_status_correct", "").strip()
        precision_status_assessed = (
            int(metadata["precision_status_case"]) == 1
            and key[3] in PRECISION_STATUS_METHODS
        )
        if precision_status_assessed:
            precision_correct = require_integer(
                row, "precision_status_correct", key
            )
            if precision_correct not in {0, 1}:
                raise ValueError(
                    f"result row {key} has invalid precision_status_correct="
                    f"{precision_correct}"
                )
        elif precision_correct_text:
            raise ValueError(
                f"unassessed result row {key} has precision_status_correct="
                f"{precision_correct_text!r}"
            )
        for field in ("tolerance", "regularization_ratio"):
            value = require_finite_float(row, field, key)
            expected_value = float(metadata[field])
            if not floats_match(value, expected_value):
                raise ValueError(
                    f"result row {key} has {field}={value}, expected "
                    f"{expected_value} from the formal config"
                )

        if not row.get("error", "").strip():
            status = require_integer(row, "status", key)
            stop_reason = require_integer(row, "stop_reason", key)
            if status not in STATUS_NAMES:
                raise ValueError(f"result row {key} has unknown status {status}")
            if stop_reason not in STOP_REASON_NAMES:
                raise ValueError(
                    f"result row {key} has unknown stop_reason {stop_reason}"
                )
            if primary_accuracy(row) is None:
                field = (
                    "relative_energy_error_external"
                    if float(metadata["regularization_ratio"]) > 0.0
                    else "backward_error_external"
                )
                raise ValueError(
                    f"result row {key} has no finite primary external accuracy "
                    f"in {field}"
                )


def method_order(methods: set[str]) -> list[str]:
    return [method for method in CANONICAL_METHODS if method in methods]


def terminal_outcome(row: dict[str, str]) -> str:
    if row.get("error", "").strip():
        return "exception"
    status = integer(row, "status")
    if status == 0:
        return "success" if passed(row) else "success / external failure"
    reason = integer(row, "stop_reason")
    return f"{STATUS_NAMES[status]} / {STOP_REASON_NAMES[reason]}"


def outcome_summary(rows: list[dict[str, str]]) -> str:
    counts: dict[str, int] = {}
    for row in rows:
        outcome = terminal_outcome(row)
        counts[outcome] = counts.get(outcome, 0) + 1
    return "; ".join(
        f"{outcome}: {counts[outcome]}" for outcome in sorted(counts)
    )


def reliability_table(rows: list[dict[str, str]]) -> list[str]:
    methods = method_order({row["method"] for row in rows})
    lines = [
        "APLICUR uses explicit matrix-entry access. Its full wall time and external "
        "accuracy remain comparable, but matrix-free vector/call work is shown as `--`.",
        "Attempt medians include successful and unsuccessful terminal outcomes; an "
        "exception contributes its recorded wall time but has no solver work counters.",
        "A wall-time IQR is Q3-Q1 using inclusive sample quartiles over seeds "
        "within one case. The reliability-table summary is the median of those "
        "per-case IQRs.",
        "",
        "| method | formal pass | precision status correct | exceptions | "
        "terminal outcomes | false accepts | median attempt wall (s) | "
        "median per-case wall IQR (s) | median attempt vectors | "
        "worst accepted accuracy |",
        "|---|---:|---:|---:|---|---:|---:|---:|---:|---:|",
    ]
    for method in methods:
        selected = [row for row in rows if row["method"] == method]
        accepted = [row for row in selected if passed(row)]
        exceptions = sum(bool(row.get("error", "").strip()) for row in selected)
        false_accepts = sum(
            integer(row, "success") == 1 and integer(row, "external_success") != 1
            for row in selected
        )
        accuracies = [primary_accuracy(row) for row in accepted]
        finite_accuracies = [value for value in accuracies if value is not None]
        worst = max(finite_accuracies) if finite_accuracies else None
        vectors = (
            None
            if method == "aplicur"
            else median_field(selected, "total_vector_work")
        )
        lines.append(
            f"| {method} | {len(accepted)}/{len(selected)} | "
            f"{precision_status_summary(selected)} | {exceptions} | "
            f"{outcome_summary(selected)} | {false_accepts} | "
            f"{format_number(median_field(selected, 'wall_seconds'))} | "
            f"{format_number(median_case_wall_iqr(selected))} | "
            f"{format_number(vectors)} | "
            f"{format_number(worst)} |"
        )
    return lines


def rows_by_instance(
    rows: list[dict[str, str]],
) -> dict[tuple[str, str], dict[str, dict[str, str]]]:
    result: dict[tuple[str, str], dict[str, dict[str, str]]] = {}
    for row in rows:
        key = (row["case"], row["instance_seed"])
        result.setdefault(key, {})[row["method"]] = row
    return result


def complete_case_ratios(
    rows: list[dict[str, str]], reference: str, metric: str
) -> tuple[list[float], list[str], int]:
    instances = rows_by_instance(rows)
    precision_status_cases = {
        row["case"] for row in rows if is_precision_status_case(row)
    }
    cases = sorted(
        {case for case, _ in instances} - precision_status_cases
    )
    case_medians: list[float] = []
    excluded: list[str] = []
    paired_attempts = 0
    for case in cases:
        ratio, attempts = paired_case_ratio(rows, case, reference, metric)
        if ratio is not None:
            case_medians.append(ratio)
            paired_attempts += attempts
        else:
            excluded.append(case)
    return case_medians, excluded, paired_attempts


def paired_case_ratio(
    rows: list[dict[str, str]], case: str, reference: str, metric: str
) -> tuple[float | None, int]:
    instances = rows_by_instance(rows)
    selected = [
        methods for (name, _), methods in instances.items() if name == case
    ]
    if not selected:
        return None, 0
    pairs: list[float] = []
    for methods in selected:
        if "amfls" not in methods or reference not in methods:
            return None, 0
        amfls_row = methods["amfls"]
        reference_row = methods[reference]
        numerator = finite_float(amfls_row, metric)
        denominator = finite_float(reference_row, metric)
        if (
            not passed(amfls_row)
            or not passed(reference_row)
            or numerator is None
            or denominator is None
            or denominator <= 0.0
        ):
            return None, 0
        pairs.append(numerator / denominator)
    return statistics.median(pairs), len(pairs)


def comparison_table(rows: list[dict[str, str]]) -> list[str]:
    references = method_order({row["method"] for row in rows} - {"amfls"})
    lines = [
        "Ratios are `amfls/reference`. Each entry is the median of per-case medians; "
        "a case is included only when every seed passes formally for both methods. "
        "Precision-status cases are excluded from these performance ratios. "
        "APLICUR reports only the wall-time ratio because its explicit entry work is "
        "not a MatrixOperator vector/call count.",
        "SPIR and FOSSILS explicit dense/CSR column-norm scans count in wall time "
        "but not in the callback ledger. Their vector/call ratios therefore are not "
        "complete matrix-free work ratios.",
        "",
        "| reference | complete cases | paired attempts | wall | vectors | "
        "search vectors | block calls | excluded cases |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for reference in references:
        cells: list[str] = []
        common_excluded: set[str] = set()
        complete_case_count = 0
        paired_attempts = 0
        for _, metric in RATIO_METRICS:
            if reference == "aplicur" and metric != "wall_seconds":
                cells.append("--")
                continue
            case_values, excluded, attempts = complete_case_ratios(
                rows, reference, metric
            )
            cells.append(
                format_number(statistics.median(case_values) if case_values else None)
            )
            common_excluded.update(excluded)
            if metric == "wall_seconds":
                complete_case_count = len(case_values)
                paired_attempts = attempts
        excluded_text = ", ".join(sorted(common_excluded)) or "none"
        lines.append(
            f"| {reference} | {complete_case_count} | {paired_attempts} | "
            + " | ".join(cells)
            + f" | {excluded_text} |"
        )
    return lines


def per_case_comparison_table(rows: list[dict[str, str]]) -> list[str]:
    references = method_order({row["method"] for row in rows} - {"amfls"})
    precision_status_cases = {
        row["case"] for row in rows if is_precision_status_case(row)
    }
    cases = sorted({row["case"] for row in rows} - precision_status_cases)
    instances = rows_by_instance(rows)
    lines = [
        "Ratios are `amfls/reference` medians across paired seeds. A row is "
        "reported only when every seed passes formally for both methods; otherwise "
        "its ratios are `--`.",
        "",
        "| case | reference | complete paired seeds | wall | vectors | "
        "search vectors | block calls |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for case in cases:
        seed_count = sum(name == case for name, _ in instances)
        for reference in references:
            wall, paired_attempts = paired_case_ratio(
                rows, case, reference, "wall_seconds"
            )
            cells = [format_number(wall)]
            for _, metric in RATIO_METRICS[1:]:
                if reference == "aplicur":
                    cells.append("--")
                else:
                    ratio, _ = paired_case_ratio(rows, case, reference, metric)
                    cells.append(format_number(ratio))
            lines.append(
                f"| {case} | {reference} | {paired_attempts}/{seed_count} | "
                + " | ".join(cells)
                + " |"
            )
    return lines


def format_config_value(value: object) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return json.dumps(value)
    if isinstance(value, list):
        return ", ".join(format_config_value(item) for item in value)
    return str(value)


def protocol_table(config: dict[str, object]) -> list[str]:
    fields = (
        ("tolerance", "tolerance"),
        ("failure probability", "failure_probability"),
        ("instance seeds", "instance_seeds"),
        ("algorithm seed offset", "algorithm_seed_offset"),
        ("execution-order seed", "order_seed"),
        ("timing repetitions", "timing_repetitions"),
        ("threads", "threads"),
        ("warmup runs", "warmup_runs"),
        ("maximum iterations", "maximum_iterations"),
        ("maximum depth", "maximum_depth"),
        ("maximum auxiliary width", "maximum_auxiliary_width"),
        ("maximum epochs", "maximum_epochs"),
        ("maximum basis size", "maximum_basis_size"),
    )
    lines = [
        "| frozen suite setting | value |",
        "|---|---|",
    ]
    for label, key in fields:
        lines.append(f"| {label} | {format_config_value(config[key])} |")

    default_seeds = config["instance_seeds"]
    default_tolerance = config["tolerance"]
    default_repetitions = config["timing_repetitions"]
    lines.extend(
        [
            "",
            "Case rows below make all case overrides and spectral/RHS numeric "
            "parameters explicit.",
            "",
            "| case | tolerance | instance seeds | timing repetitions | "
            "spectral/RHS numeric parameters |",
            "|---|---:|---|---:|---|",
        ]
    )
    for case_object in config["cases"]:
        case = dict(case_object)
        parameters = "; ".join(
            f"{key}={format_config_value(case[key])}"
            for key in CASE_NUMERIC_PARAMETER_FIELDS
            if key in case
        ) or "--"
        lines.append(
            f"| {case['name']} | "
            f"{format_config_value(case.get('tolerance', default_tolerance))} | "
            f"{format_config_value(case.get('instance_seeds', default_seeds))} | "
            f"{format_config_value(case.get('timing_repetitions', default_repetitions))} | "
            f"{parameters} |"
        )
    return lines


def case_coverage_table(rows: list[dict[str, str]]) -> list[str]:
    lines = [
        "| case | shape | dimensions | rank | rank status | coherence | spectrum | "
        "consistency | rhs | regularization ratio |",
        "|---|---|---:|---:|---|---|---|---|---|---:|",
    ]
    for case in sorted({row["case"] for row in rows}):
        row = next(selected for selected in rows if selected["case"] == case)
        m = integer(row, "m")
        n = integer(row, "n")
        rank = integer(row, "rank")
        shape = "tall" if m > n else "wide" if m < n else "square"
        if is_precision_status_case(row):
            rank_status = f"full; numerical rank {rank - 1} status case"
        else:
            rank_status = "full" if rank == min(m, n) else "exact deficient"
        consistency = (
            "inconsistent"
            if row["rhs"] in {"inconsistent", "noisy_rhs"}
            else "consistent"
        )
        lines.append(
            f"| {case} | {shape} | {m} x {n} | {rank} | {rank_status} | "
            f"{row['coherence']} | {row['spectrum']} | {consistency} | "
            f"{row['rhs']} | {format_number(finite_float(row, 'regularization_ratio'))} |"
        )
    return lines


def case_table(rows: list[dict[str, str]]) -> list[str]:
    methods = method_order({row["method"] for row in rows})
    lines = [
        "| case | method | formal pass | precision status correct | "
        "terminal outcomes | median attempt wall (s) | wall IQR across seeds (s) | "
        "attempt vectors | "
        "attempt block calls | accepted accuracy | solution error | null component | "
        "width | depth |",
        "|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for case in sorted({row["case"] for row in rows}):
        for method in methods:
            selected = [
                row
                for row in rows
                if row["case"] == case and row["method"] == method
            ]
            accepted = [row for row in selected if passed(row)]
            has_operator_counts = method != "aplicur"
            vectors = (
                median_field(selected, "total_vector_work")
                if has_operator_counts
                else None
            )
            block_calls = (
                median_field(selected, "total_block_calls")
                if has_operator_counts
                else None
            )
            width = (
                median_field(selected, "auxiliary_width")
                if has_operator_counts
                else None
            )
            depth = (
                median_field(selected, "depth") if has_operator_counts else None
            )
            lines.append(
                f"| {case} | {method} | {len(accepted)}/{len(selected)} | "
                f"{precision_status_summary(selected)} | "
                f"{outcome_summary(selected)} | "
                f"{format_number(median_field(selected, 'wall_seconds'))} | "
                f"{format_number(iqr_field(selected, 'wall_seconds'))} | "
                f"{format_number(vectors)} | "
                f"{format_number(block_calls)} | "
                f"{format_number(median_primary_accuracy(accepted))} | "
                f"{format_number(median_field(accepted, 'relative_solution_error'))} | "
                f"{format_number(median_field(accepted, 'relative_null_component'))} | "
                f"{format_number(width)} | "
                f"{format_number(depth)} |"
            )
    return lines


def render(
    rows: list[dict[str, str]],
    inputs: list[Path],
    config_paths: list[Path],
) -> str:
    configs = load_configs(config_paths)
    lines = [
        "# Final large-matrix study",
        "",
        "Inputs: " + ", ".join(path.name for path in inputs) + ".",
        "",
    ]
    for suite in sorted({row["suite"] for row in rows}):
        if suite not in configs:
            raise ValueError(f"no formal config supplied for suite: {suite}")
        suite_rows = [row for row in rows if row["suite"] == suite]
        lines.extend([f"## {suite}", "", "### Frozen experiment protocol", ""])
        lines.extend(protocol_table(configs[suite]))
        lines.extend(["", "### Formal case coverage", ""])
        lines.extend(case_coverage_table(suite_rows))
        lines.extend(["", "### Reliability and accuracy", ""])
        lines.extend(reliability_table(suite_rows))
        lines.extend(["", "### Case-complete AMFLS comparisons", ""])
        lines.extend(comparison_table(suite_rows))
        lines.extend(["", "### Per-case AMFLS comparisons", ""])
        lines.extend(per_case_comparison_table(suite_rows))
        lines.extend(["", "### Per-case results", ""])
        lines.extend(case_table(suite_rows))
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    rows = load_rows(args.input)
    validate_grid(rows, args.config)
    report = render(rows, args.input, args.config)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report, encoding="utf-8")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
