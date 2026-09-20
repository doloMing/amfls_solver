from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import os
from pathlib import Path
import sys
import time

import numpy as np
import scipy.io
import scipy.sparse
import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import run_comparative_study as comparative  # noqa: E402


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

APPLICATION_FIELDS = [
    *comparative.FIELDS[: comparative.FIELDS.index("rank")],
    "nnz",
    *comparative.FIELDS[comparative.FIELDS.index("rank") :],
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the frozen large sparse application comparison."
    )
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--case",
        dest="case_name",
        help="Run one named case from the registered formal configuration.",
    )
    parser.add_argument(
        "--method",
        dest="selected_methods",
        action="append",
        help="Run one configured method. Repeat to select multiple methods.",
    )
    parser.add_argument(
        "--seed",
        dest="selected_seeds",
        action="append",
        type=int,
        help="Run one configured instance seed. Repeat to select multiple seeds.",
    )
    return parser.parse_args()


def local_input_path(config_path: Path, value: object) -> Path:
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = config_path.parent / path
    return path.resolve()


def read_matrix_market(path: Path) -> scipy.sparse.csr_matrix:
    if not path.is_file():
        raise FileNotFoundError(path)
    if path.suffix == ".gz":
        with gzip.open(path, "rb") as handle:
            value = scipy.io.mmread(handle, spmatrix=True)
    else:
        with path.open("rb") as handle:
            value = scipy.io.mmread(handle, spmatrix=True)
    if scipy.sparse.issparse(value):
        if np.iscomplexobj(value.data):
            raise ValueError(f"complex application matrix is not supported: {path}")
        matrix = value.astype(np.float64).tocsr()
    else:
        dense = np.asarray(value)
        if np.iscomplexobj(dense):
            raise ValueError(f"complex application matrix is not supported: {path}")
        matrix = scipy.sparse.csr_matrix(np.asarray(dense, dtype=np.float64))
    matrix.sum_duplicates()
    matrix.eliminate_zeros()
    matrix.sort_indices()
    if matrix.ndim != 2 or min(matrix.shape) <= 0:
        raise ValueError(
            f"application matrix must be nonempty and two-dimensional: {path}"
        )
    if matrix.nnz == 0 or not np.isfinite(matrix.data).all():
        raise ValueError(f"application matrix must contain finite nonzeros: {path}")
    return matrix


def validate_leading_negative_identity(
    matrix: scipy.sparse.csr_matrix,
    path: Path,
) -> None:
    dimension = matrix.shape[0]
    if matrix.shape[1] < dimension:
        raise ValueError(
            f"formal rail matrix must have at least {dimension} columns: {path}"
        )
    leading = matrix[:, :dimension].tocsr()
    leading.sum_duplicates()
    leading.eliminate_zeros()
    leading.sort_indices()
    expected_offsets = np.arange(dimension + 1, dtype=leading.indptr.dtype)
    expected_indices = np.arange(dimension, dtype=leading.indices.dtype)
    if (
        leading.nnz != dimension
        or not np.array_equal(leading.indptr, expected_offsets)
        or not np.array_equal(leading.indices, expected_indices)
        or not np.array_equal(leading.data, -np.ones(dimension, dtype=np.float64))
    ):
        raise ValueError(
            f"formal rail matrix does not have an exact leading -I block after "
            f"duplicate merging and zero removal: {path}"
        )


def read_rhs(path: Path, rows: int) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(path)
    if path.suffix == ".gz":
        with gzip.open(path, "rb") as handle:
            value = scipy.io.mmread(handle, spmatrix=True)
    else:
        with path.open("rb") as handle:
            value = scipy.io.mmread(handle, spmatrix=True)
    if scipy.sparse.issparse(value):
        if np.iscomplexobj(value.data):
            raise ValueError(f"complex right-hand side is not supported: {path}")
        array = value.toarray()
    else:
        array = np.asarray(value)
    if np.iscomplexobj(array):
        raise ValueError(f"complex right-hand side is not supported: {path}")
    if array.shape not in {(rows,), (rows, 1), (1, rows)}:
        raise ValueError(f"right-hand side must be a vector of length {rows}: {path}")
    vector = np.asarray(array, dtype=np.float64).reshape(-1)
    if vector.shape != (rows,) or not np.isfinite(vector).all():
        raise ValueError(f"right-hand side must contain {rows} finite values: {path}")
    norm = float(np.linalg.norm(vector))
    if norm == 0.0:
        raise ValueError(f"right-hand side cannot be zero: {path}")
    return np.ascontiguousarray(vector / norm)


def torch_csr(matrix: scipy.sparse.csr_matrix) -> torch.Tensor:
    row_offsets = np.asarray(matrix.indptr, dtype=np.int64)
    column_indices = np.asarray(matrix.indices, dtype=np.int64)
    values = np.asarray(matrix.data, dtype=np.float64)
    return torch.sparse_csr_tensor(
        torch.from_numpy(row_offsets),
        torch.from_numpy(column_indices),
        torch.from_numpy(values),
        size=matrix.shape,
        dtype=torch.float64,
        device="cpu",
    )


def operator_norm_lower_estimate(
    matrix: scipy.sparse.csr_matrix,
    iterations: int,
    seed: int,
) -> float:
    generator = np.random.default_rng(seed)
    direction = generator.standard_normal(matrix.shape[1])
    direction /= np.linalg.norm(direction)
    for _ in range(iterations):
        image = np.asarray(matrix @ direction).reshape(-1)
        adjoint_image = np.asarray(matrix.T @ image).reshape(-1)
        norm = float(np.linalg.norm(adjoint_image))
        if not math.isfinite(norm) or norm == 0.0:
            raise ValueError(
                "failed to estimate a positive application operator norm"
            )
        direction = adjoint_image / norm
    estimate = float(np.linalg.norm(np.asarray(matrix @ direction).reshape(-1)))
    if not math.isfinite(estimate) or estimate <= 0.0:
        raise ValueError("failed to estimate a positive application operator norm")
    return estimate


def noisy_rhs(
    matrix: scipy.sparse.csr_matrix,
    seed: int,
    noise_ratio: float,
) -> np.ndarray:
    generator = np.random.default_rng(seed)
    target = generator.standard_normal(matrix.shape[1])
    target_norm = float(np.linalg.norm(target))
    if target_norm == 0.0:
        raise RuntimeError("failed to generate a nonzero application target")
    target /= target_norm
    signal = np.asarray(matrix @ target).reshape(-1)
    signal_norm = float(np.linalg.norm(signal))
    if signal_norm == 0.0:
        raise RuntimeError("generated application target lies in the null space")
    if noise_ratio > 0.0:
        noise = generator.standard_normal(matrix.shape[0])
        noise_norm = float(np.linalg.norm(noise))
        if noise_norm == 0.0:
            raise RuntimeError("failed to generate application noise")
        signal = signal + noise_ratio * signal_norm * noise / noise_norm
    rhs_norm = float(np.linalg.norm(signal))
    if rhs_norm == 0.0 or not np.isfinite(signal).all():
        raise RuntimeError("failed to generate a finite nonzero application rhs")
    return np.ascontiguousarray(signal / rhs_norm)


def validate_config(config: dict[str, object]) -> None:
    if not str(config["suite"]).strip():
        raise ValueError("suite must be nonempty")
    methods = tuple(str(value) for value in config["methods"])
    if methods != FORMAL_METHODS:
        raise ValueError(
            "the sparse application study uses the single frozen main-comparison "
            f"method list {FORMAL_METHODS}"
        )
    if int(config["fixed_rbgk_auxiliary_width"]) != 8:
        raise ValueError("fixed_rbgk_auxiliary_width must equal 8")
    tolerance = float(config["tolerance"])
    if not math.isfinite(tolerance) or not 0.0 < tolerance < 1.0:
        raise ValueError("tolerance must be finite and lie between zero and one")
    for name in (
        "maximum_iterations",
        "maximum_depth",
        "maximum_auxiliary_width",
        "maximum_basis_size",
        "maximum_epochs",
        "threads",
        "timing_repetitions",
        "operator_norm_iterations",
    ):
        value = float(config[name])
        if not math.isfinite(value) or value <= 0.0 or not value.is_integer():
            raise ValueError(f"{name} must be a positive integer")
    probability = float(config["failure_probability"])
    if not math.isfinite(probability) or not 0.0 < probability < 1.0:
        raise ValueError(
            "failure_probability must be finite and lie between zero and one"
        )
    lsrn_oversampling = float(config["lsrn_oversampling"])
    lsrn_rank_tolerance = float(config["lsrn_relative_rank_tolerance"])
    if not math.isfinite(lsrn_oversampling) or lsrn_oversampling <= 1.0:
        raise ValueError("lsrn_oversampling must be finite and greater than one")
    if not math.isfinite(lsrn_rank_tolerance) or lsrn_rank_tolerance < 0.0:
        raise ValueError(
            "lsrn_relative_rank_tolerance must be finite and nonnegative"
        )
    distortion = float(config["sparse_embedding_distortion"])
    embedding_probability = float(
        config["sparse_embedding_failure_probability"]
    )
    if not math.isfinite(distortion) or not 0.0 < distortion < 1.0:
        raise ValueError(
            "sparse_embedding_distortion must be finite and lie between zero and one"
        )
    if (
        not math.isfinite(embedding_probability)
        or not 0.0 < embedding_probability < 1.0
    ):
        raise ValueError(
            "sparse_embedding_failure_probability must be finite and lie between "
            "zero and one"
        )
    for name in (
        "sparse_embedding_relative_rank_tolerance",
        "sparse_embedding_absolute_rank_tolerance",
    ):
        value = float(config[name])
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(f"{name} must be finite and nonnegative")
    for name in (
        "lsrn_sketch_block_size",
        "sparse_embedding_nonzeros",
        "sparse_embedding_sketch_block_size",
        "spir_embedding_nonzeros",
        "spir_maximum_inner_iterations",
        "spir_sketch_block_size",
        "fossils_embedding_nonzeros",
        "fossils_maximum_inner_iterations",
        "fossils_sketch_block_size",
    ):
        value = float(config[name])
        if not math.isfinite(value) or value <= 0.0 or not value.is_integer():
            raise ValueError(f"{name} must be a positive integer")
    fossils_distortion_safety = float(config.get("fossils_distortion_safety", 1.0))
    if not math.isfinite(fossils_distortion_safety) or fossils_distortion_safety <= 0.0:
        raise ValueError("fossils_distortion_safety must be finite and positive")
    warmup_runs = float(config.get("warmup_runs", 0))
    if (
        not math.isfinite(warmup_runs)
        or warmup_runs < 0.0
        or not warmup_runs.is_integer()
    ):
        raise ValueError("warmup_runs must be a nonnegative integer")
    for name in ("algorithm_seed_offset", "order_seed", "operator_norm_seed"):
        value = float(config[name])
        if not math.isfinite(value) or not value.is_integer():
            raise ValueError(f"{name} must be an integer")
    raw_seeds = [float(value) for value in config["instance_seeds"]]
    if any(not math.isfinite(value) or not value.is_integer() for value in raw_seeds):
        raise ValueError("instance_seeds must contain integers")
    seeds = [int(value) for value in raw_seeds]
    if not seeds or len(seeds) != len(set(seeds)):
        raise ValueError("instance_seeds must be nonempty and distinct")
    cases = list(config["cases"])
    names = [str(case["name"]) for case in cases]
    if not names or len(names) != len(set(names)):
        raise ValueError("case names must be nonempty and distinct")
    for case in cases:
        if "instance_seeds" in case:
            raise ValueError(
                f"{case['name']} must use the sparse application's frozen "
                "top-level instance_seeds"
            )
        m = int(case["m"])
        n = int(case["n"])
        rank = int(case["rank"])
        if m <= n or rank != n:
            raise ValueError(
                f"{case['name']} must declare a tall, full-column-rank "
                "transposed application problem"
            )
        if not str(case["matrix_path"]).strip():
            raise ValueError(f"{case['name']} matrix_path must be nonempty")
        if not isinstance(case.get("transpose", False), bool):
            raise ValueError(f"{case['name']} transpose must be Boolean")
        mode = str(case["rhs_mode"])
        if mode not in {"noisy_rhs", "provided"}:
            raise ValueError(f"unsupported rhs_mode for {case['name']}: {mode}")
        if mode == "provided" and "rhs_path" not in case:
            raise ValueError(f"{case['name']} requires rhs_path")
        noise_ratio = float(case.get("noise_ratio", 0.0))
        if not math.isfinite(noise_ratio) or noise_ratio < 0.0:
            raise ValueError(
                f"{case['name']} noise_ratio must be finite and nonnegative"
            )
        for name in (
            "sparse_embedding_sketch_rows",
            "spir_sketch_rows",
            "fossils_sketch_rows",
        ):
            value = float(case[name])
            if not math.isfinite(value) or value <= 0.0 or not value.is_integer():
                raise ValueError(
                    f"{case['name']} {name} must be a positive integer"
                )
        if int(config["sparse_embedding_nonzeros"]) > int(
            case["sparse_embedding_sketch_rows"]
        ):
            raise ValueError(
                f"{case['name']} sparse-embedding density exceeds sketch rows"
            )


def load_case(
    case: dict[str, object],
    config_path: Path,
    config: dict[str, object],
) -> dict[str, object]:
    matrix_path = local_input_path(config_path, case["matrix_path"])
    matrix = read_matrix_market(matrix_path)
    if bool(case.get("transpose", False)):
        validate_leading_negative_identity(matrix, matrix_path)
        matrix = matrix.T.tocsr()
    matrix.sort_indices()
    rows, columns = matrix.shape
    expected_shape = (int(case["m"]), int(case["n"]))
    if (rows, columns) != expected_shape:
        raise ValueError(
            f"{case['name']} loaded shape {(rows, columns)} does not match "
            f"the frozen shape {expected_shape}"
        )
    if int(case["rank"]) != columns:
        raise ValueError(
            f"{case['name']} rank must equal the matrix column count "
            f"{columns}"
        )
    if rows <= columns:
        raise ValueError(
            f"{case['name']} must be tall for the "
            "common ordinary least-squares baseline grid"
        )
    for name in (
        "maximum_iterations",
        "maximum_depth",
        "maximum_basis_size",
    ):
        if int(config[name]) < columns:
            raise ValueError(
                f"{case['name']} {name} must be at least the column count "
                f"{columns}"
            )
    case_config = dict(config)
    for name in (
        "sparse_embedding_sketch_rows",
        "spir_sketch_rows",
        "fossils_sketch_rows",
    ):
        rows_value = int(case[name])
        if rows_value < columns:
            raise ValueError(f"{case['name']} {name} must be at least {columns}")
        case_config[name] = rows_value
    norm_estimate = operator_norm_lower_estimate(
        matrix,
        int(config["operator_norm_iterations"]),
        int(config["operator_norm_seed"]),
    )
    return {
        "matrix_scipy": matrix,
        "matrix": torch_csr(matrix),
        "operator_norm": norm_estimate,
        "m": rows,
        "n": columns,
        "nnz": int(matrix.nnz),
        "matrix_path": matrix_path,
        "config": case_config,
    }


def make_problem(
    loaded: dict[str, object],
    case: dict[str, object],
    config_path: Path,
    instance_seed: int,
) -> dict[str, object]:
    matrix = loaded["matrix_scipy"]
    mode = str(case["rhs_mode"])
    if mode == "provided":
        rhs_path = local_input_path(config_path, case["rhs_path"])
        rhs = read_rhs(rhs_path, matrix.shape[0])
    else:
        rhs = noisy_rhs(
            matrix,
            instance_seed,
            float(case.get("noise_ratio", 0.0)),
        )
    return {
        "matrix": loaded["matrix"],
        "matrix_scipy": matrix,
        "b": torch.from_numpy(rhs),
        "b_numpy": rhs,
        "operator_norm": loaded["operator_norm"],
        "regularization": 0.0,
    }


def ratio(numerator: float, denominator: float) -> float:
    if numerator == 0.0:
        return 0.0
    if denominator == 0.0:
        return math.inf
    return numerator / denominator


def diagnostics(
    problem: dict[str, object],
    result,
    tolerance: float,
) -> dict[str, float | int]:
    matrix = problem["matrix_scipy"]
    rhs = problem["b_numpy"]
    solution = np.asarray(result.solution.detach().cpu().numpy(), dtype=np.float64)
    residual = np.asarray(matrix @ solution).reshape(-1) - rhs
    normal = np.asarray(matrix.T @ residual).reshape(-1)
    residual_norm = float(np.linalg.norm(residual))
    normal_norm = float(np.linalg.norm(normal))
    solution_norm = float(np.linalg.norm(solution))
    rhs_norm = float(np.linalg.norm(rhs))
    operator_norm = float(problem["operator_norm"])
    compatible = ratio(
        residual_norm,
        operator_norm * solution_norm + rhs_norm,
    )
    least_squares = ratio(normal_norm, operator_norm * residual_norm)
    backward = min(compatible, least_squares)
    slack = 256.0 * torch.finfo(torch.float64).eps + 1.0e-7 * tolerance
    return {
        "external_success": int(backward <= tolerance + slack),
        "objective": float(np.dot(residual, residual)),
        "residual_norm": residual_norm,
        "normal_residual_norm": normal_norm,
        "compatible_backward_error_external": compatible,
        "least_squares_backward_error_external": least_squares,
        "backward_error_external": backward,
    }


def empty_row(
    suite: str,
    case: dict[str, object],
    loaded: dict[str, object],
    instance_seed: int,
    algorithm_seed: int,
    method: str,
    tolerance: float,
) -> dict[str, object]:
    row = {name: "" for name in APPLICATION_FIELDS}
    row.update(
        {
            "suite": suite,
            "case": case["name"],
            "instance_seed": instance_seed,
            "algorithm_seed": algorithm_seed,
            "method": method,
            "m": loaded["m"],
            "n": loaded["n"],
            "rank": loaded["n"],
            "nnz": loaded["nnz"],
            "coherence": "application",
            "spectrum": "application",
            "rhs": case["rhs_mode"],
            "regularization": 0.0,
            "regularization_ratio": 0.0,
            "tolerance": tolerance,
            "precision_status_case": 0,
            "precision_status_correct": "",
        }
    )
    return row


def measured_row(
    suite: str,
    case: dict[str, object],
    loaded: dict[str, object],
    instance_seed: int,
    algorithm_seed: int,
    method: str,
    problem: dict[str, object],
    result,
    wall_seconds: float,
    peak_resident_memory_mib: float,
    additional_resident_memory_mib: float,
    kernel_seconds: float,
    tolerance: float,
) -> dict[str, object]:
    values = diagnostics(problem, result, tolerance)
    row = empty_row(
        suite,
        case,
        loaded,
        instance_seed,
        algorithm_seed,
        method,
        tolerance,
    )
    row.update(
        {
            "success": int(result.success),
            "external_success": values["external_success"],
            "status": result.status,
            "stop_reason": result.stop_reason,
            "objective": values["objective"],
            "residual_norm": values["residual_norm"],
            "normal_residual_norm": values["normal_residual_norm"],
            "compatible_backward_error_external": values[
                "compatible_backward_error_external"
            ],
            "least_squares_backward_error_external": values[
                "least_squares_backward_error_external"
            ],
            "backward_error_external": values["backward_error_external"],
            "reported_accuracy_upper": result.backward_error_upper_bound,
            "a_columns": result.a_columns,
            "at_columns": result.at_columns,
            "total_vector_work": result.a_columns + result.at_columns,
            "a_block_calls": result.a_block_calls,
            "at_block_calls": result.at_block_calls,
            "total_block_calls": result.a_block_calls + result.at_block_calls,
            "search_vector_work": result.search_a_columns
            + result.search_at_columns,
            "validation_vector_work": result.validation_a_columns
            + result.validation_at_columns,
            "search_block_calls": result.search_a_block_calls
            + result.search_at_block_calls,
            "validation_block_calls": result.validation_a_block_calls
            + result.validation_at_block_calls,
            "sketch_a_columns": result.sketch_a_columns,
            "sketch_at_columns": result.sketch_at_columns,
            "sketch_a_block_calls": result.sketch_a_block_calls,
            "sketch_at_block_calls": result.sketch_at_block_calls,
            "iterative_a_columns": result.iterative_a_columns,
            "iterative_at_columns": result.iterative_at_columns,
            "iterative_a_block_calls": result.iterative_a_block_calls,
            "iterative_at_block_calls": result.iterative_at_block_calls,
            "gaussian_random_columns": result.gaussian_random_columns,
            "gaussian_random_values": result.gaussian_random_values,
            "gaussian_random_block_requests": (
                result.gaussian_random_block_requests
            ),
            "iterations": result.iterations,
            "depth": result.depth,
            "auxiliary_width": result.auxiliary_width,
            "basis_rank": result.basis_rank,
            "kernel_seconds": kernel_seconds,
            "a_seconds": result.a_seconds,
            "at_seconds": result.at_seconds,
            "orthogonalization_seconds": result.orthogonalization_seconds,
            "projected_solve_seconds": result.projected_solve_seconds,
            "validation_seconds": result.validation_seconds,
            "other_seconds": result.other_seconds,
            "wall_seconds": wall_seconds,
            "peak_resident_memory_mib": peak_resident_memory_mib,
            "additional_resident_memory_mib": additional_resident_memory_mib,
        }
    )
    return row


def measure_instance(
    *,
    suite: str,
    case: dict[str, object],
    loaded: dict[str, object],
    instance_seed: int,
    order_index: int,
    methods: list[str],
    problem: dict[str, object],
    config: dict[str, object],
    tolerance: float,
    base_algorithm_seed: int,
) -> list[dict[str, object]]:
    repetitions = int(config["timing_repetitions"])
    measurements: dict[str, dict[str, object]] = {
        method: {
            "repetitions": [],
            "error": None,
            "error_wall": None,
            "error_peak_memory": None,
            "error_additional_memory": None,
        }
        for method in methods
    }
    for repetition in range(repetitions):
        order = comparative.balanced_method_order(
            methods,
            order_index * repetitions + repetition,
            int(config["order_seed"]),
        )
        for method in order:
            measurement = measurements[method]
            if measurement["error"] is not None:
                continue
            algorithm_seed = comparative.method_seed(base_algorithm_seed, method)
            comparative.release_unused_memory()
            memory = comparative.ResidentMemorySampler()
            memory.start()
            start = time.perf_counter()
            try:
                result = comparative.run_method(
                    method,
                    problem,
                    config,
                    algorithm_seed,
                )
                wall_seconds = time.perf_counter() - start
                peak_memory, additional_memory = memory.stop()
                measurement["repetitions"].append(
                    (wall_seconds, result, peak_memory, additional_memory)
                )
                print(
                    f"finished {case['name']} seed={instance_seed} "
                    f"repetition={repetition + 1}/{repetitions} method={method} "
                    f"wall={wall_seconds:.6f}s success={int(result.success)}",
                    flush=True,
                )
            except Exception as error:
                wall_seconds = time.perf_counter() - start
                peak_memory, additional_memory = memory.stop()
                measurement["error"] = f"{type(error).__name__}: {error}"
                measurement["error_wall"] = wall_seconds
                measurement["error_peak_memory"] = peak_memory
                measurement["error_additional_memory"] = additional_memory
                print(
                    f"finished {case['name']} seed={instance_seed} "
                    f"repetition={repetition + 1}/{repetitions} method={method} "
                    f"wall={wall_seconds:.6f}s error={measurement['error']}",
                    flush=True,
                )

    rows: list[dict[str, object]] = []
    for method in methods:
        measurement = measurements[method]
        algorithm_seed = comparative.method_seed(base_algorithm_seed, method)
        if measurement["error"] is None:
            wall_seconds, result, peak_memory, additional_memory = (
                comparative.median_wall_repetition(
                measurement["repetitions"]
            )
            )
            row = measured_row(
                suite,
                case,
                loaded,
                instance_seed,
                algorithm_seed,
                method,
                problem,
                result,
                wall_seconds,
                peak_memory,
                additional_memory,
                result.total_seconds,
                tolerance,
            )
        else:
            row = empty_row(
                suite,
                case,
                loaded,
                instance_seed,
                algorithm_seed,
                method,
                tolerance,
            )
            row["success"] = 0
            row["external_success"] = 0
            row["wall_seconds"] = measurement["error_wall"]
            row["peak_resident_memory_mib"] = measurement[
                "error_peak_memory"
            ]
            row["additional_resident_memory_mib"] = measurement[
                "error_additional_memory"
            ]
            row["error"] = measurement["error"]
        rows.append(row)
    return rows


def main() -> None:
    args = parse_args()
    if not args.library.is_file():
        raise FileNotFoundError(args.library)
    if not args.config.is_file():
        raise FileNotFoundError(args.config)
    if args.output.exists():
        raise FileExistsError(args.output)

    config = json.loads(args.config.read_text(encoding="utf-8"))
    validate_config(config)
    all_cases = list(config["cases"])
    selected_cases = list(enumerate(all_cases))
    if args.case_name is not None:
        selected_cases = [
            (case_index, case)
            for case_index, case in selected_cases
            if str(case["name"]) == args.case_name
        ]
        if len(selected_cases) != 1:
            raise ValueError(
                f"formal case {args.case_name!r} was not found exactly once"
            )
    for case in all_cases:
        matrix_path = local_input_path(args.config, case["matrix_path"])
        if not matrix_path.is_file():
            raise FileNotFoundError(matrix_path)
        if str(case["rhs_mode"]) == "provided":
            rhs_path = local_input_path(args.config, case["rhs_path"])
            if not rhs_path.is_file():
                raise FileNotFoundError(rhs_path)

    threads = int(config["threads"])
    torch.set_num_threads(threads)
    os.environ["OMP_NUM_THREADS"] = str(threads)
    os.environ["OPENBLAS_NUM_THREADS"] = str(threads)
    comparative.load_library(args.library)

    configured_methods = [str(value) for value in config["methods"]]
    configured_seeds = [int(value) for value in config["instance_seeds"]]
    methods = (
        configured_methods
        if args.selected_methods is None
        else [str(value) for value in args.selected_methods]
    )
    seeds = (
        configured_seeds
        if args.selected_seeds is None
        else [int(value) for value in args.selected_seeds]
    )
    if (
        not methods
        or len(methods) != len(set(methods))
        or any(method not in configured_methods for method in methods)
    ):
        raise ValueError("selected methods must be distinct configured methods")
    if (
        not seeds
        or len(seeds) != len(set(seeds))
        or any(seed not in configured_seeds for seed in seeds)
    ):
        raise ValueError("selected seeds must be distinct configured seeds")
    tolerance = float(config["tolerance"])
    algorithm_seed_offset = int(config["algorithm_seed_offset"])
    suite = str(config["suite"])
    rows: list[dict[str, object]] = []

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8", newline="") as handle:
        csv.DictWriter(handle, fieldnames=APPLICATION_FIELDS).writeheader()

    warm_loaded = None
    if int(config.get("warmup_runs", 0)) > 0:
        warm_case = all_cases[0]
        warm_loaded = load_case(warm_case, args.config, config)
        warm_problem = make_problem(warm_loaded, warm_case, args.config, seeds[0])
        warm_seed = seeds[0] + algorithm_seed_offset
        for warmup_index in range(int(config["warmup_runs"])):
            for method in methods:
                start = time.perf_counter()
                print(
                    f"starting warmup={warmup_index + 1}/"
                    f"{int(config['warmup_runs'])} method={method}",
                    flush=True,
                )
                try:
                    comparative.run_method(
                        method,
                        warm_problem,
                        warm_loaded["config"],
                        comparative.method_seed(warm_seed, method),
                    )
                    print(
                        f"finished warmup={warmup_index + 1}/"
                        f"{int(config['warmup_runs'])} method={method} "
                        f"wall={time.perf_counter() - start:.6f}s",
                        flush=True,
                    )
                except Exception as error:
                    print(
                        f"warmup {method} failed and was ignored: "
                        f"{type(error).__name__}: {error}",
                        file=sys.stderr,
                        flush=True,
                    )
        del warm_problem
        if selected_cases[0][0] != 0:
            warm_loaded = None

    for case_index, case in selected_cases:
        if case_index == 0 and warm_loaded is not None:
            loaded = warm_loaded
        else:
            loaded = load_case(case, args.config, config)
        print(
            f"loaded {case['name']} shape=({loaded['m']},{loaded['n']}) "
            f"nnz={loaded['nnz']}",
            flush=True,
        )
        for seed_index, instance_seed in enumerate(seeds):
            print(
                f"starting {case['name']} seed={instance_seed} "
                f"({seed_index + 1}/{len(seeds)})",
                flush=True,
            )
            problem = make_problem(loaded, case, args.config, instance_seed)
            base_algorithm_seed = instance_seed + algorithm_seed_offset
            instance_rows = measure_instance(
                suite=suite,
                case=case,
                loaded=loaded,
                instance_seed=instance_seed,
                order_index=seed_index + case_index * len(seeds),
                methods=methods,
                problem=problem,
                config=loaded["config"],
                tolerance=tolerance,
                base_algorithm_seed=base_algorithm_seed,
            )
            rows.extend(instance_rows)
            for row in instance_rows:
                print(
                    f"{case['name']} seed={instance_seed} {row['method']} "
                    f"success={row['success']} vectors={row['total_vector_work']} "
                    f"blocks={row['total_block_calls']}",
                    flush=True,
                )
            with args.output.open("a", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=APPLICATION_FIELDS)
                writer.writerows(instance_rows)
                handle.flush()

    expected = len(selected_cases) * len(seeds) * len(methods)
    if len(rows) != expected:
        raise RuntimeError("incomplete result grid")
    observed = {
        (row["case"], row["instance_seed"], row["method"]) for row in rows
    }
    if len(observed) != expected:
        raise RuntimeError("duplicate result rows")

    print(f"completed {len(rows)} rows in {args.output}", flush=True)


if __name__ == "__main__":
    main()
