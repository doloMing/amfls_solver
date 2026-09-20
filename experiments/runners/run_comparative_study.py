from __future__ import annotations

import argparse
import ctypes
import csv
import gc
import json
import math
import os
from pathlib import Path
import random
import sys
import threading
import time

import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))

from amfls import (  # noqa: E402
    load_library,
    run_amfls as run_production_amfls,
    run_aplicur,
    run_fixed_rbgk,
    run_fossils,
    run_lsmb,
    run_lsmr,
    run_lsrn,
    run_lsqr,
    run_randomized_block_cg,
    run_sparse_embedding_lsqr,
    run_spir,
)


FIELDS = [
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
    "regularization",
    "regularization_ratio",
    "tolerance",
    "success",
    "external_success",
    "precision_status_case",
    "precision_status_correct",
    "status",
    "stop_reason",
    "objective",
    "reference_objective",
    "relative_objective_gap",
    "residual_norm",
    "normal_residual_norm",
    "compatible_backward_error_external",
    "least_squares_backward_error_external",
    "backward_error_external",
    "relative_energy_error_external",
    "reported_accuracy_upper",
    "relative_solution_error",
    "relative_null_component",
    "a_columns",
    "at_columns",
    "total_vector_work",
    "a_block_calls",
    "at_block_calls",
    "total_block_calls",
    "search_vector_work",
    "validation_vector_work",
    "search_block_calls",
    "validation_block_calls",
    "sketch_a_columns",
    "sketch_at_columns",
    "sketch_a_block_calls",
    "sketch_at_block_calls",
    "iterative_a_columns",
    "iterative_at_columns",
    "iterative_a_block_calls",
    "iterative_at_block_calls",
    "gaussian_random_columns",
    "gaussian_random_values",
    "gaussian_random_block_requests",
    "iterations",
    "depth",
    "auxiliary_width",
    "basis_rank",
    "kernel_seconds",
    "a_seconds",
    "at_seconds",
    "orthogonalization_seconds",
    "projected_solve_seconds",
    "validation_seconds",
    "other_seconds",
    "wall_seconds",
    "peak_resident_memory_mib",
    "additional_resident_memory_mib",
    "error",
]

INTERNAL_TIMING_FIELDS = (
    "a_seconds",
    "at_seconds",
    "orthogonalization_seconds",
    "projected_solve_seconds",
    "validation_seconds",
    "other_seconds",
)

STATUS_SUCCESS = 0
STATUS_PRECISION_LIMIT = 3
STOP_REASON_PRECISION_LIMIT = 6
PRECISION_STATUS_METHODS = frozenset({"amfls", "fixed_rbgk"})
MEMORY_SAMPLE_SECONDS = 0.002


def resident_memory_mib() -> float:
    with Path("/proc/self/statm").open("r", encoding="utf-8") as handle:
        resident_pages = int(handle.read().split()[1])
    return resident_pages * os.sysconf("SC_PAGE_SIZE") / (1024.0 * 1024.0)


def release_unused_memory() -> None:
    gc.collect()
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except (AttributeError, OSError):
        pass


class ResidentMemorySampler:
    def __init__(self) -> None:
        self._stop = threading.Event()
        self._started = threading.Event()
        self._baseline = 0.0
        self._peak = 0.0
        self._thread = threading.Thread(target=self._sample, daemon=True)

    def _sample(self) -> None:
        self._baseline = resident_memory_mib()
        self._peak = self._baseline
        self._started.set()
        while not self._stop.wait(MEMORY_SAMPLE_SECONDS):
            self._peak = max(self._peak, resident_memory_mib())
        self._peak = max(self._peak, resident_memory_mib())

    def start(self) -> None:
        self._thread.start()
        self._started.wait()

    def stop(self) -> tuple[float, float]:
        self._stop.set()
        self._thread.join()
        return self._peak, max(0.0, self._peak - self._baseline)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one frozen formal dense comparison configuration."
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


def orthonormal_columns(
    rows: int,
    columns: int,
    generator: torch.Generator,
    coherence: str,
) -> torch.Tensor:
    if coherence == "coherent":
        values = torch.zeros((rows, columns), dtype=torch.float64)
        diagonal = torch.arange(columns)
        values[diagonal, diagonal] = 1.0
        return values
    if coherence != "haar":
        raise ValueError(f"unknown coherence: {coherence}")
    values = torch.randn(
        (rows, columns), generator=generator, dtype=torch.float64
    )
    q, r = torch.linalg.qr(values, mode="reduced")
    signs = torch.sign(torch.diagonal(r))
    signs[signs == 0.0] = 1.0
    return q * signs


def singular_values(case: dict[str, object], rank: int) -> torch.Tensor:
    kind = str(case["spectrum"])
    index = torch.arange(rank, dtype=torch.float64)
    if kind == "flat":
        return torch.ones(rank, dtype=torch.float64)
    if kind == "outliers":
        count = min(int(case.get("outlier_count", 1)), rank)
        ratio = float(case.get("outlier_ratio", 1.0e4))
        tail_minimum = float(case.get("tail_minimum", 0.2))
        values = torch.linspace(
            1.0, tail_minimum, rank, dtype=torch.float64
        )
        values[:count] = ratio
        return values
    if kind == "separated_outliers":
        count = min(int(case.get("outlier_count", 1)), rank)
        ratio = float(case.get("outlier_ratio", 1.0e4))
        smallest_ratio = float(case.get("smallest_outlier_ratio", 10.0))
        tail_minimum = float(case.get("tail_minimum", 0.2))
        values = torch.linspace(
            1.0, tail_minimum, rank, dtype=torch.float64
        )
        if count == 1:
            values[0] = ratio
        else:
            values[:count] = torch.logspace(
                math.log10(ratio),
                math.log10(smallest_ratio),
                count,
                dtype=torch.float64,
            )
        return values
    if kind == "clusters":
        values = torch.full((rank,), 0.1, dtype=torch.float64)
        first = min(rank, max(1, rank // 16))
        second = min(rank, max(first + 1, rank // 2))
        values[:first] = 100.0
        values[first:second] = 1.0
        return values
    if kind == "polynomial":
        alpha = float(case.get("alpha", 1.0))
        return (index + 1.0).pow(-alpha)
    if kind == "exponential":
        endpoint = float(case.get("endpoint", 1.0e-8))
        if rank == 1:
            return torch.ones(1, dtype=torch.float64)
        return torch.logspace(
            0.0, math.log10(endpoint), rank, dtype=torch.float64
        )
    if kind == "smooth":
        if rank == 1:
            return torch.ones(1, dtype=torch.float64)
        scaled = index / float(rank - 1)
        return torch.exp(-4.0 * scaled.pow(1.5))
    if kind == "numerical":
        endpoint = float(case["endpoint"])
        if rank == 1:
            return torch.ones(1, dtype=torch.float64)
        return torch.logspace(
            0.0, math.log10(endpoint), rank, dtype=torch.float64
        )
    if kind == "precision_two_mode":
        values = torch.ones(rank, dtype=torch.float64)
        values[-1] = float(case["two_mode_small_singular_value"])
        return values
    raise ValueError(f"unknown spectrum: {kind}")


def make_problem(case: dict[str, object], seed: int) -> dict[str, object]:
    m = int(case["m"])
    n = int(case["n"])
    rank = int(case["rank"])
    if rank <= 0 or rank > min(m, n):
        raise ValueError(f"invalid rank {rank} for shape ({m},{n})")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    coherence = str(case.get("coherence", "haar"))
    left = orthonormal_columns(m, rank, generator, coherence)
    right = orthonormal_columns(n, rank, generator, coherence)
    sigma = singular_values(case, rank)
    if coherence == "coherent":
        matrix = torch.zeros((m, n), dtype=torch.float64)
        diagonal = torch.arange(rank)
        matrix[diagonal, diagonal] = sigma
    else:
        matrix = ((left * sigma) @ right.T).contiguous()

    coefficients = torch.randn(
        rank, generator=generator, dtype=torch.float64
    )
    rhs_kind = str(case["rhs"])
    exact_coordinate_inconsistent = (
        coherence == "coherent"
        and rhs_kind == "inconsistent"
        and rank < min(m, n)
        and float(case.get("regularization_ratio", 0.0)) == 0.0
    )
    if exact_coordinate_inconsistent:
        coefficients = torch.sign(coefficients)
        coefficients[coefficients == 0.0] = 1.0
    if rhs_kind in {"random", "consistent", "inconsistent"}:
        pass
    elif rhs_kind == "top_loaded":
        coefficients[max(1, rank // 8) :] = 0.0
    elif rhs_kind == "tail_loaded":
        coefficients[: max(0, rank - max(1, rank // 8))] = 0.0
    elif rhs_kind == "outlier_orthogonal":
        coefficients[: min(int(case.get("outlier_count", 1)), rank)] = 0.0
    elif rhs_kind == "energy_balanced":
        ratio = float(case.get("regularization_ratio", 0.0))
        regularization = ratio * float(sigma[0]) ** 2
        coefficients = (sigma.square() + regularization).rsqrt()
        signs = torch.sign(
            torch.randn(rank, generator=generator, dtype=torch.float64)
        )
        signs[signs == 0.0] = 1.0
        coefficients *= signs
    else:
        raise ValueError(f"unknown right-hand side: {rhs_kind}")

    signal = left @ (sigma * coefficients)
    b = signal.clone()
    if rhs_kind == "inconsistent":
        if rank >= m:
            raise ValueError("an inconsistent case requires rank < m")
        if exact_coordinate_inconsistent:
            noise = torch.zeros(m, dtype=torch.float64)
            noise[rank:] = torch.randn(
                m - rank, generator=generator, dtype=torch.float64
            )
        else:
            noise = torch.randn(m, generator=generator, dtype=torch.float64)
            noise -= left @ (left.T @ noise)
        noise_norm = float(torch.linalg.vector_norm(noise))
        if noise_norm == 0.0:
            raise RuntimeError("failed to construct an inconsistent component")
        signal_norm = float(torch.linalg.vector_norm(signal))
        b += float(case.get("noise_ratio", 0.05)) * signal_norm * noise / noise_norm

    regularization_ratio = float(case.get("regularization_ratio", 0.0))
    regularization = regularization_ratio * float(sigma[0]) ** 2
    shrinkage = sigma.square() / (sigma.square() + regularization)
    reference = right @ (shrinkage * coefficients)

    b_norm = float(torch.linalg.vector_norm(b))
    if b_norm == 0.0:
        raise ValueError("right-hand side cannot be zero")
    if exact_coordinate_inconsistent:
        _, exponent = math.frexp(b_norm)
        exact_scale = math.ldexp(1.0, -exponent)
        b *= exact_scale
        reference *= exact_scale
    else:
        b /= b_norm
        reference /= b_norm

    residual = matrix @ reference - b
    reference_objective = float(
        torch.dot(residual, residual)
        + regularization * torch.dot(reference, reference)
    )
    return {
        "matrix": matrix,
        "b": b.contiguous(),
        "reference": reference.contiguous(),
        "row_basis": right,
        "operator_norm": float(sigma[0]),
        "regularization": regularization,
        "reference_objective": reference_objective,
    }


def run_method(
    method: str,
    problem: dict[str, object],
    config: dict[str, object],
    algorithm_seed: int,
):
    matrix = problem["matrix"]
    b = problem["b"]
    regularization = float(problem["regularization"])
    tolerance = float(config["tolerance"])
    maximum_iterations = int(config["maximum_iterations"])
    maximum_depth = int(config["maximum_depth"])
    maximum_width = int(config["maximum_auxiliary_width"])
    maximum_basis = int(config["maximum_basis_size"])
    failure_probability = float(config["failure_probability"])

    if method == "fixed_rbgk":
        return run_fixed_rbgk(
            matrix,
            b,
            auxiliary_width=int(config["fixed_rbgk_auxiliary_width"]),
            maximum_depth=maximum_depth,
            maximum_basis_size=maximum_basis,
            regularization=regularization,
            tolerance=tolerance,
            failure_probability=failure_probability,
            seed=algorithm_seed,
        )
    if method == "lsqr":
        return run_lsqr(
            matrix,
            b,
            regularization=regularization,
            tolerance=tolerance,
            maximum_iterations=maximum_iterations,
        )
    if method == "lsmr":
        return run_lsmr(
            matrix,
            b,
            regularization=regularization,
            tolerance=tolerance,
            maximum_iterations=maximum_iterations,
        )
    if method == "lsmb":
        return run_lsmb(
            matrix,
            b,
            regularization=regularization,
            tolerance=tolerance,
            maximum_iterations=maximum_iterations,
        )
    if method == "lsrn":
        return run_lsrn(
            matrix,
            b,
            regularization=regularization,
            tolerance=tolerance,
            oversampling=float(config.get("lsrn_oversampling", 2.0)),
            relative_rank_tolerance=float(
                config.get("lsrn_relative_rank_tolerance", 1.0e-12)
            ),
            maximum_iterations=maximum_iterations,
            sketch_block_size=int(config.get("lsrn_sketch_block_size", 0)),
            seed=algorithm_seed,
        )
    if method == "sparse_embedding_lsqr":
        return run_sparse_embedding_lsqr(
            matrix,
            b,
            regularization=regularization,
            tolerance=tolerance,
            embedding_distortion=float(
                config["sparse_embedding_distortion"]
            ),
            embedding_failure_probability=float(
                config["sparse_embedding_failure_probability"]
            ),
            sketch_rows=int(config["sparse_embedding_sketch_rows"]),
            embedding_nonzeros=int(
                config["sparse_embedding_nonzeros"]
            ),
            relative_rank_tolerance=float(
                config["sparse_embedding_relative_rank_tolerance"]
            ),
            absolute_rank_tolerance=float(
                config["sparse_embedding_absolute_rank_tolerance"]
            ),
            maximum_iterations=maximum_iterations,
            sketch_block_size=int(
                config["sparse_embedding_sketch_block_size"]
            ),
            seed=algorithm_seed,
        )
    if method == "spir":
        return run_spir(
            matrix,
            b,
            regularization=regularization,
            tolerance=tolerance,
            sketch_rows=int(config.get("spir_sketch_rows", 0)),
            embedding_nonzeros=int(
                config.get("spir_embedding_nonzeros", 8)
            ),
            maximum_inner_iterations=int(
                config.get("spir_maximum_inner_iterations", 50)
            ),
            sketch_block_size=int(config.get("spir_sketch_block_size", 32)),
            seed=algorithm_seed,
        )
    if method == "fossils":
        return run_fossils(
            matrix,
            b,
            regularization=regularization,
            tolerance=tolerance,
            sketch_rows=int(config.get("fossils_sketch_rows", 0)),
            embedding_nonzeros=int(
                config.get("fossils_embedding_nonzeros", 8)
            ),
            maximum_inner_iterations=int(
                config.get("fossils_maximum_inner_iterations", 100)
            ),
            distortion_safety=float(
                config.get("fossils_distortion_safety", 1.0)
            ),
            sketch_block_size=int(
                config.get("fossils_sketch_block_size", 32)
            ),
            seed=algorithm_seed,
        )
    if method == "aplicur":
        return run_aplicur(
            matrix,
            b,
            regularization=regularization,
            tolerance=tolerance,
            block_size=int(config.get("aplicur_block_size", 8)),
            sparse_sign_nonzeros=int(
                config.get("aplicur_sparse_sign_nonzeros", 8)
            ),
            spectral_probe_count=int(
                config.get("aplicur_spectral_probe_count", 10)
            ),
            cur_tolerance=float(config.get("aplicur_cur_tolerance", 0.0)),
            re_preconditioning_tolerance=float(
                config.get("aplicur_re_preconditioning_tolerance", 10.0)
            ),
            dynamic_stopping_tolerance=float(
                config.get("aplicur_dynamic_stopping_tolerance", 150.0)
            ),
            maximum_iterations=maximum_iterations,
            relative_rank_tolerance=float(
                config.get("aplicur_relative_rank_tolerance", 1.0e-12)
            ),
            seed=algorithm_seed,
        )
    if method == "rbcg":
        return run_randomized_block_cg(
            matrix,
            b,
            regularization=regularization,
            tolerance=tolerance,
            random_block_size=int(config["rbcg_block_size"]),
            maximum_depth=maximum_depth,
            seed=algorithm_seed,
        )
    if method == "amfls":
        return run_production_amfls(
            matrix,
            b,
            regularization=regularization,
            tolerance=tolerance,
            failure_probability=failure_probability,
            maximum_epochs=int(config["maximum_epochs"]),
            maximum_depth=maximum_depth,
            maximum_auxiliary_width=maximum_width,
            maximum_basis_size=maximum_basis,
            seed=algorithm_seed,
        )
    raise ValueError(f"unknown method: {method}")


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
    matrix = problem["matrix"]
    b = problem["b"]
    reference = problem["reference"]
    row_basis = problem["row_basis"]
    regularization = float(problem["regularization"])
    operator_norm = float(problem["operator_norm"])
    solution = result.solution

    residual = matrix @ solution - b
    normal = matrix.T @ residual + regularization * solution
    residual_norm = float(torch.linalg.vector_norm(residual))
    normal_norm = float(torch.linalg.vector_norm(normal))
    solution_norm = float(torch.linalg.vector_norm(solution))
    b_norm = float(torch.linalg.vector_norm(b))
    augmented_residual = math.sqrt(
        residual_norm * residual_norm
        + regularization * solution_norm * solution_norm
    )
    augmented_norm = math.sqrt(operator_norm * operator_norm + regularization)
    compatible = ratio(
        augmented_residual, augmented_norm * solution_norm + b_norm
    )
    least_squares = ratio(
        normal_norm, augmented_norm * augmented_residual
    )
    backward = min(compatible, least_squares)

    error = solution - reference
    error_norm = float(torch.linalg.vector_norm(error))
    reference_norm = float(torch.linalg.vector_norm(reference))
    error_a_norm = float(torch.linalg.vector_norm(matrix @ error))
    reference_a_norm = float(torch.linalg.vector_norm(matrix @ reference))
    energy_error = math.sqrt(
        error_a_norm * error_a_norm + regularization * error_norm * error_norm
    )
    reference_energy = math.sqrt(
        reference_a_norm * reference_a_norm
        + regularization * reference_norm * reference_norm
    )
    relative_energy = ratio(energy_error, reference_energy)
    primary = relative_energy if regularization > 0.0 else backward
    slack = 256.0 * torch.finfo(torch.float64).eps + 1.0e-7 * tolerance

    objective = float(
        torch.dot(residual, residual)
        + regularization * torch.dot(solution, solution)
    )
    reference_objective = float(problem["reference_objective"])
    objective_scale = max(reference_objective, torch.finfo(torch.float64).tiny)
    if regularization > 0.0 or row_basis.shape[1] >= matrix.shape[1]:
        relative_null_component = 0.0
        minimum_norm_satisfied = True
    else:
        null_component = solution - row_basis @ (row_basis.T @ solution)
        relative_null_component = ratio(
            float(torch.linalg.vector_norm(null_component)), reference_norm
        )
        minimum_norm_satisfied = relative_null_component <= tolerance + slack
    return {
        "external_success": int(
            primary <= tolerance + slack and minimum_norm_satisfied
        ),
        "objective": objective,
        "relative_objective_gap": (objective - reference_objective) / objective_scale,
        "residual_norm": residual_norm,
        "normal_residual_norm": normal_norm,
        "compatible_backward_error_external": compatible,
        "least_squares_backward_error_external": least_squares,
        "backward_error_external": backward,
        "relative_energy_error_external": relative_energy,
        "relative_solution_error": ratio(error_norm, reference_norm),
        "relative_null_component": relative_null_component,
    }


def evaluate_precision_status_correctness(
    case: dict[str, object],
    problem: dict[str, object],
    result,
    values: dict[str, float | int],
    tolerance: float,
) -> int:
    if not bool(case.get("precision_status_case", False)):
        return 0
    if float(problem["regularization"]) != 0.0:
        return 0

    status = int(result.status)
    internal_upper_bound = float(result.backward_error_upper_bound)
    external_success = int(values["external_success"])
    if status == STATUS_SUCCESS:
        return int(
            math.isfinite(internal_upper_bound)
            and internal_upper_bound <= tolerance
            and external_success == 1
        )
    return int(
        status == STATUS_PRECISION_LIMIT
        and int(result.stop_reason) == STOP_REASON_PRECISION_LIMIT
        and math.isfinite(internal_upper_bound)
        and internal_upper_bound > tolerance
    )


def precision_status_is_assessed(
    case: dict[str, object], method: str
) -> bool:
    return (
        bool(case.get("precision_status_case", False))
        and method in PRECISION_STATUS_METHODS
    )


def empty_row(
    suite: str,
    case: dict[str, object],
    instance_seed: int,
    algorithm_seed: int,
    method: str,
    regularization: float,
    tolerance: float,
) -> dict[str, object]:
    row = {name: "" for name in FIELDS}
    row.update(
        {
            "suite": suite,
            "case": case["name"],
            "instance_seed": instance_seed,
            "algorithm_seed": algorithm_seed,
            "method": method,
            "m": case["m"],
            "n": case["n"],
            "rank": case["rank"],
            "coherence": case.get("coherence", "haar"),
            "spectrum": case["spectrum"],
            "rhs": case["rhs"],
            "regularization": regularization,
            "regularization_ratio": case.get("regularization_ratio", 0.0),
            "tolerance": tolerance,
            "precision_status_case": int(
                bool(case.get("precision_status_case", False))
            ),
            "precision_status_correct": (
                0 if precision_status_is_assessed(case, method) else ""
            ),
        }
    )
    return row


def measured_row(
    suite: str,
    case: dict[str, object],
    instance_seed: int,
    algorithm_seed: int,
    method: str,
    problem: dict[str, object],
    result,
    wall_seconds: float,
    peak_resident_memory_mib: float,
    additional_resident_memory_mib: float,
    tolerance: float,
) -> dict[str, object]:
    values = diagnostics(problem, result, tolerance)
    regularization = float(problem["regularization"])
    row = empty_row(
        suite,
        case,
        instance_seed,
        algorithm_seed,
        method,
        regularization,
        tolerance,
    )
    reported_accuracy = (
        result.relative_energy_error_upper_bound
        if regularization > 0.0
        else result.backward_error_upper_bound
    )
    row.update(
        {
            "success": int(result.success),
            "external_success": values["external_success"],
            "precision_status_correct": (
                evaluate_precision_status_correctness(
                    case, problem, result, values, tolerance
                )
                if precision_status_is_assessed(case, method)
                else ""
            ),
            "status": result.status,
            "stop_reason": result.stop_reason,
            "objective": values["objective"],
            "reference_objective": problem["reference_objective"],
            "relative_objective_gap": values["relative_objective_gap"],
            "residual_norm": values["residual_norm"],
            "normal_residual_norm": values["normal_residual_norm"],
            "compatible_backward_error_external": values[
                "compatible_backward_error_external"
            ],
            "least_squares_backward_error_external": values[
                "least_squares_backward_error_external"
            ],
            "backward_error_external": values["backward_error_external"],
            "relative_energy_error_external": values[
                "relative_energy_error_external"
            ],
            "reported_accuracy_upper": reported_accuracy,
            "relative_solution_error": values["relative_solution_error"],
            "relative_null_component": values["relative_null_component"],
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
            "kernel_seconds": result.total_seconds,
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


def validate_config(config: dict[str, object]) -> None:
    required_positive = (
        "tolerance",
        "maximum_iterations",
        "maximum_depth",
        "maximum_auxiliary_width",
        "maximum_basis_size",
        "maximum_epochs",
        "timing_repetitions",
        "threads",
    )
    for name in required_positive:
        if float(config[name]) <= 0.0:
            raise ValueError(f"{name} must be positive")
    probability = float(config["failure_probability"])
    if not 0.0 < probability < 1.0:
        raise ValueError("failure_probability must lie between zero and one")
    methods = [str(value) for value in config["methods"]]
    seeds = [int(value) for value in config["instance_seeds"]]
    if not methods or len(methods) != len(set(methods)):
        raise ValueError("methods must be nonempty and distinct")
    if not seeds or len(seeds) != len(set(seeds)):
        raise ValueError("instance_seeds must be nonempty and distinct")
    cases = config["cases"]
    precision_status_case_count = 0
    allowed_spectra = {
        "flat",
        "outliers",
        "separated_outliers",
        "clusters",
        "polynomial",
        "exponential",
        "smooth",
        "numerical",
        "precision_two_mode",
    }
    allowed_right_hand_sides = {
        "random",
        "consistent",
        "inconsistent",
        "top_loaded",
        "tail_loaded",
        "outlier_orthogonal",
        "energy_balanced",
    }
    for case in cases:
        m = int(case["m"])
        n = int(case["n"])
        rank = int(case["rank"])
        spectrum = str(case["spectrum"])
        rhs = str(case["rhs"])
        coherence = str(case.get("coherence", "haar"))
        regularization_ratio = float(case.get("regularization_ratio", 0.0))
        raw_precision_status_case = case.get("precision_status_case", False)
        if not isinstance(raw_precision_status_case, bool):
            raise ValueError(
                f"precision_status_case must be Boolean in {case['name']}"
            )
        precision_status_case = raw_precision_status_case
        precision_status_case_count += int(precision_status_case)
        if m <= 0 or n <= 0 or rank <= 0 or rank > min(m, n):
            raise ValueError(
                f"invalid dimensions or rank in formal case {case['name']}"
            )
        if spectrum not in allowed_spectra:
            raise ValueError(f"unknown spectrum in formal case {case['name']}")
        if rhs not in allowed_right_hand_sides:
            raise ValueError(
                f"unknown right-hand side in formal case {case['name']}"
            )
        if coherence not in {"haar", "coherent"}:
            raise ValueError(f"unknown coherence in formal case {case['name']}")
        if regularization_ratio < 0.0 or not math.isfinite(
            regularization_ratio
        ):
            raise ValueError(
                f"invalid regularization ratio in formal case {case['name']}"
            )
        case_tolerance = float(case.get("tolerance", config["tolerance"]))
        if not 0.0 < case_tolerance < 1.0 or not math.isfinite(
            case_tolerance
        ):
            raise ValueError(
                f"invalid tolerance in formal case {case['name']}"
            )
        case_seeds = [
            int(value) for value in case.get("instance_seeds", seeds)
        ]
        if not case_seeds or len(case_seeds) != len(set(case_seeds)):
            raise ValueError(
                f"invalid instance seeds in formal case {case['name']}"
            )
        if rhs == "inconsistent" and rank >= m:
            raise ValueError(
                f"inconsistent formal case {case['name']} requires rank < m"
            )
        if spectrum in {"exponential", "numerical"}:
            endpoint = float(case["endpoint"])
            if not 0.0 < endpoint <= 1.0:
                raise ValueError(
                    f"invalid spectrum endpoint in formal case {case['name']}"
                )
        if spectrum == "polynomial" and float(case.get("alpha", 1.0)) <= 0.0:
            raise ValueError(
                f"invalid polynomial exponent in formal case {case['name']}"
            )
        if spectrum == "separated_outliers":
            ratio = float(case.get("outlier_ratio", 0.0))
            smallest_ratio = float(
                case.get("smallest_outlier_ratio", 0.0)
            )
            if not ratio > smallest_ratio > 1.0:
                raise ValueError(
                    "separated outliers require outlier_ratio > "
                    f"smallest_outlier_ratio > 1 in {case['name']}"
                )
        if spectrum == "precision_two_mode":
            small = float(case["two_mode_small_singular_value"])
            scalar_rank_cutoff = (
                64.0
                * torch.finfo(torch.float64).eps
                * max(m, n)
            )
            if (
                not precision_status_case
                or m != n
                or rank != n
                or rank < 2
                or coherence != "coherent"
                or rhs != "energy_balanced"
                or regularization_ratio != 0.0
                or not math.isfinite(small)
                or not case_tolerance < small < scalar_rank_cutoff
                or int(case.get("timing_repetitions", 0)) != 1
                or len(case_seeds) != 1
            ):
                raise ValueError(
                    "precision_two_mode must be one coherent, ordinary, "
                    "full-rank square status case with two positive FP64 "
                    "modes, one explicit seed, one timing repetition, and "
                    f"tolerance < small mode < scalar rank cutoff: {case['name']}"
                )
        elif precision_status_case:
            raise ValueError(
                f"precision status case {case['name']} must use "
                "precision_two_mode"
            )
    if precision_status_case_count > 1:
        raise ValueError("formal studies accept at most one precision status case")
    if cases:
        largest_problem_dimension = max(
            min(int(case["m"]), int(case["n"])) for case in cases
        )
        for limit in (
            "maximum_iterations",
            "maximum_depth",
            "maximum_basis_size",
        ):
            if int(config[limit]) < largest_problem_dimension:
                raise ValueError(
                    f"{limit} must not truncate a formal problem dimension"
                )
    if not methods or len(methods) != len(set(methods)):
        raise ValueError("methods must be nonempty and distinct")
    allowed_methods = {
        "amfls",
        "fixed_rbgk",
        "lsqr",
        "lsmr",
        "lsmb",
        "lsrn",
        "sparse_embedding_lsqr",
        "spir",
        "fossils",
        "aplicur",
        "rbcg",
    }
    unknown_methods = sorted(set(methods) - allowed_methods)
    if unknown_methods:
        raise ValueError(
            "formal studies accept only canonical methods; unknown methods: "
            + ", ".join(unknown_methods)
        )
    if methods.count("amfls") != 1:
        raise ValueError("formal studies require exactly one amfls method")
    if "fixed_rbgk" in methods and int(
        config.get("fixed_rbgk_auxiliary_width", 0)
    ) != 8:
        raise ValueError(
            "fixed_rbgk_auxiliary_width must equal 8 in formal studies"
        )
    if ({"spir", "fossils"} & set(methods)) and any(
        int(case["m"]) < int(case["n"]) for case in cases
    ):
        raise ValueError("SPIR and FOSSILS formal cases must satisfy m >= n")
    if "sparse_embedding_lsqr" in methods:
        distortion = float(config.get("sparse_embedding_distortion", 0.0))
        embedding_probability = float(
            config.get("sparse_embedding_failure_probability", 0.0)
        )
        sketch_rows = int(config.get("sparse_embedding_sketch_rows", 0))
        embedding_nonzeros = int(
            config.get("sparse_embedding_nonzeros", 0)
        )
        sketch_block_size = int(
            config.get("sparse_embedding_sketch_block_size", 0)
        )
        if not 0.0 < distortion < 1.0:
            raise ValueError(
                "sparse_embedding_distortion must lie between zero and one"
            )
        if not 0.0 < embedding_probability < 1.0:
            raise ValueError(
                "sparse_embedding_failure_probability must lie between zero and one"
            )
        if (
            sketch_rows <= 0
            or embedding_nonzeros <= 0
            or embedding_nonzeros > sketch_rows
            or sketch_block_size <= 0
        ):
            raise ValueError(
                "sparse-embedding rows, density, and block size are invalid"
            )
        if any(
            float(case.get("regularization_ratio", 0.0)) != 0.0
            or int(case["m"]) <= int(case["n"])
            or int(case["rank"]) != int(case["n"])
            or sketch_rows < int(case["n"])
            for case in cases
        ):
            raise ValueError(
                "sparse_embedding_lsqr requires ordinary, strictly tall, "
                "full-column-rank cases and at least n sketch rows"
            )
    if "rbcg" in methods:
        if int(config.get("rbcg_block_size", 0)) <= 0:
            raise ValueError(
                "rbcg_block_size must be positive when rbcg is selected"
            )
        if any(
            float(case.get("regularization_ratio", 0.0)) <= 0.0
            for case in cases
        ):
            raise ValueError(
                "rbcg requires strictly positive regularization_ratio "
                "for every formal case"
            )
    if not seeds or len(seeds) != len(set(seeds)):
        raise ValueError("instance seeds must be nonempty and distinct")
    names = [str(case["name"]) for case in cases]
    if not names or len(names) != len(set(names)):
        raise ValueError("case names must be nonempty and distinct")


def method_seed(base_seed: int, method: str) -> int:
    if method == "fixed_rbgk":
        return base_seed + 1_000_003
    if method == "lsrn":
        return base_seed + 2_000_003
    if method == "sparse_embedding_lsqr":
        return base_seed + 3_000_007
    if method == "spir":
        return base_seed + 4_000_009
    if method == "fossils":
        return base_seed + 5_000_011
    if method == "aplicur":
        return base_seed + 6_000_013
    if method == "rbcg":
        return base_seed + 7_000_027
    return base_seed


def balanced_method_order(
    methods: list[str], seed_index: int, order_seed: int
) -> list[str]:
    count = len(methods)
    if count % 2 != 0:
        unused = "__unused_method__"
        return [
            method
            for method in balanced_method_order(
                [*methods, unused], seed_index, order_seed
            )
            if method != unused
        ]
    labels = methods.copy()
    random.Random(order_seed).shuffle(labels)
    first = [0]
    low = 1
    high = count - 1
    while len(first) < count:
        first.append(low)
        low += 1
        if len(first) < count:
            first.append(high)
            high -= 1
    shift = seed_index % count
    row = [labels[(index + shift) % count] for index in first]
    if (seed_index // count) % 2 == 1:
        row.reverse()
    return row


def median_wall_repetition(
    repetitions: list[tuple[float, object, float, float]],
) -> tuple[float, object, float, float]:
    if not repetitions:
        raise ValueError("at least one successful timing repetition is required")
    ordered = sorted(repetitions, key=lambda repetition: repetition[0])
    return ordered[len(ordered) // 2]


def measure_methods_for_instance(
    *,
    suite: str,
    case: dict[str, object],
    instance_seed: int,
    order_index: int,
    methods: list[str],
    problem: dict[str, object],
    config: dict[str, object],
    tolerance: float,
    base_algorithm_seed: int,
) -> list[dict[str, object]]:
    repetitions = int(config.get("timing_repetitions", 1))
    if repetitions <= 0:
        raise ValueError("timing_repetitions must be positive")
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
        order = balanced_method_order(
            methods,
            order_index * repetitions + repetition,
            int(config["order_seed"]),
        )
        for method in order:
            measurement = measurements[method]
            if measurement["error"] is not None:
                continue
            algorithm_seed = method_seed(base_algorithm_seed, method)
            release_unused_memory()
            memory = ResidentMemorySampler()
            memory.start()
            start = time.perf_counter()
            try:
                result = run_method(method, problem, config, algorithm_seed)
                wall_seconds = time.perf_counter() - start
                peak_memory, additional_memory = memory.stop()
                measurement["repetitions"].append(
                    (wall_seconds, result, peak_memory, additional_memory)
                )
                print(
                    f"finished {case['name']} seed={instance_seed} "
                    f"repetition={repetition + 1}/{repetitions} "
                    f"method={method} wall={wall_seconds:.6f}s "
                    f"success={int(result.success)}",
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
                    f"repetition={repetition + 1}/{repetitions} "
                    f"method={method} wall={wall_seconds:.6f}s "
                    f"error={measurement['error']}",
                    flush=True,
                )

    rows: list[dict[str, object]] = []
    for method in methods:
        measurement = measurements[method]
        algorithm_seed = method_seed(base_algorithm_seed, method)
        if measurement["error"] is None:
            wall_seconds, result, peak_memory, additional_memory = median_wall_repetition(
                measurement["repetitions"]
            )
            row = measured_row(
                suite,
                case,
                instance_seed,
                algorithm_seed,
                method,
                problem,
                result,
                wall_seconds,
                peak_memory,
                additional_memory,
                tolerance,
            )
        else:
            row = empty_row(
                suite,
                case,
                instance_seed,
                algorithm_seed,
                method,
                float(problem["regularization"]),
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
    threads = int(config["threads"])
    torch.set_num_threads(threads)
    os.environ["OMP_NUM_THREADS"] = str(threads)
    os.environ["OPENBLAS_NUM_THREADS"] = str(threads)
    load_library(args.library)

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

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8", newline="") as handle:
        csv.DictWriter(handle, fieldnames=FIELDS).writeheader()

    if int(config.get("warmup_runs", 0)) > 0:
        warm_case = all_cases[0]
        warm_problem = make_problem(warm_case, seeds[0])
        warm_config = dict(config)
        warm_config["tolerance"] = float(
            warm_case.get("tolerance", tolerance)
        )
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
                    run_method(
                        method,
                        warm_problem,
                        warm_config,
                        method_seed(warm_seed, method),
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

    rows: list[dict[str, object]] = []
    for case_index, case in selected_cases:
        case_tolerance = float(case.get("tolerance", tolerance))
        case_config = dict(config)
        case_config["tolerance"] = case_tolerance
        case_config["timing_repetitions"] = int(
            case.get("timing_repetitions", config["timing_repetitions"])
        )
        configured_case_seeds = [
            int(value)
            for value in case.get("instance_seeds", configured_seeds)
        ]
        case_seeds = (
            configured_case_seeds
            if args.selected_seeds is None
            else [seed for seed in seeds if seed in configured_case_seeds]
        )
        if not case_seeds:
            raise ValueError(
                f"no selected seed is configured for case {case['name']}"
            )
        for seed_index, instance_seed in enumerate(case_seeds):
            problem = make_problem(case, instance_seed)
            base_algorithm_seed = instance_seed + algorithm_seed_offset
            instance_rows = measure_methods_for_instance(
                suite=suite,
                case=case,
                instance_seed=instance_seed,
                order_index=seed_index + case_index * len(seeds),
                methods=methods,
                problem=problem,
                config=case_config,
                tolerance=case_tolerance,
                base_algorithm_seed=base_algorithm_seed,
            )
            for row in instance_rows:
                rows.append(row)
                print(
                    f"{case['name']} seed={instance_seed} {row['method']} "
                    f"success={row['success']} vectors={row['total_vector_work']} "
                    f"blocks={row['total_block_calls']}",
                    flush=True,
                )
            with args.output.open("a", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=FIELDS)
                writer.writerows(instance_rows)
                handle.flush()

    expected = sum(
        len(case.get("instance_seeds", configured_seeds))
        if args.selected_seeds is None
        else sum(
            seed
            in {
                int(value)
                for value in case.get("instance_seeds", configured_seeds)
            }
            for seed in seeds
        )
        for _, case in selected_cases
    ) * len(methods)
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
