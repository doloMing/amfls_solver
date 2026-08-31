from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch


@dataclass(frozen=True)
class SolveResult:
    solution: torch.Tensor
    objective: float
    residual_norm: float
    augmented_residual_norm: float
    normal_residual_norm: float
    solution_norm: float
    solution_energy_norm: float
    operator_norm_lower_bound: float
    augmented_operator_norm_lower_bound: float
    compatible_backward_error_upper_bound: float
    least_squares_backward_error_upper_bound: float
    backward_error_upper_bound: float
    relative_normal_residual_upper_bound: float
    energy_error_upper_bound: float
    relative_energy_error_upper_bound: float
    objective_gap_upper_bound: float
    ridge_base_energy_error_upper_bound: float | None
    ridge_corrected_energy_error_upper_bound: float | None
    ridge_correction_gamma: float | None
    ridge_correction_z_h_z: float | None
    ridge_correction_two_abs_z_t_q: float | None
    ridge_correction_q_norm_squared: float | None
    a_columns: int
    at_columns: int
    a_block_calls: int
    at_block_calls: int
    search_a_columns: int
    search_at_columns: int
    validation_a_columns: int
    validation_at_columns: int
    iterations: int
    depth: int
    auxiliary_width: int
    basis_rank: int
    sketch_a_columns: int
    sketch_at_columns: int
    sketch_a_block_calls: int
    sketch_at_block_calls: int
    iterative_a_columns: int
    iterative_at_columns: int
    iterative_a_block_calls: int
    iterative_at_block_calls: int
    gaussian_random_columns: int
    gaussian_random_values: int
    search_a_block_calls: int
    search_at_block_calls: int
    validation_a_block_calls: int
    validation_at_block_calls: int
    base_validation_a_columns: int
    base_validation_at_columns: int
    base_validation_a_block_calls: int
    base_validation_at_block_calls: int
    ridge_correction_a_columns: int
    ridge_correction_at_columns: int
    ridge_correction_a_block_calls: int
    ridge_correction_at_block_calls: int
    gaussian_random_block_requests: int
    ridge_correction_disposition: int
    total_seconds: float
    a_seconds: float
    at_seconds: float
    orthogonalization_seconds: float
    projected_solve_seconds: float
    validation_seconds: float
    other_seconds: float
    base_validation_seconds: float
    ridge_correction_validation_seconds: float
    trace: torch.Tensor
    trace_float: torch.Tensor
    trace_int: torch.Tensor
    status: int
    stop_reason: int

    @property
    def success(self) -> bool:
        return self.status == 0

    @property
    def search_vector_work(self) -> int:
        return self.search_a_columns + self.search_at_columns

def load_library(path: str | Path) -> Path:
    library = Path(path).expanduser().resolve()
    if not library.is_file():
        raise FileNotFoundError(f"AMFLS Torch library does not exist: {library}")
    torch.ops.load_library(str(library))
    return library


def _check_inputs(matrix: torch.Tensor, b: torch.Tensor) -> None:
    if matrix.device.type != "cpu" or b.device.type != "cpu":
        raise ValueError("the current binding supports CPU tensors only")
    if matrix.dtype != torch.float64 or b.dtype != torch.float64:
        raise ValueError("matrix and b must use torch.float64")
    if matrix.layout not in (torch.strided, torch.sparse_csr):
        raise ValueError("matrix must use dense strided or sparse CSR storage")
    if b.layout != torch.strided:
        raise ValueError("b must use dense strided storage")
    if matrix.ndim != 2 or b.ndim != 1 or matrix.shape[0] != b.shape[0]:
        raise ValueError("matrix must have shape (m,n) and b must have shape (m,)")


_UINT64_MODULUS = 1 << 64
_INT64_SIGN_BIT = 1 << 63


def _canonical_uint64_to_int64_transport(value: int, name: str) -> int:
    """Encode an unsigned seed or stream for Torch's signed integer argument."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer in [0, 2**64 - 1]")
    if value < 0 or value >= _UINT64_MODULUS:
        raise ValueError(f"{name} must be an integer in [0, 2**64 - 1]")
    return value if value < _INT64_SIGN_BIT else value - _UINT64_MODULUS


def _require_tensor(
    value: object,
    name: str,
    dtype: torch.dtype,
    shape: tuple[int | None, ...],
) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise RuntimeError(f"AMFLS {name} must be a tensor")
    if value.device.type != "cpu":
        raise RuntimeError(f"AMFLS {name} must be on CPU")
    if value.dtype != dtype:
        raise RuntimeError(
            f"AMFLS {name} must use {dtype}, received {value.dtype}"
        )
    if value.ndim != len(shape):
        raise RuntimeError(
            f"AMFLS {name} must have rank {len(shape)}, received {value.ndim}"
        )
    for axis, expected in enumerate(shape):
        if expected is not None and value.shape[axis] != expected:
            raise RuntimeError(
                f"AMFLS {name} axis {axis} must have length {expected}, "
                f"received {value.shape[axis]}"
            )
    return value


def _require_scalar_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise RuntimeError(f"AMFLS {name} must be an integer scalar")
    return value


def _decode(raw: tuple[object, ...]) -> SolveResult:
    if not isinstance(raw, tuple) or len(raw) != 8:
        raise RuntimeError("AMFLS binding result must be an eight-item tuple")

    solution, metrics, counters, timings, trace_float, trace_int, status, reason = raw
    solution = _require_tensor(solution, "solution", torch.float64, (None,))
    metrics = _require_tensor(metrics, "metrics", torch.float64, (21,))
    counters = _require_tensor(counters, "counters", torch.int64, (36,))
    timings = _require_tensor(timings, "timings", torch.float64, (9,))
    trace_float = _require_tensor(
        trace_float, "trace_float", torch.float64, (None, 29)
    )
    trace_int = _require_tensor(
        trace_int,
        "trace_int",
        torch.int64,
        (trace_float.shape[0], 4),
    )
    if trace_int.shape[0] > 0 and not bool(
        torch.all((trace_int[:, 2] == 0) | (trace_int[:, 2] == 1))
    ):
        raise RuntimeError("AMFLS trace success flags must be zero or one")

    return SolveResult(
        solution=solution,
        objective=float(metrics[0]),
        residual_norm=float(metrics[1]),
        augmented_residual_norm=float(metrics[2]),
        normal_residual_norm=float(metrics[3]),
        solution_norm=float(metrics[4]),
        solution_energy_norm=float(metrics[5]),
        operator_norm_lower_bound=float(metrics[6]),
        augmented_operator_norm_lower_bound=float(metrics[7]),
        compatible_backward_error_upper_bound=float(metrics[8]),
        least_squares_backward_error_upper_bound=float(metrics[9]),
        backward_error_upper_bound=float(metrics[10]),
        relative_normal_residual_upper_bound=float(metrics[11]),
        energy_error_upper_bound=float(metrics[12]),
        relative_energy_error_upper_bound=float(metrics[13]),
        objective_gap_upper_bound=float(metrics[14]),
        ridge_base_energy_error_upper_bound=float(metrics[15]),
        ridge_corrected_energy_error_upper_bound=float(metrics[16]),
        ridge_correction_gamma=float(metrics[17]),
        ridge_correction_z_h_z=float(metrics[18]),
        ridge_correction_two_abs_z_t_q=float(metrics[19]),
        ridge_correction_q_norm_squared=float(metrics[20]),
        a_columns=int(counters[0]),
        at_columns=int(counters[1]),
        a_block_calls=int(counters[2]),
        at_block_calls=int(counters[3]),
        search_a_columns=int(counters[4]),
        search_at_columns=int(counters[5]),
        validation_a_columns=int(counters[6]),
        validation_at_columns=int(counters[7]),
        iterations=int(counters[8]),
        depth=int(counters[9]),
        auxiliary_width=int(counters[10]),
        basis_rank=int(counters[11]),
        sketch_a_columns=int(counters[12]),
        sketch_at_columns=int(counters[13]),
        sketch_a_block_calls=int(counters[14]),
        sketch_at_block_calls=int(counters[15]),
        iterative_a_columns=int(counters[16]),
        iterative_at_columns=int(counters[17]),
        iterative_a_block_calls=int(counters[18]),
        iterative_at_block_calls=int(counters[19]),
        gaussian_random_columns=int(counters[20]),
        gaussian_random_values=int(counters[21]),
        search_a_block_calls=int(counters[22]),
        search_at_block_calls=int(counters[23]),
        validation_a_block_calls=int(counters[24]),
        validation_at_block_calls=int(counters[25]),
        base_validation_a_columns=int(counters[26]),
        base_validation_at_columns=int(counters[27]),
        base_validation_a_block_calls=int(counters[28]),
        base_validation_at_block_calls=int(counters[29]),
        ridge_correction_a_columns=int(counters[30]),
        ridge_correction_at_columns=int(counters[31]),
        ridge_correction_a_block_calls=int(counters[32]),
        ridge_correction_at_block_calls=int(counters[33]),
        gaussian_random_block_requests=int(counters[34]),
        ridge_correction_disposition=int(counters[35]),
        total_seconds=float(timings[0]),
        a_seconds=float(timings[1]),
        at_seconds=float(timings[2]),
        orthogonalization_seconds=float(timings[3]),
        projected_solve_seconds=float(timings[4]),
        validation_seconds=float(timings[5]),
        other_seconds=float(timings[6]),
        base_validation_seconds=float(timings[7]),
        ridge_correction_validation_seconds=float(timings[8]),
        trace=trace_float[:, :18],
        trace_float=trace_float,
        trace_int=trace_int,
        status=_require_scalar_int(status, "status"),
        stop_reason=_require_scalar_int(reason, "stop_reason"),
    )



def run_amfls(
    matrix: torch.Tensor,
    b: torch.Tensor,
    *,
    regularization: float = 0.0,
    tolerance: float = 1e-8,
    failure_probability: float = 1e-6,
    maximum_epochs: int = 0,
    maximum_depth: int = 0,
    maximum_auxiliary_width: int = 0,
    maximum_basis_size: int = 0,
    seed: int = 0,
    stream: int = 0,
) -> SolveResult:
    _check_inputs(matrix, b)
    seed_transport = _canonical_uint64_to_int64_transport(seed, "seed")
    stream_transport = _canonical_uint64_to_int64_transport(stream, "stream")
    return _decode(
        torch.ops.amfls.run_amfls(
            matrix,
            b,
            regularization,
            tolerance,
            failure_probability,
            maximum_epochs,
            maximum_depth,
            maximum_auxiliary_width,
            maximum_basis_size,
            seed_transport,
            stream_transport,
        )
    )


def run_fixed_rbgk(
    matrix: torch.Tensor,
    b: torch.Tensor,
    *,
    auxiliary_width: int,
    maximum_depth: int,
    maximum_basis_size: int = 0,
    regularization: float = 0.0,
    tolerance: float = 1e-8,
    failure_probability: float = 1e-6,
    seed: int = 0,
    stream: int = 0,
) -> SolveResult:
    _check_inputs(matrix, b)
    seed_transport = _canonical_uint64_to_int64_transport(seed, "seed")
    stream_transport = _canonical_uint64_to_int64_transport(stream, "stream")
    return _decode(
        torch.ops.amfls.run_fixed_rbgk(
            matrix,
            b,
            regularization,
            tolerance,
            failure_probability,
            auxiliary_width,
            maximum_depth,
            maximum_basis_size,
            seed_transport,
            stream_transport,
        )
    )


def run_lsqr(
    matrix: torch.Tensor,
    b: torch.Tensor,
    *,
    regularization: float = 0.0,
    tolerance: float = 1e-8,
    maximum_iterations: int = 0,
) -> SolveResult:
    _check_inputs(matrix, b)
    return _decode(
        torch.ops.amfls.run_lsqr(
            matrix,
            b,
            regularization,
            tolerance,
            maximum_iterations,
        )
    )


def run_lsmr(
    matrix: torch.Tensor,
    b: torch.Tensor,
    *,
    regularization: float = 0.0,
    tolerance: float = 1e-8,
    maximum_iterations: int = 0,
) -> SolveResult:
    _check_inputs(matrix, b)
    return _decode(
        torch.ops.amfls.run_lsmr(
            matrix,
            b,
            regularization,
            tolerance,
            maximum_iterations,
        )
    )


def run_lsmb(
    matrix: torch.Tensor,
    b: torch.Tensor,
    *,
    regularization: float = 0.0,
    tolerance: float = 1e-8,
    maximum_iterations: int = 0,
) -> SolveResult:
    _check_inputs(matrix, b)
    return _decode(
        torch.ops.amfls.run_lsmb(
            matrix,
            b,
            regularization,
            tolerance,
            maximum_iterations,
        )
    )


def run_lsrn(
    matrix: torch.Tensor,
    b: torch.Tensor,
    *,
    regularization: float = 0.0,
    tolerance: float = 1e-8,
    oversampling: float = 2.0,
    relative_rank_tolerance: float = 1e-12,
    absolute_rank_tolerance: float = 0.0,
    maximum_iterations: int = 0,
    sketch_block_size: int = 0,
    seed: int = 0,
    stream: int = 0,
) -> SolveResult:
    _check_inputs(matrix, b)
    seed_transport = _canonical_uint64_to_int64_transport(seed, "seed")
    stream_transport = _canonical_uint64_to_int64_transport(stream, "stream")
    return _decode(
        torch.ops.amfls.run_lsrn(
            matrix,
            b,
            regularization,
            tolerance,
            oversampling,
            relative_rank_tolerance,
            absolute_rank_tolerance,
            maximum_iterations,
            sketch_block_size,
            seed_transport,
            stream_transport,
        )
    )


def run_randomized_block_cg(
    matrix: torch.Tensor,
    b: torch.Tensor,
    *,
    regularization: float,
    tolerance: float = 1e-8,
    random_block_size: int = 8,
    maximum_depth: int = 0,
    seed: int = 0,
    stream: int = 0,
) -> SolveResult:
    _check_inputs(matrix, b)
    seed_transport = _canonical_uint64_to_int64_transport(seed, "seed")
    stream_transport = _canonical_uint64_to_int64_transport(stream, "stream")
    return _decode(
        torch.ops.amfls.run_randomized_block_cg(
            matrix,
            b,
            regularization,
            tolerance,
            random_block_size,
            maximum_depth,
            seed_transport,
            stream_transport,
        )
    )


def run_sparse_embedding_lsqr(
    matrix: torch.Tensor,
    b: torch.Tensor,
    *,
    regularization: float = 0.0,
    tolerance: float = 1e-8,
    embedding_distortion: float = 0.5,
    embedding_failure_probability: float = 0.1,
    sketch_rows: int = 0,
    embedding_nonzeros: int = 1,
    relative_rank_tolerance: float = 1e-12,
    absolute_rank_tolerance: float = 0.0,
    maximum_iterations: int = 0,
    sketch_block_size: int = 0,
    seed: int = 0,
    stream: int = 0,
) -> SolveResult:
    _check_inputs(matrix, b)
    seed_transport = _canonical_uint64_to_int64_transport(seed, "seed")
    stream_transport = _canonical_uint64_to_int64_transport(stream, "stream")
    return _decode(
        torch.ops.amfls.run_sparse_embedding_lsqr(
            matrix,
            b,
            regularization,
            tolerance,
            embedding_distortion,
            embedding_failure_probability,
            sketch_rows,
            embedding_nonzeros,
            relative_rank_tolerance,
            absolute_rank_tolerance,
            maximum_iterations,
            sketch_block_size,
            seed_transport,
            stream_transport,
        )
    )


def run_spir(
    matrix: torch.Tensor,
    b: torch.Tensor,
    *,
    regularization: float = 0.0,
    tolerance: float = 1e-8,
    sketch_rows: int = 0,
    embedding_nonzeros: int = 8,
    maximum_inner_iterations: int = 50,
    relative_rank_tolerance: float = 6.661338147750939e-15,
    absolute_rank_tolerance: float = 0.0,
    sketch_block_size: int = 32,
    seed: int = 0,
    stream: int = 0,
) -> SolveResult:
    _check_inputs(matrix, b)
    seed_transport = _canonical_uint64_to_int64_transport(seed, "seed")
    stream_transport = _canonical_uint64_to_int64_transport(stream, "stream")
    return _decode(
        torch.ops.amfls.run_spir(
            matrix,
            b,
            regularization,
            tolerance,
            sketch_rows,
            embedding_nonzeros,
            maximum_inner_iterations,
            relative_rank_tolerance,
            absolute_rank_tolerance,
            sketch_block_size,
            seed_transport,
            stream_transport,
        )
    )


def run_fossils(
    matrix: torch.Tensor,
    b: torch.Tensor,
    *,
    regularization: float = 0.0,
    tolerance: float = 1e-8,
    sketch_rows: int = 0,
    embedding_nonzeros: int = 8,
    maximum_inner_iterations: int = 100,
    distortion_safety: float = 1.0,
    relative_rank_tolerance: float = 6.661338147750939e-15,
    absolute_rank_tolerance: float = 0.0,
    sketch_block_size: int = 32,
    seed: int = 0,
    stream: int = 0,
) -> SolveResult:
    _check_inputs(matrix, b)
    seed_transport = _canonical_uint64_to_int64_transport(seed, "seed")
    stream_transport = _canonical_uint64_to_int64_transport(stream, "stream")
    return _decode(
        torch.ops.amfls.run_fossils(
            matrix,
            b,
            regularization,
            tolerance,
            sketch_rows,
            embedding_nonzeros,
            maximum_inner_iterations,
            distortion_safety,
            relative_rank_tolerance,
            absolute_rank_tolerance,
            sketch_block_size,
            seed_transport,
            stream_transport,
        )
    )


def run_aplicur(
    matrix: torch.Tensor,
    b: torch.Tensor,
    *,
    regularization: float,
    tolerance: float = 1e-8,
    block_size: int = 8,
    sparse_sign_nonzeros: int = 8,
    spectral_probe_count: int = 10,
    cur_tolerance: float = 0.0,
    re_preconditioning_tolerance: float = 10.0,
    dynamic_stopping_tolerance: float = 150.0,
    maximum_iterations: int = 0,
    relative_rank_tolerance: float = 1e-12,
    absolute_rank_tolerance: float = 0.0,
    seed: int = 0,
) -> SolveResult:
    _check_inputs(matrix, b)
    if matrix.layout != torch.strided:
        raise ValueError(
            "APLICUR requires a dense strided matrix because it selects "
            "explicit rows and columns"
        )
    seed_transport = _canonical_uint64_to_int64_transport(seed, "seed")
    return _decode(
        torch.ops.amfls.run_aplicur(
            matrix,
            b,
            regularization,
            tolerance,
            block_size,
            sparse_sign_nonzeros,
            spectral_probe_count,
            cur_tolerance,
            re_preconditioning_tolerance,
            dynamic_stopping_tolerance,
            maximum_iterations,
            relative_rank_tolerance,
            absolute_rank_tolerance,
            seed_transport,
        )
    )
