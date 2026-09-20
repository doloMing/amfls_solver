from __future__ import annotations

import argparse
import gzip
import math
from pathlib import Path
from typing import Any

import numpy as np
import scipy.io
import scipy.sparse
import scipy.sparse.linalg
from scipy.sparse.csgraph import maximum_bipartite_matching, structural_rank


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit conservative numerical-rank diagnostics for one sparse "
            "application matrix."
        )
    )
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument(
        "--transpose",
        action="store_true",
        help="transpose the Matrix Market input before auditing",
    )
    parser.add_argument(
        "--relative-rank-tolerance", type=float, default=1.0e-12
    )
    parser.add_argument(
        "--propack-maxiter",
        type=int,
        default=0,
        help="zero selects max(1000, 10*n)",
    )
    return parser.parse_args()


def read_matrix(path: Path) -> scipy.sparse.csr_matrix:
    if not path.is_file():
        raise FileNotFoundError(path)
    if path.suffix == ".gz":
        with gzip.open(path, "rb") as handle:
            value = scipy.io.mmread(handle, spmatrix=True)
    else:
        with path.open("rb") as handle:
            value = scipy.io.mmread(handle, spmatrix=True)
    matrix = (
        value.astype(np.float64).tocsr()
        if scipy.sparse.issparse(value)
        else scipy.sparse.csr_matrix(np.asarray(value, dtype=np.float64))
    )
    matrix.sum_duplicates()
    matrix.eliminate_zeros()
    matrix.sort_indices()
    if matrix.ndim != 2 or min(matrix.shape) <= 0 or matrix.nnz == 0:
        raise ValueError("matrix must be a nonempty sparse matrix")
    if not np.isfinite(matrix.data).all():
        raise ValueError("matrix entries must be finite")
    return matrix


def solve_residual(
    minor: scipy.sparse.csc_matrix,
    factor: Any,
) -> float:
    dimension = minor.shape[0]
    indices = np.arange(dimension, dtype=np.float64)
    probes = np.column_stack(
        (
            np.ones(dimension, dtype=np.float64),
            np.where((indices.astype(np.int64) & 1) == 0, 1.0, -1.0),
            (indices + 1.0) / max(1, dimension),
        )
    )
    probes /= np.linalg.norm(probes, axis=0, keepdims=True)
    solutions = factor.solve(probes)
    residual = minor @ solutions - probes
    denominator = (
        float(scipy.sparse.linalg.norm(minor))
        * float(np.linalg.norm(solutions))
        + float(np.linalg.norm(probes))
    )
    return float(np.linalg.norm(residual)) / denominator


def inverse_one_norm_estimate(factor: Any, dimension: int) -> float:
    operator = scipy.sparse.linalg.LinearOperator(
        shape=(dimension, dimension),
        dtype=np.dtype(np.float64),
        matvec=lambda vector: factor.solve(np.asarray(vector)),
        rmatvec=lambda vector: factor.solve(np.asarray(vector), trans="T"),
        matmat=lambda block: factor.solve(np.asarray(block)),
        rmatmat=lambda block: factor.solve(np.asarray(block), trans="T"),
    )
    return float(
        scipy.sparse.linalg.onenormest(operator, t=min(8, dimension), itmax=20)
    )


def minor_diagnostic(
    matrix: scipy.sparse.csr_matrix,
    matched_rows: np.ndarray,
) -> dict[str, object]:
    columns = matrix.shape[1]
    diagnostic: dict[str, object] = {
        "minor_factorization_succeeded": False,
        "minor_solve_relative_residual": math.nan,
        "minor_one_norm": math.nan,
        "minor_inverse_one_norm_estimate": math.nan,
        "minor_reciprocal_one_condition_estimate": math.nan,
        "minor_min_abs_u_diagonal": math.nan,
        "minor_max_abs_u_diagonal": math.nan,
        "minor_diagnostic": "inconclusive",
    }
    minor = matrix[matched_rows, :].tocsc()
    try:
        factor = scipy.sparse.linalg.splu(minor)
    except Exception as error:
        diagnostic["minor_factorization_error"] = (
            f"{type(error).__name__}: {error}"
        )
        return diagnostic

    diagnostic["minor_factorization_succeeded"] = True
    try:
        relative_residual = solve_residual(minor, factor)
    except Exception as error:
        diagnostic["minor_solve_error"] = f"{type(error).__name__}: {error}"
        return diagnostic
    diagnostic["minor_solve_relative_residual"] = relative_residual

    try:
        inverse_norm = inverse_one_norm_estimate(factor, columns)
    except Exception as error:
        diagnostic["minor_inverse_estimate_error"] = (
            f"{type(error).__name__}: {error}"
        )
        return diagnostic

    minor_one_norm = float(abs(minor).sum(axis=0).max())
    diagonal = np.abs(factor.U.diagonal())
    if math.isfinite(inverse_norm) and inverse_norm > 0.0:
        reciprocal_condition = 1.0 / (minor_one_norm * inverse_norm)
    else:
        reciprocal_condition = math.nan
    diagnostic.update(
        {
            "minor_one_norm": minor_one_norm,
            "minor_inverse_one_norm_estimate": inverse_norm,
            "minor_reciprocal_one_condition_estimate":
                reciprocal_condition,
            "minor_min_abs_u_diagonal": float(diagonal.min()),
            "minor_max_abs_u_diagonal": float(diagonal.max()),
        }
    )
    return diagnostic


def propack_diagnostic(
    matrix: scipy.sparse.csr_matrix,
    sigma_max_lower_bound: float,
    relative_rank_tolerance: float,
    maximum_iterations: int,
) -> dict[str, object]:
    diagnostic: dict[str, object] = {
        "propack_used": True,
        "propack_returned": False,
        "propack_smallest_singular_value": math.nan,
        "propack_relative_singular_triplet_residual": math.nan,
        "propack_witness_image_norm": math.nan,
        "propack_witness_ratio": math.nan,
        "propack_decision": "ambiguous",
    }
    try:
        # For a tall matrix, PROPACK applied directly can expose the exact
        # left-null-space zeros instead of the smallest value among the n
        # singular values of interest.  Applying it to A* has the same
        # nonzero singular values and avoids those extra left-null directions.
        left_transposed, singular_values, right_transpose_transposed = (
            scipy.sparse.linalg.svds(
                matrix.T.tocsr(),
                k=1,
                which="SM",
                solver="propack",
                tol=0.0,
                maxiter=maximum_iterations,
                return_singular_vectors=True,
                rng=np.random.default_rng(0),
            )
        )
    except Exception as error:
        diagnostic["propack_error"] = f"{type(error).__name__}: {error}"
        return diagnostic

    singular_value = float(singular_values[0])
    right_vector = np.asarray(left_transposed[:, 0], dtype=np.float64)
    left_vector = right_transpose_transposed[0, :]
    right_norm = float(np.linalg.norm(right_vector))
    if not math.isfinite(right_norm) or right_norm == 0.0:
        diagnostic["propack_error"] = "returned right vector is not usable"
        return diagnostic
    right_vector /= right_norm
    left_vector = np.asarray(left_vector, dtype=np.float64) / right_norm
    witness_image = np.asarray(matrix @ right_vector).reshape(-1)
    witness_image_norm = float(np.linalg.norm(witness_image))
    witness_ratio = witness_image_norm / sigma_max_lower_bound
    left_residual = matrix @ right_vector - singular_value * left_vector
    right_residual = matrix.T @ left_vector - singular_value * right_vector
    relative_residual = math.hypot(
        float(np.linalg.norm(left_residual)),
        float(np.linalg.norm(right_residual)),
    ) / sigma_max_lower_bound
    decision = (
        "numerically_rank_deficient"
        if math.isfinite(witness_ratio)
        and witness_ratio < relative_rank_tolerance
        else "ambiguous"
    )
    diagnostic.update(
        {
            "propack_returned": True,
            "propack_smallest_singular_value": singular_value,
            "propack_relative_singular_triplet_residual": relative_residual,
            "propack_witness_image_norm": witness_image_norm,
            "propack_witness_ratio": witness_ratio,
            "propack_decision": decision,
        }
    )
    return diagnostic


def audit_matrix(
    matrix: scipy.sparse.csr_matrix,
    relative_rank_tolerance: float = 1.0e-12,
    propack_maximum_iterations: int = 0,
) -> dict[str, object]:
    if (
        not math.isfinite(relative_rank_tolerance)
        or relative_rank_tolerance <= 0.0
        or relative_rank_tolerance >= 1.0
    ):
        raise ValueError("relative rank tolerance must lie strictly in (0, 1)")
    matrix = matrix.astype(np.float64).tocsr()
    matrix.sum_duplicates()
    matrix.eliminate_zeros()
    matrix.sort_indices()
    rows, columns = matrix.shape
    if rows < columns:
        raise ValueError("rank audit requires rows greater than or equal to columns")
    if columns < 2:
        raise ValueError("rank audit requires at least two columns")

    pattern_rank = int(structural_rank(matrix))
    operator_frobenius_norm = float(scipy.sparse.linalg.norm(matrix))
    squared_column_norms = np.asarray(
        matrix.multiply(matrix).sum(axis=0)
    ).reshape(-1)
    sigma_max_lower_bound = math.sqrt(float(squared_column_norms.max()))
    result: dict[str, object] = {
        "rows": rows,
        "columns": columns,
        "nonzeros": int(matrix.nnz),
        "structural_rank": pattern_rank,
        "relative_rank_tolerance": relative_rank_tolerance,
        "operator_frobenius_norm": operator_frobenius_norm,
        "sigma_max_lower_bound": sigma_max_lower_bound,
        "propack_used": False,
    }
    if pattern_rank < columns:
        result.update(
            {
                "decision": "rank_deficient",
                "numerical_rank_decision": "less_than_n",
                "minimum_norm_external_check_required": True,
            }
        )
        return result

    matched_rows = maximum_bipartite_matching(matrix, perm_type="row")
    if matched_rows.shape != (columns,) or np.any(matched_rows < 0):
        raise RuntimeError("structural rank and maximum matching disagree")
    result["matched_minor_order"] = columns
    minor = minor_diagnostic(
        matrix,
        matched_rows,
    )
    result.update(minor)

    maximum_iterations = (
        propack_maximum_iterations
        if propack_maximum_iterations > 0
        else max(1000, 10 * columns)
    )
    fallback = propack_diagnostic(
        matrix,
        sigma_max_lower_bound,
        relative_rank_tolerance,
        maximum_iterations,
    )
    result.update(fallback)
    fallback_decision = str(fallback["propack_decision"])
    if fallback_decision == "numerically_rank_deficient":
        result.update(
            {
                "decision": "rank_deficient",
                "numerical_rank_decision": "less_than_n",
                "minimum_norm_external_check_required": True,
            }
        )
    else:
        result.update(
            {
                "decision": "ambiguous",
                "numerical_rank_decision": "undetermined",
                "minimum_norm_external_check_required": True,
            }
        )
    return result


def format_value(value: object) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return f"{value:.17g}" if math.isfinite(value) else "not_available"
    return str(value)


def main() -> None:
    args = parse_args()
    matrix = read_matrix(args.matrix)
    if args.transpose:
        matrix = matrix.T.tocsr()
        matrix.sort_indices()
    result = audit_matrix(
        matrix,
        relative_rank_tolerance=float(args.relative_rank_tolerance),
        propack_maximum_iterations=int(args.propack_maxiter),
    )
    for name, value in result.items():
        print(f"{name}={format_value(value)}")


if __name__ == "__main__":
    main()
