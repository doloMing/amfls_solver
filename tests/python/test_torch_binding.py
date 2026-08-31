from __future__ import annotations

import math
from pathlib import Path
import sys

import torch


if len(sys.argv) != 3:
    raise RuntimeError("expected the Torch library and project root")

library = Path(sys.argv[1]).resolve()
project_root = Path(sys.argv[2]).resolve()
sys.path.insert(0, str(project_root / "python"))

from amfls import (  # noqa: E402
    SolveResult,
    load_library,
    run_amfls,
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


load_library(library)

matrix = torch.tensor(
    [
        [3.0, 0.2, 0.0, 0.1],
        [0.1, 2.5, 0.3, 0.0],
        [0.0, 0.2, 2.0, 0.4],
        [0.2, 0.0, 0.1, 1.5],
        [1.0, -0.3, 0.2, 0.1],
        [0.1, 0.5, -0.2, 0.7],
        [0.4, 0.1, 0.8, -0.1],
        [-0.2, 0.3, 0.0, 0.9],
    ],
    dtype=torch.float64,
)
truth = torch.tensor([1.0, -2.0, 0.5, 1.5], dtype=torch.float64)
right_hand_side = matrix @ truth


def check_result(result: SolveResult, label: str) -> None:
    assert isinstance(result, SolveResult), label
    assert result.solution.shape == (matrix.shape[1],), label
    assert result.solution.dtype == torch.float64, label
    assert math.isfinite(result.residual_norm), label
    assert result.a_columns == result.search_a_columns + result.validation_a_columns
    assert result.at_columns == result.search_at_columns + result.validation_at_columns
    assert result.trace_float.ndim == 2 and result.trace_float.shape[1] == 29
    assert result.trace_int.shape == (result.trace_float.shape[0], 4)
    assert torch.equal(result.trace, result.trace_float[:, :18])
    if result.trace_int.shape[0] > 0:
        assert bool(torch.all((result.trace_int[:, 2] == 0) |
                              (result.trace_int[:, 2] == 1)))


ordinary_results = {
    "amfls": run_amfls(
        matrix,
        right_hand_side,
        tolerance=1e-10,
        maximum_epochs=4,
        maximum_depth=4,
        maximum_auxiliary_width=2,
        maximum_basis_size=4,
        seed=11,
    ),
    "fixed": run_fixed_rbgk(
        matrix,
        right_hand_side,
        auxiliary_width=1,
        maximum_depth=4,
        maximum_basis_size=4,
        tolerance=1e-10,
        seed=11,
    ),
    "lsqr": run_lsqr(matrix, right_hand_side, tolerance=1e-10),
    "lsmr": run_lsmr(matrix, right_hand_side, tolerance=1e-10),
    "lsmb": run_lsmb(matrix, right_hand_side, tolerance=1e-10),
    "lsrn": run_lsrn(
        matrix,
        right_hand_side,
        tolerance=1e-10,
        oversampling=2.0,
        seed=11,
    ),
    "sparse": run_sparse_embedding_lsqr(
        matrix,
        right_hand_side,
        tolerance=1e-10,
        sketch_rows=16,
        embedding_nonzeros=2,
        seed=11,
    ),
    "spir": run_spir(
        matrix,
        right_hand_side,
        tolerance=1e-10,
        sketch_rows=16,
        embedding_nonzeros=2,
        seed=11,
    ),
    "fossils": run_fossils(
        matrix,
        right_hand_side,
        tolerance=1e-10,
        sketch_rows=16,
        embedding_nonzeros=2,
        seed=11,
    ),
}

for label, result in ordinary_results.items():
    check_result(result, label)

assert ordinary_results["lsqr"].success
assert ordinary_results["amfls"].success
assert torch.linalg.vector_norm(
    ordinary_results["amfls"].solution - truth
) < 1e-8

padded_matrix = torch.empty(
    (matrix.shape[0], 2 * matrix.shape[1]), dtype=torch.float64
)
padded_matrix[:, ::2] = matrix
noncontiguous_matrix = padded_matrix[:, ::2]
assert not noncontiguous_matrix.is_contiguous()
noncontiguous_result = run_lsqr(
    noncontiguous_matrix,
    right_hand_side,
    tolerance=1e-10,
)
check_result(noncontiguous_result, "noncontiguous_dense")
assert noncontiguous_result.success
assert torch.linalg.vector_norm(noncontiguous_result.solution - truth) < 1e-8

ridge_results = {
    "rbcg": run_randomized_block_cg(
        matrix,
        right_hand_side,
        regularization=0.1,
        tolerance=1e-8,
        random_block_size=2,
        maximum_depth=8,
        seed=7,
    ),
    "aplicur": run_aplicur(
        matrix,
        right_hand_side,
        regularization=0.1,
        tolerance=1e-8,
        block_size=2,
        sparse_sign_nonzeros=2,
        spectral_probe_count=2,
        seed=7,
    ),
}

for label, result in ridge_results.items():
    check_result(result, label)

noncontiguous_aplicur = run_aplicur(
    noncontiguous_matrix,
    right_hand_side,
    regularization=0.1,
    tolerance=1e-8,
    block_size=2,
    sparse_sign_nonzeros=2,
    spectral_probe_count=2,
    seed=7,
)
check_result(noncontiguous_aplicur, "noncontiguous_aplicur")
assert noncontiguous_aplicur.success
assert torch.equal(
    noncontiguous_aplicur.solution,
    ridge_results["aplicur"].solution,
)

csr_matrix = matrix.to_sparse_csr()
csr_results = {
    "amfls_csr": run_amfls(
        csr_matrix,
        right_hand_side,
        tolerance=1e-10,
        maximum_epochs=4,
        maximum_depth=4,
        maximum_auxiliary_width=2,
        maximum_basis_size=4,
        seed=11,
    ),
    "fixed_csr": run_fixed_rbgk(
        csr_matrix,
        right_hand_side,
        auxiliary_width=1,
        maximum_depth=4,
        maximum_basis_size=4,
        tolerance=1e-10,
        seed=11,
    ),
    "lsqr_csr": run_lsqr(csr_matrix, right_hand_side, tolerance=1e-10),
    "lsmr_csr": run_lsmr(csr_matrix, right_hand_side, tolerance=1e-10),
    "lsmb_csr": run_lsmb(csr_matrix, right_hand_side, tolerance=1e-10),
    "lsrn_csr": run_lsrn(
        csr_matrix,
        right_hand_side,
        tolerance=1e-10,
        oversampling=2.0,
        seed=11,
    ),
    "sparse_embedding_csr": run_sparse_embedding_lsqr(
        csr_matrix,
        right_hand_side,
        tolerance=1e-10,
        sketch_rows=16,
        embedding_nonzeros=2,
        seed=11,
    ),
    "spir_csr": run_spir(
        csr_matrix,
        right_hand_side,
        tolerance=1e-10,
        sketch_rows=16,
        embedding_nonzeros=2,
        seed=11,
    ),
    "fossils_csr": run_fossils(
        csr_matrix,
        right_hand_side,
        tolerance=1e-10,
        sketch_rows=16,
        embedding_nonzeros=2,
        seed=11,
    ),
    "rbcg_csr": run_randomized_block_cg(
        csr_matrix,
        right_hand_side,
        regularization=0.1,
        tolerance=1e-8,
        random_block_size=2,
        maximum_depth=8,
        seed=7,
    ),
}

for label, result in csr_results.items():
    check_result(result, label)

assert csr_results["lsqr_csr"].success
assert csr_results["amfls_csr"].success
assert torch.linalg.vector_norm(
    csr_results["amfls_csr"].solution - truth
) < 1e-8

try:
    run_aplicur(
        csr_matrix,
        right_hand_side,
        regularization=0.1,
    )
except ValueError:
    pass
else:
    raise AssertionError("APLICUR must reject sparse CSR input")

high_seed = (1 << 64) - 1
first = run_amfls(matrix, right_hand_side, seed=high_seed)
second = run_amfls(matrix, right_hand_side, seed=high_seed)
assert torch.equal(first.solution, second.solution)

try:
    run_amfls(matrix, right_hand_side, seed=-1)
except ValueError:
    pass
else:
    raise AssertionError("negative public seed must be rejected")

try:
    run_amfls(
        matrix,
        right_hand_side,
        maximum_auxiliary_width=-1,
    )
except ValueError:
    pass
else:
    raise AssertionError("negative AMFLS auxiliary-width limit must be rejected")

try:
    run_lsqr(matrix.to(torch.float32), right_hand_side)
except ValueError:
    pass
else:
    raise AssertionError("float32 input must be rejected")
