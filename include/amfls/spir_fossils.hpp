#pragma once

#include <cstdint>
#include <limits>

#include "amfls/least_squares_result.hpp"
#include "amfls/matrix_operator.hpp"

namespace amfls {

// Clean-room implementation of Algorithms 1 and 3 together with the
// production recommendations in Section 4.2 of
// Epperly--Meier--Nakatsukasa, "Fast randomized least-squares solvers can be
// just as accurate and stable as classical direct solvers" (2025).
//
// Scope: real FP64, m >= n, ordinary least squares.  A nonzero
// regularization value is rejected because the paper's numerical-rank
// stabilization is not the same problem as user-requested ridge regression.
// The implementation normalizes the columns, uses the paper's SVD
// preconditioner and sparse sign embedding, switches from the first
// refinement at the recommended update-size threshold, and uses the sketched
// Karlson--Walden posterior estimate in the second refinement.  Directions
// below the declared SVD cutoff are truncated.  The returned, unscaled
// candidate is finally tested on the original problem.
//
// Deliberate implementation choices:
//   * zero sketch_rows selects 2*n for SPIR;
//   * embedding_nonzeros is capped at sketch_rows (the paper uses eight);
//   * maximum_inner_iterations is a per-refinement safety cap;
//   * sketch_block_size partitions sketch callbacks and the column-norm
//     callbacks required by an opaque operator; stored matrices use a direct
//     entrywise norm pass.  It does not change the sampled embedding.
struct SpirOptions {
    double regularization = 0.0;
    double tolerance = 1e-8;
    int sketch_rows = 0;
    int embedding_nonzeros = 8;
    int maximum_inner_iterations = 50;
    double relative_rank_tolerance =
        30.0 * std::numeric_limits<double>::epsilon();
    double absolute_rank_tolerance = 0.0;
    int sketch_block_size = 32;
    std::uint64_t seed = 0;
    std::uint64_t stream = 0;
};

LeastSquaresResult solve_spir(
    const MatrixOperator& matrix,
    const double* right_hand_side,
    const SpirOptions& options);

// Clean-room implementation of Algorithms 1, 2, and the core iteration of
// Algorithm 4 in the same paper.  The shape, regularization, embedding, SVD
// truncation, and shared validation scope are the same as SPIR above.
//
// zero sketch_rows selects the paper's practical 12*n setting.  With
// eta = distortion_safety*sqrt(retained_rank/sketch_rows), the coefficients
// are alpha=(1-eta^2)^2 and beta=eta^2.  The default fixed cap follows the
// 100-update limit in Algorithm 4.  Adaptive phase switching and posterior
// stopping are always active and use the paper's fixed constants.
struct FossilsOptions {
    double regularization = 0.0;
    double tolerance = 1e-8;
    int sketch_rows = 0;
    int embedding_nonzeros = 8;
    int maximum_inner_iterations = 100;
    double distortion_safety = 1.0;
    double relative_rank_tolerance =
        30.0 * std::numeric_limits<double>::epsilon();
    double absolute_rank_tolerance = 0.0;
    int sketch_block_size = 32;
    std::uint64_t seed = 0;
    std::uint64_t stream = 0;
};

LeastSquaresResult solve_fossils(
    const MatrixOperator& matrix,
    const double* right_hand_side,
    const FossilsOptions& options);

}  // namespace amfls
