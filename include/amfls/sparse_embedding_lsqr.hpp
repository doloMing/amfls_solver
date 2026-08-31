#pragma once

#include <cstdint>

#include "amfls/least_squares_result.hpp"
#include "amfls/matrix_operator.hpp"

namespace amfls {

// Sparse-embedding right-preconditioned LSQR for full-column-rank,
// overdetermined ordinary least-squares problems.  Ridge, square/wide
// operators, and a rank-deficient embedded matrix are outside this baseline.
struct SparseEmbeddingLsqrOptions {
    double regularization = 0.0;
    double tolerance = 1e-8;

    // With sketch_rows == 0, the embedding has
    // ceil((cols^2 + cols) /
    //      (embedding_distortion^2 * embedding_failure_probability)) rows.
    // For embedding_nonzeros == 1 this is the sufficient dimension in
    // Meng--Mahoney, Theorem 1.  A positive sketch_rows overrides the formula
    // and must be at least cols.
    double embedding_distortion = 0.5;
    double embedding_failure_probability = 0.1;
    int sketch_rows = 0;

    // Each original row is sent to this many distinct sketch rows with
    // independent signs scaled by 1 / sqrt(embedding_nonzeros).  One recovers
    // the one-nonzero embedding when this value is one.
    int embedding_nonzeros = 1;

    // A finite embedded singular value sigma is retained when
    // sigma > max(relative_rank_tolerance * sigma_max,
    //             absolute_rank_tolerance)
    // and its reciprocal is finite.  -1 selects a dimension-scaled FP64
    // cutoff.  All original columns must be retained before LSQR starts.
    double relative_rank_tolerance = 1e-12;
    double absolute_rank_tolerance = 0.0;

    // Zero means the original column count for iterations and all embedding
    // rows in one A* callback for sketch_block_size.
    int maximum_iterations = 0;
    int sketch_block_size = 0;

    // These select the reproducible pseudorandom row assignments and signs.
    std::uint64_t seed = 0;
    std::uint64_t stream = 0;
};

LeastSquaresResult solve_sparse_embedding_lsqr(
    const MatrixOperator& matrix,
    const double* right_hand_side,
    const SparseEmbeddingLsqrOptions& options);

}  // namespace amfls
