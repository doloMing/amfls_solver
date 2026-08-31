#pragma once

#include <cstdint>

#include "amfls/least_squares_result.hpp"
#include "amfls/matrix_operator.hpp"

namespace amfls {

// Clean-room realization of Definition 3.1 and equation (3.1) in
// Chen--Huber--Lin--Zaid, "Preconditioning without a preconditioner using
// randomized block Krylov subspace methods," ETNA 65 (2026), 63--92,
// arXiv:2501.18717, applied to the ridge normal system
//
//     (A^* A + regularization I) x = A^* b.
//
// `random_block_size` is the paper's Gaussian augmentation width ell.  The
// actual starting block [A^* b, Omega] therefore has ell + 1 columns.
// `maximum_depth` is the block-Krylov order t; zero selects the domain
// dimension, the natural exact-arithmetic cap.  The Gaussian Omega uses the
// repository's seed/stream-addressed standard-normal generator.
//
// This baseline deliberately requires positive regularization.  That makes
// the shifted normal system positive definite for every rectangular or
// rank-deficient MatrixOperator without imposing an unverifiable full-column-
// rank assumption on A.
struct RandomizedBlockCgOptions {
    double regularization = 0.0;
    double tolerance = 1e-8;
    int random_block_size = 8;
    int maximum_depth = 0;
    std::uint64_t seed = 0;
    std::uint64_t stream = 0;
};

LeastSquaresResult solve_randomized_block_cg(
    const MatrixOperator& matrix,
    const double* right_hand_side,
    const RandomizedBlockCgOptions& options);

}  // namespace amfls
