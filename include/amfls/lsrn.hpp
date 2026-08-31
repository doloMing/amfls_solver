#pragma once

#include <cstdint>

#include "amfls/least_squares_result.hpp"
#include "amfls/matrix_operator.hpp"

namespace amfls {

struct LsrnOptions {
    double regularization = 0.0;
    double tolerance = 1e-8;
    double oversampling = 2.0;

    // A finite singular value sigma is retained when
    // sigma > max(relative_rank_tolerance * sigma_max,
    //             absolute_rank_tolerance) and its reciprocal is finite.
    // The latter is an explicit FP64 construction guard.  The 1e-12 default
    // matches the reference algorithm's practical RCOND; -1 selects a
    // dimension-scaled FP64 cutoff when explicitly requested.
    double relative_rank_tolerance = 1e-12;
    double absolute_rank_tolerance = 0.0;

    int maximum_iterations = 0;
    int sketch_block_size = 0;
    std::uint64_t seed = 0;
    std::uint64_t stream = 0;
};

LeastSquaresResult solve_lsrn(
    const MatrixOperator& matrix,
    const double* right_hand_side,
    const LsrnOptions& options);

}  // namespace amfls
