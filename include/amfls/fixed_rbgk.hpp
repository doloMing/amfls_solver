#pragma once

#include <cstdint>

#include "amfls/least_squares_result.hpp"
#include "amfls/matrix_operator.hpp"

namespace amfls {

struct FixedRbgkOptions {
    double regularization = 0.0;
    double tolerance = 1e-8;
    double failure_probability = 1e-6;
    int auxiliary_width = 4;
    int maximum_depth = 32;
    int maximum_basis_size = 0;
    std::uint64_t seed = 0;
    std::uint64_t stream = 0;
    bool stop_early = true;
};

LeastSquaresResult solve_fixed_rbgk(
    const MatrixOperator& matrix,
    const double* right_hand_side,
    const FixedRbgkOptions& options);

}  // namespace amfls
