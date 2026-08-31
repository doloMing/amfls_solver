#pragma once

#include <cstdint>

#include "amfls/least_squares_result.hpp"
#include "amfls/matrix_operator.hpp"

namespace amfls {

struct AmflsOptions {
    double regularization = 0.0;
    double tolerance = 1e-8;
    double failure_probability = 1e-6;
    int maximum_epochs = 0;
    int maximum_basis_size = 0;
    std::uint64_t seed = 0;
    std::uint64_t stream = 0;
    int maximum_depth = 0;
    int maximum_auxiliary_width = 0;
};

LeastSquaresResult solve_amfls(
    const MatrixOperator& matrix,
    const double* right_hand_side,
    const AmflsOptions& options);

}  // namespace amfls
