#pragma once

#include "amfls/least_squares_result.hpp"
#include "amfls/matrix_operator.hpp"

namespace amfls {

struct LsmrOptions {
    double regularization = 0.0;
    double tolerance = 1e-8;
    int maximum_iterations = 0;
};

LeastSquaresResult solve_lsmr(
    const MatrixOperator& matrix,
    const double* right_hand_side,
    const LsmrOptions& options);

}  // namespace amfls
