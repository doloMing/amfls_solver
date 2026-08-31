#pragma once

#include <vector>

#include "algorithms/mathematics/linalg/matrix.hpp"

namespace amfls::math {

struct SvdResult {
    Matrix u;
    std::vector<double> singular_values;
    Matrix vt;
};

struct MinimumNormLeastSquaresResult {
    std::vector<double> solution;
    int rank = 0;
};

SvdResult thin_svd(const Matrix& matrix);

// Compute a thin SVD while allowing LAPACK to overwrite matrix storage.
SvdResult thin_svd(Matrix&& matrix);

// Solve min ||matrix*x-rhs||_2 with rank-revealing complete orthogonal
// factorization.  The tolerance controls the effective triangular rank.
MinimumNormLeastSquaresResult minimum_norm_least_squares(
    const Matrix& matrix,
    const double* right_hand_side,
    double relative_tolerance);

// Solve the same problem while allowing LAPACK to overwrite matrix storage.
MinimumNormLeastSquaresResult minimum_norm_least_squares(
    Matrix&& matrix,
    const double* right_hand_side,
    double relative_tolerance);

// Return an orthonormal basis for range(matrix), using a scale-aware SVD cut.
Matrix numerical_column_space(
    const Matrix& matrix,
    double relative_tolerance = -1.0,
    double absolute_tolerance = 0.0);

}  // namespace amfls::math
