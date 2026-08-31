#include "algorithms/mathematics/linalg/svd.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

#include "algorithms/mathematics/linalg/blas_lapack.hpp"

extern "C" {
void dgesdd_(
    const char*, const int*, const int*, double*, const int*, double*,
    double*, const int*, double*, const int*, double*, const int*, int*, int*);
void dgelsy_(
    const int*, const int*, const int*, double*, const int*, double*,
    const int*, int*, const double*, int*, double*, const int*, int*);
}

namespace amfls::math {

SvdResult thin_svd(Matrix&& matrix) {
    const int rows = matrix.rows();
    const int cols = matrix.cols();
    const int count = std::min(rows, cols);
    SvdResult result{
        Matrix(rows, count), std::vector<double>(count), Matrix(count, cols)};
    if (count == 0) {
        return result;
    }

    const char vectors = 'S';
    const int stride = matrix.leading_dimension();
    const int left_stride = result.u.leading_dimension();
    const int right_stride = result.vt.leading_dimension();
    int workspace_size = -1;
    double workspace_query = 0.0;
    std::vector<int> integer_workspace(8 * count);
    int info = 0;
    dgesdd_(
        &vectors, &rows, &cols, matrix.data(), &stride,
        result.singular_values.data(), result.u.data(), &left_stride,
        result.vt.data(), &right_stride, &workspace_query, &workspace_size,
        integer_workspace.data(), &info);
    if (info != 0) {
        throw std::runtime_error("dgesdd workspace query failed");
    }
    workspace_size = std::max(1, static_cast<int>(std::ceil(workspace_query)));
    std::vector<double> workspace(workspace_size);
    dgesdd_(
        &vectors, &rows, &cols, matrix.data(), &stride,
        result.singular_values.data(), result.u.data(), &left_stride,
        result.vt.data(), &right_stride, workspace.data(), &workspace_size,
        integer_workspace.data(), &info);
    if (info != 0) {
        throw std::runtime_error("dgesdd failed");
    }
    return result;
}

SvdResult thin_svd(const Matrix& matrix) {
    Matrix work = matrix;
    return thin_svd(std::move(work));
}

namespace {

void validate_minimum_norm_least_squares_inputs(
    const Matrix& matrix,
    const double* right_hand_side,
    double relative_tolerance) {
    const int rows = matrix.rows();
    const int cols = matrix.cols();
    if (rows <= 0 || cols < 0) {
        throw std::invalid_argument(
            "minimum-norm least-squares matrix dimensions are invalid");
    }
    if (right_hand_side == nullptr) {
        throw std::invalid_argument(
            "minimum-norm least-squares right-hand side must not be null");
    }
    if (!std::isfinite(relative_tolerance) ||
        relative_tolerance < 0.0 || relative_tolerance >= 1.0) {
        throw std::invalid_argument(
            "minimum-norm least-squares tolerance must lie in [0,1)");
    }
}

MinimumNormLeastSquaresResult minimum_norm_least_squares_impl(
    Matrix work_matrix,
    const double* right_hand_side,
    double relative_tolerance) {
    const int rows = work_matrix.rows();
    const int cols = work_matrix.cols();
    const int right_hand_side_stride = std::max(rows, cols);
    std::vector<double> work_rhs(right_hand_side_stride, 0.0);
    std::copy(
        right_hand_side, right_hand_side + rows, work_rhs.begin());
    MinimumNormLeastSquaresResult result;
    result.solution.resize(cols);
    if (cols == 0) {
        return result;
    }

    const int right_hand_sides = 1;
    const int matrix_stride = work_matrix.leading_dimension();
    std::vector<int> pivot(cols, 0);
    int workspace_size = -1;
    double workspace_query = 0.0;
    int info = 0;
    dgelsy_(
        &rows,
        &cols,
        &right_hand_sides,
        work_matrix.data(),
        &matrix_stride,
        work_rhs.data(),
        &right_hand_side_stride,
        pivot.data(),
        &relative_tolerance,
        &result.rank,
        &workspace_query,
        &workspace_size,
        &info);
    if (info != 0) {
        throw std::runtime_error("dgelsy workspace query failed");
    }

    workspace_size = std::max(
        1, static_cast<int>(std::ceil(workspace_query)));
    std::vector<double> workspace(workspace_size);
    std::fill(pivot.begin(), pivot.end(), 0);
    dgelsy_(
        &rows,
        &cols,
        &right_hand_sides,
        work_matrix.data(),
        &matrix_stride,
        work_rhs.data(),
        &right_hand_side_stride,
        pivot.data(),
        &relative_tolerance,
        &result.rank,
        workspace.data(),
        &workspace_size,
        &info);
    if (info != 0) {
        throw std::runtime_error("dgelsy failed");
    }
    std::copy(work_rhs.begin(), work_rhs.begin() + cols, result.solution.begin());
    return result;
}

}  // namespace

MinimumNormLeastSquaresResult minimum_norm_least_squares(
    const Matrix& matrix,
    const double* right_hand_side,
    double relative_tolerance) {
    validate_minimum_norm_least_squares_inputs(
        matrix, right_hand_side, relative_tolerance);
    return minimum_norm_least_squares_impl(
        Matrix(matrix), right_hand_side, relative_tolerance);
}

MinimumNormLeastSquaresResult minimum_norm_least_squares(
    Matrix&& matrix,
    const double* right_hand_side,
    double relative_tolerance) {
    validate_minimum_norm_least_squares_inputs(
        matrix, right_hand_side, relative_tolerance);
    return minimum_norm_least_squares_impl(
        std::move(matrix), right_hand_side, relative_tolerance);
}

Matrix numerical_column_space(
    const Matrix& matrix,
    double relative_tolerance,
    double absolute_tolerance) {
    SvdResult svd = thin_svd(matrix);
    if (svd.singular_values.empty() || svd.singular_values.front() == 0.0) {
        return Matrix(matrix.rows(), 0);
    }
    if (!std::isfinite(relative_tolerance) || relative_tolerance < 0.0) {
        if (relative_tolerance != -1.0) {
            throw std::invalid_argument(
                "relative SVD tolerance must be nonnegative or the default sentinel");
        }
    }
    if (!std::isfinite(absolute_tolerance) || absolute_tolerance < 0.0) {
        throw std::invalid_argument("absolute SVD tolerance must be nonnegative");
    }
    if (relative_tolerance < 0.0) {
        relative_tolerance =
            16.0 * std::numeric_limits<double>::epsilon() *
            std::max(matrix.rows(), matrix.cols());
    }
    const double cutoff = std::max(
        relative_tolerance * svd.singular_values.front(), absolute_tolerance);
    int rank = 0;
    while (rank < static_cast<int>(svd.singular_values.size()) &&
           svd.singular_values[rank] > cutoff) {
        ++rank;
    }
    svd.u.truncate_columns(rank);
    return std::move(svd.u);
}

}  // namespace amfls::math
