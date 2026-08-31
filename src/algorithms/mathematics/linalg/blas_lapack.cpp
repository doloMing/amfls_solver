#include "algorithms/mathematics/linalg/blas_lapack.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

extern "C" {
void dgemm_(
    const char*, const char*, const int*, const int*, const int*,
    const double*, const double*, const int*, const double*, const int*,
    const double*, double*, const int*);
void dgemv_(
    const char*, const int*, const int*, const double*, const double*,
    const int*, const double*, const int*, const double*, double*, const int*);
double dnrm2_(const int*, const double*, const int*);
double ddot_(const int*, const double*, const int*, const double*, const int*);
void dptsv_(
    const int*, const int*, double*, double*, double*, const int*, int*);
void dptcon_(
    const int*, const double*, const double*, const double*, double*,
    double*, int*);
void dposv_(
    const char*, const int*, const int*, double*, const int*, double*,
    const int*, int*);
}

namespace amfls::math {

void gemm(
    const Matrix& left,
    bool transpose_left,
    const Matrix& right,
    bool transpose_right,
    double alpha,
    double beta,
    Matrix& result) {
    const int left_rows = transpose_left ? left.cols() : left.rows();
    const int left_cols = transpose_left ? left.rows() : left.cols();
    const int right_rows = transpose_right ? right.cols() : right.rows();
    const int right_cols = transpose_right ? right.rows() : right.cols();
    if (left_cols != right_rows || result.rows() != left_rows ||
        result.cols() != right_cols) {
        throw std::invalid_argument("matrix multiply dimensions do not match");
    }
    if (result.empty()) {
        return;
    }
    if (left_cols == 0) {
        for (int index = 0; index < result.size(); ++index) {
            result.data()[index] *= beta;
        }
        return;
    }
    const char left_flag = transpose_left ? 'T' : 'N';
    const char right_flag = transpose_right ? 'T' : 'N';
    const int left_stride = left.leading_dimension();
    const int right_stride = right.leading_dimension();
    const int result_stride = result.leading_dimension();
    dgemm_(
        &left_flag, &right_flag, &left_rows, &right_cols, &left_cols,
        &alpha, left.data(), &left_stride, right.data(), &right_stride,
        &beta, result.data(), &result_stride);
}

void gemm_column_block(
    const Matrix& left,
    int first_left_column,
    int left_column_count,
    const Matrix& right,
    bool transpose_right,
    double alpha,
    double beta,
    Matrix& result) {
    const int right_rows = transpose_right ? right.cols() : right.rows();
    const int right_cols = transpose_right ? right.rows() : right.cols();
    if (first_left_column < 0 || left_column_count < 0 ||
        first_left_column + left_column_count > left.cols() ||
        left_column_count != right_rows || result.rows() != left.rows() ||
        result.cols() != right_cols) {
        throw std::invalid_argument(
            "matrix column-block multiply dimensions do not match");
    }
    if (result.empty()) {
        return;
    }
    if (left_column_count == 0) {
        for (int index = 0; index < result.size(); ++index) {
            result.data()[index] *= beta;
        }
        return;
    }
    const char left_flag = 'N';
    const char right_flag = transpose_right ? 'T' : 'N';
    const int left_rows = left.rows();
    const int left_stride = left.leading_dimension();
    const int right_stride = right.leading_dimension();
    const int result_stride = result.leading_dimension();
    dgemm_(
        &left_flag,
        &right_flag,
        &left_rows,
        &right_cols,
        &left_column_count,
        &alpha,
        left.column_data(first_left_column),
        &left_stride,
        right.data(),
        &right_stride,
        &beta,
        result.data(),
        &result_stride);
}

void gemv(
    const Matrix& matrix,
    bool transpose,
    const double* input,
    double alpha,
    double beta,
    double* output) {
    if (input == nullptr || output == nullptr) {
        throw std::invalid_argument("matrix-vector input must not be null");
    }
    const char flag = transpose ? 'T' : 'N';
    const int rows = matrix.rows();
    const int cols = matrix.cols();
    const int stride = matrix.leading_dimension();
    const int increment = 1;
    dgemv_(
        &flag,
        &rows,
        &cols,
        &alpha,
        matrix.data(),
        &stride,
        input,
        &increment,
        &beta,
        output,
        &increment);
}

Matrix multiply(const Matrix& left, const Matrix& right) {
    Matrix result(left.rows(), right.cols());
    gemm(left, false, right, false, 1.0, 0.0, result);
    return result;
}

Matrix transpose_multiply(const Matrix& left, const Matrix& right) {
    Matrix result(left.cols(), right.cols());
    gemm(left, true, right, false, 1.0, 0.0, result);
    return result;
}

double vector_norm(const double* values, int size) {
    const int stride = 1;
    return dnrm2_(&size, values, &stride);
}

double vector_dot(const double* left, const double* right, int size) {
    const int stride = 1;
    return ddot_(&size, left, &stride, right, &stride);
}

SpdTridiagonalSolve solve_spd_tridiagonal(
    const std::vector<double>& diagonal,
    const std::vector<double>& off_diagonal,
    const std::vector<double>& right_hand_side) {
    const int size = static_cast<int>(diagonal.size());
    if (size <= 0 ||
        off_diagonal.size() != static_cast<std::size_t>(size - 1) ||
        right_hand_side.size() != static_cast<std::size_t>(size)) {
        throw std::invalid_argument(
            "SPD tridiagonal dimensions do not match");
    }
    double matrix_one_norm = 0.0;
    for (int index = 0; index < size; ++index) {
        double row_sum = std::abs(diagonal[index]);
        if (index > 0) {
            row_sum += std::abs(off_diagonal[index - 1]);
        }
        if (index + 1 < size) {
            row_sum += std::abs(off_diagonal[index]);
        }
        matrix_one_norm = std::max(matrix_one_norm, row_sum);
    }

    std::vector<double> factored_diagonal = diagonal;
    std::vector<double> factored_off_diagonal = off_diagonal;
    SpdTridiagonalSolve result;
    result.solution = right_hand_side;
    const int right_hand_sides = 1;
    const int stride = size;
    int info = 0;
    dptsv_(
        &size,
        &right_hand_sides,
        factored_diagonal.data(),
        factored_off_diagonal.data(),
        result.solution.data(),
        &stride,
        &info);
    if (info != 0) {
        result.solution.clear();
        return result;
    }
    std::vector<double> workspace(2 * size);
    dptcon_(
        &size,
        factored_diagonal.data(),
        factored_off_diagonal.data(),
        &matrix_one_norm,
        &result.reciprocal_condition,
        workspace.data(),
        &info);
    if (info != 0 || !std::isfinite(result.reciprocal_condition)) {
        result.solution.clear();
        result.reciprocal_condition = 0.0;
        return result;
    }
    result.positive_definite = true;
    return result;
}

DenseSpdSolve solve_dense_spd(
    Matrix matrix,
    const std::vector<double>& right_hand_side) {
    const int size = matrix.rows();
    if (size <= 0 || matrix.cols() != size ||
        right_hand_side.size() != static_cast<std::size_t>(size)) {
        throw std::invalid_argument(
            "dense SPD dimensions do not match");
    }
    DenseSpdSolve result;
    result.solution = right_hand_side;
    const char triangle = 'L';
    const int right_hand_sides = 1;
    const int stride = matrix.leading_dimension();
    int info = 0;
    dposv_(
        &triangle,
        &size,
        &right_hand_sides,
        matrix.data(),
        &stride,
        result.solution.data(),
        &size,
        &info);
    if (info != 0 || !std::all_of(
            result.solution.begin(),
            result.solution.end(),
            [](double value) { return std::isfinite(value); })) {
        result.solution.clear();
        return result;
    }
    result.positive_definite = true;
    return result;
}

}  // namespace amfls::math
