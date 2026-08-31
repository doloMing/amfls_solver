#pragma once

#include <vector>

#include "algorithms/mathematics/linalg/matrix.hpp"

namespace amfls::math {

struct SpdTridiagonalSolve {
    std::vector<double> solution;
    double reciprocal_condition = 0.0;
    bool positive_definite = false;
};

struct DenseSpdSolve {
    std::vector<double> solution;
    bool positive_definite = false;
};

void gemm(
    const Matrix& left,
    bool transpose_left,
    const Matrix& right,
    bool transpose_right,
    double alpha,
    double beta,
    Matrix& result);

void gemm_column_block(
    const Matrix& left,
    int first_left_column,
    int left_column_count,
    const Matrix& right,
    bool transpose_right,
    double alpha,
    double beta,
    Matrix& result);

void gemv(
    const Matrix& matrix,
    bool transpose,
    const double* input,
    double alpha,
    double beta,
    double* output);

Matrix multiply(const Matrix& left, const Matrix& right);
Matrix transpose_multiply(const Matrix& left, const Matrix& right);
double vector_norm(const double* values, int size);
double vector_dot(const double* left, const double* right, int size);

SpdTridiagonalSolve solve_spd_tridiagonal(
    const std::vector<double>& diagonal,
    const std::vector<double>& off_diagonal,
    const std::vector<double>& right_hand_side);

DenseSpdSolve solve_dense_spd(
    Matrix matrix,
    const std::vector<double>& right_hand_side);

}  // namespace amfls::math
