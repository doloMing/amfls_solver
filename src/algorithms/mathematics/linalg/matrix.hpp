#pragma once

#include <vector>

namespace amfls::math {

// Owning FP64 column-major matrix used at the BLAS/LAPACK boundary.
class Matrix {
public:
    Matrix();
    Matrix(int rows, int cols);
    Matrix(int rows, int cols, double value);
    Matrix(int rows, int cols, const double* values);

    int rows() const;
    int cols() const;
    int leading_dimension() const;
    int size() const;
    bool empty() const;

    double* data();
    const double* data() const;
    double* column_data(int col);
    const double* column_data(int col) const;
    double& operator()(int row, int col);
    double operator()(int row, int col) const;

    void fill(double value);
    void append_columns(const Matrix& block);
    void append_zero_columns(int count);
    void truncate_columns(int count);
    void give_values_to(std::vector<double>& output);

private:
    int rows_ = 0;
    int cols_ = 0;
    std::vector<double> values_;
};

// LAPACK-style scaled sum of squares.  This permits a cache-friendly pass
// over interleaved rows or columns without losing the overflow/underflow
// protection of a BLAS Euclidean norm.
class ScaledSumOfSquares {
public:
    void add(double value) noexcept;
    double norm() const noexcept;

private:
    double scale_ = 0.0;
    double sum_squares_ = 1.0;
    bool has_nan_ = false;
};

Matrix copy_columns(const Matrix& source, int first_col, int count);
void set_block(Matrix& target, int first_row, int first_col, const Matrix& block);
Matrix join_columns(const Matrix& left, const Matrix& right);
double squared_frobenius_norm(const Matrix& matrix);
double frobenius_norm(const Matrix& matrix);
std::vector<double> row_norms(const Matrix& matrix);

}  // namespace amfls::math
