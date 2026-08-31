#include "algorithms/mathematics/linalg/matrix.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

extern "C" {
double dnrm2_(const int*, const double*, const int*);
}

namespace amfls::math {

Matrix::Matrix() = default;

Matrix::Matrix(int rows, int cols) : Matrix(rows, cols, 0.0) {}

Matrix::Matrix(int rows, int cols, double value) : rows_(rows), cols_(cols) {
    if (rows < 0 || cols < 0) {
        throw std::invalid_argument("matrix dimensions must be nonnegative");
    }
    const long long count = static_cast<long long>(rows) * cols;
    if (count > std::numeric_limits<int>::max()) {
        throw std::length_error("matrix is too large for the LP64 backend");
    }
    values_.assign(static_cast<std::size_t>(count), value);
}

Matrix::Matrix(int rows, int cols, const double* values) : Matrix(rows, cols) {
    if (!values_.empty()) {
        std::copy(values, values + values_.size(), values_.begin());
    }
}

int Matrix::rows() const { return rows_; }
int Matrix::cols() const { return cols_; }
int Matrix::leading_dimension() const { return std::max(1, rows_); }
int Matrix::size() const { return static_cast<int>(values_.size()); }
bool Matrix::empty() const { return rows_ == 0 || cols_ == 0; }
double* Matrix::data() { return values_.data(); }
const double* Matrix::data() const { return values_.data(); }
double* Matrix::column_data(int col) { return values_.data() + col * rows_; }
const double* Matrix::column_data(int col) const {
    return values_.data() + col * rows_;
}
double& Matrix::operator()(int row, int col) { return values_[row + col * rows_]; }
double Matrix::operator()(int row, int col) const {
    return values_[row + col * rows_];
}

void Matrix::fill(double value) { std::fill(values_.begin(), values_.end(), value); }

void Matrix::append_columns(const Matrix& block) {
    if (rows_ != block.rows()) {
        throw std::invalid_argument("matrix row counts do not match");
    }
    const long long new_cols = static_cast<long long>(cols_) + block.cols();
    const long long new_size = static_cast<long long>(rows_) * new_cols;
    if (new_cols > std::numeric_limits<int>::max() ||
        new_size > std::numeric_limits<int>::max()) {
        throw std::length_error("matrix is too large for the LP64 backend");
    }
    values_.insert(values_.end(), block.data(), block.data() + block.size());
    cols_ = static_cast<int>(new_cols);
}

void Matrix::append_zero_columns(int count) {
    if (count < 0) {
        throw std::invalid_argument("matrix column count must be nonnegative");
    }
    const long long new_cols = static_cast<long long>(cols_) + count;
    const long long new_size = static_cast<long long>(rows_) * new_cols;
    if (new_cols > std::numeric_limits<int>::max() ||
        new_size > std::numeric_limits<int>::max()) {
        throw std::length_error("matrix is too large for the LP64 backend");
    }
    values_.resize(static_cast<std::size_t>(new_size), 0.0);
    cols_ = static_cast<int>(new_cols);
}

void Matrix::truncate_columns(int count) {
    if (count < 0 || count > cols_) {
        throw std::invalid_argument("column count is out of range");
    }
    values_.resize(static_cast<std::size_t>(rows_) * count);
    cols_ = count;
}

void Matrix::give_values_to(std::vector<double>& output) {
    output.clear();
    output.swap(values_);
    rows_ = 0;
    cols_ = 0;
}

void ScaledSumOfSquares::add(double value) noexcept {
    const double magnitude = std::abs(value);
    if (std::isnan(magnitude)) {
        has_nan_ = true;
        return;
    }
    if (std::isinf(magnitude)) {
        scale_ = std::numeric_limits<double>::infinity();
        sum_squares_ = 1.0;
        return;
    }
    if (magnitude == 0.0 || std::isinf(scale_)) {
        return;
    }
    if (scale_ < magnitude) {
        const double ratio = scale_ / magnitude;
        sum_squares_ = 1.0 + sum_squares_ * ratio * ratio;
        scale_ = magnitude;
    } else {
        const double ratio = magnitude / scale_;
        sum_squares_ += ratio * ratio;
    }
}

double ScaledSumOfSquares::norm() const noexcept {
    if (has_nan_) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    if (scale_ == 0.0 || std::isinf(scale_)) {
        return scale_;
    }
    return scale_ * std::sqrt(sum_squares_);
}

Matrix copy_columns(const Matrix& source, int first_col, int count) {
    if (first_col < 0 || count < 0 || first_col + count > source.cols()) {
        throw std::invalid_argument("matrix columns are out of range");
    }
    Matrix result(source.rows(), count);
    if (result.size() > 0) {
        std::copy(
            source.column_data(first_col),
            source.column_data(first_col) + result.size(),
            result.data());
    }
    return result;
}

void set_block(Matrix& target, int first_row, int first_col, const Matrix& block) {
    if (first_row < 0 || first_col < 0 ||
        first_row + block.rows() > target.rows() ||
        first_col + block.cols() > target.cols()) {
        throw std::invalid_argument("matrix block is out of range");
    }
    for (int col = 0; col < block.cols(); ++col) {
        for (int row = 0; row < block.rows(); ++row) {
            target(first_row + row, first_col + col) = block(row, col);
        }
    }
}

Matrix join_columns(const Matrix& left, const Matrix& right) {
    if (left.rows() != right.rows()) {
        throw std::invalid_argument("matrix row counts do not match");
    }
    Matrix result(left.rows(), left.cols() + right.cols());
    set_block(result, 0, 0, left);
    set_block(result, 0, left.cols(), right);
    return result;
}

double squared_frobenius_norm(const Matrix& matrix) {
    long double sum = 0.0L;
    for (int index = 0; index < matrix.size(); ++index) {
        const long double value = matrix.data()[index];
        sum += value * value;
    }
    return static_cast<double>(sum);
}

double frobenius_norm(const Matrix& matrix) {
    const int count = matrix.size();
    if (count == 0) {
        return 0.0;
    }
    const int stride = 1;
    return dnrm2_(&count, matrix.data(), &stride);
}

std::vector<double> row_norms(const Matrix& matrix) {
    std::vector<ScaledSumOfSquares> accumulators(
        static_cast<std::size_t>(matrix.rows()));
    for (int column = 0; column < matrix.cols(); ++column) {
        const double* values = matrix.column_data(column);
        for (int row = 0; row < matrix.rows(); ++row) {
            accumulators[static_cast<std::size_t>(row)].add(values[row]);
        }
    }
    std::vector<double> norms(static_cast<std::size_t>(matrix.rows()));
    for (int row = 0; row < matrix.rows(); ++row) {
        norms[static_cast<std::size_t>(row)] =
            accumulators[static_cast<std::size_t>(row)].norm();
    }
    return norms;
}

}  // namespace amfls::math
