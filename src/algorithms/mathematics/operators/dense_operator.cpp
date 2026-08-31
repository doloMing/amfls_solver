#include "algorithms/mathematics/operators/dense_operator.hpp"

#include <algorithm>
#include <cfenv>
#include <cmath>
#include <stdexcept>
#include <vector>

#include "algorithms/mathematics/floating_point_bounds.hpp"

extern "C" {
void dgemm_(
    const char*, const char*, const int*, const int*, const int*,
    const double*, const double*, const int*, const double*, const int*,
    const double*, double*, const int*);
}

namespace amfls::math {

DenseOperator::DenseOperator(int rows, int cols, const double* values)
    : rows_(rows), cols_(cols), values_(values),
      operator_norm_upper_bound_(std::numeric_limits<double>::infinity()),
      operator_norm_lower_bound_(0.0) {
    if (rows <= 0 || cols <= 0 || values == nullptr) {
        throw std::invalid_argument("dense operator input is invalid");
    }
    std::vector<double> row_sums(static_cast<std::size_t>(rows), 0.0);
    std::vector<double> row_square_sums(
        static_cast<std::size_t>(rows), 0.0);
    double maximum_column_sum_upper = 0.0;
    double maximum_two_norm_lower = 0.0;
    double maximum_entry = 0.0;
    double frobenius_squared_upper = 0.0;
    bool certifiable = true;
    for (int column = 0; column < cols; ++column) {
        double column_sum = 0.0;
        double column_square_sum = 0.0;
        for (int row = 0; row < rows; ++row) {
            const double magnitude = std::abs(
                values[row + column * rows]);
            if (!std::isfinite(magnitude)) {
                certifiable = false;
                continue;
            }
            maximum_entry = std::max(maximum_entry, magnitude);
            row_sums[static_cast<std::size_t>(row)] += magnitude;
            row_square_sums[static_cast<std::size_t>(row)] = std::fma(
                magnitude,
                magnitude,
                row_square_sums[static_cast<std::size_t>(row)]);
            column_sum += magnitude;
            column_square_sum = std::fma(
                magnitude, magnitude, column_square_sum);
        }
        maximum_column_sum_upper = std::max(
            maximum_column_sum_upper,
            fp::positive_sum_upper_bound(column_sum, rows));
        maximum_two_norm_lower = std::max(
            maximum_two_norm_lower,
            fp::downward_sqrt(fp::positive_sum_lower_bound(
                column_square_sum, rows)));
        frobenius_squared_upper = fp::upward_add(
            frobenius_squared_upper,
            fp::positive_fma_square_sum_upper_bound(
                column_square_sum, rows));
    }
    double maximum_row_sum_upper = 0.0;
    for (double row_sum : row_sums) {
        maximum_row_sum_upper = std::max(
            maximum_row_sum_upper,
            fp::positive_sum_upper_bound(row_sum, cols));
    }
    for (double row_square_sum : row_square_sums) {
        maximum_two_norm_lower = std::max(
            maximum_two_norm_lower,
            fp::downward_sqrt(fp::positive_sum_lower_bound(
                row_square_sum, cols)));
    }
    if (certifiable && std::fegetround() == FE_TONEAREST &&
        fp::gradual_underflow_is_active()) {
        const double norm_one_infinity_upper = fp::upward_multiply(
            fp::upward_sqrt(maximum_column_sum_upper),
            fp::upward_sqrt(maximum_row_sum_upper));
        operator_norm_upper_bound_ = std::min(
            norm_one_infinity_upper,
            fp::upward_sqrt(frobenius_squared_upper));
        operator_norm_lower_bound_ = std::max(
            maximum_entry, maximum_two_norm_lower);
    }
    if (!std::isfinite(operator_norm_upper_bound_) ||
        operator_norm_lower_bound_ > operator_norm_upper_bound_) {
        operator_norm_upper_bound_ = std::numeric_limits<double>::infinity();
        operator_norm_lower_bound_ = 0.0;
    }
}

int DenseOperator::rows() const { return rows_; }
int DenseOperator::cols() const { return cols_; }

double DenseOperator::relative_block_product_cost(
    int block_cols) const noexcept {
    return block_cols > 0
        ? 1.0
        : std::numeric_limits<double>::infinity();
}
const double* DenseOperator::values() const noexcept { return values_; }

MatrixOperatorValidationErrorModel DenseOperator::validation_error_model()
    const noexcept {
    return {
        operator_norm_upper_bound_,
        operator_norm_lower_bound_,
        2LL * static_cast<long long>(cols_) + 2LL,
        2LL * static_cast<long long>(rows_) + 2LL};
}

bool DenseOperator::supports_validation_refinement() const noexcept {
    return true;
}

bool DenseOperator::apply_validation_refinement(
    const long double* x,
    long double* y) const noexcept {
    if (x == nullptr || y == nullptr) {
        return false;
    }
    std::fill(y, y + rows_, 0.0L);
    for (int column = 0; column < cols_; ++column) {
        const long double input = x[column];
        if (!std::isfinite(input)) {
            return false;
        }
        for (int row = 0; row < rows_; ++row) {
            y[row] +=
                static_cast<long double>(values_[row + column * rows_]) *
                input;
        }
    }
    return std::all_of(y, y + rows_, [](long double value) {
        return std::isfinite(value);
    });
}

bool DenseOperator::apply_transpose_validation_refinement(
    const long double* y,
    long double* x) const noexcept {
    if (x == nullptr || y == nullptr) {
        return false;
    }
    for (int column = 0; column < cols_; ++column) {
        long double sum = 0.0L;
        for (int row = 0; row < rows_; ++row) {
            sum +=
                static_cast<long double>(values_[row + column * rows_]) *
                y[row];
        }
        if (!std::isfinite(sum)) {
            return false;
        }
        x[column] = sum;
    }
    return true;
}

void DenseOperator::apply(const double* x, int block_cols, double* y) const {
    const char no_transpose = 'N';
    const double alpha = 1.0;
    const double beta = 0.0;
    dgemm_(
        &no_transpose, &no_transpose, &rows_, &block_cols, &cols_,
        &alpha, values_, &rows_, x, &cols_, &beta, y, &rows_);
}

void DenseOperator::apply_transpose(
    const double* y,
    int block_cols,
    double* x) const {
    const char transpose = 'T';
    const char no_transpose = 'N';
    const double alpha = 1.0;
    const double beta = 0.0;
    dgemm_(
        &transpose, &no_transpose, &cols_, &block_cols, &rows_,
        &alpha, values_, &rows_, y, &rows_, &beta, x, &cols_);
}

}  // namespace amfls::math
