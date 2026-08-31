#include "algorithms/mathematics/operators/row_major_dense_operator.hpp"

#include <algorithm>
#include <cfenv>
#include <cmath>
#include <stdexcept>
#include <vector>

#include "algorithms/mathematics/floating_point_bounds.hpp"

extern "C" {
void dgemv_(
    const char*, const int*, const int*, const double*, const double*,
    const int*, const double*, const int*, const double*, double*, const int*);
void dgemm_(
    const char*, const char*, const int*, const int*, const int*,
    const double*, const double*, const int*, const double*, const int*,
    const double*, double*, const int*);
}

namespace amfls::math {

RowMajorDenseOperator::RowMajorDenseOperator(
    int rows,
    int cols,
    const double* values)
    : rows_(rows), cols_(cols), values_(values),
      operator_norm_upper_bound_(std::numeric_limits<double>::infinity()),
      operator_norm_lower_bound_(0.0) {
    if (rows <= 0 || cols <= 0 || values == nullptr) {
        throw std::invalid_argument("row-major dense operator input is invalid");
    }
    std::vector<double> column_sums(static_cast<std::size_t>(cols), 0.0);
    std::vector<double> column_square_sums(
        static_cast<std::size_t>(cols), 0.0);
    double maximum_row_sum_upper = 0.0;
    double maximum_two_norm_lower = 0.0;
    double maximum_entry = 0.0;
    double frobenius_squared_upper = 0.0;
    bool certifiable = true;
    for (int row = 0; row < rows; ++row) {
        double row_sum = 0.0;
        double row_square_sum = 0.0;
        for (int column = 0; column < cols; ++column) {
            const double magnitude = std::abs(
                values[row * cols + column]);
            if (!std::isfinite(magnitude)) {
                certifiable = false;
                continue;
            }
            maximum_entry = std::max(maximum_entry, magnitude);
            row_sum += magnitude;
            column_sums[static_cast<std::size_t>(column)] += magnitude;
            row_square_sum = std::fma(
                magnitude, magnitude, row_square_sum);
            column_square_sums[static_cast<std::size_t>(column)] = std::fma(
                magnitude,
                magnitude,
                column_square_sums[static_cast<std::size_t>(column)]);
        }
        maximum_row_sum_upper = std::max(
            maximum_row_sum_upper,
            fp::positive_sum_upper_bound(row_sum, cols));
        maximum_two_norm_lower = std::max(
            maximum_two_norm_lower,
            fp::downward_sqrt(fp::positive_sum_lower_bound(
                row_square_sum, cols)));
    }
    double maximum_column_sum_upper = 0.0;
    for (double column_sum : column_sums) {
        maximum_column_sum_upper = std::max(
            maximum_column_sum_upper,
            fp::positive_sum_upper_bound(column_sum, rows));
    }
    for (double column_square_sum : column_square_sums) {
        maximum_two_norm_lower = std::max(
            maximum_two_norm_lower,
            fp::downward_sqrt(fp::positive_sum_lower_bound(
                column_square_sum, rows)));
        frobenius_squared_upper = fp::upward_add(
            frobenius_squared_upper,
            fp::positive_fma_square_sum_upper_bound(
                column_square_sum, rows));
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

int RowMajorDenseOperator::rows() const { return rows_; }
int RowMajorDenseOperator::cols() const { return cols_; }

double RowMajorDenseOperator::relative_block_product_cost(
    int block_cols) const noexcept {
    return block_cols > 0
        ? 1.0
        : std::numeric_limits<double>::infinity();
}
const double* RowMajorDenseOperator::values() const noexcept { return values_; }

MatrixOperatorValidationErrorModel
RowMajorDenseOperator::validation_error_model() const noexcept {
    return {
        operator_norm_upper_bound_,
        operator_norm_lower_bound_,
        2LL * static_cast<long long>(cols_) + 2LL,
        2LL * static_cast<long long>(rows_) + 2LL};
}

bool RowMajorDenseOperator::supports_validation_refinement() const noexcept {
    return true;
}

bool RowMajorDenseOperator::apply_validation_refinement(
    const long double* x,
    long double* y) const noexcept {
    if (x == nullptr || y == nullptr) {
        return false;
    }
    for (int row = 0; row < rows_; ++row) {
        long double sum = 0.0L;
        for (int column = 0; column < cols_; ++column) {
            sum +=
                static_cast<long double>(values_[row * cols_ + column]) *
                x[column];
        }
        if (!std::isfinite(sum)) {
            return false;
        }
        y[row] = sum;
    }
    return true;
}

bool RowMajorDenseOperator::apply_transpose_validation_refinement(
    const long double* y,
    long double* x) const noexcept {
    if (x == nullptr || y == nullptr) {
        return false;
    }
    std::fill(x, x + cols_, 0.0L);
    for (int row = 0; row < rows_; ++row) {
        const long double input = y[row];
        if (!std::isfinite(input)) {
            return false;
        }
        for (int column = 0; column < cols_; ++column) {
            x[column] +=
                static_cast<long double>(values_[row * cols_ + column]) *
                input;
        }
    }
    return std::all_of(x, x + cols_, [](long double value) {
        return std::isfinite(value);
    });
}

void RowMajorDenseOperator::apply(
    const double* x,
    int block_cols,
    double* y) const {
    if (x == nullptr || y == nullptr || block_cols <= 0) {
        throw std::invalid_argument("row-major dense apply block is invalid");
    }
    // Row-major A is the same memory as column-major A^T.
    const char transpose = 'T';
    const char no_transpose = 'N';
    const double alpha = 1.0;
    const double beta = 0.0;
    if (block_cols == 1) {
        const int increment = 1;
        dgemv_(
            &transpose,
            &cols_,
            &rows_,
            &alpha,
            values_,
            &cols_,
            x,
            &increment,
            &beta,
            y,
            &increment);
        return;
    }
    dgemm_(
        &transpose,
        &no_transpose,
        &rows_,
        &block_cols,
        &cols_,
        &alpha,
        values_,
        &cols_,
        x,
        &cols_,
        &beta,
        y,
        &rows_);
}

void RowMajorDenseOperator::apply_transpose(
    const double* y,
    int block_cols,
    double* x) const {
    if (x == nullptr || y == nullptr || block_cols <= 0) {
        throw std::invalid_argument(
            "row-major dense transpose block is invalid");
    }
    const char no_transpose = 'N';
    const double alpha = 1.0;
    const double beta = 0.0;
    if (block_cols == 1) {
        const int increment = 1;
        dgemv_(
            &no_transpose,
            &cols_,
            &rows_,
            &alpha,
            values_,
            &cols_,
            y,
            &increment,
            &beta,
            x,
            &increment);
        return;
    }
    dgemm_(
        &no_transpose,
        &no_transpose,
        &cols_,
        &block_cols,
        &rows_,
        &alpha,
        values_,
        &cols_,
        y,
        &rows_,
        &beta,
        x,
        &cols_);
}

}  // namespace amfls::math
