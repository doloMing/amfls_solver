#include "algorithms/mathematics/operators/sparse_csr_operator.hpp"

#include <algorithm>
#include <cfenv>
#include <cstddef>
#include <cmath>
#include <stdexcept>
#include <vector>

#include "algorithms/mathematics/floating_point_bounds.hpp"

namespace amfls::math {

SparseCsrOperator::SparseCsrOperator(
    int rows,
    int cols,
    const std::int64_t* row_offsets,
    const std::int64_t* column_indices,
    const double* values,
    std::int64_t nonzeros)
    : rows_(rows),
      cols_(cols),
      row_offsets_(row_offsets),
      column_indices_(column_indices),
      values_(values),
      nonzeros_(nonzeros),
      maximum_row_nonzeros_(0),
      maximum_column_nonzeros_(0),
      operator_norm_upper_bound_(std::numeric_limits<double>::infinity()),
      operator_norm_lower_bound_(0.0) {
    if (rows <= 0 || cols <= 0 || row_offsets == nullptr || nonzeros < 0) {
        throw std::invalid_argument("CSR operator dimensions are invalid");
    }
    if (nonzeros > 0 && (column_indices == nullptr || values == nullptr)) {
        throw std::invalid_argument("CSR operator arrays are missing");
    }
    if (row_offsets[0] != 0 || row_offsets[rows] != nonzeros) {
        throw std::invalid_argument("CSR row offsets do not span the values");
    }
    std::vector<std::int64_t> column_nonzeros(
        static_cast<std::size_t>(cols), 0);
    std::vector<double> column_sums(static_cast<std::size_t>(cols), 0.0);
    std::vector<double> column_square_sums(
        static_cast<std::size_t>(cols), 0.0);
    std::vector<int> row_stamp(static_cast<std::size_t>(cols), -1);
    double maximum_row_sum_upper = 0.0;
    double maximum_two_norm_lower = 0.0;
    double maximum_entry = 0.0;
    double frobenius_squared_upper = 0.0;
    for (int row = 0; row < rows; ++row) {
        if (row_offsets[row] > row_offsets[row + 1]) {
            throw std::invalid_argument("CSR row offsets must be nondecreasing");
        }
        const std::int64_t row_nonzeros =
            row_offsets[row + 1] - row_offsets[row];
        maximum_row_nonzeros_ = std::max(
            maximum_row_nonzeros_, row_nonzeros);
        double row_sum = 0.0;
        double row_square_sum = 0.0;
        for (std::int64_t entry = row_offsets[row];
             entry < row_offsets[row + 1];
             ++entry) {
            if (column_indices[entry] < 0 ||
                column_indices[entry] >= cols) {
                throw std::invalid_argument(
                    "CSR column index is out of range");
            }
            if (!std::isfinite(values[entry])) {
                throw std::invalid_argument("CSR values must be finite");
            }
            const std::size_t column = static_cast<std::size_t>(
                column_indices[entry]);
            if (row_stamp[column] == row) {
                throw std::invalid_argument(
                    "CSR rows must not contain duplicate columns");
            }
            row_stamp[column] = row;
            ++column_nonzeros[column];
            maximum_column_nonzeros_ = std::max(
                maximum_column_nonzeros_, column_nonzeros[column]);
            const double magnitude = std::abs(values[entry]);
            maximum_entry = std::max(maximum_entry, magnitude);
            row_sum += magnitude;
            column_sums[column] += magnitude;
            row_square_sum = std::fma(
                magnitude, magnitude, row_square_sum);
            column_square_sums[column] = std::fma(
                magnitude, magnitude, column_square_sums[column]);
        }
        maximum_row_sum_upper = std::max(
            maximum_row_sum_upper,
            fp::positive_sum_upper_bound(row_sum, row_nonzeros));
        maximum_two_norm_lower = std::max(
            maximum_two_norm_lower,
            fp::downward_sqrt(fp::positive_sum_lower_bound(
                row_square_sum, row_nonzeros)));
    }
    double maximum_column_sum_upper = 0.0;
    for (int column = 0; column < cols_; ++column) {
        const std::size_t index = static_cast<std::size_t>(column);
        maximum_column_sum_upper = std::max(
            maximum_column_sum_upper,
            fp::positive_sum_upper_bound(
                column_sums[index], column_nonzeros[index]));
        maximum_two_norm_lower = std::max(
            maximum_two_norm_lower,
            fp::downward_sqrt(fp::positive_sum_lower_bound(
                column_square_sums[index], column_nonzeros[index])));
        frobenius_squared_upper = fp::upward_add(
            frobenius_squared_upper,
            fp::positive_fma_square_sum_upper_bound(
                column_square_sums[index], column_nonzeros[index]));
    }
    if (std::fegetround() == FE_TONEAREST &&
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

int SparseCsrOperator::rows() const { return rows_; }
int SparseCsrOperator::cols() const { return cols_; }
const std::int64_t* SparseCsrOperator::row_offsets() const noexcept {
    return row_offsets_;
}
const std::int64_t* SparseCsrOperator::column_indices() const noexcept {
    return column_indices_;
}
const double* SparseCsrOperator::values() const noexcept { return values_; }

MatrixOperatorValidationErrorModel SparseCsrOperator::validation_error_model()
    const noexcept {
    return {
        operator_norm_upper_bound_,
        operator_norm_lower_bound_,
        2LL * maximum_row_nonzeros_ + 2LL,
        2LL * maximum_column_nonzeros_ + 2LL};
}

bool SparseCsrOperator::supports_validation_refinement() const noexcept {
    return true;
}

bool SparseCsrOperator::apply_validation_refinement(
    const long double* x,
    long double* y) const noexcept {
    if (x == nullptr || y == nullptr) {
        return false;
    }
    for (int row = 0; row < rows_; ++row) {
        long double sum = 0.0L;
        for (std::int64_t entry = row_offsets_[row];
             entry < row_offsets_[row + 1];
             ++entry) {
            sum += static_cast<long double>(values_[entry]) *
                x[static_cast<std::size_t>(column_indices_[entry])];
        }
        if (!std::isfinite(sum)) {
            return false;
        }
        y[row] = sum;
    }
    return true;
}

bool SparseCsrOperator::apply_transpose_validation_refinement(
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
        for (std::int64_t entry = row_offsets_[row];
             entry < row_offsets_[row + 1];
             ++entry) {
            const std::size_t column = static_cast<std::size_t>(
                column_indices_[entry]);
            x[column] +=
                static_cast<long double>(values_[entry]) * input;
        }
    }
    return std::all_of(x, x + cols_, [](long double value) {
        return std::isfinite(value);
    });
}

void SparseCsrOperator::apply(
    const double* x,
    int block_cols,
    double* y) const {
    if (x == nullptr || y == nullptr || block_cols <= 0) {
        throw std::invalid_argument("CSR apply block is invalid");
    }
    const std::size_t output_size =
        static_cast<std::size_t>(rows_) * static_cast<std::size_t>(block_cols);
    std::fill(y, y + output_size, 0.0);

    for (int row = 0; row < rows_; ++row) {
        for (std::int64_t entry = row_offsets_[row];
             entry < row_offsets_[row + 1];
             ++entry) {
            const std::size_t col =
                static_cast<std::size_t>(column_indices_[entry]);
            const double matrix_value = values_[entry];
            for (int block = 0; block < block_cols; ++block) {
                const std::size_t x_index =
                    static_cast<std::size_t>(block) *
                        static_cast<std::size_t>(cols_) +
                    col;
                const std::size_t y_index =
                    static_cast<std::size_t>(block) *
                        static_cast<std::size_t>(rows_) +
                    static_cast<std::size_t>(row);
                y[y_index] =
                    std::fma(matrix_value, x[x_index], y[y_index]);
            }
        }
    }
}

void SparseCsrOperator::apply_transpose(
    const double* y,
    int block_cols,
    double* x) const {
    if (x == nullptr || y == nullptr || block_cols <= 0) {
        throw std::invalid_argument("CSR transpose block is invalid");
    }
    const std::size_t output_size =
        static_cast<std::size_t>(cols_) * static_cast<std::size_t>(block_cols);
    std::fill(x, x + output_size, 0.0);

    for (int row = 0; row < rows_; ++row) {
        for (std::int64_t entry = row_offsets_[row];
             entry < row_offsets_[row + 1];
             ++entry) {
            const std::size_t col =
                static_cast<std::size_t>(column_indices_[entry]);
            const double matrix_value = values_[entry];
            for (int block = 0; block < block_cols; ++block) {
                const std::size_t y_index =
                    static_cast<std::size_t>(block) *
                        static_cast<std::size_t>(rows_) +
                    static_cast<std::size_t>(row);
                const std::size_t x_index =
                    static_cast<std::size_t>(block) *
                        static_cast<std::size_t>(cols_) +
                    col;
                x[x_index] =
                    std::fma(matrix_value, y[y_index], x[x_index]);
            }
        }
    }
}

}  // namespace amfls::math
