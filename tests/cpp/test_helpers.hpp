#pragma once

#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "amfls/matrix_operator.hpp"
#include "algorithms/mathematics/floating_point_bounds.hpp"

inline void require_test(bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

inline void require_near(
    double actual,
    double expected,
    double tolerance,
    const std::string& message) {
    if (!std::isfinite(actual) || std::abs(actual - expected) > tolerance) {
        throw std::runtime_error(
            message + ": actual=" + std::to_string(actual) +
            " expected=" + std::to_string(expected));
    }
}

class DenseTestOperator final : public amfls::MatrixOperator {
public:
    DenseTestOperator(int rows, int cols, std::vector<double> column_major)
        : rows_(rows), cols_(cols), values_(std::move(column_major)) {
        require_test(
            static_cast<int>(values_.size()) == rows * cols,
            "dense test matrix has the wrong size");
        operator_norm_upper_bound_ =
            amfls::math::fp::frobenius_norm_upper_bound(values_);
        for (double value : values_) {
            operator_norm_lower_bound_ = std::max(
                operator_norm_lower_bound_, std::abs(value));
        }
    }

    int rows() const override { return rows_; }
    int cols() const override { return cols_; }

    void apply(const double* x, int block_cols, double* y) const override {
        for (int col = 0; col < block_cols; ++col) {
            for (int row = 0; row < rows_; ++row) {
                long double sum = 0.0L;
                for (int inner = 0; inner < cols_; ++inner) {
                    sum += values_[row + inner * rows_] *
                           x[inner + col * cols_];
                }
                y[row + col * rows_] = static_cast<double>(sum);
            }
        }
    }

    void apply_transpose(
        const double* y,
        int block_cols,
        double* x) const override {
        for (int col = 0; col < block_cols; ++col) {
            for (int row = 0; row < cols_; ++row) {
                long double sum = 0.0L;
                for (int inner = 0; inner < rows_; ++inner) {
                    sum += values_[inner + row * rows_] *
                           y[inner + col * rows_];
                }
                x[row + col * cols_] = static_cast<double>(sum);
            }
        }
    }

    amfls::MatrixOperatorValidationErrorModel validation_error_model()
        const noexcept override {
        return {
            operator_norm_upper_bound_,
            operator_norm_lower_bound_,
            2LL * static_cast<long long>(cols_) + 2LL,
            2LL * static_cast<long long>(rows_) + 2LL};
    }

private:
    int rows_;
    int cols_;
    std::vector<double> values_;
    double operator_norm_upper_bound_ =
        std::numeric_limits<double>::infinity();
    double operator_norm_lower_bound_ = 0.0;
};
