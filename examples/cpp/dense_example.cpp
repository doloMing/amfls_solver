#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

#include "amfls/amfls.hpp"

namespace {

class DiagonalOperator final : public amfls::MatrixOperator {
public:
    explicit DiagonalOperator(std::vector<double> diagonal)
        : diagonal_(std::move(diagonal)) {
        for (double value : diagonal_) {
            if (!std::isfinite(value)) {
                throw std::invalid_argument(
                    "diagonal entries must be finite");
            }
            operator_norm_ = std::max(operator_norm_, std::abs(value));
        }
    }

    int rows() const override { return static_cast<int>(diagonal_.size()); }
    int cols() const override { return static_cast<int>(diagonal_.size()); }

    void apply(const double* x, int block_cols, double* y) const override {
        for (int col = 0; col < block_cols; ++col) {
            for (int row = 0; row < rows(); ++row) {
                y[row + col * rows()] = diagonal_[row] * x[row + col * rows()];
            }
        }
    }

    void apply_transpose(
        const double* y,
        int block_cols,
        double* x) const override {
        apply(y, block_cols, x);
    }

    amfls::MatrixOperatorValidationErrorModel validation_error_model()
        const noexcept override {
        return {operator_norm_, operator_norm_, 1, 1};
    }

private:
    // This operator owns the diagonal and never mutates it, so the cached
    // exact spectral norm remains valid for every callback.
    std::vector<double> diagonal_;
    double operator_norm_ = 0.0;
};

}  // namespace

int main() {
    DiagonalOperator matrix({100.0, 3.0, 2.0, 1.0});
    const std::vector<double> b{100.0, 6.0, 6.0, 4.0};
    amfls::AmflsOptions options;
    options.tolerance = 1e-10;
    options.failure_probability = 1e-6;
    options.seed = 7;

    const amfls::LeastSquaresResult result =
        amfls::solve_amfls(matrix, b.data(), options);
    std::cout << "status=" << static_cast<int>(result.status)
              << " backward_error_upper=" << result.backward_error_upper_bound
              << " basis_rank=" << result.basis_rank << '\n';
    for (double value : result.solution) {
        std::cout << value << ' ';
    }
    std::cout << '\n';
    return result.status == amfls::SolverStatus::success ? 0 : 1;
}
