#pragma once

#include "amfls/matrix_operator.hpp"

namespace amfls::math {

// Non-owning FP64 row-major dense operator. The caller owns values and must
// keep every entry unchanged for the operator's lifetime: validation norm
// bounds are scanned and cached by the constructor.
class RowMajorDenseOperator final : public MatrixOperator {
public:
    RowMajorDenseOperator(int rows, int cols, const double* values);

    int rows() const override;
    int cols() const override;
    void apply(const double* x, int block_cols, double* y) const override;
    void apply_transpose(
        const double* y,
        int block_cols,
        double* x) const override;
    double relative_block_product_cost(
        int block_cols) const noexcept override;
    MatrixOperatorValidationErrorModel validation_error_model()
        const noexcept override;
    bool supports_validation_refinement() const noexcept override;
    bool apply_validation_refinement(
        const long double* x,
        long double* y) const noexcept override;
    bool apply_transpose_validation_refinement(
        const long double* y,
        long double* x) const noexcept override;

    // Internal read-only access for algorithms whose standard explicit-matrix
    // setup performs a single pass over stored entries.
    const double* values() const noexcept;

private:
    int rows_;
    int cols_;
    const double* values_;
    double operator_norm_upper_bound_;
    double operator_norm_lower_bound_;
};

}  // namespace amfls::math
