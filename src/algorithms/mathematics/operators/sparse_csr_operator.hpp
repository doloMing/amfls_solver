#pragma once

#include <cstdint>

#include "amfls/matrix_operator.hpp"

namespace amfls::math {

// Non-owning FP64 compressed-sparse-row operator.  The caller owns all three
// CSR arrays and must keep the structure and values unchanged for the
// operator's lifetime: validation norm bounds and sparsity counts are scanned
// and cached by the constructor.
class SparseCsrOperator final : public MatrixOperator {
public:
    SparseCsrOperator(
        int rows,
        int cols,
        const std::int64_t* row_offsets,
        const std::int64_t* column_indices,
        const double* values,
        std::int64_t nonzeros);

    int rows() const override;
    int cols() const override;
    void apply(const double* x, int block_cols, double* y) const override;
    void apply_transpose(
        const double* y,
        int block_cols,
        double* x) const override;
    MatrixOperatorValidationErrorModel validation_error_model()
        const noexcept override;
    bool supports_validation_refinement() const noexcept override;
    bool apply_validation_refinement(
        const long double* x,
        long double* y) const noexcept override;
    bool apply_transpose_validation_refinement(
        const long double* y,
        long double* x) const noexcept override;

    // Internal read-only CSR access used by algorithms whose standard sparse
    // formulation operates on matrix entries rather than black-box products.
    // The returned arrays retain the non-owning lifetime of this operator.
    const std::int64_t* row_offsets() const noexcept;
    const std::int64_t* column_indices() const noexcept;
    const double* values() const noexcept;

private:
    int rows_;
    int cols_;
    const std::int64_t* row_offsets_;
    const std::int64_t* column_indices_;
    const double* values_;
    std::int64_t nonzeros_;
    std::int64_t maximum_row_nonzeros_;
    std::int64_t maximum_column_nonzeros_;
    double operator_norm_upper_bound_;
    double operator_norm_lower_bound_;
};

}  // namespace amfls::math
