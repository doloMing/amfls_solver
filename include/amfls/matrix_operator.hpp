#pragma once

#include <limits>

namespace amfls {

// A normwise FP64 model for the fresh scalar validation callbacks. If the
// model is valid, operator_norm_lower_bound is a lower bound for ||A||_2 and
// operator_norm_upper_bound is an upper bound for || |A| ||_2, hence also for
// ||A||_2. The two step counts bound the number of rounded multiply/add
// operations along one output dot product. The common validator combines this
// envelope with outward-
// rounded norm and scalar arithmetic. This contract assumes IEEE-754 binary64
// round-to-nearest arithmetic with gradual underflow; a nonfinite envelope
// disables certification, so no finite certified bound relies on an
// overflowing intermediate. The built-in dense, row-major dense, and CSR
// operators provide such a model. The default deliberately supplies none: a
// custom MatrixOperator remains source compatible but cannot produce certified
// solver success until it overrides validation_error_model() with a valid
// model.
struct MatrixOperatorValidationErrorModel {
    double operator_norm_upper_bound =
        std::numeric_limits<double>::infinity();
    double operator_norm_lower_bound = 0.0;
    long long apply_rounding_steps = 0;
    long long apply_transpose_rounding_steps = 0;
};

class MatrixOperator {
public:
    virtual ~MatrixOperator() = default;

    virtual int rows() const = 0;
    virtual int cols() const = 0;

    // X is cols() by block_cols and Y is rows() by block_cols.
    // Both blocks use column-major storage.
    virtual void apply(const double* x, int block_cols, double* y) const = 0;

    // Y is rows() by block_cols and X is cols() by block_cols.
    virtual void apply_transpose(
        const double* y,
        int block_cols,
        double* x) const = 0;

    // Relative cost used only by the adaptive width scheduler.  The default
    // charges one unit per vector, which is conservative for an operator that
    // exposes no batching model.  Operators whose block callbacks amortize a
    // shared pass may override this without changing solver correctness.
    virtual double relative_block_product_cost(
        int block_cols) const noexcept {
        return block_cols > 0
            ? static_cast<double>(block_cols)
            : std::numeric_limits<double>::infinity();
    }

    virtual MatrixOperatorValidationErrorModel validation_error_model()
        const noexcept {
        return {};
    }

    // Built-in stored operators may provide a validation-only extended
    // accumulator.  The common validator uses this path only when the fast
    // binary64 ridge envelope is dominated by its callback-rounding radius.
    // Search products and the returned solution remain binary64.  Custom
    // operators need not implement these methods.  Overriding support to true
    // promises ordinary long-double multiply/add dot products over immutable
    // stored binary64 entries, with the same lengths and absolute-matrix norm
    // bound declared by validation_error_model().
    virtual bool supports_validation_refinement() const noexcept {
        return false;
    }

    virtual bool apply_validation_refinement(
        const long double* x,
        long double* y) const noexcept {
        (void)x;
        (void)y;
        return false;
    }

    virtual bool apply_transpose_validation_refinement(
        const long double* y,
        long double* x) const noexcept {
        (void)y;
        (void)x;
        return false;
    }
};

}  // namespace amfls
