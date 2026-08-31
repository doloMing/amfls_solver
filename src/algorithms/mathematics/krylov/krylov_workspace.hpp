#pragma once

#include <vector>

#include "amfls/least_squares_result.hpp"
#include "amfls/matrix_operator.hpp"
#include "algorithms/mathematics/krylov/candidate_validation.hpp"
#include "algorithms/mathematics/linalg/matrix.hpp"

namespace amfls::math {

struct ProjectedCandidate {
    std::vector<double> solution;
    int basis_rank = 0;
};

struct BlockRange {
    int first = 0;
    int width = 0;
};

struct BlockGolubKahanStepAudit {
    int requested_injected_columns = 0;
    int retained_injected_columns = 0;
    int left_input_width = 0;
    int retained_right_width = 0;
    int retained_next_left_width = 0;
    int basis_rank_before = 0;
    int basis_rank_after = 0;
    double right_rank_cutoff = 0.0;
    double right_discarded_residual_norm = 0.0;
    double right_active_discarded_residual_norm = 0.0;
    double left_rank_cutoff = 0.0;
    double left_discarded_residual_norm = 0.0;
    bool right_rank_loss = false;
    bool left_rank_loss = false;
    bool right_positive_finite_below_cutoff = false;
    bool left_positive_finite_below_cutoff = false;
    bool right_nonfinite_residual = false;
    bool left_nonfinite_residual = false;
    bool right_basis_cap_truncation = false;
    bool left_basis_cap_truncation = false;
};

// A positive finite direction was removed by the numerical rank cutoff even
// though the configured basis capacity could have retained it.  A
// rolling-left process must not consume another frontier after this event,
// because the discarded direction is not represented in the retained V
// space.
bool step_has_numerical_rank_truncation(
    const BlockGolubKahanStepAudit& step) noexcept;

bool step_has_nonfinite_breakdown(
    const BlockGolubKahanStepAudit& step) noexcept;

struct KrylovOrthogonalizationAudit {
    double projected_residual_norm = 0.0;
    double rank_cutoff = 0.0;
    int retained_columns = 0;
    bool positive_finite_below_cutoff = false;
    bool nonfinite_residual = false;
};

Matrix orthogonalize_krylov_block(
    Matrix block,
    const Matrix& basis,
    int basis_limit,
    int operator_rows,
    int operator_cols,
    RunStatistics& statistics,
    std::vector<double>* coefficient_workspace = nullptr,
    KrylovOrthogonalizationAudit* audit = nullptr);

ProjectedCandidate solve_projected_basis(
    const double* right_hand_side,
    double regularization,
    const Matrix& basis,
    const Matrix& applied_basis,
    RunStatistics& statistics);

LeastSquaresResult evaluate_projected_basis(
    const MatrixOperator& validation_matrix,
    const double* right_hand_side,
    double regularization,
    double tolerance,
    const Matrix& basis,
    const Matrix& applied_basis,
    RunStatistics& statistics,
    int epoch,
    int depth,
    int auxiliary_width);

// One variable-width block Golub--Kahan process.  `initialize` constructs
// the first left block from b and records an optional auxiliary block as a
// locally stabilized pending P*Omega, but deliberately performs no operator
// callback.  Each
// successful `advance_level` applies A* once to the current left frontier and
// pending P*Omega in one block call, creates a right block, and then applies A
// once to create the replacement left frontier.  Pending P*Omega has the same
// quotient-space effect as the injection lemma but is not assigned a
// conceptual U coordinate.
//
// V is retained in full and every right residual is reorthogonalized against
// it. Only the active, not-yet-consumed block of U is stored physically.
// Historical conceptual U
// coordinates remain represented by T and g=U'*b.  Exactly,
// A*U_consumed subset span(V) makes U_consumed'*A V_new zero, so forming the
// next left frontier requires projection against only the active U block.
// Candidate construction extracts a compact rectangular T from geometrically
// grown leading-dimension storage; original-problem validation remains
// independent.
class BlockGolubKahanWorkspace {
public:
    BlockGolubKahanWorkspace(
        const MatrixOperator& matrix,
        const double* right_hand_side,
        double regularization,
        int maximum_basis_size,
        RunStatistics& statistics);

    void initialize(Matrix initial_auxiliary_left);
    int append_left_seed(Matrix left_seed);
    bool advance_level();

    LeastSquaresResult evaluate(
        double tolerance,
        int epoch,
        int auxiliary_width);

    int depth() const noexcept;
    int basis_rank() const noexcept;
    int basis_limit() const noexcept;
    int active_left_width() const noexcept;
    bool can_advance() const noexcept;
    double right_hand_side_norm() const noexcept;
    const Matrix& right_basis() const noexcept;
    // The physically retained, unconsumed left frontier only.  Historical
    // conceptual U columns are represented by projected_operator()/g.
    const Matrix& left_basis() const noexcept;
    // Return logical compact T.  Internal leading-dimension capacity grows
    // geometrically, so T is copied only on logarithmically many row growths.
    Matrix projected_operator() const;
    const std::vector<double>& projected_rhs() const noexcept;
    const BlockGolubKahanStepAudit& last_step_audit() const noexcept;

private:
    void apply_search(const Matrix& input, Matrix& output);
    void apply_transpose_search(const Matrix& input, Matrix& output);
    void append_projected_rhs(const Matrix& left_block);
    void ensure_projected_row_capacity(int required_rows);
    Matrix compact_projected_operator() const;
    Matrix compact_projected_prefix(int rows, int cols) const;

    const MatrixOperator& matrix_;
    Matrix right_hand_side_;
    double right_hand_side_norm_ = 0.0;
    double regularization_;
    int basis_limit_;
    Matrix left_basis_;
    Matrix pending_left_seed_;
    Matrix right_basis_;
    // Physical rows are a geometrically grown leading-dimension capacity;
    // projected_rhs_.size() is the logical row count.  Physical columns are
    // exactly the current logical T columns.
    Matrix projected_operator_;
    std::vector<double> projected_rhs_;
    // `first` is a conceptual T/g row; `left_basis_` contains exactly
    // `width` physical columns and no preceding U history.
    BlockRange active_left_;
    BlockRange previous_right_;
    Matrix frontier_coupling_;
    // Scalar residual storage is recycled after its normalized column has
    // been consumed.  Operator callbacks overwrite every entry, so reuse
    // removes allocation/zero-fill work without changing arithmetic.
    Matrix scalar_right_residual_buffer_;
    Matrix scalar_left_residual_buffer_;
    // Reused by scalar DGKS projections.  Its size only grows with the
    // largest basis seen by this workspace, avoiding one allocation per
    // half-step without changing the projection arithmetic.
    std::vector<double> scalar_projection_coefficients_;
    std::vector<double> scalar_accumulated_coefficients_;
    int pending_requested_columns_ = 0;
    int pending_retained_columns_ = 0;
    int depth_ = 0;
    bool initialized_ = false;
    bool continuation_reliable_ = true;
    RunStatistics& statistics_;
    BlockGolubKahanStepAudit last_step_audit_;
};

IterationRecord make_iteration_record(
    const LeastSquaresResult& result,
    int epoch,
    int depth,
    int auxiliary_width);

}  // namespace amfls::math
