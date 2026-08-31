#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

#include "algorithms/mathematics/krylov/krylov_workspace.hpp"
#include "algorithms/mathematics/linalg/blas_lapack.hpp"
#include "algorithms/mathematics/linalg/svd.hpp"
#include "algorithms/mathematics/random/gaussian.hpp"
#include "test_helpers.hpp"

namespace {

using amfls::math::Matrix;
using amfls::math::BlockGolubKahanWorkspace;

class NonfiniteLeftHalfOperator final : public amfls::MatrixOperator {
public:
    int rows() const override { return 1; }
    int cols() const override { return 1; }

    void apply(const double*, int block_cols, double* output) const override {
        for (int col = 0; col < block_cols; ++col) {
            output[col] = std::numeric_limits<double>::infinity();
        }
    }

    void apply_transpose(
        const double* input,
        int block_cols,
        double* output) const override {
        for (int col = 0; col < block_cols; ++col) {
            output[col] = input[col];
        }
    }
};

double relative_matrix_difference(
    const Matrix& left,
    const Matrix& right) {
    require_test(
        left.rows() == right.rows() && left.cols() == right.cols(),
        "matrix difference dimensions");
    long double difference_squared = 0.0L;
    long double reference_squared = 0.0L;
    for (int index = 0; index < left.size(); ++index) {
        const long double difference =
            static_cast<long double>(left.data()[index]) -
            right.data()[index];
        difference_squared += difference * difference;
        const long double reference = left.data()[index];
        reference_squared += reference * reference;
    }
    if (reference_squared == 0.0L) {
        return std::sqrt(static_cast<double>(difference_squared));
    }
    return std::sqrt(static_cast<double>(
        difference_squared / reference_squared));
}

void require_orthonormal(
    const Matrix& basis,
    double tolerance,
    const std::string& label) {
    Matrix gram = amfls::math::transpose_multiply(basis, basis);
    for (int col = 0; col < gram.cols(); ++col) {
        for (int row = 0; row < gram.rows(); ++row) {
            require_near(
                gram(row, col),
                row == col ? 1.0 : 0.0,
                tolerance,
                label);
        }
    }
}

void test_variable_block_gkb_projection_and_injection() {
    constexpr int rows = 6;
    constexpr int cols = 4;
    std::vector<double> values(rows * cols, 0.0);
    values[0 + 0 * rows] = 4.0;
    values[1 + 1 * rows] = 3.0;
    values[2 + 2 * rows] = 2.0;
    values[3 + 3 * rows] = 1.0;
    DenseTestOperator matrix(rows, cols, std::move(values));
    const std::vector<double> right_hand_side{
        1.0, -2.0, 3.0, -4.0, 0.5, -0.25};

    amfls::RunStatistics statistics;
    BlockGolubKahanWorkspace workspace(
        matrix, right_hand_side.data(), 0.0, cols, statistics);
    workspace.initialize(Matrix(rows, 0));
    require_test(
        workspace.active_left_width() == 1 && workspace.depth() == 0,
        "GKB initialization must leave one unconsumed b frontier");
    require_test(
        workspace.advance_level() && workspace.depth() == 1 &&
            workspace.basis_rank() == 1,
        "first scalar GKB level");

    const Matrix consumed_active = workspace.left_basis();
    const Matrix projected_before_injection =
        workspace.projected_operator();
    const std::vector<double> rhs_before_injection =
        workspace.projected_rhs();
    require_test(
        consumed_active.cols() == workspace.active_left_width() &&
            projected_before_injection.rows() > consumed_active.cols(),
        "only the active U frontier is retained physically");
    require_test(
        workspace.append_left_seed(consumed_active) == 0 &&
            workspace.last_step_audit().requested_injected_columns == 1 &&
            workspace.last_step_audit().retained_injected_columns == 0,
        "pending Omega is rank-revealed against active U locally");

    Matrix injected(rows, 2);
    injected(0, 0) = 1.0;
    injected(1, 0) = 1.0;
    injected(4, 0) = 2.0;
    injected(2, 1) = -1.0;
    injected(3, 1) = 0.5;
    injected(5, 1) = 3.0;
    const int retained = workspace.append_left_seed(injected);
    require_test(
        retained == 2 &&
            workspace.active_left_width() == consumed_active.cols() &&
            workspace.left_basis().cols() == consumed_active.cols(),
        "injection must remain pending instead of widening conceptual U");
    require_test(
        relative_matrix_difference(
            projected_before_injection,
            workspace.projected_operator()) == 0.0 &&
            workspace.projected_rhs() == rhs_before_injection,
        "pending Omega must not create T rows or projected RHS entries");
    const Matrix right_before_advance = workspace.right_basis();
    const long long at_columns_before = statistics.search_at_columns;
    require_test(
        workspace.advance_level() && workspace.depth() == 2,
        "widened GKB level");
    const Matrix& active_u = workspace.left_basis();
    const Matrix& v = workspace.right_basis();
    const Matrix projected = workspace.projected_operator();
    require_orthonormal(active_u, 2e-12, "active GKB left orthogonality");
    require_orthonormal(v, 2e-12, "GKB right orthogonality");
    require_test(
        active_u.cols() == workspace.active_left_width() &&
            projected.rows() ==
                static_cast<int>(workspace.projected_rhs().size()) &&
            projected.rows() ==
                projected_before_injection.rows() + active_u.cols() &&
            projected.cols() == v.cols(),
        "compact T must retain conceptual rows while U storage rolls");
    require_test(
        workspace.last_step_audit().left_input_width ==
                consumed_active.cols() + retained &&
            workspace.last_step_audit().requested_injected_columns == 3 &&
            workspace.last_step_audit().retained_injected_columns == 2 &&
            statistics.search_at_columns ==
                at_columns_before + consumed_active.cols() + retained,
        "active U and stabilized pending Omega share one A* block call");

    // Once U_active is consumed, A*U_active belongs to the retained V space.
    // This is the exact relation that makes all older U projections of
    // A*V_new vanish and permits rolling storage.
    Matrix transposed_consumed(cols, consumed_active.cols());
    matrix.apply_transpose(
        consumed_active.data(),
        consumed_active.cols(),
        transposed_consumed.data());
    for (int pass = 0; pass < 2; ++pass) {
        Matrix coefficients = amfls::math::transpose_multiply(
            v, transposed_consumed);
        amfls::math::gemm(
            v,
            false,
            coefficients,
            false,
            -1.0,
            1.0,
            transposed_consumed);
    }
    require_test(
        amfls::math::frobenius_norm(transposed_consumed) < 3e-12,
        "consumed U must map into the retained V space");

    const int new_right_first = right_before_advance.cols();
    const int new_right_width =
        workspace.last_step_audit().retained_right_width;
    const int consumed_left_first =
        projected_before_injection.rows() - consumed_active.cols();
    const int new_left_first = projected_before_injection.rows();
    Matrix new_right = amfls::math::copy_columns(
        v, new_right_first, new_right_width);
    Matrix applied_new_right(rows, new_right_width);
    matrix.apply(
        new_right.data(), new_right.cols(), applied_new_right.data());
    Matrix locally_represented(rows, new_right_width);
    for (int col = 0; col < new_right_width; ++col) {
        for (int local = 0; local < consumed_active.cols(); ++local) {
            const double coefficient = projected(
                consumed_left_first + local,
                new_right_first + col);
            for (int row = 0; row < rows; ++row) {
                locally_represented(row, col) +=
                    consumed_active(row, local) * coefficient;
            }
        }
        for (int local = 0; local < active_u.cols(); ++local) {
            const double coefficient = projected(
                new_left_first + local,
                new_right_first + col);
            for (int row = 0; row < rows; ++row) {
                locally_represented(row, col) +=
                    active_u(row, local) * coefficient;
            }
        }
        for (int historical_row = 0;
             historical_row < consumed_left_first;
             ++historical_row) {
            require_near(
                projected(historical_row, new_right_first + col),
                0.0,
                2e-13,
                "new V has zero coefficients on consumed historical U");
        }
    }
    require_test(
        relative_matrix_difference(
            applied_new_right, locally_represented) < 3e-12,
        "A V_new needs only consumed-active and replacement-active U");
    for (int col = 0; col < active_u.cols(); ++col) {
        require_near(
            workspace.projected_rhs()[new_left_first + col],
            amfls::math::vector_dot(
                active_u.column_data(col), right_hand_side.data(), rows),
            2e-13,
            "replacement-active projected RHS U^T b");
    }
    require_test(
        statistics.search_a_block_calls == 2 &&
            statistics.search_at_block_calls == 2,
        "one A and one A* block call per GKB level");
    const auto candidate = workspace.evaluate(1e-12, 0, retained);
    require_test(
        candidate.operator_norm_lower_bound > 0.0,
        "ordinary validation constructs a certified norm lower bound");
    require_test(
        candidate.operator_norm_lower_bound <=
            4.0 * (1.0 + 8.0 * std::numeric_limits<double>::epsilon()),
        "certified lower bound must not exceed the explicit operator norm");
}

void test_gkb_wide_minimum_norm_and_ridge_projection() {
    constexpr int rows = 2;
    constexpr int cols = 3;
    const std::vector<double> values{
        2.0, 0.0,
        0.0, 1.0,
        0.0, 0.0};
    DenseTestOperator matrix(rows, cols, values);
    const std::vector<double> right_hand_side{4.0, 3.0};

    const auto solve = [&](double regularization) {
        amfls::RunStatistics statistics;
        BlockGolubKahanWorkspace workspace(
            matrix,
            right_hand_side.data(),
            regularization,
            rows,
            statistics);
        workspace.initialize(Matrix(rows, 0));
        while (workspace.can_advance()) {
            if (!workspace.advance_level()) {
                break;
            }
        }
        return workspace.evaluate(1e-12, 0, 0);
    };

    const auto ordinary = solve(0.0);
    require_near(ordinary.solution[0], 2.0, 2e-12, "wide minimum norm x0");
    require_near(ordinary.solution[1], 3.0, 2e-12, "wide minimum norm x1");
    require_near(ordinary.solution[2], 0.0, 2e-12, "wide null component");

    const auto ridge = solve(1.0);
    require_near(ridge.solution[0], 1.6, 2e-12, "wide ridge x0");
    require_near(ridge.solution[1], 1.5, 2e-12, "wide ridge x1");
    require_near(ridge.solution[2], 0.0, 2e-12, "wide ridge null component");
    require_test(
        ridge.operator_norm_lower_bound > 0.0 &&
            ridge.operator_norm_lower_bound <=
                2.0 * (1.0 + 8.0 * std::numeric_limits<double>::epsilon()),
        "ridge validation constructs a certified norm lower bound");
}

void test_scalar_no_pending_path_matches_explicit_projected_basis() {
    constexpr int rows = 7;
    constexpr int cols = 5;
    std::vector<double> values(rows * cols, 0.0);
    for (int col = 0; col < cols; ++col) {
        for (int row = 0; row < rows; ++row) {
            values[row + col * rows] =
                0.15 * (row + 1) * (col + 2) +
                (row == col ? 2.0 : 0.0) +
                (row == col + 1 ? -0.4 : 0.0);
        }
    }
    DenseTestOperator matrix(rows, cols, std::move(values));
    const std::vector<double> right_hand_side{
        1.0, -0.5, 2.0, -1.5, 0.75, 0.25, -0.125};

    const auto run = [&](double regularization) {
        amfls::RunStatistics statistics;
        BlockGolubKahanWorkspace workspace(
            matrix,
            right_hand_side.data(),
            regularization,
            cols,
            statistics);
        workspace.initialize(Matrix(rows, 0));
        constexpr int levels = 3;
        for (int level = 1; level <= levels; ++level) {
            require_test(
                workspace.advance_level(),
                "scalar no-pending GKB advance");
            const auto& audit = workspace.last_step_audit();
            require_test(
                workspace.depth() == level &&
                    workspace.basis_rank() == level &&
                    workspace.active_left_width() == 1 &&
                    audit.requested_injected_columns == 0 &&
                    audit.retained_injected_columns == 0 &&
                    audit.left_input_width == 1 &&
                    audit.retained_right_width == 1 &&
                    audit.retained_next_left_width == 1 &&
                    statistics.search_a_columns == level &&
                    statistics.search_at_columns == level &&
                    statistics.search_a_block_calls == level &&
                    statistics.search_at_block_calls == level,
                "scalar no-pending state and operator accounting");
        }

        const Matrix basis = workspace.right_basis();
        Matrix applied_basis(rows, basis.cols());
        matrix.apply(basis.data(), basis.cols(), applied_basis.data());
        amfls::RunStatistics explicit_statistics;
        const auto explicit_candidate = amfls::math::evaluate_projected_basis(
            matrix,
            right_hand_side.data(),
            regularization,
            1e-11,
            basis,
            applied_basis,
            explicit_statistics,
            0,
            levels,
            0);
        const auto recurrence_candidate = workspace.evaluate(1e-11, 0, 0);
        require_test(
            recurrence_candidate.status == explicit_candidate.status &&
                recurrence_candidate.stop_reason ==
                    explicit_candidate.stop_reason &&
                recurrence_candidate.solution.size() ==
                    explicit_candidate.solution.size(),
            "scalar recurrence and explicit projected candidates agree");
        for (int index = 0;
             index < static_cast<int>(recurrence_candidate.solution.size());
             ++index) {
            require_near(
                recurrence_candidate.solution[index],
                explicit_candidate.solution[index],
                3e-12,
                "scalar recurrence matches explicit projected solution");
        }
        require_test(
            statistics.search_a_columns == levels &&
                statistics.search_at_columns == levels &&
                statistics.validation_a_columns ==
                    (regularization > 0.0 ? 2 : 1) &&
                statistics.validation_at_columns ==
                    (regularization > 0.0 ? 2 : 1),
            "scalar evaluation preserves search and validation accounting");
    };

    run(0.0);
    run(0.25);
}

void test_numerical_truncation_closes_rolling_continuation() {
    constexpr int rows = 3;
    constexpr int cols = 2;
    constexpr double discarded_left_component = 1e-15;
    const std::vector<double> values{
        1.0, discarded_left_component, 0.0,
        0.0, 0.0, 2.0};
    DenseTestOperator matrix(rows, cols, values);
    const std::vector<double> right_hand_side{1.0, 0.0, 0.0};
    amfls::RunStatistics statistics;
    BlockGolubKahanWorkspace workspace(
        matrix,
        right_hand_side.data(),
        0.0,
        cols,
        statistics);
    workspace.initialize(Matrix(rows, 0));
    require_test(
        workspace.advance_level(),
        "numerical-truncation GKB level");
    const auto& audit = workspace.last_step_audit();
    require_test(
        amfls::math::step_has_numerical_rank_truncation(audit) &&
            audit.left_discarded_residual_norm >=
                0.5 * discarded_left_component &&
        workspace.left_basis().cols() == 0 &&
            workspace.active_left_width() == 0 &&
            !workspace.can_advance(),
        "positive below-cutoff loss closes rolling continuation");

    bool rejected_restart = false;
    try {
        Matrix pending(rows, 1);
        pending(2, 0) = 1.0;
        (void)workspace.append_left_seed(pending);
    } catch (const std::logic_error&) {
        rejected_restart = true;
    }
    require_test(
        rejected_restart,
        "rolling workspace must reject restart after numerical truncation");
}

void test_basis_cap_is_reported_separately() {
    constexpr int dimension = 12;
    constexpr int basis_limit = 10;
    std::vector<double> values(dimension * dimension, 0.0);
    for (int index = 0; index < dimension; ++index) {
        values[index + index * dimension] = index + 1.0;
    }
    DenseTestOperator matrix(dimension, dimension, std::move(values));
    std::vector<double> right_hand_side(dimension, 1.0);
    Matrix initial_auxiliary(dimension, 2);
    for (int row = 0; row < dimension; ++row) {
        const double coordinate = row + 1.0;
        initial_auxiliary(row, 0) = std::sin(coordinate);
        initial_auxiliary(row, 1) = std::cos(0.7 * coordinate);
    }

    amfls::RunStatistics statistics;
    BlockGolubKahanWorkspace workspace(
        matrix,
        right_hand_side.data(),
        0.0,
        basis_limit,
        statistics);
    workspace.initialize(initial_auxiliary);
    require_test(
        workspace.active_left_width() == 1 &&
            workspace.left_basis().cols() == 1 &&
            workspace.projected_rhs().size() == 1,
        "initial auxiliary Omega stays pending outside conceptual U");

    const int expected_basis_ranks[]{3, 6, 9, 10};
    const int expected_right_widths[]{3, 3, 3, 1};
    const int expected_next_left_widths[]{3, 3, 3, 1};
    for (int level = 0; level < 4; ++level) {
        require_test(
            workspace.advance_level(),
            "basis-cap variable-width level");
        const auto& audit = workspace.last_step_audit();
        require_test(
            workspace.basis_rank() == expected_basis_ranks[level] &&
                audit.retained_right_width ==
                    expected_right_widths[level] &&
                workspace.active_left_width() ==
                    expected_next_left_widths[level],
                "basis-capped variable-width ranks: level=" +
                std::to_string(level) + " basis=" +
                std::to_string(workspace.basis_rank()) + " right=" +
                std::to_string(audit.retained_right_width) + " active=" +
                std::to_string(workspace.active_left_width()));
        require_test(
            !audit.right_rank_loss &&
                audit.right_basis_cap_truncation == (level == 3) &&
                !audit.left_basis_cap_truncation,
            "basis cap and numerical rank loss must remain distinct");
    }
    require_test(
        !workspace.can_advance(),
        "a full basis cap is terminal without numerical-rank classification");
}

void test_current_step_below_cutoff_deflation_audit() {
    constexpr int dimension = 4;
    const double small = std::ldexp(1.0, -45);
    std::vector<double> values(dimension * dimension, 0.0);
    for (int index = 0; index < dimension; ++index) {
        values[index + index * dimension] = 1.0;
    }
    values[(dimension - 1) + (dimension - 1) * dimension] = small;
    DenseTestOperator matrix(dimension, dimension, std::move(values));
    const std::vector<double> right_hand_side{1.0, 0.0, 0.0, small};
    amfls::RunStatistics statistics;
    BlockGolubKahanWorkspace workspace(
        matrix,
        right_hand_side.data(),
        0.0,
        dimension,
        statistics);
    workspace.initialize(Matrix(dimension, 0));
    require_test(
        workspace.advance_level(),
        "two-mode precision audit completes its retained right step");
    const auto& audit = workspace.last_step_audit();
    require_test(
        audit.retained_right_width == 1 &&
            audit.retained_next_left_width == 0 &&
            audit.left_rank_loss &&
            audit.left_positive_finite_below_cutoff &&
            audit.left_discarded_residual_norm > 0.0 &&
            audit.left_discarded_residual_norm <= audit.left_rank_cutoff &&
            !workspace.can_advance(),
        "current step exposes causal positive below-cutoff frontier exhaustion");
}

void test_pending_only_right_truncation_does_not_close_continuation() {
    constexpr int dimension = 2;
    const double pending_scale = std::ldexp(1.0, -60);
    const std::vector<double> values{
        1.0, 0.0,
        0.0, pending_scale};
    DenseTestOperator matrix(dimension, dimension, values);
    const std::vector<double> right_hand_side{1.0, 0.0};
    amfls::RunStatistics statistics;
    BlockGolubKahanWorkspace workspace(
        matrix,
        right_hand_side.data(),
        0.0,
        dimension,
        statistics);
    workspace.initialize(Matrix(dimension, 0));
    Matrix pending(dimension, 1);
    pending(1, 0) = 1.0;
    require_test(
        workspace.append_left_seed(pending) == 1,
        "pending-only truncation seed");
    require_test(
        workspace.advance_level(),
        "pending-only truncation completes the active right direction");
    const auto& audit = workspace.last_step_audit();
    require_test(
        audit.right_rank_loss &&
            audit.right_positive_finite_below_cutoff &&
            audit.right_discarded_residual_norm > 0.0 &&
            audit.right_active_discarded_residual_norm == 0.0 &&
            !amfls::math::step_has_numerical_rank_truncation(audit),
        "discarded pending-only right direction is not a rolling defect");
    require_test(
        workspace.append_left_seed(pending) == 1,
        "pending-only discard must not prohibit a later pending direction");
}

void test_nonfinite_half_step_closes_with_numerical_breakdown_audit() {
    const std::vector<double> values{
        std::numeric_limits<double>::infinity()};
    DenseTestOperator matrix(1, 1, values);
    const std::vector<double> right_hand_side{1.0};
    amfls::RunStatistics statistics;
    BlockGolubKahanWorkspace workspace(
        matrix,
        right_hand_side.data(),
        0.0,
        1,
        statistics);
    workspace.initialize(Matrix(1, 0));
    require_test(
        !workspace.advance_level(),
        "nonfinite right half-step cannot create a level");
    require_test(
        amfls::math::step_has_nonfinite_breakdown(
            workspace.last_step_audit()) &&
            !workspace.can_advance(),
        "nonfinite right half-step permanently closes continuation");
}

void test_nonfinite_left_half_step_does_not_commit_checkpoint() {
    NonfiniteLeftHalfOperator matrix;
    const std::vector<double> right_hand_side{1.0};
    amfls::RunStatistics statistics;
    BlockGolubKahanWorkspace workspace(
        matrix,
        right_hand_side.data(),
        0.0,
        1,
        statistics);
    workspace.initialize(Matrix(1, 0));
    require_test(
        !workspace.advance_level(),
        "nonfinite left half-step cannot complete a level");
    require_test(
        amfls::math::step_has_nonfinite_breakdown(
            workspace.last_step_audit()) &&
            workspace.right_basis().cols() == 0 &&
            workspace.projected_operator().cols() == 0 &&
            workspace.depth() == 0 &&
            !workspace.can_advance(),
        "nonfinite left half-step leaves the old checkpoint transactional");
}

void test_krylov_block_audit_distinguishes_zero_and_cutoff() {
    Matrix empty_basis(3, 0);
    Matrix zero_block(3, 1);
    amfls::RunStatistics zero_statistics;
    amfls::math::KrylovOrthogonalizationAudit zero_audit;
    Matrix zero_result = amfls::math::orthogonalize_krylov_block(
        std::move(zero_block),
        empty_basis,
        3,
        3,
        3,
        zero_statistics,
        nullptr,
        &zero_audit);
    require_test(
        zero_result.cols() == 0 &&
            zero_audit.projected_residual_norm == 0.0 &&
            !zero_audit.positive_finite_below_cutoff &&
            !zero_audit.nonfinite_residual,
        "exactly zero Krylov residual is not numerical deflation");

    Matrix basis(3, 1);
    basis(0, 0) = 1.0;
    Matrix weak_block(3, 1);
    weak_block(0, 0) = 1.0;
    weak_block(1, 0) = 1.0e-30;
    amfls::RunStatistics weak_statistics;
    amfls::math::KrylovOrthogonalizationAudit weak_audit;
    Matrix weak_result = amfls::math::orthogonalize_krylov_block(
        std::move(weak_block),
        basis,
        3,
        3,
        3,
        weak_statistics,
        nullptr,
        &weak_audit);
    require_test(
        weak_result.cols() == 0 &&
            weak_audit.projected_residual_norm > 0.0 &&
            weak_audit.positive_finite_below_cutoff &&
            !weak_audit.nonfinite_residual,
        "positive residual removed by the cutoff is numerical deflation");
}

void test_raw_initial_gaussian_preserves_fixed_seed_state() {
    constexpr int rows = 7;
    constexpr int cols = 5;
    std::vector<double> values(rows * cols, 0.0);
    for (int index = 0; index < cols; ++index) {
        values[index + index * rows] = 5.0 - 0.75 * index;
    }
    DenseTestOperator matrix(rows, cols, std::move(values));
    const std::vector<double> right_hand_side{
        3.0, -4.0, 2.0, 1.0, -0.5, 0.25, -0.125};
    constexpr int auxiliary_width = 2;
    constexpr std::uint64_t seed = 29;
    constexpr std::uint64_t stream = 7;

    Matrix raw = amfls::math::gaussian_matrix(
        rows, auxiliary_width, seed, stream, 0);
    Matrix preprojected = amfls::math::gaussian_matrix_orthogonal_to(
        rows,
        auxiliary_width,
        right_hand_side.data(),
        seed,
        stream,
        0);

    amfls::RunStatistics raw_statistics;
    amfls::RunStatistics reference_statistics;
    BlockGolubKahanWorkspace raw_workspace(
        matrix,
        right_hand_side.data(),
        0.0,
        cols,
        raw_statistics);
    BlockGolubKahanWorkspace reference_workspace(
        matrix,
        right_hand_side.data(),
        0.0,
        cols,
        reference_statistics);
    const double expected_rhs_norm = amfls::math::vector_norm(
        right_hand_side.data(), rows);
    require_near(
        raw_workspace.right_hand_side_norm(),
        expected_rhs_norm,
        0.0,
        "workspace caches the fixed-seed RHS norm");

    raw_workspace.initialize(raw);
    reference_workspace.initialize(preprojected);
    require_test(
        raw_workspace.projected_rhs() == reference_workspace.projected_rhs() &&
            raw_workspace.projected_rhs().size() == 1 &&
            raw_workspace.projected_rhs().front() == expected_rhs_norm,
        "raw and preprojected fixed seeds have the same initial projected RHS");
    require_test(
        raw_workspace.advance_level() && reference_workspace.advance_level(),
        "raw and preprojected fixed seeds complete the first block level");

    const amfls::LeastSquaresResult raw_result = raw_workspace.evaluate(
        1e-10, 0, auxiliary_width);
    const amfls::LeastSquaresResult reference_result =
        reference_workspace.evaluate(1e-10, 0, auxiliary_width);
    require_test(
        raw_result.status == reference_result.status &&
            raw_result.stop_reason == reference_result.stop_reason &&
            raw_result.depth == reference_result.depth &&
            raw_result.basis_rank == reference_result.basis_rank &&
            raw_workspace.projected_rhs().size() ==
                reference_workspace.projected_rhs().size(),
        "raw fixed seed preserves projected status and dimensions");
    for (int index = 0;
         index < static_cast<int>(raw_result.solution.size());
         ++index) {
        require_near(
            raw_result.solution[index],
            reference_result.solution[index],
            2e-12,
            "raw fixed seed preserves the projected solution");
    }
    for (int index = 0;
         index < static_cast<int>(raw_workspace.projected_rhs().size());
         ++index) {
        require_near(
            raw_workspace.projected_rhs()[index],
            reference_workspace.projected_rhs()[index],
            2e-12,
            "raw fixed seed preserves the completed projected RHS");
    }
    require_test(
        raw_statistics.search_a_columns ==
                reference_statistics.search_a_columns &&
            raw_statistics.search_at_columns ==
                reference_statistics.search_at_columns &&
            raw_statistics.search_a_block_calls ==
                reference_statistics.search_a_block_calls &&
            raw_statistics.search_at_block_calls ==
                reference_statistics.search_at_block_calls &&
            raw_statistics.validation_a_columns ==
                reference_statistics.validation_a_columns &&
            raw_statistics.validation_at_columns ==
                reference_statistics.validation_at_columns,
        "raw fixed seed preserves operator accounting");
}


}  // namespace

int main() {
    test_variable_block_gkb_projection_and_injection();
    test_gkb_wide_minimum_norm_and_ridge_projection();
    test_scalar_no_pending_path_matches_explicit_projected_basis();
    test_numerical_truncation_closes_rolling_continuation();
    test_basis_cap_is_reported_separately();
    test_current_step_below_cutoff_deflation_audit();
    test_pending_only_right_truncation_does_not_close_continuation();
    test_nonfinite_half_step_closes_with_numerical_breakdown_audit();
    test_nonfinite_left_half_step_does_not_commit_checkpoint();
    test_krylov_block_audit_distinguishes_zero_and_cutoff();
    test_raw_initial_gaussian_preserves_fixed_seed_state();
}
