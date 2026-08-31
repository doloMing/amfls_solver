#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include "amfls/amfls.hpp"
#include "amfls/fixed_rbgk.hpp"
#include "amfls/lsmb.hpp"
#include "amfls/lsmr.hpp"
#include "amfls/lsqr.hpp"
#include "amfls/lsrn.hpp"
#include "algorithms/amfls/amfls_internal.hpp"
#include "algorithms/mathematics/krylov/certificate_checkpoint_schedule.hpp"
#include "algorithms/mathematics/krylov/krylov_workspace.hpp"
#include "algorithms/mathematics/linalg/blas_lapack.hpp"
#include "algorithms/mathematics/linalg/svd.hpp"
#include "algorithms/lsmb/lsmb_internal.hpp"
#include "test_helpers.hpp"

namespace {

DenseTestOperator diagonal_operator(const std::vector<double>& diagonal) {
    const int size = static_cast<int>(diagonal.size());
    std::vector<double> values(size * size, 0.0);
    for (int index = 0; index < size; ++index) {
        values[index + index * size] = diagonal[index];
    }
    return DenseTestOperator(size, size, std::move(values));
}

class StatefulTransposeTestOperator final : public amfls::MatrixOperator {
public:
    int rows() const override { return 1; }
    int cols() const override { return 1; }

    void apply(const double*, int block_cols, double* output) const override {
        for (int col = 0; col < block_cols; ++col) {
            output[col] = 0.0;
        }
    }

    void apply_transpose(
        const double* input,
        int block_cols,
        double* output) const override {
        for (int col = 0; col < block_cols; ++col) {
            output[col] = transpose_calls_ == 0 ? 0.0 : input[col];
        }
        ++transpose_calls_;
    }

private:
    mutable int transpose_calls_ = 0;
};

class RidgeCorrectionFailureOperator final : public amfls::MatrixOperator {
public:
    enum class Mode {
        invalid_curvature,
        partial_after_a,
        nonfinite_after_at,
        step_overflow_after_finite_pair,
        squared_diagnostic_overflow,
        infinite_corrected_bound
    };

    explicit RidgeCorrectionFailureOperator(Mode mode) : mode_(mode) {}

    int rows() const override { return 1; }
    int cols() const override { return 1; }

    void apply(const double* input, int block_cols, double* output) const override {
        for (int col = 0; col < block_cols; ++col) {
            if (apply_calls_ > 0 && mode_ == Mode::partial_after_a) {
                output[col] = std::numeric_limits<double>::quiet_NaN();
            } else if (apply_calls_ > 0 &&
                       mode_ == Mode::step_overflow_after_finite_pair) {
                output[col] = 0.0;
            } else if (apply_calls_ > 0 &&
                       mode_ == Mode::squared_diagnostic_overflow) {
                output[col] = std::sqrt(3.0);
            } else if (apply_calls_ > 0 &&
                       mode_ == Mode::infinite_corrected_bound) {
                output[col] = 1.0;
            } else {
                output[col] = input[col];
            }
        }
        ++apply_calls_;
    }

    void apply_transpose(
        const double* input,
        int block_cols,
        double* output) const override {
        for (int col = 0; col < block_cols; ++col) {
            if (transpose_calls_ > 0 &&
                mode_ == Mode::invalid_curvature) {
                output[col] = -2.0;
            } else if (transpose_calls_ > 0 &&
                       mode_ == Mode::nonfinite_after_at) {
                output[col] = std::numeric_limits<double>::quiet_NaN();
            } else if (transpose_calls_ > 0 &&
                       mode_ == Mode::step_overflow_after_finite_pair) {
                output[col] = 0.0;
            } else if (mode_ == Mode::squared_diagnostic_overflow) {
                output[col] = transpose_calls_ == 0 ? 1e200 : 3.0;
            } else if (mode_ == Mode::infinite_corrected_bound) {
                output[col] = transpose_calls_ == 0
                    ? 0.8 * std::numeric_limits<double>::max()
                    : 0.0;
            } else {
                output[col] = input[col];
            }
        }
        ++transpose_calls_;
    }

private:
    Mode mode_;
    mutable int apply_calls_ = 0;
    mutable int transpose_calls_ = 0;
};

void check_phase_accounting(
    const amfls::RunStatistics& statistics,
    const std::string& label) {
    require_test(
        statistics.a_columns ==
            statistics.search_a_columns + statistics.validation_a_columns &&
            statistics.at_columns == statistics.search_at_columns +
                statistics.validation_at_columns,
        label + " column partition");
    require_test(
        statistics.a_block_calls == statistics.search_a_block_calls +
            statistics.validation_a_block_calls &&
            statistics.at_block_calls == statistics.search_at_block_calls +
                statistics.validation_at_block_calls,
        label + " callback partition");
    require_test(
        statistics.validation_a_columns ==
            statistics.base_validation_a_columns +
                statistics.ridge_correction_a_columns &&
            statistics.validation_at_columns ==
                statistics.base_validation_at_columns +
                    statistics.ridge_correction_at_columns,
        label + " validation column partition");
    require_test(
        statistics.validation_a_block_calls ==
            statistics.base_validation_a_block_calls +
                statistics.ridge_correction_a_block_calls &&
            statistics.validation_at_block_calls ==
                statistics.base_validation_at_block_calls +
                    statistics.ridge_correction_at_block_calls,
        label + " validation callback partition");
    require_test(
        statistics.validation_seconds ==
            statistics.base_validation_seconds +
                statistics.ridge_correction_validation_seconds &&
            statistics.base_validation_seconds >= 0.0 &&
            statistics.ridge_correction_validation_seconds >= 0.0,
        label + " validation timing partition");
}

void require_common_numerical_terminal(
    const amfls::LeastSquaresResult& result,
    const std::string& label) {
    require_test(
        result.status == amfls::SolverStatus::numerical_breakdown &&
            result.stop_reason == amfls::StopReason::numerical_breakdown,
        label + " terminal numerical status");
    require_test(!result.trace.empty(), label + " numerical trace");
    require_test(
        result.trace.back().contract_status ==
                amfls::SolverStatus::numerical_breakdown &&
            result.trace.back().contract_stop_reason ==
                amfls::StopReason::numerical_breakdown &&
            !result.trace.back().contract_passed,
        label + " common-validator trace status");
}

void check_solution(
    const amfls::LeastSquaresResult& result,
    const std::vector<double>& expected,
    double tolerance,
    const std::string& label) {
    require_test(result.status == amfls::SolverStatus::success, label + " status");
    require_test(result.solution.size() == expected.size(), label + " solution size");
    for (int index = 0; index < static_cast<int>(expected.size()); ++index) {
        require_near(result.solution[index], expected[index], tolerance, label);
    }
    require_test(
        result.backward_error_upper_bound <= tolerance * 10.0,
        label + " backward-error certificate");
}

void check_lsmr_accounting(
    const amfls::LeastSquaresResult& result,
    int expected_iterations) {
    require_test(
        result.iterations == expected_iterations,
        "LSMR iteration count");
    require_test(
        !result.trace.empty() &&
            result.trace.back().depth == expected_iterations,
        "LSMR trace must end at the committed iteration");
    require_test(
        result.statistics.search_a_columns == expected_iterations &&
            result.statistics.search_at_columns == expected_iterations + 1,
        "LSMR search column accounting");
    require_test(
        result.statistics.validation_a_columns ==
                static_cast<long long>(result.trace.size()) &&
            result.statistics.validation_at_columns ==
                static_cast<long long>(result.trace.size()),
        "LSMR validation column accounting");
    require_test(
        result.statistics.a_columns ==
                expected_iterations +
                    static_cast<long long>(result.trace.size()) &&
            result.statistics.at_columns ==
                expected_iterations + 1 +
                    static_cast<long long>(result.trace.size()),
        "LSMR total column accounting");
    require_test(
        result.statistics.a_block_calls == result.statistics.a_columns &&
            result.statistics.at_block_calls == result.statistics.at_columns,
        "scalar LSMR callback and column counts must agree");
    require_test(
        result.statistics.search_a_block_calls ==
                result.statistics.search_a_columns &&
            result.statistics.search_at_block_calls ==
                result.statistics.search_at_columns,
        "scalar LSMR search callback accounting");
    check_phase_accounting(result.statistics, "LSMR");
    for (int checkpoint = 0;
         checkpoint < static_cast<int>(result.trace.size());
         ++checkpoint) {
        const auto& record = result.trace[checkpoint];
        require_test(
            (expected_iterations == 0 || record.depth > 0) &&
                record.basis_rank == record.depth &&
                record.auxiliary_width == 0,
            "LSMR trace coordinates");
        require_test(
            record.a_columns == record.depth + checkpoint + 1 &&
                record.at_columns == record.depth + checkpoint + 2 &&
                record.a_block_calls == record.a_columns &&
                record.at_block_calls == record.at_columns,
            "LSMR checkpoint accounting");
    }
}

void check_lsmb_accounting(
    const amfls::LeastSquaresResult& result,
    int expected_iterations) {
    require_test(
        result.iterations == expected_iterations,
        "LSMB iteration count");
    require_test(
        !result.trace.empty() &&
            result.trace.back().depth == expected_iterations,
        "LSMB trace must end at the committed iteration");
    require_test(
        result.statistics.search_a_columns == expected_iterations &&
            result.statistics.search_at_columns == expected_iterations + 1,
        "LSMB search column accounting");
    require_test(
        result.statistics.validation_a_columns ==
                static_cast<long long>(result.trace.size()) &&
            result.statistics.validation_at_columns ==
                static_cast<long long>(result.trace.size()),
        "LSMB validation column accounting");
    require_test(
        result.statistics.a_columns ==
                expected_iterations +
                    static_cast<long long>(result.trace.size()) &&
            result.statistics.at_columns ==
                expected_iterations + 1 +
                    static_cast<long long>(result.trace.size()),
        "LSMB total column accounting");
    require_test(
        result.statistics.a_block_calls == result.statistics.a_columns &&
            result.statistics.at_block_calls == result.statistics.at_columns,
        "scalar LSMB callback and column counts must agree");
    require_test(
        result.statistics.search_a_block_calls ==
                result.statistics.search_a_columns &&
            result.statistics.search_at_block_calls ==
                result.statistics.search_at_columns,
        "scalar LSMB search callback accounting");
    check_phase_accounting(result.statistics, "LSMB");
    for (int checkpoint = 0;
         checkpoint < static_cast<int>(result.trace.size());
         ++checkpoint) {
        const auto& record = result.trace[checkpoint];
        require_test(
            (expected_iterations == 0 || record.depth > 0) &&
                record.basis_rank == record.depth &&
                record.auxiliary_width == 0,
            "LSMB trace coordinates");
        require_test(
            record.a_columns == record.depth + checkpoint + 1 &&
                record.at_columns == record.depth + checkpoint + 2 &&
                record.a_block_calls == record.a_columns &&
                record.at_block_calls == record.at_columns,
            "LSMB checkpoint accounting");
    }
}

void check_one_step_exact_beta_accounting(
    const amfls::LeastSquaresResult& result,
    const std::string& label) {
    require_test(
        result.iterations == 1 && result.trace.size() == 1 &&
            result.trace.back().depth == 1,
        label + " exact-beta terminal iteration");
    require_test(
        result.statistics.search_a_columns == 1 &&
            result.statistics.search_at_columns == 1 &&
            result.statistics.search_a_block_calls == 1 &&
            result.statistics.search_at_block_calls == 1,
        label + " exact beta must not schedule a known-zero A* product");
    require_test(
        result.statistics.base_validation_a_columns == 1 &&
            result.statistics.base_validation_at_columns == 1 &&
            result.statistics.a_columns == 2 &&
            result.statistics.at_columns == 2 &&
            result.trace.back().a_columns == 2 &&
            result.trace.back().at_columns == 2,
        label + " exact-beta terminal validation accounting");
    check_phase_accounting(result.statistics, label);
}

}  // namespace

int main() {
    amfls::detail::ProgressHistory deepen_history;
    amfls::detail::record_deepen_progress(
        deepen_history, std::exp(8.0), std::exp(4.0), 1, 1, 2);
    amfls::detail::record_deepen_progress(
        deepen_history, std::exp(4.0), std::exp(2.0), 1, 1, 2);
    require_test(
        deepen_history.same_width_level_interval_count == 2 &&
            deepen_history.active_left_width == 1,
        "AMFLS retains two same-width per-level contractions");
    require_near(
        deepen_history.same_width_level_log_contractions[0],
        2.0,
        1e-14,
        "AMFLS previous per-level contraction");
    require_near(
        deepen_history.same_width_level_log_contractions[1],
        1.0,
        1e-14,
        "AMFLS recent per-level contraction");

    amfls::detail::ProgressHistory reset_history = deepen_history;
    amfls::detail::reset_progress_history(reset_history);
    require_test(
        reset_history.same_width_level_interval_count == 0 &&
            reset_history.active_left_width == -1 &&
            !std::isfinite(
                reset_history.same_width_level_operator_seconds[0]) &&
            !std::isfinite(
                reset_history.same_width_level_local_seconds[0]),
        "AMFLS width change clears same-width progress history");

    amfls::detail::ProgressHistory width_bound_history;
    amfls::detail::record_deepen_progress(
        width_bound_history, 8.0, 4.0, 2, 2);
    amfls::detail::record_deepen_progress(
        width_bound_history, 4.0, 2.0, 2, 2);
    require_test(
        width_bound_history.active_left_width == 2 &&
            width_bound_history.same_width_level_interval_count == 2,
        "AMFLS progress history binds stable active-left width");
    amfls::detail::record_deepen_progress(
        width_bound_history, 2.0, 1.0, 2, 1);
    require_test(
        width_bound_history.active_left_width == -1 &&
            width_bound_history.same_width_level_interval_count == 0,
        "AMFLS crossing an active-left width resets and skips the interval");
    require_test(
        amfls::detail::next_dyadic_active_width_increment(1, 0, 15, 15) ==
                1 &&
            amfls::detail::next_dyadic_active_width_increment(
                2, 1, 15, 14) == 2 &&
            amfls::detail::next_dyadic_active_width_increment(
                4, 3, 15, 12) == 4 &&
            amfls::detail::next_dyadic_active_width_increment(
                8, 7, 15, 8) == 8 &&
            amfls::detail::next_dyadic_active_width_increment(
                4, 3, 6, 3) == 3,
        "AMFLS doubles active width to the resource maximum");
    require_test(
        amfls::detail::next_dyadic_active_width_increment(
            7, 6, 6, 8) == 0 &&
            amfls::detail::next_dyadic_active_width_increment(
                2, 1, 15, 1) == 0 &&
            amfls::detail::next_dyadic_active_width_increment(
                4, 3, 6, 2) == 0 &&
            amfls::detail::next_dyadic_active_width_increment(
                0, 0, 6, 2) == 0,
        "AMFLS active width rejects completed, incomplete, and malformed "
        "stages");
    // A requested three-column prefix that retains only one auxiliary
    // direction leaves active width two.  The next stage doubles that
    // retained frontier, while all three generated columns still consume
    // budget.  Exhausting the generated-column budget forbids another stage.
    require_test(
        amfls::detail::next_dyadic_active_width_increment(
            2, 3, 15, 12) == 2 &&
            amfls::detail::next_dyadic_active_width_increment(
                2, 14, 15, 1) == 1 &&
            amfls::detail::next_dyadic_active_width_increment(
                2, 15, 15, 8) == 0,
        "AMFLS partial deflation separates retained width from generated "
        "column budget");
    require_test(
        amfls::detail::maximum_feedback_feasible_width_increment(
            64, 70, 1, 2) == 2 &&
            amfls::detail::next_dyadic_active_width_increment(
                1, 0, 16, 2) == 1 &&
            amfls::detail::maximum_feedback_feasible_width_increment(
                64, 68, 1, 2) == 1 &&
            amfls::detail::maximum_feedback_feasible_width_increment(
                64, 67, 1, 2) == 0,
        "AMFLS staged widening fits its trial and immediate propagation "
        "level");
    require_test(
        amfls::detail::maximum_feedback_feasible_width_increment(
            -1, 16, 1, 2) == 0 &&
            amfls::detail::maximum_feedback_feasible_width_increment(
                17, 16, 1, 2) == 0 &&
            amfls::detail::maximum_feedback_feasible_width_increment(
                8, 16, -1, 2) == 0 &&
            amfls::detail::maximum_feedback_feasible_width_increment(
                8, 16, 1, 0) == 0,
        "AMFLS feedback-horizon width rejects malformed inputs");
    amfls::detail::ProgressHistory slowdown_history;
    amfls::detail::record_deepen_progress(
        slowdown_history, std::exp(11.0), std::exp(5.0),
        1, 1, 2, 2.0, 0.02);
    amfls::detail::record_deepen_progress(
        slowdown_history, std::exp(5.0), std::exp(1.0),
        1, 1, 2, 20.0, 0.02);
    amfls::detail::record_deepen_progress(
        slowdown_history, std::exp(1.0), 1.0,
        1, 1, 1, 12.0, 0.02);
    const auto slowdown_candidate =
        amfls::detail::make_initial_width_candidate(
            1, 1.0, std::exp(-12.0), 8, 100, 5, 100, 1,
            slowdown_history, 1.0, 1.0);
    require_near(
        slowdown_candidate.prior_level_log_contraction,
        3.0,
        1e-14,
        "AMFLS pooled prior same-width contraction");
    require_near(
        slowdown_candidate.recent_level_log_contraction,
        5.0 / 3.0,
        1e-14,
        "AMFLS recent contraction is weighted by interval levels");
    require_near(
        slowdown_candidate.incumbent_estimated_seconds,
        80.08,
        1e-12,
        "AMFLS conservative incumbent search cost");
    require_near(
        slowdown_candidate.target_estimated_seconds,
        48.4,
        1e-12,
        "AMFLS conservative target search cost");
    require_test(
        slowdown_history.prior_same_width_levels == 2 &&
            slowdown_candidate.history_mature &&
            slowdown_candidate.prior_history_mature &&
            slowdown_candidate.persistent_slowdown &&
            !slowdown_candidate.long_horizon_pressure &&
            slowdown_candidate.widening_necessary &&
            slowdown_candidate.forecast_levels == 8 &&
            slowdown_candidate.matched_horizon == 8 &&
            slowdown_candidate.incumbent_levels == 8 &&
            slowdown_candidate.target_levels == 4 &&
            slowdown_candidate.candidate,
        "AMFLS persistent slowdown admits a strictly cheaper dense width");

    amfls::detail::ProgressHistory long_horizon_history;
    amfls::detail::record_deepen_progress(
        long_horizon_history, std::exp(7.0), std::exp(6.0),
        1, 1, 1, 1.0, 0.01);
    amfls::detail::record_deepen_progress(
        long_horizon_history, std::exp(6.0), std::exp(2.0),
        1, 1, 2, 20.0, 0.02);
    amfls::detail::record_deepen_progress(
        long_horizon_history, std::exp(2.0), 1.0,
        1, 1, 1, 12.0, 0.02);
    const double long_horizon_tolerance = std::exp(-128.0);
    const auto long_horizon_candidate =
        amfls::detail::make_initial_width_candidate(
            1, 1.0, long_horizon_tolerance, 8, 100, 5, 100, 1,
            long_horizon_history, 1.0, 1.0);
    require_test(
        !long_horizon_candidate.persistent_slowdown &&
            long_horizon_candidate.long_horizon_pressure &&
            long_horizon_candidate.widening_necessary &&
            long_horizon_candidate.forecast_levels == 64 &&
            long_horizon_candidate.target_remaining_levels == 47 &&
            long_horizon_candidate.matched_horizon == 64 &&
            long_horizon_candidate.incumbent_levels == 64 &&
            long_horizon_candidate.target_levels == 32 &&
            long_horizon_candidate.candidate,
        "AMFLS long-depth pressure admits a strictly cheaper dense width");

    amfls::detail::ProgressHistory strict_boundary_history;
    amfls::detail::record_deepen_progress(
        strict_boundary_history, std::exp(5.0), std::exp(3.0),
        1, 1, 1, 10.0, 0.01);
    amfls::detail::record_deepen_progress(
        strict_boundary_history, std::exp(3.0), std::exp(1.0),
        1, 1, 1, 10.0, 0.01);
    amfls::detail::record_deepen_progress(
        strict_boundary_history, std::exp(1.0), 1.0,
        1, 1, 1, 12.0, 0.02);
    const auto strict_boundary_candidate =
        amfls::detail::make_initial_width_candidate(
            1, 1.0, std::exp(-95.0), 0, 64, 0, 128, 1,
            strict_boundary_history, 1.0, 1.0);
    require_test(
        strict_boundary_candidate.forecast_levels == 64 &&
            strict_boundary_candidate.target_remaining_levels == 64 &&
            !strict_boundary_candidate.persistent_slowdown &&
            !strict_boundary_candidate.long_horizon_pressure &&
            !strict_boundary_candidate.widening_necessary &&
            strict_boundary_candidate.pass_reduction_admissible &&
            !strict_boundary_candidate.candidate,
        "AMFLS necessity comparisons are strict at both boundaries");

    amfls::detail::ProgressHistory linear_cost_history;
    amfls::detail::record_deepen_progress(
        linear_cost_history, std::exp(7.0), std::exp(6.0),
        1, 1, 1, 10.0, 0.0);
    amfls::detail::record_deepen_progress(
        linear_cost_history, std::exp(6.0), std::exp(2.0),
        1, 1, 2, 20.0, 0.0);
    amfls::detail::record_deepen_progress(
        linear_cost_history, std::exp(2.0), 1.0,
        1, 1, 1, 10.0, 0.0);
    const auto linear_cost_candidate =
        amfls::detail::make_initial_width_candidate(
            1, 1.0, long_horizon_tolerance, 8, 100, 5, 100, 1,
            linear_cost_history, 1.0, 2.0);
    require_test(
        linear_cost_candidate.long_horizon_pressure &&
            linear_cost_candidate.observed_costs_mature &&
            !linear_cost_candidate.pass_reduction_admissible &&
            !linear_cost_candidate.candidate,
        "AMFLS rejects equal-work linear CSR block costs");

    amfls::detail::ProgressHistory local_cost_history;
    amfls::detail::record_deepen_progress(
        local_cost_history, std::exp(7.0), std::exp(6.0),
        1, 1, 1, 1.0, 10.0);
    amfls::detail::record_deepen_progress(
        local_cost_history, std::exp(6.0), std::exp(2.0),
        1, 1, 2, 2.0, 20.0);
    amfls::detail::record_deepen_progress(
        local_cost_history, std::exp(2.0), 1.0,
        1, 1, 1, 1.0, 10.0);
    const auto local_cost_candidate =
        amfls::detail::make_initial_width_candidate(
            1, 1.0, long_horizon_tolerance, 8, 100, 5, 100, 1,
            local_cost_history, 1.0, 1.0);
    require_near(
        local_cost_candidate.incumbent_estimated_seconds,
        704.0,
        1e-12,
        "AMFLS local-dominated incumbent cost");
    require_near(
        local_cost_candidate.target_estimated_seconds,
        1352.0,
        1e-12,
        "AMFLS local-dominated target cost");
    require_test(
        local_cost_candidate.long_horizon_pressure &&
            !local_cost_candidate.pass_reduction_admissible &&
            !local_cost_candidate.candidate,
        "AMFLS rejects widths whose local block cost exceeds pass savings");

    amfls::detail::ProgressHistory width_two_history;
    amfls::detail::record_deepen_progress(
        width_two_history, std::exp(7.0), std::exp(6.0),
        2, 2, 1, 10.0, 0.01);
    amfls::detail::record_deepen_progress(
        width_two_history, std::exp(6.0), std::exp(2.0),
        2, 2, 2, 20.0, 0.02);
    amfls::detail::record_deepen_progress(
        width_two_history, std::exp(2.0), 1.0,
        2, 2, 1, 12.0, 0.02);
    const auto untruncated_horizon_candidate =
        amfls::detail::make_initial_width_candidate(
            2, 1.0, long_horizon_tolerance, 5, 15, 10, 100, 2,
            width_two_history, 1.0, 1.0);
    const auto feasible_full_horizon_candidate =
        amfls::detail::make_initial_width_candidate(
            2, 1.0, long_horizon_tolerance, 5, 100, 10, 106, 2,
            width_two_history, 1.0, 1.0);
    const auto target_resource_candidate =
        amfls::detail::make_initial_width_candidate(
            2, 1.0, long_horizon_tolerance, 5, 15, 11, 30, 2,
            width_two_history, 1.0, 1.0);
    require_test(
        untruncated_horizon_candidate.forecast_levels == 64 &&
            untruncated_horizon_candidate.matched_horizon == 90 &&
            untruncated_horizon_candidate.incumbent_levels == 45 &&
            untruncated_horizon_candidate.target_levels == 23 &&
            untruncated_horizon_candidate.target_remaining_levels == 10 &&
            !untruncated_horizon_candidate.candidate &&
            feasible_full_horizon_candidate.matched_horizon == 96 &&
            feasible_full_horizon_candidate.incumbent_levels == 48 &&
            feasible_full_horizon_candidate.target_levels == 24 &&
            feasible_full_horizon_candidate.target_remaining_levels == 24 &&
            feasible_full_horizon_candidate.candidate &&
            target_resource_candidate.matched_horizon == 19 &&
            target_resource_candidate.target_levels == 5 &&
            target_resource_candidate.target_remaining_levels == 4 &&
            !target_resource_candidate.candidate,
        "AMFLS horizon is not shortened before the target resource gate");

    amfls::detail::MatchedHorizonPlan horizon_plan;
    require_test(
        amfls::detail::start_matched_horizon_plan(
            horizon_plan, 10, 30, 18, 3, 6) &&
            horizon_plan.end_basis_rank == 28 &&
            horizon_plan.origin_active_width == 3 &&
            amfls::detail::remaining_matched_horizon(horizon_plan, 10) == 18 &&
            amfls::detail::remaining_matched_horizon(horizon_plan, 20) == 8 &&
            amfls::detail::remaining_matched_horizon(horizon_plan, 28) == 0 &&
            amfls::detail::remaining_matched_horizon(horizon_plan, 30) == 0,
        "AMFLS matched horizon is consumed by actual retained basis rank");
    require_test(
        amfls::detail::record_matched_horizon_cost(
            horizon_plan, 6, 6, 1, 10.0, 0.1) &&
            amfls::detail::record_matched_horizon_cost(
                horizon_plan, 6, 6, 2, 22.0, 0.4) &&
            horizon_plan.cost_sample_count == 2 &&
            !amfls::detail::record_matched_horizon_cost(
                horizon_plan, 6, 6, 1, 10.0, 0.1),
        "AMFLS plan retains exactly two finite same-width cost samples");
    amfls::detail::MatchedHorizonPlan boundary_plan;
    require_test(
        amfls::detail::start_matched_horizon_plan(
            boundary_plan,
            std::numeric_limits<int>::max() - 18,
            std::numeric_limits<int>::max(),
            18,
            1,
            2) &&
            boundary_plan.end_basis_rank ==
                std::numeric_limits<int>::max() &&
            amfls::detail::remaining_matched_horizon(
                boundary_plan,
                std::numeric_limits<int>::max() - 8) == 8 &&
            amfls::detail::remaining_matched_horizon(
                boundary_plan,
                std::numeric_limits<int>::max()) == 0,
        "AMFLS matched horizon handles the signed-integer boundary");

    amfls::detail::ProgressHistory immature_history;
    amfls::detail::record_deepen_progress(
        immature_history, 8.0, 4.0, 1, 1, 1, 1.0, 0.0);
    amfls::detail::ProgressHistory zero_progress_history;
    amfls::detail::record_deepen_progress(
        zero_progress_history, 4.0, 4.0, 1, 1, 1, 1.0, 0.0);
    amfls::detail::ProgressHistory transient_progress_history;
    amfls::detail::record_deepen_progress(
        transient_progress_history, 8.0, 8.0, 1, 1, 1, 1.0, 0.0);
    amfls::detail::record_deepen_progress(
        transient_progress_history, 8.0, 8.0, 1, 1, 1, 1.0, 0.0);
    amfls::detail::record_deepen_progress(
        transient_progress_history, 8.0, 8.0, 1, 1, 1, 1.0, 0.0);
    amfls::detail::record_deepen_progress(
        transient_progress_history, 8.0, 4.0, 1, 1, 2, 2.0, 0.0);
    amfls::detail::record_deepen_progress(
        zero_progress_history, 4.0, 4.0, 1, 1, 1, 1.0, 0.0);
    amfls::detail::ProgressHistory invalid_cost_history =
        long_horizon_history;
    invalid_cost_history.same_width_level_operator_seconds[0] =
        std::numeric_limits<double>::infinity();
    const auto invalid_certificate_candidate =
        amfls::detail::make_initial_width_candidate(
            1, std::numeric_limits<double>::infinity(),
            long_horizon_tolerance, 8, 100, 5, 100, 1,
            long_horizon_history, 1.0, 1.0);
    const auto immature_candidate =
        amfls::detail::make_initial_width_candidate(
            1, 1.0, long_horizon_tolerance, 8, 100, 5, 100, 1,
            immature_history, 1.0, 1.0);
    const auto zero_progress_candidate =
        amfls::detail::make_initial_width_candidate(
            1, 1.0, long_horizon_tolerance, 8, 100, 5, 100, 1,
            zero_progress_history, 1.0, 1.0);
    const auto transient_progress_candidate =
        amfls::detail::make_initial_width_candidate(
            1, 1.0, long_horizon_tolerance, 8, 100, 5, 100, 1,
            transient_progress_history, 1.0, 1.0);
    const auto invalid_cost_candidate =
        amfls::detail::make_initial_width_candidate(
            1, 1.0, long_horizon_tolerance, 8, 100, 5, 100, 1,
            invalid_cost_history, 1.0, 1.0);
    const auto invalid_relative_cost_candidate =
        amfls::detail::make_initial_width_candidate(
            1, 1.0, long_horizon_tolerance, 8, 100, 5, 100, 1,
            long_horizon_history, 1.0,
            std::numeric_limits<double>::infinity());
    const double finite_costs[2] = {1.0, 1.0};
    const double zero_local_costs[2] = {0.0, 0.0};
    const auto one_sample_candidate =
        amfls::detail::make_horizon_width_candidate(
            1, 8, 0, 10, 0, 8, 1,
            finite_costs, zero_local_costs, 1, 1.0, 1.0);
    const auto shallow_wide_candidate =
        amfls::detail::make_horizon_width_candidate(
            16, 160, 0, 100, 0, 160, 16,
            finite_costs, zero_local_costs, 2, 1.0, 1.0, true);
    const auto first_reopened_width_candidate =
        amfls::detail::make_horizon_width_candidate(
            16, 160, 0, 100, 0, 160, 16,
            finite_costs, zero_local_costs, 2, 1.0, 1.0, false);
    const auto overflow_width_candidate =
        amfls::detail::make_horizon_width_candidate(
            1, 8, 0, std::numeric_limits<int>::max(), 0,
            std::numeric_limits<int>::max(),
            std::numeric_limits<int>::max(),
            finite_costs, finite_costs, 2, 1.0, 1.0);
    require_test(
        !invalid_certificate_candidate.candidate &&
            !invalid_certificate_candidate.finite_current_certificate &&
            !immature_candidate.candidate &&
            !immature_candidate.history_mature &&
            !zero_progress_candidate.candidate &&
            transient_progress_candidate.long_horizon_pressure &&
            !transient_progress_candidate.prior_history_mature &&
            !transient_progress_candidate.widening_necessary &&
            !transient_progress_candidate.candidate &&
            !invalid_cost_candidate.candidate &&
            !invalid_cost_candidate.observed_costs_mature &&
            !invalid_relative_cost_candidate.candidate &&
            !invalid_relative_cost_candidate.observed_costs_mature &&
            !one_sample_candidate.observed_costs_mature &&
            !one_sample_candidate.candidate &&
            shallow_wide_candidate.target_levels == 5 &&
            shallow_wide_candidate.target_remaining_levels == 5 &&
            !shallow_wide_candidate.width_depth_admissible &&
            !shallow_wide_candidate.candidate &&
            first_reopened_width_candidate.width_depth_admissible &&
            first_reopened_width_candidate.candidate &&
            !overflow_width_candidate.candidate &&
            !overflow_width_candidate.stage_feasible,
        "AMFLS width decisions reject incomplete and nonfinite evidence");

    static_assert(
        static_cast<int>(amfls::RidgeCorrectionDisposition::not_applicable) ==
        0);
    static_assert(
        static_cast<int>(
            amfls::RidgeCorrectionDisposition::skipped_base_pass) == 1);
    static_assert(
        static_cast<int>(
            amfls::RidgeCorrectionDisposition::skipped_zero_energy) == 2);
    static_assert(
        static_cast<int>(
            amfls::RidgeCorrectionDisposition::skipped_nonfinite_candidate) ==
        3);
    static_assert(
        static_cast<int>(
            amfls::RidgeCorrectionDisposition::attempted_improved) == 4);
    static_assert(
        static_cast<int>(
            amfls::RidgeCorrectionDisposition::attempted_not_improved) == 5);
    static_assert(
        static_cast<int>(
            amfls::RidgeCorrectionDisposition::attempted_invalid_curvature) ==
        6);
    static_assert(
        static_cast<int>(
            amfls::RidgeCorrectionDisposition::attempted_partial_failure) ==
        7);
    static_assert(
        static_cast<int>(
            amfls::RidgeCorrectionDisposition::attempted_nonfinite) == 8);

    double cubic_gamma = -1.0;
    require_test(
        amfls::detail::solve_lsmb_gamma(
            0.0, 0.0, 1.0, -0.25, 0.75, false, false, cubic_gamma),
        "LSMB safeguarded cubic must bracket a regular root");
    require_near(cubic_gamma, 0.25, 1e-15, "LSMB cubic root");
    require_test(
        amfls::detail::solve_lsmb_gamma(
            1.0, -1.0, 0.0, 0.0, 0.0, true, true, cubic_gamma) &&
            cubic_gamma == 0.0,
        "LSMB degenerate cubic must select the zero endpoint");
    require_test(
        !amfls::detail::solve_lsmb_gamma(
            std::numeric_limits<double>::infinity(),
            0.0,
            1.0,
            -0.5,
            0.5,
            false,
            false,
            cubic_gamma),
        "LSMB cubic must reject nonfinite coefficients");
    require_test(
        amfls::detail::solve_lsmb_gamma(
            1.0, -1.5, 0.66, -0.08, 0.08, false, false,
            cubic_gamma) &&
            cubic_gamma > 0.0 && cubic_gamma < 1.0 &&
            std::abs(amfls::detail::evaluate_lsmb_cubic(
                1.0, -1.5, 0.66, -0.08, cubic_gamma)) < 1e-14,
        "LSMB safeguarded cubic must return a valid root when roots repeat");
    require_test(
        amfls::detail::solve_lsmb_gamma(
            0.0, 0.0, 1.0, -1.0, 0.0, false, true, cubic_gamma) &&
            cubic_gamma == 1.0,
        "LSMB cubic must select the exact upper endpoint");
    require_test(
        !amfls::detail::solve_lsmb_gamma(
            0.0, 0.0, 1.0, -1e-300, -1e-320, false, false,
            cubic_gamma),
        "LSMB cubic must reject a rounded negative upper endpoint");

    // The product of the two LSQR backward-error branches can be quadratically
    // too small.  This exact construction has C(x)=S(x)=delta but the legacy
    // normal-equation ratio eta(x)=delta^2.  A tolerance between those scales
    // must not be reported as success.
    constexpr double delta = 1e-3;
    const double t = (1.0 - delta * delta) / (2.0 * delta);
    amfls::LeastSquaresResult stopping_counterexample;
    stopping_counterexample.residual_norm = 1.0;
    stopping_counterexample.normal_residual_norm = delta;
    stopping_counterexample.solution_norm = t;
    stopping_counterexample.operator_norm_lower_bound = 1.0;
    amfls::math::evaluate_accuracy_contract(
        stopping_counterexample,
        std::hypot(t, 1.0),
        0.0,
        10.0 * delta * delta);
    require_near(
        stopping_counterexample.compatible_backward_error_upper_bound,
        delta,
        1e-14,
        "compatible backward-error branch");
    require_near(
        stopping_counterexample.least_squares_backward_error_upper_bound,
        delta,
        1e-14,
        "least-squares backward-error branch");
    require_near(
        stopping_counterexample.relative_normal_residual_upper_bound,
        delta * delta,
        1e-14,
        "legacy normal-equation ratio");
    require_test(
        stopping_counterexample.status == amfls::SolverStatus::work_limit,
        "a product-sized normal residual must not certify backward accuracy");

    // For fixed ridge, an augmented backward error can be small while the
    // relative H_lambda-energy error is one.  The ridge success gate therefore
    // uses the structured energy bound, not the unstructured augmented test.
    amfls::LeastSquaresResult ridge_counterexample;
    ridge_counterexample.residual_norm = 1.0;
    ridge_counterexample.normal_residual_norm = 1.0;
    ridge_counterexample.solution_norm = 0.0;
    ridge_counterexample.solution_energy_norm = 0.0;
    ridge_counterexample.operator_norm_lower_bound = 1e6;
    amfls::math::evaluate_accuracy_contract(
        ridge_counterexample, 1.0, 1.0, 1e-3);
    require_test(
        ridge_counterexample.backward_error_upper_bound < 1e-3,
        "ridge augmented backward error should expose the counterexample");
    require_test(
        ridge_counterexample.status == amfls::SolverStatus::work_limit,
        "ridge success must require the relative energy bound");

    // A finite nominal stationarity numerator is not enough when a finite
    // operator error envelope cannot be represented.  For
    // A=[DBL_MAX,0]^T the next outward binary64 Frobenius bound is +inf, so
    // the validator must retain the nominal diagnostics without certifying.
    const double largest_finite = std::numeric_limits<double>::max();
    DenseTestOperator denominator_overflow_operator(
        2, 1, {largest_finite, 0.0});
    const std::vector<double> denominator_overflow_b{0.0, 2.0};
    const std::vector<double> denominator_overflow_x{0.0};
    amfls::RunStatistics denominator_overflow_statistics;
    const auto denominator_overflow_candidate =
        amfls::math::validate_original_candidate(
            denominator_overflow_operator,
            denominator_overflow_b.data(),
            std::span<const double>(denominator_overflow_x),
            0.0,
            1e-12,
            largest_finite,
            denominator_overflow_statistics);
    require_test(
        denominator_overflow_candidate.status ==
                amfls::SolverStatus::work_limit &&
            denominator_overflow_candidate.normal_residual_norm == 0.0 &&
            denominator_overflow_candidate.objective == 4.0 &&
            std::isinf(
                denominator_overflow_candidate.backward_error_upper_bound),
        "unrepresentable callback envelope cannot certify nominal stationarity");
    check_phase_accounting(
        denominator_overflow_statistics,
        "certificate-denominator-overflow validator");

    // Ordinary validation has an explicit not-applicable correction audit;
    // its two scalar callbacks are entirely in the base partition.
    DenseTestOperator ordinary_identity = diagonal_operator({1.0});
    const std::vector<double> ordinary_b{1.0};
    const std::vector<double> ordinary_x{1.0};
    amfls::RunStatistics ordinary_statistics;
    const auto ordinary_candidate = amfls::math::validate_original_candidate(
        ordinary_identity,
        ordinary_b.data(),
        std::span<const double>(ordinary_x),
        0.0,
        1e-8,
        1.0,
        ordinary_statistics);
    require_test(
        ordinary_candidate.ridge_correction_disposition ==
                amfls::RidgeCorrectionDisposition::not_applicable &&
            std::isnan(
                ordinary_candidate.ridge_base_energy_error_upper_bound) &&
            std::isnan(
                ordinary_candidate.ridge_corrected_energy_error_upper_bound) &&
            std::isnan(ordinary_candidate.ridge_correction_gamma) &&
            ordinary_statistics.base_validation_a_columns == 1 &&
            ordinary_statistics.base_validation_at_columns == 1 &&
            ordinary_statistics.ridge_correction_a_columns == 0 &&
            ordinary_statistics.ridge_correction_at_columns == 0,
        "ordinary ridge-correction audit is explicitly not applicable");
    check_phase_accounting(ordinary_statistics, "ordinary validator");

    // A gradient concentrated in a stiff eigendirection makes the old
    // ||g||/sqrt(lambda) bound arbitrarily loose.  The shared original-
    // candidate validator must spend one additional A/A* pair, retain the
    // cross term, and certify this candidate with the one-step bound.
    DenseTestOperator correction_operator = diagonal_operator({1e6, 1.0});
    const std::vector<double> correction_b{0.0, 2.0};
    const std::vector<double> correction_x{1e-12, 1.0};
    amfls::RunStatistics correction_statistics;
    const auto corrected_candidate = amfls::math::validate_original_candidate(
        correction_operator,
        correction_b.data(),
        std::span<const double>(correction_x),
        1.0,
        1e-5,
        1.0,
        correction_statistics);
    require_test(
        corrected_candidate.status == amfls::SolverStatus::work_limit &&
            corrected_candidate.ridge_correction_disposition ==
                amfls::RidgeCorrectionDisposition::attempted_improved,
        "one-step ridge correction strict-roundoff contract");
    require_test(
        corrected_candidate.normal_residual_norm > 1.0 &&
            corrected_candidate.energy_error_upper_bound <
                corrected_candidate.ridge_base_energy_error_upper_bound &&
            corrected_candidate.relative_energy_error_upper_bound > 1e-5,
        "one-step ridge correction improves without crossing its FP64 floor");
    const double exact_correction_error =
        std::sqrt((1e12 + 1.0) * 1e-24);
    require_test(
        corrected_candidate.energy_error_upper_bound >=
            exact_correction_error * (1.0 - 1e-12),
        "one-step ridge correction must cover the analytic energy error");
    require_test(
        corrected_candidate.objective_gap_upper_bound >=
            corrected_candidate.energy_error_upper_bound *
                corrected_candidate.energy_error_upper_bound,
        "corrected ridge objective-gap outward bound");
    require_test(
        corrected_candidate.ridge_base_energy_error_upper_bound >=
            corrected_candidate.normal_residual_norm,
        "lambda-only ridge audit is outward rounded");
    require_near(
        corrected_candidate.ridge_correction_gamma,
        1.0 / (1e12 + 1.0),
        1e-24,
        "analytic ridge correction gamma");
    require_test(
        corrected_candidate.ridge_corrected_energy_error_upper_bound *
                corrected_candidate.ridge_corrected_energy_error_upper_bound >=
            corrected_candidate.ridge_correction_z_h_z +
                corrected_candidate.ridge_correction_two_abs_z_t_q +
                corrected_candidate.ridge_correction_q_norm_squared,
        "ridge correction outward decomposition");
    require_near(
        corrected_candidate.energy_error_upper_bound,
        corrected_candidate.ridge_corrected_energy_error_upper_bound,
        1e-20,
        "improved ridge audit selects d_corr");
    require_test(
        correction_statistics.validation_a_columns == 2 &&
            correction_statistics.validation_at_columns == 2 &&
            correction_statistics.base_validation_a_columns == 1 &&
            correction_statistics.base_validation_at_columns == 1 &&
            correction_statistics.ridge_correction_a_columns == 1 &&
            correction_statistics.ridge_correction_at_columns == 1 &&
            correction_statistics.a_block_calls == 2 &&
            correction_statistics.at_block_calls == 2,
        "one-step ridge correction validation accounting");
    check_phase_accounting(correction_statistics, "improved ridge validator");

    // With A=diag(2,1), lambda=4, x=(1,1), and b=(2,3), the
    // correction has a nonzero q and therefore locks the q^2/lambda scaling:
    // g=(4,2), gamma=5/37, zHz=3700/1369, q^2=720/1369,
    // and d_corr^2=(3700+720/4)/1369=3880/1369.
    DenseTestOperator scaled_ridge_operator = diagonal_operator({2.0, 1.0});
    const std::vector<double> scaled_ridge_b{2.0, 3.0};
    const std::vector<double> scaled_ridge_x{1.0, 1.0};
    amfls::RunStatistics scaled_ridge_statistics;
    const auto scaled_ridge_candidate =
        amfls::math::validate_original_candidate(
            scaled_ridge_operator,
            scaled_ridge_b.data(),
            std::span<const double>(scaled_ridge_x),
            4.0,
            1e-8,
            2.0,
            scaled_ridge_statistics);
    require_test(
        scaled_ridge_candidate.ridge_correction_disposition ==
            amfls::RidgeCorrectionDisposition::attempted_improved,
        "scaled ridge correction disposition");
    require_near(
        scaled_ridge_candidate.ridge_base_energy_error_upper_bound,
        std::sqrt(5.0),
        1e-14,
        "scaled ridge d0");
    require_near(
        scaled_ridge_candidate.ridge_correction_gamma,
        5.0 / 37.0,
        1e-15,
        "scaled ridge gamma");
    require_test(
        scaled_ridge_candidate.ridge_correction_z_h_z >=
                3700.0 / 1369.0 &&
            scaled_ridge_candidate.ridge_correction_two_abs_z_t_q >= 0.0 &&
            scaled_ridge_candidate.ridge_correction_q_norm_squared >=
                720.0 / 1369.0,
        "scaled ridge correction terms are conservative");
    require_test(
        scaled_ridge_candidate.ridge_corrected_energy_error_upper_bound >=
                std::sqrt(3880.0 / 1369.0) &&
            scaled_ridge_candidate.energy_error_upper_bound ==
                scaled_ridge_candidate
                    .ridge_corrected_energy_error_upper_bound,
        "scaled ridge selected bound covers the analytic correction");
    check_phase_accounting(
        scaled_ridge_statistics, "scaled analytic ridge validator");

    // A computed coefficient can exceed the finite double range without
    // invalidating the normalized correction.  Here A=0, lambda is the
    // smallest subnormal, g=lambda, u=z=1, q=0, and gamma=1/lambda=+inf;
    // d0 and d_corr are the same finite sqrt(lambda), so no improvement is
    // selected and the extended-real gamma remains an audit value.
    DenseTestOperator zero_scalar_operator(1, 1, {0.0});
    const std::vector<double> zero_scalar_b{0.0};
    const std::vector<double> unit_scalar_x{1.0};
    amfls::RunStatistics infinite_gamma_statistics;
    const auto infinite_gamma_candidate =
        amfls::math::validate_original_candidate(
            zero_scalar_operator,
            zero_scalar_b.data(),
            std::span<const double>(unit_scalar_x),
            std::numeric_limits<double>::denorm_min(),
            1e-8,
            0.0,
            infinite_gamma_statistics);
    require_test(
        infinite_gamma_candidate.status == amfls::SolverStatus::work_limit &&
            infinite_gamma_candidate.ridge_correction_disposition ==
                amfls::RidgeCorrectionDisposition::skipped_zero_energy &&
            std::isnan(infinite_gamma_candidate.ridge_correction_gamma) &&
            std::isfinite(infinite_gamma_candidate
                              .ridge_base_energy_error_upper_bound) &&
            infinite_gamma_candidate.energy_error_upper_bound ==
                infinite_gamma_candidate
                    .ridge_base_energy_error_upper_bound,
        "subnormal ridge data retain the strict base bound");
    check_phase_accounting(
        infinite_gamma_statistics, "infinite-gamma validator");

    // Even with a stored zero transpose product, lambda*x can round to zero
    // from a nonzero exact subnormal.  The explicit denormal radius must keep
    // that stored zero from becoming a false exact-stationarity certificate.
    const std::vector<double> quarter_scalar_x{0.25};
    amfls::RunStatistics ridge_product_underflow_statistics;
    const auto ridge_product_underflow_candidate =
        amfls::math::validate_original_candidate(
            zero_scalar_operator,
            zero_scalar_b.data(),
            std::span<const double>(quarter_scalar_x),
            std::numeric_limits<double>::denorm_min(),
            1e-8,
            0.0,
            ridge_product_underflow_statistics);
    require_test(
        ridge_product_underflow_candidate.normal_residual_norm == 0.0 &&
            ridge_product_underflow_candidate.status ==
                amfls::SolverStatus::work_limit &&
            ridge_product_underflow_candidate.energy_error_upper_bound > 0.0,
        "ridge-product underflow cannot certify false stationarity");
    check_phase_accounting(
        ridge_product_underflow_statistics,
        "ridge-product-underflow validator");

    // A passing old gate and an iterate with zero H-energy both skip the
    // optional pair.  The latter cannot acquire a positive relative-error
    // denominator from an inverse-energy correction.
    DenseTestOperator identity_operator = diagonal_operator({1.0, 1.0});
    const std::vector<double> old_pass_b{2.0 - 1e-10, 2.0};
    const std::vector<double> old_pass_x{1.0, 1.0};
    amfls::RunStatistics old_pass_statistics;
    const auto old_pass_candidate = amfls::math::validate_original_candidate(
        identity_operator,
        old_pass_b.data(),
        std::span<const double>(old_pass_x),
        1.0,
        1e-8,
        1.0,
        old_pass_statistics);
    require_test(
        old_pass_candidate.status == amfls::SolverStatus::success &&
            old_pass_candidate.ridge_correction_disposition ==
                amfls::RidgeCorrectionDisposition::skipped_base_pass &&
            std::isfinite(
                old_pass_candidate.ridge_base_energy_error_upper_bound) &&
            std::isnan(
                old_pass_candidate.ridge_corrected_energy_error_upper_bound) &&
            old_pass_statistics.validation_a_columns == 1 &&
            old_pass_statistics.validation_at_columns == 1,
        "old ridge bound pass must skip correction products");
    check_phase_accounting(old_pass_statistics, "base-pass ridge validator");

    const std::vector<double> zero_candidate_x{0.0, 0.0};
    const std::vector<double> nonzero_b{1.0, 0.0};
    amfls::RunStatistics zero_candidate_statistics;
    const auto zero_candidate = amfls::math::validate_original_candidate(
        identity_operator,
        nonzero_b.data(),
        std::span<const double>(zero_candidate_x),
        1.0,
        1e-8,
        1.0,
        zero_candidate_statistics);
    require_test(
        zero_candidate.status == amfls::SolverStatus::work_limit &&
            zero_candidate.ridge_correction_disposition ==
                amfls::RidgeCorrectionDisposition::skipped_zero_energy &&
            zero_candidate_statistics.validation_a_columns == 1 &&
            zero_candidate_statistics.validation_at_columns == 1,
        "zero-energy ridge candidate must skip correction products");
    check_phase_accounting(zero_candidate_statistics, "zero-energy validator");

    // The correction is not uniformly sharper.  A mostly soft gradient with
    // a tiny stiff component leaves a large correction residual, so the
    // minimum with the old lambda-only bound is essential.
    DenseTestOperator mixed_gradient_operator =
        diagonal_operator({0.0, 1e6});
    const std::vector<double> mixed_gradient_b{0.0, 0.0};
    const std::vector<double> mixed_gradient_x{1.0, 1e-18};
    amfls::RunStatistics mixed_gradient_statistics;
    const auto mixed_gradient_candidate =
        amfls::math::validate_original_candidate(
            mixed_gradient_operator,
            mixed_gradient_b.data(),
            std::span<const double>(mixed_gradient_x),
            1.0,
            1e-8,
            1.0,
            mixed_gradient_statistics);
    require_test(
        mixed_gradient_candidate.energy_error_upper_bound ==
                mixed_gradient_candidate
                    .ridge_base_energy_error_upper_bound &&
            mixed_gradient_candidate.energy_error_upper_bound >=
                mixed_gradient_candidate.normal_residual_norm,
        "ridge correction must retain the better strict base bound");
    require_test(
        mixed_gradient_candidate.status == amfls::SolverStatus::work_limit &&
            mixed_gradient_candidate.ridge_correction_disposition ==
                amfls::RidgeCorrectionDisposition::attempted_not_improved &&
            mixed_gradient_candidate.ridge_corrected_energy_error_upper_bound >=
                mixed_gradient_candidate.ridge_base_energy_error_upper_bound &&
            mixed_gradient_statistics.validation_a_columns == 2 &&
            mixed_gradient_statistics.validation_at_columns == 2,
        "non-improving ridge correction accounting");
    check_phase_accounting(
        mixed_gradient_statistics, "non-improving ridge validator");

    // Nonfinite candidate primitives skip before the optional pair and leave
    // all uncomputed correction diagnostics as quiet NaNs.
    const std::vector<double> nonfinite_x{
        std::numeric_limits<double>::quiet_NaN()};
    amfls::RunStatistics nonfinite_input_statistics;
    const auto nonfinite_input_candidate =
        amfls::math::validate_original_candidate(
            ordinary_identity,
            ordinary_b.data(),
            std::span<const double>(nonfinite_x),
            1.0,
            1e-8,
            1.0,
            nonfinite_input_statistics);
    require_test(
        nonfinite_input_candidate.ridge_correction_disposition ==
                amfls::RidgeCorrectionDisposition::skipped_nonfinite_candidate &&
            std::isnan(nonfinite_input_candidate
                           .ridge_base_energy_error_upper_bound) &&
            std::isnan(nonfinite_input_candidate
                           .ridge_corrected_energy_error_upper_bound) &&
            nonfinite_input_statistics.ridge_correction_a_columns == 0 &&
            nonfinite_input_statistics.ridge_correction_at_columns == 0,
        "nonfinite ridge input skip audit");
    check_phase_accounting(
        nonfinite_input_statistics, "nonfinite-input validator");

    const double maximum_finite = std::numeric_limits<double>::max();
    const std::vector<double> overflowing_gradient_x{
        maximum_finite, maximum_finite};
    const std::vector<double> zero_b_two{0.0, 0.0};
    amfls::RunStatistics infinite_d0_statistics;
    const auto infinite_d0_candidate =
        amfls::math::validate_original_candidate(
            identity_operator,
            zero_b_two.data(),
            std::span<const double>(overflowing_gradient_x),
            1.0,
            1e-8,
            1.0,
            infinite_d0_statistics);
    require_test(
        std::isinf(infinite_d0_candidate.normal_residual_norm) &&
            std::isinf(infinite_d0_candidate
                           .ridge_base_energy_error_upper_bound) &&
            std::isinf(infinite_d0_candidate.energy_error_upper_bound) &&
            infinite_d0_candidate.ridge_correction_disposition ==
                amfls::RidgeCorrectionDisposition::skipped_nonfinite_candidate,
        "computed infinite d0 remains extended-real infinity");
    check_phase_accounting(infinite_d0_statistics, "infinite-d0 validator");

    // Every candidate component is finite, but the two-component solution
    // norm exceeds DBL_MAX.  For the zero ordinary operator both residual
    // numerators are exactly zero; this must be numerical breakdown, not a
    // zero-numerator success.
    const double overflowing_component = std::ldexp(1.5, 1023);
    const std::vector<double> finite_overflow_x{
        overflowing_component, overflowing_component};
    DenseTestOperator zero_wide_operator(1, 2, {0.0, 0.0});
    const std::vector<double> zero_scalar_rhs{0.0};
    amfls::RunStatistics ordinary_norm_overflow_statistics;
    const auto ordinary_norm_overflow =
        amfls::math::validate_original_candidate(
            zero_wide_operator,
            zero_scalar_rhs.data(),
            std::span<const double>(finite_overflow_x),
            0.0,
            1e-8,
            0.0,
            ordinary_norm_overflow_statistics);
    require_test(
        std::all_of(
            finite_overflow_x.begin(),
            finite_overflow_x.end(),
            [](double value) { return std::isfinite(value); }) &&
            std::isinf(ordinary_norm_overflow.solution_norm) &&
            ordinary_norm_overflow.normal_residual_norm == 0.0 &&
            ordinary_norm_overflow.status ==
                amfls::SolverStatus::numerical_breakdown &&
            ordinary_norm_overflow.stop_reason ==
                amfls::StopReason::numerical_breakdown,
        "ordinary finite-component norm overflow cannot certify success");
    check_phase_accounting(
        ordinary_norm_overflow_statistics,
        "ordinary solution-norm-overflow validator");

    // Powers of two make a ridge stationarity cancellation exact.  With
    // A=sqrt(lambda)=2^-537, lambda=2^-1074, and
    // b=2*A*x, each finite x component is stationary while ||x|| overflows.
    // The old zero-bound shortcut could report exact stationarity here.
    const double square_root_denorm = std::ldexp(1.0, -537);
    const double stationary_rhs_value =
        2.0 * square_root_denorm * overflowing_component;
    DenseTestOperator ridge_norm_overflow_operator = diagonal_operator(
        {square_root_denorm, square_root_denorm});
    const std::vector<double> ridge_norm_overflow_b{
        stationary_rhs_value, stationary_rhs_value};
    amfls::RunStatistics ridge_norm_overflow_statistics;
    const auto ridge_norm_overflow =
        amfls::math::validate_original_candidate(
            ridge_norm_overflow_operator,
            ridge_norm_overflow_b.data(),
            std::span<const double>(finite_overflow_x),
            std::numeric_limits<double>::denorm_min(),
            1e-8,
            square_root_denorm,
            ridge_norm_overflow_statistics);
    require_test(
        std::isfinite(stationary_rhs_value) &&
            ridge_norm_overflow.normal_residual_norm == 0.0 &&
            std::isinf(ridge_norm_overflow.solution_norm) &&
            std::isinf(ridge_norm_overflow.solution_energy_norm) &&
            ridge_norm_overflow.status ==
                amfls::SolverStatus::numerical_breakdown &&
            ridge_norm_overflow.stop_reason ==
                amfls::StopReason::numerical_breakdown &&
            ridge_norm_overflow.ridge_correction_disposition ==
                amfls::RidgeCorrectionDisposition::skipped_nonfinite_candidate,
        "ridge finite-component norm overflow cannot certify stationarity");
    check_phase_accounting(
        ridge_norm_overflow_statistics,
        "ridge solution-norm-overflow validator");

    // All gate norms can remain finite while the explicitly stored ridge
    // objective overflows.  Exact cancellation gives g=0 for A=1,
    // lambda=1, b=1e200, x=5e199; +inf objective is nevertheless a
    // nonfinite candidate primitive and cannot certify stationarity.
    DenseTestOperator objective_overflow_operator = diagonal_operator({1.0});
    const std::vector<double> objective_overflow_b{1e200};
    const std::vector<double> objective_overflow_x{5e199};
    amfls::RunStatistics objective_overflow_statistics;
    const auto objective_overflow_candidate =
        amfls::math::validate_original_candidate(
            objective_overflow_operator,
            objective_overflow_b.data(),
            std::span<const double>(objective_overflow_x),
            1.0,
            1e-8,
            1.0,
            objective_overflow_statistics);
    require_test(
        objective_overflow_candidate.normal_residual_norm == 0.0 &&
            std::isfinite(objective_overflow_candidate.residual_norm) &&
            std::isfinite(objective_overflow_candidate.solution_norm) &&
            std::isfinite(
                objective_overflow_candidate.solution_energy_norm) &&
            std::isinf(objective_overflow_candidate.objective) &&
            objective_overflow_candidate.status ==
                amfls::SolverStatus::numerical_breakdown &&
            objective_overflow_candidate.stop_reason ==
                amfls::StopReason::numerical_breakdown &&
            objective_overflow_candidate.ridge_correction_disposition ==
                amfls::RidgeCorrectionDisposition::skipped_nonfinite_candidate,
        "objective overflow cannot certify exact ridge stationarity");
    check_phase_accounting(
        objective_overflow_statistics, "objective-overflow validator");

    amfls::LsqrOptions objective_overflow_lsqr_options;
    objective_overflow_lsqr_options.regularization = 1.0;
    objective_overflow_lsqr_options.tolerance = 1e-8;
    objective_overflow_lsqr_options.maximum_iterations = 1;
    const auto objective_overflow_lsqr = amfls::solve_lsqr(
        objective_overflow_operator,
        objective_overflow_b.data(),
        objective_overflow_lsqr_options);
    require_common_numerical_terminal(
        objective_overflow_lsqr, "objective-overflow LSQR");
    check_phase_accounting(
        objective_overflow_lsqr.statistics, "objective-overflow LSQR");

    // The two RHS components are finite but their Euclidean norm is +inf.
    // Every public solver must stop at its first common validation and retain
    // the exact numerical-breakdown status; no recurrence/cap/epoch policy
    // may rewrite it.  The zero operator keeps every search construction
    // finite and isolates the validator/terminal policy.
    DenseTestOperator rhs_norm_overflow_operator(2, 1, {0.0, 0.0});
    const std::vector<double> rhs_norm_overflow_b{
        largest_finite, largest_finite};

    amfls::LsqrOptions rhs_overflow_lsqr_options;
    rhs_overflow_lsqr_options.maximum_iterations = 1;
    const auto rhs_overflow_lsqr = amfls::solve_lsqr(
        rhs_norm_overflow_operator,
        rhs_norm_overflow_b.data(),
        rhs_overflow_lsqr_options);
    require_common_numerical_terminal(rhs_overflow_lsqr, "RHS-overflow LSQR");
    check_phase_accounting(rhs_overflow_lsqr.statistics, "RHS-overflow LSQR");

    amfls::LsmrOptions rhs_overflow_lsmr_options;
    rhs_overflow_lsmr_options.maximum_iterations = 1;
    const auto rhs_overflow_lsmr = amfls::solve_lsmr(
        rhs_norm_overflow_operator,
        rhs_norm_overflow_b.data(),
        rhs_overflow_lsmr_options);
    require_common_numerical_terminal(rhs_overflow_lsmr, "RHS-overflow LSMR");
    check_phase_accounting(rhs_overflow_lsmr.statistics, "RHS-overflow LSMR");

    amfls::LsmbOptions rhs_overflow_lsmb_options;
    rhs_overflow_lsmb_options.maximum_iterations = 1;
    const auto rhs_overflow_lsmb = amfls::solve_lsmb(
        rhs_norm_overflow_operator,
        rhs_norm_overflow_b.data(),
        rhs_overflow_lsmb_options);
    require_common_numerical_terminal(rhs_overflow_lsmb, "RHS-overflow LSMB");
    check_phase_accounting(rhs_overflow_lsmb.statistics, "RHS-overflow LSMB");

    amfls::LsrnOptions rhs_overflow_lsrn_options;
    rhs_overflow_lsrn_options.maximum_iterations = 1;
    const auto rhs_overflow_lsrn = amfls::solve_lsrn(
        rhs_norm_overflow_operator,
        rhs_norm_overflow_b.data(),
        rhs_overflow_lsrn_options);
    require_common_numerical_terminal(rhs_overflow_lsrn, "RHS-overflow LSRN");
    check_phase_accounting(rhs_overflow_lsrn.statistics, "RHS-overflow LSRN");

    amfls::FixedRbgkOptions rhs_overflow_fixed_options;
    rhs_overflow_fixed_options.auxiliary_width = 0;
    rhs_overflow_fixed_options.maximum_depth = 1;
    rhs_overflow_fixed_options.stop_early = false;
    const auto rhs_overflow_fixed = amfls::solve_fixed_rbgk(
        rhs_norm_overflow_operator,
        rhs_norm_overflow_b.data(),
        rhs_overflow_fixed_options);
    require_common_numerical_terminal(
        rhs_overflow_fixed, "RHS-overflow fixed RBGK");
    check_phase_accounting(
        rhs_overflow_fixed.statistics, "RHS-overflow fixed RBGK");

    amfls::AmflsOptions rhs_overflow_amfls_options;
    rhs_overflow_amfls_options.maximum_depth = 1;
    const auto rhs_overflow_amfls = amfls::solve_amfls(
        rhs_norm_overflow_operator,
        rhs_norm_overflow_b.data(),
        rhs_overflow_amfls_options);
    require_common_numerical_terminal(
        rhs_overflow_amfls, "RHS-overflow AMFLS");
    check_phase_accounting(
        rhs_overflow_amfls.statistics, "RHS-overflow AMFLS");

    const std::vector<double> failure_b{0.0};
    const std::vector<double> failure_x{1.0};
    // A custom callback without an explicit error model remains callable, but
    // neither its nominal base quantities nor its optional correction may be
    // promoted to a strict certificate.
    RidgeCorrectionFailureOperator unmodelled_operator(
        RidgeCorrectionFailureOperator::Mode::invalid_curvature);
    amfls::RunStatistics unmodelled_statistics;
    const auto unmodelled_candidate =
        amfls::math::validate_original_candidate(
            unmodelled_operator,
            failure_b.data(),
            std::span<const double>(failure_x),
            1.0,
            1e-8,
            1.0,
            unmodelled_statistics);
    require_test(
        unmodelled_candidate.status == amfls::SolverStatus::work_limit &&
            std::isinf(unmodelled_candidate.energy_error_upper_bound) &&
            unmodelled_candidate.ridge_correction_disposition ==
                amfls::RidgeCorrectionDisposition::skipped_nonfinite_candidate &&
            unmodelled_statistics.ridge_correction_a_columns == 0 &&
            unmodelled_statistics.ridge_correction_at_columns == 0,
        "unmodelled custom callback cannot certify or correct");
    check_phase_accounting(
        unmodelled_statistics, "unmodelled custom validator");

    // A nominally zero ridge target has an undefined relative energy ratio.
    // With only a conservative normwise callback model, an exactly zero
    // transpose product cannot be distinguished from cancellation below its
    // error radius, so it must not be promoted to strict stationarity.
    DenseTestOperator null_rhs_operator(2, 1, {1.0, 0.0});
    const std::vector<double> null_rhs{0.0, 1.0};
    amfls::LsqrOptions null_rhs_lsqr_options;
    null_rhs_lsqr_options.regularization = 1.0;
    null_rhs_lsqr_options.tolerance = 1e-12;
    const auto null_rhs_lsqr = amfls::solve_lsqr(
        null_rhs_operator, null_rhs.data(), null_rhs_lsqr_options);
    require_test(
        null_rhs_lsqr.solution == std::vector<double>{0.0} &&
            null_rhs_lsqr.status == amfls::SolverStatus::work_limit &&
            null_rhs_lsqr.stop_reason ==
                amfls::StopReason::maximum_depth &&
            std::isinf(null_rhs_lsqr.relative_energy_error_upper_bound),
        "ridge zero target cannot bypass its callback error radius");
    amfls::LsmrOptions null_rhs_lsmr_options;
    null_rhs_lsmr_options.regularization = 1.0;
    null_rhs_lsmr_options.tolerance = 1e-12;
    const auto null_rhs_lsmr = amfls::solve_lsmr(
        null_rhs_operator, null_rhs.data(), null_rhs_lsmr_options);
    require_test(
        null_rhs_lsmr.solution == std::vector<double>{0.0} &&
            null_rhs_lsmr.status == amfls::SolverStatus::work_limit &&
            null_rhs_lsmr.stop_reason ==
                amfls::StopReason::exhausted_search_space,
        "LSMR ridge zero target stop reason");
    check_lsmr_accounting(null_rhs_lsmr, 0);
    amfls::LsmbOptions null_rhs_lsmb_options;
    null_rhs_lsmb_options.regularization = 1.0;
    null_rhs_lsmb_options.tolerance = 1e-12;
    const auto null_rhs_lsmb = amfls::solve_lsmb(
        null_rhs_operator, null_rhs.data(), null_rhs_lsmb_options);
    require_test(
        null_rhs_lsmb.solution == std::vector<double>{0.0} &&
            null_rhs_lsmb.status == amfls::SolverStatus::work_limit &&
            null_rhs_lsmb.stop_reason ==
                amfls::StopReason::exhausted_search_space,
        "LSMB ridge zero target stop reason");
    check_lsmb_accounting(null_rhs_lsmb, 0);

    // This deliberately stateful callback violates the fixed-operator
    // contract: initialization reports alpha=0, while independent validation
    // does not.  The defensive path must still return an explicit reason.
    StatefulTransposeTestOperator stateful_operator;
    const std::vector<double> stateful_b{1.0};
    amfls::LsmrOptions stateful_options;
    stateful_options.tolerance = 1e-12;
    const auto stateful_lsmr = amfls::solve_lsmr(
        stateful_operator, stateful_b.data(), stateful_options);
    require_test(
        stateful_lsmr.status == amfls::SolverStatus::work_limit &&
            stateful_lsmr.stop_reason ==
                amfls::StopReason::exhausted_search_space,
        "LSMR alpha-zero validation disagreement status");
    StatefulTransposeTestOperator stateful_lsmb_operator;
    amfls::LsmbOptions stateful_lsmb_options;
    stateful_lsmb_options.tolerance = 1e-12;
    const auto stateful_lsmb = amfls::solve_lsmb(
        stateful_lsmb_operator,
        stateful_b.data(),
        stateful_lsmb_options);
    require_test(
        stateful_lsmb.status == amfls::SolverStatus::work_limit &&
            stateful_lsmb.stop_reason ==
                amfls::StopReason::exhausted_search_space,
        "LSMB alpha-zero validation disagreement status");

    DenseTestOperator zero_operator(2, 1, {0.0, 0.0});
    amfls::AmflsOptions zero_operator_options;
    zero_operator_options.tolerance = 1e-12;
    const auto zero_operator_result = amfls::solve_amfls(
        zero_operator, null_rhs.data(), zero_operator_options);
    check_solution(
        zero_operator_result, {0.0}, 0.0, "zero operator minimum norm");

    DenseTestOperator diagonal = diagonal_operator({100.0, 3.0, 2.0, 1.0});
    const std::vector<double> b{100.0, 6.0, 6.0, 4.0};
    const std::vector<double> expected{1.0, 2.0, 3.0, 4.0};

    amfls::LsqrOptions lsqr_options;
    lsqr_options.tolerance = 1e-11;
    lsqr_options.maximum_iterations = 8;
    const auto lsqr = amfls::solve_lsqr(diagonal, b.data(), lsqr_options);
    check_solution(lsqr, expected, 1e-9, "LSQR");
    require_test(
        lsqr.statistics.search_a_block_calls ==
                lsqr.statistics.search_a_columns &&
            lsqr.statistics.search_at_block_calls ==
                lsqr.statistics.search_at_columns &&
            lsqr.statistics.search_a_columns == lsqr.iterations &&
            lsqr.statistics.search_at_columns == lsqr.iterations,
        "scalar LSQR must not prepare a transpose step after success");
    check_phase_accounting(lsqr.statistics, "LSQR");

    DenseTestOperator one_mode_lsqr_operator = diagonal_operator({2.0});
    const std::vector<double> one_mode_lsqr_b{4.0};
    amfls::LsqrOptions one_mode_lsqr_options;
    one_mode_lsqr_options.tolerance = 1e-12;
    one_mode_lsqr_options.maximum_iterations = 4;
    const auto one_mode_lsqr = amfls::solve_lsqr(
        one_mode_lsqr_operator,
        one_mode_lsqr_b.data(),
        one_mode_lsqr_options);
    check_solution(one_mode_lsqr, {2.0}, 1e-14, "one-mode LSQR");
    require_test(
        one_mode_lsqr.iterations == 1 &&
            one_mode_lsqr.statistics.search_a_columns == 1 &&
            one_mode_lsqr.statistics.search_at_columns == 1,
        "exact beta breakdown must not schedule an unused LSQR A*");

    amfls::LsmrOptions lsmr_options;
    lsmr_options.tolerance = 1e-11;
    lsmr_options.maximum_iterations = 8;
    const auto lsmr = amfls::solve_lsmr(
        diagonal, b.data(), lsmr_options);
    check_solution(lsmr, expected, 1e-9, "LSMR");
    check_lsmr_accounting(lsmr, lsmr.iterations);

    amfls::LsmrOptions one_mode_lsmr_options;
    one_mode_lsmr_options.tolerance = 1e-12;
    one_mode_lsmr_options.maximum_iterations = 4;
    const auto one_mode_lsmr = amfls::solve_lsmr(
        one_mode_lsqr_operator,
        one_mode_lsqr_b.data(),
        one_mode_lsmr_options);
    check_solution(one_mode_lsmr, {2.0}, 1e-14, "one-mode LSMR");
    check_one_step_exact_beta_accounting(one_mode_lsmr, "LSMR");

    amfls::LsmbOptions lsmb_options;
    lsmb_options.tolerance = 1e-11;
    lsmb_options.maximum_iterations = 8;
    const auto lsmb = amfls::solve_lsmb(
        diagonal, b.data(), lsmb_options);
    check_solution(lsmb, expected, 1e-9, "LSMB");
    check_lsmb_accounting(lsmb, lsmb.iterations);

    amfls::LsmbOptions one_mode_lsmb_options;
    one_mode_lsmb_options.tolerance = 1e-12;
    one_mode_lsmb_options.maximum_iterations = 4;
    const auto one_mode_lsmb = amfls::solve_lsmb(
        one_mode_lsqr_operator,
        one_mode_lsqr_b.data(),
        one_mode_lsmb_options);
    check_solution(one_mode_lsmb, {2.0}, 1e-14, "one-mode LSMB");
    check_one_step_exact_beta_accounting(one_mode_lsmb, "LSMB");

    // All Krylov solvers use the same certificate cadence without changing
    // their search recurrences.  A normal nontrivial path starts at depth one
    // rather than paying for x=0.
    require_test(
        amfls::math::is_certificate_checkpoint(1, 0.0) &&
            amfls::math::is_certificate_checkpoint(2, 0.0) &&
            amfls::math::is_certificate_checkpoint(3, 0.0) &&
            amfls::math::is_certificate_checkpoint(4, 0.0) &&
            !amfls::math::is_certificate_checkpoint(5, 0.0) &&
            amfls::math::is_certificate_checkpoint(6, 0.0) &&
            !amfls::math::is_certificate_checkpoint(7, 0.0) &&
            amfls::math::is_certificate_checkpoint(8, 0.0) &&
            !amfls::math::is_certificate_checkpoint(3, 1.0) &&
            amfls::math::is_certificate_checkpoint(4, 1.0),
        "certificate checkpoint sequence");

    // A positive log contraction may insert a validation before the next
    // static checkpoint.  These two formal-study observations predict
    // 256 + ceil(64 log(3.53e-8/1e-8) / log(3.85e-5/3.53e-8)) = 268.
    amfls::math::CertificateCheckpointSchedule ordinary_prediction(
        0.0, 1e-8);
    amfls::LeastSquaresResult previous_ordinary_certificate;
    previous_ordinary_certificate.compatible_backward_error_upper_bound =
        3.85e-5;
    previous_ordinary_certificate.least_squares_backward_error_upper_bound =
        1.0;
    ordinary_prediction.record_evaluation(
        192, previous_ordinary_certificate);
    amfls::LeastSquaresResult current_ordinary_certificate;
    current_ordinary_certificate.compatible_backward_error_upper_bound =
        3.53e-8;
    current_ordinary_certificate.least_squares_backward_error_upper_bound =
        1.0;
    ordinary_prediction.record_evaluation(
        256, current_ordinary_certificate);
    require_test(
        !ordinary_prediction.should_evaluate(267) &&
            ordinary_prediction.should_evaluate(268),
        "ordinary certificate crossing prediction");
    amfls::LeastSquaresResult failed_predicted_certificate =
        current_ordinary_certificate;
    failed_predicted_certificate.compatible_backward_error_upper_bound =
        2.0e-8;
    ordinary_prediction.record_evaluation(
        268, failed_predicted_certificate);
    require_test(
        !ordinary_prediction.should_evaluate(269) &&
            !ordinary_prediction.should_evaluate(383) &&
            ordinary_prediction.should_evaluate(384),
        "failed prediction waits for the static interval endpoint");

    // The two ordinary stopping branches are predicted independently and
    // the earlier valid crossing wins.
    amfls::math::CertificateCheckpointSchedule ordinary_branch_minimum(
        0.0, 1e-10);
    previous_ordinary_certificate.compatible_backward_error_upper_bound =
        1e-4;
    previous_ordinary_certificate.least_squares_backward_error_upper_bound =
        1e-4;
    ordinary_branch_minimum.record_evaluation(
        192, previous_ordinary_certificate);
    current_ordinary_certificate.compatible_backward_error_upper_bound =
        1e-6;
    current_ordinary_certificate.least_squares_backward_error_upper_bound =
        1e-8;
    ordinary_branch_minimum.record_evaluation(
        256, current_ordinary_certificate);
    require_test(
        !ordinary_branch_minimum.should_evaluate(287) &&
            ordinary_branch_minimum.should_evaluate(288),
        "ordinary branch-minimum crossing prediction");

    // Ridge scheduling uses only relative energy.  Invalid or noncontracting
    // observations introduce no checkpoint, so the static cadence remains
    // the fallback upper bound.
    amfls::math::CertificateCheckpointSchedule ridge_prediction(1.0, 1e-8);
    amfls::LeastSquaresResult previous_ridge_certificate;
    previous_ridge_certificate.relative_energy_error_upper_bound = 1e-4;
    previous_ridge_certificate.compatible_backward_error_upper_bound = 1e10;
    previous_ridge_certificate.least_squares_backward_error_upper_bound =
        1e10;
    ridge_prediction.record_evaluation(128, previous_ridge_certificate);
    amfls::LeastSquaresResult current_ridge_certificate;
    current_ridge_certificate.relative_energy_error_upper_bound = 1e-6;
    current_ridge_certificate.compatible_backward_error_upper_bound = 1e-12;
    current_ridge_certificate.least_squares_backward_error_upper_bound =
        1e-12;
    ridge_prediction.record_evaluation(256, current_ridge_certificate);
    require_test(
        !ridge_prediction.should_evaluate(383) &&
            ridge_prediction.should_evaluate(384),
        "ridge relative-energy crossing prediction");

    amfls::math::CertificateCheckpointSchedule invalid_prediction(1.0, 1e-8);
    previous_ridge_certificate.relative_energy_error_upper_bound =
        std::numeric_limits<double>::infinity();
    invalid_prediction.record_evaluation(128, previous_ridge_certificate);
    current_ridge_certificate.relative_energy_error_upper_bound = 1e-4;
    invalid_prediction.record_evaluation(256, current_ridge_certificate);
    require_test(
        !invalid_prediction.should_evaluate(384) &&
            !invalid_prediction.should_evaluate(511) &&
            invalid_prediction.should_evaluate(512),
        "invalid contraction falls back to ridge static cadence");

    DenseTestOperator checkpoint_operator = diagonal_operator(
        {1000.0, 300.0, 90.0, 27.0, 9.0, 4.0,
         2.0, 1.0, 0.5, 0.25, 0.125, 0.0625});
    const std::vector<double> checkpoint_b{
        1.0, -2.0, 3.0, -4.0, 5.0, -6.0,
        7.0, -8.0, 9.0, -10.0, 11.0, -12.0};
    const std::vector<int> expected_checkpoint_depths{1, 2, 3, 4, 6, 7};
    const auto require_checkpoint_trace = [&](
        const amfls::LeastSquaresResult& scheduled,
        const std::string& label) {
        require_test(
            scheduled.status == amfls::SolverStatus::work_limit &&
                scheduled.stop_reason == amfls::StopReason::maximum_depth &&
                scheduled.iterations == 7 &&
                scheduled.trace.size() == expected_checkpoint_depths.size(),
            label + " terminal state");
        for (int index = 0;
             index < static_cast<int>(expected_checkpoint_depths.size());
             ++index) {
            require_test(
                scheduled.trace[index].depth == expected_checkpoint_depths[index],
                label + " checkpoint depth");
        }
        require_test(
            scheduled.statistics.base_validation_a_columns ==
                    static_cast<long long>(expected_checkpoint_depths.size()) &&
                scheduled.statistics.base_validation_at_columns ==
                    static_cast<long long>(expected_checkpoint_depths.size()),
            label + " validation count");
    };

    amfls::LsqrOptions checkpoint_lsqr_options;
    checkpoint_lsqr_options.tolerance = 1e-30;
    checkpoint_lsqr_options.maximum_iterations = 7;
    const auto checkpoint_lsqr_result = amfls::solve_lsqr(
        checkpoint_operator, checkpoint_b.data(), checkpoint_lsqr_options);
    require_checkpoint_trace(checkpoint_lsqr_result, "LSQR");
    require_test(
        checkpoint_lsqr_result.statistics.search_a_columns == 7 &&
            checkpoint_lsqr_result.statistics.search_at_columns == 7,
        "maximum-depth LSQR must not prepare an unused transpose step");

    amfls::LsmrOptions checkpoint_lsmr_options;
    checkpoint_lsmr_options.tolerance = 1e-30;
    checkpoint_lsmr_options.maximum_iterations = 7;
    require_checkpoint_trace(
        amfls::solve_lsmr(
            checkpoint_operator, checkpoint_b.data(), checkpoint_lsmr_options),
        "LSMR");

    amfls::LsmbOptions checkpoint_lsmb_options;
    checkpoint_lsmb_options.tolerance = 1e-30;
    checkpoint_lsmb_options.maximum_iterations = 7;
    require_checkpoint_trace(
        amfls::solve_lsmb(
            checkpoint_operator, checkpoint_b.data(), checkpoint_lsmb_options),
        "LSMB");

    amfls::FixedRbgkOptions checkpoint_fixed_options;
    checkpoint_fixed_options.tolerance = 1e-30;
    checkpoint_fixed_options.auxiliary_width = 0;
    checkpoint_fixed_options.maximum_depth = 7;
    checkpoint_fixed_options.stop_early = false;
    const auto checkpoint_fixed_result = amfls::solve_fixed_rbgk(
        checkpoint_operator, checkpoint_b.data(), checkpoint_fixed_options);
    require_checkpoint_trace(checkpoint_fixed_result, "fixed RBGK p=0");

    // The public regularization coefficient is the square of the `damp`
    // parameter in Fong--Saunders.  This two-mode problem has an analytic
    // ridge solution x_i = a_i b_i / (a_i^2 + lambda).
    DenseTestOperator ridge_diagonal = diagonal_operator({3.0, 1.0});
    const std::vector<double> ridge_b{6.0, 2.0};
    amfls::LsmrOptions ridge_lsmr_options;
    ridge_lsmr_options.regularization = 4.0;
    ridge_lsmr_options.tolerance = 1e-12;
    ridge_lsmr_options.maximum_iterations = 4;
    const auto ridge_lsmr = amfls::solve_lsmr(
        ridge_diagonal, ridge_b.data(), ridge_lsmr_options);
    check_solution(
        ridge_lsmr,
        {18.0 / 13.0, 2.0 / 5.0},
        1e-12,
        "regularized LSMR");
    require_test(
        ridge_lsmr.stop_reason ==
            amfls::StopReason::relative_energy_error,
        "regularized LSMR must use the shared ridge success gate");

    amfls::LsmbOptions ridge_lsmb_options;
    ridge_lsmb_options.regularization = 4.0;
    ridge_lsmb_options.tolerance = 1e-12;
    ridge_lsmb_options.maximum_iterations = 4;
    const auto ridge_lsmb = amfls::solve_lsmb(
        ridge_diagonal, ridge_b.data(), ridge_lsmb_options);
    check_solution(
        ridge_lsmb,
        {18.0 / 13.0, 2.0 / 5.0},
        1e-12,
        "regularized LSMB");
    require_test(
        ridge_lsmb.stop_reason ==
            amfls::StopReason::relative_energy_error,
        "regularized LSMB must use the shared ridge success gate");

    // An inconsistent tall problem exercises the MINRES-on-normal-equations
    // branch rather than only compatible square systems.
    DenseTestOperator inconsistent_tall(
        3, 2, {1.0, 0.0, 1.0, 0.0, 1.0, 1.0});
    const std::vector<double> inconsistent_b{1.0, 2.0, 0.0};
    const auto inconsistent_lsmr = amfls::solve_lsmr(
        inconsistent_tall, inconsistent_b.data(), lsmr_options);
    check_solution(
        inconsistent_lsmr, {0.0, 1.0}, 1e-12, "inconsistent LSMR");
    const auto inconsistent_lsmb = amfls::solve_lsmb(
        inconsistent_tall, inconsistent_b.data(), lsmb_options);
    check_solution(
        inconsistent_lsmb, {0.0, 1.0}, 1e-12, "inconsistent LSMB");

    // A capped run audits the same explicit checkpoint schedule as LSQR:
    // one initial A* search call, then one A/A* search pair and one A/A*
    // validation pair per iteration.
    amfls::LsmrOptions capped_lsmr_options;
    capped_lsmr_options.tolerance = 1e-15;
    capped_lsmr_options.maximum_iterations = 2;
    const auto capped_lsmr = amfls::solve_lsmr(
        diagonal, b.data(), capped_lsmr_options);
    require_test(
        capped_lsmr.status == amfls::SolverStatus::work_limit &&
            capped_lsmr.stop_reason == amfls::StopReason::maximum_depth,
        "capped LSMR status");
    check_lsmr_accounting(capped_lsmr, 2);

    amfls::LsmbOptions capped_lsmb_options;
    capped_lsmb_options.tolerance = 1e-15;
    capped_lsmb_options.maximum_iterations = 2;
    const auto capped_lsmb = amfls::solve_lsmb(
        diagonal, b.data(), capped_lsmb_options);
    require_test(
        capped_lsmb.status == amfls::SolverStatus::work_limit &&
            capped_lsmb.stop_reason == amfls::StopReason::maximum_depth,
        "capped LSMB status");
    check_lsmb_accounting(capped_lsmb, 2);

    // At one step LSMR is MINRES, not CG/LSQR, on the normal equations.
    // For q=A^T b and H=A^T A, its scalar coefficient is
    // (q^T Hq)/||Hq||^2 = 41/365 on this problem.
    DenseTestOperator one_step_operator = diagonal_operator({3.0, 1.0});
    const std::vector<double> one_step_b{1.0, 1.0};
    amfls::LsmrOptions one_step_options;
    one_step_options.tolerance = 1e-15;
    one_step_options.maximum_iterations = 1;
    const auto one_step_lsmr = amfls::solve_lsmr(
        one_step_operator, one_step_b.data(), one_step_options);
    require_near(
        one_step_lsmr.solution[0], 123.0 / 365.0, 1e-14,
        "one-step LSMR first component");
    require_near(
        one_step_lsmr.solution[1], 41.0 / 365.0, 1e-14,
        "one-step LSMR second component");
    check_lsmr_accounting(one_step_lsmr, 1);

    // Golden values obtained by evaluating the public Hallman--Gu MATLAB
    // recurrence with its defaults tau=Inf and sigEst=0.  This audits the
    // cubic bridge point, not merely the LSQR or LSMR endpoint.
    amfls::LsmbOptions one_step_lsmb_options;
    one_step_lsmb_options.tolerance = 1e-15;
    one_step_lsmb_options.maximum_iterations = 1;
    const auto one_step_lsmb = amfls::solve_lsmb(
        one_step_operator, one_step_b.data(), one_step_lsmb_options);
    require_near(
        one_step_lsmb.solution[0],
        0.34110166093631106,
        2e-14,
        "one-step LSMB MATLAB golden first component");
    require_near(
        one_step_lsmb.solution[1],
        0.11370055364543702,
        2e-14,
        "one-step LSMB MATLAB golden second component");
    check_lsmb_accounting(one_step_lsmb, 1);

    // The bridge equation is homogeneous in b.  Squaring its unscaled
    // primitives used to underflow p(0) for this nonzero right-hand side and
    // incorrectly select the LSQR endpoint gamma=0.
    constexpr double tiny_rhs_scale = 1e-200;
    const std::vector<double> tiny_one_step_b{
        tiny_rhs_scale, tiny_rhs_scale};
    const auto tiny_one_step_lsmb = amfls::solve_lsmb(
        one_step_operator,
        tiny_one_step_b.data(),
        one_step_lsmb_options);
    require_near(
        tiny_one_step_lsmb.solution[0] / tiny_rhs_scale,
        one_step_lsmb.solution[0],
        2e-14,
        "one-step LSMB first-component RHS scale invariance");
    require_near(
        tiny_one_step_lsmb.solution[1] / tiny_rhs_scale,
        one_step_lsmb.solution[1],
        2e-14,
        "one-step LSMB second-component RHS scale invariance");
    check_lsmb_accounting(tiny_one_step_lsmb, 1);

    // With ridge coefficient four, H=A^T A+4I and the one-step MINRES
    // coefficient becomes (q^T Hq)/||Hq||^2 = 61/773.  This directly checks
    // the damping rotation's contribution to the solution recurrence.
    amfls::LsmrOptions one_step_ridge_options = one_step_options;
    one_step_ridge_options.regularization = 4.0;
    const auto one_step_ridge_lsmr = amfls::solve_lsmr(
        one_step_operator, one_step_b.data(), one_step_ridge_options);
    require_near(
        one_step_ridge_lsmr.solution[0], 183.0 / 773.0, 1e-14,
        "one-step ridge LSMR first component");
    require_near(
        one_step_ridge_lsmr.solution[1], 61.0 / 773.0, 1e-14,
        "one-step ridge LSMR second component");
    const auto same_audit_value = [](double left, double right) {
        return left == right || (std::isnan(left) && std::isnan(right));
    };
    require_test(
        !one_step_ridge_lsmr.trace.empty() &&
            one_step_ridge_lsmr.trace.back().contract_status ==
                one_step_ridge_lsmr.status &&
            one_step_ridge_lsmr.trace.back().contract_stop_reason ==
                one_step_ridge_lsmr.stop_reason &&
            one_step_ridge_lsmr.trace.back().contract_passed ==
                (one_step_ridge_lsmr.status ==
                 amfls::SolverStatus::success) &&
            same_audit_value(
                one_step_ridge_lsmr.trace.back().solution_norm,
                one_step_ridge_lsmr.solution_norm) &&
            same_audit_value(
                one_step_ridge_lsmr.trace.back().solution_energy_norm,
                one_step_ridge_lsmr.solution_energy_norm) &&
            same_audit_value(
                one_step_ridge_lsmr.trace.back()
                    .augmented_operator_norm_lower_bound,
                one_step_ridge_lsmr
                    .augmented_operator_norm_lower_bound) &&
            same_audit_value(
                one_step_ridge_lsmr.trace.back().energy_error_upper_bound,
                one_step_ridge_lsmr.energy_error_upper_bound) &&
            same_audit_value(
                one_step_ridge_lsmr.trace.back().objective_gap_upper_bound,
                one_step_ridge_lsmr.objective_gap_upper_bound) &&
            one_step_ridge_lsmr.trace.back()
                    .ridge_correction_disposition ==
                one_step_ridge_lsmr.ridge_correction_disposition &&
            same_audit_value(
                one_step_ridge_lsmr.trace.back()
                    .ridge_base_energy_error_upper_bound,
                one_step_ridge_lsmr.ridge_base_energy_error_upper_bound) &&
            same_audit_value(
                one_step_ridge_lsmr.trace.back()
                    .ridge_corrected_energy_error_upper_bound,
                one_step_ridge_lsmr
                    .ridge_corrected_energy_error_upper_bound) &&
            same_audit_value(
                one_step_ridge_lsmr.trace.back().ridge_correction_gamma,
                one_step_ridge_lsmr.ridge_correction_gamma) &&
            same_audit_value(
                one_step_ridge_lsmr.trace.back().ridge_correction_z_h_z,
                one_step_ridge_lsmr.ridge_correction_z_h_z) &&
            same_audit_value(
                one_step_ridge_lsmr.trace.back()
                    .ridge_correction_two_abs_z_t_q,
                one_step_ridge_lsmr.ridge_correction_two_abs_z_t_q) &&
            same_audit_value(
                one_step_ridge_lsmr.trace.back()
                    .ridge_correction_q_norm_squared,
                one_step_ridge_lsmr.ridge_correction_q_norm_squared),
        "iteration record must retain candidate contract and ridge audit");

    // Exact-zero and nonfinite recurrence denominators must return an
    // explicit numerical-breakdown status instead of silently skipping the
    // update.  Exact beta breakdown has a known-zero transpose image and
    // therefore does not schedule an extra A^T callback.
    for (double fragile_scale : {1e-200, 1e308}) {
        DenseTestOperator fragile_operator =
            diagonal_operator({fragile_scale});
        const std::vector<double> fragile_b{1.0};
        const auto fragile_lsmr = amfls::solve_lsmr(
            fragile_operator, fragile_b.data(), one_step_options);
        require_test(
            fragile_lsmr.status ==
                    amfls::SolverStatus::numerical_breakdown &&
                fragile_lsmr.stop_reason ==
                    amfls::StopReason::numerical_breakdown,
            "LSMR recurrence denominator breakdown status");
        if (fragile_scale == 1e-200) {
            check_one_step_exact_beta_accounting(
                fragile_lsmr, "fragile LSMR");
        } else {
            check_lsmr_accounting(fragile_lsmr, 1);
        }
    }

    amfls::FixedRbgkOptions fixed_options;
    fixed_options.tolerance = 1e-11;
    fixed_options.auxiliary_width = 0;
    fixed_options.maximum_depth = 8;
    const auto fixed = amfls::solve_fixed_rbgk(
        diagonal, b.data(), fixed_options);
    check_solution(fixed, expected, 1e-9, "fixed p=0");
    check_phase_accounting(fixed.statistics, "fixed RBGK");
    amfls::FixedRbgkOptions zero_rhs_fixed_options = fixed_options;
    zero_rhs_fixed_options.auxiliary_width = 4;
    const std::vector<double> zero_fixed_rhs(b.size(), 0.0);
    const auto zero_rhs_fixed = amfls::solve_fixed_rbgk(
        diagonal, zero_fixed_rhs.data(), zero_rhs_fixed_options);
    require_test(
        zero_rhs_fixed.status == amfls::SolverStatus::success &&
            zero_rhs_fixed.statistics.gaussian_random_columns == 0 &&
            zero_rhs_fixed.statistics.search_a_columns == 0 &&
            zero_rhs_fixed.statistics.search_at_columns == 0,
        "fixed RBGK zero right-hand side must return without random or "
        "search work");
    amfls::FixedRbgkOptions oversized_fixed_options = fixed_options;
    oversized_fixed_options.auxiliary_width =
        std::numeric_limits<int>::max();
    bool oversized_fixed_rejected = false;
    try {
        (void)amfls::solve_fixed_rbgk(
            diagonal, b.data(), oversized_fixed_options);
    } catch (const std::length_error& error) {
        oversized_fixed_rejected = std::string(error.what()) ==
            "fixed RBGK seed width exceeds LP64 limits";
    }
    require_test(
        oversized_fixed_rejected,
        "fixed RBGK must reject auxiliary-width addition overflow before "
        "matrix allocation");

    amfls::FixedRbgkOptions basis_limited_fixed_options = fixed_options;
    basis_limited_fixed_options.auxiliary_width = 2;
    basis_limited_fixed_options.maximum_basis_size = 2;
    const auto basis_limited_fixed = amfls::solve_fixed_rbgk(
        diagonal, b.data(), basis_limited_fixed_options);
    require_test(
        basis_limited_fixed.auxiliary_width == 1 &&
            basis_limited_fixed.statistics.gaussian_random_columns == 1,
        "fixed RBGK must report the effective width retained by its basis "
        "limit");

    amfls::AmflsOptions adaptive_options;
    adaptive_options.tolerance = 1e-11;
    adaptive_options.maximum_epochs = 5;
    adaptive_options.seed = 11;
    const amfls::LeastSquaresResult adaptive =
        amfls::solve_amfls(diagonal, b.data(), adaptive_options);
    check_solution(adaptive, expected, 1e-9, "AMFLS");
    check_phase_accounting(adaptive.statistics, "AMFLS");
    require_test(adaptive.statistics.search_a_columns > 0, "AMFLS A work counter");
    require_test(adaptive.statistics.search_at_columns > 0, "AMFLS AT work counter");
    bool valid_amfls_checkpoints =
        !adaptive.trace.empty() && adaptive.trace.front().depth == 1;
    for (std::size_t index = 1;
         index < adaptive.trace.size();
         ++index) {
        const amfls::IterationRecord& previous = adaptive.trace[index - 1];
        const amfls::IterationRecord& current = adaptive.trace[index];
        const bool widening_checkpoint =
            current.auxiliary_width > previous.auxiliary_width;
        const bool terminal_checkpoint =
            index + 1 == adaptive.trace.size();
        valid_amfls_checkpoints = valid_amfls_checkpoints &&
            current.depth > previous.depth &&
            (amfls::math::is_certificate_checkpoint(
                 current.depth, adaptive_options.regularization) ||
             widening_checkpoint || terminal_checkpoint);
    }
    require_test(
        valid_amfls_checkpoints,
        "AMFLS evaluates each depth once; an off-cadence checkpoint is a "
        "staged widening candidate or the terminal candidate");

    DenseTestOperator three_cluster = diagonal_operator(
        {8.0, 8.0, 2.0, 2.0, 0.5, 0.5});
    const std::vector<double> three_cluster_b{
        8.0, -16.0, 6.0, -8.0, 2.5, -3.0};
    amfls::AmflsOptions three_cluster_options;
    three_cluster_options.tolerance = 1e-11;
    three_cluster_options.maximum_epochs = 6;
    three_cluster_options.seed = 23;
    const auto three_cluster_result = amfls::solve_amfls(
        three_cluster, three_cluster_b.data(), three_cluster_options);
    check_solution(
        three_cluster_result,
        {1.0, -2.0, 3.0, -4.0, 5.0, -6.0},
        1e-9,
        "AMFLS clustered spectrum");

    // A problem certified by the initial width-one level must return before
    // any random directions are generated.
    DenseTestOperator flat = diagonal_operator({2.0, 2.0, 2.0, 2.0});
    const std::vector<double> flat_b{2.0, -4.0, 6.0, -8.0};
    amfls::AmflsOptions flat_options;
    flat_options.tolerance = 1e-12;
    const auto flat_result =
        amfls::solve_amfls(flat, flat_b.data(), flat_options);
    check_solution(
        flat_result, {1.0, -2.0, 3.0, -4.0}, 1e-12, "AMFLS initial return");
    require_test(
        flat_result.auxiliary_width == 0 &&
            flat_result.statistics.gaussian_random_columns == 0,
        "one-step AMFLS problem must not widen");

    // A positive finite mode below the scale-aware block cutoff exhausts the
    // first GKB frontier.  Both ordinary backward-error branches remain
    // above the requested tolerance, and causal numerical truncation takes
    // priority before any widening-chain admission.
    const double precision_mode = std::ldexp(1.0, -45);
    DenseTestOperator precision_operator = diagonal_operator(
        {1.0, 1.0, 1.0, precision_mode});
    const std::vector<double> precision_b{
        1.0, 0.0, 0.0, precision_mode};
    amfls::AmflsOptions precision_options;
    precision_options.tolerance = std::ldexp(1.0, -49);
    precision_options.maximum_epochs = 4;
    const auto precision_result = amfls::solve_amfls(
        precision_operator, precision_b.data(), precision_options);
    require_test(
        precision_result.status == amfls::SolverStatus::precision_limit &&
            precision_result.stop_reason ==
                amfls::StopReason::precision_limit &&
            std::isfinite(
                precision_result.compatible_backward_error_upper_bound) &&
            std::isfinite(
                precision_result.least_squares_backward_error_upper_bound) &&
            precision_result.compatible_backward_error_upper_bound >
                precision_options.tolerance &&
            precision_result.least_squares_backward_error_upper_bound >
                precision_options.tolerance,
        "AMFLS causal numerical deflation precision-limit status");

    // A user depth cap has higher priority than the same causal numerical
    // deflation evidence.
    amfls::AmflsOptions precision_depth_limited = precision_options;
    precision_depth_limited.maximum_depth = 1;
    const auto precision_depth_result = amfls::solve_amfls(
        precision_operator, precision_b.data(), precision_depth_limited);
    require_test(
        precision_depth_result.status == amfls::SolverStatus::work_limit &&
            precision_depth_result.stop_reason ==
                amfls::StopReason::maximum_depth,
        "AMFLS user resource status precedes precision-limit status");

    // A failed run returns the best candidate actually observed, even
    // when the terminal candidate is worse under the nonmonotone stopping
    // quantity.
    DenseTestOperator nonmonotone = diagonal_operator(
        {10.0, std::sqrt(10.0), 1.0 / std::sqrt(10.0)});
    const std::vector<double> nonmonotone_b{
        0.1, 40.0 / std::sqrt(10.0), 40.0 * std::sqrt(10.0)};
    amfls::AmflsOptions nonmonotone_options;
    nonmonotone_options.tolerance = 1e-15;
    nonmonotone_options.maximum_depth = 2;
    nonmonotone_options.maximum_basis_size = 2;
    nonmonotone_options.seed = 4;
    const auto nonmonotone_result = amfls::solve_amfls(
        nonmonotone, nonmonotone_b.data(), nonmonotone_options);
    double trace_best = std::numeric_limits<double>::infinity();
    for (const auto& record : nonmonotone_result.trace) {
        trace_best = std::min(trace_best, record.backward_error_upper_bound);
    }
    require_near(
        nonmonotone_result.backward_error_upper_bound,
        trace_best,
        1e-14,
        "failed AMFLS run must retain the best observed candidate");

    // If every ridge relative-energy bound is infinite, AMFLS still
    // returns an evaluated candidate rather than a zero-filled placeholder.
    DenseTestOperator infinite_accuracy = diagonal_operator({1e-6, 1e-3});
    const std::vector<double> infinite_b{1.0, 1e-6};
    amfls::AmflsOptions infinite_options;
    infinite_options.regularization = 1e-12;
    infinite_options.tolerance = 1e-15;
    infinite_options.maximum_depth = 1;
    infinite_options.maximum_basis_size = 1;
    const auto infinite_result = amfls::solve_amfls(
        infinite_accuracy, infinite_b.data(), infinite_options);
    require_test(!infinite_result.trace.empty(), "all-infinite AMFLS trace");
    require_near(
        infinite_result.objective,
        infinite_result.trace.front().objective,
        1e-14,
        "all-infinite AMFLS run must return an evaluated candidate");

    // stop_early=false requests the terminal fixed-depth candidate even if a
    // loose contract already passes at depth one.
    DenseTestOperator forced_depth = diagonal_operator({3.0, 2.0, 1.0});
    const std::vector<double> forced_b{3.0, 2.0, 1.0};
    amfls::FixedRbgkOptions forced_options;
    forced_options.tolerance = 0.99;
    forced_options.auxiliary_width = 0;
    forced_options.maximum_depth = 2;
    forced_options.stop_early = false;
    const auto forced_result = amfls::solve_fixed_rbgk(
        forced_depth, forced_b.data(), forced_options);
    require_test(
        forced_result.depth == 2 && forced_result.trace.size() == 2,
        "fixed-depth mode must not stop at the first passing checkpoint");

    // Projection of random columns away from b must be scale safe; squaring
    // this RHS norm underflows even though the vector itself is representable.
    const std::vector<double> tiny_b{3e-200, 2e-200, 1e-200};
    amfls::FixedRbgkOptions tiny_fixed_options;
    tiny_fixed_options.tolerance = 1e-4;
    tiny_fixed_options.auxiliary_width = 1;
    tiny_fixed_options.maximum_depth = 3;
    const auto tiny_fixed = amfls::solve_fixed_rbgk(
        forced_depth, tiny_b.data(), tiny_fixed_options);
    require_test(
        std::isfinite(tiny_fixed.objective),
        "fixed random projection must not underflow the RHS scale");
    amfls::AmflsOptions tiny_amfls_options;
    tiny_amfls_options.tolerance = 1e-4;
    tiny_amfls_options.maximum_auxiliary_width = 1;
    const auto tiny_amfls = amfls::solve_amfls(
        forced_depth, tiny_b.data(), tiny_amfls_options);
    require_test(
        std::isfinite(tiny_amfls.objective),
        "AMFLS random projection must not underflow the RHS scale");

    // At a fixed depth, adding random starting columns cannot increase the
    // projected objective in exact arithmetic.
    DenseTestOperator comparison = diagonal_operator(
        {1000.0, 20.0, 8.0, 4.0, 2.0, 1.0});
    const std::vector<double> comparison_b{1.0, 2.0, -1.0, 3.0, 0.5, -2.0};
    amfls::FixedRbgkOptions narrow;
    narrow.tolerance = 1e-15;
    narrow.auxiliary_width = 0;
    narrow.maximum_depth = 1;
    narrow.stop_early = false;
    amfls::FixedRbgkOptions wide = narrow;
    wide.auxiliary_width = 2;
    wide.seed = 5;
    const auto narrow_result =
        amfls::solve_fixed_rbgk(comparison, comparison_b.data(), narrow);
    const auto wide_result =
        amfls::solve_fixed_rbgk(comparison, comparison_b.data(), wide);
    require_test(
        wide_result.objective <= narrow_result.objective + 1e-11,
        "augmented projected objective must dominate narrow Krylov");

    // Rank-deficient underdetermined system: the right Krylov space lies in
    // range(A^T), so the returned exact solution is minimum norm.
    DenseTestOperator wide_rank_deficient(
        2, 3, {1.0, 0.0, 0.0, 2.0, 0.0, 0.0});
    const std::vector<double> wide_b{1.0, 2.0};
    amfls::FixedRbgkOptions rank_options;
    rank_options.tolerance = 1e-12;
    rank_options.auxiliary_width = 1;
    rank_options.maximum_depth = 3;
    const auto rank_result = amfls::solve_fixed_rbgk(
        wide_rank_deficient, wide_b.data(), rank_options);
    check_solution(rank_result, {1.0, 1.0, 0.0}, 1e-10, "minimum norm");
    amfls::AmflsOptions rank_amfls_options;
    rank_amfls_options.tolerance = 1e-12;
    rank_amfls_options.maximum_epochs = 3;
    const auto rank_amfls_result = amfls::solve_amfls(
        wide_rank_deficient, wide_b.data(), rank_amfls_options);
    check_solution(
        rank_amfls_result,
        {1.0, 1.0, 0.0},
        1e-10,
        "AMFLS minimum norm");
    require_test(
        rank_amfls_result.auxiliary_width == 0,
        "AMFLS minimum-norm solve must not widen unnecessarily");
    amfls::LsmrOptions wide_lsmr_options;
    wide_lsmr_options.tolerance = 1e-12;
    wide_lsmr_options.maximum_iterations = 4;
    const auto wide_lsmr = amfls::solve_lsmr(
        wide_rank_deficient, wide_b.data(), wide_lsmr_options);
    check_solution(
        wide_lsmr, {1.0, 1.0, 0.0}, 1e-10, "LSMR minimum norm");
    amfls::LsmbOptions wide_lsmb_options;
    wide_lsmb_options.tolerance = 1e-12;
    wide_lsmb_options.maximum_iterations = 4;
    const auto wide_lsmb = amfls::solve_lsmb(
        wide_rank_deficient, wide_b.data(), wide_lsmb_options);
    check_solution(
        wide_lsmb, {1.0, 1.0, 0.0}, 1e-10, "LSMB minimum norm");

    bool rejected_negative_lsmb_limit = false;
    try {
        amfls::LsmbOptions invalid_lsmb_options;
        invalid_lsmb_options.maximum_iterations = -1;
        (void)amfls::solve_lsmb(
            wide_rank_deficient, wide_b.data(), invalid_lsmb_options);
    } catch (const std::invalid_argument&) {
        rejected_negative_lsmb_limit = true;
    }
    require_test(
        rejected_negative_lsmb_limit,
        "LSMB must reject a negative iteration limit");

    bool rejected_negative_lsmb_regularization = false;
    try {
        amfls::LsmbOptions invalid_lsmb_options;
        invalid_lsmb_options.regularization = -1.0;
        (void)amfls::solve_lsmb(
            wide_rank_deficient, wide_b.data(), invalid_lsmb_options);
    } catch (const std::invalid_argument&) {
        rejected_negative_lsmb_regularization = true;
    }
    require_test(
        rejected_negative_lsmb_regularization,
        "LSMB must reject negative regularization");

    // Numerical breakdown must not turn a roundoff-sized projection residual
    // into an additional unit vector.  The exact matrix has rank five, while
    // AMFLS advances its rank-revealing variable-width GKB process.
    constexpr int breakdown_rows = 8;
    constexpr int breakdown_cols = 12;
    constexpr int breakdown_rank = 5;
    std::vector<double> breakdown_values(
        breakdown_rows * breakdown_cols, 0.0);
    std::vector<double> breakdown_b(breakdown_rows, 0.125);
    std::vector<double> breakdown_expected(breakdown_cols, 0.0);
    for (int index = 0; index < breakdown_rank; ++index) {
        const double sigma = 1.0 / (index + 1.0);
        const double value = index + 1.0;
        breakdown_values[index + index * breakdown_rows] = sigma;
        breakdown_b[index] = sigma * value;
        breakdown_expected[index] = value;
    }
    DenseTestOperator breakdown_operator(
        breakdown_rows, breakdown_cols, std::move(breakdown_values));
    amfls::AmflsOptions breakdown_options;
    breakdown_options.tolerance = 1e-12;
    breakdown_options.maximum_epochs = 6;
    breakdown_options.seed = 29;
    const auto breakdown_result = amfls::solve_amfls(
        breakdown_operator, breakdown_b.data(), breakdown_options);
    check_solution(
        breakdown_result,
        breakdown_expected,
        1e-10,
        "rank-revealing numerical breakdown");
    require_test(
        breakdown_result.basis_rank <= breakdown_rank,
        "roundoff residuals must not increase the numerical basis rank");

    std::cout << "test_solvers passed\n";
}
