#include "amfls/fixed_rbgk.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <stdexcept>
#include <utility>
#include <vector>

#include "algorithms/mathematics/checked_arithmetic.hpp"
#include "algorithms/mathematics/krylov/certificate_checkpoint_schedule.hpp"
#include "algorithms/mathematics/krylov/krylov_workspace.hpp"
#include "algorithms/mathematics/linalg/matrix.hpp"
#include "algorithms/mathematics/random/gaussian.hpp"

namespace amfls {

LeastSquaresResult solve_fixed_rbgk(
    const MatrixOperator& matrix,
    const double* right_hand_side,
    const FixedRbgkOptions& options) {
    math::validate_common_inputs(
        matrix, right_hand_side, options.regularization, options.tolerance);
    if (!std::isfinite(options.failure_probability) ||
        options.failure_probability <= 0.0 ||
        options.failure_probability >= 1.0) {
        throw std::invalid_argument(
            "failure_probability must lie strictly between zero and one");
    }
    if (options.auxiliary_width < 0) {
        throw std::invalid_argument("auxiliary_width must be nonnegative");
    }
    if (options.maximum_depth <= 0) {
        throw std::invalid_argument("maximum_depth must be positive");
    }
    if (options.maximum_basis_size < 0) {
        throw std::invalid_argument("maximum_basis_size must be nonnegative");
    }
    (void)math::checked_add(
        options.auxiliary_width,
        1,
        "fixed RBGK seed width exceeds LP64 limits");

    const auto total_start = std::chrono::steady_clock::now();
    RunStatistics statistics;
    math::BlockGolubKahanWorkspace workspace(
        matrix,
        right_hand_side,
        options.regularization,
        options.maximum_basis_size,
        statistics);
    const bool zero_right_hand_side =
        workspace.right_hand_side_norm() == 0.0;
    const int auxiliary_width = zero_right_hand_side
        ? 0
        : std::min(
              options.auxiliary_width,
              std::max(0, workspace.basis_limit() - 1));

    std::vector<IterationRecord> trace;
    LeastSquaresResult result;
    int depth = 0;
    int evaluated_depth = -1;
    math::CertificateCheckpointSchedule checkpoint_schedule(
        options.regularization, options.tolerance);
    const auto evaluate_candidate = [&] {
        result = workspace.evaluate(
            options.tolerance, 0, auxiliary_width);
        checkpoint_schedule.record_evaluation(depth, result);
        evaluated_depth = depth;
        trace.push_back(result.trace.front());
        result.trace.clear();
    };
    const auto evaluate_terminal_candidate = [&] {
        if (evaluated_depth != depth) {
            evaluate_candidate();
        }
        return math::candidate_validation_failed_numerically(result);
    };

    math::Matrix initial_auxiliary_left(
        matrix.rows(), auxiliary_width);
    if (auxiliary_width > 0) {
        initial_auxiliary_left = math::gaussian_matrix(
            matrix.rows(),
            auxiliary_width,
            options.seed,
            options.stream,
            0);
        math::checked_counter_add(
            statistics.gaussian_random_block_requests,
            1,
            "fixed RBGK statistics exceed signed 64-bit range");
        math::checked_counter_add(
            statistics.gaussian_random_columns,
            auxiliary_width,
            "fixed RBGK statistics exceed signed 64-bit range");
        math::checked_counter_add(
            statistics.gaussian_random_values,
            math::checked_nonnegative_multiply(
                matrix.rows(),
                auxiliary_width,
                "fixed RBGK statistics exceed signed 64-bit range"),
            "fixed RBGK statistics exceed signed 64-bit range");
    }

    workspace.initialize(std::move(initial_auxiliary_left));
    if (workspace.can_advance()) {
        (void)workspace.advance_level();
    }
    depth = workspace.depth();
    evaluate_candidate();
    bool continuation_truncated_numerically =
        math::step_has_numerical_rank_truncation(
            workspace.last_step_audit());
    bool search_broke_numerically =
        math::step_has_nonfinite_breakdown(
            workspace.last_step_audit());

    while (true) {
        if (math::candidate_validation_failed_numerically(result)) {
            break;
        }
        if (result.status == SolverStatus::success && options.stop_early) {
            break;
        }
        if (search_broke_numerically) {
            if (evaluate_terminal_candidate()) {
                break;
            }
            result.status = SolverStatus::numerical_breakdown;
            result.stop_reason = StopReason::numerical_breakdown;
            break;
        }
        if (workspace.basis_rank() >= workspace.basis_limit()) {
            if (evaluate_terminal_candidate()) {
                break;
            }
            if (result.status != SolverStatus::success) {
                result.status = SolverStatus::basis_limit;
                result.stop_reason = StopReason::maximum_basis;
            }
            break;
        }
        if (depth >= options.maximum_depth) {
            if (evaluate_terminal_candidate()) {
                break;
            }
            if (result.status != SolverStatus::success) {
                result.status = SolverStatus::work_limit;
                result.stop_reason = StopReason::maximum_depth;
            }
            break;
        }
        if (continuation_truncated_numerically) {
            if (evaluate_terminal_candidate()) {
                break;
            }
            if (result.status != SolverStatus::success) {
                result.status = SolverStatus::precision_limit;
                result.stop_reason = StopReason::precision_limit;
            }
            break;
        }
        if (!workspace.can_advance()) {
            if (evaluate_terminal_candidate()) {
                break;
            }
            if (result.status != SolverStatus::success) {
                result.status = SolverStatus::work_limit;
                result.stop_reason = StopReason::exhausted_search_space;
            }
            break;
        }

        const bool extension_succeeded = workspace.advance_level();
        const math::BlockGolubKahanStepAudit& step =
            workspace.last_step_audit();
        continuation_truncated_numerically =
            math::step_has_numerical_rank_truncation(step);
        search_broke_numerically =
            math::step_has_nonfinite_breakdown(step);
        if (!extension_succeeded) {
            if (!step.right_rank_loss && !search_broke_numerically) {
                throw std::logic_error(
                    "failed fixed extension did not report rank loss");
            }
            continue;
        }
        depth = workspace.depth();
        const bool terminal_depth =
            depth >= options.maximum_depth ||
            workspace.basis_rank() >= workspace.basis_limit() ||
            !workspace.can_advance() ||
            continuation_truncated_numerically ||
            search_broke_numerically;
        if (checkpoint_schedule.should_evaluate(depth) || terminal_depth) {
            evaluate_candidate();
        }
    }

    result.iterations = depth;
    result.depth = depth;
    result.trace = std::move(trace);
    statistics.total_seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - total_start).count();
    result.statistics = statistics;
    return result;
}

}  // namespace amfls
