#include "amfls/amfls.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

#include "algorithms/amfls/amfls_internal.hpp"
#include "algorithms/mathematics/checked_arithmetic.hpp"
#include "algorithms/mathematics/krylov/certificate_checkpoint_schedule.hpp"
#include "algorithms/mathematics/krylov/krylov_workspace.hpp"
#include "algorithms/mathematics/linalg/blas_lapack.hpp"
#include "algorithms/mathematics/linalg/matrix.hpp"
#include "algorithms/mathematics/random/gaussian.hpp"

namespace amfls {
namespace {

constexpr const char* amfls_counter_overflow =
    "AMFLS statistics exceed signed 64-bit range";

int default_epochs(int dimension) {
    int epochs = 1;
    int covered = 1;
    while (covered < dimension && epochs < 30) {
        covered *= 2;
        ++epochs;
    }
    return epochs;
}

int next_certificate_checkpoint(
    int current_depth,
    int maximum_depth,
    const math::CertificateCheckpointSchedule& schedule) {
    int checkpoint = current_depth + 1;
    while (checkpoint < maximum_depth &&
           !schedule.should_evaluate(checkpoint)) {
        ++checkpoint;
    }
    return checkpoint;
}

double active_certificate(
    const LeastSquaresResult& result,
    double regularization) noexcept {
    return regularization > 0.0
        ? result.relative_energy_error_upper_bound
        : result.backward_error_upper_bound;
}

}  // namespace

LeastSquaresResult solve_amfls(
    const MatrixOperator& matrix,
    const double* right_hand_side,
    const AmflsOptions& options) {
    math::validate_common_inputs(
        matrix, right_hand_side, options.regularization, options.tolerance);
    if (!std::isfinite(options.failure_probability) ||
        options.failure_probability <= 0.0 ||
        options.failure_probability >= 1.0) {
        throw std::invalid_argument(
            "failure_probability must lie strictly between zero and one");
    }
    if (options.maximum_epochs < 0 || options.maximum_depth < 0 ||
        options.maximum_auxiliary_width < 0 ||
        options.maximum_basis_size < 0) {
        throw std::invalid_argument("work limits must be nonnegative");
    }

    const auto total_start = std::chrono::steady_clock::now();
    RunStatistics statistics;
    math::BlockGolubKahanWorkspace workspace(
        matrix,
        right_hand_side,
        options.regularization,
        options.maximum_basis_size,
        statistics);

    const int dimension = std::min(matrix.rows(), matrix.cols());
    const int maximum_depth = options.maximum_depth > 0
        ? std::min(options.maximum_depth, dimension)
        : dimension;
    const int maximum_auxiliary_width = std::min(
        options.maximum_auxiliary_width > 0
            ? options.maximum_auxiliary_width
            : std::max(0, workspace.basis_limit() - 1),
        std::max(0, workspace.basis_limit() - 1));
    const int maximum_widening_epochs = options.maximum_epochs > 0
        ? options.maximum_epochs
        : default_epochs(dimension);

    std::vector<IterationRecord> trace;
    math::CertificateCheckpointSchedule checkpoint_schedule(
        options.regularization, options.tolerance);
    int evaluations = 0;
    int last_evaluated_depth = -1;
    int auxiliary_width = 0;
    LeastSquaresResult result;
    LeastSquaresResult best_result;
    double best_certificate = std::numeric_limits<double>::infinity();
    bool has_evaluated_candidate = false;

    const auto record_candidate = [&](
        LeastSquaresResult candidate,
        bool train_checkpoint_schedule) {
        if (candidate.trace.size() != 1) {
            throw std::logic_error(
                "AMFLS candidate must contain exactly one trace record");
        }
        if (candidate.depth == last_evaluated_depth) {
            throw std::logic_error(
                "AMFLS evaluated one checkpoint more than once");
        }
        last_evaluated_depth = candidate.depth;
        if (train_checkpoint_schedule) {
            checkpoint_schedule.record_evaluation(candidate.depth, candidate);
        }
        trace.push_back(candidate.trace.front());
        candidate.trace.clear();
        ++evaluations;

        const double certificate =
            active_certificate(candidate, options.regularization);
        const bool better = !has_evaluated_candidate ||
            (std::isfinite(certificate) &&
             !std::isfinite(best_certificate)) ||
            certificate < best_certificate;
        if (better) {
            best_certificate = certificate;
            best_result = candidate;
        }
        has_evaluated_candidate = true;
        result = std::move(candidate);
    };

    const auto evaluate_checkpoint = [&](bool train_checkpoint_schedule) {
        record_candidate(workspace.evaluate(
            options.tolerance, evaluations, auxiliary_width),
            train_checkpoint_schedule);
    };

    const auto finish = [&](LeastSquaresResult final_result) {
        final_result.iterations = workspace.depth();
        final_result.trace = trace;
        statistics.total_seconds = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - total_start).count();
        final_result.statistics = statistics;
        return final_result;
    };

    // The unique AMFLS process begins at width one.  Initialization performs
    // no operator work; the first level is completed before the first and only
    // evaluation of the depth-one checkpoint.
    workspace.initialize(math::Matrix(matrix.rows(), 0));
    if (workspace.can_advance() && maximum_depth > 0) {
        (void)workspace.advance_level();
    }
    evaluate_checkpoint(true);
    if (math::candidate_validation_failed_numerically(result) ||
        result.status == SolverStatus::success) {
        return finish(result);
    }

    bool continuation_truncated_numerically =
        math::step_has_numerical_rank_truncation(
            workspace.last_step_audit());
    bool search_broke_numerically =
        math::step_has_nonfinite_breakdown(
            workspace.last_step_audit());

    detail::ProgressHistory deepen_history;
    detail::MatchedHorizonPlan width_plan;
    int widening_epoch = 0;
    std::uint64_t random_columns_generated = 0;
    std::vector<double> normalized_right_hand_side;
    bool right_hand_side_direction_initialized = false;
    const auto gaussian_block_orthogonal_to_right_hand_side = [&](
        int columns,
        std::uint64_t first_column) {
        // Generate first so that the first widening stage retains the exact
        // operation order of gaussian_matrix_orthogonal_to.  The RHS is fixed
        // for the solve, hence its normalized direction can then be reused by
        // every later cumulative-prefix increment without changing the
        // counter-addressed Gaussian stream.
        math::Matrix random = math::gaussian_matrix(
            matrix.rows(),
            columns,
            options.seed,
            options.stream,
            first_column);
        if (!right_hand_side_direction_initialized) {
            const double right_hand_side_norm = math::vector_norm(
                right_hand_side, matrix.rows());
            if (right_hand_side_norm != 0.0) {
                normalized_right_hand_side.resize(matrix.rows());
                for (int row = 0; row < matrix.rows(); ++row) {
                    normalized_right_hand_side[row] =
                        right_hand_side[row] / right_hand_side_norm;
                }
            }
            right_hand_side_direction_initialized = true;
        }
        if (normalized_right_hand_side.empty()) {
            return random;
        }
        for (int col = 0; col < columns; ++col) {
            const double coefficient = math::vector_dot(
                normalized_right_hand_side.data(),
                random.column_data(col),
                matrix.rows());
            for (int row = 0; row < matrix.rows(); ++row) {
                random(row, col) -= coefficient *
                    normalized_right_hand_side[row];
            }
        }
        return random;
    };
    bool feedback_pending = false;
    int feedback_checkpoint_target_depth = 0;
    int widening_stage_start_basis_rank = 0;
    int widening_stage_origin_width = 0;
    int widening_stage_target_width = 0;

    while (true) {
        // Begin at the preceding checkpoint boundary.  Terminal checks are
        // included whenever this iteration proceeds to another checkpoint.
        if (search_broke_numerically) {
            best_result.status = SolverStatus::numerical_breakdown;
            best_result.stop_reason = StopReason::numerical_breakdown;
            return finish(best_result);
        }
        if (workspace.basis_rank() >= workspace.basis_limit()) {
            best_result.status = SolverStatus::basis_limit;
            best_result.stop_reason = StopReason::maximum_basis;
            return finish(best_result);
        }
        if (workspace.depth() >= maximum_depth) {
            best_result.status = SolverStatus::work_limit;
            best_result.stop_reason = StopReason::maximum_depth;
            return finish(best_result);
        }
        if (continuation_truncated_numerically) {
            best_result.status = SolverStatus::precision_limit;
            best_result.stop_reason = StopReason::precision_limit;
            return finish(best_result);
        }

        // The interval begins at the preceding evaluated candidate so that
        // every scheduling computation is charged to the action it precedes.
        const double controller_interval_certificate =
            active_certificate(result, options.regularization);

        const int incumbent_frontier_width =
            workspace.active_left_width();
        const int remaining_basis_rank = std::max(
            0, workspace.basis_limit() - workspace.basis_rank());
        const int available_basis_increment = std::max(
            0,
            remaining_basis_rank - incumbent_frontier_width);
        const int remaining_auxiliary_width = std::max(
            0, maximum_auxiliary_width - auxiliary_width);
        const int width_cap = std::min(
            available_basis_increment, remaining_auxiliary_width);
        const bool complete_feedback_checkpoint_available =
            maximum_depth - workspace.depth() >= 2;
        const int feedback_checkpoint_depth =
            complete_feedback_checkpoint_available
            ? workspace.depth() + 2
            : workspace.depth();
        const int levels_to_feedback_checkpoint =
            feedback_checkpoint_depth - workspace.depth();
        const int feedback_feasible_width_increment =
            complete_feedback_checkpoint_available
            ? detail::maximum_feedback_feasible_width_increment(
                  workspace.basis_rank(),
                  workspace.basis_limit(),
                  incumbent_frontier_width,
                  levels_to_feedback_checkpoint)
            : 0;
        const int feasible_width_cap = std::min(
            width_cap, feedback_feasible_width_increment);
        if (width_plan.end_basis_rank >= 0 &&
            detail::remaining_matched_horizon(
                width_plan, workspace.basis_rank()) == 0) {
            detail::reset_matched_horizon_plan(width_plan);
        }
        const int remaining_horizon =
            detail::remaining_matched_horizon(
                width_plan, workspace.basis_rank());
        const bool matched_horizon_active = remaining_horizon > 0;
        const bool collecting_matched_cost = matched_horizon_active &&
            !feedback_pending && width_plan.cost_sample_count < 2;
        const bool width_decision_ready = !feedback_pending &&
            !collecting_matched_cost;
        const bool widening_stage_available =
            width_decision_ready &&
            widening_epoch < maximum_widening_epochs;
        const int next_prefix_increment = widening_stage_available
            ? detail::next_dyadic_active_width_increment(
                  incumbent_frontier_width,
                  auxiliary_width,
                  maximum_auxiliary_width,
                  feasible_width_cap)
            : 0;

        int requested_width = 0;
        int requested_matched_horizon = 0;
        if (widening_stage_available && next_prefix_increment > 0) {
            const int target_active_width =
                incumbent_frontier_width + next_prefix_increment;
            const detail::WidthDecision decision = matched_horizon_active
                ? detail::make_horizon_width_candidate(
                      next_prefix_increment,
                      remaining_horizon,
                      workspace.depth(),
                      maximum_depth,
                      workspace.basis_rank(),
                      workspace.basis_limit(),
                      incumbent_frontier_width,
                      width_plan.level_operator_seconds,
                      width_plan.level_local_seconds,
                      width_plan.cost_sample_count,
                      matrix.relative_block_product_cost(
                          incumbent_frontier_width),
                      matrix.relative_block_product_cost(
                          target_active_width),
                      width_plan.origin_active_width > 1)
                : detail::make_initial_width_candidate(
                      next_prefix_increment,
                      controller_interval_certificate,
                      options.tolerance,
                      workspace.depth(),
                      maximum_depth,
                      workspace.basis_rank(),
                      workspace.basis_limit(),
                      incumbent_frontier_width,
                      deepen_history,
                      matrix.relative_block_product_cost(
                          incumbent_frontier_width),
                      matrix.relative_block_product_cost(
                          target_active_width));
            if (decision.candidate) {
                requested_width = next_prefix_increment;
                if (!matched_horizon_active) {
                    requested_matched_horizon = decision.matched_horizon;
                }
            }
        }
        if (matched_horizon_active && width_decision_ready &&
            requested_width == 0) {
            detail::reset_matched_horizon_plan(width_plan);
        }

        if (requested_width > 0) {
            feedback_checkpoint_target_depth = feedback_checkpoint_depth;
            widening_stage_start_basis_rank = workspace.basis_rank();
            widening_stage_origin_width = matched_horizon_active
                ? width_plan.origin_active_width
                : incumbent_frontier_width;
            widening_stage_target_width =
                incumbent_frontier_width + requested_width;
            detail::reset_progress_history(deepen_history);
            checkpoint_schedule = math::CertificateCheckpointSchedule(
                options.regularization, options.tolerance);
            math::Matrix random =
                gaussian_block_orthogonal_to_right_hand_side(
                requested_width,
                random_columns_generated);
            math::checked_counter_add(
                statistics.gaussian_random_block_requests,
                1,
                amfls_counter_overflow);
            math::checked_counter_add(
                statistics.gaussian_random_columns,
                requested_width,
                amfls_counter_overflow);
            math::checked_counter_add(
                statistics.gaussian_random_values,
                math::checked_nonnegative_multiply(
                    matrix.rows(),
                    requested_width,
                    amfls_counter_overflow),
                amfls_counter_overflow);
            random_columns_generated = math::checked_add(
                random_columns_generated,
                static_cast<std::uint64_t>(requested_width),
                "AMFLS Gaussian column offset exceeds uint64 range");
            auxiliary_width = math::checked_add(
                auxiliary_width,
                requested_width,
                "AMFLS auxiliary width exceeds LP64 limits");
            ++widening_epoch;

            (void)workspace.append_left_seed(std::move(random));
            bool advanced = false;
            if (workspace.can_advance()) {
                advanced = workspace.advance_level();
            }

            if (advanced) {
                if (requested_matched_horizon > 0) {
                    (void)detail::start_matched_horizon_plan(
                        width_plan,
                        widening_stage_start_basis_rank,
                        workspace.basis_limit(),
                        requested_matched_horizon,
                        widening_stage_origin_width,
                        widening_stage_target_width);
                } else {
                    detail::reset_matched_horizon_costs(
                        width_plan, widening_stage_target_width);
                }
                evaluate_checkpoint(false);
                continuation_truncated_numerically =
                    math::step_has_numerical_rank_truncation(
                        workspace.last_step_audit());
                search_broke_numerically =
                    math::step_has_nonfinite_breakdown(
                        workspace.last_step_audit());
                if (math::candidate_validation_failed_numerically(result) ||
                    result.status == SolverStatus::success) {
                    return finish(result);
                }
                const bool reliable_trial =
                    !continuation_truncated_numerically &&
                    !search_broke_numerically &&
                    workspace.basis_rank() >
                        widening_stage_start_basis_rank &&
                    workspace.active_left_width() ==
                        widening_stage_target_width;
                if (!reliable_trial ||
                    detail::remaining_matched_horizon(
                        width_plan, workspace.basis_rank()) == 0) {
                    detail::reset_matched_horizon_plan(width_plan);
                    feedback_pending = false;
                } else {
                    // The next fixed-width level supplies the first actual
                    // cost observation at the new width.  Its certificate
                    // change is not used as ordinary convergence evidence.
                    feedback_pending = true;
                }
            } else {
                continuation_truncated_numerically =
                    math::step_has_numerical_rank_truncation(
                        workspace.last_step_audit());
                search_broke_numerically =
                    math::step_has_nonfinite_breakdown(
                        workspace.last_step_audit());
                feedback_pending = false;
                detail::reset_matched_horizon_plan(width_plan);
            }

            continue;
        }

        if (workspace.can_advance()) {
            const int starting_depth = workspace.depth();
            const int starting_basis_rank = workspace.basis_rank();
            const bool matched_cost_interval = !feedback_pending &&
                detail::remaining_matched_horizon(
                    width_plan, starting_basis_rank) > 0 &&
                width_plan.cost_sample_count < 2;
            const int target_depth = feedback_pending
                ? feedback_checkpoint_target_depth
                : matched_cost_interval
                    ? std::min(starting_depth + 1, maximum_depth)
                    : next_certificate_checkpoint(
                          starting_depth,
                          maximum_depth,
                          checkpoint_schedule);
            const auto search_interval_start =
                std::chrono::steady_clock::now();
            const double search_a_seconds_before = statistics.a_seconds;
            const double search_at_seconds_before = statistics.at_seconds;

            while (workspace.can_advance() &&
                   workspace.depth() < target_depth) {
                const bool advanced = workspace.advance_level();
                continuation_truncated_numerically =
                    math::step_has_numerical_rank_truncation(
                        workspace.last_step_audit());
                search_broke_numerically =
                    math::step_has_nonfinite_breakdown(
                        workspace.last_step_audit());
                if (!advanced || continuation_truncated_numerically ||
                    search_broke_numerically) {
                    break;
                }
            }
            const double interval_search_seconds =
                std::chrono::duration<double>(
                    std::chrono::steady_clock::now() -
                    search_interval_start).count();
            const double interval_operator_seconds = std::max(
                0.0,
                statistics.a_seconds - search_a_seconds_before +
                    statistics.at_seconds - search_at_seconds_before);
            const double interval_local_seconds = std::max(
                0.0,
                interval_search_seconds - interval_operator_seconds);

            if (workspace.depth() > starting_depth) {
                evaluate_checkpoint(
                    !feedback_pending && !matched_cost_interval);
                const double current_certificate =
                    active_certificate(result, options.regularization);
                if (math::candidate_validation_failed_numerically(result) ||
                    result.status == SolverStatus::success) {
                    return finish(result);
                }
                if (feedback_pending &&
                    workspace.depth() >= feedback_checkpoint_target_depth) {
                    feedback_pending = false;
                    const bool reliable_feedback =
                        !continuation_truncated_numerically &&
                        !search_broke_numerically &&
                        workspace.basis_rank() > starting_basis_rank &&
                        workspace.basis_rank() >
                            widening_stage_start_basis_rank &&
                        incumbent_frontier_width ==
                            widening_stage_target_width &&
                        workspace.active_left_width() ==
                            widening_stage_target_width;
                    const bool recorded = reliable_feedback &&
                        detail::record_matched_horizon_cost(
                            width_plan,
                            incumbent_frontier_width,
                            workspace.active_left_width(),
                            workspace.depth() - starting_depth,
                            interval_operator_seconds,
                            interval_local_seconds);
                    if (!recorded ||
                        detail::remaining_matched_horizon(
                            width_plan, workspace.basis_rank()) == 0) {
                        detail::reset_matched_horizon_plan(width_plan);
                    }
                } else if (matched_cost_interval) {
                    const bool reliable_cost_interval =
                        !continuation_truncated_numerically &&
                        !search_broke_numerically &&
                        workspace.basis_rank() > starting_basis_rank &&
                        incumbent_frontier_width ==
                            workspace.active_left_width();
                    const bool recorded = reliable_cost_interval &&
                        detail::record_matched_horizon_cost(
                            width_plan,
                            incumbent_frontier_width,
                            workspace.active_left_width(),
                            workspace.depth() - starting_depth,
                            interval_operator_seconds,
                            interval_local_seconds);
                    if (!recorded ||
                        detail::remaining_matched_horizon(
                            width_plan, workspace.basis_rank()) == 0) {
                        detail::reset_matched_horizon_plan(width_plan);
                    }
                } else if (!feedback_pending) {
                    detail::record_deepen_progress(
                        deepen_history,
                        controller_interval_certificate,
                        current_certificate,
                        incumbent_frontier_width,
                        workspace.active_left_width(),
                        workspace.depth() - starting_depth,
                        interval_operator_seconds,
                        interval_local_seconds);
                }
                continue;
            }

            // A rank-zero right half step leaves the immutable recurrence T
            // and the candidate unchanged.  The next pass assigns either the
            // numerical-truncation status or exact search-space exhaustion.
            if (feedback_pending || matched_cost_interval) {
                feedback_pending = false;
                detail::reset_matched_horizon_plan(width_plan);
            }
            continue;
        }

        StopReason terminal_reason = StopReason::exhausted_search_space;
        if (!feedback_pending &&
            widening_epoch >= maximum_widening_epochs &&
            auxiliary_width < maximum_auxiliary_width &&
            available_basis_increment > 0) {
            terminal_reason = StopReason::maximum_epochs;
        }
        best_result.status = SolverStatus::work_limit;
        best_result.stop_reason = terminal_reason;
        return finish(best_result);
    }
}

}  // namespace amfls
