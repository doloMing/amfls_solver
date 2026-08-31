#pragma once

#include <algorithm>
#include <cmath>
#include <limits>

namespace amfls::detail {

// Raw certificate progress and per-level search cost in the latest two
// ordinary same-width intervals.  Older same-width progress is pooled so the
// controller can distinguish a persistent slowdown from a stable scalar
// trajectory without depending on a checkpoint partition.  Widening trials
// and matched-horizon feedback intervals are never recorded here.
struct ProgressHistory {
    double same_width_log_contractions[2] = {};
    double same_width_level_log_contractions[2] = {};
    int same_width_interval_levels[2] = {};
    double same_width_level_operator_seconds[2] = {
        std::numeric_limits<double>::infinity(),
        std::numeric_limits<double>::infinity()};
    double same_width_level_local_seconds[2] = {
        std::numeric_limits<double>::infinity(),
        std::numeric_limits<double>::infinity()};
    int same_width_level_interval_count = 0;
    double prior_same_width_log_contraction = 0.0;
    int prior_same_width_levels = 0;
    int active_left_width = -1;
};

inline void reset_progress_history(
    ProgressHistory& history) noexcept {
    history = {};
}

inline void record_deepen_progress(
    ProgressHistory& history,
    double previous,
    double current,
    int interval_start_width = 0,
    int interval_end_width = 0,
    int interval_levels = 1,
    double interval_operator_seconds =
        std::numeric_limits<double>::infinity(),
    double interval_local_seconds =
        std::numeric_limits<double>::infinity()) noexcept {
    if (interval_start_width <= 0 || interval_end_width <= 0 ||
        interval_start_width != interval_end_width) {
        reset_progress_history(history);
        return;
    }
    if (history.active_left_width != interval_start_width) {
        reset_progress_history(history);
        history.active_left_width = interval_start_width;
    }
    if (interval_levels <= 0 || !(previous > 0.0) || !(current > 0.0) ||
        !std::isfinite(previous) || !std::isfinite(current)) {
        return;
    }
    const long double previous_long = static_cast<long double>(previous);
    const long double current_long = static_cast<long double>(current);
    const long double progress = previous > current
        ? std::log(previous_long) - std::log(current_long)
        : 0.0L;
    if (!(progress >= 0.0L) || !std::isfinite(progress)) {
        return;
    }
    const long double level_log_contraction =
        progress / static_cast<long double>(interval_levels);
    const double level_log_contraction_double =
        level_log_contraction > 0.0L &&
            std::isfinite(level_log_contraction)
        ? static_cast<double>(std::min(
              level_log_contraction,
              static_cast<long double>(
                  std::numeric_limits<double>::max())))
        : 0.0;
    const bool finite_interval_cost =
        interval_operator_seconds >= 0.0 &&
        interval_local_seconds >= 0.0 &&
        std::isfinite(interval_operator_seconds) &&
        std::isfinite(interval_local_seconds) &&
        interval_operator_seconds + interval_local_seconds > 0.0 &&
        std::isfinite(interval_operator_seconds + interval_local_seconds);
    const double level_operator_seconds = finite_interval_cost
        ? interval_operator_seconds / static_cast<double>(interval_levels)
        : std::numeric_limits<double>::infinity();
    const double level_local_seconds = finite_interval_cost
        ? interval_local_seconds / static_cast<double>(interval_levels)
        : std::numeric_limits<double>::infinity();
    if (history.same_width_level_interval_count < 2) {
        const int index = history.same_width_level_interval_count;
        history.same_width_log_contractions[index] =
            static_cast<double>(progress);
        history.same_width_level_log_contractions[index] =
            level_log_contraction_double;
        history.same_width_interval_levels[index] = interval_levels;
        history.same_width_level_operator_seconds[index] =
            level_operator_seconds;
        history.same_width_level_local_seconds[index] =
            level_local_seconds;
        ++history.same_width_level_interval_count;
    } else {
        const long double pooled_prior =
            static_cast<long double>(
                history.prior_same_width_log_contraction) +
            static_cast<long double>(
                history.same_width_log_contractions[0]);
        if (history.prior_same_width_levels >
                std::numeric_limits<int>::max() -
                    history.same_width_interval_levels[0] ||
            !std::isfinite(pooled_prior) ||
            pooled_prior > static_cast<long double>(
                std::numeric_limits<double>::max())) {
            history.prior_same_width_log_contraction =
                std::numeric_limits<double>::infinity();
            history.prior_same_width_levels =
                std::numeric_limits<int>::max();
        } else {
            history.prior_same_width_log_contraction =
                static_cast<double>(pooled_prior);
            history.prior_same_width_levels +=
                history.same_width_interval_levels[0];
        }
        history.same_width_log_contractions[0] =
            history.same_width_log_contractions[1];
        history.same_width_log_contractions[1] =
            static_cast<double>(progress);
        history.same_width_level_log_contractions[0] =
            history.same_width_level_log_contractions[1];
        history.same_width_level_log_contractions[1] =
            level_log_contraction_double;
        history.same_width_interval_levels[0] =
            history.same_width_interval_levels[1];
        history.same_width_interval_levels[1] = interval_levels;
        history.same_width_level_operator_seconds[0] =
            history.same_width_level_operator_seconds[1];
        history.same_width_level_operator_seconds[1] =
            level_operator_seconds;
        history.same_width_level_local_seconds[0] =
            history.same_width_level_local_seconds[1];
        history.same_width_level_local_seconds[1] =
            level_local_seconds;
    }
}

// Double the retained active block width along 1, 2, 4, ... .  Generated
// auxiliary columns remain the budget coordinate because rank-deflated
// columns still consumed random work.  Returning zero means that the budget
// is exhausted or the complete next stage does not fit the currently
// available resources.
inline int next_dyadic_active_width_increment(
    int active_left_width,
    int generated_auxiliary_width,
    int maximum_generated_auxiliary_width,
    int available_increment) noexcept {
    if (active_left_width <= 0 || generated_auxiliary_width < 0 ||
        maximum_generated_auxiliary_width <= generated_auxiliary_width ||
        active_left_width > generated_auxiliary_width + 1 ||
        available_increment <= 0) {
        return 0;
    }

    const long long maximum_active_width =
        static_cast<long long>(active_left_width) +
        maximum_generated_auxiliary_width - generated_auxiliary_width;
    const long long current_active_width =
        active_left_width;
    const long long next_active_width = current_active_width >
            maximum_active_width / 2
        ? maximum_active_width
        : std::min(2 * current_active_width, maximum_active_width);
    if (next_active_width <= current_active_width) {
        return 0;
    }
    const long long requested_width =
        next_active_width - current_active_width;
    return requested_width <= available_increment &&
            requested_width <= std::numeric_limits<int>::max()
        ? static_cast<int>(requested_width)
        : 0;
}

inline int ceiling_ratio(int numerator, int denominator) noexcept {
    if (numerator <= 0 || denominator <= 0) {
        return 0;
    }
    return 1 + (numerator - 1) / denominator;
}

// A widening stage consumes its trial level and the immediately following
// feedback level.  Return the largest increment for which that complete path
// fits in the remaining basis in the no-rank-loss case.
inline int maximum_feedback_feasible_width_increment(
    int current_basis_rank,
    int basis_limit,
    int current_frontier_width,
    int levels_to_feedback_checkpoint) noexcept {
    if (current_basis_rank < 0 || basis_limit < current_basis_rank ||
        current_frontier_width < 0 || levels_to_feedback_checkpoint <= 0) {
        return 0;
    }
    const int remaining_basis = basis_limit - current_basis_rank;
    const int maximum_complete_level_width =
        remaining_basis / levels_to_feedback_checkpoint;
    return std::max(
        0, maximum_complete_level_width - current_frontier_width);
}

struct WidthDecision {
    double recent_level_log_contraction = 0.0;
    double prior_level_log_contraction = 0.0;
    double remaining_log_contraction = 0.0;
    int forecast_levels = 0;
    int incumbent_levels = 0;
    int target_levels = 0;
    int target_remaining_levels = 0;
    int matched_horizon = 0;
    double incumbent_estimated_seconds =
        std::numeric_limits<double>::infinity();
    double target_estimated_seconds =
        std::numeric_limits<double>::infinity();
    bool history_mature = false;
    bool prior_history_mature = false;
    bool persistent_slowdown = false;
    bool long_horizon_pressure = false;
    bool widening_necessary = false;
    bool observed_costs_mature = false;
    bool width_depth_admissible = true;
    bool pass_reduction_admissible = false;
    bool stage_feasible = false;
    bool finite_current_certificate = false;
    bool finite_resource_boundary = false;
    bool candidate = false;
};

inline void apply_width_cost_gate(
    WidthDecision& decision,
    int active_left_width,
    int target_width,
    const double* level_operator_seconds,
    const double* level_local_seconds,
    int cost_sample_count,
    double incumbent_relative_cost,
    double target_relative_cost) noexcept {
    if (decision.incumbent_levels <= 0 || decision.target_levels <= 0 ||
        active_left_width <= 0 || target_width <= active_left_width ||
        cost_sample_count != 2 ||
        !(incumbent_relative_cost > 0.0) ||
        !(target_relative_cost > 0.0) ||
        !std::isfinite(incumbent_relative_cost) ||
        !std::isfinite(target_relative_cost)) {
        return;
    }

    const long double width_ratio =
        static_cast<long double>(target_width) /
        static_cast<long double>(active_left_width);
    const long double width_ratio_squared = width_ratio * width_ratio;
    const long double operator_cost_ratio =
        static_cast<long double>(target_relative_cost) /
        static_cast<long double>(incumbent_relative_cost);
    const long double incumbent_operator_cost =
        static_cast<long double>(decision.incumbent_levels) *
        static_cast<long double>(incumbent_relative_cost);
    const long double target_operator_cost =
        static_cast<long double>(decision.target_levels) *
        static_cast<long double>(target_relative_cost);

    long double best_incumbent_seconds =
        std::numeric_limits<long double>::infinity();
    long double worst_target_seconds = 0.0L;
    bool finite_observed_costs = true;
    for (int index = 0; index < cost_sample_count; ++index) {
        const long double operator_seconds = level_operator_seconds[index];
        const long double local_seconds = level_local_seconds[index];
        if (!(operator_seconds >= 0.0L) || !(local_seconds >= 0.0L) ||
            !std::isfinite(operator_seconds) ||
            !std::isfinite(local_seconds) ||
            !(operator_seconds + local_seconds > 0.0L)) {
            finite_observed_costs = false;
            break;
        }
        const long double incumbent_seconds =
            static_cast<long double>(decision.incumbent_levels) *
            (operator_seconds + local_seconds);
        const long double target_local_seconds =
            local_seconds * width_ratio_squared;
        const long double target_seconds =
            static_cast<long double>(decision.target_levels) *
                (operator_seconds * operator_cost_ratio +
                 target_local_seconds) +
            target_local_seconds;
        if (!std::isfinite(incumbent_seconds) ||
            !std::isfinite(target_seconds)) {
            finite_observed_costs = false;
            break;
        }
        best_incumbent_seconds = std::min(
            best_incumbent_seconds, incumbent_seconds);
        worst_target_seconds = std::max(
            worst_target_seconds, target_seconds);
    }
    if (finite_observed_costs) {
        const long double double_max =
            std::numeric_limits<double>::max();
        finite_observed_costs = best_incumbent_seconds <= double_max &&
            worst_target_seconds <= double_max;
        if (finite_observed_costs) {
            decision.incumbent_estimated_seconds =
                static_cast<double>(best_incumbent_seconds);
            decision.target_estimated_seconds =
                static_cast<double>(worst_target_seconds);
        }
    }
    decision.observed_costs_mature = finite_observed_costs;
    decision.pass_reduction_admissible =
        decision.width_depth_admissible &&
        decision.target_levels < decision.incumbent_levels &&
        target_operator_cost < incumbent_operator_cost &&
        decision.observed_costs_mature &&
        decision.target_estimated_seconds <
            decision.incumbent_estimated_seconds;
}

// Compare two widths over one already approved finite right-rank horizon.
// The horizon is never enlarged here.  Operator time follows the MatrixOperator
// batching model, local block algebra is conservatively scaled quadratically,
// and every proposed injection is charged one additional target local level.
inline WidthDecision make_horizon_width_candidate(
    int proposed_stage_increment,
    int remaining_horizon,
    int current_depth,
    int maximum_depth,
    int current_basis_rank,
    int basis_limit,
    int active_left_width,
    const double* level_operator_seconds,
    const double* level_local_seconds,
    int cost_sample_count,
    double incumbent_relative_cost = 1.0,
    double target_relative_cost = 1.0,
    bool require_width_depth_balance = false) noexcept {
    WidthDecision decision;
    decision.stage_feasible = proposed_stage_increment > 0 &&
        active_left_width > 0 &&
        proposed_stage_increment <=
            std::numeric_limits<int>::max() - active_left_width;
    decision.finite_resource_boundary = current_depth >= 0 &&
        maximum_depth >= current_depth && current_basis_rank >= 0 &&
        basis_limit >= current_basis_rank && active_left_width > 0 &&
        remaining_horizon > 0 &&
        remaining_horizon <= basis_limit - current_basis_rank;
    if (!decision.stage_feasible || !decision.finite_resource_boundary) {
        return decision;
    }

    const int target_width =
        active_left_width + proposed_stage_increment;
    decision.matched_horizon = remaining_horizon;
    decision.incumbent_levels = ceiling_ratio(
        remaining_horizon, active_left_width);
    decision.target_levels = ceiling_ratio(
        remaining_horizon, target_width);
    decision.target_remaining_levels = std::min(
        maximum_depth - current_depth,
        (basis_limit - current_basis_rank) / target_width);
    if (decision.target_levels < 2 ||
        decision.target_levels > decision.target_remaining_levels) {
        return decision;
    }
    decision.width_depth_admissible =
        !require_width_depth_balance ||
        decision.target_levels > target_width;

    apply_width_cost_gate(
        decision,
        active_left_width,
        target_width,
        level_operator_seconds,
        level_local_seconds,
        cost_sample_count,
        incumbent_relative_cost,
        target_relative_cost);
    decision.candidate = decision.pass_reduction_admissible;
    return decision;
}

// Open a finite matched-horizon plan only when widening is necessary: either
// both recent intervals have slowed relative to earlier same-width progress,
// or the scalar forecast is longer than the next width can execute with the
// remaining resources.  The subsequent cost comparison remains strict.
inline WidthDecision make_initial_width_candidate(
    int proposed_stage_increment,
    double current_certificate,
    double tolerance,
    int current_depth,
    int maximum_depth,
    int current_basis_rank,
    int basis_limit,
    int active_left_width,
    const ProgressHistory& history,
    double incumbent_relative_cost = 1.0,
    double target_relative_cost = 1.0) noexcept {
    WidthDecision decision;
    decision.finite_current_certificate = current_certificate > 0.0 &&
        tolerance > 0.0 && current_certificate > tolerance &&
        std::isfinite(current_certificate) && std::isfinite(tolerance);
    decision.history_mature =
        history.active_left_width == active_left_width &&
        history.same_width_level_interval_count >= 2;
    if (!decision.finite_current_certificate || !decision.history_mature ||
        current_depth < 0 || maximum_depth < current_depth ||
        current_basis_rank < 0 || basis_limit < current_basis_rank ||
        active_left_width <= 0 || proposed_stage_increment <= 0 ||
        proposed_stage_increment >
            std::numeric_limits<int>::max() - active_left_width) {
        return decision;
    }

    const long long recent_levels =
        static_cast<long long>(history.same_width_interval_levels[0]) +
        static_cast<long long>(history.same_width_interval_levels[1]);
    const long double recent_progress =
        static_cast<long double>(history.same_width_log_contractions[0]) +
        static_cast<long double>(history.same_width_log_contractions[1]);
    if (recent_levels <= 0 || !(recent_progress > 0.0L) ||
        !std::isfinite(recent_progress)) {
        return decision;
    }
    decision.recent_level_log_contraction = static_cast<double>(
        recent_progress / static_cast<long double>(recent_levels));
    decision.remaining_log_contraction =
        std::log(current_certificate) - std::log(tolerance);
    if (!(decision.recent_level_log_contraction > 0.0) ||
        !std::isfinite(decision.recent_level_log_contraction) ||
        !(decision.remaining_log_contraction > 0.0) ||
        !std::isfinite(decision.remaining_log_contraction)) {
        return decision;
    }

    const long double raw_forecast =
        static_cast<long double>(decision.remaining_log_contraction) /
        static_cast<long double>(decision.recent_level_log_contraction);
    if (!(raw_forecast > 0.0L) || !std::isfinite(raw_forecast)) {
        return decision;
    }
    decision.forecast_levels = raw_forecast >=
            static_cast<long double>(std::numeric_limits<int>::max())
        ? std::numeric_limits<int>::max()
        : std::max(1, static_cast<int>(std::ceil(raw_forecast)));

    decision.prior_history_mature =
        history.prior_same_width_levels > 0 &&
        history.prior_same_width_log_contraction > 0.0 &&
        std::isfinite(history.prior_same_width_log_contraction);
    if (decision.prior_history_mature) {
        decision.prior_level_log_contraction =
            history.prior_same_width_log_contraction /
            static_cast<double>(history.prior_same_width_levels);
        decision.persistent_slowdown =
            std::isfinite(decision.prior_level_log_contraction) &&
            history.same_width_level_log_contractions[0] > 0.0 &&
            history.same_width_level_log_contractions[1] > 0.0 &&
            history.same_width_level_log_contractions[0] <
                decision.prior_level_log_contraction &&
            history.same_width_level_log_contractions[1] <
                decision.prior_level_log_contraction;
    }

    const long long raw_horizon =
        static_cast<long long>(active_left_width) *
        static_cast<long long>(decision.forecast_levels);
    const int remaining_basis = basis_limit - current_basis_rank;
    const int matched_horizon = static_cast<int>(std::min(
        raw_horizon, static_cast<long long>(remaining_basis)));

    WidthDecision cost_decision = make_horizon_width_candidate(
        proposed_stage_increment,
        matched_horizon,
        current_depth,
        maximum_depth,
        current_basis_rank,
        basis_limit,
        active_left_width,
        history.same_width_level_operator_seconds,
        history.same_width_level_local_seconds,
        2,
        incumbent_relative_cost,
        target_relative_cost);
    cost_decision.finite_current_certificate =
        decision.finite_current_certificate;
    cost_decision.history_mature = decision.history_mature;
    cost_decision.prior_history_mature =
        decision.prior_history_mature;
    cost_decision.persistent_slowdown =
        decision.persistent_slowdown;
    cost_decision.recent_level_log_contraction =
        decision.recent_level_log_contraction;
    cost_decision.prior_level_log_contraction =
        decision.prior_level_log_contraction;
    cost_decision.remaining_log_contraction =
        decision.remaining_log_contraction;
    cost_decision.forecast_levels = decision.forecast_levels;
    cost_decision.long_horizon_pressure =
        cost_decision.target_remaining_levels > 0 &&
        decision.forecast_levels >
            cost_decision.target_remaining_levels;
    const bool positive_recent_progress =
        history.same_width_level_log_contractions[0] > 0.0 &&
        history.same_width_level_log_contractions[1] > 0.0;
    cost_decision.widening_necessary =
        cost_decision.prior_history_mature &&
        positive_recent_progress &&
        (cost_decision.persistent_slowdown ||
         cost_decision.long_horizon_pressure);
    cost_decision.candidate = cost_decision.candidate &&
        cost_decision.widening_necessary;
    return cost_decision;
}

struct MatchedHorizonPlan {
    int end_basis_rank = -1;
    int origin_active_width = -1;
    int active_left_width = -1;
    double level_operator_seconds[2] = {
        std::numeric_limits<double>::infinity(),
        std::numeric_limits<double>::infinity()};
    double level_local_seconds[2] = {
        std::numeric_limits<double>::infinity(),
        std::numeric_limits<double>::infinity()};
    int cost_sample_count = 0;
};

inline void reset_matched_horizon_plan(
    MatchedHorizonPlan& plan) noexcept {
    plan = {};
}

inline int remaining_matched_horizon(
    const MatchedHorizonPlan& plan,
    int current_basis_rank) noexcept {
    if (plan.end_basis_rank < 0 || current_basis_rank < 0 ||
        current_basis_rank >= plan.end_basis_rank) {
        return 0;
    }
    return plan.end_basis_rank - current_basis_rank;
}

inline bool start_matched_horizon_plan(
    MatchedHorizonPlan& plan,
    int current_basis_rank,
    int basis_limit,
    int matched_horizon,
    int origin_active_width,
    int target_active_width) noexcept {
    if (current_basis_rank < 0 || basis_limit < current_basis_rank ||
        matched_horizon <= 0 ||
        matched_horizon > basis_limit - current_basis_rank ||
        origin_active_width <= 0 ||
        target_active_width <= 0) {
        reset_matched_horizon_plan(plan);
        return false;
    }
    plan = {};
    plan.end_basis_rank = current_basis_rank + matched_horizon;
    plan.origin_active_width = origin_active_width;
    plan.active_left_width = target_active_width;
    return true;
}

inline void reset_matched_horizon_costs(
    MatchedHorizonPlan& plan,
    int active_left_width) noexcept {
    plan.active_left_width = active_left_width;
    plan.level_operator_seconds[0] =
        std::numeric_limits<double>::infinity();
    plan.level_operator_seconds[1] =
        std::numeric_limits<double>::infinity();
    plan.level_local_seconds[0] =
        std::numeric_limits<double>::infinity();
    plan.level_local_seconds[1] =
        std::numeric_limits<double>::infinity();
    plan.cost_sample_count = 0;
}

inline bool record_matched_horizon_cost(
    MatchedHorizonPlan& plan,
    int interval_start_width,
    int interval_end_width,
    int interval_levels,
    double interval_operator_seconds,
    double interval_local_seconds) noexcept {
    if (plan.end_basis_rank < 0 || interval_start_width <= 0 ||
        interval_start_width != interval_end_width ||
        interval_start_width != plan.active_left_width ||
        interval_levels <= 0 || interval_operator_seconds < 0.0 ||
        interval_local_seconds < 0.0 ||
        !std::isfinite(interval_operator_seconds) ||
        !std::isfinite(interval_local_seconds) ||
        !(interval_operator_seconds + interval_local_seconds > 0.0) ||
        plan.cost_sample_count >= 2) {
        return false;
    }
    const int index = plan.cost_sample_count;
    plan.level_operator_seconds[index] =
        interval_operator_seconds / static_cast<double>(interval_levels);
    plan.level_local_seconds[index] =
        interval_local_seconds / static_cast<double>(interval_levels);
    ++plan.cost_sample_count;
    return true;
}

}  // namespace amfls::detail
