#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>

#include "amfls/least_squares_result.hpp"

namespace amfls::torch_binding::result_tensors {

inline constexpr std::size_t metrics_width = 21;
inline constexpr std::size_t counters_width = 36;
inline constexpr std::size_t timings_width = 9;
inline constexpr std::size_t trace_float_width = 29;
inline constexpr std::size_t trace_int_width = 4;

using Metrics = std::array<double, metrics_width>;
using Counters = std::array<std::int64_t, counters_width>;
using Timings = std::array<double, timings_width>;
using TraceFloatRow = std::array<double, trace_float_width>;
using TraceIntRow = std::array<std::int64_t, trace_int_width>;

inline Metrics metrics(const LeastSquaresResult& result) {
    return {
        result.objective,
        result.residual_norm,
        result.augmented_residual_norm,
        result.normal_residual_norm,
        result.solution_norm,
        result.solution_energy_norm,
        result.operator_norm_lower_bound,
        result.augmented_operator_norm_lower_bound,
        result.compatible_backward_error_upper_bound,
        result.least_squares_backward_error_upper_bound,
        result.backward_error_upper_bound,
        result.relative_normal_residual_upper_bound,
        result.energy_error_upper_bound,
        result.relative_energy_error_upper_bound,
        result.objective_gap_upper_bound,
        result.ridge_base_energy_error_upper_bound,
        result.ridge_corrected_energy_error_upper_bound,
        result.ridge_correction_gamma,
        result.ridge_correction_z_h_z,
        result.ridge_correction_two_abs_z_t_q,
        result.ridge_correction_q_norm_squared,
    };
}

inline Counters counters(const LeastSquaresResult& result) {
    const RunStatistics& statistics = result.statistics;
    const auto i64 = [](auto value) {
        return static_cast<std::int64_t>(value);
    };
    return {
        i64(statistics.a_columns),
        i64(statistics.at_columns),
        i64(statistics.a_block_calls),
        i64(statistics.at_block_calls),
        i64(statistics.search_a_columns),
        i64(statistics.search_at_columns),
        i64(statistics.validation_a_columns),
        i64(statistics.validation_at_columns),
        i64(result.iterations),
        i64(result.depth),
        i64(result.auxiliary_width),
        i64(result.basis_rank),
        i64(statistics.sketch_a_columns),
        i64(statistics.sketch_at_columns),
        i64(statistics.sketch_a_block_calls),
        i64(statistics.sketch_at_block_calls),
        i64(statistics.iterative_a_columns),
        i64(statistics.iterative_at_columns),
        i64(statistics.iterative_a_block_calls),
        i64(statistics.iterative_at_block_calls),
        i64(statistics.gaussian_random_columns),
        i64(statistics.gaussian_random_values),
        i64(statistics.search_a_block_calls),
        i64(statistics.search_at_block_calls),
        i64(statistics.validation_a_block_calls),
        i64(statistics.validation_at_block_calls),
        i64(statistics.base_validation_a_columns),
        i64(statistics.base_validation_at_columns),
        i64(statistics.base_validation_a_block_calls),
        i64(statistics.base_validation_at_block_calls),
        i64(statistics.ridge_correction_a_columns),
        i64(statistics.ridge_correction_at_columns),
        i64(statistics.ridge_correction_a_block_calls),
        i64(statistics.ridge_correction_at_block_calls),
        i64(statistics.gaussian_random_block_requests),
        i64(result.ridge_correction_disposition),
    };
}

inline Timings timings(const LeastSquaresResult& result) {
    const RunStatistics& statistics = result.statistics;
    const double unclassified_seconds = std::max(
        0.0,
        statistics.total_seconds - statistics.a_seconds -
            statistics.at_seconds - statistics.orthogonalization_seconds -
            statistics.projected_solve_seconds);
    return {
        statistics.total_seconds,
        statistics.a_seconds,
        statistics.at_seconds,
        statistics.orthogonalization_seconds,
        statistics.projected_solve_seconds,
        statistics.validation_seconds,
        unclassified_seconds,
        statistics.base_validation_seconds,
        statistics.ridge_correction_validation_seconds,
    };
}

inline TraceFloatRow trace_float(const IterationRecord& record) {
    return {
        static_cast<double>(record.epoch),
        static_cast<double>(record.depth),
        static_cast<double>(record.auxiliary_width),
        static_cast<double>(record.basis_rank),
        static_cast<double>(record.a_columns),
        static_cast<double>(record.at_columns),
        static_cast<double>(record.a_block_calls),
        static_cast<double>(record.at_block_calls),
        record.objective,
        record.residual_norm,
        record.augmented_residual_norm,
        record.normal_residual_norm,
        record.operator_norm_lower_bound,
        record.compatible_backward_error_upper_bound,
        record.least_squares_backward_error_upper_bound,
        record.backward_error_upper_bound,
        record.relative_normal_residual_upper_bound,
        record.relative_energy_error_upper_bound,
        record.solution_norm,
        record.solution_energy_norm,
        record.augmented_operator_norm_lower_bound,
        record.energy_error_upper_bound,
        record.objective_gap_upper_bound,
        record.ridge_base_energy_error_upper_bound,
        record.ridge_corrected_energy_error_upper_bound,
        record.ridge_correction_gamma,
        record.ridge_correction_z_h_z,
        record.ridge_correction_two_abs_z_t_q,
        record.ridge_correction_q_norm_squared,
    };
}

inline TraceIntRow trace_int(const IterationRecord& record) {
    return {
        static_cast<std::int64_t>(record.contract_status),
        static_cast<std::int64_t>(record.contract_stop_reason),
        static_cast<std::int64_t>(record.contract_passed),
        static_cast<std::int64_t>(record.ridge_correction_disposition),
    };
}

}  // namespace amfls::torch_binding::result_tensors
