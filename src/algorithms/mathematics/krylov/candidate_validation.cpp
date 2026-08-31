#include "algorithms/mathematics/krylov/candidate_validation.hpp"

#include <algorithm>
#include <chrono>
#include <cfenv>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

#include "algorithms/mathematics/floating_point_bounds.hpp"
#include "algorithms/mathematics/linalg/blas_lapack.hpp"

namespace amfls::math {
namespace {

using Clock = std::chrono::steady_clock;

enum class ValidationPhase { base, ridge_correction };

struct RidgeCorrectionAudit {
    RidgeCorrectionDisposition disposition =
        RidgeCorrectionDisposition::attempted_nonfinite;
    double corrected_bound = std::numeric_limits<double>::quiet_NaN();
    double gamma = std::numeric_limits<double>::quiet_NaN();
    double z_h_z = std::numeric_limits<double>::quiet_NaN();
    double two_abs_z_t_q = std::numeric_limits<double>::quiet_NaN();
    double q_norm_squared = std::numeric_limits<double>::quiet_NaN();
};

struct BaseValidationEnvelope {
    bool certified = false;
    double operator_norm_upper_bound =
        std::numeric_limits<double>::infinity();
    double operator_norm_lower_bound = 0.0;
    double residual_lower = 0.0;
    double residual_upper = std::numeric_limits<double>::infinity();
    double gradient_upper = std::numeric_limits<double>::infinity();
    double solution_lower = 0.0;
    double solution_upper = std::numeric_limits<double>::infinity();
    double right_hand_side_lower = 0.0;
    double applied_solution_lower = 0.0;
    double applied_solution_upper = std::numeric_limits<double>::infinity();
    double solution_energy_lower = 0.0;
    double gradient_error_radius = std::numeric_limits<double>::infinity();
    MatrixOperatorValidationErrorModel model;
};

double elapsed(Clock::time_point start) {
    return std::chrono::duration<double>(Clock::now() - start).count();
}

bool all_finite(std::span<const double> values) {
    return std::all_of(
        values.begin(), values.end(), [](double value) {
            return std::isfinite(value);
        });
}

bool all_finite(std::span<const long double> values) {
    return std::all_of(
        values.begin(), values.end(), [](long double value) {
            return std::isfinite(value);
        });
}

bool validation_model_is_available(
    const MatrixOperatorValidationErrorModel& model) noexcept {
    // The gamma bounds below are round-to-nearest bounds. Gradual underflow is
    // part of the public callback model; platforms configured to flush
    // subnormals must not claim this certificate.
    return std::numeric_limits<double>::is_iec559 &&
        fp::gradual_underflow_is_active() &&
        std::fegetround() == FE_TONEAREST &&
        std::isfinite(model.operator_norm_upper_bound) &&
        model.operator_norm_upper_bound >= 0.0 &&
        std::isfinite(model.operator_norm_lower_bound) &&
        model.operator_norm_lower_bound >= 0.0 &&
        model.operator_norm_lower_bound <=
            model.operator_norm_upper_bound &&
        model.apply_rounding_steps > 0 &&
        model.apply_transpose_rounding_steps > 0;
}

double underflow_radius(long long operations, int output_size) noexcept {
    if (operations <= 0 || output_size <= 0) {
        return 0.0;
    }
    const long double radius =
        static_cast<long double>(operations) *
        std::sqrt(static_cast<long double>(output_size)) *
        static_cast<long double>(std::numeric_limits<double>::denorm_min());
    return fp::upward_from_long_double(radius);
}

double callback_error_radius(
    const MatrixOperatorValidationErrorModel& model,
    bool transpose,
    double input_norm_upper,
    int output_size) noexcept {
    if (!validation_model_is_available(model) ||
        !std::isfinite(input_norm_upper) || input_norm_upper < 0.0) {
        return std::numeric_limits<double>::infinity();
    }
    if (model.operator_norm_upper_bound == 0.0 ||
        input_norm_upper == 0.0) {
        return 0.0;
    }
    const long long operations = transpose
        ? model.apply_transpose_rounding_steps
        : model.apply_rounding_steps;
    const double gamma = fp::binary64_gamma(operations);
    const double main_term = fp::upward_multiply(
        gamma,
        fp::upward_multiply(
            model.operator_norm_upper_bound, input_norm_upper));
    return fp::upward_add(
        main_term, underflow_radius(operations, output_size));
}

double one_operation_radius(
    double stored_result_norm_upper,
    bool can_underflow,
    int output_size) noexcept {
    if (!std::isfinite(stored_result_norm_upper) ||
        stored_result_norm_upper < 0.0) {
        return std::numeric_limits<double>::infinity();
    }
    double radius = fp::upward_multiply(
        fp::binary64_gamma(2), stored_result_norm_upper);
    if (can_underflow) {
        radius = fp::upward_add(
            radius, underflow_radius(1, output_size));
    }
    return radius;
}

fp::NonnegativeInterval enlarge_norm_interval(
    fp::NonnegativeInterval stored,
    double radius) noexcept {
    if (!std::isfinite(radius) || radius < 0.0) {
        return {};
    }
    return {
        fp::downward_difference(stored.lower, radius),
        fp::upward_add(stored.upper, radius)};
}

fp::NonnegativeInterval augmented_norm_interval(
    fp::NonnegativeInterval primary,
    fp::NonnegativeInterval secondary,
    double regularization) noexcept {
    const double lower_squared = fp::downward_add(
        fp::downward_multiply(primary.lower, primary.lower),
        fp::downward_multiply(
            regularization,
            fp::downward_multiply(secondary.lower, secondary.lower)));
    const double upper_squared = fp::upward_add(
        fp::upward_multiply(primary.upper, primary.upper),
        fp::upward_multiply(
            regularization,
            fp::upward_multiply(secondary.upper, secondary.upper)));
    return {
        fp::downward_sqrt(lower_squared),
        fp::upward_sqrt(upper_squared)};
}

double certified_ratio(double numerator, double denominator) {
    if (!std::isfinite(numerator)) {
        return std::numeric_limits<double>::infinity();
    }
    if (numerator == 0.0) {
        return 0.0;
    }
    if (!std::isfinite(denominator) || denominator <= 0.0) {
        return std::numeric_limits<double>::infinity();
    }
    return numerator / denominator;
}

double relative_energy_bound(double absolute_bound, double iterate_energy) {
    if (!std::isfinite(iterate_energy)) {
        return std::numeric_limits<double>::infinity();
    }
    if (absolute_bound == 0.0) {
        return 0.0;
    }
    if (!std::isfinite(absolute_bound) ||
        iterate_energy <= 0.0) {
        return std::numeric_limits<double>::infinity();
    }
    const double scaled_bound = absolute_bound / iterate_energy;
    if (!std::isfinite(scaled_bound) || scaled_bound >= 1.0) {
        return std::numeric_limits<double>::infinity();
    }
    return scaled_bound / (1.0 - scaled_bound);
}

void apply_validation(
    const MatrixOperator& matrix,
    const double* input,
    double* output,
    RunStatistics& statistics,
    ValidationPhase phase) {
    const auto start = Clock::now();
    matrix.apply(input, 1, output);
    const double seconds = elapsed(start);
    statistics.a_seconds += seconds;
    ++statistics.a_columns;
    ++statistics.validation_a_columns;
    ++statistics.a_block_calls;
    ++statistics.validation_a_block_calls;
    if (phase == ValidationPhase::base) {
        ++statistics.base_validation_a_columns;
        ++statistics.base_validation_a_block_calls;
        statistics.base_validation_seconds += seconds;
    } else {
        ++statistics.ridge_correction_a_columns;
        ++statistics.ridge_correction_a_block_calls;
        statistics.ridge_correction_validation_seconds += seconds;
    }
    statistics.validation_seconds = statistics.base_validation_seconds +
        statistics.ridge_correction_validation_seconds;
}

void apply_transpose_validation(
    const MatrixOperator& matrix,
    const double* input,
    double* output,
    RunStatistics& statistics,
    ValidationPhase phase) {
    const auto start = Clock::now();
    matrix.apply_transpose(input, 1, output);
    const double seconds = elapsed(start);
    statistics.at_seconds += seconds;
    ++statistics.at_columns;
    ++statistics.validation_at_columns;
    ++statistics.at_block_calls;
    ++statistics.validation_at_block_calls;
    if (phase == ValidationPhase::base) {
        ++statistics.base_validation_at_columns;
        ++statistics.base_validation_at_block_calls;
        statistics.base_validation_seconds += seconds;
    } else {
        ++statistics.ridge_correction_at_columns;
        ++statistics.ridge_correction_at_block_calls;
        statistics.ridge_correction_validation_seconds += seconds;
    }
    statistics.validation_seconds = statistics.base_validation_seconds +
        statistics.ridge_correction_validation_seconds;
}

bool long_double_validation_environment_is_available() noexcept {
    if (std::fegetround() != FE_TONEAREST ||
        !std::numeric_limits<long double>::is_iec559 ||
        std::numeric_limits<long double>::radix != 2 ||
        std::numeric_limits<long double>::digits <=
            std::numeric_limits<double>::digits ||
        std::numeric_limits<long double>::max_exponent <
            std::numeric_limits<double>::max_exponent ||
        std::numeric_limits<long double>::min_exponent >
            std::numeric_limits<double>::min_exponent ||
        std::numeric_limits<long double>::has_denorm != std::denorm_present) {
        return false;
    }
    volatile long double smallest_subnormal =
        std::numeric_limits<long double>::denorm_min();
    volatile long double one = 1.0L;
    volatile long double preserved_subnormal = smallest_subnormal * one;
    volatile long double smallest_normal =
        std::numeric_limits<long double>::min();
    volatile long double half = 0.5L;
    volatile long double produced_subnormal = smallest_normal * half;
    return preserved_subnormal ==
            std::numeric_limits<long double>::denorm_min() &&
        produced_subnormal > 0.0L;
}

double long_double_gamma(long long operations) noexcept {
    if (operations < 0) {
        return std::numeric_limits<double>::infinity();
    }
    const long double unit_roundoff =
        0.5L * std::numeric_limits<long double>::epsilon();
    const long double product =
        static_cast<long double>(operations) * unit_roundoff;
    if (!(product < 1.0L)) {
        return std::numeric_limits<double>::infinity();
    }
    return fp::upward_from_long_double(product / (1.0L - product));
}

double long_double_underflow_radius(
    long long operations,
    int output_size) noexcept {
    if (operations <= 0 || output_size <= 0) {
        return 0.0;
    }
    const long double radius =
        static_cast<long double>(operations) *
        std::sqrt(static_cast<long double>(output_size)) *
        std::numeric_limits<long double>::denorm_min();
    return fp::upward_from_long_double(radius);
}

double long_double_callback_error_radius(
    const MatrixOperatorValidationErrorModel& model,
    long long dot_length,
    double input_norm_upper,
    int output_size) noexcept {
    if (!validation_model_is_available(model) || dot_length <= 0 ||
        !std::isfinite(input_norm_upper) || input_norm_upper < 0.0) {
        return std::numeric_limits<double>::infinity();
    }
    if (model.operator_norm_upper_bound == 0.0 ||
        input_norm_upper == 0.0) {
        return 0.0;
    }
    if (dot_length > std::numeric_limits<long long>::max() / 2) {
        return std::numeric_limits<double>::infinity();
    }
    // The extended callback forms each dot-product term with one rounded
    // multiplication and one rounded addition.  gamma_(2 k) therefore also
    // covers implementations that contract some terms into an FMA.
    const long long operations = 2 * dot_length;
    const double main_term = fp::upward_multiply(
        long_double_gamma(operations),
        fp::upward_multiply(
            model.operator_norm_upper_bound, input_norm_upper));
    return fp::upward_add(
        main_term,
        long_double_underflow_radius(operations, output_size));
}

double long_double_one_operation_radius(
    double stored_result_norm_upper,
    bool can_underflow,
    int output_size) noexcept {
    if (!std::isfinite(stored_result_norm_upper) ||
        stored_result_norm_upper < 0.0) {
        return std::numeric_limits<double>::infinity();
    }
    double radius = fp::upward_multiply(
        long_double_gamma(2), stored_result_norm_upper);
    if (can_underflow) {
        radius = fp::upward_add(
            radius, long_double_underflow_radius(1, output_size));
    }
    return radius;
}

bool apply_validation_refinement_counted(
    const MatrixOperator& matrix,
    const long double* input,
    long double* output,
    RunStatistics& statistics) {
    const auto start = Clock::now();
    const bool finite = matrix.apply_validation_refinement(input, output);
    const double seconds = elapsed(start);
    statistics.a_seconds += seconds;
    ++statistics.a_columns;
    ++statistics.validation_a_columns;
    ++statistics.a_block_calls;
    ++statistics.validation_a_block_calls;
    ++statistics.base_validation_a_columns;
    ++statistics.base_validation_a_block_calls;
    statistics.base_validation_seconds += seconds;
    statistics.validation_seconds = statistics.base_validation_seconds +
        statistics.ridge_correction_validation_seconds;
    return finite;
}

bool apply_transpose_validation_refinement_counted(
    const MatrixOperator& matrix,
    const long double* input,
    long double* output,
    RunStatistics& statistics) {
    const auto start = Clock::now();
    const bool finite =
        matrix.apply_transpose_validation_refinement(input, output);
    const double seconds = elapsed(start);
    statistics.at_seconds += seconds;
    ++statistics.at_columns;
    ++statistics.validation_at_columns;
    ++statistics.at_block_calls;
    ++statistics.validation_at_block_calls;
    ++statistics.base_validation_at_columns;
    ++statistics.base_validation_at_block_calls;
    statistics.base_validation_seconds += seconds;
    statistics.validation_seconds = statistics.base_validation_seconds +
        statistics.ridge_correction_validation_seconds;
    return finite;
}

RidgeCorrectionAudit one_step_ridge_energy_bound(
    const MatrixOperator& matrix,
    std::span<const double> gradient,
    double gradient_norm,
    double regularization,
    RunStatistics& statistics,
    const BaseValidationEnvelope& base_envelope) {
    RidgeCorrectionAudit audit;
    if (!(gradient_norm > 0.0) || !std::isfinite(gradient_norm) ||
        !all_finite(gradient)) {
        audit.disposition =
            RidgeCorrectionDisposition::skipped_nonfinite_candidate;
        return audit;
    }

    std::vector<double> direction(matrix.cols());
    for (int col = 0; col < matrix.cols(); ++col) {
        direction[col] = gradient[col] / gradient_norm;
    }
    if (!all_finite(direction)) {
        audit.disposition =
            RidgeCorrectionDisposition::skipped_nonfinite_candidate;
        return audit;
    }
    const fp::NonnegativeInterval direction_norm =
        fp::norm_interval(direction);

    std::vector<double> applied_direction(matrix.rows());
    apply_validation(
        matrix,
        direction.data(),
        applied_direction.data(),
        statistics,
        ValidationPhase::ridge_correction);
    if (!all_finite(applied_direction)) {
        // A completed, but its unusable result prevents the A* callback.
        audit.disposition =
            RidgeCorrectionDisposition::attempted_partial_failure;
        return audit;
    }
    const fp::NonnegativeInterval applied_direction_norm =
        fp::norm_interval(applied_direction);
    const double applied_direction_error = callback_error_radius(
        base_envelope.model,
        false,
        direction_norm.upper,
        matrix.rows());

    std::vector<double> transposed_direction(matrix.cols());
    apply_transpose_validation(
        matrix,
        applied_direction.data(),
        transposed_direction.data(),
        statistics,
        ValidationPhase::ridge_correction);
    const double transpose_direction_error = callback_error_radius(
        base_envelope.model,
        true,
        applied_direction_norm.upper,
        matrix.cols());
    std::vector<double> hessian_direction = transposed_direction;
    for (int col = 0; col < matrix.cols(); ++col) {
        hessian_direction[col] = std::fma(
            regularization, direction[col], hessian_direction[col]);
    }
    if (!all_finite(hessian_direction)) {
        audit.disposition =
            RidgeCorrectionDisposition::attempted_nonfinite;
        return audit;
    }
    const fp::NonnegativeInterval hessian_direction_norm =
        fp::norm_interval(hessian_direction);
    const double propagated_applied_error = fp::upward_multiply(
        base_envelope.operator_norm_upper_bound,
        applied_direction_error);
    const double hessian_sum_error = one_operation_radius(
        hessian_direction_norm.upper,
        regularization > 0.0 && direction_norm.upper > 0.0,
        matrix.cols());
    const double hessian_error = fp::upward_add(
        fp::upward_add(
            transpose_direction_error, propagated_applied_error),
        hessian_sum_error);

    const double curvature = vector_dot(
        direction.data(), hessian_direction.data(), matrix.cols());
    if (!std::isfinite(curvature)) {
        audit.disposition =
            RidgeCorrectionDisposition::attempted_nonfinite;
        return audit;
    }
    if (!(curvature > 0.0)) {
        audit.disposition =
            RidgeCorrectionDisposition::attempted_invalid_curvature;
        return audit;
    }
    const double step = gradient_norm / curvature;
    if (!(step > 0.0) || !std::isfinite(step)) {
        audit.disposition =
            RidgeCorrectionDisposition::attempted_nonfinite;
        return audit;
    }

    // The correction is z=step*direction and Az=step*A*direction.  Their
    // norms and cross term can be scaled from the three vectors already in
    // hand; materializing two more full vectors would only repeat those
    // elementwise products at every unsuccessful ridge checkpoint.
    for (int col = 0; col < matrix.cols(); ++col) {
        hessian_direction[col] = std::fma(
            -step, hessian_direction[col], gradient[col]);
    }
    if (!all_finite(hessian_direction)) {
        audit.disposition =
            RidgeCorrectionDisposition::attempted_nonfinite;
        return audit;
    }
    const fp::NonnegativeInterval correction_residual_norm =
        fp::norm_interval(hessian_direction);
    const double correction_axpy_error = one_operation_radius(
        correction_residual_norm.upper,
        gradient_norm > 0.0 && hessian_direction_norm.upper > 0.0,
        matrix.cols());
    const double correction_residual_error = fp::upward_add(
        fp::upward_add(
            base_envelope.gradient_error_radius,
            fp::upward_multiply(step, hessian_error)),
        correction_axpy_error);
    const double q_upper = fp::upward_add(
        correction_residual_norm.upper, correction_residual_error);
    const double applied_direction_true_upper = fp::upward_add(
        applied_direction_norm.upper, applied_direction_error);

    const double step_squared = fp::upward_multiply(step, step);
    const double direction_energy_squared = fp::upward_add(
        fp::upward_multiply(
            applied_direction_true_upper,
            applied_direction_true_upper),
        fp::upward_multiply(
            regularization,
            fp::upward_multiply(
                direction_norm.upper, direction_norm.upper)));
    const double z_h_z = fp::upward_multiply(
        step_squared, direction_energy_squared);
    // Cauchy--Schwarz avoids treating the nominal dot product as exact.  It
    // is conservative but needs no additional callback or public audit data.
    const double cross = fp::upward_multiply(
        step,
        fp::upward_multiply(direction_norm.upper, q_upper));
    const double two_cross = fp::upward_multiply(2.0, cross);
    const double q_norm_squared = fp::upward_multiply(q_upper, q_upper);
    const double q_energy_term = fp::upward_divide(
        q_norm_squared, regularization);
    const double corrected_squared = fp::upward_add(
        z_h_z, fp::upward_add(two_cross, q_energy_term));
    const double corrected_bound = fp::upward_sqrt(corrected_squared);
    if (std::isnan(corrected_bound)) {
        audit.disposition =
            RidgeCorrectionDisposition::attempted_nonfinite;
        return audit;
    }
    audit.corrected_bound = corrected_bound;
    // Gamma is a nominal diagnostic.  Certification treats the stored
    // direction and positive stored step as an arbitrary z, for which the
    // complete correction inequality remains valid.
    audit.gamma = 1.0 / curvature;
    audit.z_h_z = z_h_z;
    audit.two_abs_z_t_q = two_cross;
    audit.q_norm_squared = q_norm_squared;
    // The caller compares d_corr with d0 without recomputing either bound.
    audit.disposition = RidgeCorrectionDisposition::attempted_not_improved;
    return audit;
}

}  // namespace

void validate_common_inputs(
    const MatrixOperator& matrix,
    const double* right_hand_side,
    double regularization,
    double tolerance) {
    if (matrix.rows() <= 0 || matrix.cols() <= 0) {
        throw std::invalid_argument("operator dimensions must be positive");
    }
    if (right_hand_side == nullptr) {
        throw std::invalid_argument("right-hand side cannot be null");
    }
    if (!std::isfinite(regularization) || regularization < 0.0) {
        throw std::invalid_argument("regularization must be finite and nonnegative");
    }
    if (!std::isfinite(tolerance) || tolerance <= 0.0 || tolerance >= 1.0) {
        throw std::invalid_argument("tolerance must lie strictly between zero and one");
    }
    for (int row = 0; row < matrix.rows(); ++row) {
        if (!std::isfinite(right_hand_side[row])) {
            throw std::invalid_argument("right-hand side must be finite");
        }
    }
}

void evaluate_accuracy_contract(
    LeastSquaresResult& result,
    double right_hand_side_norm,
    double regularization,
    double tolerance,
    std::optional<double> ridge_energy_error_upper_bound) {
    const double regularization_root = std::sqrt(regularization);
    result.augmented_residual_norm = std::hypot(
        result.residual_norm,
        regularization_root * result.solution_norm);
    result.augmented_operator_norm_lower_bound = std::hypot(
        result.operator_norm_lower_bound, regularization_root);

    const double compatible_denominator =
        result.augmented_operator_norm_lower_bound * result.solution_norm +
        right_hand_side_norm;
    result.compatible_backward_error_upper_bound = certified_ratio(
        result.augmented_residual_norm, compatible_denominator);

    const double least_squares_denominator =
        result.augmented_operator_norm_lower_bound *
        result.augmented_residual_norm;
    result.least_squares_backward_error_upper_bound = certified_ratio(
        result.normal_residual_norm, least_squares_denominator);
    result.backward_error_upper_bound = std::min(
        result.compatible_backward_error_upper_bound,
        result.least_squares_backward_error_upper_bound);

    const double normal_equation_denominator =
        (result.operator_norm_lower_bound * result.operator_norm_lower_bound +
         regularization) * result.solution_norm +
        result.operator_norm_lower_bound * right_hand_side_norm;
    result.relative_normal_residual_upper_bound = certified_ratio(
        result.normal_residual_norm, normal_equation_denominator);

    if (regularization > 0.0) {
        const double requested_bound = ridge_energy_error_upper_bound.value_or(
            result.normal_residual_norm / regularization_root);
        result.energy_error_upper_bound =
            requested_bound >= 0.0
            ? requested_bound
            : std::numeric_limits<double>::infinity();
        result.objective_gap_upper_bound =
            result.energy_error_upper_bound * result.energy_error_upper_bound;
        result.relative_energy_error_upper_bound = relative_energy_bound(
            result.energy_error_upper_bound, result.solution_energy_norm);
    } else if (result.normal_residual_norm == 0.0) {
        result.energy_error_upper_bound = 0.0;
        result.relative_energy_error_upper_bound = 0.0;
        result.objective_gap_upper_bound = 0.0;
    } else {
        result.energy_error_upper_bound =
            std::numeric_limits<double>::infinity();
        result.relative_energy_error_upper_bound =
            std::numeric_limits<double>::infinity();
        result.objective_gap_upper_bound =
            std::numeric_limits<double>::infinity();
    }

    const auto finite_norm = [](double value) {
        return std::isfinite(value) && value >= 0.0;
    };
    const bool finite_contract_primitives =
        finite_norm(right_hand_side_norm) &&
        finite_norm(result.residual_norm) &&
        finite_norm(result.augmented_residual_norm) &&
        finite_norm(result.normal_residual_norm) &&
        finite_norm(result.solution_norm) &&
        finite_norm(result.solution_energy_norm) &&
        finite_norm(result.objective) &&
        finite_norm(result.operator_norm_lower_bound) &&
        finite_norm(result.augmented_operator_norm_lower_bound);
    if (!finite_contract_primitives) {
        result.status = SolverStatus::numerical_breakdown;
        result.stop_reason = StopReason::numerical_breakdown;
        return;
    }

    const bool passed = regularization > 0.0
        ? result.relative_energy_error_upper_bound <= tolerance
        : result.backward_error_upper_bound <= tolerance;
    result.status = passed
        ? SolverStatus::success
        : SolverStatus::work_limit;
    if (result.normal_residual_norm == 0.0) {
        result.stop_reason = StopReason::exact_stationarity;
    } else if (result.status == SolverStatus::success &&
               regularization > 0.0) {
        result.stop_reason = StopReason::relative_energy_error;
    } else if (result.status == SolverStatus::success &&
               result.compatible_backward_error_upper_bound <=
                   result.least_squares_backward_error_upper_bound) {
        result.stop_reason = StopReason::compatible_backward_error;
    } else if (result.status == SolverStatus::success) {
        result.stop_reason = StopReason::least_squares_backward_error;
    } else {
        result.stop_reason = StopReason::maximum_depth;
    }
}

void evaluate_certified_accuracy_contract(
    LeastSquaresResult& result,
    const BaseValidationEnvelope& envelope,
    double regularization,
    double tolerance,
    std::optional<double> ridge_energy_error_upper_bound = std::nullopt) {
    if (!envelope.certified) {
        result.compatible_backward_error_upper_bound =
            std::numeric_limits<double>::infinity();
        result.least_squares_backward_error_upper_bound =
            std::numeric_limits<double>::infinity();
        result.backward_error_upper_bound =
            std::numeric_limits<double>::infinity();
        result.relative_normal_residual_upper_bound =
            std::numeric_limits<double>::infinity();
        result.energy_error_upper_bound =
            std::numeric_limits<double>::infinity();
        result.relative_energy_error_upper_bound =
            std::numeric_limits<double>::infinity();
        result.objective_gap_upper_bound =
            std::numeric_limits<double>::infinity();
        result.status = SolverStatus::work_limit;
        result.stop_reason = StopReason::maximum_depth;
        return;
    }

    const fp::NonnegativeInterval solution{
        envelope.solution_lower, envelope.solution_upper};
    const fp::NonnegativeInterval residual{
        envelope.residual_lower, envelope.residual_upper};
    const fp::NonnegativeInterval augmented_residual =
        augmented_norm_interval(residual, solution, regularization);

    const double augmented_operator_squared = fp::downward_add(
        fp::downward_multiply(
            result.operator_norm_lower_bound,
            result.operator_norm_lower_bound),
        regularization);
    result.augmented_operator_norm_lower_bound =
        fp::downward_sqrt(augmented_operator_squared);

    const double compatible_denominator = fp::downward_add(
        fp::downward_multiply(
            result.augmented_operator_norm_lower_bound,
            envelope.solution_lower),
        envelope.right_hand_side_lower);
    result.compatible_backward_error_upper_bound = fp::upward_divide(
        augmented_residual.upper, compatible_denominator);

    const double least_squares_denominator = fp::downward_multiply(
        result.augmented_operator_norm_lower_bound,
        augmented_residual.lower);
    result.least_squares_backward_error_upper_bound = fp::upward_divide(
        envelope.gradient_upper, least_squares_denominator);
    result.backward_error_upper_bound = std::min(
        result.compatible_backward_error_upper_bound,
        result.least_squares_backward_error_upper_bound);

    const double normal_denominator = fp::downward_add(
        fp::downward_multiply(
            fp::downward_add(
                fp::downward_multiply(
                    result.operator_norm_lower_bound,
                    result.operator_norm_lower_bound),
                regularization),
            envelope.solution_lower),
        fp::downward_multiply(
            result.operator_norm_lower_bound,
            envelope.right_hand_side_lower));
    result.relative_normal_residual_upper_bound = fp::upward_divide(
        envelope.gradient_upper, normal_denominator);

    if (regularization > 0.0) {
        const double regularization_root_lower =
            fp::downward_sqrt(regularization);
        const double base_bound = fp::upward_divide(
            envelope.gradient_upper, regularization_root_lower);
        result.energy_error_upper_bound =
            ridge_energy_error_upper_bound.value_or(base_bound);
        result.objective_gap_upper_bound = fp::upward_multiply(
            result.energy_error_upper_bound,
            result.energy_error_upper_bound);
        const double relative_denominator = fp::downward_difference(
            envelope.solution_energy_lower,
            result.energy_error_upper_bound);
        result.relative_energy_error_upper_bound = fp::upward_divide(
            result.energy_error_upper_bound, relative_denominator);
    } else if (envelope.gradient_upper == 0.0) {
        result.energy_error_upper_bound = 0.0;
        result.relative_energy_error_upper_bound = 0.0;
        result.objective_gap_upper_bound = 0.0;
    } else {
        result.energy_error_upper_bound =
            std::numeric_limits<double>::infinity();
        result.relative_energy_error_upper_bound =
            std::numeric_limits<double>::infinity();
        result.objective_gap_upper_bound =
            std::numeric_limits<double>::infinity();
    }

    const bool passed = envelope.gradient_upper == 0.0 ||
        (regularization > 0.0
             ? result.relative_energy_error_upper_bound <= tolerance
             : result.backward_error_upper_bound <= tolerance);
    result.status = passed
        ? SolverStatus::success
        : SolverStatus::work_limit;
    if (envelope.gradient_upper == 0.0) {
        result.stop_reason = StopReason::exact_stationarity;
    } else if (passed && regularization > 0.0) {
        result.stop_reason = StopReason::relative_energy_error;
    } else if (passed &&
               result.compatible_backward_error_upper_bound <=
                   result.least_squares_backward_error_upper_bound) {
        result.stop_reason = StopReason::compatible_backward_error;
    } else if (passed) {
        result.stop_reason = StopReason::least_squares_backward_error;
    } else {
        result.stop_reason = StopReason::maximum_depth;
    }
}

bool candidate_validation_failed_numerically(
    const LeastSquaresResult& result) noexcept {
    return result.status == SolverStatus::numerical_breakdown ||
        result.stop_reason == StopReason::numerical_breakdown;
}

BaseValidationEnvelope build_base_validation_envelope(
    const MatrixOperator& matrix,
    std::span<const double> right_hand_side,
    std::span<const double> solution,
    std::span<const double> applied_solution,
    std::span<const double> residual,
    std::span<const double> transposed_residual,
    std::span<const double> gradient,
    double regularization) {
    BaseValidationEnvelope envelope;
    envelope.model = matrix.validation_error_model();
    envelope.operator_norm_upper_bound =
        envelope.model.operator_norm_upper_bound;
    if (!validation_model_is_available(envelope.model)) {
        return envelope;
    }

    const fp::NonnegativeInterval b_norm =
        fp::norm_interval(right_hand_side);
    const fp::NonnegativeInterval x_norm = fp::norm_interval(solution);
    const fp::NonnegativeInterval applied_norm =
        fp::norm_interval(applied_solution);
    const fp::NonnegativeInterval stored_residual_norm =
        fp::norm_interval(residual);
    const fp::NonnegativeInterval transpose_norm =
        fp::norm_interval(transposed_residual);
    const fp::NonnegativeInterval stored_gradient_norm =
        fp::norm_interval(gradient);

    const double apply_error = callback_error_radius(
        envelope.model, false, x_norm.upper, matrix.rows());
    // Binary64 subtraction of two stored binary64 numbers obeys the usual
    // one-operation relative model; using the stored result makes the bound
    // tight under cancellation and preserves an exact zero difference.
    const double subtraction_error = one_operation_radius(
        stored_residual_norm.upper,
        applied_norm.upper > 0.0 || b_norm.upper > 0.0,
        matrix.rows());
    const double residual_error = fp::upward_add(
        apply_error, subtraction_error);
    const fp::NonnegativeInterval true_residual = enlarge_norm_interval(
        stored_residual_norm, residual_error);

    const double transpose_error = callback_error_radius(
        envelope.model,
        true,
        stored_residual_norm.upper,
        matrix.cols());
    const bool ridge_sum_can_underflow =
        regularization > 0.0 && x_norm.upper > 0.0;
    const double ridge_sum_error = one_operation_radius(
        stored_gradient_norm.upper,
        ridge_sum_can_underflow,
        matrix.cols());
    const double propagated_residual_error = fp::upward_multiply(
        envelope.operator_norm_upper_bound, residual_error);
    const double gradient_error = fp::upward_add(
        fp::upward_add(transpose_error, propagated_residual_error),
        ridge_sum_error);
    const fp::NonnegativeInterval true_gradient = enlarge_norm_interval(
        stored_gradient_norm, gradient_error);
    const fp::NonnegativeInterval true_applied = enlarge_norm_interval(
        applied_norm, apply_error);

    double operator_lower = envelope.model.operator_norm_lower_bound;
    if (x_norm.upper > 0.0) {
        operator_lower = std::max(
            operator_lower,
            fp::downward_divide(
                fp::downward_difference(applied_norm.lower, apply_error),
                x_norm.upper));
    }
    if (stored_residual_norm.upper > 0.0) {
        operator_lower = std::max(
            operator_lower,
            fp::downward_divide(
                fp::downward_difference(
                    transpose_norm.lower, transpose_error),
                stored_residual_norm.upper));
    }

    const fp::NonnegativeInterval solution_energy =
        augmented_norm_interval(true_applied, x_norm, regularization);
    envelope.residual_lower = true_residual.lower;
    envelope.residual_upper = true_residual.upper;
    envelope.gradient_upper = true_gradient.upper;
    envelope.solution_lower = x_norm.lower;
    envelope.solution_upper = x_norm.upper;
    envelope.right_hand_side_lower = b_norm.lower;
    envelope.applied_solution_lower = true_applied.lower;
    envelope.applied_solution_upper = true_applied.upper;
    envelope.solution_energy_lower = solution_energy.lower;
    envelope.gradient_error_radius = gradient_error;
    envelope.certified =
        std::isfinite(envelope.residual_upper) &&
        std::isfinite(envelope.gradient_upper) &&
        std::isfinite(envelope.solution_upper) &&
        std::isfinite(envelope.operator_norm_upper_bound);
    envelope.applied_solution_lower = std::max(
        0.0, envelope.applied_solution_lower);
    envelope.right_hand_side_lower = std::max(
        0.0, envelope.right_hand_side_lower);
    envelope.operator_norm_lower_bound = operator_lower;
    return envelope;
}

namespace {

struct RefinedBaseValidation {
    BaseValidationEnvelope envelope;
    std::vector<double> applied_solution;
    std::vector<double> residual;
    std::vector<double> transposed_residual;
    std::vector<double> gradient;
};

bool cast_validation_vector(
    std::span<const long double> source,
    std::vector<double>& destination) {
    destination.resize(source.size());
    for (std::size_t index = 0; index < source.size(); ++index) {
        destination[index] = static_cast<double>(source[index]);
        if (!std::isfinite(destination[index])) {
            return false;
        }
    }
    return true;
}

std::optional<RefinedBaseValidation> refine_base_validation(
    const MatrixOperator& matrix,
    std::span<const double> right_hand_side,
    std::span<const double> solution,
    double regularization,
    RunStatistics& statistics) {
    if (!matrix.supports_validation_refinement() ||
        !long_double_validation_environment_is_available()) {
        return std::nullopt;
    }
    const MatrixOperatorValidationErrorModel model =
        matrix.validation_error_model();
    if (!validation_model_is_available(model)) {
        return std::nullopt;
    }

    const fp::NonnegativeInterval b_norm =
        fp::norm_interval(right_hand_side);
    const fp::NonnegativeInterval x_norm = fp::norm_interval(solution);
    if (!std::isfinite(b_norm.upper) || !std::isfinite(x_norm.upper)) {
        return std::nullopt;
    }

    std::vector<long double> long_solution(solution.begin(), solution.end());
    std::vector<long double> long_applied(
        static_cast<std::size_t>(matrix.rows()));
    if (!apply_validation_refinement_counted(
            matrix,
            long_solution.data(),
            long_applied.data(),
            statistics) ||
        !all_finite(std::span<const long double>(long_applied))) {
        return std::nullopt;
    }

    RefinedBaseValidation refined;
    if (!cast_validation_vector(
            std::span<const long double>(long_applied),
            refined.applied_solution)) {
        return std::nullopt;
    }
    const fp::NonnegativeInterval stored_applied_norm =
        fp::norm_interval(refined.applied_solution);
    const double applied_cast_error = one_operation_radius(
        stored_applied_norm.upper, true, matrix.rows());
    const double applied_accumulation_error =
        long_double_callback_error_radius(
            model, matrix.cols(), x_norm.upper, matrix.rows());
    const double applied_error = fp::upward_add(
        applied_accumulation_error, applied_cast_error);

    std::vector<long double> long_residual(
        static_cast<std::size_t>(matrix.rows()));
    for (int row = 0; row < matrix.rows(); ++row) {
        long_residual[static_cast<std::size_t>(row)] =
            long_applied[static_cast<std::size_t>(row)] -
            static_cast<long double>(right_hand_side[row]);
    }
    if (!all_finite(std::span<const long double>(long_residual)) ||
        !cast_validation_vector(
            std::span<const long double>(long_residual),
            refined.residual)) {
        return std::nullopt;
    }
    const fp::NonnegativeInterval stored_residual_norm =
        fp::norm_interval(refined.residual);
    const double residual_cast_error = one_operation_radius(
        stored_residual_norm.upper, true, matrix.rows());
    const double long_residual_norm_upper = fp::upward_add(
        stored_residual_norm.upper, residual_cast_error);
    const double residual_subtraction_error =
        long_double_one_operation_radius(
            long_residual_norm_upper,
            stored_applied_norm.upper > 0.0 || b_norm.upper > 0.0,
            matrix.rows());
    // The long-double residual is consumed directly by A^T, so its error
    // before the binary64 storage conversion excludes residual_cast_error.
    const double long_residual_error = fp::upward_add(
        applied_accumulation_error, residual_subtraction_error);
    const double stored_residual_error = fp::upward_add(
        long_residual_error, residual_cast_error);

    std::vector<long double> long_transposed(
        static_cast<std::size_t>(matrix.cols()));
    if (!apply_transpose_validation_refinement_counted(
            matrix,
            long_residual.data(),
            long_transposed.data(),
            statistics) ||
        !all_finite(std::span<const long double>(long_transposed)) ||
        !cast_validation_vector(
            std::span<const long double>(long_transposed),
            refined.transposed_residual)) {
        return std::nullopt;
    }
    const fp::NonnegativeInterval stored_transpose_norm =
        fp::norm_interval(refined.transposed_residual);
    const double transpose_cast_error = one_operation_radius(
        stored_transpose_norm.upper, true, matrix.cols());
    const double transpose_accumulation_error =
        long_double_callback_error_radius(
            model,
            matrix.rows(),
            long_residual_norm_upper,
            matrix.cols());

    std::vector<long double> long_gradient(
        static_cast<std::size_t>(matrix.cols()));
    for (int column = 0; column < matrix.cols(); ++column) {
        long_gradient[static_cast<std::size_t>(column)] = std::fma(
            static_cast<long double>(regularization),
            long_solution[static_cast<std::size_t>(column)],
            long_transposed[static_cast<std::size_t>(column)]);
    }
    if (!all_finite(std::span<const long double>(long_gradient)) ||
        !cast_validation_vector(
            std::span<const long double>(long_gradient),
            refined.gradient)) {
        return std::nullopt;
    }
    const fp::NonnegativeInterval stored_gradient_norm =
        fp::norm_interval(refined.gradient);
    const double gradient_cast_error = one_operation_radius(
        stored_gradient_norm.upper, true, matrix.cols());
    const double long_gradient_norm_upper = fp::upward_add(
        stored_gradient_norm.upper, gradient_cast_error);
    const double ridge_sum_error = long_double_one_operation_radius(
        long_gradient_norm_upper,
        regularization > 0.0 && x_norm.upper > 0.0,
        matrix.cols());
    const double propagated_long_residual_error = fp::upward_multiply(
        model.operator_norm_upper_bound, long_residual_error);
    const double gradient_error = fp::upward_add(
        fp::upward_add(
            transpose_accumulation_error,
            propagated_long_residual_error),
        fp::upward_add(ridge_sum_error, gradient_cast_error));

    const fp::NonnegativeInterval true_applied = enlarge_norm_interval(
        stored_applied_norm, applied_error);
    const fp::NonnegativeInterval true_residual = enlarge_norm_interval(
        stored_residual_norm, stored_residual_error);
    const fp::NonnegativeInterval true_gradient = enlarge_norm_interval(
        stored_gradient_norm, gradient_error);

    double operator_lower = model.operator_norm_lower_bound;
    if (x_norm.upper > 0.0) {
        operator_lower = std::max(
            operator_lower,
            fp::downward_divide(
                fp::downward_difference(
                    stored_applied_norm.lower, applied_error),
                x_norm.upper));
    }
    if (stored_residual_norm.upper > 0.0) {
        const double transpose_input_conversion_error =
            fp::upward_multiply(
                model.operator_norm_upper_bound, residual_cast_error);
        const double transpose_error_for_stored_residual = fp::upward_add(
            fp::upward_add(
                transpose_accumulation_error,
                transpose_cast_error),
            transpose_input_conversion_error);
        operator_lower = std::max(
            operator_lower,
            fp::downward_divide(
                fp::downward_difference(
                    stored_transpose_norm.lower,
                    transpose_error_for_stored_residual),
                stored_residual_norm.upper));
    }

    const fp::NonnegativeInterval solution_energy =
        augmented_norm_interval(true_applied, x_norm, regularization);
    BaseValidationEnvelope& envelope = refined.envelope;
    envelope.model = model;
    envelope.operator_norm_upper_bound = model.operator_norm_upper_bound;
    envelope.operator_norm_lower_bound = operator_lower;
    envelope.residual_lower = true_residual.lower;
    envelope.residual_upper = true_residual.upper;
    envelope.gradient_upper = true_gradient.upper;
    envelope.solution_lower = x_norm.lower;
    envelope.solution_upper = x_norm.upper;
    envelope.right_hand_side_lower = std::max(0.0, b_norm.lower);
    envelope.applied_solution_lower = std::max(
        0.0, true_applied.lower);
    envelope.applied_solution_upper = true_applied.upper;
    envelope.solution_energy_lower = solution_energy.lower;
    envelope.gradient_error_radius = gradient_error;
    envelope.certified =
        std::isfinite(envelope.residual_upper) &&
        std::isfinite(envelope.gradient_upper) &&
        std::isfinite(envelope.solution_upper) &&
        std::isfinite(envelope.operator_norm_upper_bound);
    if (!envelope.certified) {
        return std::nullopt;
    }
    return refined;
}

void populate_candidate_norms(
    LeastSquaresResult& result,
    std::span<const double> solution,
    std::span<const double> applied_solution,
    std::span<const double> residual,
    std::span<const double> gradient,
    double regularization) {
    const double applied_solution_norm = vector_norm(
        applied_solution.data(), static_cast<int>(applied_solution.size()));
    result.residual_norm = vector_norm(
        residual.data(), static_cast<int>(residual.size()));
    result.normal_residual_norm = vector_norm(
        gradient.data(), static_cast<int>(gradient.size()));
    result.solution_norm = vector_norm(
        solution.data(), static_cast<int>(solution.size()));
    result.solution_energy_norm = std::hypot(
        applied_solution_norm,
        std::sqrt(regularization) * result.solution_norm);
    result.objective =
        result.residual_norm * result.residual_norm +
        regularization * result.solution_norm * result.solution_norm;
}

}  // namespace

LeastSquaresResult validate_original_candidate(
    const MatrixOperator& matrix,
    const double* right_hand_side,
    std::span<const double> solution,
    double regularization,
    double tolerance,
    double operator_norm_lower_bound,
    RunStatistics& statistics) {
    if (solution.size() != static_cast<std::size_t>(matrix.cols())) {
        throw std::invalid_argument("candidate solution has the wrong size");
    }

    LeastSquaresResult result;
    result.rows = matrix.rows();
    result.cols = matrix.cols();
    result.solution.assign(solution.begin(), solution.end());
    // Search-side values currently have no callback-error audit.  Retain the
    // argument for source compatibility, but stopping uses only a fresh lower
    // bound reconstructed below from the validation pair.
    (void)operator_norm_lower_bound;
    result.operator_norm_lower_bound = 0.0;

    std::vector<double> applied_solution(matrix.rows());
    apply_validation(
        matrix,
        solution.data(),
        applied_solution.data(),
        statistics,
        ValidationPhase::base);

    std::vector<double> residual(matrix.rows());
    for (int row = 0; row < matrix.rows(); ++row) {
        residual[row] = applied_solution[row] - right_hand_side[row];
    }
    std::vector<double> transposed_residual(matrix.cols());
    apply_transpose_validation(
        matrix,
        residual.data(),
        transposed_residual.data(),
        statistics,
        ValidationPhase::base);
    std::vector<double> gradient = transposed_residual;
    for (int col = 0; col < matrix.cols(); ++col) {
        gradient[col] = std::fma(
            regularization, solution[col], gradient[col]);
    }

    populate_candidate_norms(
        result,
        solution,
        applied_solution,
        residual,
        gradient,
        regularization);
    const double right_hand_side_norm = vector_norm(
        right_hand_side, matrix.rows());
    evaluate_accuracy_contract(
        result,
        right_hand_side_norm,
        regularization,
        tolerance);

    const bool finite_candidate_vectors =
        all_finite(solution) && all_finite(applied_solution) &&
        all_finite(residual) && all_finite(transposed_residual) &&
        all_finite(gradient);
    if (!finite_candidate_vectors ||
        candidate_validation_failed_numerically(result)) {
        result.status = SolverStatus::numerical_breakdown;
        result.stop_reason = StopReason::numerical_breakdown;
    } else {
        BaseValidationEnvelope envelope =
            build_base_validation_envelope(
                matrix,
                std::span<const double>(
                    right_hand_side,
                    static_cast<std::size_t>(matrix.rows())),
                solution,
                applied_solution,
                residual,
                transposed_residual,
                gradient,
                regularization);
        result.operator_norm_lower_bound =
            envelope.operator_norm_lower_bound;
        evaluate_certified_accuracy_contract(
            result, envelope, regularization, tolerance);

        // The fast BLAS/CSR envelope remains the default.  A built-in stored
        // operator repeats the complete base chain with an extended
        // accumulator only after a failed check whose gradient
        // uncertainty is at least as large as the stored gradient itself.
        // This is a validation refinement, not a different solver path, and
        // its two matrix passes are fully charged below.
        if (result.status != SolverStatus::success &&
            envelope.certified &&
            envelope.gradient_error_radius >=
                result.normal_residual_norm) {
            std::optional<RefinedBaseValidation> refined =
                refine_base_validation(
                    matrix,
                    std::span<const double>(
                        right_hand_side,
                        static_cast<std::size_t>(matrix.rows())),
                    solution,
                    regularization,
                    statistics);
            if (refined.has_value()) {
                applied_solution = std::move(
                    refined->applied_solution);
                residual = std::move(refined->residual);
                transposed_residual = std::move(
                    refined->transposed_residual);
                gradient = std::move(refined->gradient);
                envelope = refined->envelope;
                populate_candidate_norms(
                    result,
                    solution,
                    applied_solution,
                    residual,
                    gradient,
                    regularization);
                evaluate_accuracy_contract(
                    result,
                    right_hand_side_norm,
                    regularization,
                    tolerance);
                result.operator_norm_lower_bound =
                    envelope.operator_norm_lower_bound;
                evaluate_certified_accuracy_contract(
                    result, envelope, regularization, tolerance);
            }
        }

        if (regularization > 0.0) {
            result.ridge_base_energy_error_upper_bound =
                result.energy_error_upper_bound;
            if (result.status == SolverStatus::success) {
                result.ridge_correction_disposition =
                    RidgeCorrectionDisposition::skipped_base_pass;
            } else if (!envelope.certified) {
                result.ridge_correction_disposition =
                    RidgeCorrectionDisposition::skipped_nonfinite_candidate;
            } else if (envelope.solution_energy_lower == 0.0) {
                result.ridge_correction_disposition =
                    RidgeCorrectionDisposition::skipped_zero_energy;
            } else if (!(envelope.gradient_upper > 0.0) ||
                       !std::isfinite(envelope.gradient_upper)) {
                result.ridge_correction_disposition =
                    RidgeCorrectionDisposition::skipped_nonfinite_candidate;
            } else {
                RidgeCorrectionAudit audit = one_step_ridge_energy_bound(
                    matrix,
                    gradient,
                    result.normal_residual_norm,
                    regularization,
                    statistics,
                    envelope);
                result.ridge_correction_disposition = audit.disposition;
                result.ridge_corrected_energy_error_upper_bound =
                    audit.corrected_bound;
                result.ridge_correction_gamma = audit.gamma;
                result.ridge_correction_z_h_z = audit.z_h_z;
                result.ridge_correction_two_abs_z_t_q =
                    audit.two_abs_z_t_q;
                result.ridge_correction_q_norm_squared =
                    audit.q_norm_squared;
                if (audit.disposition ==
                        RidgeCorrectionDisposition::attempted_not_improved &&
                    std::isfinite(audit.corrected_bound) &&
                    audit.corrected_bound < result.energy_error_upper_bound) {
                    result.ridge_correction_disposition =
                        RidgeCorrectionDisposition::attempted_improved;
                    evaluate_certified_accuracy_contract(
                        result,
                        envelope,
                        regularization,
                        tolerance,
                        audit.corrected_bound);
                }
            }
        }
    }

    if (regularization > 0.0 &&
        result.status == SolverStatus::numerical_breakdown) {
        result.ridge_base_energy_error_upper_bound =
            std::isnan(result.normal_residual_norm)
            ? std::numeric_limits<double>::quiet_NaN()
            : std::numeric_limits<double>::infinity();
        result.energy_error_upper_bound =
            result.ridge_base_energy_error_upper_bound;
        result.relative_energy_error_upper_bound =
            std::numeric_limits<double>::infinity();
        result.objective_gap_upper_bound =
            std::numeric_limits<double>::infinity();
        result.ridge_correction_disposition =
            RidgeCorrectionDisposition::skipped_nonfinite_candidate;
    }

    result.statistics = statistics;
    return result;
}

}  // namespace amfls::math
