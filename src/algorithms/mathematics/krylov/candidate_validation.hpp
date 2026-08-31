#pragma once

#include <optional>
#include <span>

#include "amfls/least_squares_result.hpp"
#include "amfls/matrix_operator.hpp"

namespace amfls::math {

void validate_common_inputs(
    const MatrixOperator& matrix,
    const double* right_hand_side,
    double regularization,
    double tolerance);

void evaluate_accuracy_contract(
    LeastSquaresResult& result,
    double right_hand_side_norm,
    double regularization,
    double tolerance,
    std::optional<double> ridge_energy_error_upper_bound = std::nullopt);

// True only for a common-validator numerical failure.  Solver-specific cap,
// recurrence, and solver policies may change other terminal statuses but
// must never promote or overwrite this state before applying their declared
// numerical-failure policy.
bool candidate_validation_failed_numerically(
    const LeastSquaresResult& result) noexcept;

LeastSquaresResult validate_original_candidate(
    const MatrixOperator& matrix,
    const double* right_hand_side,
    std::span<const double> solution,
    double regularization,
    double tolerance,
    double operator_norm_lower_bound,
    RunStatistics& statistics);

}  // namespace amfls::math
