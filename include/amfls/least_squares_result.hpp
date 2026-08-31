#pragma once

#include <limits>
#include <vector>

namespace amfls {

enum class SolverStatus {
    success = 0,
    work_limit = 1,
    basis_limit = 2,
    precision_limit = 3,
    numerical_breakdown = 4
};

enum class StopReason {
    certified_optimality = 0,
    exact_stationarity = 1,
    exhausted_search_space = 2,
    maximum_depth = 3,
    maximum_epochs = 4,
    maximum_basis = 5,
    precision_limit = 6,
    numerical_breakdown = 7,
    compatible_backward_error = 8,
    least_squares_backward_error = 9,
    relative_energy_error = 10
};

// Outcome of the optional one-step ridge correction.
enum class RidgeCorrectionDisposition : int {
    // Ordinary least squares (lambda == 0): the correction does not apply.
    not_applicable = 0,
    // The base ridge certificate already passed, so no correction was needed.
    skipped_base_pass = 1,
    // The candidate has zero solution-energy norm, so the relative gate
    // cannot justify attempting the optional correction.
    skipped_zero_energy = 2,
    // A candidate-level primitive or vector is nonfinite before the optional
    // correction callbacks begin.
    skipped_nonfinite_candidate = 3,
    // Both callbacks and the correction construction completed, and the
    // finite corrected bound strictly improved the bound used for stopping.
    attempted_improved = 4,
    // The completed correction produced a valid (possibly +inf) bound that
    // did not strictly improve the base bound.
    attempted_not_improved = 5,
    // Both callbacks completed, but the finite curvature was nonpositive.
    attempted_invalid_curvature = 6,
    // The correction A callback completed with an unusable result, so the
    // corresponding A* callback was not made.
    attempted_partial_failure = 7,
    // A* or a later correction-construction primitive was nonfinite/invalid.
    attempted_nonfinite = 8
};

struct RunStatistics {
    long long a_columns = 0;
    long long at_columns = 0;
    long long a_block_calls = 0;
    long long at_block_calls = 0;

    long long search_a_columns = 0;
    long long search_at_columns = 0;
    long long validation_a_columns = 0;
    long long validation_at_columns = 0;

    // Baseline phase accounting.  The sketch and iterative counters are
    // disjoint subsets of search_*; validation remains independent.  Random
    // counts refer to generated standard-normal columns and scalar values.
    long long sketch_a_columns = 0;
    long long sketch_at_columns = 0;
    long long sketch_a_block_calls = 0;
    long long sketch_at_block_calls = 0;
    long long iterative_a_columns = 0;
    long long iterative_at_columns = 0;
    long long iterative_a_block_calls = 0;
    long long iterative_at_block_calls = 0;
    long long gaussian_random_columns = 0;
    long long gaussian_random_values = 0;

    double total_seconds = 0.0;
    double a_seconds = 0.0;
    double at_seconds = 0.0;
    double orthogonalization_seconds = 0.0;
    double projected_solve_seconds = 0.0;
    double validation_seconds = 0.0;

    // Detailed validation counters. Search calls are original-operator
    // invocations. The
    // base pair evaluates Ax-b and A^*(Ax-b)+lambda*x; the correction
    // partition is the optional fresh A/A^* pair used by the one-step update.
    long long search_a_block_calls = 0;
    long long search_at_block_calls = 0;
    long long validation_a_block_calls = 0;
    long long validation_at_block_calls = 0;
    long long base_validation_a_columns = 0;
    long long base_validation_at_columns = 0;
    long long base_validation_a_block_calls = 0;
    long long base_validation_at_block_calls = 0;
    long long ridge_correction_a_columns = 0;
    long long ridge_correction_at_columns = 0;
    long long ridge_correction_a_block_calls = 0;
    long long ridge_correction_at_block_calls = 0;
    double base_validation_seconds = 0.0;
    double ridge_correction_validation_seconds = 0.0;

    long long gaussian_random_block_requests = 0;
};

struct IterationRecord {
    int epoch = 0;
    int depth = 0;
    int auxiliary_width = 0;
    int basis_rank = 0;

    long long a_columns = 0;
    long long at_columns = 0;
    long long a_block_calls = 0;
    long long at_block_calls = 0;

    double objective = 0.0;
    double residual_norm = 0.0;
    double augmented_residual_norm = 0.0;
    double normal_residual_norm = 0.0;
    double operator_norm_lower_bound = 0.0;
    // With a valid MatrixOperator validation model, these are the final
    // outward-rounded original-problem upper bounds used by the common
    // validator. Without such a model they are +infinity and cannot pass.
    double compatible_backward_error_upper_bound = 0.0;
    double least_squares_backward_error_upper_bound = 0.0;
    double backward_error_upper_bound = 0.0;
    double relative_energy_error_upper_bound = 0.0;
    double relative_normal_residual_upper_bound = 0.0;

    // Accuracy and ridge-correction values at this checkpoint.
    double solution_norm = 0.0;
    double solution_energy_norm = 0.0;
    double augmented_operator_norm_lower_bound = 0.0;
    double energy_error_upper_bound = 0.0;
    double objective_gap_upper_bound = 0.0;
    SolverStatus contract_status = SolverStatus::work_limit;
    StopReason contract_stop_reason = StopReason::maximum_depth;
    bool contract_passed = false;
    RidgeCorrectionDisposition ridge_correction_disposition =
        RidgeCorrectionDisposition::not_applicable;
    double ridge_base_energy_error_upper_bound =
        std::numeric_limits<double>::quiet_NaN();
    double ridge_corrected_energy_error_upper_bound =
        std::numeric_limits<double>::quiet_NaN();
    double ridge_correction_gamma =
        std::numeric_limits<double>::quiet_NaN();
    double ridge_correction_z_h_z =
        std::numeric_limits<double>::quiet_NaN();
    double ridge_correction_two_abs_z_t_q =
        std::numeric_limits<double>::quiet_NaN();
    double ridge_correction_q_norm_squared =
        std::numeric_limits<double>::quiet_NaN();

};

struct LeastSquaresResult {
    int rows = 0;
    int cols = 0;
    int iterations = 0;
    int depth = 0;
    int auxiliary_width = 0;
    int basis_rank = 0;

    std::vector<double> solution;
    double objective = 0.0;
    double residual_norm = 0.0;
    double augmented_residual_norm = 0.0;
    double normal_residual_norm = 0.0;
    double solution_norm = 0.0;
    double solution_energy_norm = 0.0;
    double operator_norm_lower_bound = 0.0;
    double augmented_operator_norm_lower_bound = 0.0;

    // Final outward-rounded upper bounds for the two standard backward-error
    // constructions and their minimum. The common validator includes its
    // callback and norm error envelope; without a valid MatrixOperator model
    // these values are +infinity and cannot produce success. The augmented
    // system [A; sqrt(lambda) I] is used when lambda is positive.
    double compatible_backward_error_upper_bound = 0.0;
    double least_squares_backward_error_upper_bound = 0.0;
    double backward_error_upper_bound = 0.0;

    // Retained as a historical milestone-0.1 diagnostic.
    // It is not the stopping quantity; see docs/ALGORITHM_CONTRACT.md.
    double relative_normal_residual_upper_bound = 0.0;

    // For lambda > 0, these are outward-rounded upper bounds for
    // ||x-x_lambda||_{H_lambda}, its relative form, and
    // F_lambda(x)-F_lambda(x_lambda). A ridge success is admitted only from
    // the finite relative upper bound.
    double energy_error_upper_bound = 0.0;
    double relative_energy_error_upper_bound = 0.0;
    double objective_gap_upper_bound = 0.0;

    SolverStatus status = SolverStatus::work_limit;
    StopReason stop_reason = StopReason::maximum_depth;
    RunStatistics statistics;
    std::vector<IterationRecord> trace;

    // Ridge inverse-energy correction diagnostics. d0 is the strict
    // lambda-only base bound and d_corr is the independently evaluated
    // one-step bound. Gamma is the nominal Galerkin coefficient; z_h_z,
    // two_abs_z_t_q, and q_norm_squared are outward upper bounds used in
    // true_error^2 <= z_h_z + two_abs_z_t_q +
    // q_norm_squared/lambda <= d_corr^2.
    // Inapplicable values are quiet NaNs. A correction is selected only when
    // its finite strict bound improves d0.
    RidgeCorrectionDisposition ridge_correction_disposition =
        RidgeCorrectionDisposition::not_applicable;
    double ridge_base_energy_error_upper_bound =
        std::numeric_limits<double>::quiet_NaN();
    double ridge_corrected_energy_error_upper_bound =
        std::numeric_limits<double>::quiet_NaN();
    double ridge_correction_gamma =
        std::numeric_limits<double>::quiet_NaN();
    double ridge_correction_z_h_z =
        std::numeric_limits<double>::quiet_NaN();
    double ridge_correction_two_abs_z_t_q =
        std::numeric_limits<double>::quiet_NaN();
    double ridge_correction_q_norm_squared =
        std::numeric_limits<double>::quiet_NaN();

};

}  // namespace amfls
