#include <cfenv>
#include <stdexcept>
#include <vector>

#include "algorithms/mathematics/operators/dense_operator.hpp"
#include "algorithms/mathematics/operators/row_major_dense_operator.hpp"
#include "algorithms/mathematics/krylov/candidate_validation.hpp"
#include "test_helpers.hpp"

int main() {
    const std::vector<double> values{
        2.0, -1.0,
        0.0, 3.0,
        4.0, 5.0};
    amfls::math::RowMajorDenseOperator matrix(3, 2, values.data());
    const auto model = matrix.validation_error_model();
    require_test(
        std::isfinite(model.operator_norm_upper_bound) &&
            model.operator_norm_upper_bound >= std::sqrt(55.0) &&
            model.operator_norm_lower_bound > 0.0 &&
            model.operator_norm_lower_bound <=
                model.operator_norm_upper_bound &&
            model.apply_rounding_steps == 6 &&
            model.apply_transpose_rounding_steps == 8,
        "row-major dense validation error model");
    require_test(
        matrix.relative_block_product_cost(1) == 1.0 &&
            matrix.relative_block_product_cost(8) == 1.0 &&
            !std::isfinite(matrix.relative_block_product_cost(0)),
        "row-major dense block products amortize one stored-matrix pass");

    const std::vector<double> x{
        1.0, 2.0,
        -2.0, 0.5};
    std::vector<double> applied(6);
    matrix.apply(x.data(), 2, applied.data());
    const std::vector<double> expected_applied{
        0.0, 6.0, 14.0,
        -4.5, 1.5, -5.5};
    for (int index = 0; index < 6; ++index) {
        require_near(
            applied[index],
            expected_applied[index],
            1e-14,
            "row-major dense apply");
    }

    const std::vector<double> y{
        1.0, -2.0, 0.5,
        3.0, 1.0, -4.0};
    std::vector<double> transposed(4);
    matrix.apply_transpose(y.data(), 2, transposed.data());
    const std::vector<double> expected_transposed{
        4.0, -4.5,
        -10.0, -20.0};
    for (int index = 0; index < 4; ++index) {
        require_near(
            transposed[index],
            expected_transposed[index],
            1e-14,
            "row-major dense transpose");
    }

    bool rejected = false;
    try {
        (void)amfls::math::RowMajorDenseOperator(3, 2, nullptr);
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    require_test(rejected, "row-major dense operator must reject null values");

    // Exact cancellation in the fresh residual remains a success only after
    // the callback and subtraction radii have been included.
    const std::vector<double> identity_values{1.0, 0.0, 0.0, 1.0};
    amfls::math::RowMajorDenseOperator identity(
        2, 2, identity_values.data());
    const std::vector<double> exact_solution{1.0, -1.0};
    const std::vector<double> exact_rhs{1.0, -1.0};
    amfls::RunStatistics statistics;
    const auto candidate = amfls::math::validate_original_candidate(
        identity,
        exact_rhs.data(),
        exact_solution,
        0.0,
        1e-12,
        0.0,
        statistics);
    require_test(
        candidate.status == amfls::SolverStatus::success &&
            candidate.backward_error_upper_bound >= 0.0 &&
            candidate.backward_error_upper_bound <= 1e-12 &&
            candidate.operator_norm_lower_bound > 0.0,
        "row-major dense strict cancellation certificate");

    // A large residual component in the exact left null space makes the
    // binary64 transpose-callback radius dominate an exactly zero ordinary
    // least-squares gradient. The stored operator repeats the same complete
    // validation chain in the extended accumulator and charges both passes.
    constexpr int ordinary_refinement_rows = 64;
    constexpr int ordinary_refinement_cols = 2;
    std::vector<double> ordinary_refinement_values(
        ordinary_refinement_rows * ordinary_refinement_cols, 0.0);
    ordinary_refinement_values[0] = 1.0e4;
    ordinary_refinement_values[ordinary_refinement_cols + 1] = 1.0;
    amfls::math::RowMajorDenseOperator ordinary_refinement_operator(
        ordinary_refinement_rows,
        ordinary_refinement_cols,
        ordinary_refinement_values.data());
    std::vector<double> ordinary_refinement_rhs(
        ordinary_refinement_rows, 0.0);
    ordinary_refinement_rhs[1] = 1.0;
    ordinary_refinement_rhs[2] = 1.0e5;
    const std::vector<double> ordinary_refinement_solution{0.0, 1.0};
    amfls::RunStatistics ordinary_refinement_statistics;
    const auto ordinary_refined_candidate =
        amfls::math::validate_original_candidate(
            ordinary_refinement_operator,
            ordinary_refinement_rhs.data(),
            ordinary_refinement_solution,
            0.0,
            1.0e-15,
            0.0,
            ordinary_refinement_statistics);
    require_test(
        std::numeric_limits<long double>::digits >
                std::numeric_limits<double>::digits &&
            ordinary_refined_candidate.status ==
                amfls::SolverStatus::success &&
            ordinary_refined_candidate.backward_error_upper_bound <=
                1.0e-15 &&
            ordinary_refinement_statistics.base_validation_a_columns == 2 &&
            ordinary_refinement_statistics.base_validation_at_columns == 2 &&
            ordinary_refinement_statistics.ridge_correction_a_columns == 0 &&
            ordinary_refinement_statistics.ridge_correction_at_columns == 0,
        "stored ordinary validation refines a rounding-dominated base chain");

    // A large residual component orthogonal to range(A) makes the fast
    // binary64 A^T callback envelope dominate a small high-curvature ridge
    // gradient.  The stored operator may refine the complete base chain in
    // an extended accumulator and then reuse the common one-step certificate.
    constexpr int refinement_rows = 64;
    constexpr int refinement_cols = 2;
    std::vector<double> refinement_row_major(
        refinement_rows * refinement_cols, 0.0);
    refinement_row_major[0] = 1.0e4;
    refinement_row_major[refinement_cols + 1] = 1.0;
    amfls::math::RowMajorDenseOperator refinement_operator(
        refinement_rows,
        refinement_cols,
        refinement_row_major.data());
    std::vector<double> refinement_rhs(refinement_rows, 0.0);
    refinement_rhs[1] = 2.0;
    refinement_rhs[2] = 1.0e5;
    const std::vector<double> refinement_solution{1.0e-13, 1.0};
    amfls::RunStatistics refinement_statistics;
    const auto refined_candidate =
        amfls::math::validate_original_candidate(
            refinement_operator,
            refinement_rhs.data(),
            refinement_solution,
            1.0,
            1.0e-8,
            0.0,
            refinement_statistics);
    require_test(
        std::numeric_limits<long double>::digits >
                std::numeric_limits<double>::digits &&
            refined_candidate.status == amfls::SolverStatus::success &&
            refined_candidate.relative_energy_error_upper_bound <= 1.0e-8 &&
            refined_candidate.ridge_correction_disposition ==
                amfls::RidgeCorrectionDisposition::attempted_improved &&
            refinement_statistics.base_validation_a_columns == 2 &&
            refinement_statistics.base_validation_at_columns == 2 &&
            refinement_statistics.ridge_correction_a_columns == 1 &&
            refinement_statistics.ridge_correction_at_columns == 1,
        "stored ridge validation refines a rounding-dominated base chain");

    std::vector<double> refinement_column_major(
        refinement_rows * refinement_cols, 0.0);
    refinement_column_major[0] = 1.0e4;
    refinement_column_major[1 + refinement_rows] = 1.0;
    DenseTestOperator unsupported_refinement(
        refinement_rows,
        refinement_cols,
        refinement_column_major);
    amfls::RunStatistics unsupported_statistics;
    const auto unsupported_candidate =
        amfls::math::validate_original_candidate(
            unsupported_refinement,
            refinement_rhs.data(),
            refinement_solution,
            1.0,
            1.0e-8,
            0.0,
            unsupported_statistics);
    require_test(
        unsupported_candidate.status != amfls::SolverStatus::success &&
            unsupported_statistics.base_validation_a_columns == 1 &&
            unsupported_statistics.base_validation_at_columns == 1,
        "a custom operator does not enter built-in validation refinement");

    const int original_rounding = std::fegetround();
    require_test(
        original_rounding != -1 && std::fesetround(FE_UPWARD) == 0,
        "test platform must expose its floating-point rounding mode");
    amfls::RunStatistics wrong_rounding_statistics;
    const auto wrong_rounding_refinement =
        amfls::math::validate_original_candidate(
            refinement_operator,
            refinement_rhs.data(),
            refinement_solution,
            1.0,
            1.0e-8,
            0.0,
            wrong_rounding_statistics);
    require_test(
        std::fesetround(original_rounding) == 0,
        "test must restore its floating-point rounding mode");
    require_test(
        wrong_rounding_refinement.status !=
                amfls::SolverStatus::success &&
            wrong_rounding_statistics.base_validation_a_columns == 1 &&
            wrong_rounding_statistics.base_validation_at_columns == 1 &&
            wrong_rounding_statistics.ridge_correction_a_columns == 0 &&
            wrong_rounding_statistics.ridge_correction_at_columns == 0,
        "validation refinement is disabled outside round-to-nearest");

    // A=[3 4; 0 0] has analytic spectral norm five.  Audit both dense
    // layouts against the same exact value rather than against their shared
    // one/inf-norm construction.
    const std::vector<double> analytic_row_major{3.0, 4.0, 0.0, 0.0};
    const std::vector<double> analytic_column_major{3.0, 0.0, 4.0, 0.0};
    amfls::math::RowMajorDenseOperator analytic_row_operator(
        2, 2, analytic_row_major.data());
    amfls::math::DenseOperator analytic_column_operator(
        2, 2, analytic_column_major.data());
    for (const auto analytic_model : {
             analytic_row_operator.validation_error_model(),
             analytic_column_operator.validation_error_model()}) {
        require_test(
            analytic_model.operator_norm_lower_bound > 4.9 &&
                analytic_model.operator_norm_lower_bound <= 5.0 &&
                analytic_model.operator_norm_upper_bound >= 5.0 &&
                analytic_model.operator_norm_upper_bound <=
                    5.0 * (1.0 + 1e-14) &&
                std::isfinite(analytic_model.operator_norm_upper_bound),
            "dense Frobenius model tightly contains analytic spectral norm");
    }

    // For the signed Hadamard matrix, ||A||_2=sqrt(2) while
    // || |A| ||_2=2.  Callback-error bounds require the latter upper bound;
    // audit both built-in layouts and the valid custom test model.
    const std::vector<double> cancellation_values{1.0, 1.0, 1.0, -1.0};
    amfls::math::RowMajorDenseOperator cancellation_row_operator(
        2, 2, cancellation_values.data());
    amfls::math::DenseOperator cancellation_column_operator(
        2, 2, cancellation_values.data());
    DenseTestOperator cancellation_custom_operator(
        2, 2, cancellation_values);
    for (const auto cancellation_model : {
             cancellation_row_operator.validation_error_model(),
             cancellation_column_operator.validation_error_model(),
             cancellation_custom_operator.validation_error_model()}) {
        require_test(
            cancellation_model.operator_norm_upper_bound >= 2.0 &&
                cancellation_model.operator_norm_upper_bound <=
                    2.0 * (1.0 + 1e-14) &&
                cancellation_model.operator_norm_lower_bound <=
                    std::sqrt(2.0) * (1.0 + 1e-14),
            "validation model upper-bounds the absolute-matrix norm");
    }

    const std::vector<double> zero_values(4, 0.0);
    amfls::math::RowMajorDenseOperator zero(2, 2, zero_values.data());
    const auto zero_model = zero.validation_error_model();
    require_test(
        zero_model.operator_norm_lower_bound == 0.0 &&
            zero_model.operator_norm_upper_bound == 0.0,
        "zero dense matrix retains an exact validation model");

    const std::vector<double> subnormal_value{
        std::numeric_limits<double>::denorm_min()};
    amfls::math::RowMajorDenseOperator subnormal(
        1, 1, subnormal_value.data());
    const auto subnormal_model = subnormal.validation_error_model();
    require_test(
        subnormal_model.operator_norm_lower_bound == subnormal_value[0] &&
            subnormal_model.operator_norm_upper_bound >= subnormal_value[0],
        "subnormal dense model remains nonzero and contains the exact norm");

    const std::vector<double> overflowing_sum_values{
        std::numeric_limits<double>::max(),
        std::numeric_limits<double>::max()};
    amfls::math::RowMajorDenseOperator overflowing_sum(
        1, 2, overflowing_sum_values.data());
    const auto overflowing_sum_model =
        overflowing_sum.validation_error_model();
    require_test(
        std::isinf(overflowing_sum_model.operator_norm_upper_bound) &&
            overflowing_sum_model.operator_norm_lower_bound == 0.0,
        "overflowing positive sum disables dense certification");
}
