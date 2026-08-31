#include <cfenv>
#include <cstdint>
#include <stdexcept>
#include <vector>

#include "algorithms/mathematics/operators/sparse_csr_operator.hpp"
#include "algorithms/mathematics/krylov/candidate_validation.hpp"
#include "test_helpers.hpp"

int main() {
    // The final row is empty and the final column is structurally zero.  The
    // rectangular shape also distinguishes the leading dimensions of the
    // column-major input and output blocks.
    const std::vector<std::int64_t> row_offsets{0, 2, 3, 5, 7, 7};
    const std::vector<std::int64_t> column_indices{0, 2, 1, 0, 2, 1, 2};
    const std::vector<double> values{2.0, -1.0, 3.0, 4.0, 5.0, -2.0, 1.0};
    amfls::math::SparseCsrOperator matrix(
        5,
        4,
        row_offsets.data(),
        column_indices.data(),
        values.data(),
        static_cast<std::int64_t>(values.size()));
    const auto model = matrix.validation_error_model();
    require_test(
        std::isfinite(model.operator_norm_upper_bound) &&
            model.operator_norm_upper_bound >= std::sqrt(60.0) &&
            model.operator_norm_lower_bound > 0.0 &&
            model.operator_norm_lower_bound <=
                model.operator_norm_upper_bound &&
            model.apply_rounding_steps == 6 &&
            model.apply_transpose_rounding_steps == 8,
        "CSR validation model uses maximum row and column lengths");
    require_test(
        matrix.relative_block_product_cost(1) == 1.0 &&
            matrix.relative_block_product_cost(3) == 3.0 &&
            !std::isfinite(matrix.relative_block_product_cost(0)),
        "CSR block products charge every sparse matrix-vector column");

    const std::vector<double> single_x{-2.0, 1.0, 0.5, -9.0};
    std::vector<double> single_applied(5, 17.0);
    matrix.apply(single_x.data(), 1, single_applied.data());
    const std::vector<double> expected_single_applied{
        -4.5, 3.0, -5.5, -1.5, 0.0};
    for (int index = 0; index < 5; ++index) {
        require_near(
            single_applied[index],
            expected_single_applied[index],
            1e-14,
            "single-column CSR apply");
    }

    const std::vector<double> x{
        1.0, 2.0, 3.0, 7.0,
        -2.0, 1.0, 0.5, -9.0,
        0.0, -1.0, 2.0, 11.0};
    std::vector<double> applied(15, -23.0);
    matrix.apply(x.data(), 3, applied.data());
    const std::vector<double> expected_applied{
        -1.0, 6.0, 19.0, -1.0, 0.0,
        -4.5, 3.0, -5.5, -1.5, 0.0,
        -2.0, -3.0, 10.0, 4.0, 0.0};
    for (int index = 0; index < 15; ++index) {
        require_near(
            applied[index],
            expected_applied[index],
            1e-14,
            "multi-column CSR apply");
    }

    const std::vector<double> single_y{3.0, 1.0, -4.0, -2.0, -7.0};
    std::vector<double> single_transposed(4, 31.0);
    matrix.apply_transpose(single_y.data(), 1, single_transposed.data());
    const std::vector<double> expected_single_transposed{
        -10.0, 7.0, -25.0, 0.0};
    for (int index = 0; index < 4; ++index) {
        require_near(
            single_transposed[index],
            expected_single_transposed[index],
            1e-14,
            "single-column CSR transpose");
    }

    const std::vector<double> y{
        1.0, -2.0, 0.5, 4.0, 8.0,
        3.0, 1.0, -4.0, -2.0, -7.0,
        -1.0, 0.25, 2.0, 3.0, 9.0};
    std::vector<double> transposed(12, -37.0);
    matrix.apply_transpose(y.data(), 3, transposed.data());
    const std::vector<double> expected_transposed{
        4.0, -14.0, 5.5, 0.0,
        -10.0, 7.0, -25.0, 0.0,
        6.0, -5.25, 14.0, 0.0};
    for (int index = 0; index < 12; ++index) {
        require_near(
            transposed[index],
            expected_transposed[index],
            1e-14,
            "multi-column CSR transpose");
    }

    bool rejected = false;
    const std::vector<std::int64_t> invalid_columns{0, 4, 1, 0, 2, 1, 2};
    try {
        (void)amfls::math::SparseCsrOperator(
            5,
            4,
            row_offsets.data(),
            invalid_columns.data(),
            values.data(),
            static_cast<std::int64_t>(values.size()));
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    require_test(rejected, "CSR must reject an out-of-range column");

    rejected = false;
    const std::vector<std::int64_t> duplicate_offsets{0, 2};
    const std::vector<std::int64_t> duplicate_columns{0, 0};
    const std::vector<double> duplicate_values{1.0, 2.0};
    try {
        (void)amfls::math::SparseCsrOperator(
            1,
            1,
            duplicate_offsets.data(),
            duplicate_columns.data(),
            duplicate_values.data(),
            2);
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    require_test(
        rejected,
        "CSR must reject duplicate column indices within one row");

    const std::vector<std::int64_t> identity_offsets{0, 1, 2};
    const std::vector<std::int64_t> identity_columns{0, 1};
    const std::vector<double> identity_values{1.0, 1.0};
    amfls::math::SparseCsrOperator identity(
        2,
        2,
        identity_offsets.data(),
        identity_columns.data(),
        identity_values.data(),
        2);
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
        "CSR strict cancellation certificate");

    const int original_rounding = std::fegetround();
    require_test(
        original_rounding != -1 && std::fesetround(FE_UPWARD) == 0,
        "test platform must expose its floating-point rounding mode");
    const auto wrong_rounding_candidate =
        amfls::math::validate_original_candidate(
            identity,
            exact_rhs.data(),
            exact_solution,
            0.0,
            1e-12,
            0.0,
            statistics);
    amfls::math::SparseCsrOperator constructed_under_wrong_rounding(
        2,
        2,
        identity_offsets.data(),
        identity_columns.data(),
        identity_values.data(),
        2);
    require_test(
        std::fesetround(original_rounding) == 0,
        "test must restore its floating-point rounding mode");
    require_test(
        wrong_rounding_candidate.status != amfls::SolverStatus::success &&
            std::isinf(
                wrong_rounding_candidate.backward_error_upper_bound),
        "strict certification is disabled outside round-to-nearest");
    const auto invalid_cached_model_candidate =
        amfls::math::validate_original_candidate(
            constructed_under_wrong_rounding,
            exact_rhs.data(),
            exact_solution,
            0.0,
            1e-12,
            0.0,
            statistics);
    require_test(
        invalid_cached_model_candidate.status !=
                amfls::SolverStatus::success &&
            std::isinf(
                invalid_cached_model_candidate
                    .backward_error_upper_bound),
        "bounds cached outside round-to-nearest stay uncertified");

    const std::vector<std::int64_t> analytic_offsets{0, 2, 2};
    const std::vector<std::int64_t> analytic_columns{0, 1};
    const std::vector<double> analytic_values{3.0, 4.0};
    amfls::math::SparseCsrOperator analytic(
        2,
        2,
        analytic_offsets.data(),
        analytic_columns.data(),
        analytic_values.data(),
        2);
    const auto analytic_model = analytic.validation_error_model();
    require_test(
        analytic_model.operator_norm_lower_bound > 4.9 &&
            analytic_model.operator_norm_lower_bound <= 5.0 &&
            analytic_model.operator_norm_upper_bound >= 5.0 &&
            analytic_model.operator_norm_upper_bound <=
                5.0 * (1.0 + 1e-14) &&
            std::isfinite(analytic_model.operator_norm_upper_bound),
        "CSR Frobenius model tightly contains analytic spectral norm");
}
