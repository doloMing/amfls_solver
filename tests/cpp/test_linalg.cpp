#include <algorithm>
#include <array>
#include <cfenv>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <utility>

#include "algorithms/mathematics/linalg/blas_lapack.hpp"
#include "algorithms/mathematics/linalg/matrix.hpp"
#include "algorithms/mathematics/linalg/svd.hpp"
#include "algorithms/mathematics/random/gaussian.hpp"
#include "test_helpers.hpp"

int main() {
    using amfls::math::Matrix;
    const double binary64_unit_roundoff =
        0.5 * std::numeric_limits<double>::epsilon();
    require_test(
        amfls::math::fp::gradual_underflow_is_active(),
        "test platform must preserve subnormal inputs and outputs");
    const double one_step_gamma = amfls::math::fp::binary64_gamma(1);
    const double many_step_gamma = amfls::math::fp::binary64_gamma(1024);
    require_test(
        one_step_gamma >= binary64_unit_roundoff &&
            one_step_gamma <= 2.0 * binary64_unit_roundoff &&
            many_step_gamma >= 1024.0 * binary64_unit_roundoff,
        "callback gamma must use binary64 unit roundoff");
    require_test(
        amfls::math::fp::positive_sum_upper_bound(0.0, 1024) == 0.0 &&
            amfls::math::fp::positive_sum_upper_bound(1.0, 1024) >= 1.0 &&
            amfls::math::fp::positive_sum_upper_bound(
                std::numeric_limits<double>::denorm_min(), 2) >=
                std::numeric_limits<double>::denorm_min(),
        "positive-sum outward envelope covers zero, normal, and subnormal sums");
    require_test(
        amfls::math::fp::positive_fma_square_sum_upper_bound(0.0, 1) >=
            std::numeric_limits<double>::denorm_min(),
        "FMA square-sum envelope covers a product that underflows to zero");
    const double analytic_square_lower =
        amfls::math::fp::positive_sum_lower_bound(25.0, 2);
    require_test(
        analytic_square_lower > 0.0 && analytic_square_lower <= 25.0 &&
            amfls::math::fp::positive_sum_lower_bound(
                std::numeric_limits<double>::denorm_min(), 1) == 0.0 &&
            amfls::math::fp::positive_sum_lower_bound(
                std::numeric_limits<double>::infinity(), 1) == 0.0,
        "positive-sum lower envelope covers analytic and degraded cases");
    require_test(
        amfls::math::fp::upward_multiply(
            std::numeric_limits<double>::denorm_min(),
            std::numeric_limits<double>::denorm_min()) > 0.0 &&
            amfls::math::fp::upward_divide(
                std::numeric_limits<double>::denorm_min(),
                std::numeric_limits<double>::max()) > 0.0,
        "positive outward operations cannot underflow to zero");
    const std::array<double, 2> analytic_norm_values{3.0, 4.0};
    const auto analytic_norm = amfls::math::fp::norm_interval(
        std::span<const double>(analytic_norm_values));
    require_test(
        analytic_norm.lower <= 5.0 && analytic_norm.upper >= 5.0 &&
            analytic_norm.lower > 5.0 * (1.0 - 1e-14) &&
            analytic_norm.upper < 5.0 * (1.0 + 1e-14),
        "scaled norm interval tightly contains an analytic norm");
    const std::array<double, 3> dynamic_norm_values{
        3e300, 4e300, 3e-300};
    const auto dynamic_norm = amfls::math::fp::norm_interval(
        std::span<const double>(dynamic_norm_values));
    require_test(
        dynamic_norm.lower <= 5e300 && dynamic_norm.upper >= 5e300 &&
            dynamic_norm.lower > 5e300 * (1.0 - 1e-14) &&
            dynamic_norm.upper < 5e300 * (1.0 + 1e-14),
        "scaled norm interval contains a huge-dynamic-range norm");
    const std::array<double, 1> maximum_norm_value{
        std::numeric_limits<double>::max()};
    const auto maximum_norm = amfls::math::fp::norm_interval(
        std::span<const double>(maximum_norm_value));
    require_test(
        maximum_norm.lower <= maximum_norm_value[0] &&
            maximum_norm.upper >= maximum_norm_value[0],
        "scaled norm interval contains DBL_MAX");
    const std::array<double, 1> subnormal_norm_value{
        std::numeric_limits<double>::denorm_min()};
    const auto subnormal_norm = amfls::math::fp::norm_interval(
        std::span<const double>(subnormal_norm_value));
    require_test(
        subnormal_norm.lower <= subnormal_norm_value[0] &&
            subnormal_norm.upper >= subnormal_norm_value[0] &&
            subnormal_norm.upper > 0.0,
        "scaled norm interval contains the minimum subnormal");
    const std::array<double, 1> nonfinite_norm_value{
        std::numeric_limits<double>::infinity()};
    const auto nonfinite_norm = amfls::math::fp::norm_interval(
        std::span<const double>(nonfinite_norm_value));
    require_test(
        nonfinite_norm.lower == 0.0 &&
            std::isinf(nonfinite_norm.upper),
        "scaled norm interval rejects nonfinite input");
    const int saved_rounding_mode = std::fegetround();
    require_test(
        saved_rounding_mode != -1 && std::fesetround(FE_UPWARD) == 0,
        "norm test must set a directed rounding mode");
    const auto directed_rounding_norm = amfls::math::fp::norm_interval(
        std::span<const double>(analytic_norm_values));
    require_test(
        std::fesetround(saved_rounding_mode) == 0,
        "norm test must restore round-to-nearest");
    require_test(
        directed_rounding_norm.lower == 0.0 &&
            std::isinf(directed_rounding_norm.upper),
        "scaled norm interval disables itself outside round-to-nearest");

    Matrix matrix(4, 3);
    matrix(0, 0) = 4.0;
    matrix(1, 1) = 2.0;
    matrix(2, 2) = 0.5;
    const amfls::math::SvdResult svd = amfls::math::thin_svd(matrix);
    require_near(svd.singular_values[0], 4.0, 1e-12, "largest singular value");
    require_near(svd.singular_values[1], 2.0, 1e-12, "middle singular value");
    require_near(svd.singular_values[2], 0.5, 1e-12, "smallest singular value");

    Matrix empty_matrix(0, 0);
    require_near(
        amfls::math::frobenius_norm(empty_matrix),
        0.0,
        0.0,
        "empty Frobenius norm");
    Matrix large_norm_input(2, 1);
    large_norm_input(0, 0) = 3.0e200;
    large_norm_input(1, 0) = 4.0e200;
    require_near(
        amfls::math::frobenius_norm(large_norm_input) / 5.0e200,
        1.0,
        2e-15,
        "overflow-safe Frobenius norm");
    Matrix tiny_norm_input(2, 1);
    tiny_norm_input(0, 0) = 3.0e-200;
    tiny_norm_input(1, 0) = 4.0e-200;
    require_near(
        amfls::math::frobenius_norm(tiny_norm_input) / 5.0e-200,
        1.0,
        2e-15,
        "underflow-safe Frobenius norm");

    Matrix interleaved_norm_input(2, 2);
    interleaved_norm_input(0, 0) = 3.0e200;
    interleaved_norm_input(0, 1) = 4.0e200;
    interleaved_norm_input(1, 0) = 3.0e-200;
    interleaved_norm_input(1, 1) = 4.0e-200;
    const std::vector<double> interleaved_row_norms =
        amfls::math::row_norms(interleaved_norm_input);
    require_near(
        interleaved_row_norms[0] / 5.0e200,
        1.0,
        2e-15,
        "overflow-safe interleaved row norm");
    require_near(
        interleaved_row_norms[1] / 5.0e-200,
        1.0,
        2e-15,
        "underflow-safe interleaved row norm");

    Matrix full = amfls::math::gaussian_matrix(7, 5, 13, 4, 0);
    Matrix prefix = amfls::math::gaussian_matrix(7, 2, 13, 4, 0);
    Matrix suffix = amfls::math::gaussian_matrix(7, 3, 13, 4, 2);
    for (int row = 0; row < 7; ++row) {
        for (int col = 0; col < 2; ++col) {
            require_near(full(row, col), prefix(row, col), 0.0, "Gaussian prefix replay");
        }
        for (int col = 0; col < 3; ++col) {
            require_near(full(row, col + 2), suffix(row, col), 0.0, "Gaussian suffix replay");
        }
    }

    constexpr std::array<std::uint64_t, 4> uint64_boundaries{
        0ULL,
        0x7fffffffffffffffULL,
        0x8000000000000000ULL,
        0xffffffffffffffffULL};
    constexpr std::array<double, 4> seed_boundary_gaussians{
        -0x1.51a00aae743cep-2,
        -0x1.d199f975f9c05p-2,
        0x1.67af63361327ap-1,
        -0x1.ab96d15bbf0a3p-2};
    constexpr std::array<double, 4> stream_boundary_gaussians{
        -0x1.46af8e27e24f1p+0,
        0x1.f4e39c58f4677p-1,
        0x1.b7a12aa5cb265p-1,
        -0x1.d2f6405176b19p-1};
    for (std::size_t index = 0; index < uint64_boundaries.size(); ++index) {
        const Matrix seed_sample = amfls::math::gaussian_matrix(
            1, 1, uint64_boundaries[index], 17, 0);
        require_near(
            seed_sample(0, 0),
            seed_boundary_gaussians[index],
            2e-15,
            "full-uint64 Gaussian seed boundary");
        const Matrix stream_sample = amfls::math::gaussian_matrix(
            1, 1, 23, uint64_boundaries[index], 0);
        require_near(
            stream_sample(0, 0),
            stream_boundary_gaussians[index],
            2e-15,
            "full-uint64 Gaussian stream boundary");
    }

    const double tiny_direction[7]{
        1e-200, -2e-200, 3e-200, -4e-200, 5e-200, -6e-200, 7e-200};
    Matrix projected = amfls::math::gaussian_matrix_orthogonal_to(
        7, 5, tiny_direction, 13, 4, 0);
    Matrix projected_prefix = amfls::math::gaussian_matrix_orthogonal_to(
        7, 2, tiny_direction, 13, 4, 0);
    const double tiny_norm = amfls::math::vector_norm(tiny_direction, 7);
    double normalized_direction[7];
    for (int row = 0; row < 7; ++row) {
        normalized_direction[row] = tiny_direction[row] / tiny_norm;
        for (int col = 0; col < 2; ++col) {
            require_near(
                projected(row, col),
                projected_prefix(row, col),
                0.0,
                "projected Gaussian prefix replay");
        }
    }
    for (int col = 0; col < projected.cols(); ++col) {
        require_near(
            amfls::math::vector_dot(
                normalized_direction, projected.column_data(col), 7),
            0.0,
            2e-15,
            "scale-safe Gaussian projection");
    }

    Matrix dependent(5, 3);
    for (int row = 0; row < 5; ++row) {
        dependent(row, 0) = row + 1.0;
        dependent(row, 1) = 2.0 * dependent(row, 0);
        dependent(row, 2) = row == 0 ? 1.0 : 0.0;
    }
    Matrix basis = amfls::math::numerical_column_space(dependent);
    require_test(basis.cols() == 2, "numerical column space must remove dependence");
    Matrix gram = amfls::math::transpose_multiply(basis, basis);
    require_near(gram(0, 0), 1.0, 1e-12, "basis norm 0");
    require_near(gram(1, 1), 1.0, 1e-12, "basis norm 1");
    require_near(gram(0, 1), 0.0, 1e-12, "basis orthogonality");

    const amfls::math::SpdTridiagonalSolve tridiagonal =
        amfls::math::solve_spd_tridiagonal(
            {4.0, 5.0, 6.0}, {1.0, 2.0}, {2.0, -3.0, 14.0});
    require_test(
        tridiagonal.positive_definite,
        "SPD tridiagonal factorization must succeed");
    require_test(
        tridiagonal.reciprocal_condition > 0.0 &&
            tridiagonal.reciprocal_condition <= 1.0,
        "SPD tridiagonal condition estimate");
    require_near(tridiagonal.solution[0], 1.0, 1e-13, "tridiagonal x0");
    require_near(tridiagonal.solution[1], -2.0, 1e-13, "tridiagonal x1");
    require_near(tridiagonal.solution[2], 3.0, 1e-13, "tridiagonal x2");

    Matrix dense_spd(3, 3);
    dense_spd(0, 0) = 4.0;
    dense_spd(1, 0) = 1.0;
    dense_spd(0, 1) = 1.0;
    dense_spd(1, 1) = 5.0;
    dense_spd(2, 1) = 2.0;
    dense_spd(1, 2) = 2.0;
    dense_spd(2, 2) = 6.0;
    const amfls::math::DenseSpdSolve dense =
        amfls::math::solve_dense_spd(
            std::move(dense_spd), {2.0, -3.0, 14.0});
    require_test(dense.positive_definite, "dense SPD factorization");
    require_near(dense.solution[0], 1.0, 1e-13, "dense SPD x0");
    require_near(dense.solution[1], -2.0, 1e-13, "dense SPD x1");
    require_near(dense.solution[2], 3.0, 1e-13, "dense SPD x2");

    Matrix rank_deficient(4, 3);
    rank_deficient(0, 0) = 1.0;
    rank_deficient(0, 1) = 2.0;
    rank_deficient(1, 2) = 1.0;
    const double rank_deficient_rhs[4]{3.0, 4.0, 0.0, 0.0};
    const amfls::math::MinimumNormLeastSquaresResult least_squares =
        amfls::math::minimum_norm_least_squares(
            rank_deficient, rank_deficient_rhs, 1e-12);
    require_test(least_squares.rank == 2, "rank-revealing least-squares rank");
    require_near(least_squares.solution[0], 0.6, 1e-12, "minimum-norm x0");
    require_near(least_squares.solution[1], 1.2, 1e-12, "minimum-norm x1");
    require_near(least_squares.solution[2], 4.0, 1e-12, "minimum-norm x2");
    Matrix consumed_rank_deficient = rank_deficient;
    const amfls::math::MinimumNormLeastSquaresResult consumed_least_squares =
        amfls::math::minimum_norm_least_squares(
            std::move(consumed_rank_deficient), rank_deficient_rhs, 1e-12);
    require_test(
        consumed_least_squares.rank == least_squares.rank,
        "consuming least-squares rank");
    for (int index = 0;
         index < static_cast<int>(least_squares.solution.size());
         ++index) {
        require_near(
            consumed_least_squares.solution[index],
            least_squares.solution[index],
            1e-14,
            "consuming least-squares solution");
    }

    Matrix underdetermined_rank_deficient(2, 3);
    underdetermined_rank_deficient(0, 0) = 1.0;
    underdetermined_rank_deficient(0, 1) = 1.0;
    underdetermined_rank_deficient(1, 2) = 2.0;
    const double underdetermined_rhs[2]{2.0, 4.0};
    const amfls::math::MinimumNormLeastSquaresResult underdetermined =
        amfls::math::minimum_norm_least_squares(
            underdetermined_rank_deficient, underdetermined_rhs, 1e-12);
    require_test(
        underdetermined.rank == 2,
        "underdetermined rank-revealing least-squares rank");
    require_near(
        underdetermined.solution[0],
        1.0,
        1e-12,
        "underdetermined minimum-norm x0");
    require_near(
        underdetermined.solution[1],
        1.0,
        1e-12,
        "underdetermined minimum-norm x1");
    require_near(
        underdetermined.solution[2],
        2.0,
        1e-12,
        "underdetermined minimum-norm x2");
    Matrix consumed_underdetermined = underdetermined_rank_deficient;
    const amfls::math::MinimumNormLeastSquaresResult
        consumed_underdetermined_result =
            amfls::math::minimum_norm_least_squares(
                std::move(consumed_underdetermined),
                underdetermined_rhs,
                1e-12);
    require_test(
        consumed_underdetermined_result.rank == underdetermined.rank,
        "consuming underdetermined least-squares rank");
    for (int index = 0;
         index < static_cast<int>(underdetermined.solution.size());
         ++index) {
        require_near(
            consumed_underdetermined_result.solution[index],
            underdetermined.solution[index],
            1e-14,
            "consuming underdetermined least-squares solution");
    }

    Matrix empty_columns(3, 0);
    const double empty_columns_rhs[3]{1.0, -2.0, 3.0};
    const amfls::math::MinimumNormLeastSquaresResult empty_columns_result =
        amfls::math::minimum_norm_least_squares(
            std::move(empty_columns), empty_columns_rhs, 1e-12);
    require_test(
        empty_columns_result.rank == 0 &&
            empty_columns_result.solution.empty(),
        "consuming empty least-squares problem");
    std::cout << "test_linalg passed\n";
}
