#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "amfls/lsrn.hpp"
#include "test_helpers.hpp"

namespace {

DenseTestOperator row_major_operator(
    int rows,
    int cols,
    const std::vector<double>& row_major) {
    require_test(
        static_cast<int>(row_major.size()) == rows * cols,
        "row-major test matrix size");
    std::vector<double> column_major(row_major.size());
    for (int row = 0; row < rows; ++row) {
        for (int col = 0; col < cols; ++col) {
            column_major[row + col * rows] = row_major[row * cols + col];
        }
    }
    return DenseTestOperator(rows, cols, std::move(column_major));
}

std::vector<double> apply_dense(
    const DenseTestOperator& matrix,
    const std::vector<double>& solution) {
    std::vector<double> result(matrix.rows());
    matrix.apply(solution.data(), 1, result.data());
    return result;
}

void require_solution(
    const amfls::LeastSquaresResult& result,
    const std::vector<double>& expected,
    double tolerance,
    const std::string& label) {
    require_test(
        result.status == amfls::SolverStatus::success,
        label + " must satisfy the original-candidate contract");
    require_test(
        result.solution.size() == expected.size(),
        label + " solution size");
    for (int index = 0; index < static_cast<int>(expected.size()); ++index) {
        require_near(
            result.solution[index],
            expected[index],
            tolerance,
            label + " coordinate " + std::to_string(index));
    }
}

template <class Function>
void require_invalid_argument(Function&& function, const std::string& label) {
    bool threw = false;
    try {
        function();
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    require_test(threw, label);
}

void require_identical_solution(
    const amfls::LeastSquaresResult& left,
    const amfls::LeastSquaresResult& right,
    const std::string& label) {
    require_test(
        left.status == right.status &&
            left.stop_reason == right.stop_reason &&
            left.basis_rank == right.basis_rank &&
            left.iterations == right.iterations &&
            left.solution == right.solution,
        label);
}

void check_ordinary_accounting(
    const amfls::LeastSquaresResult& result,
    bool tall,
    int sketch_size,
    int sketch_block_calls,
    long long random_values,
    const std::string& label) {
    const long long iterations = result.iterations;
    require_test(
        result.statistics.gaussian_random_block_requests ==
                sketch_block_calls &&
            result.statistics.gaussian_random_columns == sketch_size &&
            result.statistics.gaussian_random_values == random_values,
        label + " random accounting");
    require_test(
        result.statistics.iterative_a_columns == iterations &&
            result.statistics.iterative_at_columns == iterations &&
            result.statistics.iterative_a_block_calls == iterations &&
            result.statistics.iterative_at_block_calls == iterations,
        label + " terminal transpose product accounting");
    require_test(
        result.statistics.validation_a_columns ==
                static_cast<long long>(result.trace.size()) &&
            result.statistics.validation_at_columns ==
                static_cast<long long>(result.trace.size()),
        label + " validation accounting");
    if (tall) {
        require_test(
            result.statistics.sketch_a_columns == 0 &&
                result.statistics.sketch_at_columns == sketch_size &&
                result.statistics.sketch_a_block_calls == 0 &&
                result.statistics.sketch_at_block_calls == sketch_block_calls,
            label + " tall sketch accounting");
    } else {
        require_test(
            result.statistics.sketch_a_columns == sketch_size &&
                result.statistics.sketch_at_columns == 0 &&
                result.statistics.sketch_a_block_calls == sketch_block_calls &&
                result.statistics.sketch_at_block_calls == 0,
            label + " wide sketch accounting");
    }
    require_test(
        result.statistics.search_a_columns ==
                result.statistics.sketch_a_columns +
                    result.statistics.iterative_a_columns &&
            result.statistics.search_at_columns ==
                result.statistics.sketch_at_columns +
                    result.statistics.iterative_at_columns &&
            result.statistics.a_columns ==
                result.statistics.search_a_columns +
                    result.statistics.validation_a_columns &&
            result.statistics.at_columns ==
                result.statistics.search_at_columns +
                    result.statistics.validation_at_columns,
        label + " disjoint phase accounting");
    require_test(
        result.statistics.search_a_block_calls ==
                result.statistics.sketch_a_block_calls +
                    result.statistics.iterative_a_block_calls &&
            result.statistics.search_at_block_calls ==
                result.statistics.sketch_at_block_calls +
                    result.statistics.iterative_at_block_calls &&
            result.statistics.a_block_calls ==
                result.statistics.search_a_block_calls +
                    result.statistics.validation_a_block_calls &&
            result.statistics.at_block_calls ==
                result.statistics.search_at_block_calls +
                    result.statistics.validation_at_block_calls,
        label + " callback phase accounting");
    require_test(
        result.statistics.validation_a_columns ==
                result.statistics.base_validation_a_columns &&
            result.statistics.validation_at_columns ==
                result.statistics.base_validation_at_columns &&
            result.statistics.validation_a_block_calls ==
                result.statistics.base_validation_a_block_calls &&
            result.statistics.validation_at_block_calls ==
                result.statistics.base_validation_at_block_calls &&
            result.statistics.ridge_correction_a_columns == 0 &&
            result.statistics.ridge_correction_at_columns == 0 &&
            result.statistics.validation_seconds ==
                result.statistics.base_validation_seconds +
                    result.statistics.ridge_correction_validation_seconds,
        label + " base-validation partition");
    require_test(
        !result.trace.empty() &&
            result.trace.back().depth == result.iterations,
        label + " checkpoint trace accounting");
}

class ExplodingIterationOperator final : public amfls::MatrixOperator {
public:
    int rows() const override { return 2; }
    int cols() const override { return 1; }

    void apply(const double* x, int block_cols, double* y) const override {
        ++a_calls_;
        for (int col = 0; col < block_cols; ++col) {
            y[2 * col] = a_calls_ >= 2
                ? std::numeric_limits<double>::infinity()
                : x[col];
            y[2 * col + 1] = 0.0;
        }
    }

    void apply_transpose(
        const double* y,
        int block_cols,
        double* x) const override {
        ++at_calls_;
        for (int col = 0; col < block_cols; ++col) {
            x[col] = y[2 * col];
        }
    }

    int a_calls() const { return a_calls_; }
    int at_calls() const { return at_calls_; }

private:
    mutable int a_calls_ = 0;
    mutable int at_calls_ = 0;
};

}  // namespace

int main() {
    const std::uint64_t seed = 0x123456789abcdef0ULL;
    const std::uint64_t stream = 17;

    DenseTestOperator tall = row_major_operator(
        6,
        2,
        {1.0, 0.0,
         0.0, 1.0,
         1.0, 1.0,
         2.0, -1.0,
         0.5, 3.0,
         -1.0, 2.0});
    const std::vector<double> tall_expected{2.0, -1.0};
    const std::vector<double> tall_b = apply_dense(tall, tall_expected);
    amfls::LsrnOptions tall_options;
    tall_options.tolerance = 1e-10;
    tall_options.seed = seed;
    tall_options.stream = stream;
    tall_options.sketch_block_size = 2;
    const auto tall_result = amfls::solve_lsrn(
        tall, tall_b.data(), tall_options);
    require_solution(tall_result, tall_expected, 2e-10, "tall LSRN");
    require_test(tall_result.basis_rank == 2, "tall LSRN sketch rank");
    check_ordinary_accounting(
        tall_result, true, 4, 2, 24, "tall LSRN");

    require_test(
        !tall_result.trace.empty() &&
            tall_result.trace.front().depth == 1 &&
            tall_result.statistics.base_validation_a_columns ==
                static_cast<long long>(tall_result.trace.size()) &&
            tall_result.statistics.base_validation_at_columns ==
                static_cast<long long>(tall_result.trace.size()) &&
            tall_result.statistics.iterative_a_columns ==
                tall_result.iterations &&
            tall_result.statistics.iterative_at_columns ==
                tall_result.iterations,
        "LSRN must start at depth one and stop before the next A*");

    amfls::LsrnOptions capped_tall_options = tall_options;
    capped_tall_options.tolerance = 1e-15;
    capped_tall_options.maximum_iterations = 1;
    const auto capped_tall_result = amfls::solve_lsrn(
        tall, tall_b.data(), capped_tall_options);
    require_test(
        capped_tall_result.status == amfls::SolverStatus::work_limit &&
            capped_tall_result.stop_reason ==
                amfls::StopReason::maximum_depth &&
            capped_tall_result.iterations == 1 &&
            capped_tall_result.statistics.iterative_a_columns == 1 &&
            capped_tall_result.statistics.iterative_at_columns == 1,
        "maximum-depth LSRN must not prepare an unused transpose step");

    DenseTestOperator wide = row_major_operator(
        2,
        4,
        {1.0, 0.0, 1.0, 0.0,
         0.0, 1.0, 0.0, 1.0});
    const std::vector<double> wide_b{2.0, 4.0};
    const std::vector<double> wide_expected{1.0, 2.0, 1.0, 2.0};
    amfls::LsrnOptions wide_options;
    wide_options.tolerance = 1e-10;
    wide_options.seed = seed;
    wide_options.stream = stream + 1;
    wide_options.sketch_block_size = 3;
    const auto wide_result = amfls::solve_lsrn(
        wide, wide_b.data(), wide_options);
    require_solution(
        wide_result, wide_expected, 2e-10, "wide minimum-norm LSRN");
    require_test(wide_result.basis_rank == 2, "wide LSRN sketch rank");
    check_ordinary_accounting(
        wide_result, false, 4, 2, 16, "wide LSRN");

    // A = u v^T has rank one.  The right-hand side includes a component
    // orthogonal to u; the expected vector is the unique minimum-length LS
    // solution, not merely an arbitrary exact normal-equation solution.
    DenseTestOperator rank_deficient_tall = row_major_operator(
        4,
        3,
        {1.0, 2.0, 0.0,
         2.0, 4.0, 0.0,
         3.0, 6.0, 0.0,
         4.0, 8.0, 0.0});
    const std::vector<double> rank_deficient_tall_b{5.0, 5.0, 9.0, 12.0};
    amfls::LsrnOptions rank_deficient_tall_options;
    rank_deficient_tall_options.tolerance = 1e-10;
    rank_deficient_tall_options.seed = seed + 2;
    const auto rank_deficient_tall_result = amfls::solve_lsrn(
        rank_deficient_tall,
        rank_deficient_tall_b.data(),
        rank_deficient_tall_options);
    require_solution(
        rank_deficient_tall_result,
        {0.6, 1.2, 0.0},
        2e-10,
        "rank-deficient tall LSRN");
    require_test(
        rank_deficient_tall_result.basis_rank == 1,
        "rank-deficient tall numerical rank");

    DenseTestOperator rank_deficient_wide = row_major_operator(
        2,
        4,
        {1.0, -1.0, 2.0, 0.0,
         2.0, -2.0, 4.0, 0.0});
    const std::vector<double> rank_deficient_wide_b{3.0, 6.0};
    amfls::LsrnOptions rank_deficient_wide_options;
    rank_deficient_wide_options.tolerance = 1e-10;
    rank_deficient_wide_options.seed = seed + 3;
    const auto rank_deficient_wide_result = amfls::solve_lsrn(
        rank_deficient_wide,
        rank_deficient_wide_b.data(),
        rank_deficient_wide_options);
    require_solution(
        rank_deficient_wide_result,
        {0.5, -0.5, 1.0, 0.0},
        2e-10,
        "rank-deficient wide minimum-norm LSRN");
    require_test(
        rank_deficient_wide_result.basis_rank == 1,
        "rank-deficient wide numerical rank");

    DenseTestOperator ridge_tall = row_major_operator(
        3,
        2,
        {3.0, 0.0,
         0.0, 1.0,
         0.0, 0.0});
    const std::vector<double> ridge_tall_b{6.0, 2.0, 5.0};
    amfls::LsrnOptions ridge_tall_options;
    ridge_tall_options.regularization = 4.0;
    ridge_tall_options.tolerance = 1e-10;
    ridge_tall_options.seed = seed + 4;
    ridge_tall_options.sketch_block_size = 2;
    const auto ridge_tall_result = amfls::solve_lsrn(
        ridge_tall, ridge_tall_b.data(), ridge_tall_options);
    require_solution(
        ridge_tall_result,
        {18.0 / 13.0, 2.0 / 5.0},
        2e-10,
        "tall ridge LSRN");
    require_test(
        ridge_tall_result.stop_reason ==
            amfls::StopReason::relative_energy_error,
        "tall ridge LSRN shared success gate");
    require_test(
        ridge_tall_result.statistics.gaussian_random_block_requests == 2 &&
            ridge_tall_result.statistics.gaussian_random_columns == 4 &&
            ridge_tall_result.statistics.gaussian_random_values == 20 &&
            ridge_tall_result.statistics.sketch_at_columns == 4,
        "tall ridge augmented sketch accounting");

    amfls::LsrnOptions ridge_wide_options;
    ridge_wide_options.regularization = 2.0;
    ridge_wide_options.tolerance = 1e-10;
    ridge_wide_options.seed = seed + 5;
    ridge_wide_options.sketch_block_size = 2;
    const std::vector<double> ridge_wide_b{4.0, 8.0};
    const auto ridge_wide_result = amfls::solve_lsrn(
        wide, ridge_wide_b.data(), ridge_wide_options);
    require_solution(
        ridge_wide_result,
        {1.0, 2.0, 1.0, 2.0},
        2e-10,
        "wide ridge dual LSRN");
    require_test(
        ridge_wide_result.stop_reason ==
            amfls::StopReason::relative_energy_error,
        "wide ridge LSRN shared success gate");
    require_test(
        ridge_wide_result.statistics.gaussian_random_block_requests == 2 &&
            ridge_wide_result.statistics.gaussian_random_columns == 4 &&
            ridge_wide_result.statistics.gaussian_random_values == 24 &&
            ridge_wide_result.statistics.sketch_a_columns == 4,
        "wide ridge dual sketch accounting");

    // Counter-addressed Gaussian values make the materialized sketch exactly
    // independent of the callback block partition in both orientations.
    amfls::LsrnOptions tall_single_block = tall_options;
    tall_single_block.sketch_block_size = 0;
    const auto tall_single_block_result = amfls::solve_lsrn(
        tall, tall_b.data(), tall_single_block);
    require_identical_solution(
        tall_result,
        tall_single_block_result,
        "tall LSRN block-invariant solution");
    require_test(
        tall_single_block_result.statistics.gaussian_random_block_requests ==
                1 &&
            tall_result.statistics.gaussian_random_block_requests == 2 &&
            tall_single_block_result.statistics.gaussian_random_values ==
                tall_result.statistics.gaussian_random_values &&
            tall_single_block_result.statistics.sketch_at_columns ==
                tall_result.statistics.sketch_at_columns &&
            tall_single_block_result.statistics.sketch_at_block_calls == 1 &&
            tall_result.statistics.sketch_at_block_calls == 2,
        "tall LSRN block-invariant work and explicit callback counts");

    amfls::LsrnOptions wide_single_block = wide_options;
    wide_single_block.sketch_block_size = 0;
    const auto wide_single_block_result = amfls::solve_lsrn(
        wide, wide_b.data(), wide_single_block);
    require_identical_solution(
        wide_result,
        wide_single_block_result,
        "wide LSRN block-invariant solution");
    require_test(
        wide_single_block_result.statistics.gaussian_random_block_requests ==
                1 &&
            wide_result.statistics.gaussian_random_block_requests == 2 &&
            wide_single_block_result.statistics.gaussian_random_values ==
                wide_result.statistics.gaussian_random_values &&
            wide_single_block_result.statistics.sketch_a_columns ==
                wide_result.statistics.sketch_a_columns &&
            wide_single_block_result.statistics.sketch_a_block_calls == 1 &&
            wide_result.statistics.sketch_a_block_calls == 2,
        "wide LSRN block-invariant work and explicit callback counts");

    const auto repeat = amfls::solve_lsrn(
        tall, tall_b.data(), tall_options);
    require_identical_solution(
        tall_result, repeat, "LSRN fixed-seed determinism");

    // If validation encounters a nonfinite original-operator product, LSRN
    // must expose a finite committed candidate and a numerical failure.
    ExplodingIterationOperator exploding_operator;
    const std::vector<double> exploding_b{1.0, 0.0};
    amfls::LsrnOptions exploding_options;
    exploding_options.tolerance = 1e-12;
    exploding_options.seed = seed + 7;
    const auto exploding = amfls::solve_lsrn(
        exploding_operator, exploding_b.data(), exploding_options);
    require_test(
        exploding.status == amfls::SolverStatus::numerical_breakdown &&
            exploding.stop_reason == amfls::StopReason::numerical_breakdown &&
            exploding.iterations == 1 && exploding.trace.size() == 1 &&
            exploding.solution.size() == 1 &&
            std::isfinite(exploding.solution[0]),
        "LSRN nonfinite validation must return a finite committed candidate");
    require_test(
        exploding.statistics.sketch_at_columns == 2 &&
            exploding.statistics.iterative_a_columns == 1 &&
            exploding.statistics.iterative_at_columns == 1,
        "LSRN failed-product search accounting");
    require_test(
        exploding.statistics.validation_a_columns == 1 &&
            exploding.statistics.validation_at_columns == 1,
        "LSRN failed-product validation accounting");
    require_test(
        exploding.statistics.a_columns == 2 &&
            exploding.statistics.at_columns == 4,
        "LSRN failed-product total column accounting");
    require_test(
        exploding_operator.a_calls() == 2 &&
            exploding_operator.at_calls() == 3,
        "LSRN failed-product callback accounting");

    // The user-visible relative cutoff controls the effective rank.  The
    // right-hand side has no component in the tiny direction so both runs
    // remain valid original-problem solutions.
    DenseTestOperator cutoff_matrix = row_major_operator(
        3,
        2,
        {1.0, 0.0,
         0.0, 1e-8,
         0.0, 0.0});
    const std::vector<double> cutoff_b{1.0, 0.0, 0.0};
    amfls::LsrnOptions truncated_options;
    truncated_options.tolerance = 1e-10;
    truncated_options.relative_rank_tolerance = 1e-6;
    truncated_options.seed = seed + 6;
    const auto truncated = amfls::solve_lsrn(
        cutoff_matrix, cutoff_b.data(), truncated_options);
    require_solution(truncated, {1.0, 0.0}, 2e-6, "truncated-rank LSRN");
    require_test(truncated.basis_rank == 1, "explicit LSRN rank truncation");

    amfls::LsrnOptions untruncated_options = truncated_options;
    untruncated_options.relative_rank_tolerance = 0.0;
    untruncated_options.tolerance = 1e-14;
    const auto untruncated = amfls::solve_lsrn(
        cutoff_matrix, cutoff_b.data(), untruncated_options);
    require_solution(untruncated, {1.0, 0.0}, 2e-6, "untruncated LSRN");
    require_test(untruncated.basis_rank == 2, "zero LSRN rank cutoff");

    DenseTestOperator source_default_cutoff_matrix = row_major_operator(
        3,
        2,
        {1.0, 0.0,
         0.0, 1e-13,
         0.0, 0.0});
    amfls::LsrnOptions source_default_cutoff_options;
    source_default_cutoff_options.tolerance = 1e-10;
    source_default_cutoff_options.seed = seed + 8;
    const auto source_default_cutoff = amfls::solve_lsrn(
        source_default_cutoff_matrix,
        cutoff_b.data(),
        source_default_cutoff_options);
    require_solution(
        source_default_cutoff,
        {1.0, 0.0},
        2e-8,
        "source-default cutoff LSRN");
    require_test(
        source_default_cutoff.basis_rank == 1,
        "LSRN default must apply the strict 1e-12 relative rank cutoff");

    amfls::LsrnOptions empty_space_options = tall_options;
    empty_space_options.absolute_rank_tolerance = 1e100;
    const auto empty_space = amfls::solve_lsrn(
        tall, tall_b.data(), empty_space_options);
    require_test(
        empty_space.status == amfls::SolverStatus::work_limit &&
            empty_space.stop_reason ==
                amfls::StopReason::exhausted_search_space &&
            empty_space.basis_rank == 0 &&
            empty_space.trace.size() == 1 &&
            empty_space.trace.front().depth == 0 &&
            empty_space.statistics.base_validation_a_columns == 1 &&
            empty_space.statistics.base_validation_at_columns == 1,
        "an empty cutoff space must remain a non-success without a "
        "precision-limit claim");

    amfls::LsrnOptions invalid = tall_options;
    invalid.oversampling = 1.0;
    require_invalid_argument(
        [&]() { (void)amfls::solve_lsrn(tall, tall_b.data(), invalid); },
        "LSRN must reject non-oversampling");
    invalid = tall_options;
    invalid.relative_rank_tolerance = -2.0;
    require_invalid_argument(
        [&]() { (void)amfls::solve_lsrn(tall, tall_b.data(), invalid); },
        "LSRN must reject an invalid rank cutoff");
    invalid = tall_options;
    invalid.sketch_block_size = -1;
    require_invalid_argument(
        [&]() { (void)amfls::solve_lsrn(tall, tall_b.data(), invalid); },
        "LSRN must reject a negative sketch block size");

    return 0;
}
