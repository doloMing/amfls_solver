#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "amfls/randomized_block_cg.hpp"
#include "test_helpers.hpp"

namespace {

DenseTestOperator row_major_operator(
    int rows,
    int cols,
    const std::vector<double>& row_major) {
    require_test(
        static_cast<int>(row_major.size()) == rows * cols,
        "row-major randomized block CG test matrix size");
    std::vector<double> column_major(row_major.size());
    for (int row = 0; row < rows; ++row) {
        for (int col = 0; col < cols; ++col) {
            column_major[row + col * rows] = row_major[row * cols + col];
        }
    }
    return DenseTestOperator(rows, cols, std::move(column_major));
}

void require_solution(
    const amfls::LeastSquaresResult& result,
    const std::vector<double>& expected,
    double tolerance,
    const std::string& label) {
    require_test(
        result.status == amfls::SolverStatus::success,
        label + " must pass the original-problem accuracy contract");
    require_test(result.solution.size() == expected.size(), label + " size");
    for (int index = 0; index < static_cast<int>(expected.size()); ++index) {
        require_near(
            result.solution[static_cast<std::size_t>(index)],
            expected[static_cast<std::size_t>(index)],
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

}  // namespace

int main() {
    const std::uint64_t seed = 913;

    DenseTestOperator tall = row_major_operator(
        3,
        2,
        {3.0, 0.0,
         0.0, 1.0,
         0.0, 0.0});
    const std::vector<double> tall_b{6.0, 2.0, 5.0};
    amfls::RandomizedBlockCgOptions tall_options;
    tall_options.regularization = 4.0;
    tall_options.tolerance = 1e-10;
    tall_options.random_block_size = 1;
    tall_options.maximum_depth = 2;
    tall_options.seed = seed;
    const amfls::LeastSquaresResult tall_result =
        amfls::solve_randomized_block_cg(
            tall, tall_b.data(), tall_options);
    require_solution(
        tall_result,
        {18.0 / 13.0, 2.0 / 5.0},
        2e-10,
        "tall ridge randomized block CG");
    require_test(
        tall_result.stop_reason ==
                amfls::StopReason::relative_energy_error &&
            tall_result.iterations == 1 && tall_result.depth == 1 &&
            tall_result.auxiliary_width == 1 &&
            tall_result.basis_rank == 2,
        "tall parameter-to-result mapping");
    require_test(
        tall_result.statistics.gaussian_random_block_requests == 1 &&
            tall_result.statistics.gaussian_random_columns == 1 &&
            tall_result.statistics.gaussian_random_values == 2 &&
            tall_result.statistics.iterative_a_columns == 2 &&
            tall_result.statistics.iterative_a_block_calls == 1 &&
            tall_result.statistics.iterative_at_columns == 1 &&
            tall_result.statistics.iterative_at_block_calls == 1,
        "tall block-CG search and random accounting");
    require_test(
        tall_result.statistics.search_a_columns ==
                tall_result.statistics.iterative_a_columns &&
            tall_result.statistics.search_at_columns ==
                tall_result.statistics.iterative_at_columns &&
            tall_result.statistics.sketch_a_columns == 0 &&
            tall_result.statistics.sketch_at_columns == 0,
        "random augmentation generates vectors but performs no sketch product");
    DenseTestOperator wide = row_major_operator(
        2,
        4,
        {1.0, 0.0, 0.0, 0.0,
         0.0, 1.0, 0.0, 0.0});
    const std::vector<double> wide_b{3.0, 4.0};
    amfls::RandomizedBlockCgOptions wide_options;
    wide_options.regularization = 1.0;
    wide_options.tolerance = 1e-10;
    wide_options.random_block_size = 3;
    wide_options.maximum_depth = 1;
    wide_options.seed = seed + 1;
    const amfls::LeastSquaresResult wide_result =
        amfls::solve_randomized_block_cg(
            wide, wide_b.data(), wide_options);
    require_solution(
        wide_result,
        {1.5, 2.0, 0.0, 0.0},
        3e-10,
        "wide ridge randomized block CG");
    require_test(
        wide_result.basis_rank == 4 && wide_result.depth == 1 &&
            wide_result.statistics.iterative_a_columns == 4 &&
            wide_result.statistics.iterative_at_columns == 1 &&
            wide_result.statistics.gaussian_random_columns == 3 &&
            wide_result.statistics.gaussian_random_values == 12,
        "wide normal-system domain and block accounting");

    DenseTestOperator capped = row_major_operator(
        4,
        4,
        {100.0, 0.0, 0.0, 0.0,
         0.0, 10.0, 0.0, 0.0,
         0.0, 0.0, 2.0, 0.0,
         0.0, 0.0, 0.0, 1.0});
    const std::vector<double> capped_b{1.0, -2.0, 3.0, -4.0};
    amfls::RandomizedBlockCgOptions capped_options;
    capped_options.regularization = 0.5;
    capped_options.tolerance = 1e-14;
    capped_options.random_block_size = 1;
    capped_options.maximum_depth = 1;
    capped_options.seed = seed + 2;
    capped_options.stream = 7;
    const amfls::LeastSquaresResult capped_result =
        amfls::solve_randomized_block_cg(
            capped, capped_b.data(), capped_options);
    require_test(
        capped_result.status == amfls::SolverStatus::work_limit &&
            capped_result.stop_reason == amfls::StopReason::maximum_depth &&
            capped_result.depth == 1 && capped_result.basis_rank == 2 &&
            capped_result.trace.size() == 2 &&
            capped_result.statistics.base_validation_a_columns == 2 &&
            capped_result.statistics.base_validation_at_columns == 2 &&
            capped_result.statistics.base_validation_a_block_calls == 2 &&
            capped_result.statistics.base_validation_at_block_calls == 2,
        "fixed block-Krylov depth is a work limit, not a success claim");
    const amfls::LeastSquaresResult capped_repeat =
        amfls::solve_randomized_block_cg(
            capped, capped_b.data(), capped_options);
    require_test(
        capped_repeat.solution == capped_result.solution &&
            capped_repeat.status == capped_result.status &&
            capped_repeat.stop_reason == capped_result.stop_reason,
        "fixed seed and stream reproduce the capped block-CG candidate");

    amfls::RandomizedBlockCgOptions extended_options = capped_options;
    extended_options.maximum_depth = 4;
    extended_options.tolerance = 1e-10;
    const amfls::LeastSquaresResult extended_result =
        amfls::solve_randomized_block_cg(
            capped, capped_b.data(), extended_options);
    require_solution(
        extended_result,
        {100.0 / 10000.5,
         -20.0 / 100.5,
         6.0 / 4.5,
         -4.0 / 1.5},
        2e-9,
        "two-depth ridge randomized block CG");
    require_test(
        extended_result.depth == 2 && extended_result.basis_rank == 4 &&
            extended_result.statistics.iterative_a_columns == 4 &&
            extended_result.statistics.iterative_a_block_calls == 2 &&
            extended_result.statistics.iterative_at_columns == 3 &&
            extended_result.statistics.iterative_at_block_calls == 2,
        "normal-system extension uses one A/A* block pair per added depth");

    std::vector<double> scheduled_values(12 * 12, 0.0);
    const std::vector<double> scheduled_diagonal{
        1000.0, 300.0, 90.0, 27.0, 9.0, 4.0,
        2.0, 1.0, 0.5, 0.25, 0.125, 0.0625};
    for (int index = 0; index < 12; ++index) {
        scheduled_values[static_cast<std::size_t>(index * 12 + index)] =
            scheduled_diagonal[static_cast<std::size_t>(index)];
    }
    DenseTestOperator scheduled = row_major_operator(
        12, 12, scheduled_values);
    const std::vector<double> scheduled_rhs{
        1.0, -2.0, 3.0, -4.0, 5.0, -6.0,
        7.0, -8.0, 9.0, -10.0, 11.0, -12.0};
    amfls::RandomizedBlockCgOptions scheduled_options;
    scheduled_options.regularization = 0.5;
    scheduled_options.tolerance = 1e-30;
    scheduled_options.random_block_size = 1;
    scheduled_options.maximum_depth = 5;
    scheduled_options.seed = seed + 3;
    const amfls::LeastSquaresResult scheduled_result =
        amfls::solve_randomized_block_cg(
            scheduled, scheduled_rhs.data(), scheduled_options);
    require_test(
        scheduled_result.status == amfls::SolverStatus::work_limit &&
            scheduled_result.stop_reason ==
                amfls::StopReason::maximum_depth &&
            scheduled_result.iterations == 5 &&
            scheduled_result.trace.size() == 5 &&
            scheduled_result.statistics.base_validation_a_columns == 5 &&
            scheduled_result.statistics.base_validation_at_columns == 5 &&
            scheduled_result.statistics.base_validation_a_block_calls == 5 &&
            scheduled_result.statistics.base_validation_at_block_calls == 5,
        "off-cadence block-CG cap validates the terminal candidate once");
    const std::vector<int> scheduled_depths{0, 1, 2, 4, 5};
    for (int index = 0; index < static_cast<int>(scheduled_depths.size());
         ++index) {
        require_test(
            scheduled_result.trace[static_cast<std::size_t>(index)].depth ==
                scheduled_depths[static_cast<std::size_t>(index)],
            "block-CG shared checkpoint cadence");
    }

    std::vector<double> exhausted_values(8 * 8, 0.0);
    const std::vector<double> exhausted_diagonal{
        3.0, 3.0, 3.0, 2.0, 2.0, 2.0, 1.0, 1.0};
    for (int index = 0; index < 8; ++index) {
        exhausted_values[static_cast<std::size_t>(index * 8 + index)] =
            exhausted_diagonal[static_cast<std::size_t>(index)];
    }
    DenseTestOperator exhausted = row_major_operator(
        8, 8, exhausted_values);
    const std::vector<double> exhausted_rhs{
        1.0, -2.0, 3.0, -4.0, 5.0, -6.0, 7.0, -8.0};
    amfls::RandomizedBlockCgOptions exhausted_options = scheduled_options;
    exhausted_options.maximum_depth = 8;
    exhausted_options.seed = seed + 4;
    const amfls::LeastSquaresResult exhausted_result =
        amfls::solve_randomized_block_cg(
            exhausted, exhausted_rhs.data(), exhausted_options);
    require_test(
        exhausted_result.status == amfls::SolverStatus::precision_limit &&
            exhausted_result.stop_reason ==
                amfls::StopReason::precision_limit &&
            exhausted_result.iterations == 3 &&
            exhausted_result.trace.size() == 4 &&
            exhausted_result.trace.back().depth == 3 &&
            exhausted_result.statistics.base_validation_a_columns == 4 &&
            exhausted_result.statistics.base_validation_at_columns == 4,
        "block-CG numerical deflation validates the off-cadence terminal candidate");

    amfls::RandomizedBlockCgOptions invalid = tall_options;
    invalid.regularization = 0.0;
    require_invalid_argument(
        [&]() {
            (void)amfls::solve_randomized_block_cg(
                tall, tall_b.data(), invalid);
        },
        "unregularized normal equations are explicitly outside this baseline");
    invalid = tall_options;
    invalid.random_block_size = 0;
    require_invalid_argument(
        [&]() {
            (void)amfls::solve_randomized_block_cg(
                tall, tall_b.data(), invalid);
        },
        "random augmentation width must be positive");
    invalid = tall_options;
    invalid.maximum_depth = -1;
    require_invalid_argument(
        [&]() {
            (void)amfls::solve_randomized_block_cg(
                tall, tall_b.data(), invalid);
        },
        "negative block-Krylov depth is rejected");

    std::cout << "test_randomized_block_cg passed\n";
    return 0;
}
