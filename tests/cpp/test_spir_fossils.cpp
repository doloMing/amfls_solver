#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "amfls/spir_fossils.hpp"
#include "algorithms/mathematics/operators/row_major_dense_operator.hpp"
#include "algorithms/mathematics/operators/sparse_csr_operator.hpp"
#include "test_helpers.hpp"

namespace {

DenseTestOperator row_major_operator(
    int rows,
    int cols,
    const std::vector<double>& row_major) {
    require_test(
        static_cast<int>(row_major.size()) == rows * cols,
        "row-major SPIR/FOSSILS test matrix size");
    std::vector<double> column_major(row_major.size());
    for (int row = 0; row < rows; ++row) {
        for (int column = 0; column < cols; ++column) {
            column_major[row + column * rows] =
                row_major[row * cols + column];
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

void require_accounting(
    const amfls::LeastSquaresResult& result,
    int column_norm_a_columns,
    int column_norm_a_calls,
    int sketch_rows,
    int sketch_block_calls,
    const std::string& label) {
    const auto& statistics = result.statistics;
    require_test(
        statistics.sketch_a_columns == column_norm_a_columns &&
            statistics.sketch_a_block_calls == column_norm_a_calls &&
            statistics.sketch_at_columns == sketch_rows &&
            statistics.sketch_at_block_calls == sketch_block_calls,
        label + " column scaling and sketch accounting");
    require_test(
        statistics.search_a_columns ==
                statistics.sketch_a_columns +
                    statistics.iterative_a_columns &&
            statistics.search_at_columns ==
                statistics.sketch_at_columns +
                    statistics.iterative_at_columns &&
            statistics.search_a_block_calls ==
                statistics.sketch_a_block_calls +
                    statistics.iterative_a_block_calls &&
            statistics.search_at_block_calls ==
                statistics.sketch_at_block_calls +
                    statistics.iterative_at_block_calls,
        label + " disjoint search accounting");
    require_test(
        statistics.a_columns ==
                statistics.search_a_columns + statistics.validation_a_columns &&
            statistics.at_columns ==
                statistics.search_at_columns + statistics.validation_at_columns &&
            statistics.a_block_calls ==
                statistics.search_a_block_calls +
                    statistics.validation_a_block_calls &&
            statistics.at_block_calls ==
                statistics.search_at_block_calls +
                    statistics.validation_at_block_calls,
        label + " search/validation accounting");
    require_test(
        statistics.validation_a_columns ==
                statistics.base_validation_a_columns &&
            statistics.validation_at_columns ==
                statistics.base_validation_at_columns &&
            statistics.ridge_correction_a_columns == 0 &&
            statistics.ridge_correction_at_columns == 0,
        label + " ordinary validation partition");
    require_test(
        statistics.validation_a_columns == 1 &&
            statistics.validation_at_columns == 1 &&
            statistics.validation_a_block_calls == 1 &&
            statistics.validation_at_block_calls == 1,
        label + " performs one final original-problem validation");
    require_test(
        result.trace.size() == 1,
        label + " records only the final externally validated candidate");
}

void require_same_work_statistics(
    const amfls::LeastSquaresResult& first,
    const amfls::LeastSquaresResult& second,
    const std::string& label) {
    const auto& left = first.statistics;
    const auto& right = second.statistics;
    require_test(
        left.a_columns == right.a_columns &&
            left.at_columns == right.at_columns &&
            left.a_block_calls == right.a_block_calls &&
            left.at_block_calls == right.at_block_calls &&
            left.search_a_columns == right.search_a_columns &&
            left.search_at_columns == right.search_at_columns &&
            left.validation_a_columns == right.validation_a_columns &&
            left.validation_at_columns == right.validation_at_columns &&
            left.sketch_a_columns == right.sketch_a_columns &&
            left.sketch_at_columns == right.sketch_at_columns &&
            left.sketch_a_block_calls == right.sketch_a_block_calls &&
            left.sketch_at_block_calls == right.sketch_at_block_calls &&
            left.iterative_a_columns == right.iterative_a_columns &&
            left.iterative_at_columns == right.iterative_at_columns &&
            left.iterative_a_block_calls ==
                right.iterative_a_block_calls &&
            left.iterative_at_block_calls ==
                right.iterative_at_block_calls,
        label);
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
    const std::vector<double> tall_row_major{
        1.0, 0.0,
        0.0, 1.0,
        1.0, 1.0,
        2.0, -1.0,
        -1.0, 2.0,
        0.5, 3.0,
        3.0, 0.5,
        -2.0, -1.0};
    DenseTestOperator tall = row_major_operator(
        8,
        2,
        tall_row_major);
    const std::vector<double> right_hand_side{
        2.5, -0.5, 1.0, 6.0, -4.0, -0.5, 8.0, -3.0};
    // The explicit normal equations give the following solution for this
    // inconsistent problem.
    const std::vector<double> expected{
        2.5066063348416288, -0.7543891402714932};

    amfls::SpirOptions spir_options;
    spir_options.tolerance = 1e-10;
    spir_options.sketch_rows = 16;
    spir_options.embedding_nonzeros = 6;
    spir_options.maximum_inner_iterations = 12;
    spir_options.sketch_block_size = 5;
    spir_options.seed = 711;
    spir_options.stream = 3;
    const amfls::LeastSquaresResult spir = amfls::solve_spir(
        tall, right_hand_side.data(), spir_options);
    require_solution(spir, expected, 2e-10, "SPIR");
    require_test(
        spir.auxiliary_width == 16 && spir.basis_rank == 2 &&
            spir.iterations > 0 && spir.depth >= 1 && spir.depth <= 2,
        "SPIR result parameter mapping");
    require_accounting(spir, 2, 1, 16, 4, "SPIR");

    amfls::math::RowMajorDenseOperator stored_dense_tall(
        8, 2, tall_row_major.data());
    const amfls::LeastSquaresResult stored_dense_spir = amfls::solve_spir(
        stored_dense_tall, right_hand_side.data(), spir_options);
    require_solution(
        stored_dense_spir,
        spir.solution,
        2e-12,
        "stored-dense SPIR equivalence");
    require_accounting(
        stored_dense_spir, 0, 0, 16, 4, "stored-dense SPIR");

    amfls::FossilsOptions fossils_options;
    fossils_options.tolerance = 1e-10;
    fossils_options.sketch_rows = 24;
    fossils_options.embedding_nonzeros = 8;
    fossils_options.maximum_inner_iterations = 100;
    fossils_options.sketch_block_size = 7;
    fossils_options.seed = 913;
    fossils_options.stream = 5;
    const amfls::LeastSquaresResult fossils = amfls::solve_fossils(
        tall, right_hand_side.data(), fossils_options);
    require_solution(fossils, expected, 3e-9, "FOSSILS");
    require_test(
        fossils.auxiliary_width == 24 && fossils.basis_rank == 2 &&
            fossils.iterations > 0 && fossils.depth >= 1 &&
            fossils.depth <= 2,
        "FOSSILS result parameter mapping");
    require_accounting(fossils, 2, 1, 24, 4, "FOSSILS");

    DenseTestOperator difficult = row_major_operator(
        6,
        2,
        {1.0, 1.0,
         0.0, 1e-8,
         0.0, 0.0,
         0.0, 0.0,
         0.0, 0.0,
         0.0, 0.0});
    const std::vector<double> difficult_rhs{
        2.0, 1e-8, 1.0, 0.0, 0.0, 0.0};
    amfls::FossilsOptions difficult_fossils_options;
    difficult_fossils_options.tolerance = 1e-16;
    difficult_fossils_options.sketch_rows = 48;
    difficult_fossils_options.maximum_inner_iterations = 100;
    difficult_fossils_options.seed = 1901;
    const auto difficult_fossils = amfls::solve_fossils(
        difficult, difficult_rhs.data(), difficult_fossils_options);
    amfls::FossilsOptions loose_fossils_options =
        difficult_fossils_options;
    loose_fossils_options.tolerance = 1e-8;
    const auto loose_fossils = amfls::solve_fossils(
        difficult, difficult_rhs.data(), loose_fossils_options);
    require_test(
        difficult_fossils.status !=
                amfls::SolverStatus::numerical_breakdown &&
            difficult_fossils.depth == 2 &&
            difficult_fossils.iterations > 0 &&
            difficult_fossils.iterations <
                2 * difficult_fossils_options.maximum_inner_iterations &&
            loose_fossils.status !=
                amfls::SolverStatus::numerical_breakdown &&
            loose_fossils.depth == 1 &&
            difficult_fossils.iterations ==
                loose_fossils.iterations + 15,
        "FOSSILS uses the requested posterior tolerance and completes the "
        "next posterior interval after a four-times-near pass");
    require_accounting(
        difficult_fossils, 2, 1, 48, 2, "adaptive FOSSILS");
    require_accounting(
        loose_fossils, 2, 1, 48, 2, "loose adaptive FOSSILS");

    const std::vector<std::int64_t> csr_row_offsets{
        0, 1, 2, 4, 6, 8, 10, 12, 14};
    const std::vector<std::int64_t> csr_column_indices{
        0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1};
    const std::vector<double> csr_values{
        1.0, 1.0, 1.0, 1.0, 2.0, -1.0, -1.0,
        2.0, 0.5, 3.0, 3.0, 0.5, -2.0, -1.0};
    amfls::math::SparseCsrOperator sparse_tall(
        8,
        2,
        csr_row_offsets.data(),
        csr_column_indices.data(),
        csr_values.data(),
        static_cast<std::int64_t>(csr_values.size()));

    const amfls::LeastSquaresResult native_spir = amfls::solve_spir(
        sparse_tall, right_hand_side.data(), spir_options);
    const amfls::LeastSquaresResult repeated_native_spir = amfls::solve_spir(
        sparse_tall, right_hand_side.data(), spir_options);
    require_solution(
        native_spir,
        spir.solution,
        2e-13,
        "CSR-native SPIR equivalence");
    require_test(
        repeated_native_spir.solution == native_spir.solution &&
            repeated_native_spir.status == native_spir.status &&
            repeated_native_spir.stop_reason == native_spir.stop_reason &&
            repeated_native_spir.iterations == native_spir.iterations &&
            repeated_native_spir.basis_rank == native_spir.basis_rank,
        "CSR-native SPIR is deterministic");
    require_same_work_statistics(
        native_spir,
        repeated_native_spir,
        "repeated CSR-native SPIR preserves work statistics");
    require_accounting(native_spir, 0, 0, 0, 0, "CSR-native SPIR");

    const amfls::LeastSquaresResult native_fossils = amfls::solve_fossils(
        sparse_tall, right_hand_side.data(), fossils_options);
    const amfls::LeastSquaresResult repeated_native_fossils =
        amfls::solve_fossils(
            sparse_tall, right_hand_side.data(), fossils_options);
    require_solution(
        native_fossils,
        fossils.solution,
        2e-12,
        "CSR-native FOSSILS equivalence");
    require_test(
        repeated_native_fossils.solution == native_fossils.solution &&
            repeated_native_fossils.status == native_fossils.status &&
            repeated_native_fossils.stop_reason ==
                native_fossils.stop_reason &&
            repeated_native_fossils.iterations == native_fossils.iterations &&
            repeated_native_fossils.basis_rank == native_fossils.basis_rank,
        "CSR-native FOSSILS is deterministic");
    require_same_work_statistics(
        native_fossils,
        repeated_native_fossils,
        "repeated CSR-native FOSSILS preserves work statistics");
    require_accounting(
        native_fossils, 0, 0, 0, 0, "CSR-native FOSSILS");

    DenseTestOperator wide = row_major_operator(
        2, 3, {1.0, 0.0, 0.0, 0.0, 1.0, 0.0});
    const std::vector<double> wide_rhs{1.0, 2.0};
    require_invalid_argument(
        [&]() {
            (void)amfls::solve_spir(
                wide, wide_rhs.data(), amfls::SpirOptions{});
        },
        "SPIR wide shape is outside the declared baseline scope");

    amfls::SpirOptions ridge_spir;
    ridge_spir.regularization = 1.0;
    require_invalid_argument(
        [&]() {
            (void)amfls::solve_spir(
                tall, right_hand_side.data(), ridge_spir);
        },
        "SPIR rejects ridge regularization");

    amfls::FossilsOptions ridge_fossils;
    ridge_fossils.regularization = 1.0;
    require_invalid_argument(
        [&]() {
            (void)amfls::solve_fossils(
                tall, right_hand_side.data(), ridge_fossils);
        },
        "FOSSILS rejects ridge regularization");

    amfls::FossilsOptions invalid_distortion;
    invalid_distortion.sketch_rows = tall.cols();
    require_invalid_argument(
        [&]() {
            (void)amfls::solve_fossils(
                tall, right_hand_side.data(), invalid_distortion);
        },
        "FOSSILS rejects a unit distortion estimate");

    amfls::SpirOptions invalid_sketch;
    invalid_sketch.sketch_rows = -1;
    require_invalid_argument(
        [&]() {
            (void)amfls::solve_spir(
                tall, right_hand_side.data(), invalid_sketch);
        },
        "SPIR rejects a negative sketch size");

    std::cout << "test_spir_fossils passed\n";
    return 0;
}
