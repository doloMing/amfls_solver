#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "amfls/sparse_embedding_lsqr.hpp"
#include "algorithms/mathematics/operators/sparse_csr_operator.hpp"
#include "test_helpers.hpp"

namespace {

DenseTestOperator row_major_operator(
    int rows,
    int cols,
    const std::vector<double>& row_major) {
    require_test(
        static_cast<int>(row_major.size()) == rows * cols,
        "row-major sparse-embedding LSQR test matrix size");
    std::vector<double> column_major(row_major.size());
    for (int row = 0; row < rows; ++row) {
        for (int col = 0; col < cols; ++col) {
            column_major[row + col * rows] = row_major[row * cols + col];
        }
    }
    return DenseTestOperator(rows, cols, std::move(column_major));
}

class OneShotNonfiniteApplyOperator final : public amfls::MatrixOperator {
public:
    OneShotNonfiniteApplyOperator(
        const amfls::MatrixOperator& delegate,
        int failure_call)
        : delegate_(delegate), failure_call_(failure_call) {}

    int rows() const override { return delegate_.rows(); }
    int cols() const override { return delegate_.cols(); }

    void apply(const double* input, int block_cols, double* output) const override {
        ++apply_calls_;
        if (apply_calls_ == failure_call_) {
            for (int index = 0; index < rows() * block_cols; ++index) {
                output[index] = std::numeric_limits<double>::quiet_NaN();
            }
            return;
        }
        delegate_.apply(input, block_cols, output);
    }

    void apply_transpose(
        const double* input,
        int block_cols,
        double* output) const override {
        delegate_.apply_transpose(input, block_cols, output);
    }

private:
    const amfls::MatrixOperator& delegate_;
    int failure_call_ = 0;
    mutable int apply_calls_ = 0;
};

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

void require_accounting(
    const amfls::LeastSquaresResult& result,
    int expected_sketch_columns,
    int expected_sketch_block_calls,
    const std::string& label) {
    const long long iterations = result.iterations;
    require_test(
        result.statistics.sketch_a_columns == 0 &&
            result.statistics.sketch_a_block_calls == 0 &&
            result.statistics.sketch_at_columns == expected_sketch_columns &&
            result.statistics.sketch_at_block_calls ==
                expected_sketch_block_calls,
        label + " sketch uses only block A* callbacks");
    require_test(
        result.statistics.iterative_a_columns == iterations &&
            result.statistics.iterative_a_block_calls == iterations &&
            result.statistics.iterative_at_columns == iterations &&
            result.statistics.iterative_at_block_calls == iterations,
        label + " right-preconditioned LSQR callback accounting");
    require_test(
        result.statistics.validation_a_columns ==
                static_cast<long long>(result.trace.size()) &&
            result.statistics.validation_at_columns ==
                static_cast<long long>(result.trace.size()) &&
            result.statistics.validation_a_block_calls ==
                static_cast<long long>(result.trace.size()) &&
            result.statistics.validation_at_block_calls ==
                static_cast<long long>(result.trace.size()),
        label + " one fresh original validation per checkpoint");
    require_test(
        result.statistics.search_a_columns ==
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
        label + " disjoint search and validation column ledgers");
    require_test(
        result.statistics.search_a_block_calls ==
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
        label + " disjoint callback ledgers");
    require_test(
        result.statistics.gaussian_random_block_requests == 0 &&
            result.statistics.gaussian_random_columns == 0 &&
            result.statistics.gaussian_random_values == 0,
        label + " sparse signs are not charged as Gaussian work");
    require_test(
        !result.trace.empty() && result.trace.front().depth == 0 &&
            result.trace.back().depth == result.iterations,
        label + " terminal checkpoint");
    for (std::size_t index = 1; index < result.trace.size(); ++index) {
        require_test(
            result.trace[index - 1].depth < result.trace[index].depth,
            label + " checkpoint depths are strictly increasing");
    }
    require_test(
        result.statistics.base_validation_a_columns ==
                static_cast<long long>(result.trace.size()) &&
            result.statistics.base_validation_at_columns ==
                static_cast<long long>(result.trace.size()) &&
            result.statistics.base_validation_a_block_calls ==
                static_cast<long long>(result.trace.size()) &&
            result.statistics.base_validation_at_block_calls ==
                static_cast<long long>(result.trace.size()),
        label + " one base validation per checkpoint");
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

}  // namespace

int main() {
    DenseTestOperator tall = row_major_operator(
        4,
        2,
        {1.0, 0.0,
         0.0, 1.0,
         1.0, 1.0,
         1.0, -1.0});
    const std::vector<double> right_hand_side{2.0, -1.0, 1.0, 4.0};

    amfls::SparseEmbeddingLsqrOptions options;
    options.tolerance = 1e-10;
    options.embedding_distortion = 0.5;
    options.embedding_failure_probability = 0.5;
    options.sketch_block_size = 7;
    options.seed = 0x123456789abcdef0ULL;
    options.stream = 17;
    const amfls::LeastSquaresResult result =
        amfls::solve_sparse_embedding_lsqr(
            tall, right_hand_side.data(), options);
    require_solution(
        result,
        {7.0 / 3.0, -4.0 / 3.0},
        3e-10,
        "theorem-sized sparse-embedding LSQR");
    require_test(
        result.auxiliary_width == 48 && result.basis_rank == 2 &&
            result.iterations >= 1 && result.iterations <= 2,
        "theoretical sketch formula and LSQR iteration mapping");
    require_accounting(result, 48, 7, "theorem-sized solve");
    amfls::SparseEmbeddingLsqrOptions single_block = options;
    single_block.sketch_block_size = 0;
    const amfls::LeastSquaresResult repeated =
        amfls::solve_sparse_embedding_lsqr(
            tall, right_hand_side.data(), single_block);
    require_solution(
        repeated,
        {7.0 / 3.0, -4.0 / 3.0},
        3e-10,
        "single-block sparse-embedding LSQR");
    require_test(
        repeated.solution == result.solution &&
            repeated.iterations == result.iterations &&
            repeated.basis_rank == result.basis_rank &&
            repeated.statistics.sketch_at_block_calls == 1 &&
            result.statistics.sketch_at_block_calls == 7,
        "fixed seed is independent of the sketch callback partition");
    require_accounting(repeated, 48, 1, "single-block solve");

    amfls::SparseEmbeddingLsqrOptions practical = options;
    practical.sketch_rows = 24;
    practical.embedding_nonzeros = 4;
    practical.sketch_block_size = 5;
    practical.seed += 1;
    const amfls::LeastSquaresResult practical_result =
        amfls::solve_sparse_embedding_lsqr(
            tall, right_hand_side.data(), practical);
    require_solution(
        practical_result,
        {7.0 / 3.0, -4.0 / 3.0},
        3e-10,
        "explicit-row sparse-embedding LSQR");
    require_test(
        practical_result.auxiliary_width == 24,
        "explicit sketch_rows overrides the theorem-sized formula");
    require_accounting(
        practical_result, 24, 5, "explicit-row solve");

    const std::vector<std::int64_t> csr_row_offsets{0, 1, 2, 4, 6};
    const std::vector<std::int64_t> csr_column_indices{0, 1, 0, 1, 0, 1};
    const std::vector<double> csr_values{1.0, 1.0, 1.0, 1.0, 1.0, -1.0};
    amfls::math::SparseCsrOperator sparse_tall(
        4,
        2,
        csr_row_offsets.data(),
        csr_column_indices.data(),
        csr_values.data(),
        static_cast<std::int64_t>(csr_values.size()));
    const amfls::LeastSquaresResult native_result =
        amfls::solve_sparse_embedding_lsqr(
            sparse_tall, right_hand_side.data(), practical);
    const amfls::LeastSquaresResult native_repeated =
        amfls::solve_sparse_embedding_lsqr(
            sparse_tall, right_hand_side.data(), practical);
    require_solution(
        native_result,
        practical_result.solution,
        2e-13,
        "CSR-native sparse-embedding LSQR equivalence");
    require_test(
        native_repeated.solution == native_result.solution &&
            native_repeated.status == native_result.status &&
            native_repeated.stop_reason == native_result.stop_reason &&
            native_repeated.iterations == native_result.iterations &&
            native_repeated.basis_rank == native_result.basis_rank,
        "CSR-native sparse-embedding LSQR is deterministic");
    require_same_work_statistics(
        native_result,
        native_repeated,
        "repeated CSR-native solve preserves work statistics");
    require_accounting(native_result, 0, 0, "CSR-native solve");
    require_test(
        practical_result.statistics.sketch_at_columns == 24 &&
            practical_result.statistics.sketch_at_block_calls == 5 &&
            native_result.statistics.sketch_at_columns == 0 &&
            native_result.statistics.sketch_at_block_calls == 0,
        "CSR-native sketch construction records no nonexistent operator callbacks");

    constexpr int scheduled_rows = 20;
    constexpr int scheduled_cols = 10;
    std::vector<double> scheduled_values(
        scheduled_rows * scheduled_cols);
    for (int row = 0; row < scheduled_rows; ++row) {
        for (int col = 0; col < scheduled_cols; ++col) {
            const int centered = ((row + 3) * (col + 5) + 2 * row + col) % 19;
            double value = static_cast<double>(centered - 9) / 11.0;
            if (row == col) {
                value += 3.0;
            }
            scheduled_values[static_cast<std::size_t>(
                row * scheduled_cols + col)] = value;
        }
    }
    DenseTestOperator scheduled = row_major_operator(
        scheduled_rows, scheduled_cols, scheduled_values);
    std::vector<double> scheduled_rhs(scheduled_rows);
    for (int row = 0; row < scheduled_rows; ++row) {
        scheduled_rhs[static_cast<std::size_t>(row)] =
            static_cast<double>((7 * row) % 13 - 6);
    }
    amfls::SparseEmbeddingLsqrOptions scheduled_options = options;
    scheduled_options.tolerance = 1e-30;
    scheduled_options.sketch_rows = 20;
    scheduled_options.sketch_block_size = scheduled_cols;
    scheduled_options.maximum_iterations = 7;
    scheduled_options.seed += 5;
    const amfls::LeastSquaresResult scheduled_result =
        amfls::solve_sparse_embedding_lsqr(
            scheduled, scheduled_rhs.data(), scheduled_options);
    const std::vector<int> expected_checkpoint_depths{
        0, 1, 2, 3, 4, 6, 7};
    require_test(
        scheduled_result.status == amfls::SolverStatus::work_limit &&
            scheduled_result.stop_reason ==
                amfls::StopReason::maximum_depth &&
            scheduled_result.iterations == 7 &&
            scheduled_result.trace.size() ==
                expected_checkpoint_depths.size(),
        "sparse-embedding LSQR off-cadence iteration cap: status=" +
            std::to_string(static_cast<int>(scheduled_result.status)) +
            " stop=" +
            std::to_string(static_cast<int>(scheduled_result.stop_reason)) +
            " iterations=" + std::to_string(scheduled_result.iterations) +
            " trace=" + std::to_string(scheduled_result.trace.size()));
    for (int index = 0;
         index < static_cast<int>(expected_checkpoint_depths.size());
         ++index) {
        require_test(
            scheduled_result.trace[static_cast<std::size_t>(index)].depth ==
                expected_checkpoint_depths[static_cast<std::size_t>(index)],
            "sparse-embedding LSQR shared checkpoint cadence");
    }
    require_accounting(
        scheduled_result,
        20,
        2,
        "scheduled sparse-embedding solve");

    OneShotNonfiniteApplyOperator numerical_failure(scheduled, 11);
    const amfls::LeastSquaresResult numerical_result =
        amfls::solve_sparse_embedding_lsqr(
            numerical_failure, scheduled_rhs.data(), scheduled_options);
    const std::vector<int> expected_numerical_depths{0, 1, 2, 3, 4, 5};
    require_test(
        numerical_result.status == amfls::SolverStatus::numerical_breakdown &&
            numerical_result.stop_reason ==
                amfls::StopReason::numerical_breakdown &&
            numerical_result.iterations == 5 &&
            numerical_result.trace.size() ==
                expected_numerical_depths.size() &&
            numerical_result.statistics.base_validation_a_columns == 6 &&
            numerical_result.statistics.base_validation_at_columns == 6,
        "sparse-embedding numerical failure validates the skipped incumbent");
    for (int index = 0;
         index < static_cast<int>(expected_numerical_depths.size());
         ++index) {
        require_test(
            numerical_result.trace[static_cast<std::size_t>(index)].depth ==
                expected_numerical_depths[static_cast<std::size_t>(index)],
            "sparse-embedding numerical terminal checkpoint");
    }
    for (double value : numerical_result.solution) {
        require_test(
            std::isfinite(value),
            "sparse-embedding numerical terminal candidate is finite");
    }

    DenseTestOperator rank_deficient = row_major_operator(
        4,
        2,
        {1.0, 2.0,
         2.0, 4.0,
         3.0, 6.0,
         4.0, 8.0});
    const std::vector<double> rank_deficient_rhs{1.0, 2.0, 3.0, 4.0};
    amfls::SparseEmbeddingLsqrOptions rank_options = options;
    rank_options.sketch_rows = 32;
    rank_options.sketch_block_size = 8;
    const amfls::LeastSquaresResult rank_result =
        amfls::solve_sparse_embedding_lsqr(
            rank_deficient,
            rank_deficient_rhs.data(),
            rank_options);
    require_test(
        rank_result.status == amfls::SolverStatus::numerical_breakdown &&
            rank_result.stop_reason ==
                amfls::StopReason::numerical_breakdown &&
            rank_result.basis_rank == 1 && rank_result.iterations == 0 &&
            rank_result.statistics.sketch_at_columns == 32 &&
            rank_result.statistics.iterative_a_columns == 0 &&
            rank_result.statistics.iterative_at_columns == 0 &&
            rank_result.statistics.validation_a_columns == 1 &&
            rank_result.statistics.validation_at_columns == 1,
        "rank-deficient embedded systems are rejected after one validated candidate");

    amfls::SparseEmbeddingLsqrOptions invalid = options;
    invalid.regularization = 1.0;
    require_invalid_argument(
        [&]() {
            (void)amfls::solve_sparse_embedding_lsqr(
                tall, right_hand_side.data(), invalid);
        },
        "ridge is explicitly outside this baseline");

    DenseTestOperator square = row_major_operator(
        2, 2, {1.0, 0.0, 0.0, 1.0});
    const std::vector<double> square_rhs{1.0, 2.0};
    require_invalid_argument(
        [&]() {
            (void)amfls::solve_sparse_embedding_lsqr(
                square, square_rhs.data(), options);
        },
        "square systems are outside the tall-only baseline");

    DenseTestOperator wide = row_major_operator(
        2,
        3,
        {1.0, 0.0, 0.0,
         0.0, 1.0, 0.0});
    const std::vector<double> wide_rhs{1.0, 2.0};
    require_invalid_argument(
        [&]() {
            (void)amfls::solve_sparse_embedding_lsqr(
                wide, wide_rhs.data(), options);
        },
        "wide systems are outside the tall-only baseline");

    invalid = options;
    invalid.sketch_rows = 1;
    require_invalid_argument(
        [&]() {
            (void)amfls::solve_sparse_embedding_lsqr(
                tall, right_hand_side.data(), invalid);
        },
        "explicit sketch rows cannot be smaller than the column count");

    invalid = options;
    invalid.embedding_nonzeros = 0;
    require_invalid_argument(
        [&]() {
            (void)amfls::solve_sparse_embedding_lsqr(
                tall, right_hand_side.data(), invalid);
        },
        "sparse-embedding density must be positive");

    invalid = options;
    invalid.sketch_rows = 4;
    invalid.embedding_nonzeros = 5;
    require_invalid_argument(
        [&]() {
            (void)amfls::solve_sparse_embedding_lsqr(
                tall, right_hand_side.data(), invalid);
        },
        "sparse-embedding density cannot exceed the sketch rows");

    invalid = options;
    invalid.embedding_distortion = 1.0;
    require_invalid_argument(
        [&]() {
            (void)amfls::solve_sparse_embedding_lsqr(
                tall, right_hand_side.data(), invalid);
        },
        "embedding distortion must define a proper OSE tolerance");

    std::cout << "test_sparse_embedding_lsqr passed\n";
    return 0;
}
