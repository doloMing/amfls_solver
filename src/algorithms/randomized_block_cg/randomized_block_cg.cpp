#include "amfls/randomized_block_cg.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

#include "algorithms/mathematics/checked_arithmetic.hpp"
#include "algorithms/mathematics/krylov/candidate_validation.hpp"
#include "algorithms/mathematics/krylov/certificate_checkpoint_schedule.hpp"
#include "algorithms/mathematics/krylov/krylov_workspace.hpp"
#include "algorithms/mathematics/linalg/blas_lapack.hpp"
#include "algorithms/mathematics/linalg/matrix.hpp"
#include "algorithms/mathematics/random/gaussian.hpp"

namespace amfls {
namespace {

using Clock = std::chrono::steady_clock;

constexpr const char* counter_overflow =
    "randomized block CG statistics exceed signed 64-bit range";

struct ProjectedNormalSystem {
    math::Matrix matrix;
    std::vector<double> right_hand_side;
};

double elapsed(Clock::time_point start) {
    return std::chrono::duration<double>(Clock::now() - start).count();
}

bool all_finite(const math::Matrix& matrix) {
    return std::all_of(
        matrix.data(), matrix.data() + matrix.size(), [](double value) {
            return std::isfinite(value);
        });
}

void apply_search(
    const MatrixOperator& matrix,
    const math::Matrix& input,
    math::Matrix& output,
    RunStatistics& statistics) {
    const auto start = Clock::now();
    matrix.apply(input.data(), input.cols(), output.data());
    statistics.a_seconds += elapsed(start);
    math::checked_counter_add(
        statistics.a_columns, input.cols(), counter_overflow);
    math::checked_counter_add(
        statistics.a_block_calls, 1, counter_overflow);
    math::checked_counter_add(
        statistics.search_a_columns, input.cols(), counter_overflow);
    math::checked_counter_add(
        statistics.search_a_block_calls, 1, counter_overflow);
    math::checked_counter_add(
        statistics.iterative_a_columns, input.cols(), counter_overflow);
    math::checked_counter_add(
        statistics.iterative_a_block_calls, 1, counter_overflow);
}

void apply_transpose_search(
    const MatrixOperator& matrix,
    const double* input,
    int block_columns,
    math::Matrix& output,
    RunStatistics& statistics) {
    const auto start = Clock::now();
    matrix.apply_transpose(input, block_columns, output.data());
    statistics.at_seconds += elapsed(start);
    math::checked_counter_add(
        statistics.at_columns, block_columns, counter_overflow);
    math::checked_counter_add(
        statistics.at_block_calls, 1, counter_overflow);
    math::checked_counter_add(
        statistics.search_at_columns, block_columns, counter_overflow);
    math::checked_counter_add(
        statistics.search_at_block_calls, 1, counter_overflow);
    math::checked_counter_add(
        statistics.iterative_at_columns, block_columns, counter_overflow);
    math::checked_counter_add(
        statistics.iterative_at_block_calls, 1, counter_overflow);
}

math::Matrix gaussian_augmentation(
    int dimension,
    const RandomizedBlockCgOptions& options,
    RunStatistics& statistics) {
    math::Matrix random = math::gaussian_matrix(
        dimension,
        options.random_block_size,
        options.seed,
        options.stream,
        0);
    math::checked_counter_add(
        statistics.gaussian_random_block_requests, 1, counter_overflow);
    math::checked_counter_add(
        statistics.gaussian_random_columns,
        options.random_block_size,
        counter_overflow);
    math::checked_counter_add(
        statistics.gaussian_random_values,
        math::checked_nonnegative_multiply(
            dimension, options.random_block_size, counter_overflow),
        counter_overflow);
    return random;
}

LeastSquaresResult validate_zero_candidate(
    const MatrixOperator& matrix,
    const double* right_hand_side,
    const RandomizedBlockCgOptions& options,
    RunStatistics& statistics) {
    const std::vector<double> zero(
        static_cast<std::size_t>(matrix.cols()), 0.0);
    LeastSquaresResult result = math::validate_original_candidate(
        matrix,
        right_hand_side,
        std::span<const double>(zero),
        options.regularization,
        options.tolerance,
        0.0,
        statistics);
    result.iterations = 0;
    result.depth = 0;
    result.auxiliary_width = options.random_block_size;
    result.basis_rank = 0;
    result.trace.push_back(math::make_iteration_record(
        result, 0, 0, options.random_block_size));
    return result;
}

void finish_result(
    LeastSquaresResult& result,
    std::vector<IterationRecord> trace,
    RunStatistics& statistics,
    Clock::time_point total_start) {
    result.trace = std::move(trace);
    statistics.total_seconds = elapsed(total_start);
    result.statistics = statistics;
}

void mark_numerical_breakdown(LeastSquaresResult& result) {
    result.status = SolverStatus::numerical_breakdown;
    result.stop_reason = StopReason::numerical_breakdown;
}

void append_projected_normal_block(
    const math::Matrix& applied_basis,
    const math::Matrix& applied_frontier,
    const double* right_hand_side,
    ProjectedNormalSystem& projected,
    RunStatistics& statistics) {
    if (applied_basis.rows() != applied_frontier.rows() ||
        projected.matrix.rows() != applied_basis.cols() ||
        projected.matrix.cols() != applied_basis.cols() ||
        projected.right_hand_side.size() !=
            static_cast<std::size_t>(applied_basis.cols())) {
        throw std::logic_error(
            "randomized block CG projected system is inconsistent");
    }
    const auto start = Clock::now();
    const int old_size = applied_basis.cols();
    const int added = applied_frontier.cols();
    const int new_size = math::checked_add(
        old_size,
        added,
        "randomized block CG projected dimension exceeds LP64 limits");
    math::Matrix grown(new_size, new_size);
    if (old_size > 0) {
        math::set_block(grown, 0, 0, projected.matrix);
        const math::Matrix cross = math::transpose_multiply(
            applied_basis, applied_frontier);
        for (int col = 0; col < added; ++col) {
            for (int row = 0; row < old_size; ++row) {
                grown(row, old_size + col) = cross(row, col);
                grown(old_size + col, row) = cross(row, col);
            }
        }
    }
    const math::Matrix diagonal_block = math::transpose_multiply(
        applied_frontier, applied_frontier);
    math::set_block(grown, old_size, old_size, diagonal_block);

    projected.right_hand_side.resize(
        static_cast<std::size_t>(new_size));
    for (int col = 0; col < added; ++col) {
        projected.right_hand_side[
            static_cast<std::size_t>(old_size + col)] = math::vector_dot(
                applied_frontier.column_data(col),
                right_hand_side,
                applied_frontier.rows());
    }
    projected.matrix = std::move(grown);
    statistics.projected_solve_seconds += elapsed(start);
}

LeastSquaresResult evaluate_projected_normal_system(
    const MatrixOperator& matrix,
    const double* right_hand_side,
    const RandomizedBlockCgOptions& options,
    const math::Matrix& basis,
    const ProjectedNormalSystem& projected,
    RunStatistics& statistics,
    int depth) {
    if (projected.matrix.rows() != basis.cols() ||
        projected.matrix.cols() != basis.cols() ||
        projected.right_hand_side.size() !=
            static_cast<std::size_t>(basis.cols())) {
        throw std::logic_error(
            "randomized block CG basis and projected system do not match");
    }
    const auto start = Clock::now();
    math::Matrix shifted = projected.matrix;
    for (int index = 0; index < shifted.rows(); ++index) {
        shifted(index, index) += options.regularization;
    }
    math::DenseSpdSolve reduced = math::solve_dense_spd(
        std::move(shifted), projected.right_hand_side);
    std::vector<double> solution(
        static_cast<std::size_t>(matrix.cols()), 0.0);
    if (reduced.positive_definite) {
        math::gemv(
            basis,
            false,
            reduced.solution.data(),
            1.0,
            0.0,
            solution.data());
    }
    statistics.projected_solve_seconds += elapsed(start);

    LeastSquaresResult result = math::validate_original_candidate(
        matrix,
        right_hand_side,
        std::span<const double>(solution),
        options.regularization,
        options.tolerance,
        0.0,
        statistics);
    if (!reduced.positive_definite &&
        result.status != SolverStatus::success) {
        mark_numerical_breakdown(result);
    }
    result.iterations = depth;
    result.depth = depth;
    result.auxiliary_width = options.random_block_size;
    result.basis_rank = basis.cols();
    result.trace.push_back(math::make_iteration_record(
        result, 0, depth, options.random_block_size));
    return result;
}

}  // namespace

LeastSquaresResult solve_randomized_block_cg(
    const MatrixOperator& matrix,
    const double* right_hand_side,
    const RandomizedBlockCgOptions& options) {
    math::validate_common_inputs(
        matrix, right_hand_side, options.regularization, options.tolerance);
    if (!(options.regularization > 0.0)) {
        throw std::invalid_argument(
            "randomized block CG requires positive regularization");
    }
    if (options.random_block_size <= 0) {
        throw std::invalid_argument(
            "random_block_size must be positive");
    }
    if (options.maximum_depth < 0) {
        throw std::invalid_argument(
            "maximum_depth must be nonnegative");
    }
    const int starting_width = math::checked_add(
        options.random_block_size,
        1,
        "randomized block CG starting width exceeds LP64 limits");

    const auto total_start = Clock::now();
    RunStatistics statistics;
    LeastSquaresResult result = validate_zero_candidate(
        matrix, right_hand_side, options, statistics);
    math::CertificateCheckpointSchedule checkpoint_schedule(
        options.regularization, options.tolerance);
    checkpoint_schedule.record_evaluation(0, result);
    std::vector<IterationRecord> trace{result.trace.front()};
    result.trace.clear();
    if (math::candidate_validation_failed_numerically(result) ||
        result.status == SolverStatus::success) {
        finish_result(
            result, std::move(trace), statistics, total_start);
        return result;
    }

    // For ridge regression the paper's positive-semidefinite matrix is
    // H=A^*A, its system right-hand side is c=A^*b, and its shift mu is the
    // least-squares regularization.  We build K_t(H,[c,Omega]); adding the
    // shift during expansion is unnecessary by Krylov shift invariance.
    math::Matrix normal_right_hand_side(matrix.cols(), 1);
    apply_transpose_search(
        matrix,
        right_hand_side,
        1,
        normal_right_hand_side,
        statistics);
    if (!all_finite(normal_right_hand_side)) {
        mark_numerical_breakdown(result);
        finish_result(
            result, std::move(trace), statistics, total_start);
        return result;
    }

    math::Matrix start(matrix.cols(), starting_width);
    for (int row = 0; row < matrix.cols(); ++row) {
        start(row, 0) = normal_right_hand_side(row, 0);
    }
    math::Matrix random = gaussian_augmentation(
        matrix.cols(), options, statistics);
    math::set_block(start, 0, 1, random);

    math::Matrix basis(matrix.cols(), 0);
    math::Matrix applied_basis(matrix.rows(), 0);
    ProjectedNormalSystem projected{math::Matrix(0, 0), {}};
    math::KrylovOrthogonalizationAudit initial_orthogonalization;
    math::Matrix frontier = math::orthogonalize_krylov_block(
        std::move(start),
        basis,
        matrix.cols(),
        matrix.cols(),
        matrix.cols(),
        statistics,
        nullptr,
        &initial_orthogonalization);
    if (frontier.cols() == 0) {
        if (initial_orthogonalization.nonfinite_residual) {
            mark_numerical_breakdown(result);
        } else if (
            initial_orthogonalization.positive_finite_below_cutoff) {
            result.status = SolverStatus::precision_limit;
            result.stop_reason = StopReason::precision_limit;
        } else {
            result.status = SolverStatus::work_limit;
            result.stop_reason = StopReason::exhausted_search_space;
        }
        finish_result(
            result, std::move(trace), statistics, total_start);
        return result;
    }

    const int maximum_depth = options.maximum_depth > 0
        ? std::min(options.maximum_depth, matrix.cols())
        : matrix.cols();
    int depth = 0;
    int evaluated_depth = 0;
    const auto evaluate_current_candidate = [&] {
        result = evaluate_projected_normal_system(
            matrix,
            right_hand_side,
            options,
            basis,
            projected,
            statistics,
            depth);
        checkpoint_schedule.record_evaluation(depth, result);
        trace.push_back(result.trace.front());
        result.trace.clear();
        evaluated_depth = depth;
    };
    const auto evaluate_terminal_candidate = [&] {
        if (evaluated_depth != depth) {
            evaluate_current_candidate();
        }
        return math::candidate_validation_failed_numerically(result) ||
            result.status == SolverStatus::success;
    };
    while (true) {
        math::Matrix applied_frontier(matrix.rows(), frontier.cols());
        apply_search(matrix, frontier, applied_frontier, statistics);
        if (!all_finite(applied_frontier)) {
            if (!evaluate_terminal_candidate() &&
                result.status != SolverStatus::success) {
                mark_numerical_breakdown(result);
            }
            break;
        }

        append_projected_normal_block(
            applied_basis,
            applied_frontier,
            right_hand_side,
            projected,
            statistics);
        basis.append_columns(frontier);
        applied_basis.append_columns(applied_frontier);
        depth = math::checked_add(
            depth,
            1,
            "randomized block CG depth exceeds LP64 limits");

        const bool basis_complete = basis.cols() >= matrix.cols();
        const bool depth_complete = depth >= maximum_depth;
        if (checkpoint_schedule.should_evaluate(depth) ||
            basis_complete || depth_complete) {
            evaluate_current_candidate();
            if (math::candidate_validation_failed_numerically(result) ||
                result.status == SolverStatus::success) {
                break;
            }
        }
        if (basis_complete) {
            result.status = SolverStatus::precision_limit;
            result.stop_reason = StopReason::precision_limit;
            break;
        }
        if (depth_complete) {
            result.status = SolverStatus::work_limit;
            result.stop_reason = StopReason::maximum_depth;
            break;
        }

        math::Matrix normal_frontier(matrix.cols(), frontier.cols());
        apply_transpose_search(
            matrix,
            applied_frontier.data(),
            applied_frontier.cols(),
            normal_frontier,
            statistics);
        if (!all_finite(normal_frontier)) {
            if (!evaluate_terminal_candidate() &&
                result.status != SolverStatus::success) {
                mark_numerical_breakdown(result);
            }
            break;
        }

        math::KrylovOrthogonalizationAudit orthogonalization;
        math::Matrix next = math::orthogonalize_krylov_block(
            std::move(normal_frontier),
            basis,
            matrix.cols(),
            matrix.cols(),
            matrix.cols(),
            statistics,
            nullptr,
            &orthogonalization);
        if (next.cols() == 0) {
            if (!evaluate_terminal_candidate() &&
                result.status != SolverStatus::success) {
                if (orthogonalization.nonfinite_residual) {
                    mark_numerical_breakdown(result);
                } else if (
                    orthogonalization.positive_finite_below_cutoff) {
                    result.status = SolverStatus::precision_limit;
                    result.stop_reason = StopReason::precision_limit;
                } else {
                    result.status = SolverStatus::work_limit;
                    result.stop_reason = StopReason::exhausted_search_space;
                }
            }
            break;
        }
        frontier = std::move(next);
    }

    result.iterations = depth;
    result.depth = depth;
    result.auxiliary_width = options.random_block_size;
    result.basis_rank = basis.cols();
    finish_result(result, std::move(trace), statistics, total_start);
    return result;
}

}  // namespace amfls
