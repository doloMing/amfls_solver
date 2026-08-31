#include "amfls/sparse_embedding_lsqr.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <limits>
#include <random>
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
#include "algorithms/mathematics/linalg/svd.hpp"
#include "algorithms/mathematics/operators/sparse_csr_operator.hpp"

namespace amfls {
namespace {

using Clock = std::chrono::steady_clock;

constexpr const char* counter_overflow =
    "sparse-embedding LSQR statistics exceed signed 64-bit range";

enum class ProductPhase { sketch, iteration };

struct RowAssignments {
    int nonzeros = 0;
    std::vector<int> destinations;
    std::vector<double> signs;
};

struct RightPreconditioner {
    math::Matrix factor;
    int rank = 0;
    bool usable = false;
};

double elapsed(Clock::time_point start) {
    return std::chrono::duration<double>(Clock::now() - start).count();
}

bool all_finite(std::span<const double> values) {
    return std::all_of(
        values.begin(), values.end(), [](double value) {
            return std::isfinite(value);
        });
}

void scale(std::vector<double>& values, double factor) {
    for (double& value : values) {
        value *= factor;
    }
}

void matrix_vector_product(
    const math::Matrix& matrix,
    const double* input,
    double* output) {
    if (matrix.rows() == 0) {
        return;
    }
    if (matrix.cols() == 0) {
        std::fill(output, output + matrix.rows(), 0.0);
        return;
    }
    math::gemv(matrix, false, input, 1.0, 0.0, output);
}

void transpose_matrix_vector_product(
    const math::Matrix& matrix,
    const double* input,
    double* output) {
    if (matrix.cols() == 0) {
        return;
    }
    if (matrix.rows() == 0) {
        std::fill(output, output + matrix.cols(), 0.0);
        return;
    }
    math::gemv(matrix, true, input, 1.0, 0.0, output);
}

void apply_original(
    const MatrixOperator& matrix,
    const double* input,
    int block_cols,
    double* output,
    RunStatistics& statistics) {
    const auto start = Clock::now();
    matrix.apply(input, block_cols, output);
    statistics.a_seconds += elapsed(start);
    math::checked_counter_add(
        statistics.a_columns, block_cols, counter_overflow);
    math::checked_counter_add(
        statistics.search_a_columns, block_cols, counter_overflow);
    math::checked_counter_add(
        statistics.a_block_calls, 1, counter_overflow);
    math::checked_counter_add(
        statistics.search_a_block_calls, 1, counter_overflow);
    math::checked_counter_add(
        statistics.iterative_a_columns, block_cols, counter_overflow);
    math::checked_counter_add(
        statistics.iterative_a_block_calls, 1, counter_overflow);
}

void apply_original_transpose(
    const MatrixOperator& matrix,
    const double* input,
    int block_cols,
    double* output,
    ProductPhase phase,
    RunStatistics& statistics) {
    const auto start = Clock::now();
    matrix.apply_transpose(input, block_cols, output);
    statistics.at_seconds += elapsed(start);
    math::checked_counter_add(
        statistics.at_columns, block_cols, counter_overflow);
    math::checked_counter_add(
        statistics.search_at_columns, block_cols, counter_overflow);
    math::checked_counter_add(
        statistics.at_block_calls, 1, counter_overflow);
    math::checked_counter_add(
        statistics.search_at_block_calls, 1, counter_overflow);
    if (phase == ProductPhase::sketch) {
        math::checked_counter_add(
            statistics.sketch_at_columns, block_cols, counter_overflow);
        math::checked_counter_add(
            statistics.sketch_at_block_calls, 1, counter_overflow);
    } else {
        math::checked_counter_add(
            statistics.iterative_at_columns, block_cols, counter_overflow);
        math::checked_counter_add(
            statistics.iterative_at_block_calls, 1, counter_overflow);
    }
}

void validate_options(
    const MatrixOperator& matrix,
    const SparseEmbeddingLsqrOptions& options) {
    if (options.regularization != 0.0) {
        throw std::invalid_argument(
            "sparse-embedding LSQR supports ordinary least squares only");
    }
    if (matrix.rows() <= matrix.cols()) {
        throw std::invalid_argument(
            "sparse-embedding LSQR requires rows greater than columns");
    }
    if (!std::isfinite(options.embedding_distortion) ||
        options.embedding_distortion <= 0.0 ||
        options.embedding_distortion >= 1.0) {
        throw std::invalid_argument(
            "embedding_distortion must lie strictly between zero and one");
    }
    if (!std::isfinite(options.embedding_failure_probability) ||
        options.embedding_failure_probability <= 0.0 ||
        options.embedding_failure_probability >= 1.0) {
        throw std::invalid_argument(
            "embedding_failure_probability must lie strictly between zero and one");
    }
    if ((!std::isfinite(options.relative_rank_tolerance) ||
         options.relative_rank_tolerance < 0.0) &&
        options.relative_rank_tolerance != -1.0) {
        throw std::invalid_argument(
            "relative_rank_tolerance must be nonnegative or -1");
    }
    if (!std::isfinite(options.absolute_rank_tolerance) ||
        options.absolute_rank_tolerance < 0.0) {
        throw std::invalid_argument(
            "absolute_rank_tolerance must be finite and nonnegative");
    }
    if (options.embedding_nonzeros <= 0) {
        throw std::invalid_argument(
            "embedding_nonzeros must be positive");
    }
    if (options.sketch_rows < 0 || options.sketch_block_size < 0 ||
        options.maximum_iterations < 0) {
        throw std::invalid_argument(
            "sparse-embedding LSQR work limits must be nonnegative");
    }
    if (options.sketch_rows > 0 &&
        options.sketch_rows < matrix.cols()) {
        throw std::invalid_argument(
            "sketch_rows must be zero or at least the column count");
    }
    if (options.sketch_rows > 0 &&
        options.embedding_nonzeros > options.sketch_rows) {
        throw std::invalid_argument(
            "embedding_nonzeros cannot exceed sketch_rows");
    }
}

int resolved_sketch_rows(
    int columns,
    const SparseEmbeddingLsqrOptions& options) {
    if (options.sketch_rows > 0) {
        return options.sketch_rows;
    }
    const long double dimension = columns;
    const long double distortion = options.embedding_distortion;
    const long double failure = options.embedding_failure_probability;
    const long double requested = std::ceil(
        (dimension * dimension + dimension) /
        (distortion * distortion * failure));
    if (!std::isfinite(requested) ||
        requested > std::numeric_limits<int>::max()) {
        throw std::length_error(
            "sparse-embedding row count exceeds LP64 limits");
    }
    return std::max(columns, static_cast<int>(requested));
}

std::uint64_t draw_below(
    std::mt19937_64& engine,
    std::uint64_t upper_bound) {
    const std::uint64_t rejection_threshold =
        (std::uint64_t{0} - upper_bound) % upper_bound;
    for (;;) {
        const std::uint64_t value = engine();
        if (value >= rejection_threshold) {
            return value % upper_bound;
        }
    }
}

RowAssignments make_row_assignments(
    int original_rows,
    int sketch_rows,
    const SparseEmbeddingLsqrOptions& options) {
    const std::array<std::uint32_t, 4> seed_words{
        static_cast<std::uint32_t>(options.seed),
        static_cast<std::uint32_t>(options.seed >> 32U),
        static_cast<std::uint32_t>(options.stream),
        static_cast<std::uint32_t>(options.stream >> 32U)};
    std::seed_seq sequence(seed_words.begin(), seed_words.end());
    std::mt19937_64 engine(sequence);

    RowAssignments assignments;
    assignments.nonzeros = options.embedding_nonzeros;
    const long long assignment_count = math::checked_nonnegative_multiply(
        original_rows,
        assignments.nonzeros,
        "sparse-embedding assignment count exceeds LP64 limits");
    assignments.destinations.resize(
        static_cast<std::size_t>(assignment_count));
    assignments.signs.resize(static_cast<std::size_t>(assignment_count));
    const auto destination_count = static_cast<std::uint64_t>(sketch_rows);
    const double scale = 1.0 / std::sqrt(
        static_cast<double>(assignments.nonzeros));
    for (int row = 0; row < original_rows; ++row) {
        const std::size_t first = static_cast<std::size_t>(row) *
            static_cast<std::size_t>(assignments.nonzeros);
        for (int slot = 0; slot < assignments.nonzeros; ++slot) {
            int destination = 0;
            bool duplicate = false;
            do {
                destination = static_cast<int>(
                    draw_below(engine, destination_count));
                duplicate = false;
                for (int previous = 0; previous < slot; ++previous) {
                    if (assignments.destinations[
                            first + static_cast<std::size_t>(previous)] ==
                        destination) {
                        duplicate = true;
                        break;
                    }
                }
            } while (duplicate);
            const std::size_t index =
                first + static_cast<std::size_t>(slot);
            assignments.destinations[index] = destination;
            assignments.signs[index] =
                (engine() & std::uint64_t{1}) == 0 ? -scale : scale;
        }
    }
    return assignments;
}

void build_native_csr_embedding(
    const math::SparseCsrOperator& matrix,
    const RowAssignments& assignments,
    math::Matrix& embedded) {
    for (int row = 0; row < matrix.rows(); ++row) {
        const std::size_t first = static_cast<std::size_t>(row) *
            static_cast<std::size_t>(assignments.nonzeros);
        for (int slot = 0; slot < assignments.nonzeros; ++slot) {
            const std::size_t index =
                first + static_cast<std::size_t>(slot);
            const int destination = assignments.destinations[index];
            const double sign = assignments.signs[index];
            for (std::int64_t entry = matrix.row_offsets()[row];
                 entry < matrix.row_offsets()[row + 1];
                 ++entry) {
                const int column =
                    static_cast<int>(matrix.column_indices()[entry]);
                embedded(destination, column) = std::fma(
                    sign,
                    matrix.values()[entry],
                    embedded(destination, column));
            }
        }
    }
}

int retained_rank(
    const math::SvdResult& svd,
    int sketch_rows,
    int columns,
    const SparseEmbeddingLsqrOptions& options) {
    if (svd.singular_values.empty() ||
        !(svd.singular_values.front() > 0.0) ||
        !std::isfinite(svd.singular_values.front())) {
        return 0;
    }
    double relative_tolerance = options.relative_rank_tolerance;
    if (relative_tolerance < 0.0) {
        relative_tolerance =
            16.0 * std::numeric_limits<double>::epsilon() *
            std::max(sketch_rows, columns);
    }
    const double cutoff = std::max(
        relative_tolerance * svd.singular_values.front(),
        options.absolute_rank_tolerance);
    int rank = 0;
    while (rank < static_cast<int>(svd.singular_values.size()) &&
           svd.singular_values[rank] > cutoff &&
           std::isfinite(svd.singular_values[rank]) &&
           std::isfinite(1.0 / svd.singular_values[rank])) {
        ++rank;
    }
    return rank;
}

RightPreconditioner build_preconditioner(
    const MatrixOperator& matrix,
    const SparseEmbeddingLsqrOptions& options,
    int sketch_rows,
    RunStatistics& statistics) {
    const RowAssignments assignments = make_row_assignments(
        matrix.rows(), sketch_rows, options);
    const int block_size = options.sketch_block_size > 0
        ? std::min(options.sketch_block_size, sketch_rows)
        : sketch_rows;
    math::Matrix embedded(sketch_rows, matrix.cols());

    if (const auto* sparse =
            dynamic_cast<const math::SparseCsrOperator*>(&matrix)) {
        // For explicit CSR, form S*A by accumulating each signed source-row
        // contribution directly.  The generic path below forms the same
        // black-box product A* S* = (S A)*.
        build_native_csr_embedding(
            *sparse,
            assignments,
            embedded);
    } else {
        for (int first = 0; first < sketch_rows;) {
            const int count = std::min(block_size, sketch_rows - first);
            math::Matrix sketch_transpose_block(matrix.rows(), count);
            const int end = math::checked_add(
                first,
                count,
                "sparse-embedding block endpoint exceeds LP64 limits");
            for (int row = 0; row < matrix.rows(); ++row) {
                const std::size_t assignment_first =
                    static_cast<std::size_t>(row) *
                    static_cast<std::size_t>(assignments.nonzeros);
                for (int slot = 0; slot < assignments.nonzeros; ++slot) {
                    const std::size_t index = assignment_first +
                        static_cast<std::size_t>(slot);
                    const int destination = assignments.destinations[index];
                    if (destination >= first && destination < end) {
                        sketch_transpose_block(
                            row, destination - first) +=
                            assignments.signs[index];
                    }
                }
            }
            math::Matrix product(matrix.cols(), count);
            apply_original_transpose(
                matrix,
                sketch_transpose_block.data(),
                count,
                product.data(),
                ProductPhase::sketch,
                statistics);
            for (int embedded_row = 0;
                 embedded_row < count;
                 ++embedded_row) {
                for (int col = 0; col < matrix.cols(); ++col) {
                    embedded(first + embedded_row, col) =
                        product(col, embedded_row);
                }
            }
            first = end;
        }
    }

    if (!all_finite(std::span<const double>(
            embedded.data(), static_cast<std::size_t>(embedded.size())))) {
        return {math::Matrix(matrix.cols(), 0), 0, false};
    }

    const auto decomposition_start = Clock::now();
    math::SvdResult svd = math::thin_svd(embedded);
    statistics.projected_solve_seconds += elapsed(decomposition_start);
    const int rank = retained_rank(
        svd, sketch_rows, matrix.cols(), options);
    math::Matrix factor(matrix.cols(), rank);
    for (int col = 0; col < rank; ++col) {
        const double inverse = 1.0 / svd.singular_values[col];
        for (int row = 0; row < matrix.cols(); ++row) {
            factor(row, col) = svd.vt(col, row) * inverse;
        }
    }
    const bool finite_factor = all_finite(std::span<const double>(
        factor.data(), static_cast<std::size_t>(factor.size())));
    const bool usable = rank == matrix.cols() && finite_factor;
    return {std::move(factor), rank, usable};
}

class RightPreconditionedSystem {
public:
    RightPreconditionedSystem(
        const MatrixOperator& matrix,
        const math::Matrix& factor,
        RunStatistics& statistics)
        : matrix_(matrix),
          factor_(factor),
          statistics_(statistics) {}

    std::vector<double> original_candidate(
        const std::vector<double>& inner_solution) const {
        std::vector<double> result(matrix_.cols());
        matrix_vector_product(
            factor_, inner_solution.data(), result.data());
        return result;
    }

    bool apply(const double* input, double* output) {
        std::vector<double> original_direction(matrix_.cols());
        matrix_vector_product(
            factor_, input, original_direction.data());
        if (!all_finite(std::span<const double>(original_direction))) {
            return false;
        }
        apply_original(
            matrix_,
            original_direction.data(),
            1,
            output,
            statistics_);
        return all_finite(std::span<const double>(
            output, static_cast<std::size_t>(matrix_.rows())));
    }

    bool apply_transpose(const double* input, double* output) {
        std::vector<double> original_transpose(matrix_.cols());
        apply_original_transpose(
            matrix_,
            input,
            1,
            original_transpose.data(),
            ProductPhase::iteration,
            statistics_);
        if (!all_finite(std::span<const double>(original_transpose))) {
            return false;
        }
        transpose_matrix_vector_product(
            factor_, original_transpose.data(), output);
        return all_finite(std::span<const double>(
            output, static_cast<std::size_t>(matrix_.cols())));
    }

private:
    const MatrixOperator& matrix_;
    const math::Matrix& factor_;
    RunStatistics& statistics_;
};

LeastSquaresResult validate_candidate(
    const MatrixOperator& matrix,
    const double* right_hand_side,
    const SparseEmbeddingLsqrOptions& options,
    const std::vector<double>& original_solution,
    int iteration,
    int sketch_rows,
    int rank,
    RunStatistics& statistics) {
    LeastSquaresResult result = math::validate_original_candidate(
        matrix,
        right_hand_side,
        std::span<const double>(original_solution),
        0.0,
        options.tolerance,
        0.0,
        statistics);
    result.iterations = iteration;
    result.depth = iteration;
    result.auxiliary_width = sketch_rows;
    result.basis_rank = rank;
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

void set_numerical_breakdown(LeastSquaresResult& result) {
    result.status = SolverStatus::numerical_breakdown;
    result.stop_reason = StopReason::numerical_breakdown;
}

}  // namespace

LeastSquaresResult solve_sparse_embedding_lsqr(
    const MatrixOperator& matrix,
    const double* right_hand_side,
    const SparseEmbeddingLsqrOptions& options) {
    math::validate_common_inputs(
        matrix, right_hand_side, options.regularization, options.tolerance);
    validate_options(matrix, options);

    const auto total_start = Clock::now();
    RunStatistics statistics;
    const int sketch_rows = resolved_sketch_rows(matrix.cols(), options);
    if (options.embedding_nonzeros > sketch_rows) {
        throw std::invalid_argument(
            "embedding_nonzeros cannot exceed resolved sketch rows");
    }
    RightPreconditioner preconditioner = build_preconditioner(
        matrix,
        options,
        sketch_rows,
        statistics);

    std::vector<double> original_solution(matrix.cols(), 0.0);
    LeastSquaresResult result = validate_candidate(
        matrix,
        right_hand_side,
        options,
        original_solution,
        0,
        sketch_rows,
        preconditioner.rank,
        statistics);
    if (!math::candidate_validation_failed_numerically(result) &&
        result.status != SolverStatus::success &&
        !preconditioner.usable) {
        set_numerical_breakdown(result);
    }
    math::CertificateCheckpointSchedule checkpoint_schedule(
        options.regularization, options.tolerance);
    checkpoint_schedule.record_evaluation(0, result);
    std::vector<IterationRecord> trace;
    trace.push_back(math::make_iteration_record(
        result, 0, 0, sketch_rows));
    if (math::candidate_validation_failed_numerically(result) ||
        result.status == SolverStatus::success ||
        !preconditioner.usable) {
        finish_result(result, std::move(trace), statistics, total_start);
        return result;
    }

    RightPreconditionedSystem system(
        matrix,
        preconditioner.factor,
        statistics);
    std::vector<double> inner_solution(matrix.cols(), 0.0);
    int committed_iteration = 0;
    int evaluated_iteration = 0;
    const auto evaluate_current_candidate = [&] {
        original_solution = system.original_candidate(inner_solution);
        result = validate_candidate(
            matrix,
            right_hand_side,
            options,
            original_solution,
            committed_iteration,
            sketch_rows,
            preconditioner.rank,
            statistics);
        checkpoint_schedule.record_evaluation(
            committed_iteration, result);
        trace.push_back(math::make_iteration_record(
            result, 0, committed_iteration, sketch_rows));
        evaluated_iteration = committed_iteration;
    };
    const auto record_numerical_terminal = [&] {
        if (evaluated_iteration != committed_iteration) {
            evaluate_current_candidate();
        }
        if (result.status != SolverStatus::success &&
            !math::candidate_validation_failed_numerically(result)) {
            set_numerical_breakdown(result);
        }
    };
    std::vector<double> u(
        right_hand_side, right_hand_side + matrix.rows());
    double beta = math::vector_norm(u.data(), matrix.rows());
    if (!(beta > 0.0) || !std::isfinite(beta)) {
        if (std::isfinite(beta)) {
            result.status = SolverStatus::work_limit;
            result.stop_reason = StopReason::exhausted_search_space;
        } else {
            set_numerical_breakdown(result);
        }
        finish_result(result, std::move(trace), statistics, total_start);
        return result;
    }
    scale(u, 1.0 / beta);

    std::vector<double> v(matrix.cols());
    if (!system.apply_transpose(u.data(), v.data())) {
        set_numerical_breakdown(result);
        finish_result(result, std::move(trace), statistics, total_start);
        return result;
    }
    double alpha = math::vector_norm(v.data(), matrix.cols());
    if (!(alpha > 0.0) || !std::isfinite(alpha)) {
        if (std::isfinite(alpha)) {
            result.status = SolverStatus::work_limit;
            result.stop_reason = StopReason::exhausted_search_space;
        } else {
            set_numerical_breakdown(result);
        }
        finish_result(result, std::move(trace), statistics, total_start);
        return result;
    }
    scale(v, 1.0 / alpha);
    std::vector<double> w = v;
    double rhobar = alpha;
    double phibar = beta;
    const int maximum_iterations = options.maximum_iterations > 0
        ? options.maximum_iterations
        : matrix.cols();

    std::vector<double> next_u(matrix.rows());
    std::vector<double> next_v(matrix.cols());
    std::vector<double> candidate_inner(matrix.cols());
    std::vector<double> candidate_w(matrix.cols());
    for (int iteration = 1; iteration <= maximum_iterations; ++iteration) {
        if (!system.apply(v.data(), next_u.data())) {
            record_numerical_terminal();
            break;
        }
        for (int row = 0; row < matrix.rows(); ++row) {
            next_u[row] -= alpha * u[row];
        }
        beta = math::vector_norm(next_u.data(), matrix.rows());
        if (!std::isfinite(beta) ||
            !all_finite(std::span<const double>(next_u))) {
            record_numerical_terminal();
            break;
        }
        if (beta > 0.0) {
            scale(next_u, 1.0 / beta);
        }

        // The current LSQR candidate needs beta but not the next A* product.
        // Commit and, when due, validate it before preparing continuation.
        const double rho = std::hypot(rhobar, beta);
        if (!(rho > 0.0) || !std::isfinite(rho)) {
            record_numerical_terminal();
            break;
        }
        const double cosine = rhobar / rho;
        const double sine = beta / rho;
        const double phi = cosine * phibar;
        phibar = sine * phibar;
        const double update = phi / rho;
        if (!std::isfinite(cosine) || !std::isfinite(sine) ||
            !std::isfinite(phi) || !std::isfinite(phibar) ||
            !std::isfinite(update)) {
            record_numerical_terminal();
            break;
        }

        candidate_inner = inner_solution;
        for (int col = 0; col < matrix.cols(); ++col) {
            candidate_inner[col] += update * w[col];
        }
        if (!all_finite(std::span<const double>(candidate_inner)) ||
            !all_finite(std::span<const double>(next_u))) {
            record_numerical_terminal();
            break;
        }

        inner_solution.swap(candidate_inner);
        u.swap(next_u);
        committed_iteration = iteration;

        const bool beta_breakdown = beta == 0.0;
        const bool checkpoint =
            checkpoint_schedule.should_evaluate(iteration) ||
            beta_breakdown || iteration == maximum_iterations;
        bool candidate_was_evaluated = false;
        if (checkpoint) {
            evaluate_current_candidate();
            candidate_was_evaluated = true;
            if (math::candidate_validation_failed_numerically(result) ||
                result.status == SolverStatus::success) {
                break;
            }
            if (beta_breakdown) {
                result.status = SolverStatus::work_limit;
                result.stop_reason = StopReason::exhausted_search_space;
                break;
            }
            if (iteration == maximum_iterations) {
                result.status = SolverStatus::work_limit;
                result.stop_reason = StopReason::maximum_depth;
                break;
            }
        }

        if (!system.apply_transpose(u.data(), next_v.data())) {
            record_numerical_terminal();
            break;
        }
        for (int col = 0; col < matrix.cols(); ++col) {
            next_v[col] -= beta * v[col];
        }
        const double next_alpha = math::vector_norm(
            next_v.data(), matrix.cols());
        bool continuation_is_finite =
            std::isfinite(next_alpha) &&
            all_finite(std::span<const double>(next_v));
        if (next_alpha > 0.0 && std::isfinite(next_alpha)) {
            scale(next_v, 1.0 / next_alpha);
        }

        const double theta = sine * next_alpha;
        const double next_rhobar = -cosine * next_alpha;
        const double recurrence = theta / rho;
        continuation_is_finite = continuation_is_finite &&
            std::isfinite(theta) && std::isfinite(next_rhobar) &&
            std::isfinite(recurrence);
        if (continuation_is_finite && next_alpha > 0.0) {
            for (int col = 0; col < matrix.cols(); ++col) {
                candidate_w[col] = next_v[col] - recurrence * w[col];
            }
            continuation_is_finite =
                all_finite(std::span<const double>(candidate_w));
        }

        if (!continuation_is_finite || next_alpha == 0.0) {
            if (!candidate_was_evaluated) {
                evaluate_current_candidate();
                if (math::candidate_validation_failed_numerically(result) ||
                    result.status == SolverStatus::success) {
                    break;
                }
            }
            result.status = continuation_is_finite
                ? SolverStatus::work_limit
                : SolverStatus::numerical_breakdown;
            result.stop_reason = continuation_is_finite
                ? StopReason::exhausted_search_space
                : StopReason::numerical_breakdown;
            break;
        }

        alpha = next_alpha;
        rhobar = next_rhobar;
        w.swap(candidate_w);
        v.swap(next_v);
    }

    finish_result(result, std::move(trace), statistics, total_start);
    return result;
}

}  // namespace amfls
