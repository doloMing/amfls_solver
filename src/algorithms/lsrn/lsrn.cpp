#include "amfls/lsrn.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <limits>
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
#include "algorithms/mathematics/random/gaussian.hpp"

namespace amfls {
namespace {

using Clock = std::chrono::steady_clock;

constexpr const char* lsrn_counter_overflow =
    "LSRN statistics exceed signed 64-bit range";

enum class ProductPhase { sketch, iteration };

struct Preconditioner {
    bool tall = true;
    math::Matrix factor;
    int rank = 0;
    int sketch_size = 0;
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

void apply_original(
    const MatrixOperator& matrix,
    const double* input,
    int block_cols,
    double* output,
    ProductPhase phase,
    RunStatistics& statistics) {
    const auto start = Clock::now();
    matrix.apply(input, block_cols, output);
    statistics.a_seconds += elapsed(start);
    math::checked_counter_add(
        statistics.a_columns, block_cols, lsrn_counter_overflow);
    math::checked_counter_add(
        statistics.search_a_columns, block_cols, lsrn_counter_overflow);
    math::checked_counter_add(
        statistics.a_block_calls, 1, lsrn_counter_overflow);
    math::checked_counter_add(
        statistics.search_a_block_calls, 1, lsrn_counter_overflow);
    if (phase == ProductPhase::sketch) {
        math::checked_counter_add(
            statistics.sketch_a_columns, block_cols, lsrn_counter_overflow);
        math::checked_counter_add(
            statistics.sketch_a_block_calls, 1, lsrn_counter_overflow);
    } else {
        math::checked_counter_add(
            statistics.iterative_a_columns, block_cols, lsrn_counter_overflow);
        math::checked_counter_add(
            statistics.iterative_a_block_calls, 1, lsrn_counter_overflow);
    }
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
        statistics.at_columns, block_cols, lsrn_counter_overflow);
    math::checked_counter_add(
        statistics.search_at_columns, block_cols, lsrn_counter_overflow);
    math::checked_counter_add(
        statistics.at_block_calls, 1, lsrn_counter_overflow);
    math::checked_counter_add(
        statistics.search_at_block_calls, 1, lsrn_counter_overflow);
    if (phase == ProductPhase::sketch) {
        math::checked_counter_add(
            statistics.sketch_at_columns, block_cols, lsrn_counter_overflow);
        math::checked_counter_add(
            statistics.sketch_at_block_calls, 1, lsrn_counter_overflow);
    } else {
        math::checked_counter_add(
            statistics.iterative_at_columns, block_cols, lsrn_counter_overflow);
        math::checked_counter_add(
            statistics.iterative_at_block_calls, 1, lsrn_counter_overflow);
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

int checked_system_dimension(int first, int second) {
    const long long total = static_cast<long long>(first) + second;
    if (total > std::numeric_limits<int>::max()) {
        throw std::length_error("augmented LSRN dimension exceeds LP64 limits");
    }
    return static_cast<int>(total);
}

int sketch_size(double oversampling, int dimension) {
    const long double requested = std::ceil(
        static_cast<long double>(oversampling) * dimension);
    if (requested > std::numeric_limits<int>::max()) {
        throw std::length_error("LSRN sketch size exceeds LP64 limits");
    }
    int size = static_cast<int>(requested);
    if (size <= dimension) {
        if (dimension == std::numeric_limits<int>::max()) {
            throw std::length_error("LSRN sketch cannot exceed its dimension");
        }
        size = dimension + 1;
    }
    return size;
}

math::Matrix gaussian_block(
    int rows,
    int cols,
    const LsrnOptions& options,
    std::uint64_t first_column,
    RunStatistics& statistics) {
    math::Matrix result = math::gaussian_matrix(
        rows,
        cols,
        options.seed,
        options.stream,
        first_column);
    if (cols > 0) {
        math::checked_counter_add(
            statistics.gaussian_random_block_requests,
            1,
            lsrn_counter_overflow);
    }
    math::checked_counter_add(
        statistics.gaussian_random_columns, cols, lsrn_counter_overflow);
    math::checked_counter_add(
        statistics.gaussian_random_values,
        math::checked_nonnegative_multiply(
            rows, cols, lsrn_counter_overflow),
        lsrn_counter_overflow);
    return result;
}

int retained_rank(
    const math::SvdResult& svd,
    int sketch_rows,
    int sketch_cols,
    const LsrnOptions& options) {
    if (svd.singular_values.empty() ||
        !(svd.singular_values.front() > 0.0) ||
        !std::isfinite(svd.singular_values.front())) {
        return 0;
    }
    double relative_tolerance = options.relative_rank_tolerance;
    if (relative_tolerance < 0.0) {
        relative_tolerance =
            16.0 * std::numeric_limits<double>::epsilon() *
            std::max(sketch_rows, sketch_cols);
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

Preconditioner build_tall_preconditioner(
    const MatrixOperator& matrix,
    const LsrnOptions& options,
    RunStatistics& statistics) {
    const bool ridge = options.regularization > 0.0;
    const double regularization_root = std::sqrt(options.regularization);
    const int random_rows = ridge
        ? checked_system_dimension(matrix.rows(), matrix.cols())
        : matrix.rows();
    const int size = sketch_size(options.oversampling, matrix.cols());
    const int block_size = options.sketch_block_size > 0
        ? std::min(options.sketch_block_size, size)
        : size;
    // Materialize B^T G^T in column blocks.  Its left singular vectors are
    // the V-tilde vectors of G B, so N = V-tilde Sigma-tilde^{-1} can be
    // formed without storing the transposed sketch.  For ridge,
    // B = [A; sqrt(lambda) I].
    math::Matrix sketch_transpose(matrix.cols(), size);

    for (int first = 0; first < size;) {
        const int count = std::min(block_size, size - first);
        math::Matrix random = gaussian_block(
            random_rows,
            count,
            options,
            static_cast<std::uint64_t>(first),
            statistics);
        math::Matrix top;
        const double* sketch_input = random.data();
        if (ridge) {
            top = math::Matrix(matrix.rows(), count);
            for (int col = 0; col < count; ++col) {
                for (int row = 0; row < matrix.rows(); ++row) {
                    top(row, col) = random(row, col);
                }
            }
            sketch_input = top.data();
        }
        math::Matrix product(matrix.cols(), count);
        apply_original_transpose(
            matrix,
            sketch_input,
            count,
            product.data(),
            ProductPhase::sketch,
            statistics);
        if (ridge) {
            for (int col = 0; col < count; ++col) {
                for (int row = 0; row < matrix.cols(); ++row) {
                    product(row, col) = std::fma(
                        regularization_root,
                        random(matrix.rows() + row, col),
                        product(row, col));
                }
            }
        }
        for (int col = 0; col < count; ++col) {
            for (int row = 0; row < matrix.cols(); ++row) {
                sketch_transpose(row, first + col) = product(row, col);
            }
        }
        first = math::checked_add(
            first, count, "LSRN sketch loop index exceeds LP64 limits");
    }

    const auto svd_start = Clock::now();
    math::SvdResult svd = math::thin_svd(sketch_transpose);
    statistics.projected_solve_seconds += elapsed(svd_start);
    const int rank = retained_rank(
        svd, size, matrix.cols(), options);
    math::Matrix factor(matrix.cols(), rank);
    for (int col = 0; col < rank; ++col) {
        const double inverse = 1.0 / svd.singular_values[col];
        for (int row = 0; row < matrix.cols(); ++row) {
            factor(row, col) = svd.u(row, col) * inverse;
        }
    }
    return {true, std::move(factor), rank, size};
}

Preconditioner build_wide_preconditioner(
    const MatrixOperator& matrix,
    const LsrnOptions& options,
    RunStatistics& statistics) {
    const bool ridge = options.regularization > 0.0;
    const double regularization_root = std::sqrt(options.regularization);
    const int random_rows = ridge
        ? checked_system_dimension(matrix.cols(), matrix.rows())
        : matrix.cols();
    const int size = sketch_size(options.oversampling, matrix.rows());
    const int block_size = options.sketch_block_size > 0
        ? std::min(options.sketch_block_size, size)
        : size;
    // Materialize D G in column blocks and form
    // M = U-tilde Sigma-tilde^{-1}.  Ordinary least squares uses D=A;
    // ridge uses the equivalent underdetermined system
    // D=[A, sqrt(lambda) I].
    math::Matrix sketch(matrix.rows(), size);

    for (int first = 0; first < size;) {
        const int count = std::min(block_size, size - first);
        math::Matrix random = gaussian_block(
            random_rows,
            count,
            options,
            static_cast<std::uint64_t>(first),
            statistics);
        math::Matrix top;
        const double* sketch_input = random.data();
        if (ridge) {
            top = math::Matrix(matrix.cols(), count);
            for (int col = 0; col < count; ++col) {
                for (int row = 0; row < matrix.cols(); ++row) {
                    top(row, col) = random(row, col);
                }
            }
            sketch_input = top.data();
        }
        math::Matrix product(matrix.rows(), count);
        apply_original(
            matrix,
            sketch_input,
            count,
            product.data(),
            ProductPhase::sketch,
            statistics);
        if (ridge) {
            for (int col = 0; col < count; ++col) {
                for (int row = 0; row < matrix.rows(); ++row) {
                    product(row, col) = std::fma(
                        regularization_root,
                        random(matrix.cols() + row, col),
                        product(row, col));
                }
            }
        }
        for (int col = 0; col < count; ++col) {
            for (int row = 0; row < matrix.rows(); ++row) {
                sketch(row, first + col) = product(row, col);
            }
        }
        first = math::checked_add(
            first, count, "LSRN sketch loop index exceeds LP64 limits");
    }

    const auto svd_start = Clock::now();
    math::SvdResult svd = math::thin_svd(sketch);
    statistics.projected_solve_seconds += elapsed(svd_start);
    const int rank = retained_rank(
        svd, matrix.rows(), size, options);
    math::Matrix factor(matrix.rows(), rank);
    for (int col = 0; col < rank; ++col) {
        const double inverse = 1.0 / svd.singular_values[col];
        for (int row = 0; row < matrix.rows(); ++row) {
            factor(row, col) = svd.u(row, col) * inverse;
        }
    }
    return {false, std::move(factor), rank, size};
}

class PreconditionedSystem {
public:
    PreconditionedSystem(
        const MatrixOperator& matrix,
        double regularization,
        Preconditioner preconditioner,
        RunStatistics& statistics)
        : matrix_(matrix),
          regularization_(regularization),
          regularization_root_(std::sqrt(regularization)),
          preconditioner_(std::move(preconditioner)),
          original_workspace_(
              preconditioner_.tall ? matrix.cols() : matrix.rows()),
          statistics_(statistics) {}

    int rows() const {
        if (preconditioner_.tall) {
            return regularization_ > 0.0
                ? checked_system_dimension(matrix_.rows(), matrix_.cols())
                : matrix_.rows();
        }
        return preconditioner_.rank;
    }

    int cols() const {
        if (preconditioner_.tall) {
            return preconditioner_.rank;
        }
        return regularization_ > 0.0
            ? checked_system_dimension(matrix_.cols(), matrix_.rows())
            : matrix_.cols();
    }

    int rank() const { return preconditioner_.rank; }
    int sketch_size() const { return preconditioner_.sketch_size; }

    std::vector<double> transformed_rhs(const double* right_hand_side) const {
        if (preconditioner_.tall) {
            std::vector<double> result(rows(), 0.0);
            std::copy(
                right_hand_side,
                right_hand_side + matrix_.rows(),
                result.begin());
            return result;
        }
        std::vector<double> result(preconditioner_.rank);
        transpose_matrix_vector_product(
            preconditioner_.factor,
            right_hand_side,
            result.data());
        return result;
    }

    void original_candidate(
        const std::vector<double>& inner_solution,
        std::vector<double>& result) const {
        result.resize(matrix_.cols());
        if (preconditioner_.tall) {
            matrix_vector_product(
                preconditioner_.factor,
                inner_solution.data(),
                result.data());
        } else {
            std::copy_n(inner_solution.begin(), matrix_.cols(), result.begin());
        }
    }

    void apply(const double* input, double* output) {
        if (preconditioner_.tall) {
            matrix_vector_product(
                preconditioner_.factor,
                input,
                original_workspace_.data());
            apply_original(
                matrix_,
                original_workspace_.data(),
                1,
                output,
                ProductPhase::iteration,
                statistics_);
            if (regularization_ > 0.0) {
                for (int col = 0; col < matrix_.cols(); ++col) {
                    output[matrix_.rows() + col] =
                        regularization_root_ * original_workspace_[col];
                }
            }
            return;
        }

        apply_original(
            matrix_,
            input,
            1,
            original_workspace_.data(),
            ProductPhase::iteration,
            statistics_);
        if (regularization_ > 0.0) {
            for (int row = 0; row < matrix_.rows(); ++row) {
                original_workspace_[row] = std::fma(
                    regularization_root_,
                    input[matrix_.cols() + row],
                    original_workspace_[row]);
            }
        }
        transpose_matrix_vector_product(
            preconditioner_.factor,
            original_workspace_.data(),
            output);
    }

    void apply_transpose(const double* input, double* output) {
        if (preconditioner_.tall) {
            apply_original_transpose(
                matrix_,
                input,
                1,
                original_workspace_.data(),
                ProductPhase::iteration,
                statistics_);
            if (regularization_ > 0.0) {
                for (int col = 0; col < matrix_.cols(); ++col) {
                    original_workspace_[col] = std::fma(
                        regularization_root_,
                        input[matrix_.rows() + col],
                        original_workspace_[col]);
                }
            }
            transpose_matrix_vector_product(
                preconditioner_.factor,
                original_workspace_.data(),
                output);
            return;
        }

        matrix_vector_product(
            preconditioner_.factor,
            input,
            original_workspace_.data());
        apply_original_transpose(
            matrix_,
            original_workspace_.data(),
            1,
            output,
            ProductPhase::iteration,
            statistics_);
        if (regularization_ > 0.0) {
            for (int row = 0; row < matrix_.rows(); ++row) {
                output[matrix_.cols() + row] =
                    regularization_root_ * original_workspace_[row];
            }
        }
    }

private:
    const MatrixOperator& matrix_;
    double regularization_;
    double regularization_root_;
    Preconditioner preconditioner_;
    std::vector<double> original_workspace_;
    RunStatistics& statistics_;
};

int predicted_iteration_limit(
    int sketch_size_value,
    int rank,
    double tolerance) {
    if (rank <= 0) {
        return 0;
    }
    const double ratio = static_cast<double>(sketch_size_value) / rank;
    if (!(ratio > 1.0) || !std::isfinite(ratio)) {
        return rank;
    }
    // This is the paper's practical engineering rule obtained by dropping
    // the probability-tail parameter.  It is a work limit, not a theorem or
    // an accuracy certificate; only validate_original_candidate can succeed.
    const double estimate =
        2.0 * (std::log(2.0) - std::log(tolerance)) / std::log(ratio);
    if (!std::isfinite(estimate) ||
        estimate >= std::numeric_limits<int>::max()) {
        return std::numeric_limits<int>::max();
    }
    return std::max(1, static_cast<int>(std::ceil(estimate)));
}

LeastSquaresResult validate_inner_candidate(
    const MatrixOperator& matrix,
    const double* right_hand_side,
    const LsrnOptions& options,
    const PreconditionedSystem& system,
    const std::vector<double>& inner_solution,
    std::vector<double>& original_solution_workspace,
    int iteration,
    RunStatistics& statistics) {
    system.original_candidate(inner_solution, original_solution_workspace);
    LeastSquaresResult result = math::validate_original_candidate(
        matrix,
        right_hand_side,
        std::span<const double>(original_solution_workspace),
        options.regularization,
        options.tolerance,
        0.0,
        statistics);
    result.iterations = iteration;
    result.depth = iteration;
    result.auxiliary_width = system.sketch_size();
    result.basis_rank = system.rank();
    result.trace.push_back(math::make_iteration_record(
        result, 0, iteration, system.sketch_size()));
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

void validate_lsrn_options(const LsrnOptions& options) {
    if (!std::isfinite(options.oversampling) || options.oversampling <= 1.0) {
        throw std::invalid_argument("oversampling must be finite and greater than one");
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
    if (options.maximum_iterations < 0 || options.sketch_block_size < 0) {
        throw std::invalid_argument("LSRN work limits must be nonnegative");
    }
}

}  // namespace

LeastSquaresResult solve_lsrn(
    const MatrixOperator& matrix,
    const double* right_hand_side,
    const LsrnOptions& options) {
    math::validate_common_inputs(
        matrix, right_hand_side, options.regularization, options.tolerance);
    validate_lsrn_options(options);

    const auto total_start = Clock::now();
    RunStatistics statistics;
    Preconditioner preconditioner = matrix.rows() >= matrix.cols()
        ? build_tall_preconditioner(
              matrix, options, statistics)
        : build_wide_preconditioner(
              matrix, options, statistics);
    PreconditionedSystem system(
        matrix,
        options.regularization,
        std::move(preconditioner),
        statistics);

    std::vector<double> inner_solution(system.cols(), 0.0);
    std::vector<double> original_solution_workspace(matrix.cols());
    std::vector<IterationRecord> trace;
    LeastSquaresResult result;
    int committed_iteration = 0;
    math::CertificateCheckpointSchedule checkpoint_schedule(
        options.regularization, options.tolerance);
    const auto evaluate_current_candidate = [&](int iteration) {
        result = validate_inner_candidate(
            matrix,
            right_hand_side,
            options,
            system,
            inner_solution,
            original_solution_workspace,
            iteration,
            statistics);
        checkpoint_schedule.record_evaluation(iteration, result);
        trace.push_back(result.trace.front());
        result.trace.clear();
    };
    const auto finish_now = [&] {
        finish_result(result, std::move(trace), statistics, total_start);
    };

    if (system.rank() == 0) {
        evaluate_current_candidate(0);
        if (math::candidate_validation_failed_numerically(result) ||
            result.status == SolverStatus::success) {
            finish_now();
            return result;
        }
        if (system.rank() == 0) {
            result.status = SolverStatus::work_limit;
            result.stop_reason = StopReason::exhausted_search_space;
            finish_now();
            return result;
        }
    }

    std::vector<double> u = system.transformed_rhs(right_hand_side);
    double beta = math::vector_norm(u.data(), system.rows());
    if (!(beta > 0.0) || !std::isfinite(beta)) {
        evaluate_current_candidate(0);
        if (math::candidate_validation_failed_numerically(result) ||
            result.status == SolverStatus::success) {
            finish_now();
            return result;
        }
        result.status = std::isfinite(beta)
            ? SolverStatus::work_limit
            : SolverStatus::numerical_breakdown;
        result.stop_reason = std::isfinite(beta)
            ? StopReason::exhausted_search_space
            : StopReason::numerical_breakdown;
        finish_now();
        return result;
    }
    scale(u, 1.0 / beta);

    std::vector<double> v(system.cols(), 0.0);
    system.apply_transpose(u.data(), v.data());
    double alpha = math::vector_norm(v.data(), system.cols());
    if (!(alpha > 0.0) || !std::isfinite(alpha) ||
        !all_finite(std::span<const double>(v))) {
        evaluate_current_candidate(0);
        if (math::candidate_validation_failed_numerically(result) ||
            result.status == SolverStatus::success) {
            finish_now();
            return result;
        }
        result.status = std::isfinite(alpha)
            ? SolverStatus::work_limit
            : SolverStatus::numerical_breakdown;
        result.stop_reason = std::isfinite(alpha)
            ? StopReason::exhausted_search_space
            : StopReason::numerical_breakdown;
        finish_now();
        return result;
    }
    scale(v, 1.0 / alpha);
    std::vector<double> w = v;
    double rhobar = alpha;
    double phibar = beta;
    const int maximum_iterations = options.maximum_iterations > 0
        ? options.maximum_iterations
        : predicted_iteration_limit(
              system.sketch_size(), system.rank(), options.tolerance);

    std::vector<double> next_u(system.rows());
    std::vector<double> next_v(system.cols());
    std::vector<double> candidate_inner_solution(system.cols());
    std::vector<double> candidate_w(system.cols());
    const auto record_inner_failure = [&] {
        if (trace.empty() ||
            trace.back().depth != committed_iteration) {
            evaluate_current_candidate(committed_iteration);
        }
        if (result.status != SolverStatus::success) {
            result.status = SolverStatus::numerical_breakdown;
            result.stop_reason = StopReason::numerical_breakdown;
        }
    };
    for (int iteration = 1; iteration <= maximum_iterations; ++iteration) {
        system.apply(v.data(), next_u.data());
        for (int row = 0; row < system.rows(); ++row) {
            next_u[row] -= alpha * u[row];
        }
        beta = math::vector_norm(next_u.data(), system.rows());
        if (!std::isfinite(beta) ||
            !all_finite(std::span<const double>(next_u))) {
            record_inner_failure();
            break;
        }
        if (beta > 0.0) {
            scale(next_u, 1.0 / beta);
        }

        // As in LSQR, x_k needs beta_{k+1} but not A^*u_{k+1}.  Delay the
        // transpose product until the current original-coordinate candidate
        // has failed its certificate and another inner step is required.
        const double rho = std::hypot(rhobar, beta);
        if (!(rho > 0.0) || !std::isfinite(rho)) {
            record_inner_failure();
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
            record_inner_failure();
            break;
        }

        candidate_inner_solution = inner_solution;
        for (int col = 0; col < system.cols(); ++col) {
            candidate_inner_solution[col] += update * w[col];
        }
        const bool finite_candidate =
            all_finite(std::span<const double>(candidate_inner_solution)) &&
            all_finite(std::span<const double>(next_u));
        if (!finite_candidate) {
            record_inner_failure();
            break;
        }

        inner_solution.swap(candidate_inner_solution);
        u.swap(next_u);
        committed_iteration = iteration;

        const bool beta_breakdown = beta == 0.0;
        const bool checkpoint =
            checkpoint_schedule.should_evaluate(iteration) ||
            beta_breakdown || iteration == maximum_iterations;
        bool candidate_was_evaluated = false;
        if (checkpoint) {
            evaluate_current_candidate(iteration);
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

        system.apply_transpose(u.data(), next_v.data());
        for (int col = 0; col < system.cols(); ++col) {
            next_v[col] -= beta * v[col];
        }
        const double next_alpha = math::vector_norm(
            next_v.data(), system.cols());
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
            for (int col = 0; col < system.cols(); ++col) {
                candidate_w[col] =
                    next_v[col] - recurrence * w[col];
            }
            continuation_is_finite =
                all_finite(std::span<const double>(candidate_w));
        }

        if (!continuation_is_finite || next_alpha == 0.0) {
            if (!candidate_was_evaluated) {
                evaluate_current_candidate(iteration);
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
