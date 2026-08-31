#include "amfls/lsqr.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

#include "algorithms/mathematics/krylov/certificate_checkpoint_schedule.hpp"
#include "algorithms/mathematics/krylov/krylov_workspace.hpp"
#include "algorithms/mathematics/linalg/blas_lapack.hpp"

namespace amfls {
namespace {

using Clock = std::chrono::steady_clock;

void apply_search(
    const MatrixOperator& matrix,
    const double* input,
    double* output,
    RunStatistics& statistics) {
    const auto start = Clock::now();
    matrix.apply(input, 1, output);
    const double seconds =
        std::chrono::duration<double>(Clock::now() - start).count();
    statistics.a_seconds += seconds;
    ++statistics.a_columns;
    ++statistics.a_block_calls;
    ++statistics.search_a_columns;
    ++statistics.search_a_block_calls;
}

void apply_transpose_search(
    const MatrixOperator& matrix,
    const double* input,
    double* output,
    RunStatistics& statistics) {
    const auto start = Clock::now();
    matrix.apply_transpose(input, 1, output);
    const double seconds =
        std::chrono::duration<double>(Clock::now() - start).count();
    statistics.at_seconds += seconds;
    ++statistics.at_columns;
    ++statistics.at_block_calls;
    ++statistics.search_at_columns;
    ++statistics.search_at_block_calls;
}

void scale(std::vector<double>& values, double factor) {
    for (double& value : values) {
        value *= factor;
    }
}

LeastSquaresResult validate_iterate(
    const MatrixOperator& matrix,
    const double* right_hand_side,
    const std::vector<double>& solution,
    double regularization,
    double tolerance,
    int iteration,
    RunStatistics& statistics) {
    LeastSquaresResult result = math::validate_original_candidate(
        matrix,
        right_hand_side,
        std::span<const double>(solution),
        regularization,
        tolerance,
        0.0,
        statistics);
    result.iterations = iteration;
    result.depth = iteration;
    result.basis_rank = iteration;
    result.trace.push_back(math::make_iteration_record(
        result, 0, iteration, 0));
    return result;
}

}  // namespace

LeastSquaresResult solve_lsqr(
    const MatrixOperator& matrix,
    const double* right_hand_side,
    const LsqrOptions& options) {
    math::validate_common_inputs(
        matrix, right_hand_side, options.regularization, options.tolerance);
    if (options.maximum_iterations < 0) {
        throw std::invalid_argument("maximum_iterations must be nonnegative");
    }
    const int maximum_iterations = options.maximum_iterations > 0
        ? options.maximum_iterations
        : std::min(matrix.rows(), matrix.cols());
    const auto total_start = Clock::now();
    RunStatistics statistics;

    std::vector<double> u(right_hand_side, right_hand_side + matrix.rows());
    double beta = math::vector_norm(u.data(), matrix.rows());
    std::vector<double> x(matrix.cols(), 0.0);
    if (beta > 0.0) {
        scale(u, 1.0 / beta);
    }

    std::vector<double> v(matrix.cols(), 0.0);
    apply_transpose_search(matrix, u.data(), v.data(), statistics);
    double alpha = math::vector_norm(v.data(), matrix.cols());
    if (alpha > 0.0) {
        scale(v, 1.0 / alpha);
    }
    std::vector<double> w = v;
    double rhobar = alpha;
    double phibar = beta;
    const double damp = std::sqrt(options.regularization);
    std::vector<IterationRecord> trace;
    LeastSquaresResult result;
    math::CertificateCheckpointSchedule checkpoint_schedule(
        options.regularization, options.tolerance);
    const auto evaluate_candidate = [&](int iteration) {
        result = validate_iterate(
            matrix,
            right_hand_side,
            x,
            options.regularization,
            options.tolerance,
            iteration,
            statistics);
        checkpoint_schedule.record_evaluation(iteration, result);
        trace.push_back(result.trace.front());
        result.trace.clear();
    };
    const bool validate_zero_candidate =
        !std::isfinite(beta) || !std::isfinite(alpha) || alpha == 0.0;
    if (validate_zero_candidate) {
        evaluate_candidate(0);
        if (math::candidate_validation_failed_numerically(result) ||
            result.status == SolverStatus::success || alpha == 0.0) {
            result.trace = std::move(trace);
            statistics.total_seconds = std::chrono::duration<double>(
                Clock::now() - total_start).count();
            result.statistics = statistics;
            return result;
        }
    }

    std::vector<double> av(matrix.rows());
    std::vector<double> atu(matrix.cols());
    for (int iteration = 1; iteration <= maximum_iterations; ++iteration) {
        apply_search(matrix, v.data(), av.data(), statistics);
        for (int row = 0; row < matrix.rows(); ++row) {
            av[row] -= alpha * u[row];
        }
        beta = math::vector_norm(av.data(), matrix.rows());
        if (beta > 0.0) {
            scale(av, 1.0 / beta);
        }

        // The current LSQR candidate needs beta_{k+1}, but not
        // A^*u_{k+1} or alpha_{k+1}.  Form and certify x_k before preparing
        // the next bidiagonalization step so a successful or terminal
        // checkpoint does not pay for an unused transpose product.
        const double rhobar1 = std::hypot(rhobar, damp);
        const double cs1 = rhobar1 == 0.0 ? 1.0 : rhobar / rhobar1;
        phibar *= cs1;
        const double rho = std::hypot(rhobar1, beta);
        const double cs = rho == 0.0 ? 1.0 : rhobar1 / rho;
        const double sn = rho == 0.0 ? 0.0 : beta / rho;
        const double phi = cs * phibar;
        phibar = sn * phibar;
        bool candidate_recurrence_is_finite =
            std::isfinite(beta) && std::isfinite(rhobar1) &&
            std::isfinite(cs1) && std::isfinite(rho) &&
            std::isfinite(cs) && std::isfinite(sn) &&
            std::isfinite(phi) && std::isfinite(phibar);
        if (rho != 0.0) {
            const double update = phi / rho;
            candidate_recurrence_is_finite =
                candidate_recurrence_is_finite && std::isfinite(update);
            for (int col = 0; col < matrix.cols(); ++col) {
                x[col] += update * w[col];
            }
        }
        candidate_recurrence_is_finite =
            candidate_recurrence_is_finite &&
            std::all_of(x.begin(), x.end(), [](double value) {
                return std::isfinite(value);
            });
        u.swap(av);

        const bool beta_breakdown = beta == 0.0;
        const bool checkpoint =
            checkpoint_schedule.should_evaluate(iteration) ||
            beta_breakdown || !candidate_recurrence_is_finite ||
            iteration == maximum_iterations;
        bool candidate_was_evaluated = false;
        if (checkpoint) {
            evaluate_candidate(iteration);
            candidate_was_evaluated = true;
            if (math::candidate_validation_failed_numerically(result) ||
                result.status == SolverStatus::success) {
                break;
            }
            if (!candidate_recurrence_is_finite) {
                result.status = SolverStatus::numerical_breakdown;
                result.stop_reason = StopReason::numerical_breakdown;
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

        apply_transpose_search(matrix, u.data(), atu.data(), statistics);
        for (int col = 0; col < matrix.cols(); ++col) {
            atu[col] -= beta * v[col];
        }
        const double next_alpha = math::vector_norm(
            atu.data(), matrix.cols());
        bool continuation_is_finite =
            std::isfinite(next_alpha) &&
            std::all_of(atu.begin(), atu.end(), [](double value) {
                return std::isfinite(value);
            });
        if (next_alpha > 0.0 && std::isfinite(next_alpha)) {
            scale(atu, 1.0 / next_alpha);
        }

        const double theta = sn * next_alpha;
        const double next_rhobar = -cs * next_alpha;
        const double recurrence = theta / rho;
        continuation_is_finite = continuation_is_finite &&
            std::isfinite(theta) && std::isfinite(next_rhobar) &&
            std::isfinite(recurrence);
        if (continuation_is_finite && next_alpha > 0.0) {
            for (int col = 0; col < matrix.cols(); ++col) {
                w[col] = atu[col] - recurrence * w[col];
            }
            continuation_is_finite = std::all_of(
                w.begin(), w.end(), [](double value) {
                    return std::isfinite(value);
                });
        }

        if (!continuation_is_finite || next_alpha == 0.0) {
            if (!candidate_was_evaluated) {
                evaluate_candidate(iteration);
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
        v.swap(atu);
    }

    result.trace = std::move(trace);
    statistics.total_seconds = std::chrono::duration<double>(
        Clock::now() - total_start).count();
    result.statistics = statistics;
    return result;
}

}  // namespace amfls
