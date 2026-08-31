#include "amfls/lsmr.hpp"

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

struct SymmetricGivens {
    double cosine;
    double sine;
    double radius;
};

SymmetricGivens symmetric_givens(double first, double second) {
    const double radius = std::hypot(first, second);
    if (radius == 0.0) {
        return {1.0, 0.0, 0.0};
    }
    return {first / radius, second / radius, radius};
}

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

bool all_finite(const std::vector<double>& values) {
    return std::all_of(
        values.begin(), values.end(), [](double value) {
            return std::isfinite(value);
        });
}

void mark_numerical_breakdown(LeastSquaresResult& result) {
    result.status = SolverStatus::numerical_breakdown;
    result.stop_reason = StopReason::numerical_breakdown;
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

LeastSquaresResult solve_lsmr(
    const MatrixOperator& matrix,
    const double* right_hand_side,
    const LsmrOptions& options) {
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

    // Fong--Saunders (2011), section 5.2, with their regularization
    // parameter `damp` satisfying damp^2 = options.regularization.
    std::vector<double> u(right_hand_side, right_hand_side + matrix.rows());
    double beta = math::vector_norm(u.data(), matrix.rows());
    if (beta > 0.0 && std::isfinite(beta)) {
        scale(u, 1.0 / beta);
    }

    std::vector<double> v(matrix.cols(), 0.0);
    apply_transpose_search(matrix, u.data(), v.data(), statistics);
    double alpha = math::vector_norm(v.data(), matrix.cols());
    if (alpha > 0.0 && std::isfinite(alpha)) {
        scale(v, 1.0 / alpha);
    }

    std::vector<double> x(matrix.cols(), 0.0);
    std::vector<double> h = v;
    std::vector<double> hbar(matrix.cols(), 0.0);
    double alphabar = alpha;
    double zetabar = alpha * beta;
    double rho = 1.0;
    double rhobar = 1.0;
    double cbar = 1.0;
    double sbar = 0.0;
    const double damp = std::sqrt(options.regularization);

    std::vector<IterationRecord> trace;
    LeastSquaresResult result;
    math::CertificateCheckpointSchedule checkpoint_schedule(
        options.regularization, options.tolerance);
    bool initial_numerical_breakdown =
        !std::isfinite(beta) || !std::isfinite(alpha);
    const bool validate_zero_candidate =
        initial_numerical_breakdown || alpha == 0.0;
    if (validate_zero_candidate) {
        result = validate_iterate(
            matrix,
            right_hand_side,
            x,
            options.regularization,
            options.tolerance,
            0,
            statistics);
        checkpoint_schedule.record_evaluation(0, result);
        trace.push_back(result.trace.front());
        initial_numerical_breakdown = initial_numerical_breakdown ||
            math::candidate_validation_failed_numerically(result);
        if (initial_numerical_breakdown) {
            mark_numerical_breakdown(result);
        } else if (alpha == 0.0 && result.status != SolverStatus::success) {
            // Under the fixed-linear-adjoint operator contract this branch is
            // unreachable: alpha=0 is exact stationarity at x=0.  Keep an
            // explicit failure reason if a stateful or inconsistent callback
            // makes the independent validation disagree.
            result.status = SolverStatus::work_limit;
            result.stop_reason = StopReason::exhausted_search_space;
        }
        if (result.status == SolverStatus::success ||
            initial_numerical_breakdown || alpha == 0.0) {
            result.trace = std::move(trace);
            statistics.total_seconds = std::chrono::duration<double>(
                Clock::now() - total_start).count();
            result.statistics = statistics;
            return result;
        }
    }

    std::vector<double> av(matrix.rows());
    std::vector<double> atu(matrix.cols());
    std::vector<double> next_hbar(matrix.cols());
    std::vector<double> next_x(matrix.cols());
    std::vector<double> next_h(matrix.cols());
    for (int iteration = 1; iteration <= maximum_iterations; ++iteration) {
        // Continue the Golub--Kahan bidiagonalization using only A and A*.
        apply_search(matrix, v.data(), av.data(), statistics);
        bool recurrence_is_valid = all_finite(av);
        for (int row = 0; row < matrix.rows(); ++row) {
            av[row] -= alpha * u[row];
        }
        beta = math::vector_norm(av.data(), matrix.rows());
        recurrence_is_valid = recurrence_is_valid && std::isfinite(beta);
        if (beta > 0.0 && std::isfinite(beta)) {
            scale(av, 1.0 / beta);
        }

        // A nonbreakdown LSMR candidate x_k depends on alpha_{k+1}, so this
        // transpose half-step belongs to the current iteration rather than
        // to an optional continuation.  At exact beta breakdown, however,
        // u_{k+1}=0 and linearity gives A^*u_{k+1}=0.  Infer alpha_{k+1}=0
        // without paying for a callback whose result is already known.
        if (beta == 0.0) {
            std::fill(atu.begin(), atu.end(), 0.0);
            alpha = 0.0;
        } else {
            apply_transpose_search(
                matrix, av.data(), atu.data(), statistics);
            for (int col = 0; col < matrix.cols(); ++col) {
                atu[col] -= beta * v[col];
            }
            alpha = math::vector_norm(atu.data(), matrix.cols());
            recurrence_is_valid =
                recurrence_is_valid && std::isfinite(alpha);
            if (alpha > 0.0 && std::isfinite(alpha)) {
                scale(atu, 1.0 / alpha);
            }
        }

        // Apply the damping rotation, the QR rotation for B_k, and the
        // MINRES rotation for R_k^T exactly in the order of section 5.2.
        const SymmetricGivens damping_rotation =
            symmetric_givens(alphabar, damp);
        const double rho_old = rho;
        const SymmetricGivens bidiagonal_rotation =
            symmetric_givens(damping_rotation.radius, beta);
        rho = bidiagonal_rotation.radius;
        const double theta_new = bidiagonal_rotation.sine * alpha;
        alphabar = bidiagonal_rotation.cosine * alpha;

        const double rhobar_old = rhobar;
        const double theta_bar = sbar * rho;
        const double rho_temporary = cbar * rho;
        const SymmetricGivens minres_rotation =
            symmetric_givens(rho_temporary, theta_new);
        cbar = minres_rotation.cosine;
        sbar = minres_rotation.sine;
        rhobar = minres_rotation.radius;
        const double zeta = cbar * zetabar;
        zetabar = -sbar * zetabar;

        const double hbar_denominator = rho_old * rhobar_old;
        const double solution_denominator = rho * rhobar;
        recurrence_is_valid = recurrence_is_valid &&
            std::isfinite(damping_rotation.cosine) &&
            std::isfinite(damping_rotation.sine) &&
            std::isfinite(damping_rotation.radius) &&
            std::isfinite(bidiagonal_rotation.cosine) &&
            std::isfinite(bidiagonal_rotation.sine) &&
            std::isfinite(rho) && std::isfinite(theta_new) &&
            std::isfinite(alphabar) && std::isfinite(theta_bar) &&
            std::isfinite(rho_temporary) &&
            std::isfinite(minres_rotation.cosine) &&
            std::isfinite(minres_rotation.sine) &&
            std::isfinite(rhobar) && std::isfinite(zeta) &&
            std::isfinite(zetabar) &&
            std::isfinite(hbar_denominator) &&
            std::isfinite(solution_denominator) &&
            hbar_denominator != 0.0 && solution_denominator != 0.0 &&
            rho != 0.0;
        if (recurrence_is_valid) {
            const double hbar_recurrence =
                theta_bar * rho / hbar_denominator;
            const double solution_update = zeta / solution_denominator;
            const double h_recurrence = theta_new / rho;
            recurrence_is_valid =
                std::isfinite(hbar_recurrence) &&
                std::isfinite(solution_update) &&
                std::isfinite(h_recurrence);
            next_x = x;
            for (int col = 0; col < matrix.cols(); ++col) {
                next_hbar[col] =
                    h[col] - hbar_recurrence * hbar[col];
                next_x[col] += solution_update * next_hbar[col];
                next_h[col] = atu[col] - h_recurrence * h[col];
            }
            recurrence_is_valid = recurrence_is_valid &&
                all_finite(next_hbar) && all_finite(next_x) &&
                all_finite(next_h);
            if (recurrence_is_valid) {
                hbar.swap(next_hbar);
                x.swap(next_x);
                h.swap(next_h);
            }
        }
        u.swap(av);
        v.swap(atu);

        const bool exact_search_breakdown = alpha == 0.0 || beta == 0.0;
        const bool checkpoint =
            checkpoint_schedule.should_evaluate(iteration) ||
            !recurrence_is_valid || exact_search_breakdown ||
            iteration == maximum_iterations;
        if (!checkpoint) {
            continue;
        }

        // Unlike the source implementation's recurrence-based estimates,
        // the AMFLS baseline protocol explicitly recomputes both residuals.
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
        if (math::candidate_validation_failed_numerically(result)) {
            break;
        }
        if (!recurrence_is_valid) {
            mark_numerical_breakdown(result);
            break;
        }
        if (result.status == SolverStatus::success) {
            break;
        }
        if (exact_search_breakdown) {
            result.status = SolverStatus::work_limit;
            result.stop_reason = StopReason::exhausted_search_space;
            break;
        }
        if (iteration == maximum_iterations) {
            result.status = SolverStatus::work_limit;
            result.stop_reason = StopReason::maximum_depth;
        }
    }

    result.trace = std::move(trace);
    statistics.total_seconds = std::chrono::duration<double>(
        Clock::now() - total_start).count();
    result.statistics = statistics;
    return result;
}

}  // namespace amfls
