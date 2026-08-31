#include "amfls/lsmb.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

#include "algorithms/mathematics/krylov/certificate_checkpoint_schedule.hpp"
#include "algorithms/mathematics/krylov/krylov_workspace.hpp"
#include "algorithms/mathematics/linalg/blas_lapack.hpp"
#include "algorithms/lsmb/lsmb_internal.hpp"

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

LeastSquaresResult solve_lsmb(
    const MatrixOperator& matrix,
    const double* right_hand_side,
    const LsmbOptions& options) {
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

    // Hallman--Gu use `lambda` as a damping parameter.  The public AMFLS
    // option is the coefficient of ||x||^2, hence damp^2=regularization.
    const double damp = std::sqrt(options.regularization);
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

    double rho_u = alpha;
    double phi_bar = beta;
    double rho = 1.0;
    double rho_bar = 1.0;
    double c_bar = 1.0;
    double s_bar = 0.0;
    std::vector<double> h = v;
    std::vector<double> h_bar(matrix.cols(), 0.0);
    std::vector<double> x_c(matrix.cols(), 0.0);

    double rho_tilde_u = 1.0;
    double norm_r_base = 0.0;
    double zeta = 0.0;
    double z_bar = alpha * beta;
    double zeta_tilde = 0.0;
    double zeta_hat = 0.0;
    double theta_tilde = 0.0;
    double norm_z_hat = 0.0;
    double c_hat = 1.0;
    double s_hat = 0.0;

    std::vector<IterationRecord> trace;
    LeastSquaresResult result;
    math::CertificateCheckpointSchedule checkpoint_schedule(
        options.regularization, options.tolerance);
    bool initial_numerical_breakdown =
        !std::isfinite(beta) || !std::isfinite(alpha) ||
        !std::isfinite(z_bar);
    const bool validate_zero_candidate =
        initial_numerical_breakdown || alpha == 0.0;
    if (validate_zero_candidate) {
        result = validate_iterate(
            matrix,
            right_hand_side,
            x_c,
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
    std::vector<double> next_h_bar(matrix.cols());
    std::vector<double> next_x_c(matrix.cols());
    std::vector<double> next_h(matrix.cols());
    std::vector<double> candidate(matrix.cols());
    std::vector<double> corrected_candidate(matrix.cols());
    for (int iteration = 1; iteration <= maximum_iterations; ++iteration) {
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

        // A nonbreakdown LSMB bridge point x_k depends on alpha_{k+1}, so
        // this transpose half-step belongs to the current iteration.  At
        // exact beta breakdown, u_{k+1}=0 and linearity gives
        // A^*u_{k+1}=0.  Infer alpha_{k+1}=0 without an unnecessary callback.
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

        const SymmetricGivens regularization_rotation =
            symmetric_givens(rho_u, damp);
        norm_r_base = std::hypot(
            norm_r_base,
            regularization_rotation.sine * phi_bar);
        phi_bar *= regularization_rotation.cosine;

        const double rho_old = rho;
        const SymmetricGivens bidiagonal_rotation = symmetric_givens(
            regularization_rotation.radius, beta);
        rho = bidiagonal_rotation.radius;
        const double theta = bidiagonal_rotation.sine * alpha;
        rho_u = bidiagonal_rotation.cosine * alpha;
        const double phi = bidiagonal_rotation.cosine * phi_bar;
        phi_bar = -bidiagonal_rotation.sine * phi_bar;

        const double rho_bar_old = rho_bar;
        const double theta_bar = s_bar * rho;
        const double rho_bar_u = c_bar * rho;
        const SymmetricGivens minres_rotation =
            symmetric_givens(rho_bar_u, theta);
        c_bar = minres_rotation.cosine;
        s_bar = minres_rotation.sine;
        rho_bar = minres_rotation.radius;
        const double zeta_old = zeta;
        zeta = c_bar * z_bar;
        z_bar = -s_bar * z_bar;
        const double theta_wide_hat = s_bar * rho_u;
        const double rho_wide_hat = c_bar * rho_u;

        const double h_bar_denominator = rho_old * rho_bar_old;
        recurrence_is_valid = recurrence_is_valid &&
            std::isfinite(norm_r_base) &&
            std::isfinite(regularization_rotation.cosine) &&
            std::isfinite(regularization_rotation.sine) &&
            std::isfinite(rho) && std::isfinite(theta) &&
            std::isfinite(rho_u) && std::isfinite(phi) &&
            std::isfinite(phi_bar) && std::isfinite(theta_bar) &&
            std::isfinite(rho_bar_u) && std::isfinite(c_bar) &&
            std::isfinite(s_bar) && std::isfinite(rho_bar) &&
            std::isfinite(zeta) && std::isfinite(z_bar) &&
            std::isfinite(theta_wide_hat) &&
            std::isfinite(rho_wide_hat) &&
            std::isfinite(h_bar_denominator) &&
            h_bar_denominator != 0.0 && rho != 0.0;

        if (recurrence_is_valid) {
            const double h_bar_recurrence =
                theta_bar * rho / h_bar_denominator;
            const double solution_update = phi / rho;
            const double h_recurrence = theta / rho;
            recurrence_is_valid =
                std::isfinite(h_bar_recurrence) &&
                std::isfinite(solution_update) &&
                std::isfinite(h_recurrence);
            next_x_c = x_c;
            for (int col = 0; col < matrix.cols(); ++col) {
                next_h_bar[col] =
                    h[col] - h_bar_recurrence * h_bar[col];
                next_x_c[col] += solution_update * h[col];
                next_h[col] = atu[col] - h_recurrence * h[col];
            }
            recurrence_is_valid = recurrence_is_valid &&
                all_finite(next_h_bar) && all_finite(next_x_c) &&
                all_finite(next_h);
            if (recurrence_is_valid) {
                h_bar.swap(next_h_bar);
                x_c.swap(next_x_c);
                h.swap(next_h);
            }
        }
        u.swap(av);
        v.swap(atu);

        candidate = x_c;
        const bool exact_search_breakdown = alpha == 0.0 || beta == 0.0;
        if (recurrence_is_valid && !exact_search_breakdown) {
            // Primary experiments fix Hallman--Gu's native parameters to
            // tau=Inf and sigma_est=0.  The latter makes c_tilde=1 before
            // the implicit ridge row is included.
            const double rho_circ = std::hypot(rho_u, damp);
            const double c_error = rho_circ == 0.0
                ? 0.0
                : rho_u / rho_circ;

            const double theta_tilde_old = theta_tilde;
            const SymmetricGivens residual_rotation =
                symmetric_givens(rho_tilde_u, theta_bar);
            const double rho_tilde = residual_rotation.radius;
            theta_tilde = residual_rotation.sine * rho_bar;
            rho_tilde_u = residual_rotation.cosine * rho_bar;

            recurrence_is_valid = rho_tilde != 0.0 &&
                rho_tilde_u != 0.0 && std::isfinite(rho_circ) &&
                std::isfinite(c_error) &&
                std::isfinite(theta_tilde) &&
                std::isfinite(rho_tilde_u);
            double q_residual = 0.0;
            if (recurrence_is_valid) {
                q_residual =
                    phi_bar * theta_wide_hat / rho_tilde_u;
                recurrence_is_valid = std::isfinite(q_residual);
            }

            const double rho_hat_u = c_hat * rho_tilde;
            const double theta_hat = s_hat * rho_tilde;
            const SymmetricGivens solution_rotation =
                symmetric_givens(rho_hat_u, theta_tilde);
            c_hat = solution_rotation.cosine;
            s_hat = solution_rotation.sine;
            const double rho_hat = solution_rotation.radius;

            const double zeta_tilde_old = zeta_tilde;
            double zeta_tilde_u = 0.0;
            if (recurrence_is_valid && rho_hat != 0.0) {
                zeta_tilde =
                    (zeta_old - theta_tilde_old * zeta_tilde_old) /
                    rho_tilde;
                zeta_tilde_u =
                    (zeta - theta_tilde * zeta_tilde) / rho_tilde_u;
            } else {
                recurrence_is_valid = false;
            }

            const SymmetricGivens ddots_rotation = symmetric_givens(
                rho_hat_u,
                residual_rotation.sine * rho_bar_u);
            const double rho_ddots = ddots_rotation.radius;
            const double theta_ddots =
                ddots_rotation.sine * residual_rotation.cosine *
                rho_bar_u;
            const double rho_ddots_u =
                ddots_rotation.cosine * residual_rotation.cosine *
                rho_bar_u;
            const double zeta_hat_old = zeta_hat;
            double zeta_hat_u = 0.0;
            double zeta_hat_uu = 0.0;
            if (recurrence_is_valid && rho_ddots != 0.0 &&
                rho_ddots_u != 0.0) {
                zeta_hat =
                    (zeta_tilde - theta_hat * zeta_hat_old) / rho_hat;
                zeta_hat_u =
                    (zeta_tilde - theta_hat * zeta_hat_old) /
                    rho_ddots;
                zeta_hat_uu =
                    (zeta_tilde_u - theta_ddots * zeta_hat_u) /
                    rho_ddots_u;
                norm_z_hat = std::hypot(norm_z_hat, zeta_hat_old);
            } else {
                recurrence_is_valid = false;
            }
            const double norm_x_base =
                std::hypot(norm_z_hat, zeta_hat_u);

            // rho_wide_hat/c_error equals c_bar*rho_circ, avoiding 0/0.
            const double rho_error = c_bar * rho_circ;
            const double d1 = q_residual;
            const double d2 = rho_error / rho_ddots_u;
            const double d1_d2 = d1 * d2;
            const double d3 = zeta_hat_uu * rho_error - d1_d2;
            const double d4 = norm_x_base * rho_error;
            const double upper_endpoint_term = zeta_hat_uu * rho_error;
            const double norm_r_c = std::hypot(norm_r_base, phi_bar);
            // All polynomial coefficients are homogeneous of degree two in
            // (d1, d1*d2, d3, d4, ||r_C||).  Normalize those primitives
            // before squaring so changing the scale of b neither underflows
            // p(0) nor overflows an otherwise identical bridge equation.
            const double polynomial_scale = std::max({
                std::abs(d1),
                std::abs(d1_d2),
                std::abs(d3),
                std::abs(d4),
                norm_r_c});
            const double scaled_d1 = polynomial_scale == 0.0
                ? 0.0
                : d1 / polynomial_scale;
            const double scaled_d1_d2 = polynomial_scale == 0.0
                ? 0.0
                : d1_d2 / polynomial_scale;
            const double scaled_d3 = polynomial_scale == 0.0
                ? 0.0
                : d3 / polynomial_scale;
            const double scaled_d4 = polynomial_scale == 0.0
                ? 0.0
                : d4 / polynomial_scale;
            const double scaled_norm_r_c = polynomial_scale == 0.0
                ? 0.0
                : norm_r_c / polynomial_scale;
            const double scaled_upper_endpoint_term =
                polynomial_scale == 0.0
                ? 0.0
                : upper_endpoint_term / polynomial_scale;
            const double cubic =
                scaled_d1 * scaled_d1 +
                scaled_d1_d2 * scaled_d1_d2;
            const double quadratic =
                2.0 * scaled_d1_d2 * scaled_d3 -
                scaled_d1 * scaled_d1;
            // tau_native=Inf makes the source term
            // (rho_error/tau_native)^2 exactly zero.
            const double linear =
                scaled_d4 * scaled_d4 + scaled_d3 * scaled_d3 +
                scaled_norm_r_c * scaled_norm_r_c;
            const double constant =
                -scaled_norm_r_c * scaled_norm_r_c;
            const double upper_endpoint_value =
                scaled_upper_endpoint_term *
                    scaled_upper_endpoint_term +
                scaled_d4 * scaled_d4;
            const bool lower_endpoint_is_root = norm_r_c == 0.0;
            const bool upper_endpoint_is_root =
                rho_error == 0.0 ||
                (zeta_hat_uu == 0.0 && norm_x_base == 0.0);
            double gamma = 0.0;
            recurrence_is_valid = recurrence_is_valid &&
                std::isfinite(zeta_tilde) &&
                std::isfinite(zeta_tilde_u) &&
                std::isfinite(zeta_hat) &&
                std::isfinite(zeta_hat_u) &&
                std::isfinite(zeta_hat_uu) &&
                std::isfinite(norm_z_hat) &&
                std::isfinite(norm_x_base) &&
                std::isfinite(rho_error) && std::isfinite(d2) &&
                std::isfinite(d1_d2) && std::isfinite(d3) &&
                std::isfinite(d4) &&
                std::isfinite(upper_endpoint_term) &&
                std::isfinite(polynomial_scale) &&
                detail::solve_lsmb_gamma(
                    cubic,
                    quadratic,
                    linear,
                    constant,
                    upper_endpoint_value,
                    lower_endpoint_is_root,
                    upper_endpoint_is_root,
                    gamma);

            const double correction_denominator = rho * rho_bar;
            if (recurrence_is_valid && correction_denominator != 0.0 &&
                std::isfinite(correction_denominator)) {
                const double correction =
                    gamma * phi_bar * theta_wide_hat /
                    correction_denominator;
                recurrence_is_valid = std::isfinite(correction);
                if (recurrence_is_valid) {
                    corrected_candidate = candidate;
                    for (int col = 0; col < matrix.cols(); ++col) {
                        corrected_candidate[col] +=
                            correction * h_bar[col];
                    }
                    recurrence_is_valid = all_finite(corrected_candidate);
                    if (recurrence_is_valid) {
                        candidate.swap(corrected_candidate);
                    }
                }
            } else {
                recurrence_is_valid = false;
            }
        }

        const bool checkpoint =
            checkpoint_schedule.should_evaluate(iteration) ||
            !recurrence_is_valid || exact_search_breakdown ||
            iteration == maximum_iterations;
        if (!checkpoint) {
            continue;
        }

        result = validate_iterate(
            matrix,
            right_hand_side,
            candidate,
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
