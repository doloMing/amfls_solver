#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "amfls/aplicur.hpp"
#include "algorithms/mathematics/krylov/certificate_checkpoint_schedule.hpp"
#include "test_helpers.hpp"

namespace {

void require_solution(
    const amfls::LeastSquaresResult& result,
    const std::vector<double>& expected,
    double tolerance,
    const std::string& label) {
    require_test(
        result.status == amfls::SolverStatus::success,
        label + " must satisfy the original-problem accuracy contract");
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
    double regularization,
    double tolerance,
    const std::string& label) {
    const auto& statistics = result.statistics;
    require_test(
        statistics.search_a_columns == statistics.iterative_a_columns &&
            statistics.search_at_columns == statistics.iterative_at_columns &&
            statistics.sketch_a_columns == 0 &&
            statistics.sketch_at_columns == 0,
        label + " explicit CUR work is not reported as operator callbacks");
    require_test(
        statistics.iterative_a_columns == result.iterations + result.depth &&
            statistics.iterative_a_block_calls ==
                result.iterations + result.depth &&
            statistics.iterative_at_columns == result.iterations &&
            statistics.iterative_at_block_calls == result.iterations,
        label + " delayed PLSQR continuation accounting");
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
        statistics.gaussian_random_columns > 0 &&
            statistics.gaussian_random_values > 0,
        label + " randomized spectral-error probes");
    require_test(!result.trace.empty(), label + " iteration trace");
    require_test(
        result.trace.back().depth == result.iterations,
        label + " terminal candidate validation");
    for (std::size_t index = 1; index < result.trace.size(); ++index) {
        require_test(
            result.trace[index - 1].depth < result.trace[index].depth,
            label + " checkpoint iterations are strictly increasing");
    }
    const long long checkpoint_count =
        static_cast<long long>(result.trace.size());
    require_test(
        statistics.base_validation_a_columns ==
                statistics.base_validation_at_columns &&
            statistics.base_validation_a_columns ==
                statistics.base_validation_a_block_calls &&
            statistics.base_validation_a_columns ==
                statistics.base_validation_at_block_calls,
        label + " paired base-validation operator accounting");
    require_test(
        statistics.base_validation_a_columns >= checkpoint_count,
        label + " at least one base validation per checkpoint");
    const long long validation_refinement_count =
        statistics.base_validation_a_columns - checkpoint_count;
    require_test(
        validation_refinement_count <=
            (regularization > 0.0 ? checkpoint_count : 0),
        label + " at most one ridge validation refinement per checkpoint");

    amfls::math::CertificateCheckpointSchedule schedule(
        regularization, tolerance);
    int previous_depth = -1;
    int previous_epoch = 0;
    for (std::size_t index = 0; index < result.trace.size(); ++index) {
        const auto& record = result.trace[index];
        require_test(
            record.epoch >= previous_epoch,
            label + " phase indices are monotone");
        for (int depth = previous_depth + 1; depth < record.depth; ++depth) {
            require_test(
                !schedule.should_evaluate(depth),
                label + " shared schedule cannot skip a due checkpoint");
        }
        const bool phase_terminal =
            index + 1 == result.trace.size() ||
            result.trace[index + 1].epoch != record.epoch;
        require_test(
            schedule.should_evaluate(record.depth) || phase_terminal,
            label + " checkpoint is scheduled or terminates a phase");
        amfls::LeastSquaresResult certificate;
        certificate.compatible_backward_error_upper_bound =
            record.compatible_backward_error_upper_bound;
        certificate.least_squares_backward_error_upper_bound =
            record.least_squares_backward_error_upper_bound;
        certificate.relative_energy_error_upper_bound =
            record.relative_energy_error_upper_bound;
        schedule.record_evaluation(record.depth, certificate);
        previous_depth = record.depth;
        previous_epoch = record.epoch;
    }
    require_test(
        result.trace.back().epoch == result.depth,
        label + " terminal checkpoint phase");
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
    // Row-major 6-by-3 matrix with mutually orthogonal columns.  Hence the
    // regularized reference solution is available coordinate by coordinate.
    const std::vector<double> matrix{
        4.0, 0.0, 0.0,
        0.0, 2.0, 0.0,
        0.0, 0.0, 0.5,
        1.0, 0.0, 0.0,
        0.0, -1.0, 0.0,
        0.0, 0.0, 3.0};
    const std::vector<double> right_hand_side{8.0, -2.0, 1.0, 1.0, 3.0, 6.0};
    const std::vector<double> expected{
        1.9400352733686066,
        -1.3972055888223553,
        1.9978401727861772};

    amfls::AplicurOptions options;
    options.regularization = 1e-2;
    options.tolerance = 1e-10;
    options.block_size = 1;
    options.sparse_sign_nonzeros = 2;
    options.spectral_probe_count = 10;
    options.cur_tolerance = 1e-12;
    options.maximum_iterations = 30;
    options.seed = 712;
    const amfls::LeastSquaresResult tall = amfls::solve_aplicur(
        matrix.data(), 6, 3, right_hand_side.data(), options);
    require_solution(tall, expected, 2e-10, "APLICUR tall ridge");
    require_test(
        tall.auxiliary_width == 3 && tall.basis_rank == 3 &&
            tall.iterations > 0 && tall.depth >= 2,
        "APLICUR rank and phase metadata");
    require_accounting(
        tall,
        options.regularization,
        options.tolerance,
        "APLICUR tall ridge");
    // The explicit baseline also exercises the paper's general rectangular
    // formulation on a wide ridge problem.
    const std::vector<double> wide_matrix{
        2.0, 0.0, 0.0,
        0.0, 1.0, 0.0};
    const std::vector<double> wide_rhs{4.0, 3.0};
    const std::vector<double> wide_expected{
        1.9801980198019802,
        2.8846153846153846,
        0.0};
    amfls::AplicurOptions wide_options = options;
    wide_options.regularization = 4e-2;
    wide_options.block_size = 2;
    wide_options.cur_tolerance = 1e-12;
    wide_options.seed = 913;
    const amfls::LeastSquaresResult wide = amfls::solve_aplicur(
        wide_matrix.data(), 2, 3, wide_rhs.data(), wide_options);
    require_solution(wide, wide_expected, 3e-10, "APLICUR wide ridge");
    require_test(
        wide.auxiliary_width == 2 && wide.basis_rank == 2,
        "APLICUR wide CUR rank");

    const std::vector<double> coupled_matrix{
        1.0, 2.0, 0.0,
        0.0, 1.0, 3.0,
        2.0, -1.0, 1.0,
        1.0, 0.0, 1.0,
        -1.0, 2.0, 2.0,
        3.0, 1.0, -2.0,
        0.5, -1.0, 4.0};
    const std::vector<double> coupled_rhs{
        1.0, -2.0, 3.0, 0.5, 4.0, -1.0, 2.0};
    const std::vector<double> coupled_expected{
        0.1714946475059758,
        0.15912324823333435,
        0.45716482850465107};
    amfls::AplicurOptions coupled_options = options;
    coupled_options.regularization = 3e-2;
    coupled_options.seed = 1117;
    const amfls::LeastSquaresResult coupled = amfls::solve_aplicur(
        coupled_matrix.data(),
        7,
        3,
        coupled_rhs.data(),
        coupled_options);
    require_solution(
        coupled,
        coupled_expected,
        3e-10,
        "APLICUR coupled ridge");
    require_accounting(
        coupled,
        coupled_options.regularization,
        coupled_options.tolerance,
        "APLICUR coupled ridge");

    constexpr int multiphase_rows = 12;
    constexpr int multiphase_cols = 6;
    std::vector<double> multiphase_matrix(
        multiphase_rows * multiphase_cols, 0.0);
    const std::vector<double> multiphase_scales{
        10.0, 6.0, 3.0, 1.5, 0.75, 0.25};
    for (int col = 0; col < multiphase_cols; ++col) {
        multiphase_matrix[static_cast<std::size_t>(
            col * multiphase_cols + col)] =
            multiphase_scales[static_cast<std::size_t>(col)];
        multiphase_matrix[static_cast<std::size_t>(
            (multiphase_cols + col) * multiphase_cols + col)] = 1.0;
    }
    const std::vector<double> multiphase_rhs{
        5.0, -4.0, 3.0, -2.0, 1.0, -0.5,
        1.0, -1.0, 2.0, -2.0, 3.0, -3.0};
    amfls::AplicurOptions multiphase_options = options;
    multiphase_options.re_preconditioning_tolerance = 1.000001;
    multiphase_options.cur_tolerance = 1e-14;
    multiphase_options.maximum_iterations = 50;
    multiphase_options.seed = 1429;
    const amfls::LeastSquaresResult multiphase = amfls::solve_aplicur(
        multiphase_matrix.data(),
        multiphase_rows,
        multiphase_cols,
        multiphase_rhs.data(),
        multiphase_options);
    require_solution(
        multiphase,
        {51.0 / 101.01,
         -25.0 / 37.01,
         11.0 / 10.01,
         -5.0 / 3.26,
         3.75 / 1.5725,
         -3.125 / 1.0725},
        5e-10,
        "APLICUR multiphase ridge");
    require_test(
        multiphase.auxiliary_width > multiphase_options.block_size &&
            multiphase.basis_rank > 0,
        "APLICUR reuses incrementally appended CUR and row-QR state");
    require_accounting(
        multiphase,
        multiphase_options.regularization,
        multiphase_options.tolerance,
        "APLICUR multiphase ridge");
    bool has_off_cadence_phase_terminal = false;
    std::string phase_trace;
    for (std::size_t index = 0; index + 1 < multiphase.trace.size(); ++index) {
        phase_trace += " (" + std::to_string(multiphase.trace[index].epoch) +
            "," + std::to_string(multiphase.trace[index].depth) + ")";
        if (multiphase.trace[index].epoch !=
                multiphase.trace[index + 1].epoch &&
            !amfls::math::is_certificate_checkpoint(
                multiphase.trace[index].depth,
                multiphase_options.regularization)) {
            has_off_cadence_phase_terminal = true;
        }
    }
    require_test(
        has_off_cadence_phase_terminal,
        "APLICUR validates an off-cadence phase terminal before restarting:" +
            phase_trace);

    amfls::AplicurOptions capped_options = options;
    capped_options.tolerance = 1e-30;
    capped_options.maximum_iterations = 3;
    const amfls::LeastSquaresResult capped = amfls::solve_aplicur(
        coupled_matrix.data(),
        7,
        3,
        coupled_rhs.data(),
        capped_options);
    require_test(
        capped.status == amfls::SolverStatus::work_limit &&
            capped.stop_reason == amfls::StopReason::maximum_depth &&
            capped.iterations == 3,
        "APLICUR off-cadence iteration cap");
    require_accounting(
        capped,
        capped_options.regularization,
        capped_options.tolerance,
        "APLICUR capped ridge");

    amfls::AplicurOptions unregularized = options;
    unregularized.regularization = 0.0;
    require_invalid_argument(
        [&]() {
            (void)amfls::solve_aplicur(
                matrix.data(), 6, 3, right_hand_side.data(), unregularized);
        },
        "APLICUR declares Algorithm 3's positive-regularization scope");

    amfls::AplicurOptions invalid_block = options;
    invalid_block.block_size = 0;
    require_invalid_argument(
        [&]() {
            (void)amfls::solve_aplicur(
                matrix.data(), 6, 3, right_hand_side.data(), invalid_block);
        },
        "APLICUR rejects a zero block size");

    std::vector<double> nonfinite_matrix = matrix;
    nonfinite_matrix[0] = std::numeric_limits<double>::infinity();
    require_invalid_argument(
        [&]() {
            (void)amfls::solve_aplicur(
                nonfinite_matrix.data(),
                6,
                3,
                right_hand_side.data(),
                options);
        },
        "APLICUR rejects a nonfinite explicit matrix");

    std::cout << "test_aplicur passed\n";
    return 0;
}
