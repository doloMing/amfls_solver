#pragma once

#include <algorithm>
#include <cmath>
#include <limits>

#include "amfls/least_squares_result.hpp"

namespace amfls::math {

// Static certificate cadence shared by AMFLS and the iterative baselines.
// Ordinary least squares checks power-of-two endpoints and their 3/2
// midpoints; ridge checks only power-of-two endpoints.  Terminal states are
// checked by each solver independently.
inline bool is_certificate_checkpoint(
    int iteration,
    double regularization) noexcept {
    if (iteration <= 1) {
        return true;
    }
    int odd_part = iteration;
    while (odd_part % 2 == 0) {
        odd_part /= 2;
    }
    return odd_part == 1 || (regularization == 0.0 && odd_part == 3);
}

// The static cadence remains an upper bound on the distance between
// validations.  Between static checkpoints, extrapolate only an observed
// positive logarithmic contraction of the stopping certificate.  The
// extrapolation contains no fitted parameter: it is the crossing iteration
// of the straight line through the two adjacent observations in log scale.
// At most one predicted validation is admitted between two static
// checkpoints.  A failed prediction therefore cannot recursively create a
// sequence of off-cadence validations; the next observation is the static
// endpoint of the interval.
class CertificateCheckpointSchedule {
public:
    CertificateCheckpointSchedule(
        double regularization,
        double tolerance) noexcept
        : regularization_(regularization), tolerance_(tolerance) {}

    [[nodiscard]] bool should_evaluate(int iteration) const noexcept {
        return is_certificate_checkpoint(iteration, regularization_) ||
            iteration >= predicted_iteration_;
    }

    void record_evaluation(
        int iteration,
        const amfls::LeastSquaresResult& result) noexcept {
        predicted_iteration_ = std::numeric_limits<int>::max();
        const bool static_evaluation =
            is_certificate_checkpoint(iteration, regularization_);
        if (static_evaluation && has_previous_ &&
            iteration > previous_iteration_) {
            if (regularization_ > 0.0) {
                predicted_iteration_ = predicted_crossing(
                    previous_relative_energy_error_,
                    result.relative_energy_error_upper_bound,
                    previous_iteration_,
                    iteration,
                    tolerance_);
            } else {
                const int compatible_prediction = predicted_crossing(
                    previous_compatible_backward_error_,
                    result.compatible_backward_error_upper_bound,
                    previous_iteration_,
                    iteration,
                    tolerance_);
                const int least_squares_prediction = predicted_crossing(
                    previous_least_squares_backward_error_,
                    result.least_squares_backward_error_upper_bound,
                    previous_iteration_,
                    iteration,
                    tolerance_);
                predicted_iteration_ = std::min(
                    compatible_prediction, least_squares_prediction);
            }
        }

        previous_iteration_ = iteration;
        previous_compatible_backward_error_ =
            result.compatible_backward_error_upper_bound;
        previous_least_squares_backward_error_ =
            result.least_squares_backward_error_upper_bound;
        previous_relative_energy_error_ =
            result.relative_energy_error_upper_bound;
        has_previous_ = true;
    }

private:
    static int predicted_crossing(
        double previous,
        double current,
        int previous_iteration,
        int current_iteration,
        double tolerance) noexcept {
        if (!(previous > current) || !(current > tolerance) ||
            !(tolerance > 0.0) || !std::isfinite(previous) ||
            !std::isfinite(current) || !std::isfinite(tolerance)) {
            return std::numeric_limits<int>::max();
        }

        const long double log_distance =
            std::log(static_cast<long double>(current)) -
            std::log(static_cast<long double>(tolerance));
        const long double log_contraction =
            std::log(static_cast<long double>(previous)) -
            std::log(static_cast<long double>(current));
        const long double iteration_distance =
            static_cast<long double>(current_iteration - previous_iteration);
        const long double offset = std::ceil(
            iteration_distance * log_distance / log_contraction);
        if (!(offset > 0.0L) || !std::isfinite(offset) ||
            offset > static_cast<long double>(
                std::numeric_limits<int>::max() - current_iteration)) {
            return std::numeric_limits<int>::max();
        }
        return current_iteration + static_cast<int>(offset);
    }

    double regularization_ = 0.0;
    double tolerance_ = 0.0;
    int predicted_iteration_ = std::numeric_limits<int>::max();
    int previous_iteration_ = 0;
    double previous_compatible_backward_error_ = 0.0;
    double previous_least_squares_backward_error_ = 0.0;
    double previous_relative_energy_error_ = 0.0;
    bool has_previous_ = false;
};

}  // namespace amfls::math
