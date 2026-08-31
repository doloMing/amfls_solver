#pragma once

#include <algorithm>
#include <cmath>

namespace amfls::detail {

inline double evaluate_lsmb_cubic(
    double cubic,
    double quadratic,
    double linear,
    double constant,
    double value) {
    return ((cubic * value + quadratic) * value + linear) * value +
        constant;
}

inline bool solve_lsmb_gamma(
    double cubic,
    double quadratic,
    double linear,
    double constant,
    double upper_endpoint_value,
    bool lower_endpoint_is_root,
    bool upper_endpoint_is_root,
    double& gamma) {
    if (!std::isfinite(cubic) || !std::isfinite(quadratic) ||
        !std::isfinite(linear) || !std::isfinite(constant) ||
        !std::isfinite(upper_endpoint_value)) {
        return false;
    }
    // Endpoint roots are properties of the unsquared Hallman--Gu primitives,
    // not of their rounded polynomial coefficients.  In particular, a
    // nonzero residual may square to zero.  The caller therefore supplies the
    // exact endpoint predicates separately from p(0) and p(1).
    if (lower_endpoint_is_root) {
        gamma = 0.0;
        return true;
    }
    if (upper_endpoint_is_root) {
        gamma = 1.0;
        return true;
    }
    if (constant > 0.0 || upper_endpoint_value < 0.0) {
        return false;
    }

    double lower = 0.0;
    double upper = 1.0;
    // Continue to adjacent FP64 numbers rather than imposing an absolute
    // root scale.  The conservative cap exceeds the 1074 binary halvings
    // needed to traverse the IEEE-754 FP64 subnormal range from one to zero.
    for (int iteration = 0; iteration < 2048; ++iteration) {
        const double midpoint = lower + 0.5 * (upper - lower);
        if (midpoint == lower || midpoint == upper) {
            break;
        }
        const double midpoint_value = evaluate_lsmb_cubic(
            cubic, quadratic, linear, constant, midpoint);
        if (!std::isfinite(midpoint_value)) {
            return false;
        }
        if (midpoint_value <= 0.0) {
            lower = midpoint;
        } else {
            upper = midpoint;
        }
    }
    gamma = lower + 0.5 * (upper - lower);
    return std::isfinite(gamma) && gamma >= 0.0 && gamma <= 1.0;
}

}  // namespace amfls::detail
