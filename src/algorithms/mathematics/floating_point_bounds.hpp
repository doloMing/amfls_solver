#pragma once

#include <algorithm>
#include <cfenv>
#include <cmath>
#include <cstddef>
#include <limits>
#include <span>

namespace amfls::math::fp {

struct NonnegativeInterval {
    double lower = 0.0;
    double upper = std::numeric_limits<double>::infinity();
};

inline bool gradual_underflow_is_active() noexcept {
    if (std::numeric_limits<double>::has_denorm != std::denorm_present) {
        return false;
    }
    // Volatile operands force runtime arithmetic. The first product detects
    // denormals-are-zero inputs; the second detects flush-to-zero outputs.
    volatile double smallest_subnormal =
        std::numeric_limits<double>::denorm_min();
    volatile double one = 1.0;
    volatile double preserved_subnormal = smallest_subnormal * one;
    volatile double smallest_normal = std::numeric_limits<double>::min();
    volatile double half = 0.5;
    volatile double produced_subnormal = smallest_normal * half;
    return preserved_subnormal ==
            std::numeric_limits<double>::denorm_min() &&
        produced_subnormal > 0.0;
}

inline double upward_from_long_double(long double value) noexcept {
    if (!(value >= 0.0L) || !std::isfinite(value) ||
        value > static_cast<long double>(
                    std::numeric_limits<double>::max())) {
        return std::numeric_limits<double>::infinity();
    }
    if (value == 0.0L) {
        return 0.0;
    }
    double rounded = static_cast<double>(value);
    if (static_cast<long double>(rounded) < value) {
        rounded = std::nextafter(
            rounded, std::numeric_limits<double>::infinity());
    }
    return std::nextafter(
        rounded, std::numeric_limits<double>::infinity());
}

inline double downward_from_long_double(long double value) noexcept {
    if (!(value > 0.0L) || !std::isfinite(value)) {
        return 0.0;
    }
    if (value > static_cast<long double>(
                    std::numeric_limits<double>::max())) {
        return std::numeric_limits<double>::max();
    }
    double rounded = static_cast<double>(value);
    if (static_cast<long double>(rounded) > value) {
        rounded = std::nextafter(rounded, 0.0);
    }
    return std::nextafter(rounded, 0.0);
}

inline long double accumulation_gamma(std::size_t operations) noexcept {
    const long double unit_roundoff =
        0.5L * std::numeric_limits<long double>::epsilon();
    const long double product =
        static_cast<long double>(operations) * unit_roundoff;
    if (!(product < 1.0L)) {
        return std::numeric_limits<long double>::infinity();
    }
    return product / (1.0L - product);
}

// The validation-only norm uses widened FMA accumulation.  All binary64
// squares fit in the exponent range of the usual extended long double.  On a
// platform where they do not, or where accumulation cannot be bounded, the
// interval safely becomes [0,+inf].
inline NonnegativeInterval norm_interval_from_squared_sum(
    long double sum,
    std::size_t accumulated_terms) noexcept {
    constexpr int double_largest_square_exponent =
        2 * std::numeric_limits<double>::max_exponent;
    constexpr int double_smallest_square_exponent =
        2 * (std::numeric_limits<double>::min_exponent -
             std::numeric_limits<double>::digits);
    constexpr int long_double_smallest_exponent =
        std::numeric_limits<long double>::min_exponent -
        std::numeric_limits<long double>::digits;
    if constexpr (
        std::numeric_limits<long double>::max_exponent <=
                double_largest_square_exponent ||
        long_double_smallest_exponent > double_smallest_square_exponent) {
        return {};
    }
    if (!(sum >= 0.0L) || !std::isfinite(sum)) {
        return {};
    }
    if (sum == 0.0L) {
        return {0.0, 0.0};
    }
    const long double gamma = accumulation_gamma(accumulated_terms);
    if (!std::isfinite(gamma) || !(gamma < 1.0L)) {
        return {};
    }
    const long double sqrt_roundoff =
        2.0L * std::numeric_limits<long double>::epsilon();
    const long double lower_squared = sum / (1.0L + gamma);
    const long double upper_squared = sum / (1.0L - gamma);
    const long double lower =
        std::sqrt(lower_squared) * (1.0L - sqrt_roundoff);
    const long double upper =
        std::sqrt(upper_squared) * (1.0L + sqrt_roundoff);
    return {
        downward_from_long_double(lower),
        upward_from_long_double(upper)};
}

inline double binary64_gamma(long long operations) noexcept {
    if (operations < 0) {
        return std::numeric_limits<double>::infinity();
    }
    const long double unit_roundoff =
        0.5L * std::numeric_limits<double>::epsilon();
    const long double product =
        static_cast<long double>(operations) * unit_roundoff;
    if (!(product < 1.0L)) {
        return std::numeric_limits<double>::infinity();
    }
    return upward_from_long_double(product / (1.0L - product));
}

inline double upward_add(double left, double right) noexcept {
    return upward_from_long_double(
        static_cast<long double>(left) +
        static_cast<long double>(right));
}

inline double upward_multiply(double left, double right) noexcept {
    if (left < 0.0 || right < 0.0) {
        return std::numeric_limits<double>::infinity();
    }
    const long double product =
        static_cast<long double>(left) * static_cast<long double>(right);
    if (left > 0.0 && right > 0.0 && product == 0.0L) {
        return std::numeric_limits<double>::denorm_min();
    }
    return upward_from_long_double(product);
}

inline double downward_add(double left, double right) noexcept {
    if (left < 0.0 || right < 0.0 ||
        !std::isfinite(left) || !std::isfinite(right)) {
        return 0.0;
    }
    return downward_from_long_double(
        static_cast<long double>(left) +
        static_cast<long double>(right));
}

inline double downward_multiply(double left, double right) noexcept {
    if (left < 0.0 || right < 0.0 ||
        !std::isfinite(left) || !std::isfinite(right)) {
        return 0.0;
    }
    return downward_from_long_double(
        static_cast<long double>(left) *
        static_cast<long double>(right));
}

inline double downward_divide(double numerator, double denominator) noexcept {
    if (!(numerator > 0.0) || !(denominator > 0.0) ||
        !std::isfinite(numerator) || !std::isfinite(denominator)) {
        return 0.0;
    }
    return downward_from_long_double(
        static_cast<long double>(numerator) /
        static_cast<long double>(denominator));
}

inline double downward_difference(double left, double right) noexcept {
    if (!(left > right) || !std::isfinite(left) || right < 0.0 ||
        !std::isfinite(right)) {
        return 0.0;
    }
    return downward_from_long_double(
        static_cast<long double>(left) -
        static_cast<long double>(right));
}

inline double upward_divide(double numerator, double denominator) noexcept {
    if (numerator == 0.0) {
        return 0.0;
    }
    if (!(denominator > 0.0) || numerator < 0.0 ||
        !std::isfinite(numerator) || !std::isfinite(denominator)) {
        return std::numeric_limits<double>::infinity();
    }
    const long double quotient =
        static_cast<long double>(numerator) /
        static_cast<long double>(denominator);
    if (quotient == 0.0L) {
        return std::numeric_limits<double>::denorm_min();
    }
    return upward_from_long_double(quotient);
}

// Let stored_sum be the result of accumulating nonnegative binary64 terms.
// With round-to-nearest and gradual underflow,
//   stored_sum >= (1-gamma_k) exact_sum - k*denorm_min.
// Solving this inequality for exact_sum gives the outward upper bound below.
inline double positive_sum_upper_bound(
    double stored_sum,
    long long operations) noexcept {
    if (stored_sum < 0.0 || !std::isfinite(stored_sum) || operations < 0) {
        return std::numeric_limits<double>::infinity();
    }
    // Under gradual underflow a positive sum of nonnegative stored binary64
    // terms cannot round to zero. Preserve an exact zero-matrix model.
    if (stored_sum == 0.0) {
        return 0.0;
    }
    const double gamma = binary64_gamma(operations);
    const double denominator = downward_difference(1.0, gamma);
    if (!(denominator > 0.0)) {
        return std::numeric_limits<double>::infinity();
    }
    const double absolute_roundoff = upward_from_long_double(
        static_cast<long double>(operations) *
        static_cast<long double>(std::numeric_limits<double>::denorm_min()));
    return upward_divide(
        upward_add(stored_sum, absolute_roundoff), denominator);
}

// The companion form for fma(a,a,sum) must also cover a positive exact
// product that rounds to zero before the stored sum becomes nonzero.  Unlike
// a sum of already stored nonnegative terms, stored_sum==0 therefore does
// not prove that the exact square sum is zero.
inline double positive_fma_square_sum_upper_bound(
    double stored_sum,
    long long operations) noexcept {
    if (stored_sum < 0.0 || !std::isfinite(stored_sum) || operations < 0) {
        return std::numeric_limits<double>::infinity();
    }
    const double gamma = binary64_gamma(operations);
    const double denominator = downward_difference(1.0, gamma);
    if (!(denominator > 0.0)) {
        return std::numeric_limits<double>::infinity();
    }
    const double absolute_roundoff = upward_from_long_double(
        static_cast<long double>(operations) *
        static_cast<long double>(std::numeric_limits<double>::denorm_min()));
    return upward_divide(
        upward_add(stored_sum, absolute_roundoff), denominator);
}

// Companion lower bound for a nonnegative sum accumulated by one rounded
// binary64 operation per term (in particular, fma(a,a,sum)). From
//   stored_sum <= (1+gamma_k) exact_sum + k*denorm_min
// we obtain the outward lower envelope below.
inline double positive_sum_lower_bound(
    double stored_sum,
    long long operations) noexcept {
    if (!(stored_sum > 0.0) || !std::isfinite(stored_sum) || operations < 0) {
        return 0.0;
    }
    const double gamma = binary64_gamma(operations);
    const double denominator = upward_add(1.0, gamma);
    if (!std::isfinite(denominator)) {
        return 0.0;
    }
    const double absolute_roundoff = upward_from_long_double(
        static_cast<long double>(operations) *
        static_cast<long double>(std::numeric_limits<double>::denorm_min()));
    return downward_divide(
        downward_difference(stored_sum, absolute_roundoff), denominator);
}

inline double upward_sqrt(double value) noexcept;
inline double downward_sqrt(double value) noexcept;

inline double upward_scale_power_of_two(
    double value,
    int exponent) noexcept {
    if (value == 0.0) {
        return 0.0;
    }
    if (!(value > 0.0) || !std::isfinite(value)) {
        return std::numeric_limits<double>::infinity();
    }
    const double scaled = std::scalbn(value, exponent);
    if (std::isinf(scaled)) {
        return scaled;
    }
    if (scaled == 0.0) {
        return std::numeric_limits<double>::denorm_min();
    }
    return std::nextafter(scaled, std::numeric_limits<double>::infinity());
}

inline double downward_scale_power_of_two(
    double value,
    int exponent) noexcept {
    if (!(value > 0.0) || !std::isfinite(value)) {
        return 0.0;
    }
    const double scaled = std::scalbn(value, exponent);
    if (std::isinf(scaled)) {
        return std::numeric_limits<double>::max();
    }
    if (scaled == 0.0) {
        return 0.0;
    }
    return std::nextafter(scaled, 0.0);
}

// Two-pass scaled binary64 norm. A power-of-two scale keeps the largest
// stored component in [1/2,1), so the FMA square sum cannot overflow. Scaling
// is exact except when a component enters the subnormal range; a normwise
// sqrt(k)*denorm_min radius covers every such rounding before outward rescale.
inline NonnegativeInterval norm_interval(
    std::span<const double> values) noexcept {
    if (!std::numeric_limits<double>::is_iec559 ||
        std::fegetround() != FE_TONEAREST ||
        !gradual_underflow_is_active()) {
        return {};
    }
    double maximum = 0.0;
    for (double value : values) {
        if (!std::isfinite(value)) {
            return {};
        }
        maximum = std::max(maximum, std::abs(value));
    }
    if (maximum == 0.0) {
        return {0.0, 0.0};
    }
    if (values.size() > static_cast<std::size_t>(
            std::numeric_limits<long long>::max())) {
        return {};
    }

    int rescale_exponent = 0;
    (void)std::frexp(maximum, &rescale_exponent);
    double scaled_square_sum = 0.0;
    for (double value : values) {
        const double scaled = std::scalbn(
            std::abs(value), -rescale_exponent);
        scaled_square_sum = std::fma(
            scaled, scaled, scaled_square_sum);
    }
    const long long operations = static_cast<long long>(values.size());
    const double squared_lower = positive_sum_lower_bound(
        scaled_square_sum, operations);
    const double squared_upper = positive_sum_upper_bound(
        scaled_square_sum, operations);
    const double scaled_lower = downward_sqrt(squared_lower);
    const double scaled_upper = upward_sqrt(squared_upper);
    // Keep denorm_min in long double throughout: forming 0.5*denorm_min in
    // binary64 would vanish. One full denorm_min per component is the simpler
    // conservative absolute scaling-error model.
    const double scaling_error = upward_from_long_double(
        std::sqrt(static_cast<long double>(values.size())) *
        static_cast<long double>(std::numeric_limits<double>::denorm_min()));
    return {
        downward_scale_power_of_two(
            downward_difference(scaled_lower, scaling_error),
            rescale_exponent),
        upward_scale_power_of_two(
            upward_add(scaled_upper, scaling_error),
            rescale_exponent)};
}

inline double frobenius_norm_upper_bound(
    std::span<const double> values) noexcept {
    return norm_interval(values).upper;
}

inline double upward_sqrt(double value) noexcept {
    if (value < 0.0 || !std::isfinite(value)) {
        return value == std::numeric_limits<double>::infinity()
            ? value
            : std::numeric_limits<double>::infinity();
    }
    return upward_from_long_double(
        std::sqrt(static_cast<long double>(value)) *
        (1.0L + 2.0L * std::numeric_limits<long double>::epsilon()));
}

inline double downward_sqrt(double value) noexcept {
    if (!(value > 0.0) || !std::isfinite(value)) {
        return 0.0;
    }
    return downward_from_long_double(
        std::sqrt(static_cast<long double>(value)) *
        (1.0L - 2.0L * std::numeric_limits<long double>::epsilon()));
}

}  // namespace amfls::math::fp
