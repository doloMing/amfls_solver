#pragma once

#include <concepts>
#include <limits>
#include <stdexcept>

namespace amfls::math {

template <std::signed_integral Integer>
Integer checked_add(Integer first, Integer second, const char* message) {
    const Integer maximum = std::numeric_limits<Integer>::max();
    const Integer minimum = std::numeric_limits<Integer>::min();
    if ((second > 0 && first > maximum - second) ||
        (second < 0 && first < minimum - second)) {
        throw std::length_error(message);
    }
    return static_cast<Integer>(first + second);
}

template <std::unsigned_integral Integer>
Integer checked_add(Integer first, Integer second, const char* message) {
    if (first > std::numeric_limits<Integer>::max() - second) {
        throw std::length_error(message);
    }
    return static_cast<Integer>(first + second);
}

inline long long checked_nonnegative_multiply(
    int first,
    int second,
    const char* message) {
    const long long wide_first = first;
    const long long wide_second = second;
    if (wide_first < 0 || wide_second < 0 ||
        (wide_first != 0 &&
         wide_second > std::numeric_limits<long long>::max() / wide_first)) {
        throw std::length_error(message);
    }
    return wide_first * wide_second;
}

inline void checked_counter_add(
    long long& counter,
    long long increment,
    const char* message) {
    if (counter < 0 || increment < 0) {
        throw std::length_error(message);
    }
    counter = checked_add(counter, increment, message);
}

}  // namespace amfls::math
