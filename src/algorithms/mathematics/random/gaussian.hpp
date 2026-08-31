#pragma once

#include <cstdint>

#include "algorithms/mathematics/linalg/matrix.hpp"

namespace amfls::math {

// Counter-addressed Gaussian block. Results do not depend on blocking/order.
Matrix gaussian_matrix(
    int rows,
    int cols,
    std::uint64_t seed,
    std::uint64_t stream,
    std::uint64_t first_column);

// Counter-addressed Gaussian block projected orthogonally away from a vector.
// The normalized projection avoids overflow/underflow from squaring its norm.
Matrix gaussian_matrix_orthogonal_to(
    int rows,
    int cols,
    const double* vector,
    std::uint64_t seed,
    std::uint64_t stream,
    std::uint64_t first_column);

}  // namespace amfls::math
