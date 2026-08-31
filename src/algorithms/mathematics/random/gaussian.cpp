#include "algorithms/mathematics/random/gaussian.hpp"

#include <cmath>
#include <limits>
#include <numbers>
#include <vector>

#include "algorithms/mathematics/linalg/blas_lapack.hpp"

namespace amfls::math {
namespace {

std::uint64_t splitmix64(std::uint64_t value) {
    value += 0x9e3779b97f4a7c15ULL;
    value = (value ^ (value >> 30U)) * 0xbf58476d1ce4e5b9ULL;
    value = (value ^ (value >> 27U)) * 0x94d049bb133111ebULL;
    return value ^ (value >> 31U);
}

double open_uniform(std::uint64_t bits) {
    constexpr double inverse = 1.0 / 9007199254740992.0;
    return (static_cast<double>((bits >> 11U) + 1U)) * inverse;
}

double gaussian_at(
    std::uint64_t seed,
    std::uint64_t stream,
    std::uint64_t index) {
    const std::uint64_t key =
        splitmix64(seed) ^ splitmix64(stream + 0xd2b74407b1ce6e93ULL);
    const double u1 = open_uniform(splitmix64(key ^ (2U * index)));
    const double u2 = open_uniform(splitmix64(key ^ (2U * index + 1U)));
    return std::sqrt(-2.0 * std::log(u1)) *
           std::cos(2.0 * std::numbers::pi * u2);
}

}  // namespace

Matrix gaussian_matrix(
    int rows,
    int cols,
    std::uint64_t seed,
    std::uint64_t stream,
    std::uint64_t first_column) {
    Matrix result(rows, cols);
    for (int col = 0; col < cols; ++col) {
        const std::uint64_t global_col = first_column + col;
        for (int row = 0; row < rows; ++row) {
            const std::uint64_t index =
                global_col * static_cast<std::uint64_t>(rows) + row;
            result(row, col) = gaussian_at(seed, stream, index);
        }
    }
    return result;
}

Matrix gaussian_matrix_orthogonal_to(
    int rows,
    int cols,
    const double* vector,
    std::uint64_t seed,
    std::uint64_t stream,
    std::uint64_t first_column) {
    Matrix result = gaussian_matrix(
        rows, cols, seed, stream, first_column);
    const double vector_norm_value = vector_norm(vector, rows);
    if (vector_norm_value == 0.0) {
        return result;
    }
    std::vector<double> normalized(rows);
    for (int row = 0; row < rows; ++row) {
        normalized[row] = vector[row] / vector_norm_value;
    }
    for (int col = 0; col < cols; ++col) {
        const double coefficient = vector_dot(
            normalized.data(), result.column_data(col), rows);
        for (int row = 0; row < rows; ++row) {
            result(row, col) -= coefficient * normalized[row];
        }
    }
    return result;
}

}  // namespace amfls::math
