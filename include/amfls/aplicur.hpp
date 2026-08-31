#pragma once

#include <cstdint>

#include "amfls/least_squares_result.hpp"

namespace amfls {

// Options for the explicit-matrix APLICUR comparison baseline.  The public
// regularization parameter is lambda in
//
//   min_x ||A x - b||_2^2 + lambda ||x||_2^2,
//
// whereas the APLICUR paper writes mu = sqrt(lambda).  Its detailed
// Algorithm 3 assumes mu > 0, so this baseline deliberately rejects the
// unregularized case rather than silently changing the published method.
struct AplicurOptions {
    double regularization = 1e-8;
    double tolerance = 1e-8;

    int block_size = 8;
    int sparse_sign_nonzeros = 8;
    int spectral_probe_count = 10;

    // An input value of zero selects 50 * sqrt(regularization), the midpoint
    // of the paper's recommended [30 mu, 100 mu] range.
    double cur_tolerance = 0.0;
    double re_preconditioning_tolerance = 10.0;
    double dynamic_stopping_tolerance = 150.0;

    // Zero selects four times min(rows, cols) total PLSQR iterations across
    // all warm-started phases.
    int maximum_iterations = 0;

    // FP64 cutoffs for the CUR-intersection solve and compact spectral factor.
    double relative_rank_tolerance = 1e-12;
    double absolute_rank_tolerance = 0.0;
    std::uint64_t seed = 0;
};

// Solve using explicit contiguous CPU FP64 row-major storage.  APLICUR selects
// individual rows and columns, so a MatrixOperator-only interface would
// misrepresent its access requirements.  The caller retains ownership of the
// matrix values for the duration of the call.
LeastSquaresResult solve_aplicur(
    const double* matrix_row_major,
    int rows,
    int cols,
    const double* right_hand_side,
    const AplicurOptions& options);

}  // namespace amfls
