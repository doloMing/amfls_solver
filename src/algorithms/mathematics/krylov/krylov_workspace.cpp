#include "algorithms/mathematics/krylov/krylov_workspace.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>

#include "algorithms/mathematics/linalg/blas_lapack.hpp"
#include "algorithms/mathematics/linalg/svd.hpp"

namespace amfls::math {
namespace {

using Clock = std::chrono::steady_clock;

double elapsed(Clock::time_point start) {
    return std::chrono::duration<double>(Clock::now() - start).count();
}

Matrix take_scalar_residual_buffer(
    Matrix& buffer,
    int rows,
    int cols) {
    if (cols == 1 && buffer.rows() == rows && buffer.cols() == 1 &&
        buffer.size() == rows) {
        Matrix result = std::move(buffer);
        buffer = Matrix();
        return result;
    }
    return Matrix(rows, cols);
}

int projected_row_limit(int basis_limit) {
    if (basis_limit >= std::numeric_limits<int>::max()) {
        throw std::length_error(
            "block Golub-Kahan projected row count exceeds LP64 limits");
    }
    return basis_limit + 1;
}

struct OrthogonalBlockFactor {
    Matrix coefficients;
    Matrix q;
    Matrix factor;
    double discarded_frobenius_norm = 0.0;
    double discarded_tracked_prefix_frobenius_norm = 0.0;
    double rank_cutoff = 0.0;
    bool positive_finite_below_cutoff = false;
    bool nonfinite_residual = false;
};

bool matrix_is_finite(const Matrix& matrix) {
    for (int index = 0; index < matrix.size(); ++index) {
        if (!std::isfinite(matrix.data()[index])) {
            return false;
        }
    }
    return true;
}

double rank_revealing_cutoff(
    double reference_scale,
    int operator_rows,
    int operator_cols,
    int basis_columns,
    int block_columns) {
    if (!(reference_scale >= 0.0) || !std::isfinite(reference_scale)) {
        return std::numeric_limits<double>::infinity();
    }
    const int dimension = std::max({
        1,
        operator_rows,
        operator_cols,
        basis_columns,
        block_columns});
    return 64.0 * std::numeric_limits<double>::epsilon() *
        dimension * reference_scale;
}

OrthogonalBlockFactor orthogonalize_and_factor(
    Matrix block,
    const Matrix& basis,
    int maximum_rank,
    int operator_rows,
    int operator_cols,
    double reference_scale,
    int tracked_prefix_columns,
    bool retain_coefficients,
    RunStatistics& statistics,
    std::vector<double>& scalar_projection_coefficients,
    std::vector<double>& scalar_accumulated_coefficients) {
    if (block.rows() != basis.rows() || maximum_rank < 0 ||
        tracked_prefix_columns < 0 ||
        tracked_prefix_columns > block.cols() ||
        operator_rows <= 0 || operator_cols <= 0) {
        throw std::invalid_argument(
            "invalid block Golub-Kahan orthogonalization dimensions");
    }

    const auto start = Clock::now();
    const bool retain_scalar_coefficients =
        block.cols() != 1 || retain_coefficients;
    Matrix coefficients(
        retain_scalar_coefficients ? basis.cols() : 0,
        block.cols());
    bool scalar_coefficients_finite = true;
    double projected_residual_norm = 0.0;
    if (block.cols() == 1 && basis.cols() > 0) {
        const double norm_before_projection = vector_norm(
            block.data(), block.rows());
        if (scalar_projection_coefficients.size() <
            static_cast<std::size_t>(basis.cols())) {
            scalar_projection_coefficients.resize(basis.cols());
        }
        double* const pass_coefficients =
            scalar_projection_coefficients.data();
        double* accumulated_coefficients = nullptr;
        if (!retain_scalar_coefficients) {
            if (scalar_accumulated_coefficients.size() <
                static_cast<std::size_t>(basis.cols())) {
                scalar_accumulated_coefficients.resize(basis.cols());
            }
            accumulated_coefficients =
                scalar_accumulated_coefficients.data();
            std::fill_n(accumulated_coefficients, basis.cols(), 0.0);
        }
        const auto projection_pass = [&] {
            gemv(
                basis,
                true,
                block.data(),
                1.0,
                0.0,
                pass_coefficients);
            for (int row = 0; row < basis.cols(); ++row) {
                if (retain_scalar_coefficients) {
                    coefficients(row, 0) += pass_coefficients[row];
                } else {
                    accumulated_coefficients[row] += pass_coefficients[row];
                }
            }
            gemv(
                basis,
                false,
                pass_coefficients,
                -1.0,
                1.0,
                block.data());
        };
        projection_pass();
        projected_residual_norm = vector_norm(
            block.data(), block.rows());
        // Daniel--Gragg--Kaufman--Stewart reorthogonalization: a second
        // projection is needed only when the first projection removed enough
        // norm that its roundoff may contaminate the retained residual.
        if (std::isfinite(norm_before_projection) &&
            std::isfinite(projected_residual_norm) &&
            projected_residual_norm <
                std::sqrt(0.5) * norm_before_projection) {
            projection_pass();
            projected_residual_norm = vector_norm(
                block.data(), block.rows());
        }
        if (!retain_scalar_coefficients) {
            scalar_coefficients_finite = std::all_of(
                accumulated_coefficients,
                accumulated_coefficients + basis.cols(),
                [](double value) { return std::isfinite(value); });
        }
    } else {
        // Block CGS2 is retained for multi-column residuals.  A Frobenius- or
        // columnwise DGKS test can miss cancellation in a linear combination
        // of columns, whereas the unconditional second block pass preserves
        // the finite-precision invariant used by the recurrence analysis.
        for (int pass = 0; pass < 2 && basis.cols() > 0; ++pass) {
            Matrix pass_coefficients = transpose_multiply(basis, block);
            for (int col = 0; col < coefficients.cols(); ++col) {
                for (int row = 0; row < coefficients.rows(); ++row) {
                    coefficients(row, col) +=
                        pass_coefficients(row, col);
                }
            }
            gemm(
                basis,
                false,
                pass_coefficients,
                false,
                -1.0,
                1.0,
                block);
        }
        projected_residual_norm = frobenius_norm(block);
    }

    const double cutoff = rank_revealing_cutoff(
        reference_scale,
        operator_rows,
        operator_cols,
        basis.cols(),
        block.cols());
    const bool finite_residual =
        std::isfinite(reference_scale) &&
        std::isfinite(projected_residual_norm) &&
        scalar_coefficients_finite &&
        matrix_is_finite(coefficients);
    OrthogonalBlockFactor result{
        std::move(coefficients),
        Matrix(block.rows(), 0),
        Matrix(0, block.cols()),
        projected_residual_norm,
        0.0,
        cutoff,
        false,
        !finite_residual};
    if (!finite_residual) {
        result.discarded_frobenius_norm =
            std::numeric_limits<double>::infinity();
        result.discarded_tracked_prefix_frobenius_norm =
            tracked_prefix_columns > 0
            ? std::numeric_limits<double>::infinity()
            : 0.0;
        statistics.orthogonalization_seconds += elapsed(start);
        return result;
    }
    if (maximum_rank == 0 || block.cols() == 0) {
        for (int col = 0; col < tracked_prefix_columns; ++col) {
            result.discarded_tracked_prefix_frobenius_norm = std::hypot(
                result.discarded_tracked_prefix_frobenius_norm,
                vector_norm(block.column_data(col), block.rows()));
        }
        statistics.orthogonalization_seconds += elapsed(start);
        return result;
    }

    if (block.cols() == 1) {
        const double residual_norm = projected_residual_norm;
        if (std::isfinite(residual_norm) && residual_norm > cutoff) {
            const double inverse_norm = 1.0 / residual_norm;
            const int rows = block.rows();
            double* const values = block.data();
            for (int row = 0; row < rows; ++row) {
                values[row] *= inverse_norm;
            }
            result.q = std::move(block);
            result.factor = Matrix(1, 1);
            result.factor(0, 0) = residual_norm;
            result.discarded_frobenius_norm = 0.0;
            result.discarded_tracked_prefix_frobenius_norm = 0.0;
        } else if (std::isfinite(residual_norm) && residual_norm > 0.0 &&
                   residual_norm <= cutoff && maximum_rank > 0) {
            result.positive_finite_below_cutoff = true;
            if (tracked_prefix_columns == 1) {
                result.discarded_tracked_prefix_frobenius_norm =
                    residual_norm;
            }
        }
        statistics.orthogonalization_seconds += elapsed(start);
        return result;
    }

    const int block_columns = block.cols();
    SvdResult svd = thin_svd(std::move(block));
    int rank = 0;
    while (rank < static_cast<int>(svd.singular_values.size()) &&
           rank < maximum_rank &&
           std::isfinite(svd.singular_values[rank]) &&
           svd.singular_values[rank] > cutoff) {
        ++rank;
    }
    if (rank < static_cast<int>(svd.singular_values.size()) &&
        rank < maximum_rank) {
        const double first_discarded = svd.singular_values[rank];
        result.positive_finite_below_cutoff =
            std::isfinite(first_discarded) && first_discarded > 0.0 &&
            first_discarded <= cutoff;
    }
    if (rank > 0) {
        svd.u.truncate_columns(rank);
        Matrix factor(rank, block_columns);
        for (int col = 0; col < block_columns; ++col) {
            for (int row = 0; row < rank; ++row) {
                factor(row, col) =
                    svd.singular_values[row] * svd.vt(row, col);
            }
        }
        result.q = std::move(svd.u);
        result.factor = std::move(factor);
    }
    result.discarded_frobenius_norm = 0.0;
    for (int index = rank;
         index < static_cast<int>(svd.singular_values.size());
         ++index) {
        result.discarded_frobenius_norm = std::hypot(
            result.discarded_frobenius_norm,
            svd.singular_values[index]);
        double tracked_right_weight = 0.0;
        for (int col = 0; col < tracked_prefix_columns; ++col) {
            tracked_right_weight = std::hypot(
                tracked_right_weight,
                svd.vt(index, col));
        }
        result.discarded_tracked_prefix_frobenius_norm = std::hypot(
            result.discarded_tracked_prefix_frobenius_norm,
            svd.singular_values[index] * tracked_right_weight);
    }
    statistics.orthogonalization_seconds += elapsed(start);
    return result;
}

double stable_ridge_filter(double singular_value, double regularization) {
    if (!(singular_value > 0.0) || !(regularization > 0.0)) {
        return 0.0;
    }
    const double regularization_root = std::sqrt(regularization);
    if (!(regularization_root > 0.0) ||
        !std::isfinite(regularization_root)) {
        return singular_value /
            (singular_value * singular_value + regularization);
    }
    if (singular_value <= regularization_root) {
        const double ratio = singular_value / regularization_root;
        return (ratio / (1.0 + ratio * ratio)) /
            regularization_root;
    }
    const double inverse_ratio =
        regularization_root / singular_value;
    return (1.0 / singular_value) /
        (1.0 + inverse_ratio * inverse_ratio);
}

}  // namespace

BlockGolubKahanWorkspace::BlockGolubKahanWorkspace(
    const MatrixOperator& matrix,
    const double* right_hand_side,
    double regularization,
    int maximum_basis_size,
    RunStatistics& statistics)
    : matrix_(matrix),
      right_hand_side_(matrix.rows(), 1),
      regularization_(regularization),
      basis_limit_(maximum_basis_size > 0
                       ? std::min(
                             maximum_basis_size,
                             std::min(matrix.rows(), matrix.cols()))
                       : std::min(matrix.rows(), matrix.cols())),
      left_basis_(matrix.rows(), 0),
      pending_left_seed_(matrix.rows(), 0),
      right_basis_(matrix.cols(), 0),
      projected_operator_(
          std::min(projected_row_limit(basis_limit_), 8), 0),
      frontier_coupling_(0, 0),
      statistics_(statistics) {
    if (matrix.rows() <= 0 || matrix.cols() <= 0) {
        throw std::invalid_argument(
            "block Golub-Kahan operator dimensions must be positive");
    }
    if (right_hand_side == nullptr) {
        throw std::invalid_argument(
            "block Golub-Kahan right-hand side must not be null");
    }
    if (!std::isfinite(regularization) || regularization < 0.0) {
        throw std::invalid_argument(
            "block Golub-Kahan regularization must be finite and nonnegative");
    }
    if (maximum_basis_size < 0) {
        throw std::invalid_argument(
            "block Golub-Kahan basis limit must be nonnegative");
    }
    for (int row = 0; row < matrix.rows(); ++row) {
        if (!std::isfinite(right_hand_side[row])) {
            throw std::invalid_argument(
                "block Golub-Kahan right-hand side must be finite");
        }
        right_hand_side_(row, 0) = right_hand_side[row];
    }
    right_hand_side_norm_ = vector_norm(
        right_hand_side_.data(), matrix_.rows());
}

void BlockGolubKahanWorkspace::apply_search(
    const Matrix& input,
    Matrix& output) {
    const auto start = Clock::now();
    matrix_.apply(input.data(), input.cols(), output.data());
    statistics_.a_seconds += elapsed(start);
    statistics_.a_columns += input.cols();
    statistics_.search_a_columns += input.cols();
    ++statistics_.a_block_calls;
    ++statistics_.search_a_block_calls;
}

void BlockGolubKahanWorkspace::apply_transpose_search(
    const Matrix& input,
    Matrix& output) {
    const auto start = Clock::now();
    matrix_.apply_transpose(input.data(), input.cols(), output.data());
    statistics_.at_seconds += elapsed(start);
    statistics_.at_columns += input.cols();
    statistics_.search_at_columns += input.cols();
    ++statistics_.at_block_calls;
    ++statistics_.search_at_block_calls;
}

void BlockGolubKahanWorkspace::append_projected_rhs(
    const Matrix& left_block) {
    for (int col = 0; col < left_block.cols(); ++col) {
        projected_rhs_.push_back(vector_dot(
            left_block.column_data(col),
            right_hand_side_.data(),
            matrix_.rows()));
    }
}

void BlockGolubKahanWorkspace::ensure_projected_row_capacity(
    int required_rows) {
    const int maximum_rows = projected_row_limit(basis_limit_);
    if (required_rows < 0 || required_rows > maximum_rows) {
        throw std::logic_error(
            "block Golub-Kahan projected row limit was exceeded");
    }
    if (required_rows <= projected_operator_.rows()) {
        return;
    }

    int grown_rows = std::max(1, projected_operator_.rows());
    while (grown_rows < required_rows) {
        if (grown_rows > maximum_rows / 2) {
            grown_rows = maximum_rows;
        } else {
            grown_rows *= 2;
        }
    }
    Matrix grown(grown_rows, projected_operator_.cols());
    set_block(grown, 0, 0, projected_operator_);
    projected_operator_ = std::move(grown);
}

Matrix BlockGolubKahanWorkspace::compact_projected_operator() const {
    return compact_projected_prefix(
        static_cast<int>(projected_rhs_.size()), right_basis_.cols());
}

Matrix BlockGolubKahanWorkspace::compact_projected_prefix(
    int rows,
    int cols) const {
    if (right_basis_.cols() != projected_operator_.cols() ||
        static_cast<int>(projected_rhs_.size()) >
            projected_operator_.rows() ||
        rows < 0 || cols < 0 ||
        rows > static_cast<int>(projected_rhs_.size()) ||
        rows > projected_operator_.rows() ||
        cols > right_basis_.cols() ||
        cols > projected_operator_.cols()) {
        throw std::logic_error(
            "block Golub-Kahan projected prefix is inconsistent");
    }
    Matrix compact(rows, cols);
    for (int col = 0; col < compact.cols(); ++col) {
        std::copy_n(
            projected_operator_.column_data(col),
            compact.rows(),
            compact.column_data(col));
    }
    return compact;
}

void BlockGolubKahanWorkspace::initialize(
    Matrix initial_auxiliary_left) {
    if (initialized_) {
        throw std::logic_error(
            "block Golub-Kahan workspace is already initialized");
    }
    if (initial_auxiliary_left.cols() > 0 &&
        initial_auxiliary_left.rows() != matrix_.rows()) {
        throw std::invalid_argument(
            "initial auxiliary block has the wrong row count");
    }
    initialized_ = true;
    last_step_audit_ = {};

    const auto start = Clock::now();
    const double right_hand_side_norm = right_hand_side_norm_;
    if (std::isfinite(right_hand_side_norm) &&
        right_hand_side_norm > 0.0 && basis_limit_ > 0) {
        Matrix first_left(matrix_.rows(), 1);
        const double inverse_norm = 1.0 / right_hand_side_norm;
        for (int row = 0; row < matrix_.rows(); ++row) {
            first_left(row, 0) = right_hand_side_(row, 0) * inverse_norm;
        }
        left_basis_ = std::move(first_left);
        projected_rhs_.push_back(right_hand_side_norm);
        active_left_ = {0, 1};
    }
    statistics_.orthogonalization_seconds += elapsed(start);

    if (right_hand_side_norm > 0.0 &&
        initial_auxiliary_left.cols() > 0) {
        append_left_seed(std::move(initial_auxiliary_left));
    }
}

int BlockGolubKahanWorkspace::append_left_seed(
    Matrix left_seed) {
    if (!initialized_) {
        throw std::logic_error(
            "block Golub-Kahan workspace is not initialized");
    }
    if (!continuation_reliable_) {
        throw std::logic_error(
            "cannot continue block Golub-Kahan after numerical truncation");
    }
    if (left_seed.cols() > 0 && left_seed.rows() != matrix_.rows()) {
        throw std::invalid_argument(
            "injected left block has the wrong row count");
    }
    if (pending_requested_columns_ == 0 &&
        pending_left_seed_.cols() == 0) {
        last_step_audit_ = {};
    }
    if (left_seed.cols() >
        std::numeric_limits<int>::max() - pending_requested_columns_) {
        throw std::overflow_error(
            "pending block Golub-Kahan width exceeds LP64 limits");
    }
    pending_requested_columns_ += left_seed.cols();
    last_step_audit_.requested_injected_columns =
        pending_requested_columns_;
    last_step_audit_.retained_injected_columns =
        pending_retained_columns_;
    if (left_seed.cols() == 0) {
        return 0;
    }
    const int available_pending_columns = std::max(
        0,
        basis_limit_ - right_basis_.cols() - left_basis_.cols() -
            pending_left_seed_.cols());
    const int maximum_rank = std::min({
        available_pending_columns,
        matrix_.rows() - left_basis_.cols() -
            pending_left_seed_.cols(),
        left_seed.cols()});
    if (maximum_rank <= 0) {
        return 0;
    }

    // Stabilize Omega only against the physically available rolling
    // frontier (and any earlier pending columns).  The resulting P*Omega is
    // a pending A* input, not a new conceptual U block: no T row or g entry
    // is created here.
    Matrix joined_left;
    const Matrix* local_left = &left_basis_;
    if (!pending_left_seed_.empty()) {
        joined_left = join_columns(left_basis_, pending_left_seed_);
        local_left = &joined_left;
    }
    const double reference_scale = left_seed.cols() == 1
        ? vector_norm(left_seed.data(), left_seed.rows())
        : frobenius_norm(left_seed);
    OrthogonalBlockFactor injected = orthogonalize_and_factor(
        std::move(left_seed),
        *local_left,
        maximum_rank,
        matrix_.rows(),
        matrix_.cols(),
        reference_scale,
        0,
        false,
        statistics_,
        scalar_projection_coefficients_,
        scalar_accumulated_coefficients_);
    const int retained = injected.q.cols();
    if (retained <= 0) {
        return 0;
    }
    if (pending_left_seed_.empty()) {
        pending_left_seed_ = std::move(injected.q);
    } else {
        pending_left_seed_.append_columns(injected.q);
    }
    pending_retained_columns_ += retained;
    last_step_audit_.retained_injected_columns =
        pending_retained_columns_;
    return retained;
}

bool BlockGolubKahanWorkspace::advance_level() {
    const int requested_injected_columns = pending_requested_columns_;
    const int retained_injected_columns = pending_retained_columns_;
    last_step_audit_ = {};
    last_step_audit_.requested_injected_columns =
        requested_injected_columns;
    last_step_audit_.retained_injected_columns =
        retained_injected_columns;
    last_step_audit_.basis_rank_before = right_basis_.cols();
    last_step_audit_.basis_rank_after = right_basis_.cols();
    last_step_audit_.left_input_width =
        left_basis_.cols() + pending_left_seed_.cols();
    if (!can_advance()) {
        return false;
    }
    if (left_basis_.cols() != active_left_.width ||
        active_left_.first + active_left_.width !=
            static_cast<int>(projected_rhs_.size())) {
        throw std::logic_error(
            "block Golub-Kahan active left frontier is inconsistent");
    }

    const int old_right_columns = right_basis_.cols();
    const int old_left_columns =
        static_cast<int>(projected_rhs_.size());
    const int active_left_width = left_basis_.cols();

    const Matrix& active_left = left_basis_;
    Matrix joined_left;
    const Matrix* combined_left_pointer = &active_left;
    if (!pending_left_seed_.empty()) {
        joined_left = join_columns(active_left, pending_left_seed_);
        combined_left_pointer = &joined_left;
    }
    const Matrix& combined_left = *combined_left_pointer;
    Matrix right_residual = take_scalar_residual_buffer(
        scalar_right_residual_buffer_,
        matrix_.cols(),
        combined_left.cols());
    apply_transpose_search(combined_left, right_residual);
    pending_left_seed_ = Matrix(matrix_.rows(), 0);
    pending_requested_columns_ = 0;
    pending_retained_columns_ = 0;
    const double right_reference_scale = combined_left.cols() == 1
        ? vector_norm(right_residual.data(), right_residual.rows())
        : frobenius_norm(right_residual);
    if (previous_right_.width > 0 && active_left_width > 0) {
        if (frontier_coupling_.rows() != active_left_width ||
            frontier_coupling_.cols() != previous_right_.width) {
            throw std::logic_error(
                "block Golub-Kahan frontier coupling dimensions are invalid");
        }
        Matrix extended_coupling;
        const Matrix* combined_coupling = &frontier_coupling_;
        if (combined_left.cols() != active_left_width) {
            extended_coupling = Matrix(
                combined_left.cols(), previous_right_.width);
            set_block(extended_coupling, 0, 0, frontier_coupling_);
            combined_coupling = &extended_coupling;
        }
        gemm_column_block(
            right_basis_,
            previous_right_.first,
            previous_right_.width,
            *combined_coupling,
            true,
            -1.0,
            1.0,
            right_residual);
    }

    const int maximum_right_rank = std::min(
        combined_left.cols(),
        basis_limit_ - right_basis_.cols());
    OrthogonalBlockFactor right_factor = orthogonalize_and_factor(
        std::move(right_residual),
        right_basis_,
        maximum_right_rank,
        matrix_.rows(),
        matrix_.cols(),
        right_reference_scale,
        active_left_width,
        false,
        statistics_,
        scalar_projection_coefficients_,
        scalar_accumulated_coefficients_);
    const int retained_right_width = right_factor.q.cols();
    last_step_audit_.retained_right_width = retained_right_width;
    last_step_audit_.right_rank_cutoff = right_factor.rank_cutoff;
    last_step_audit_.right_discarded_residual_norm =
        right_factor.discarded_frobenius_norm;
    last_step_audit_.right_active_discarded_residual_norm =
        right_factor.discarded_tracked_prefix_frobenius_norm;
    last_step_audit_.right_positive_finite_below_cutoff =
        right_factor.positive_finite_below_cutoff;
    last_step_audit_.right_nonfinite_residual =
        right_factor.nonfinite_residual;
    last_step_audit_.right_rank_loss =
        retained_right_width < maximum_right_rank;
    last_step_audit_.right_basis_cap_truncation =
        maximum_right_rank < combined_left.cols();
    if (step_has_numerical_rank_truncation(last_step_audit_) ||
        step_has_nonfinite_breakdown(last_step_audit_)) {
        continuation_reliable_ = false;
    }
    if (retained_right_width == 0) {
        left_basis_ = Matrix(matrix_.rows(), 0);
        active_left_ = {old_left_columns, 0};
        previous_right_ = {right_basis_.cols(), 0};
        frontier_coupling_ = Matrix(0, 0);
        return false;
    }

    Matrix new_right = std::move(right_factor.q);

    Matrix left_residual = take_scalar_residual_buffer(
        scalar_left_residual_buffer_,
        matrix_.rows(),
        retained_right_width);
    apply_search(new_right, left_residual);
    const double left_reference_scale = left_residual.cols() == 1
        ? vector_norm(left_residual.data(), left_residual.rows())
        : frobenius_norm(left_residual);
    Matrix leading_active_coupling;
    const Matrix* active_coupling = &right_factor.factor;
    if (right_factor.factor.cols() != active_left_width) {
        leading_active_coupling = Matrix(
            retained_right_width, active_left_width);
        for (int col = 0; col < active_left_width; ++col) {
            std::copy_n(
                right_factor.factor.column_data(col),
                retained_right_width,
                leading_active_coupling.column_data(col));
        }
        active_coupling = &leading_active_coupling;
    }
    if (active_left_width > 0) {
        gemm(
            active_left,
            false,
            *active_coupling,
            true,
            -1.0,
            1.0,
            left_residual);
    }

    const int maximum_left_rank = std::min({
        retained_right_width,
        matrix_.rows() - old_left_columns,
        projected_row_limit(basis_limit_) - old_left_columns});
    OrthogonalBlockFactor left_factor = orthogonalize_and_factor(
        std::move(left_residual),
        active_left,
        maximum_left_rank,
        matrix_.rows(),
        matrix_.cols(),
        left_reference_scale,
        0,
        true,
        statistics_,
        scalar_projection_coefficients_,
        scalar_accumulated_coefficients_);
    const int retained_next_left_width = left_factor.q.cols();
    last_step_audit_.retained_next_left_width =
        retained_next_left_width;
    last_step_audit_.left_rank_cutoff = left_factor.rank_cutoff;
    last_step_audit_.left_discarded_residual_norm =
        left_factor.discarded_frobenius_norm;
    last_step_audit_.left_positive_finite_below_cutoff =
        left_factor.positive_finite_below_cutoff;
    last_step_audit_.left_nonfinite_residual =
        left_factor.nonfinite_residual;
    last_step_audit_.left_rank_loss =
        retained_next_left_width < maximum_left_rank;
    last_step_audit_.left_basis_cap_truncation =
        maximum_left_rank < retained_right_width;
    if (step_has_numerical_rank_truncation(last_step_audit_) ||
        step_has_nonfinite_breakdown(last_step_audit_)) {
        continuation_reliable_ = false;
    }
    if (left_factor.nonfinite_residual) {
        // The right half-step is only provisional until A*'s new right block
        // has a finite A image and therefore a completed T column.  Leave
        // V/T/g/depth at the previously validated checkpoint.
        return false;
    }

    if (right_basis_.empty()) {
        right_basis_ = std::move(new_right);
    } else {
        right_basis_.append_columns(new_right);
        if (new_right.cols() == 1) {
            scalar_right_residual_buffer_ = std::move(new_right);
        }
    }

    // T's leading dimension grows geometrically, never once per level.
    // Appending a right block then adds only zero columns at the current
    // capacity, leaving all prior column offsets unchanged.
    ensure_projected_row_capacity(
        old_left_columns + retained_next_left_width);
    projected_operator_.append_zero_columns(retained_right_width);
    for (int new_col = 0;
         new_col < retained_right_width;
         ++new_col) {
        for (int local_row = 0;
             local_row < active_left_width;
             ++local_row) {
            projected_operator_(
                active_left_.first + local_row,
                old_right_columns + new_col) =
                (*active_coupling)(new_col, local_row) +
                left_factor.coefficients(local_row, new_col);
        }
        for (int new_row = 0;
             new_row < retained_next_left_width;
             ++new_row) {
            projected_operator_(
                old_left_columns + new_row,
                old_right_columns + new_col) =
                left_factor.factor(new_row, new_col);
        }
    }

    Matrix consumed_left_basis = std::move(left_basis_);
    left_basis_ = std::move(left_factor.q);
    if (consumed_left_basis.cols() == 1) {
        scalar_left_residual_buffer_ = std::move(consumed_left_basis);
    }
    append_projected_rhs(left_basis_);

    active_left_ = {
        old_left_columns,
        retained_next_left_width};
    previous_right_ = {
        old_right_columns,
        retained_right_width};
    frontier_coupling_ = std::move(left_factor.factor);
    ++depth_;
    last_step_audit_.basis_rank_after = right_basis_.cols();
    return true;
}

LeastSquaresResult BlockGolubKahanWorkspace::evaluate(
    double tolerance,
    int epoch,
    int auxiliary_width) {
    if (!initialized_) {
        throw std::logic_error(
            "block Golub-Kahan workspace is not initialized");
    }
    if (!std::isfinite(tolerance) || tolerance <= 0.0 ||
        tolerance >= 1.0) {
        throw std::invalid_argument(
            "block Golub-Kahan tolerance must lie strictly between zero and one");
    }
    const auto solve_start = Clock::now();
    Matrix projected_operator = compact_projected_operator();
    if (projected_operator.rows() !=
            static_cast<int>(projected_rhs_.size()) ||
        projected_operator.cols() != right_basis_.cols()) {
        throw std::logic_error(
            "block Golub-Kahan projected data are inconsistent");
    }

    std::vector<double> solution;
    if (right_basis_.cols() > 0) {
        Matrix reduced_coefficients(right_basis_.cols(), 1);
        const double relative_tolerance = std::min(
            0.5,
            64.0 * std::numeric_limits<double>::epsilon() *
                std::max(
                    projected_operator.rows(),
                    projected_operator.cols()));
        if (regularization_ == 0.0) {
            MinimumNormLeastSquaresResult least_squares =
                minimum_norm_least_squares(
                    std::move(projected_operator),
                    projected_rhs_.data(),
                    relative_tolerance);
            std::copy(
                least_squares.solution.begin(),
                least_squares.solution.end(),
                reduced_coefficients.data());
        } else {
            SvdResult svd = thin_svd(std::move(projected_operator));
            std::vector<double> spectral_coefficients(
                svd.singular_values.size(), 0.0);
            for (int index = 0;
                 index < static_cast<int>(svd.singular_values.size());
                 ++index) {
                const double left_coefficient = vector_dot(
                    svd.u.column_data(index),
                    projected_rhs_.data(),
                    svd.u.rows());
                spectral_coefficients[index] = stable_ridge_filter(
                    svd.singular_values[index], regularization_) *
                    left_coefficient;
            }
            for (int col = 0; col < right_basis_.cols(); ++col) {
                long double value = 0.0L;
                for (int index = 0;
                     index < static_cast<int>(spectral_coefficients.size());
                     ++index) {
                    value += static_cast<long double>(
                        svd.vt(index, col)) *
                        spectral_coefficients[index];
                }
                reduced_coefficients(col, 0) =
                    static_cast<double>(value);
            }
        }
        Matrix full_solution = multiply(
            right_basis_, reduced_coefficients);
        full_solution.give_values_to(solution);
    } else {
        solution.assign(
            static_cast<std::size_t>(matrix_.cols()), 0.0);
    }
    statistics_.projected_solve_seconds += elapsed(solve_start);

    LeastSquaresResult result = validate_original_candidate(
        matrix_,
        right_hand_side_.data(),
        solution,
        regularization_,
        tolerance,
        0.0,
        statistics_);
    result.depth = depth_;
    result.auxiliary_width = auxiliary_width;
    result.basis_rank = right_basis_.cols();
    result.trace.push_back(make_iteration_record(
        result, epoch, depth_, auxiliary_width));
    return result;
}

int BlockGolubKahanWorkspace::depth() const noexcept { return depth_; }
int BlockGolubKahanWorkspace::basis_rank() const noexcept {
    return right_basis_.cols();
}
int BlockGolubKahanWorkspace::basis_limit() const noexcept {
    return basis_limit_;
}
int BlockGolubKahanWorkspace::active_left_width() const noexcept {
    return active_left_.width;
}
bool BlockGolubKahanWorkspace::can_advance() const noexcept {
    return initialized_ &&
        continuation_reliable_ &&
        left_basis_.cols() + pending_left_seed_.cols() > 0 &&
        right_basis_.cols() < basis_limit_;
}

double BlockGolubKahanWorkspace::right_hand_side_norm() const noexcept {
    return right_hand_side_norm_;
}
const Matrix& BlockGolubKahanWorkspace::right_basis() const noexcept {
    return right_basis_;
}
const Matrix& BlockGolubKahanWorkspace::left_basis() const noexcept {
    return left_basis_;
}
Matrix BlockGolubKahanWorkspace::projected_operator() const {
    return compact_projected_operator();
}
const std::vector<double>&
BlockGolubKahanWorkspace::projected_rhs() const noexcept {
    return projected_rhs_;
}
const BlockGolubKahanStepAudit&
BlockGolubKahanWorkspace::last_step_audit() const noexcept {
    return last_step_audit_;
}

bool step_has_numerical_rank_truncation(
    const BlockGolubKahanStepAudit& step) noexcept {
    return
        (step.right_rank_loss &&
         step.right_positive_finite_below_cutoff &&
         step.right_active_discarded_residual_norm > 0.0) ||
        (step.left_rank_loss &&
         step.left_positive_finite_below_cutoff);
}

bool step_has_nonfinite_breakdown(
    const BlockGolubKahanStepAudit& step) noexcept {
    return step.right_nonfinite_residual ||
        step.left_nonfinite_residual;
}

Matrix orthogonalize_krylov_block(
    Matrix block,
    const Matrix& basis,
    int basis_limit,
    int operator_rows,
    int operator_cols,
    RunStatistics& statistics,
    std::vector<double>* coefficient_workspace,
    KrylovOrthogonalizationAudit* audit) {
    if (operator_rows <= 0 || operator_cols <= 0 ||
        block.rows() != operator_cols || basis.rows() != operator_cols ||
        basis_limit < basis.cols() ||
        basis_limit > std::min(operator_rows, operator_cols)) {
        throw std::invalid_argument(
            "invalid Krylov orthogonalization dimensions");
    }
    const auto start = Clock::now();
    const int block_columns = block.cols();
    const double input_scale = block_columns == 1
        ? vector_norm(block.data(), block.rows())
        : frobenius_norm(block);
    Matrix residual = std::move(block);
    if (block_columns == 1 && basis.cols() > 0) {
        std::vector<double> local_coefficients;
        std::vector<double>& coefficients = coefficient_workspace != nullptr
            ? *coefficient_workspace
            : local_coefficients;
        coefficients.resize(basis.cols());
        for (int pass = 0; pass < 2; ++pass) {
            gemv(
                basis,
                true,
                residual.data(),
                1.0,
                0.0,
                coefficients.data());
            gemv(
                basis,
                false,
                coefficients.data(),
                -1.0,
                1.0,
                residual.data());
        }
    } else {
        for (int pass = 0; pass < 2 && basis.cols() > 0; ++pass) {
            Matrix coefficients = transpose_multiply(basis, residual);
            gemm(basis, false, coefficients, false, -1.0, 1.0, residual);
        }
    }
    // A relative cutoff based only on `residual` is unsafe at numerical
    // breakdown: it normalizes a projection residual made entirely of
    // roundoff and promotes that noise to a unit search direction.  Compare
    // against the unprojected block as well.  The dimension term covers the
    // accumulated dot-product error in the two reorthogonalization passes;
    // For a multi-column seed, sqrt(epsilon) rejects a weak column whose
    // normalization would magnify rounding error relative to stronger peers.
    // A scalar Krylov extension has no within-block scale competition; using
    // the same threshold there can falsely deflate a meaningful tail
    // direction after a large spectral outlier.  Reaching either guard is
    // numerical deflation, not a claim about the exact rank of the operator.
    const int dimension = std::max({
        operator_rows, operator_cols, block_columns, basis.cols()});
    const double machine_epsilon = std::numeric_limits<double>::epsilon();
    const double dot_product_tolerance =
        64.0 * machine_epsilon * dimension;
    const double breakdown_relative_tolerance = block_columns == 1
        ? dot_product_tolerance
        : std::max(dot_product_tolerance, std::sqrt(machine_epsilon));
    const double breakdown_cutoff =
        breakdown_relative_tolerance * input_scale;
    const double projected_residual_norm = block_columns == 1
        ? vector_norm(residual.data(), residual.rows())
        : frobenius_norm(residual);
    Matrix q;
    if (residual.cols() == 1) {
        // The left singular vector of a single nonzero column is exactly the
        // normalized column (up to an immaterial sign).  Avoiding a general
        // SVD here removes one LAPACK setup and allocation from every scalar
        // Krylov extension while retaining the same absolute breakdown test.
        const double residual_norm = projected_residual_norm;
        if (std::isfinite(residual_norm) &&
            residual_norm > breakdown_cutoff) {
            const double inverse_norm = 1.0 / residual_norm;
            for (int row = 0; row < residual.rows(); ++row) {
                residual(row, 0) *= inverse_norm;
            }
            q = std::move(residual);
        } else {
            q = Matrix(residual.rows(), 0);
        }
    } else {
        q = numerical_column_space(residual, -1.0, breakdown_cutoff);
    }
    const int available = basis_limit - basis.cols();
    if (q.cols() > available) {
        q.truncate_columns(std::max(0, available));
    }
    if (audit != nullptr) {
        audit->projected_residual_norm = projected_residual_norm;
        audit->rank_cutoff = breakdown_cutoff;
        audit->retained_columns = q.cols();
        audit->nonfinite_residual =
            !std::isfinite(projected_residual_norm);
        audit->positive_finite_below_cutoff =
            available > 0 && q.cols() == 0 &&
            projected_residual_norm > 0.0 &&
            std::isfinite(projected_residual_norm);
    }
    statistics.orthogonalization_seconds += elapsed(start);
    return q;
}

ProjectedCandidate solve_projected_basis(
    const double* right_hand_side,
    double regularization,
    const Matrix& basis,
    const Matrix& applied_basis,
    RunStatistics& statistics) {
    if (right_hand_side == nullptr) {
        throw std::invalid_argument(
            "projected solve right-hand side must not be null");
    }
    if (!std::isfinite(regularization) || regularization < 0.0) {
        throw std::invalid_argument(
            "projected solve regularization must be finite and nonnegative");
    }
    if (basis.rows() <= 0 || applied_basis.rows() <= 0 ||
        basis.cols() != applied_basis.cols()) {
        throw std::invalid_argument(
            "projected basis and applied basis dimensions do not match");
    }
    for (int row = 0; row < applied_basis.rows(); ++row) {
        if (!std::isfinite(right_hand_side[row])) {
            throw std::invalid_argument(
                "projected solve right-hand side must be finite");
        }
    }
    const auto solve_start = Clock::now();
    Matrix reduced_coefficients(basis.cols(), 1);
    if (regularization == 0.0) {
        const double relative_tolerance =
            16.0 * std::numeric_limits<double>::epsilon() *
            std::max(applied_basis.rows(), applied_basis.cols());
        MinimumNormLeastSquaresResult least_squares =
            minimum_norm_least_squares(
                applied_basis, right_hand_side, relative_tolerance);
        std::copy(
            least_squares.solution.begin(),
            least_squares.solution.end(),
            reduced_coefficients.data());
    } else {
        SvdResult svd = thin_svd(applied_basis);
        std::vector<double> spectral_coefficients(
            svd.singular_values.size(), 0.0);
        for (int index = 0;
             index < static_cast<int>(svd.singular_values.size());
             ++index) {
            const double sigma = svd.singular_values[index];
            const double left_coefficient = vector_dot(
                svd.u.column_data(index),
                right_hand_side,
                applied_basis.rows());
            spectral_coefficients[index] =
                sigma * left_coefficient /
                (sigma * sigma + regularization);
        }
        for (int col = 0; col < basis.cols(); ++col) {
            long double value = 0.0L;
            for (int index = 0;
                 index < static_cast<int>(spectral_coefficients.size());
                 ++index) {
                value += svd.vt(index, col) * spectral_coefficients[index];
            }
            reduced_coefficients(col, 0) = static_cast<double>(value);
        }
    }
    Matrix solution = multiply(basis, reduced_coefficients);
    statistics.projected_solve_seconds += elapsed(solve_start);

    ProjectedCandidate candidate;
    solution.give_values_to(candidate.solution);
    candidate.basis_rank = basis.cols();
    return candidate;
}

LeastSquaresResult evaluate_projected_basis(
    const MatrixOperator& validation_matrix,
    const double* right_hand_side,
    double regularization,
    double tolerance,
    const Matrix& basis,
    const Matrix& applied_basis,
    RunStatistics& statistics,
    int epoch,
    int depth,
    int auxiliary_width) {
    if (basis.rows() != validation_matrix.cols() ||
        applied_basis.rows() != validation_matrix.rows()) {
        throw std::invalid_argument(
            "projected basis and applied basis dimensions do not match");
    }
    ProjectedCandidate projected = solve_projected_basis(
        right_hand_side,
        regularization,
        basis,
        applied_basis,
        statistics);

    LeastSquaresResult result = validate_original_candidate(
        validation_matrix,
        right_hand_side,
        projected.solution,
        regularization,
        tolerance,
        0.0,
        statistics);
    result.depth = depth;
    result.auxiliary_width = auxiliary_width;
    result.basis_rank = projected.basis_rank;
    result.trace.push_back(make_iteration_record(
        result, epoch, depth, auxiliary_width));
    return result;
}

IterationRecord make_iteration_record(
    const LeastSquaresResult& result,
    int epoch,
    int depth,
    int auxiliary_width) {
    IterationRecord record;
    record.epoch = epoch;
    record.depth = depth;
    record.auxiliary_width = auxiliary_width;
    record.basis_rank = result.basis_rank;
    record.a_columns = result.statistics.a_columns;
    record.at_columns = result.statistics.at_columns;
    record.a_block_calls = result.statistics.a_block_calls;
    record.at_block_calls = result.statistics.at_block_calls;
    record.contract_status = result.status;
    record.contract_stop_reason = result.stop_reason;
    record.contract_passed = result.status == SolverStatus::success;
    record.ridge_correction_disposition =
        result.ridge_correction_disposition;
    record.objective = result.objective;
    record.residual_norm = result.residual_norm;
    record.augmented_residual_norm = result.augmented_residual_norm;
    record.normal_residual_norm = result.normal_residual_norm;
    record.operator_norm_lower_bound = result.operator_norm_lower_bound;
    record.compatible_backward_error_upper_bound =
        result.compatible_backward_error_upper_bound;
    record.least_squares_backward_error_upper_bound =
        result.least_squares_backward_error_upper_bound;
    record.backward_error_upper_bound = result.backward_error_upper_bound;
    record.relative_energy_error_upper_bound =
        result.relative_energy_error_upper_bound;
    record.relative_normal_residual_upper_bound =
        result.relative_normal_residual_upper_bound;
    record.solution_norm = result.solution_norm;
    record.solution_energy_norm = result.solution_energy_norm;
    record.augmented_operator_norm_lower_bound =
        result.augmented_operator_norm_lower_bound;
    record.energy_error_upper_bound = result.energy_error_upper_bound;
    record.objective_gap_upper_bound = result.objective_gap_upper_bound;
    record.ridge_base_energy_error_upper_bound =
        result.ridge_base_energy_error_upper_bound;
    record.ridge_corrected_energy_error_upper_bound =
        result.ridge_corrected_energy_error_upper_bound;
    record.ridge_correction_gamma = result.ridge_correction_gamma;
    record.ridge_correction_z_h_z = result.ridge_correction_z_h_z;
    record.ridge_correction_two_abs_z_t_q =
        result.ridge_correction_two_abs_z_t_q;
    record.ridge_correction_q_norm_squared =
        result.ridge_correction_q_norm_squared;
    return record;
}

}  // namespace amfls::math
