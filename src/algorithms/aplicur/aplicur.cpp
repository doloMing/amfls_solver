#include "amfls/aplicur.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numeric>
#include <random>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

#include "algorithms/mathematics/krylov/candidate_validation.hpp"
#include "algorithms/mathematics/krylov/certificate_checkpoint_schedule.hpp"
#include "algorithms/mathematics/krylov/krylov_workspace.hpp"
#include "algorithms/mathematics/linalg/blas_lapack.hpp"
#include "algorithms/mathematics/linalg/matrix.hpp"
#include "algorithms/mathematics/linalg/svd.hpp"
#include "algorithms/mathematics/operators/row_major_dense_operator.hpp"

extern "C" {
void dgeqrf_(
    const int*, const int*, double*, const int*, double*, double*, const int*,
    int*);
void dorgqr_(
    const int*, const int*, const int*, double*, const int*, const double*,
    double*, const int*, int*);
void dgetrf_(const int*, const int*, double*, const int*, int*, int*);
void dgetrs_(
    const char*, const int*, const int*, const double*, const int*, const int*,
    double*, const int*, int*);
void dgecon_(
    const char*, const int*, const double*, const int*, const double*, double*,
    double*, int*, int*);
}

namespace amfls {
namespace {

using Clock = std::chrono::steady_clock;

struct Settings {
    double regularization = 0.0;
    double tolerance = 0.0;
    double cur_tolerance = 0.0;
    double re_preconditioning_tolerance = 0.0;
    double dynamic_stopping_tolerance = 0.0;
    double relative_rank_tolerance = 0.0;
    double absolute_rank_tolerance = 0.0;
    int block_size = 0;
    int sparse_sign_nonzeros = 0;
    int spectral_probe_count = 0;
    int maximum_iterations = 0;
};

struct CurState {
    std::vector<int> row_indices;
    std::vector<int> column_indices;
    math::Matrix columns;
    math::Matrix sketched_columns;
    math::Matrix row_transpose;
    math::Matrix row_q;
    math::Matrix row_upper;
    math::Matrix core_factor;
    std::vector<int> core_pivots;
    bool core_uses_pseudoinverse = false;
    math::Matrix sketched_residual;
    double residual_estimate = std::numeric_limits<double>::infinity();
};

struct QrFactor {
    math::Matrix orthonormal;
    math::Matrix upper;
};

struct Preconditioner {
    math::Matrix right_vectors;
    std::vector<double> singular_values;
    double target = 1.0;

    int rank() const {
        return static_cast<int>(singular_values.size());
    }

    double smallest_singular_value() const {
        return singular_values.empty() ? 0.0 : singular_values.back();
    }
};

struct PhaseOutcome {
    LeastSquaresResult result;
    std::vector<double> solution;
    bool numerical_failure = false;
    bool exhausted = false;
    bool dynamic_stop = false;
};

// Non-owning view of the explicit storage APLICUR needs for row/column
// selection and its sparse-sign sketch.  Iterative A/A^T products use the
// RowMajorDenseOperator over the same values.
class RowMajorMatrixView {
public:
    RowMajorMatrixView(int rows, int cols, const double* values)
        : rows_(rows), cols_(cols), values_(values) {}

    int rows() const { return rows_; }
    int cols() const { return cols_; }

    const double* row_data(int row) const {
        return values_ + static_cast<std::size_t>(row) * cols_;
    }

    double operator()(int row, int col) const {
        return row_data(row)[col];
    }

private:
    int rows_;
    int cols_;
    const double* values_;
};

double elapsed(Clock::time_point start) {
    return std::chrono::duration<double>(Clock::now() - start).count();
}

bool all_finite(std::span<const double> values) {
    return std::all_of(
        values.begin(), values.end(), [](double value) {
            return std::isfinite(value);
        });
}

bool all_finite(const math::Matrix& matrix) {
    return all_finite(
        std::span<const double>(
            matrix.data(), static_cast<std::size_t>(matrix.size())));
}

void scale(std::vector<double>& values, double factor) {
    for (double& value : values) {
        value *= factor;
    }
}

math::Matrix transpose(const math::Matrix& matrix) {
    math::Matrix result(matrix.cols(), matrix.rows());
    for (int column = 0; column < matrix.cols(); ++column) {
        for (int row = 0; row < matrix.rows(); ++row) {
            result(column, row) = matrix(row, column);
        }
    }
    return result;
}

math::Matrix select_columns(
    const math::Matrix& matrix,
    std::span<const int> indices) {
    math::Matrix result(matrix.rows(), static_cast<int>(indices.size()));
    for (int column = 0; column < static_cast<int>(indices.size()); ++column) {
        const int source = indices[static_cast<std::size_t>(column)];
        if (source < 0 || source >= matrix.cols()) {
            throw std::invalid_argument("APLICUR column index is out of range");
        }
        std::copy(
            matrix.column_data(source),
            matrix.column_data(source) + matrix.rows(),
            result.column_data(column));
    }
    return result;
}

math::Matrix select_columns(
    const RowMajorMatrixView& matrix,
    std::span<const int> indices) {
    math::Matrix result(matrix.rows(), static_cast<int>(indices.size()));
    for (int column = 0; column < static_cast<int>(indices.size()); ++column) {
        const int source = indices[static_cast<std::size_t>(column)];
        if (source < 0 || source >= matrix.cols()) {
            throw std::invalid_argument("APLICUR column index is out of range");
        }
        for (int row = 0; row < matrix.rows(); ++row) {
            result(row, column) = matrix(row, source);
        }
    }
    return result;
}

math::Matrix select_rows(
    const RowMajorMatrixView& matrix,
    std::span<const int> indices) {
    math::Matrix result(static_cast<int>(indices.size()), matrix.cols());
    for (int column = 0; column < matrix.cols(); ++column) {
        for (int row = 0; row < static_cast<int>(indices.size()); ++row) {
            const int source = indices[static_cast<std::size_t>(row)];
            if (source < 0 || source >= matrix.rows()) {
                throw std::invalid_argument("APLICUR row index is out of range");
            }
            result(row, column) = matrix(source, column);
        }
    }
    return result;
}

math::Matrix intersection(
    const RowMajorMatrixView& matrix,
    std::span<const int> row_indices,
    std::span<const int> column_indices) {
    math::Matrix result(
        static_cast<int>(row_indices.size()),
        static_cast<int>(column_indices.size()));
    for (int column = 0; column < static_cast<int>(column_indices.size());
         ++column) {
        for (int row = 0; row < static_cast<int>(row_indices.size()); ++row) {
            result(row, column) = matrix(
                row_indices[static_cast<std::size_t>(row)],
                column_indices[static_cast<std::size_t>(column)]);
        }
    }
    return result;
}

int retained_rank(
    const math::SvdResult& svd,
    double relative_tolerance,
    double absolute_tolerance) {
    if (svd.singular_values.empty() ||
        !(svd.singular_values.front() > 0.0) ||
        !std::isfinite(svd.singular_values.front())) {
        return 0;
    }
    const double cutoff = std::max(
        relative_tolerance * svd.singular_values.front(),
        absolute_tolerance);
    int rank = 0;
    while (rank < static_cast<int>(svd.singular_values.size()) &&
           svd.singular_values[static_cast<std::size_t>(rank)] > cutoff &&
           std::isfinite(svd.singular_values[static_cast<std::size_t>(rank)])) {
        ++rank;
    }
    return rank;
}

math::Matrix pseudoinverse(
    const math::Matrix& matrix,
    const Settings& settings) {
    const math::SvdResult svd = math::thin_svd(matrix);
    const int rank = retained_rank(
        svd,
        settings.relative_rank_tolerance,
        settings.absolute_rank_tolerance);
    math::Matrix result(matrix.cols(), matrix.rows());
    for (int component = 0; component < rank; ++component) {
        const double inverse =
            1.0 / svd.singular_values[static_cast<std::size_t>(component)];
        for (int column = 0; column < matrix.rows(); ++column) {
            const double left = svd.u(column, component);
            for (int row = 0; row < matrix.cols(); ++row) {
                result(row, column) +=
                    svd.vt(component, row) * inverse * left;
            }
        }
    }
    return result;
}

std::mt19937_64 make_engine(std::uint64_t seed) {
    std::seed_seq sequence{
        static_cast<std::uint32_t>(seed),
        static_cast<std::uint32_t>(seed >> 32U),
        std::uint32_t{0x41504c49U},
        std::uint32_t{0x43555231U}};
    return std::mt19937_64(sequence);
}

math::Matrix one_time_sparse_sign_sketch(
    const RowMajorMatrixView& matrix,
    int sketch_rows,
    int requested_nonzeros,
    std::mt19937_64& engine) {
    const int nonzeros = std::min(sketch_rows, requested_nonzeros);
    const double scale_factor = 1.0 / std::sqrt(static_cast<double>(nonzeros));
    math::Matrix sketch(sketch_rows, matrix.cols());
    std::uniform_int_distribution<int> location(0, sketch_rows - 1);
    std::vector<int> positions(static_cast<std::size_t>(nonzeros));
    for (int source_row = 0; source_row < matrix.rows(); ++source_row) {
        for (int entry = 0; entry < nonzeros; ++entry) {
            int candidate = 0;
            bool duplicate = false;
            do {
                candidate = location(engine);
                duplicate = std::find(
                    positions.begin(), positions.begin() + entry, candidate) !=
                    positions.begin() + entry;
            } while (duplicate);
            positions[static_cast<std::size_t>(entry)] = candidate;
        }
        for (int entry = 0; entry < nonzeros; ++entry) {
            const double sign = (engine() & std::uint64_t{1}) == 0
                ? -scale_factor
                : scale_factor;
            const int target_row = positions[static_cast<std::size_t>(entry)];
            const double* source = matrix.row_data(source_row);
            for (int column = 0; column < matrix.cols(); ++column) {
                sketch(target_row, column) += sign * source[column];
            }
        }
    }
    return sketch;
}

double spectral_residual_estimate(
    const math::Matrix& residual,
    int probe_count,
    std::mt19937_64& engine,
    RunStatistics& statistics) {
    std::normal_distribution<double> normal(0.0, 1.0);
    double maximum = 0.0;
    std::vector<double> probe(static_cast<std::size_t>(residual.cols()));
    std::vector<double> product(static_cast<std::size_t>(residual.rows()));
    for (int probe_index = 0; probe_index < probe_count; ++probe_index) {
        for (double& value : probe) {
            value = normal(engine);
        }
        std::fill(product.begin(), product.end(), 0.0);
        for (int column = 0; column < residual.cols(); ++column) {
            const double coefficient = probe[static_cast<std::size_t>(column)];
            for (int row = 0; row < residual.rows(); ++row) {
                product[static_cast<std::size_t>(row)] +=
                    residual(row, column) * coefficient;
            }
        }
        maximum = std::max(
            maximum,
            math::vector_norm(product.data(), residual.rows()));
    }
    statistics.gaussian_random_columns += probe_count;
    statistics.gaussian_random_values +=
        static_cast<long long>(residual.cols()) * probe_count;
    statistics.gaussian_random_block_requests += probe_count > 0 ? 1 : 0;
    constexpr double two_over_pi =
        0.636619772367581343075535053490057448;
    return 10.0 * std::sqrt(two_over_pi) * maximum;
}

std::vector<int> pivoted_lu_row_indices(
    math::Matrix matrix,
    int requested,
    const std::vector<bool>& forbidden) {
    if (requested < 0 || requested > matrix.rows() ||
        requested > matrix.cols() ||
        forbidden.size() != static_cast<std::size_t>(matrix.rows())) {
        throw std::invalid_argument("APLICUR pivot request is invalid");
    }
    std::vector<int> permutation(static_cast<std::size_t>(matrix.rows()));
    std::iota(permutation.begin(), permutation.end(), 0);
    for (int step = 0; step < requested; ++step) {
        int pivot_row = -1;
        double pivot_size = -1.0;
        for (int row = step; row < matrix.rows(); ++row) {
            if (forbidden[static_cast<std::size_t>(
                    permutation[static_cast<std::size_t>(row)])]) {
                continue;
            }
            const double candidate = std::abs(matrix(row, step));
            if (candidate > pivot_size) {
                pivot_size = candidate;
                pivot_row = row;
            }
        }
        if (pivot_row < 0) {
            throw std::runtime_error("APLICUR ran out of eligible pivots");
        }
        if (pivot_size == 0.0) {
            double best_tail_norm = -1.0;
            for (int row = step; row < matrix.rows(); ++row) {
                if (forbidden[static_cast<std::size_t>(
                        permutation[static_cast<std::size_t>(row)])]) {
                    continue;
                }
                long double squared_norm = 0.0L;
                for (int column = step; column < matrix.cols(); ++column) {
                    const long double value = matrix(row, column);
                    squared_norm += value * value;
                }
                const double tail_norm = static_cast<double>(squared_norm);
                if (tail_norm > best_tail_norm) {
                    best_tail_norm = tail_norm;
                    pivot_row = row;
                }
            }
        }
        if (pivot_row != step) {
            for (int column = 0; column < matrix.cols(); ++column) {
                std::swap(matrix(step, column), matrix(pivot_row, column));
            }
            std::swap(
                permutation[static_cast<std::size_t>(step)],
                permutation[static_cast<std::size_t>(pivot_row)]);
        }
        const double pivot = matrix(step, step);
        if (pivot != 0.0 && std::isfinite(pivot)) {
            for (int row = step + 1; row < matrix.rows(); ++row) {
                const double multiplier = matrix(row, step) / pivot;
                matrix(row, step) = multiplier;
                for (int column = step + 1; column < matrix.cols(); ++column) {
                    matrix(row, column) -= multiplier * matrix(step, column);
                }
            }
        }
    }
    permutation.resize(static_cast<std::size_t>(requested));
    return permutation;
}

double matrix_one_norm(const math::Matrix& matrix) {
    double result = 0.0;
    for (int column = 0; column < matrix.cols(); ++column) {
        long double sum = 0.0L;
        for (int row = 0; row < matrix.rows(); ++row) {
            sum += std::abs(matrix(row, column));
        }
        result = std::max(result, static_cast<double>(sum));
    }
    return result;
}

QrFactor thin_qr(const math::Matrix& matrix) {
    const int rows = matrix.rows();
    const int columns = matrix.cols();
    if (rows < columns) {
        throw std::invalid_argument(
            "APLICUR thin QR requires at least as many rows as columns");
    }
    QrFactor result{
        math::Matrix(rows, columns), math::Matrix(columns, columns)};
    if (columns == 0) {
        return result;
    }

    math::Matrix work = matrix;
    std::vector<double> reflectors(static_cast<std::size_t>(columns));
    const int leading_dimension = work.leading_dimension();
    int workspace_size = -1;
    double workspace_query = 0.0;
    int info = 0;
    dgeqrf_(
        &rows,
        &columns,
        work.data(),
        &leading_dimension,
        reflectors.data(),
        &workspace_query,
        &workspace_size,
        &info);
    if (info != 0) {
        throw std::runtime_error("APLICUR dgeqrf workspace query failed");
    }
    workspace_size = std::max(
        1, static_cast<int>(std::ceil(workspace_query)));
    std::vector<double> workspace(static_cast<std::size_t>(workspace_size));
    dgeqrf_(
        &rows,
        &columns,
        work.data(),
        &leading_dimension,
        reflectors.data(),
        workspace.data(),
        &workspace_size,
        &info);
    if (info != 0) {
        throw std::runtime_error("APLICUR dgeqrf failed");
    }
    for (int column = 0; column < columns; ++column) {
        for (int row = 0; row <= column; ++row) {
            result.upper(row, column) = work(row, column);
        }
    }

    workspace_size = -1;
    workspace_query = 0.0;
    dorgqr_(
        &rows,
        &columns,
        &columns,
        work.data(),
        &leading_dimension,
        reflectors.data(),
        &workspace_query,
        &workspace_size,
        &info);
    if (info != 0) {
        throw std::runtime_error("APLICUR dorgqr workspace query failed");
    }
    workspace_size = std::max(
        1, static_cast<int>(std::ceil(workspace_query)));
    workspace.assign(static_cast<std::size_t>(workspace_size), 0.0);
    dorgqr_(
        &rows,
        &columns,
        &columns,
        work.data(),
        &leading_dimension,
        reflectors.data(),
        workspace.data(),
        &workspace_size,
        &info);
    if (info != 0) {
        throw std::runtime_error("APLICUR dorgqr failed");
    }
    result.orthonormal = std::move(work);
    return result;
}

void append_columns(math::Matrix& target, const math::Matrix& block) {
    if (target.rows() == 0 && target.cols() == 0) {
        target = block;
    } else {
        target.append_columns(block);
    }
}

void append_qr(
    math::Matrix& orthonormal,
    math::Matrix& upper,
    const math::Matrix& block) {
    if (orthonormal.cols() == 0) {
        QrFactor factor = thin_qr(block);
        orthonormal = std::move(factor.orthonormal);
        upper = std::move(factor.upper);
        return;
    }
    if (orthonormal.rows() != block.rows() ||
        upper.rows() != orthonormal.cols() ||
        upper.cols() != orthonormal.cols()) {
        throw std::invalid_argument("APLICUR incremental QR dimensions mismatch");
    }

    const int old_rank = orthonormal.cols();
    const int increment = block.cols();
    math::Matrix first_projection(old_rank, increment);
    math::gemm(
        orthonormal,
        true,
        block,
        false,
        1.0,
        0.0,
        first_projection);
    math::Matrix residual = block;
    math::gemm(
        orthonormal,
        false,
        first_projection,
        false,
        -1.0,
        1.0,
        residual);
    math::Matrix second_projection(old_rank, increment);
    math::gemm(
        orthonormal,
        true,
        residual,
        false,
        1.0,
        0.0,
        second_projection);
    math::gemm(
        orthonormal,
        false,
        second_projection,
        false,
        -1.0,
        1.0,
        residual);

    QrFactor appended = thin_qr(residual);
    math::Matrix next_upper(old_rank + increment, old_rank + increment);
    math::set_block(next_upper, 0, 0, upper);
    for (int column = 0; column < increment; ++column) {
        for (int row = 0; row < old_rank; ++row) {
            next_upper(row, old_rank + column) =
                first_projection(row, column) +
                second_projection(row, column);
        }
    }
    math::set_block(
        next_upper, old_rank, old_rank, appended.upper);
    orthonormal.append_columns(appended.orthonormal);
    upper = std::move(next_upper);
}

void factor_core(
    const RowMajorMatrixView& matrix,
    CurState& state,
    const Settings& settings) {
    const math::Matrix core = intersection(
        matrix, state.row_indices, state.column_indices);
    const int rank = core.rows();
    if (rank == 0 || core.cols() != rank) {
        throw std::invalid_argument("APLICUR CUR core must be nonempty and square");
    }

    state.core_factor = core;
    state.core_pivots.assign(static_cast<std::size_t>(rank), 0);
    const int leading_dimension = state.core_factor.leading_dimension();
    int info = 0;
    dgetrf_(
        &rank,
        &rank,
        state.core_factor.data(),
        &leading_dimension,
        state.core_pivots.data(),
        &info);
    if (info < 0) {
        throw std::runtime_error("APLICUR dgetrf received an invalid argument");
    }

    bool use_pseudoinverse = info > 0;
    if (!use_pseudoinverse) {
        const double one_norm = matrix_one_norm(core);
        double reciprocal_condition = 0.0;
        std::vector<double> workspace(static_cast<std::size_t>(4 * rank));
        std::vector<int> integer_workspace(static_cast<std::size_t>(rank));
        const char norm = '1';
        dgecon_(
            &norm,
            &rank,
            state.core_factor.data(),
            &leading_dimension,
            &one_norm,
            &reciprocal_condition,
            workspace.data(),
            integer_workspace.data(),
            &info);
        if (info != 0) {
            throw std::runtime_error("APLICUR dgecon failed");
        }
        const double relative_cutoff = settings.relative_rank_tolerance;
        const double absolute_cutoff = one_norm > 0.0
            ? settings.absolute_rank_tolerance / one_norm
            : std::numeric_limits<double>::infinity();
        use_pseudoinverse =
            !std::isfinite(reciprocal_condition) ||
            reciprocal_condition <=
                std::max(relative_cutoff, absolute_cutoff);
    }
    state.core_uses_pseudoinverse = use_pseudoinverse;
    if (use_pseudoinverse) {
        state.core_factor = pseudoinverse(core, settings);
        state.core_pivots.clear();
    }
}

math::Matrix solve_core(
    const CurState& state,
    const math::Matrix& right_hand_sides) {
    const int rank = static_cast<int>(state.column_indices.size());
    if (right_hand_sides.rows() != rank ||
        state.core_factor.rows() != rank ||
        state.core_factor.cols() != rank) {
        throw std::invalid_argument("APLICUR CUR core solve dimensions mismatch");
    }
    if (state.core_uses_pseudoinverse) {
        return math::multiply(state.core_factor, right_hand_sides);
    }

    math::Matrix result = right_hand_sides;
    if (result.cols() == 0) {
        return result;
    }
    const char transpose_flag = 'N';
    const int right_hand_side_count = result.cols();
    const int factor_stride = state.core_factor.leading_dimension();
    const int result_stride = result.leading_dimension();
    int info = 0;
    dgetrs_(
        &transpose_flag,
        &rank,
        &right_hand_side_count,
        state.core_factor.data(),
        &factor_stride,
        state.core_pivots.data(),
        result.data(),
        &result_stride,
        &info);
    if (info != 0) {
        throw std::runtime_error("APLICUR dgetrs failed");
    }
    return result;
}

math::Matrix selected_cur_rows(
    const CurState& state,
    std::span<const int> column_indices) {
    const int rank = state.row_transpose.cols();
    math::Matrix result(rank, static_cast<int>(column_indices.size()));
    for (int column = 0; column < static_cast<int>(column_indices.size());
         ++column) {
        const int source = column_indices[static_cast<std::size_t>(column)];
        if (source < 0 || source >= state.row_transpose.rows()) {
            throw std::invalid_argument("APLICUR column index is out of range");
        }
        for (int row = 0; row < rank; ++row) {
            result(row, column) = state.row_transpose(source, row);
        }
    }
    return result;
}

math::Matrix current_cur_columns(
    const CurState& state,
    std::span<const int> column_indices) {
    if (state.column_indices.empty()) {
        return math::Matrix(
            state.columns.rows(), static_cast<int>(column_indices.size()));
    }
    const math::Matrix selected_rows = selected_cur_rows(
        state, column_indices);
    const math::Matrix middle = solve_core(state, selected_rows);
    return math::multiply(state.columns, middle);
}

void update_sketched_residual(
    const math::Matrix& sketch,
    CurState& state,
    const Settings& settings,
    std::mt19937_64& engine,
    RunStatistics& statistics) {
    const math::Matrix rows = transpose(state.row_transpose);
    const math::Matrix middle = solve_core(state, rows);
    const math::Matrix approximation = math::multiply(
        state.sketched_columns, middle);
    state.sketched_residual = sketch;
    for (int index = 0; index < state.sketched_residual.size(); ++index) {
        state.sketched_residual.data()[index] -= approximation.data()[index];
    }
    state.residual_estimate = spectral_residual_estimate(
        state.sketched_residual,
        settings.spectral_probe_count,
        engine,
        statistics);
}

void augment_cur(
    const RowMajorMatrixView& matrix,
    const math::Matrix& sketch,
    CurState& state,
    int increment,
    const Settings& settings,
    std::mt19937_64& engine,
    RunStatistics& statistics) {
    std::vector<bool> used_columns(
        static_cast<std::size_t>(matrix.cols()), false);
    for (int index : state.column_indices) {
        used_columns[static_cast<std::size_t>(index)] = true;
    }
    const math::Matrix row_residual_transpose = transpose(
        state.column_indices.empty() ? sketch : state.sketched_residual);
    const std::vector<int> new_columns = pivoted_lu_row_indices(
        row_residual_transpose, increment, used_columns);

    math::Matrix column_residual = select_columns(matrix, new_columns);
    if (!state.column_indices.empty()) {
        const math::Matrix approximation = current_cur_columns(
            state, new_columns);
        for (int index = 0; index < column_residual.size(); ++index) {
            column_residual.data()[index] -= approximation.data()[index];
        }
    }
    std::vector<bool> used_rows(
        static_cast<std::size_t>(matrix.rows()), false);
    for (int index : state.row_indices) {
        used_rows[static_cast<std::size_t>(index)] = true;
    }
    const std::vector<int> new_rows = pivoted_lu_row_indices(
        std::move(column_residual), increment, used_rows);

    const math::Matrix new_column_block = select_columns(matrix, new_columns);
    const math::Matrix new_sketched_column_block = select_columns(
        sketch, new_columns);
    const math::Matrix new_row_transpose = transpose(
        select_rows(matrix, new_rows));
    state.column_indices.insert(
        state.column_indices.end(), new_columns.begin(), new_columns.end());
    state.row_indices.insert(
        state.row_indices.end(), new_rows.begin(), new_rows.end());
    append_columns(state.columns, new_column_block);
    append_columns(state.sketched_columns, new_sketched_column_block);
    append_columns(state.row_transpose, new_row_transpose);
    append_qr(state.row_q, state.row_upper, new_row_transpose);
    factor_core(matrix, state, settings);
    update_sketched_residual(
        sketch, state, settings, engine, statistics);
}

Preconditioner build_preconditioner(
    const CurState& state,
    const Settings& settings) {
    Preconditioner result;
    result.right_vectors = math::Matrix(state.row_transpose.rows(), 0);
    if (state.column_indices.empty()) {
        return result;
    }

    // Algorithm 3 maintains R^T = Q_R T_R incrementally and factorizes C
    // only when a new PLSQR phase is admitted.  The regularized construction
    // still needs the compact l-by-l SVD from Section 5.2.
    const QrFactor column_factor = thin_qr(state.columns);
    const math::Matrix core_times_row_factor = solve_core(
        state, transpose(state.row_upper));
    const math::Matrix compact = math::multiply(
        column_factor.upper, core_times_row_factor);
    const math::SvdResult compact_svd = math::thin_svd(compact);
    const int rank = retained_rank(
        compact_svd,
        settings.relative_rank_tolerance,
        settings.absolute_rank_tolerance);
    if (rank == 0) {
        return result;
    }
    math::Matrix all_right_vectors(
        state.row_q.rows(),
        static_cast<int>(compact_svd.singular_values.size()));
    math::gemm(
        state.row_q,
        false,
        compact_svd.vt,
        true,
        1.0,
        0.0,
        all_right_vectors);
    result.right_vectors = math::copy_columns(all_right_vectors, 0, rank);
    result.singular_values.assign(
        compact_svd.singular_values.begin(),
        compact_svd.singular_values.begin() + rank);
    result.target = std::hypot(
        result.singular_values.back(),
        std::sqrt(settings.regularization));
    return result;
}

void apply_preconditioner_inverse(
    const Preconditioner& preconditioner,
    double regularization,
    std::span<const double> input,
    std::vector<double>& output) {
    output.assign(input.begin(), input.end());
    if (input.size() !=
        static_cast<std::size_t>(preconditioner.right_vectors.rows())) {
        throw std::invalid_argument("APLICUR preconditioner dimension mismatch");
    }
    const double mu = std::sqrt(regularization);
    for (int component = 0; component < preconditioner.rank(); ++component) {
        const double* vector =
            preconditioner.right_vectors.column_data(component);
        const double projection = math::vector_dot(
            vector, input.data(), static_cast<int>(input.size()));
        const double denominator = std::hypot(
            preconditioner.singular_values[static_cast<std::size_t>(component)],
            mu);
        const double coefficient =
            (preconditioner.target / denominator - 1.0) * projection;
        for (int index = 0; index < static_cast<int>(input.size()); ++index) {
            output[static_cast<std::size_t>(index)] +=
                coefficient * vector[index];
        }
    }
}

void apply_search_a(
    const math::RowMajorDenseOperator& matrix,
    const double* input,
    double* output,
    RunStatistics& statistics) {
    const auto start = Clock::now();
    matrix.apply(input, 1, output);
    statistics.a_seconds += elapsed(start);
    ++statistics.a_columns;
    ++statistics.a_block_calls;
    ++statistics.search_a_columns;
    ++statistics.search_a_block_calls;
    ++statistics.iterative_a_columns;
    ++statistics.iterative_a_block_calls;
}

void apply_search_at(
    const math::RowMajorDenseOperator& matrix,
    const double* input,
    double* output,
    RunStatistics& statistics) {
    const auto start = Clock::now();
    matrix.apply_transpose(input, 1, output);
    statistics.at_seconds += elapsed(start);
    ++statistics.at_columns;
    ++statistics.at_block_calls;
    ++statistics.search_at_columns;
    ++statistics.search_at_block_calls;
    ++statistics.iterative_at_columns;
    ++statistics.iterative_at_block_calls;
}

LeastSquaresResult validate_checkpoint(
    const math::RowMajorDenseOperator& matrix,
    const double* right_hand_side,
    const std::vector<double>& solution,
    const Settings& settings,
    int iterations,
    int phase,
    int cur_rank,
    int spectral_rank,
    RunStatistics& statistics) {
    LeastSquaresResult result = math::validate_original_candidate(
        matrix,
        right_hand_side,
        std::span<const double>(solution),
        settings.regularization,
        settings.tolerance,
        0.0,
        statistics);
    result.iterations = iterations;
    result.depth = phase;
    result.auxiliary_width = cur_rank;
    result.basis_rank = spectral_rank;
    result.trace.push_back(math::make_iteration_record(
        result, phase, iterations, cur_rank));
    return result;
}

PhaseOutcome run_plsqr_phase(
    const math::RowMajorDenseOperator& matrix,
    const double* right_hand_side,
    const Preconditioner& preconditioner,
    const Settings& settings,
    std::vector<double> base_solution,
    int phase,
    int cur_rank,
    int& total_iterations,
    bool final_phase,
    math::CertificateCheckpointSchedule& checkpoint_schedule,
    int& evaluated_iteration,
    RunStatistics& statistics,
    std::vector<IterationRecord>& trace,
    LeastSquaresResult current_result) {
    const int rows = matrix.rows();
    const int cols = matrix.cols();
    const double mu = std::sqrt(settings.regularization);

    std::vector<double> applied_base(static_cast<std::size_t>(rows));
    apply_search_a(
        matrix,
        base_solution.data(),
        applied_base.data(),
        statistics);
    std::vector<double> u(static_cast<std::size_t>(rows + cols));
    for (int row = 0; row < rows; ++row) {
        u[static_cast<std::size_t>(row)] =
            right_hand_side[row] - applied_base[static_cast<std::size_t>(row)];
    }
    for (int column = 0; column < cols; ++column) {
        u[static_cast<std::size_t>(rows + column)] =
            -mu * base_solution[static_cast<std::size_t>(column)];
    }
    double beta = math::vector_norm(u.data(), rows + cols);
    if (!(beta > 0.0) || !std::isfinite(beta)) {
        if (!std::isfinite(beta) &&
            current_result.status != SolverStatus::success) {
            current_result.status = SolverStatus::numerical_breakdown;
            current_result.stop_reason = StopReason::numerical_breakdown;
        }
        return PhaseOutcome{
            std::move(current_result), std::move(base_solution),
            !std::isfinite(beta), true, false};
    }
    scale(u, 1.0 / beta);

    std::vector<double> transpose_product(static_cast<std::size_t>(cols));
    apply_search_at(
        matrix,
        u.data(),
        transpose_product.data(),
        statistics);
    for (int column = 0; column < cols; ++column) {
        transpose_product[static_cast<std::size_t>(column)] +=
            mu * u[static_cast<std::size_t>(rows + column)];
    }
    std::vector<double> v;
    apply_preconditioner_inverse(
        preconditioner,
        settings.regularization,
        transpose_product,
        v);
    double alpha = math::vector_norm(v.data(), cols);
    if (!(alpha > 0.0) || !std::isfinite(alpha)) {
        if (!std::isfinite(alpha) &&
            current_result.status != SolverStatus::success) {
            current_result.status = SolverStatus::numerical_breakdown;
            current_result.stop_reason = StopReason::numerical_breakdown;
        }
        return PhaseOutcome{
            std::move(current_result), std::move(base_solution),
            !std::isfinite(alpha), true, false};
    }
    scale(v, 1.0 / alpha);

    std::vector<double> w = v;
    std::vector<double> correction(static_cast<std::size_t>(cols), 0.0);
    double rhobar = alpha;
    double phibar = beta;
    double previous_estimate = std::abs(phibar);
    double first_rate = 0.0;
    int local_iterations = 0;

    PhaseOutcome outcome;
    outcome.result = std::move(current_result);
    outcome.solution = base_solution;
    int committed_iteration = total_iterations;
    std::vector<double> original_correction;
    const auto evaluate_current_candidate = [&]() {
        apply_preconditioner_inverse(
            preconditioner,
            settings.regularization,
            correction,
            original_correction);
        outcome.solution = base_solution;
        for (int column = 0; column < cols; ++column) {
            outcome.solution[static_cast<std::size_t>(column)] +=
                original_correction[static_cast<std::size_t>(column)];
        }
        outcome.result = validate_checkpoint(
            matrix,
            right_hand_side,
            outcome.solution,
            settings,
            committed_iteration,
            phase,
            cur_rank,
            preconditioner.rank(),
            statistics);
        checkpoint_schedule.record_evaluation(
            committed_iteration, outcome.result);
        trace.push_back(outcome.result.trace.front());
        outcome.result.trace.clear();
        evaluated_iteration = committed_iteration;
    };
    const auto evaluate_terminal_candidate = [&]() {
        if (evaluated_iteration != committed_iteration) {
            evaluate_current_candidate();
        }
    };
    std::vector<double> next_u(static_cast<std::size_t>(rows + cols));
    std::vector<double> next_transpose(static_cast<std::size_t>(cols));
    std::vector<double> next_v;
    std::vector<double> candidate_correction(static_cast<std::size_t>(cols));
    std::vector<double> candidate_w(static_cast<std::size_t>(cols));
    while (total_iterations < settings.maximum_iterations) {
        std::vector<double> original_direction;
        apply_preconditioner_inverse(
            preconditioner,
            settings.regularization,
            v,
            original_direction);
        apply_search_a(
            matrix,
            original_direction.data(),
            next_u.data(),
            statistics);
        for (int row = 0; row < rows; ++row) {
            next_u[static_cast<std::size_t>(row)] -=
                alpha * u[static_cast<std::size_t>(row)];
        }
        for (int column = 0; column < cols; ++column) {
            next_u[static_cast<std::size_t>(rows + column)] =
                mu * original_direction[static_cast<std::size_t>(column)] -
                alpha * u[static_cast<std::size_t>(rows + column)];
        }
        beta = math::vector_norm(next_u.data(), rows + cols);
        if (beta > 0.0 && std::isfinite(beta)) {
            scale(next_u, 1.0 / beta);
        }

        // The current PLSQR correction needs beta but not the next A* product.
        // Map it to original coordinates only when a validation is due.
        const double rho = std::hypot(rhobar, beta);
        const double cosine = rho == 0.0 ? 1.0 : rhobar / rho;
        const double sine = rho == 0.0 ? 0.0 : beta / rho;
        const double phi = cosine * phibar;
        phibar = sine * phibar;
        candidate_correction = correction;
        if (rho != 0.0) {
            const double update = phi / rho;
            for (int column = 0; column < cols; ++column) {
                candidate_correction[static_cast<std::size_t>(column)] +=
                    update * w[static_cast<std::size_t>(column)];
            }
        }
        if (!all_finite(candidate_correction) || !all_finite(next_u) ||
            !std::isfinite(beta) || !std::isfinite(rho) ||
            !std::isfinite(cosine) || !std::isfinite(sine) ||
            !std::isfinite(phi) || !std::isfinite(phibar)) {
            evaluate_terminal_candidate();
            if (outcome.result.status != SolverStatus::success) {
                if (!math::candidate_validation_failed_numerically(
                        outcome.result)) {
                    outcome.result.status = SolverStatus::numerical_breakdown;
                    outcome.result.stop_reason =
                        StopReason::numerical_breakdown;
                }
                outcome.numerical_failure = true;
            }
            break;
        }

        correction.swap(candidate_correction);
        u.swap(next_u);
        ++total_iterations;
        ++local_iterations;
        committed_iteration = total_iterations;

        const double estimate = std::abs(phibar);
        const double difference = previous_estimate - estimate;
        double rate = 0.0;
        if (previous_estimate > 0.0 && estimate > 0.0 &&
            estimate < previous_estimate) {
            rate = std::log(previous_estimate / estimate);
        }
        if (first_rate == 0.0 && rate > 0.0) {
            first_rate = rate;
        }
        const bool rate_slowed =
            first_rate > 0.0 && rate > 0.0 &&
            first_rate / rate > settings.dynamic_stopping_tolerance;
        const bool decrease_slowed =
            difference < preconditioner.smallest_singular_value();
        previous_estimate = estimate;

        const bool dynamic_stop =
            !final_phase && local_iterations >= 2 &&
            (rate_slowed || decrease_slowed);
        const bool beta_breakdown = beta == 0.0;
        const bool maximum_iterations_reached =
            total_iterations >= settings.maximum_iterations;
        if (checkpoint_schedule.should_evaluate(total_iterations) ||
            dynamic_stop || beta_breakdown ||
            maximum_iterations_reached) {
            evaluate_current_candidate();
            if (math::candidate_validation_failed_numerically(
                    outcome.result)) {
                outcome.numerical_failure = true;
                break;
            }
            if (outcome.result.status == SolverStatus::success) {
                break;
            }
        }
        if (dynamic_stop) {
            outcome.dynamic_stop = true;
            break;
        }
        if (beta_breakdown) {
            outcome.exhausted = true;
            break;
        }
        if (maximum_iterations_reached) {
            break;
        }

        apply_search_at(
            matrix,
            u.data(),
            next_transpose.data(),
            statistics);
        for (int column = 0; column < cols; ++column) {
            next_transpose[static_cast<std::size_t>(column)] +=
                mu * u[static_cast<std::size_t>(rows + column)];
        }
        apply_preconditioner_inverse(
            preconditioner,
            settings.regularization,
            next_transpose,
            next_v);
        for (int column = 0; column < cols; ++column) {
            next_v[static_cast<std::size_t>(column)] -=
                beta * v[static_cast<std::size_t>(column)];
        }
        const double next_alpha = math::vector_norm(next_v.data(), cols);
        bool continuation_is_finite =
            std::isfinite(next_alpha) && all_finite(next_v);
        if (next_alpha > 0.0 && std::isfinite(next_alpha)) {
            scale(next_v, 1.0 / next_alpha);
        }

        const double theta = sine * next_alpha;
        const double next_rhobar = -cosine * next_alpha;
        const double recurrence = theta / rho;
        continuation_is_finite = continuation_is_finite &&
            std::isfinite(theta) && std::isfinite(next_rhobar) &&
            std::isfinite(recurrence);
        if (continuation_is_finite && next_alpha > 0.0) {
            for (int column = 0; column < cols; ++column) {
                candidate_w[static_cast<std::size_t>(column)] =
                    next_v[static_cast<std::size_t>(column)] -
                    recurrence * w[static_cast<std::size_t>(column)];
            }
            continuation_is_finite = all_finite(candidate_w);
        }
        if (!continuation_is_finite || next_alpha == 0.0) {
            evaluate_terminal_candidate();
            if (math::candidate_validation_failed_numerically(
                    outcome.result) ||
                outcome.result.status == SolverStatus::success) {
                if (math::candidate_validation_failed_numerically(
                        outcome.result)) {
                    outcome.numerical_failure = true;
                }
                break;
            }
            if (continuation_is_finite) {
                outcome.exhausted = true;
            } else {
                outcome.result.status = SolverStatus::numerical_breakdown;
                outcome.result.stop_reason = StopReason::numerical_breakdown;
                outcome.numerical_failure = true;
            }
            break;
        }

        alpha = next_alpha;
        rhobar = next_rhobar;
        w.swap(candidate_w);
        v.swap(next_v);
    }
    if (total_iterations >= settings.maximum_iterations &&
        outcome.result.status != SolverStatus::success &&
        !outcome.numerical_failure) {
        outcome.result.status = SolverStatus::work_limit;
        outcome.result.stop_reason = StopReason::maximum_depth;
    }
    return outcome;
}

Settings make_settings(
    int rows,
    int cols,
    const AplicurOptions& options) {
    if (!std::isfinite(options.regularization) ||
        !(options.regularization > 0.0)) {
        throw std::invalid_argument(
            "APLICUR Algorithm 3 requires positive regularization");
    }
    if (!std::isfinite(options.tolerance) || options.tolerance <= 0.0 ||
        options.tolerance >= 1.0) {
        throw std::invalid_argument(
            "tolerance must lie strictly between zero and one");
    }
    if (options.block_size <= 0 || options.sparse_sign_nonzeros <= 0 ||
        options.spectral_probe_count <= 0 ||
        options.maximum_iterations < 0) {
        throw std::invalid_argument(
            "APLICUR random and work parameters must be positive");
    }
    if (!std::isfinite(options.cur_tolerance) ||
        options.cur_tolerance < 0.0 ||
        !std::isfinite(options.re_preconditioning_tolerance) ||
        !(options.re_preconditioning_tolerance > 1.0) ||
        !std::isfinite(options.dynamic_stopping_tolerance) ||
        !(options.dynamic_stopping_tolerance > 1.0) ||
        !std::isfinite(options.relative_rank_tolerance) ||
        options.relative_rank_tolerance < 0.0 ||
        !std::isfinite(options.absolute_rank_tolerance) ||
        options.absolute_rank_tolerance < 0.0) {
        throw std::invalid_argument("APLICUR tolerance parameter is invalid");
    }
    const int rank_limit = std::min(rows, cols);
    const int block_size = std::min(options.block_size, rank_limit);
    const long long default_iterations = 4LL * rank_limit;
    if (default_iterations > std::numeric_limits<int>::max()) {
        throw std::length_error(
            "APLICUR default iteration limit exceeds LP64 range");
    }
    const int maximum_iterations = options.maximum_iterations > 0
        ? options.maximum_iterations
        : static_cast<int>(default_iterations);
    const double cur_tolerance = options.cur_tolerance > 0.0
        ? options.cur_tolerance
        : 50.0 * std::sqrt(options.regularization);
    return Settings{
        options.regularization,
        options.tolerance,
        cur_tolerance,
        options.re_preconditioning_tolerance,
        options.dynamic_stopping_tolerance,
        options.relative_rank_tolerance,
        options.absolute_rank_tolerance,
        block_size,
        options.sparse_sign_nonzeros,
        options.spectral_probe_count,
        maximum_iterations};
}

void finish_result(
    LeastSquaresResult& result,
    std::vector<IterationRecord> trace,
    RunStatistics& statistics,
    Clock::time_point total_start) {
    result.trace = std::move(trace);
    statistics.total_seconds = elapsed(total_start);
    result.statistics = statistics;
}

}  // namespace

LeastSquaresResult solve_aplicur(
    const double* matrix_row_major,
    int rows,
    int cols,
    const double* right_hand_side,
    const AplicurOptions& options) {
    if (rows <= 0 || cols <= 0 || matrix_row_major == nullptr) {
        throw std::invalid_argument("explicit APLICUR matrix input is invalid");
    }
    const math::RowMajorDenseOperator matrix_operator(
        rows, cols, matrix_row_major);
    math::validate_common_inputs(
        matrix_operator,
        right_hand_side,
        options.regularization,
        options.tolerance);
    const Settings settings = make_settings(rows, cols, options);
    const long long matrix_size = static_cast<long long>(rows) * cols;
    if (matrix_size > std::numeric_limits<int>::max()) {
        throw std::length_error("explicit APLICUR matrix exceeds LP64 range");
    }
    for (long long index = 0; index < matrix_size; ++index) {
        if (!std::isfinite(matrix_row_major[index])) {
            throw std::invalid_argument("explicit APLICUR matrix must be finite");
        }
    }

    const auto total_start = Clock::now();
    const RowMajorMatrixView matrix(rows, cols, matrix_row_major);
    RunStatistics statistics;
    int total_iterations = 0;
    int phase = 0;
    std::vector<double> solution(static_cast<std::size_t>(cols), 0.0);
    LeastSquaresResult result = validate_checkpoint(
        matrix_operator,
        right_hand_side,
        solution,
        settings,
        0,
        0,
        0,
        0,
        statistics);
    std::vector<IterationRecord> trace{result.trace.front()};
    result.trace.clear();
    math::CertificateCheckpointSchedule checkpoint_schedule(
        settings.regularization, settings.tolerance);
    checkpoint_schedule.record_evaluation(0, result);
    int evaluated_iteration = 0;
    if (math::candidate_validation_failed_numerically(result) ||
        result.status == SolverStatus::success) {
        finish_result(result, std::move(trace), statistics, total_start);
        return result;
    }

    const long double sketch_size_real = std::ceil(
        1.1L * static_cast<long double>(settings.block_size));
    if (sketch_size_real > std::numeric_limits<int>::max()) {
        throw std::length_error("APLICUR sketch size exceeds LP64 range");
    }
    const int sketch_rows = std::max(
        settings.block_size, static_cast<int>(sketch_size_real));
    std::mt19937_64 engine = make_engine(options.seed);
    const math::Matrix sketch = one_time_sparse_sign_sketch(
        matrix,
        sketch_rows,
        settings.sparse_sign_nonzeros,
        engine);

    CurState cur;
    cur.sketched_residual = sketch;
    double previous_gap = std::numeric_limits<double>::infinity();
    const int rank_limit = std::min(rows, cols);
    while (static_cast<int>(cur.column_indices.size()) < rank_limit) {
        const int increment = std::min(
            settings.block_size,
            rank_limit - static_cast<int>(cur.column_indices.size()));
        augment_cur(
            matrix,
            sketch,
            cur,
            increment,
            settings,
            engine,
            statistics);
        if (!std::isfinite(cur.residual_estimate) ||
            !all_finite(cur.sketched_residual)) {
            result.status = SolverStatus::numerical_breakdown;
            result.stop_reason = StopReason::numerical_breakdown;
            break;
        }

        const int cur_rank = static_cast<int>(cur.column_indices.size());
        const bool approximation_complete =
            cur.residual_estimate <= settings.cur_tolerance;
        const bool rank_complete = cur_rank == rank_limit;
        const double current_gap =
            cur.residual_estimate - settings.cur_tolerance;
        const bool sufficiently_improved =
            !std::isfinite(previous_gap) || approximation_complete ||
            (current_gap > 0.0 &&
             previous_gap / current_gap >=
                 settings.re_preconditioning_tolerance);
        if (!sufficiently_improved && !rank_complete) {
            continue;
        }

        const Preconditioner preconditioner = build_preconditioner(
            cur, settings);
        if (!all_finite(preconditioner.right_vectors) ||
            !all_finite(preconditioner.singular_values) ||
            !(preconditioner.target > 0.0) ||
            !std::isfinite(preconditioner.target)) {
            result.status = SolverStatus::numerical_breakdown;
            result.stop_reason = StopReason::numerical_breakdown;
            break;
        }
        ++phase;
        PhaseOutcome outcome = run_plsqr_phase(
            matrix_operator,
            right_hand_side,
            preconditioner,
            settings,
            solution,
            phase,
            cur_rank,
            total_iterations,
            approximation_complete || rank_complete,
            checkpoint_schedule,
            evaluated_iteration,
            statistics,
            trace,
            result);
        result = std::move(outcome.result);
        solution = std::move(outcome.solution);
        previous_gap = current_gap;
        if (outcome.numerical_failure ||
            result.status == SolverStatus::success ||
            total_iterations >= settings.maximum_iterations) {
            break;
        }
        if (approximation_complete || rank_complete) {
            if (result.status != SolverStatus::success) {
                result.status = SolverStatus::work_limit;
                result.stop_reason = outcome.exhausted
                    ? StopReason::exhausted_search_space
                    : StopReason::maximum_depth;
            }
            break;
        }
    }

    result.solution = solution;
    result.iterations = total_iterations;
    result.depth = phase;
    result.auxiliary_width = static_cast<int>(cur.column_indices.size());
    finish_result(result, std::move(trace), statistics, total_start);
    return result;
}

}  // namespace amfls
