#include "amfls/spir_fossils.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <limits>
#include <random>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

#include "algorithms/mathematics/checked_arithmetic.hpp"
#include "algorithms/mathematics/krylov/candidate_validation.hpp"
#include "algorithms/mathematics/krylov/krylov_workspace.hpp"
#include "algorithms/mathematics/linalg/blas_lapack.hpp"
#include "algorithms/mathematics/linalg/matrix.hpp"
#include "algorithms/mathematics/linalg/svd.hpp"
#include "algorithms/mathematics/operators/dense_operator.hpp"
#include "algorithms/mathematics/operators/row_major_dense_operator.hpp"
#include "algorithms/mathematics/operators/sparse_csr_operator.hpp"

namespace amfls {
namespace {

using Clock = std::chrono::steady_clock;

constexpr const char* counter_overflow =
    "SPIR/FOSSILS statistics exceed signed 64-bit range";

enum class Method { spir, fossils };
enum class ProductPhase { sketch, iteration };

struct Settings {
    Method method = Method::spir;
    double tolerance = 0.0;
    int sketch_rows = 0;
    int embedding_nonzeros = 0;
    int maximum_inner_iterations = 0;
    double distortion_safety = 1.0;
    double relative_rank_tolerance = 0.0;
    double absolute_rank_tolerance = 0.0;
    int sketch_block_size = 0;
    std::uint64_t seed = 0;
    std::uint64_t stream = 0;
};

struct Preconditioner {
    math::Matrix factor;
    std::vector<double> initial_solution;
    math::Matrix full_vt;
    std::vector<double> full_singular_values;
    double norm_estimate = 0.0;
    double frobenius_norm_estimate = 0.0;
    double condition_estimate = 0.0;
    int rank = 0;
};

struct InnerResult {
    std::vector<double> solution;
    int iterations = 0;
    bool finite = true;
};

enum class InnerDecision { continue_iteration, stop, numerical_failure };

struct PosteriorResult {
    std::vector<double> candidate;
    std::vector<double> residual;
    double estimate = std::numeric_limits<double>::infinity();
    bool finite = false;
    bool passed = false;
    bool near_pass = false;
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
        std::span<const double>(matrix.data(),
                                static_cast<std::size_t>(matrix.size())));
}

void account_a(
    RunStatistics& statistics,
    int block_columns,
    ProductPhase phase,
    double seconds) {
    statistics.a_seconds += seconds;
    math::checked_counter_add(
        statistics.a_columns, block_columns, counter_overflow);
    math::checked_counter_add(
        statistics.search_a_columns, block_columns, counter_overflow);
    math::checked_counter_add(
        statistics.a_block_calls, 1, counter_overflow);
    math::checked_counter_add(
        statistics.search_a_block_calls, 1, counter_overflow);
    if (phase == ProductPhase::sketch) {
        math::checked_counter_add(
            statistics.sketch_a_columns, block_columns, counter_overflow);
        math::checked_counter_add(
            statistics.sketch_a_block_calls, 1, counter_overflow);
    } else {
        math::checked_counter_add(
            statistics.iterative_a_columns, block_columns, counter_overflow);
        math::checked_counter_add(
            statistics.iterative_a_block_calls, 1, counter_overflow);
    }
}

void account_at(
    RunStatistics& statistics,
    int block_columns,
    ProductPhase phase,
    double seconds) {
    statistics.at_seconds += seconds;
    math::checked_counter_add(
        statistics.at_columns, block_columns, counter_overflow);
    math::checked_counter_add(
        statistics.search_at_columns, block_columns, counter_overflow);
    math::checked_counter_add(
        statistics.at_block_calls, 1, counter_overflow);
    math::checked_counter_add(
        statistics.search_at_block_calls, 1, counter_overflow);
    if (phase == ProductPhase::sketch) {
        math::checked_counter_add(
            statistics.sketch_at_columns, block_columns, counter_overflow);
        math::checked_counter_add(
            statistics.sketch_at_block_calls, 1, counter_overflow);
    } else {
        math::checked_counter_add(
            statistics.iterative_at_columns, block_columns, counter_overflow);
        math::checked_counter_add(
            statistics.iterative_at_block_calls, 1, counter_overflow);
    }
}

void apply_original(
    const MatrixOperator& matrix,
    const double* input,
    int block_columns,
    double* output,
    ProductPhase phase,
    RunStatistics& statistics) {
    const auto start = Clock::now();
    matrix.apply(input, block_columns, output);
    const double seconds = elapsed(start);
    account_a(statistics, block_columns, phase, seconds);
}

void apply_original_transpose(
    const MatrixOperator& matrix,
    const double* input,
    int block_columns,
    double* output,
    ProductPhase phase,
    RunStatistics& statistics) {
    const auto start = Clock::now();
    matrix.apply_transpose(input, block_columns, output);
    const double seconds = elapsed(start);
    account_at(statistics, block_columns, phase, seconds);
}

class ColumnScaledOperator final : public MatrixOperator {
public:
    ColumnScaledOperator(
        const MatrixOperator& original,
        const std::vector<double>& column_norms)
        : original_(original), column_norms_(column_norms) {}

    int rows() const override { return original_.rows(); }
    int cols() const override { return original_.cols(); }

    void apply(
        const double* input,
        int block_columns,
        double* output) const override {
        math::Matrix scaled_input(cols(), block_columns);
        for (int column = 0; column < block_columns; ++column) {
            for (int row = 0; row < cols(); ++row) {
                scaled_input(row, column) =
                    input[row + column * cols()] /
                    column_norms_[static_cast<std::size_t>(row)];
            }
        }
        original_.apply(scaled_input.data(), block_columns, output);
    }

    void apply_transpose(
        const double* input,
        int block_columns,
        double* output) const override {
        math::Matrix transposed(cols(), block_columns);
        original_.apply_transpose(
            input, block_columns, transposed.data());
        for (int column = 0; column < block_columns; ++column) {
            for (int row = 0; row < cols(); ++row) {
                output[row + column * cols()] =
                    transposed(row, column) /
                    column_norms_[static_cast<std::size_t>(row)];
            }
        }
    }

private:
    const MatrixOperator& original_;
    const std::vector<double>& column_norms_;
};

bool stored_column_norms(
    const MatrixOperator& matrix,
    std::vector<double>& norms) {
    norms.assign(static_cast<std::size_t>(matrix.cols()), 0.0);
    if (const auto* dense =
            dynamic_cast<const math::DenseOperator*>(&matrix)) {
        const double* values = dense->values();
        for (int column = 0; column < matrix.cols(); ++column) {
            norms[static_cast<std::size_t>(column)] = math::vector_norm(
                values + column * matrix.rows(), matrix.rows());
        }
        return true;
    }
    if (const auto* dense =
            dynamic_cast<const math::RowMajorDenseOperator*>(&matrix)) {
        const double* values = dense->values();
        std::vector<math::ScaledSumOfSquares> accumulators(
            static_cast<std::size_t>(matrix.cols()));
        for (int row = 0; row < matrix.rows(); ++row) {
            for (int column = 0; column < matrix.cols(); ++column) {
                accumulators[static_cast<std::size_t>(column)].add(
                    values[row * matrix.cols() + column]);
            }
        }
        for (int column = 0; column < matrix.cols(); ++column) {
            norms[static_cast<std::size_t>(column)] =
                accumulators[static_cast<std::size_t>(column)].norm();
        }
        return true;
    }
    if (const auto* sparse =
            dynamic_cast<const math::SparseCsrOperator*>(&matrix)) {
        std::vector<math::ScaledSumOfSquares> accumulators(
            static_cast<std::size_t>(matrix.cols()));
        for (int row = 0; row < matrix.rows(); ++row) {
            for (std::int64_t entry = sparse->row_offsets()[row];
                 entry < sparse->row_offsets()[row + 1];
                 ++entry) {
                const std::size_t column = static_cast<std::size_t>(
                    sparse->column_indices()[entry]);
                accumulators[column].add(sparse->values()[entry]);
            }
        }
        for (int column = 0; column < matrix.cols(); ++column) {
            norms[static_cast<std::size_t>(column)] =
                accumulators[static_cast<std::size_t>(column)].norm();
        }
        return true;
    }
    return false;
}

bool compute_column_norms(
    const MatrixOperator& matrix,
    int requested_block_size,
    std::vector<double>& norms,
    RunStatistics& statistics) {
    const int block_size = requested_block_size > 0
        ? std::min(requested_block_size, matrix.cols())
        : matrix.cols();
    // Stored dense/CSR matrices expose the authors' numeric-A `vecnorm` path:
    // this is an entrywise setup pass, not an operator callback.  Its wall
    // time remains in total_seconds and must not be represented as fictitious
    // A columns or calls.
    if (!stored_column_norms(matrix, norms)) {
        norms.assign(static_cast<std::size_t>(matrix.cols()), 0.0);
        for (int first = 0; first < matrix.cols();) {
            const int count = std::min(block_size, matrix.cols() - first);
            math::Matrix unit_block(matrix.cols(), count);
            for (int column = 0; column < count; ++column) {
                unit_block(first + column, column) = 1.0;
            }
            math::Matrix columns(matrix.rows(), count);
            apply_original(
                matrix,
                unit_block.data(),
                count,
                columns.data(),
                ProductPhase::sketch,
                statistics);
            for (int column = 0; column < count; ++column) {
                norms[static_cast<std::size_t>(first + column)] =
                    math::vector_norm(
                        columns.column_data(column), matrix.rows());
            }
            first = math::checked_add(
                first,
                count,
                "column-norm block index exceeds LP64 limits");
        }
    }

    for (double& norm : norms) {
        if (!std::isfinite(norm) || norm < 0.0) {
            return false;
        }
        // An exact zero column is left unchanged so rank truncation can handle
        // it without division by zero.
        if (norm == 0.0) {
            norm = 1.0;
        }
    }
    return true;
}

int scaled_dimension(int dimension, int multiplier, const char* message) {
    if (dimension > std::numeric_limits<int>::max() / multiplier) {
        throw std::length_error(message);
    }
    return dimension * multiplier;
}

void validate_scalar_options(
    double relative_rank_tolerance,
    double absolute_rank_tolerance) {
    if (!std::isfinite(relative_rank_tolerance) ||
        relative_rank_tolerance < 0.0) {
        throw std::invalid_argument(
            "relative_rank_tolerance must be finite and nonnegative");
    }
    if (!std::isfinite(absolute_rank_tolerance) ||
        absolute_rank_tolerance < 0.0) {
        throw std::invalid_argument(
            "absolute_rank_tolerance must be finite and nonnegative");
    }
}

Settings make_settings(const MatrixOperator& matrix, const SpirOptions& options) {
    if (options.regularization != 0.0) {
        throw std::invalid_argument(
            "SPIR implements ordinary least squares and rejects ridge regularization");
    }
    if (matrix.rows() < matrix.cols()) {
        throw std::invalid_argument("SPIR requires rows >= columns");
    }
    if (options.sketch_rows < 0 || options.embedding_nonzeros <= 0 ||
        options.maximum_inner_iterations <= 0 ||
        options.sketch_block_size < 0) {
        throw std::invalid_argument(
            "SPIR embedding width and inner work must be positive");
    }
    validate_scalar_options(
        options.relative_rank_tolerance,
        options.absolute_rank_tolerance);
    const int rows = options.sketch_rows > 0
        ? options.sketch_rows
        : scaled_dimension(
              matrix.cols(), 2, "SPIR default sketch size exceeds LP64 limits");
    if (rows < matrix.cols()) {
        throw std::invalid_argument("SPIR sketch_rows must be at least columns");
    }
    return Settings{
        Method::spir,
        options.tolerance,
        rows,
        std::min(options.embedding_nonzeros, rows),
        options.maximum_inner_iterations,
        1.0,
        options.relative_rank_tolerance,
        options.absolute_rank_tolerance,
        options.sketch_block_size,
        options.seed,
        options.stream};
}

Settings make_settings(
    const MatrixOperator& matrix,
    const FossilsOptions& options) {
    if (options.regularization != 0.0) {
        throw std::invalid_argument(
            "FOSSILS implements ordinary least squares and rejects ridge regularization");
    }
    if (matrix.rows() < matrix.cols()) {
        throw std::invalid_argument("FOSSILS requires rows >= columns");
    }
    if (options.sketch_rows < 0 || options.embedding_nonzeros <= 0 ||
        options.maximum_inner_iterations <= 0 ||
        options.sketch_block_size < 0) {
        throw std::invalid_argument(
            "FOSSILS embedding width and inner work must be positive");
    }
    validate_scalar_options(
        options.relative_rank_tolerance,
        options.absolute_rank_tolerance);
    if (!std::isfinite(options.distortion_safety) ||
        !(options.distortion_safety > 0.0)) {
        throw std::invalid_argument(
            "distortion_safety must be finite and positive");
    }
    const int rows = options.sketch_rows > 0
        ? options.sketch_rows
        : scaled_dimension(
              matrix.cols(),
              12,
              "FOSSILS default sketch size exceeds LP64 limits");
    if (rows < matrix.cols()) {
        throw std::invalid_argument(
            "FOSSILS sketch_rows must be at least columns");
    }
    const double eta = options.distortion_safety *
        std::sqrt(static_cast<double>(matrix.cols()) / rows);
    if (!std::isfinite(eta) || !(eta < 1.0)) {
        throw std::invalid_argument(
            "FOSSILS distortion estimate must lie strictly below one");
    }
    return Settings{
        Method::fossils,
        options.tolerance,
        rows,
        std::min(options.embedding_nonzeros, rows),
        options.maximum_inner_iterations,
        options.distortion_safety,
        options.relative_rank_tolerance,
        options.absolute_rank_tolerance,
        options.sketch_block_size,
        options.seed,
        options.stream};
}

class SparseSignPlan {
public:
    SparseSignPlan(
        int original_rows,
        int sketch_rows,
        int nonzeros,
        std::uint64_t seed,
        std::uint64_t stream)
        : original_rows_(original_rows),
          sketch_rows_(sketch_rows),
          nonzeros_(nonzeros),
          scale_(1.0 / std::sqrt(static_cast<double>(nonzeros))) {
        const long long entry_count = math::checked_nonnegative_multiply(
            original_rows, nonzeros, "sparse embedding plan is too large");
        positions_.resize(static_cast<std::size_t>(entry_count));
        signs_.resize(static_cast<std::size_t>(entry_count));

        std::seed_seq sequence{
            static_cast<std::uint32_t>(seed),
            static_cast<std::uint32_t>(seed >> 32U),
            static_cast<std::uint32_t>(stream),
            static_cast<std::uint32_t>(stream >> 32U)};
        std::mt19937_64 engine(sequence);
        std::uniform_int_distribution<int> location(0, sketch_rows - 1);
        for (int row = 0; row < original_rows_; ++row) {
            const std::size_t first = static_cast<std::size_t>(row) * nonzeros_;
            for (int entry = 0; entry < nonzeros_; ++entry) {
                int position = 0;
                bool repeated = false;
                do {
                    position = location(engine);
                    repeated = false;
                    for (int previous = 0; previous < entry; ++previous) {
                        repeated = repeated ||
                            positions_[first + previous] == position;
                    }
                } while (repeated);
                positions_[first + entry] = position;
                signs_[first + entry] =
                    (engine() & std::uint64_t{1}) == 0 ? -1 : 1;
            }
        }
    }

    math::Matrix transpose_block(int first_column, int count) const {
        math::Matrix block(original_rows_, count);
        for (int row = 0; row < original_rows_; ++row) {
            const std::size_t first = static_cast<std::size_t>(row) * nonzeros_;
            for (int entry = 0; entry < nonzeros_; ++entry) {
                const int position = positions_[first + entry];
                if (position >= first_column &&
                    position < first_column + count) {
                    block(row, position - first_column) =
                        scale_ * signs_[first + entry];
                }
            }
        }
        return block;
    }

    std::vector<double> apply_to(const double* vector) const {
        std::vector<long double> accumulated(
            static_cast<std::size_t>(sketch_rows_), 0.0L);
        for (int row = 0; row < original_rows_; ++row) {
            const std::size_t first = static_cast<std::size_t>(row) * nonzeros_;
            for (int entry = 0; entry < nonzeros_; ++entry) {
                accumulated[positions_[first + entry]] +=
                    static_cast<long double>(scale_ * signs_[first + entry]) *
                    vector[row];
            }
        }
        std::vector<double> result(static_cast<std::size_t>(sketch_rows_));
        for (int row = 0; row < sketch_rows_; ++row) {
            result[static_cast<std::size_t>(row)] =
                static_cast<double>(accumulated[static_cast<std::size_t>(row)]);
        }
        return result;
    }

    void accumulate_csr(
        const math::SparseCsrOperator& matrix,
        math::Matrix& sketch) const {
        for (int row = 0; row < original_rows_; ++row) {
            const std::size_t first =
                static_cast<std::size_t>(row) * nonzeros_;
            for (int entry = 0; entry < nonzeros_; ++entry) {
                const int destination = positions_[first + entry];
                const double weight = scale_ * signs_[first + entry];
                for (std::int64_t matrix_entry = matrix.row_offsets()[row];
                     matrix_entry < matrix.row_offsets()[row + 1];
                     ++matrix_entry) {
                    const int column = static_cast<int>(
                        matrix.column_indices()[matrix_entry]);
                    sketch(destination, column) = std::fma(
                        matrix.values()[matrix_entry],
                        weight,
                        sketch(destination, column));
                }
            }
        }
    }

private:
    int original_rows_ = 0;
    int sketch_rows_ = 0;
    int nonzeros_ = 0;
    double scale_ = 0.0;
    std::vector<int> positions_;
    std::vector<signed char> signs_;
};

int retained_rank(
    const math::SvdResult& svd,
    const Settings& settings) {
    if (svd.singular_values.empty() ||
        !(svd.singular_values.front() > 0.0) ||
        !std::isfinite(svd.singular_values.front())) {
        return 0;
    }
    const double cutoff = std::max(
        settings.relative_rank_tolerance * svd.singular_values.front(),
        settings.absolute_rank_tolerance);
    int rank = 0;
    while (rank < static_cast<int>(svd.singular_values.size()) &&
           svd.singular_values[rank] > cutoff &&
           std::isfinite(svd.singular_values[rank]) &&
           std::isfinite(1.0 / svd.singular_values[rank])) {
        ++rank;
    }
    return rank;
}

Preconditioner build_preconditioner(
    const MatrixOperator& matrix,
    const double* right_hand_side,
    const std::vector<double>& column_norms,
    const Settings& settings,
    RunStatistics& statistics) {
    SparseSignPlan plan(
        matrix.rows(),
        settings.sketch_rows,
        settings.embedding_nonzeros,
        settings.seed,
        settings.stream);
    const std::vector<double> sketched_rhs = plan.apply_to(right_hand_side);
    math::Matrix sketch(settings.sketch_rows, matrix.cols());
    const int block_size = settings.sketch_block_size > 0
        ? std::min(settings.sketch_block_size, settings.sketch_rows)
        : settings.sketch_rows;
    if (const auto* sparse =
            dynamic_cast<const math::SparseCsrOperator*>(&matrix)) {
        plan.accumulate_csr(*sparse, sketch);
    } else {
        for (int first = 0; first < settings.sketch_rows;) {
            const int count = std::min(
                block_size, settings.sketch_rows - first);
            math::Matrix sketch_transpose =
                plan.transpose_block(first, count);
            math::Matrix product(matrix.cols(), count);
            apply_original_transpose(
                matrix,
                sketch_transpose.data(),
                count,
                product.data(),
                ProductPhase::sketch,
                statistics);
            if (!all_finite(product)) {
                return {};
            }
            for (int column = 0; column < count; ++column) {
                for (int original_column = 0;
                     original_column < matrix.cols();
                     ++original_column) {
                    sketch(first + column, original_column) =
                        product(original_column, column);
                }
            }
            first = math::checked_add(
                first,
                count,
                "sparse embedding block index exceeds LP64 limits");
        }
    }
    for (int column = 0; column < matrix.cols(); ++column) {
        const double inverse_scale =
            1.0 / column_norms[static_cast<std::size_t>(column)];
        for (int row = 0; row < settings.sketch_rows; ++row) {
            sketch(row, column) *= inverse_scale;
        }
    }
    if (!all_finite(sketch) || !all_finite(sketched_rhs)) {
        return {};
    }

    const auto factor_start = Clock::now();
    math::SvdResult svd = math::thin_svd(sketch);
    statistics.projected_solve_seconds += elapsed(factor_start);
    const int rank = retained_rank(svd, settings);
    math::Matrix factor(matrix.cols(), rank);
    for (int column = 0; column < rank; ++column) {
        const double inverse = 1.0 / svd.singular_values[column];
        for (int row = 0; row < matrix.cols(); ++row) {
            factor(row, column) = svd.vt(column, row) * inverse;
        }
    }

    std::vector<double> retained_left_coefficients(
        static_cast<std::size_t>(rank));
    for (int component = 0; component < rank; ++component) {
        retained_left_coefficients[static_cast<std::size_t>(component)] =
            math::vector_dot(
                svd.u.column_data(component),
                sketched_rhs.data(),
                settings.sketch_rows);
    }
    std::vector<double> initial_solution(
        static_cast<std::size_t>(matrix.cols()), 0.0);
    if (rank > 0) {
        math::gemv(
            factor,
            false,
            retained_left_coefficients.data(),
            1.0,
            0.0,
            initial_solution.data());
    }
    const double norm_estimate = svd.singular_values.empty()
        ? 0.0
        : svd.singular_values.front();
    const double frobenius_norm_estimate = svd.singular_values.empty()
        ? 0.0
        : math::vector_norm(
              svd.singular_values.data(),
              static_cast<int>(svd.singular_values.size()));
    const double smallest = svd.singular_values.empty()
        ? 0.0
        : svd.singular_values.back();
    const double condition_estimate =
        norm_estimate > 0.0 && smallest > 0.0 &&
            std::isfinite(norm_estimate) && std::isfinite(smallest)
        ? norm_estimate / smallest
        : std::numeric_limits<double>::infinity();
    return {
        std::move(factor),
        std::move(initial_solution),
        std::move(svd.vt),
        std::move(svd.singular_values),
        norm_estimate,
        frobenius_norm_estimate,
        condition_estimate,
        rank};
}

void apply_factor(
    const math::Matrix& factor,
    const std::vector<double>& input,
    std::vector<double>& output) {
    output.assign(static_cast<std::size_t>(factor.rows()), 0.0);
    if (factor.rows() == 0 || factor.cols() == 0) {
        return;
    }
    math::gemv(factor, false, input.data(), 1.0, 0.0, output.data());
}

void apply_factor_transpose(
    const math::Matrix& factor,
    const std::vector<double>& input,
    std::vector<double>& output) {
    output.assign(static_cast<std::size_t>(factor.cols()), 0.0);
    if (factor.rows() == 0 || factor.cols() == 0) {
        return;
    }
    math::gemv(factor, true, input.data(), 1.0, 0.0, output.data());
}

bool apply_preconditioned_normal(
    const MatrixOperator& matrix,
    const math::Matrix& factor,
    const std::vector<double>& input,
    std::vector<double>& output,
    RunStatistics& statistics) {
    std::vector<double> original_direction;
    apply_factor(factor, input, original_direction);
    if (!all_finite(original_direction)) {
        return false;
    }
    std::vector<double> applied(static_cast<std::size_t>(matrix.rows()));
    apply_original(
        matrix,
        original_direction.data(),
        1,
        applied.data(),
        ProductPhase::iteration,
        statistics);
    if (!all_finite(applied)) {
        return false;
    }
    std::vector<double> transposed(static_cast<std::size_t>(matrix.cols()));
    apply_original_transpose(
        matrix,
        applied.data(),
        1,
        transposed.data(),
        ProductPhase::iteration,
        statistics);
    if (!all_finite(transposed)) {
        return false;
    }
    apply_factor_transpose(factor, transposed, output);
    return all_finite(output);
}

template <class Controller>
InnerResult conjugate_gradient(
    const MatrixOperator& matrix,
    const math::Matrix& factor,
    const std::vector<double>& right_hand_side,
    const Settings& settings,
    RunStatistics& statistics,
    Controller&& controller) {
    InnerResult result;
    // The authors' practical implementation warm-starts the inner solve at
    // dy=c before beginning CG updates.
    result.solution = right_hand_side;
    if (!all_finite(result.solution)) {
        result.finite = false;
        return result;
    }
    std::vector<double> applied_initial;
    if (!apply_preconditioned_normal(
            matrix,
            factor,
            result.solution,
            applied_initial,
            statistics)) {
        result.finite = false;
        return result;
    }
    std::vector<double> residual = right_hand_side;
    for (int index = 0; index < static_cast<int>(residual.size()); ++index) {
        residual[static_cast<std::size_t>(index)] -=
            applied_initial[static_cast<std::size_t>(index)];
    }
    std::vector<double> direction = residual;
    double residual_squared = math::vector_dot(
        residual.data(), residual.data(), static_cast<int>(residual.size()));
    if (!std::isfinite(residual_squared) || residual_squared < 0.0) {
        result.finite = false;
        return result;
    }
    if (residual_squared == 0.0) {
        return result;
    }

    for (int iteration = 0;
         iteration < settings.maximum_inner_iterations;
         ++iteration) {
        std::vector<double> applied;
        if (!apply_preconditioned_normal(
                matrix,
                factor,
                direction,
                applied,
                statistics)) {
            result.finite = false;
            return result;
        }
        const double curvature = math::vector_dot(
            direction.data(), applied.data(), static_cast<int>(direction.size()));
        if (!(curvature > 0.0) || !std::isfinite(curvature)) {
            result.finite = false;
            return result;
        }
        const double step = residual_squared / curvature;
        if (!std::isfinite(step)) {
            result.finite = false;
            return result;
        }
        std::vector<double> update(direction.size());
        for (int index = 0; index < static_cast<int>(direction.size()); ++index) {
            update[static_cast<std::size_t>(index)] =
                step * direction[static_cast<std::size_t>(index)];
            result.solution[static_cast<std::size_t>(index)] = std::fma(
                1.0,
                update[static_cast<std::size_t>(index)],
                result.solution[static_cast<std::size_t>(index)]);
            residual[static_cast<std::size_t>(index)] = std::fma(
                -step,
                applied[static_cast<std::size_t>(index)],
                residual[static_cast<std::size_t>(index)]);
        }
        ++result.iterations;
        if (!all_finite(update) || !all_finite(result.solution) ||
            !all_finite(residual)) {
            result.finite = false;
            return result;
        }
        const InnerDecision decision = controller(
            result.solution, update, result.iterations);
        if (decision == InnerDecision::numerical_failure) {
            result.finite = false;
            return result;
        }
        if (decision == InnerDecision::stop) {
            return result;
        }
        const double next_squared = math::vector_dot(
            residual.data(), residual.data(), static_cast<int>(residual.size()));
        if (!std::isfinite(next_squared) || next_squared < 0.0) {
            result.finite = false;
            return result;
        }
        if (next_squared == 0.0) {
            return result;
        }
        const double recurrence = next_squared / residual_squared;
        if (!std::isfinite(recurrence)) {
            result.finite = false;
            return result;
        }
        for (int index = 0; index < static_cast<int>(direction.size()); ++index) {
            direction[static_cast<std::size_t>(index)] = std::fma(
                recurrence,
                direction[static_cast<std::size_t>(index)],
                residual[static_cast<std::size_t>(index)]);
        }
        residual_squared = next_squared;
    }
    return result;
}

template <class Controller>
InnerResult heavy_ball(
    const MatrixOperator& matrix,
    const math::Matrix& factor,
    const std::vector<double>& right_hand_side,
    const Settings& settings,
    RunStatistics& statistics,
    Controller&& controller) {
    InnerResult result;
    result.solution = right_hand_side;
    std::vector<double> previous = result.solution;
    if (!all_finite(result.solution)) {
        result.finite = false;
        return result;
    }
    const double eta = settings.distortion_safety * std::sqrt(
        static_cast<double>(factor.cols()) / settings.sketch_rows);
    const double momentum = eta * eta;
    const double step = (1.0 - momentum) * (1.0 - momentum);

    for (int iteration = 0;
         iteration < settings.maximum_inner_iterations;
         ++iteration) {
        std::vector<double> applied;
        if (!apply_preconditioned_normal(
                matrix,
                factor,
                result.solution,
                applied,
                statistics)) {
            result.finite = false;
            return result;
        }
        std::vector<double> next = result.solution;
        std::vector<double> update(result.solution.size());
        for (int index = 0; index < static_cast<int>(next.size()); ++index) {
            update[static_cast<std::size_t>(index)] =
                step * (right_hand_side[static_cast<std::size_t>(index)] -
                        applied[static_cast<std::size_t>(index)]) +
                momentum *
                    (result.solution[static_cast<std::size_t>(index)] -
                     previous[static_cast<std::size_t>(index)]);
            next[static_cast<std::size_t>(index)] +=
                update[static_cast<std::size_t>(index)];
        }
        ++result.iterations;
        if (!all_finite(update) || !all_finite(next)) {
            result.finite = false;
            return result;
        }
        previous.swap(result.solution);
        result.solution.swap(next);
        const InnerDecision decision = controller(
            result.solution, update, result.iterations);
        if (decision == InnerDecision::numerical_failure) {
            result.finite = false;
            return result;
        }
        if (decision == InnerDecision::stop) {
            return result;
        }
    }
    return result;
}

bool refinement_right_hand_side(
    const MatrixOperator& matrix,
    const double* right_hand_side,
    const math::Matrix& factor,
    const std::vector<double>& solution,
    std::vector<double>& residual,
    std::vector<double>& transformed,
    RunStatistics& statistics) {
    std::vector<double> applied(static_cast<std::size_t>(matrix.rows()));
    apply_original(
        matrix,
        solution.data(),
        1,
        applied.data(),
        ProductPhase::iteration,
        statistics);
    if (!all_finite(applied)) {
        return false;
    }
    residual.resize(static_cast<std::size_t>(matrix.rows()));
    for (int row = 0; row < matrix.rows(); ++row) {
        residual[static_cast<std::size_t>(row)] =
            right_hand_side[row] - applied[static_cast<std::size_t>(row)];
    }
    std::vector<double> transposed(static_cast<std::size_t>(matrix.cols()));
    apply_original_transpose(
        matrix,
        residual.data(),
        1,
        transposed.data(),
        ProductPhase::iteration,
        statistics);
    if (!all_finite(transposed)) {
        return false;
    }
    apply_factor_transpose(factor, transposed, transformed);
    return all_finite(residual) && all_finite(transformed);
}

bool form_candidate(
    const math::Matrix& factor,
    const std::vector<double>& base_solution,
    const std::vector<double>& inner_solution,
    std::vector<double>& candidate) {
    std::vector<double> correction;
    apply_factor(factor, inner_solution, correction);
    if (!all_finite(correction) || correction.size() != base_solution.size()) {
        return false;
    }
    candidate = base_solution;
    for (int column = 0; column < static_cast<int>(candidate.size()); ++column) {
        candidate[static_cast<std::size_t>(column)] +=
            correction[static_cast<std::size_t>(column)];
    }
    return all_finite(candidate);
}

PosteriorResult evaluate_posterior(
    const MatrixOperator& matrix,
    const double* right_hand_side,
    const Preconditioner& preconditioner,
    const std::vector<double>& base_solution,
    const std::vector<double>& inner_solution,
    double right_hand_side_norm,
    double backward_error_tolerance,
    RunStatistics& statistics) {
    PosteriorResult result;
    if (!(right_hand_side_norm > 0.0) ||
        !(preconditioner.frobenius_norm_estimate > 0.0) ||
        !std::isfinite(right_hand_side_norm) ||
        !std::isfinite(preconditioner.frobenius_norm_estimate) ||
        preconditioner.full_vt.rows() !=
            static_cast<int>(preconditioner.full_singular_values.size()) ||
        preconditioner.full_vt.cols() != matrix.cols()) {
        return result;
    }
    if (!form_candidate(
            preconditioner.factor,
            base_solution,
            inner_solution,
            result.candidate)) {
        return result;
    }

    std::vector<double> applied(static_cast<std::size_t>(matrix.rows()));
    apply_original(
        matrix,
        result.candidate.data(),
        1,
        applied.data(),
        ProductPhase::iteration,
        statistics);
    result.residual.resize(static_cast<std::size_t>(matrix.rows()));
    for (int row = 0; row < matrix.rows(); ++row) {
        result.residual[static_cast<std::size_t>(row)] =
            right_hand_side[row] - applied[static_cast<std::size_t>(row)];
    }
    if (!all_finite(applied) || !all_finite(result.residual)) {
        return result;
    }

    std::vector<double> gradient(static_cast<std::size_t>(matrix.cols()));
    apply_original_transpose(
        matrix,
        result.residual.data(),
        1,
        gradient.data(),
        ProductPhase::iteration,
        statistics);
    if (!all_finite(gradient)) {
        return result;
    }

    const double solution_norm = math::vector_norm(
        result.candidate.data(), matrix.cols());
    const double residual_norm = math::vector_norm(
        result.residual.data(), matrix.rows());
    const double inverse_theta = right_hand_side_norm /
        preconditioner.frobenius_norm_estimate;
    const double outer_denominator = std::hypot(
        inverse_theta, solution_norm);
    if (!(outer_denominator > 0.0) ||
        !std::isfinite(outer_denominator) ||
        !std::isfinite(residual_norm)) {
        return result;
    }
    const double outer_scale = 1.0 / outer_denominator;
    const double regularizing_scale = residual_norm * outer_scale;
    std::vector<double> projections(
        preconditioner.full_singular_values.size(), 0.0);
    if (!projections.empty()) {
        math::gemv(
            preconditioner.full_vt,
            false,
            gradient.data(),
            1.0,
            0.0,
            projections.data());
    }
    for (int component = 0;
         component < static_cast<int>(
             preconditioner.full_singular_values.size());
         ++component) {
        const double projection =
            projections[static_cast<std::size_t>(component)];
        const double denominator = std::hypot(
            preconditioner.full_singular_values[
                static_cast<std::size_t>(component)],
            regularizing_scale);
        double scaled_projection = 0.0;
        if (denominator > 0.0 && std::isfinite(denominator)) {
            scaled_projection = projection / denominator;
        } else if (projection != 0.0) {
            return result;
        }
        if (!std::isfinite(scaled_projection)) {
            return result;
        }
        projections[static_cast<std::size_t>(component)] = scaled_projection;
    }
    const double projected_norm = projections.empty()
        ? 0.0
        : math::vector_norm(
              projections.data(), static_cast<int>(projections.size()));
    result.estimate = outer_scale * projected_norm;
    result.finite = std::isfinite(result.estimate);
    const double threshold = backward_error_tolerance *
        preconditioner.frobenius_norm_estimate;
    result.passed = result.finite && result.estimate <= threshold;
    result.near_pass = result.finite && result.estimate <= 4.0 * threshold;
    return result;
}

double first_phase_threshold(
    const Preconditioner& preconditioner,
    const std::vector<double>& solution,
    const std::vector<double>& residual) {
    const double solution_norm = math::vector_norm(
        solution.data(), static_cast<int>(solution.size()));
    const double residual_norm = math::vector_norm(
        residual.data(), static_cast<int>(residual.size()));
    const double condition_term = residual_norm == 0.0
        ? 0.0
        : preconditioner.condition_estimate * residual_norm;
    return 10.0 * std::numeric_limits<double>::epsilon() *
        (preconditioner.norm_estimate * solution_norm +
         0.04 * condition_term);
}

std::vector<double> unscale_solution(
    const std::vector<double>& scaled_solution,
    const std::vector<double>& column_norms) {
    std::vector<double> solution = scaled_solution;
    for (int column = 0; column < static_cast<int>(solution.size()); ++column) {
        solution[static_cast<std::size_t>(column)] /=
            column_norms[static_cast<std::size_t>(column)];
    }
    return solution;
}

LeastSquaresResult validate_checkpoint(
    const MatrixOperator& matrix,
    const double* right_hand_side,
    const Settings& settings,
    const std::vector<double>& solution,
    int inner_iterations,
    int refinements,
    int rank,
    RunStatistics& statistics) {
    LeastSquaresResult result = math::validate_original_candidate(
        matrix,
        right_hand_side,
        std::span<const double>(solution),
        0.0,
        settings.tolerance,
        0.0,
        statistics);
    result.iterations = inner_iterations;
    result.depth = refinements;
    result.auxiliary_width = settings.sketch_rows;
    result.basis_rank = rank;
    result.trace.push_back(math::make_iteration_record(
        result, refinements, refinements, settings.sketch_rows));
    return result;
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

LeastSquaresResult initial_breakdown_result(
    const MatrixOperator& matrix,
    const Settings& settings,
    int rank,
    RunStatistics& statistics,
    Clock::time_point total_start) {
    LeastSquaresResult result;
    result.rows = matrix.rows();
    result.cols = matrix.cols();
    result.solution.assign(static_cast<std::size_t>(matrix.cols()), 0.0);
    result.auxiliary_width = settings.sketch_rows;
    result.basis_rank = rank;
    result.status = SolverStatus::numerical_breakdown;
    result.stop_reason = StopReason::numerical_breakdown;
    finish_result(result, {}, statistics, total_start);
    return result;
}

LeastSquaresResult solve_impl(
    const MatrixOperator& matrix,
    const double* right_hand_side,
    const Settings& settings) {
    const auto total_start = Clock::now();
    RunStatistics statistics;
    std::vector<double> column_norms;
    if (!compute_column_norms(
            matrix,
            settings.sketch_block_size,
            column_norms,
            statistics)) {
        return initial_breakdown_result(
            matrix, settings, 0, statistics, total_start);
    }
    ColumnScaledOperator scaled_matrix(matrix, column_norms);
    Preconditioner preconditioner = build_preconditioner(
        matrix,
        right_hand_side,
        column_norms,
        settings,
        statistics);
    if (preconditioner.initial_solution.size() !=
            static_cast<std::size_t>(matrix.cols()) ||
        !all_finite(preconditioner.initial_solution) ||
        !all_finite(preconditioner.factor) ||
        !all_finite(preconditioner.full_vt) ||
        !all_finite(preconditioner.full_singular_values) ||
        !std::isfinite(preconditioner.norm_estimate) ||
        !std::isfinite(preconditioner.frobenius_norm_estimate)) {
        return initial_breakdown_result(
            matrix,
            settings,
            preconditioner.rank,
            statistics,
            total_start);
    }

    std::vector<double> scaled_solution = preconditioner.initial_solution;
    int total_inner_iterations = 0;
    int completed_refinements = 0;
    bool numerical_breakdown = false;
    const double right_hand_side_norm = math::vector_norm(
        right_hand_side, matrix.rows());

    for (int refinement = 1;
         refinement <= 2 && preconditioner.rank > 0 &&
             right_hand_side_norm > 0.0;
         ++refinement) {
        const std::vector<double> base_solution = scaled_solution;
        std::vector<double> residual;
        std::vector<double> transformed_rhs;
        if (!refinement_right_hand_side(
                scaled_matrix,
                right_hand_side,
                preconditioner.factor,
                base_solution,
                residual,
                transformed_rhs,
                statistics)) {
            numerical_breakdown = true;
            break;
        }

        const double switch_threshold = first_phase_threshold(
            preconditioner, base_solution, residual);
        PosteriorResult scheduled_posterior;
        int scheduled_iteration = -1;
        bool stop_at_next_posterior = false;
        auto controller = [&](
                              const std::vector<double>& inner_solution,
                              const std::vector<double>& update,
                              int iteration) {
            if (refinement == 1) {
                const double update_norm = math::vector_norm(
                    update.data(), static_cast<int>(update.size()));
                if (!std::isfinite(update_norm)) {
                    return InnerDecision::numerical_failure;
                }
                return update_norm <= switch_threshold
                    ? InnerDecision::stop
                    : InnerDecision::continue_iteration;
            }
            if (iteration % 5 != 0) {
                return InnerDecision::continue_iteration;
            }
            scheduled_posterior = evaluate_posterior(
                scaled_matrix,
                right_hand_side,
                preconditioner,
                base_solution,
                inner_solution,
                right_hand_side_norm,
                settings.tolerance,
                statistics);
            scheduled_iteration = iteration;
            if (!scheduled_posterior.finite) {
                return InnerDecision::numerical_failure;
            }
            if (scheduled_posterior.passed || stop_at_next_posterior) {
                return InnerDecision::stop;
            }
            if (scheduled_posterior.near_pass) {
                // Match the authors' `stopnext` rule: after a four-times-near
                // pass, perform one more five-update posterior interval.
                stop_at_next_posterior = true;
            }
            return InnerDecision::continue_iteration;
        };

        InnerResult correction;
        if (settings.method == Method::spir) {
            correction = conjugate_gradient(
                scaled_matrix,
                preconditioner.factor,
                transformed_rhs,
                settings,
                statistics,
                controller);
        } else {
            correction = heavy_ball(
                scaled_matrix,
                preconditioner.factor,
                transformed_rhs,
                settings,
                statistics,
                controller);
        }
        total_inner_iterations = math::checked_add(
            total_inner_iterations,
            correction.iterations,
            "SPIR/FOSSILS inner iteration count exceeds LP64 limits");
        if (!correction.finite) {
            numerical_breakdown = true;
            break;
        }

        PosteriorResult completed =
            scheduled_iteration == correction.iterations
            ? std::move(scheduled_posterior)
            : evaluate_posterior(
                  scaled_matrix,
                  right_hand_side,
                  preconditioner,
                  base_solution,
                  correction.solution,
                  right_hand_side_norm,
                  settings.tolerance,
                  statistics);
        if (!completed.finite) {
            numerical_breakdown = true;
            break;
        }
        scaled_solution = std::move(completed.candidate);
        completed_refinements = refinement;
        if (completed.passed) {
            break;
        }
    }

    std::vector<double> solution = unscale_solution(
        scaled_solution, column_norms);
    if (numerical_breakdown || !all_finite(solution)) {
        LeastSquaresResult result;
        result.rows = matrix.rows();
        result.cols = matrix.cols();
        result.solution = std::move(solution);
        result.iterations = total_inner_iterations;
        result.depth = completed_refinements;
        result.auxiliary_width = settings.sketch_rows;
        result.basis_rank = preconditioner.rank;
        result.status = SolverStatus::numerical_breakdown;
        result.stop_reason = StopReason::numerical_breakdown;
        finish_result(result, {}, statistics, total_start);
        return result;
    }

    LeastSquaresResult result = validate_checkpoint(
        matrix,
        right_hand_side,
        settings,
        solution,
        total_inner_iterations,
        completed_refinements,
        preconditioner.rank,
        statistics);
    std::vector<IterationRecord> trace = std::move(result.trace);
    result.trace.clear();
    if (preconditioner.rank == 0 &&
        result.status != SolverStatus::success) {
        result.stop_reason = StopReason::exhausted_search_space;
    }
    finish_result(result, std::move(trace), statistics, total_start);
    return result;
}

}  // namespace

LeastSquaresResult solve_spir(
    const MatrixOperator& matrix,
    const double* right_hand_side,
    const SpirOptions& options) {
    math::validate_common_inputs(
        matrix, right_hand_side, options.regularization, options.tolerance);
    return solve_impl(matrix, right_hand_side, make_settings(matrix, options));
}

LeastSquaresResult solve_fossils(
    const MatrixOperator& matrix,
    const double* right_hand_side,
    const FossilsOptions& options) {
    math::validate_common_inputs(
        matrix, right_hand_side, options.regularization, options.tolerance);
    return solve_impl(matrix, right_hand_side, make_settings(matrix, options));
}

}  // namespace amfls
