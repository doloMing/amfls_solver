#include <ATen/ATen.h>
#include <torch/library.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <tuple>
#include <vector>

#include "amfls/amfls.hpp"
#include "amfls/aplicur.hpp"
#include "amfls/fixed_rbgk.hpp"
#include "amfls/lsmb.hpp"
#include "amfls/lsmr.hpp"
#include "amfls/lsrn.hpp"
#include "amfls/lsqr.hpp"
#include "amfls/randomized_block_cg.hpp"
#include "amfls/sparse_embedding_lsqr.hpp"
#include "amfls/spir_fossils.hpp"
#include "algorithms/mathematics/operators/row_major_dense_operator.hpp"
#include "algorithms/mathematics/operators/sparse_csr_operator.hpp"
#include "result_tensors.hpp"

namespace amfls {
namespace {

using TorchResult = std::tuple<
    at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor,
    at::Tensor, std::int64_t, std::int64_t>;

int checked_int(std::int64_t value, const char* name) {
    TORCH_CHECK(
        value >= std::numeric_limits<int>::min() &&
            value <= std::numeric_limits<int>::max(),
        name,
        " exceeds the C++ int range");
    return static_cast<int>(value);
}

constexpr std::uint64_t uint64_from_int64_transport(std::int64_t value) {
    return static_cast<std::uint64_t>(value);
}

static_assert(uint64_from_int64_transport(0) == 0ULL);
static_assert(
    uint64_from_int64_transport(std::numeric_limits<std::int64_t>::max()) ==
    0x7fffffffffffffffULL);
static_assert(
    uint64_from_int64_transport(std::numeric_limits<std::int64_t>::min()) ==
    0x8000000000000000ULL);
static_assert(uint64_from_int64_transport(-1) == 0xffffffffffffffffULL);

void check_matrix_header(const at::Tensor& input) {
    TORCH_CHECK(input.device().is_cpu(), "matrix must be on CPU");
    TORCH_CHECK(input.scalar_type() == at::kDouble, "matrix must use float64");
    TORCH_CHECK(input.dim() == 2, "matrix must have two dimensions");
    TORCH_CHECK(input.size(0) > 0 && input.size(1) > 0, "matrix cannot be empty");
    TORCH_CHECK(
        input.size(0) <= std::numeric_limits<int>::max() &&
        input.size(1) <= std::numeric_limits<int>::max(),
        "matrix exceeds the LP64 dimension limit");
    TORCH_CHECK(
        input.layout() == at::kStrided || input.layout() == at::kSparseCsr,
        "matrix must use dense strided or sparse CSR storage");
}

class TorchMatrixOperator {
public:
    explicit TorchMatrixOperator(const at::Tensor& input) {
        check_matrix_header(input);
        rows_ = static_cast<int>(input.size(0));
        cols_ = static_cast<int>(input.size(1));

        if (input.layout() == at::kStrided) {
            dense_values_ = input.contiguous();
            operator_ = std::make_unique<math::RowMajorDenseOperator>(
                rows_, cols_, dense_values_.data_ptr<double>());
            return;
        }

        row_offsets_ = input.crow_indices().to(at::kLong).contiguous();
        column_indices_ = input.col_indices().to(at::kLong).contiguous();
        values_ = input.values().contiguous();
        TORCH_CHECK(
            row_offsets_.dim() == 1 && row_offsets_.numel() == rows_ + 1,
            "CSR row offsets must have length rows + 1");
        TORCH_CHECK(
            column_indices_.dim() == 1 && values_.dim() == 1 &&
                column_indices_.numel() == values_.numel(),
            "CSR column indices and values must be one-dimensional and equally sized");
        operator_ = std::make_unique<math::SparseCsrOperator>(
            rows_,
            cols_,
            row_offsets_.data_ptr<std::int64_t>(),
            column_indices_.data_ptr<std::int64_t>(),
            values_.data_ptr<double>(),
            values_.numel());
    }

    int rows() const { return rows_; }
    int cols() const { return cols_; }
    const MatrixOperator& get() const { return *operator_; }

private:
    int rows_ = 0;
    int cols_ = 0;
    at::Tensor dense_values_;
    at::Tensor row_offsets_;
    at::Tensor column_indices_;
    at::Tensor values_;
    std::unique_ptr<MatrixOperator> operator_;
};

std::vector<double> copy_rhs(const at::Tensor& input, int rows) {
    TORCH_CHECK(input.device().is_cpu(), "right-hand side must be on CPU");
    TORCH_CHECK(input.scalar_type() == at::kDouble, "right-hand side must use float64");
    TORCH_CHECK(input.dim() == 1 && input.size(0) == rows, "right-hand side shape mismatch");
    auto values = input.accessor<double, 1>();
    std::vector<double> output(rows);
    for (int row = 0; row < rows; ++row) {
        output[row] = values[row];
    }
    return output;
}

TorchResult copy_result(const LeastSquaresResult& result) {
    namespace output = torch_binding::result_tensors;

    at::Tensor solution = at::empty({result.cols}, at::kDouble);
    std::copy(
        result.solution.begin(), result.solution.end(), solution.data_ptr<double>());

    const output::Metrics result_metrics = output::metrics(result);
    at::Tensor metrics = at::empty(
        {static_cast<std::int64_t>(output::metrics_width)},
        at::kDouble);
    std::copy(
        result_metrics.begin(),
        result_metrics.end(),
        metrics.data_ptr<double>());

    const output::Counters result_counters = output::counters(result);
    at::Tensor counters = at::empty(
        {static_cast<std::int64_t>(output::counters_width)},
        at::kLong);
    std::copy(
        result_counters.begin(),
        result_counters.end(),
        counters.data_ptr<std::int64_t>());

    const output::Timings result_timings = output::timings(result);
    at::Tensor timings = at::empty(
        {static_cast<std::int64_t>(output::timings_width)},
        at::kDouble);
    std::copy(
        result_timings.begin(),
        result_timings.end(),
        timings.data_ptr<double>());

    const std::int64_t candidate_count =
        static_cast<std::int64_t>(result.trace.size());
    at::Tensor candidate_trace_float = at::empty(
        {candidate_count,
         static_cast<std::int64_t>(output::trace_float_width)},
        at::kDouble);
    at::Tensor candidate_trace_int = at::empty(
        {candidate_count,
         static_cast<std::int64_t>(output::trace_int_width)},
        at::kLong);
    auto candidate_float_values =
        candidate_trace_float.accessor<double, 2>();
    auto candidate_int_values =
        candidate_trace_int.accessor<std::int64_t, 2>();
    for (std::int64_t row = 0; row < candidate_count; ++row) {
        const IterationRecord& record =
            result.trace[static_cast<std::size_t>(row)];
        const output::TraceFloatRow result_float = output::trace_float(record);
        const output::TraceIntRow result_int = output::trace_int(record);
        for (std::size_t column = 0;
             column < output::trace_float_width;
             ++column) {
            candidate_float_values[row][static_cast<std::int64_t>(column)] =
                result_float[column];
        }
        for (std::size_t column = 0;
             column < output::trace_int_width;
             ++column) {
            candidate_int_values[row][static_cast<std::int64_t>(column)] =
                result_int[column];
        }
    }

    return std::make_tuple(
        solution,
        metrics,
        counters,
        timings,
        candidate_trace_float,
        candidate_trace_int,
        static_cast<std::int64_t>(result.status),
        static_cast<std::int64_t>(result.stop_reason));
}

TorchResult run_amfls(
    const at::Tensor& matrix_tensor,
    const at::Tensor& b_tensor,
    double regularization,
    double tolerance,
    double failure_probability,
    std::int64_t maximum_epochs,
    std::int64_t maximum_depth,
    std::int64_t maximum_auxiliary_width,
    std::int64_t maximum_basis_size,
    std::int64_t seed,
    std::int64_t stream) {
    TorchMatrixOperator matrix(matrix_tensor);
    std::vector<double> b = copy_rhs(b_tensor, matrix.rows());
    AmflsOptions options;
    options.regularization = regularization;
    options.tolerance = tolerance;
    options.failure_probability = failure_probability;
    options.maximum_epochs = checked_int(maximum_epochs, "maximum_epochs");
    options.maximum_depth = checked_int(maximum_depth, "maximum_depth");
    options.maximum_auxiliary_width = checked_int(
        maximum_auxiliary_width, "maximum_auxiliary_width");
    options.maximum_basis_size = checked_int(
        maximum_basis_size, "maximum_basis_size");
    options.seed = uint64_from_int64_transport(seed);
    options.stream = uint64_from_int64_transport(stream);
    return copy_result(solve_amfls(matrix.get(), b.data(), options));
}

TorchResult run_fixed_rbgk(
    const at::Tensor& matrix_tensor,
    const at::Tensor& b_tensor,
    double regularization,
    double tolerance,
    double failure_probability,
    std::int64_t auxiliary_width,
    std::int64_t maximum_depth,
    std::int64_t maximum_basis_size,
    std::int64_t seed,
    std::int64_t stream) {
    TorchMatrixOperator matrix(matrix_tensor);
    std::vector<double> b = copy_rhs(b_tensor, matrix.rows());
    FixedRbgkOptions options;
    options.regularization = regularization;
    options.tolerance = tolerance;
    options.failure_probability = failure_probability;
    options.auxiliary_width = checked_int(auxiliary_width, "auxiliary_width");
    options.maximum_depth = checked_int(maximum_depth, "maximum_depth");
    options.maximum_basis_size = checked_int(
        maximum_basis_size, "maximum_basis_size");
    options.seed = uint64_from_int64_transport(seed);
    options.stream = uint64_from_int64_transport(stream);
    return copy_result(solve_fixed_rbgk(matrix.get(), b.data(), options));
}

TorchResult run_lsqr(
    const at::Tensor& matrix_tensor,
    const at::Tensor& b_tensor,
    double regularization,
    double tolerance,
    std::int64_t maximum_iterations) {
    TorchMatrixOperator matrix(matrix_tensor);
    std::vector<double> b = copy_rhs(b_tensor, matrix.rows());
    LsqrOptions options;
    options.regularization = regularization;
    options.tolerance = tolerance;
    options.maximum_iterations = checked_int(
        maximum_iterations, "maximum_iterations");
    return copy_result(solve_lsqr(matrix.get(), b.data(), options));
}

TorchResult run_lsmr(
    const at::Tensor& matrix_tensor,
    const at::Tensor& b_tensor,
    double regularization,
    double tolerance,
    std::int64_t maximum_iterations) {
    TorchMatrixOperator matrix(matrix_tensor);
    std::vector<double> b = copy_rhs(b_tensor, matrix.rows());
    LsmrOptions options;
    options.regularization = regularization;
    options.tolerance = tolerance;
    options.maximum_iterations = checked_int(
        maximum_iterations, "maximum_iterations");
    return copy_result(solve_lsmr(matrix.get(), b.data(), options));
}

TorchResult run_lsmb(
    const at::Tensor& matrix_tensor,
    const at::Tensor& b_tensor,
    double regularization,
    double tolerance,
    std::int64_t maximum_iterations) {
    TorchMatrixOperator matrix(matrix_tensor);
    std::vector<double> b = copy_rhs(b_tensor, matrix.rows());
    LsmbOptions options;
    options.regularization = regularization;
    options.tolerance = tolerance;
    options.maximum_iterations = checked_int(
        maximum_iterations, "maximum_iterations");
    return copy_result(solve_lsmb(matrix.get(), b.data(), options));
}

TorchResult run_lsrn(
    const at::Tensor& matrix_tensor,
    const at::Tensor& b_tensor,
    double regularization,
    double tolerance,
    double oversampling,
    double relative_rank_tolerance,
    double absolute_rank_tolerance,
    std::int64_t maximum_iterations,
    std::int64_t sketch_block_size,
    std::int64_t seed,
    std::int64_t stream) {
    TorchMatrixOperator matrix(matrix_tensor);
    std::vector<double> b = copy_rhs(b_tensor, matrix.rows());
    LsrnOptions options;
    options.regularization = regularization;
    options.tolerance = tolerance;
    options.oversampling = oversampling;
    options.relative_rank_tolerance = relative_rank_tolerance;
    options.absolute_rank_tolerance = absolute_rank_tolerance;
    options.maximum_iterations = checked_int(
        maximum_iterations, "maximum_iterations");
    options.sketch_block_size = checked_int(
        sketch_block_size, "sketch_block_size");
    options.seed = uint64_from_int64_transport(seed);
    options.stream = uint64_from_int64_transport(stream);
    return copy_result(solve_lsrn(matrix.get(), b.data(), options));
}

TorchResult run_randomized_block_cg(
    const at::Tensor& matrix_tensor,
    const at::Tensor& b_tensor,
    double regularization,
    double tolerance,
    std::int64_t random_block_size,
    std::int64_t maximum_depth,
    std::int64_t seed,
    std::int64_t stream) {
    TorchMatrixOperator matrix(matrix_tensor);
    std::vector<double> b = copy_rhs(b_tensor, matrix.rows());
    RandomizedBlockCgOptions options;
    options.regularization = regularization;
    options.tolerance = tolerance;
    options.random_block_size = checked_int(
        random_block_size, "random_block_size");
    options.maximum_depth = checked_int(maximum_depth, "maximum_depth");
    options.seed = uint64_from_int64_transport(seed);
    options.stream = uint64_from_int64_transport(stream);
    return copy_result(
        solve_randomized_block_cg(matrix.get(), b.data(), options));
}

TorchResult run_sparse_embedding_lsqr(
    const at::Tensor& matrix_tensor,
    const at::Tensor& b_tensor,
    double regularization,
    double tolerance,
    double embedding_distortion,
    double embedding_failure_probability,
    std::int64_t sketch_rows,
    std::int64_t embedding_nonzeros,
    double relative_rank_tolerance,
    double absolute_rank_tolerance,
    std::int64_t maximum_iterations,
    std::int64_t sketch_block_size,
    std::int64_t seed,
    std::int64_t stream) {
    TorchMatrixOperator matrix(matrix_tensor);
    std::vector<double> b = copy_rhs(b_tensor, matrix.rows());
    SparseEmbeddingLsqrOptions options;
    options.regularization = regularization;
    options.tolerance = tolerance;
    options.embedding_distortion = embedding_distortion;
    options.embedding_failure_probability = embedding_failure_probability;
    options.sketch_rows = checked_int(sketch_rows, "sketch_rows");
    options.embedding_nonzeros = checked_int(
        embedding_nonzeros, "embedding_nonzeros");
    options.relative_rank_tolerance = relative_rank_tolerance;
    options.absolute_rank_tolerance = absolute_rank_tolerance;
    options.maximum_iterations = checked_int(
        maximum_iterations, "maximum_iterations");
    options.sketch_block_size = checked_int(
        sketch_block_size, "sketch_block_size");
    options.seed = uint64_from_int64_transport(seed);
    options.stream = uint64_from_int64_transport(stream);
    return copy_result(
        solve_sparse_embedding_lsqr(matrix.get(), b.data(), options));
}

TorchResult run_spir(
    const at::Tensor& matrix_tensor,
    const at::Tensor& b_tensor,
    double regularization,
    double tolerance,
    std::int64_t sketch_rows,
    std::int64_t embedding_nonzeros,
    std::int64_t maximum_inner_iterations,
    double relative_rank_tolerance,
    double absolute_rank_tolerance,
    std::int64_t sketch_block_size,
    std::int64_t seed,
    std::int64_t stream) {
    TorchMatrixOperator matrix(matrix_tensor);
    std::vector<double> b = copy_rhs(b_tensor, matrix.rows());
    SpirOptions options;
    options.regularization = regularization;
    options.tolerance = tolerance;
    options.sketch_rows = checked_int(sketch_rows, "sketch_rows");
    options.embedding_nonzeros = checked_int(
        embedding_nonzeros, "embedding_nonzeros");
    options.maximum_inner_iterations = checked_int(
        maximum_inner_iterations, "maximum_inner_iterations");
    options.relative_rank_tolerance = relative_rank_tolerance;
    options.absolute_rank_tolerance = absolute_rank_tolerance;
    options.sketch_block_size = checked_int(
        sketch_block_size, "sketch_block_size");
    options.seed = uint64_from_int64_transport(seed);
    options.stream = uint64_from_int64_transport(stream);
    return copy_result(solve_spir(matrix.get(), b.data(), options));
}

TorchResult run_fossils(
    const at::Tensor& matrix_tensor,
    const at::Tensor& b_tensor,
    double regularization,
    double tolerance,
    std::int64_t sketch_rows,
    std::int64_t embedding_nonzeros,
    std::int64_t maximum_inner_iterations,
    double distortion_safety,
    double relative_rank_tolerance,
    double absolute_rank_tolerance,
    std::int64_t sketch_block_size,
    std::int64_t seed,
    std::int64_t stream) {
    TorchMatrixOperator matrix(matrix_tensor);
    std::vector<double> b = copy_rhs(b_tensor, matrix.rows());
    FossilsOptions options;
    options.regularization = regularization;
    options.tolerance = tolerance;
    options.sketch_rows = checked_int(sketch_rows, "sketch_rows");
    options.embedding_nonzeros = checked_int(
        embedding_nonzeros, "embedding_nonzeros");
    options.maximum_inner_iterations = checked_int(
        maximum_inner_iterations, "maximum_inner_iterations");
    options.distortion_safety = distortion_safety;
    options.relative_rank_tolerance = relative_rank_tolerance;
    options.absolute_rank_tolerance = absolute_rank_tolerance;
    options.sketch_block_size = checked_int(
        sketch_block_size, "sketch_block_size");
    options.seed = uint64_from_int64_transport(seed);
    options.stream = uint64_from_int64_transport(stream);
    return copy_result(solve_fossils(matrix.get(), b.data(), options));
}

TorchResult run_aplicur(
    const at::Tensor& matrix_tensor,
    const at::Tensor& b_tensor,
    double regularization,
    double tolerance,
    std::int64_t block_size,
    std::int64_t sparse_sign_nonzeros,
    std::int64_t spectral_probe_count,
    double cur_tolerance,
    double re_preconditioning_tolerance,
    double dynamic_stopping_tolerance,
    std::int64_t maximum_iterations,
    double relative_rank_tolerance,
    double absolute_rank_tolerance,
    std::int64_t seed) {
    TORCH_CHECK(
        matrix_tensor.layout() == at::kStrided,
        "APLICUR requires a dense strided matrix because it selects explicit rows and columns");
    check_matrix_header(matrix_tensor);
    const at::Tensor matrix = matrix_tensor.contiguous();
    const int rows = static_cast<int>(matrix.size(0));
    const int cols = static_cast<int>(matrix.size(1));
    std::vector<double> b = copy_rhs(b_tensor, rows);
    AplicurOptions options;
    options.regularization = regularization;
    options.tolerance = tolerance;
    options.block_size = checked_int(block_size, "block_size");
    options.sparse_sign_nonzeros = checked_int(
        sparse_sign_nonzeros, "sparse_sign_nonzeros");
    options.spectral_probe_count = checked_int(
        spectral_probe_count, "spectral_probe_count");
    options.cur_tolerance = cur_tolerance;
    options.re_preconditioning_tolerance = re_preconditioning_tolerance;
    options.dynamic_stopping_tolerance = dynamic_stopping_tolerance;
    options.maximum_iterations = checked_int(
        maximum_iterations, "maximum_iterations");
    options.relative_rank_tolerance = relative_rank_tolerance;
    options.absolute_rank_tolerance = absolute_rank_tolerance;
    options.seed = uint64_from_int64_transport(seed);
    return copy_result(solve_aplicur(
        matrix.data_ptr<double>(), rows, cols, b.data(), options));
}

}  // namespace
}  // namespace amfls

TORCH_LIBRARY(amfls, module) {
    module.def(
        "run_amfls(Tensor matrix, Tensor b, float regularization, float tolerance, "
        "float failure_probability, int maximum_epochs=0, int maximum_depth=0, "
        "int maximum_auxiliary_width=0, int maximum_basis_size=0, "
        "int seed=0, int stream=0) -> "
        "(Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, int, int)");
    module.def(
        "run_fixed_rbgk(Tensor matrix, Tensor b, float regularization, float tolerance, "
        "float failure_probability, int auxiliary_width, int maximum_depth, "
        "int maximum_basis_size, int seed, int stream) -> "
        "(Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, int, int)");
    module.def(
        "run_lsqr(Tensor matrix, Tensor b, float regularization, float tolerance, "
        "int maximum_iterations=0) -> "
        "(Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, int, int)");
    module.def(
        "run_lsmr(Tensor matrix, Tensor b, float regularization, float tolerance, "
        "int maximum_iterations=0) -> "
        "(Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, int, int)");
    module.def(
        "run_lsmb(Tensor matrix, Tensor b, float regularization, float tolerance, "
        "int maximum_iterations=0) -> "
        "(Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, int, int)");
    module.def(
        "run_lsrn(Tensor matrix, Tensor b, float regularization, float tolerance, "
        "float oversampling, float relative_rank_tolerance, "
        "float absolute_rank_tolerance, int maximum_iterations=0, "
        "int sketch_block_size=0, int seed=0, int stream=0) -> "
        "(Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, int, int)");
    module.def(
        "run_randomized_block_cg(Tensor matrix, Tensor b, float regularization, "
        "float tolerance, int random_block_size, int maximum_depth=0, "
        "int seed=0, int stream=0) -> "
        "(Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, int, int)");
    module.def(
        "run_sparse_embedding_lsqr(Tensor matrix, Tensor b, float regularization, "
        "float tolerance, float embedding_distortion, "
        "float embedding_failure_probability, int sketch_rows=0, "
        "int embedding_nonzeros=1, "
        "float relative_rank_tolerance=1e-12, "
        "float absolute_rank_tolerance=0., int maximum_iterations=0, "
        "int sketch_block_size=0, int seed=0, int stream=0) -> "
        "(Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, int, int)");
    module.def(
        "run_spir(Tensor matrix, Tensor b, float regularization, float tolerance, "
        "int sketch_rows=0, int embedding_nonzeros=8, "
        "int maximum_inner_iterations=50, "
        "float relative_rank_tolerance=6.661338147750939e-15, "
        "float absolute_rank_tolerance=0., int sketch_block_size=32, "
        "int seed=0, int stream=0) -> "
        "(Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, int, int)");
    module.def(
        "run_fossils(Tensor matrix, Tensor b, float regularization, float tolerance, "
        "int sketch_rows=0, int embedding_nonzeros=8, "
        "int maximum_inner_iterations=100, float distortion_safety=1., "
        "float relative_rank_tolerance=6.661338147750939e-15, "
        "float absolute_rank_tolerance=0., int sketch_block_size=32, "
        "int seed=0, int stream=0) -> "
        "(Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, int, int)");
    module.def(
        "run_aplicur(Tensor matrix, Tensor b, float regularization, "
        "float tolerance, int block_size=8, int sparse_sign_nonzeros=8, "
        "int spectral_probe_count=10, float cur_tolerance=0., "
        "float re_preconditioning_tolerance=10., "
        "float dynamic_stopping_tolerance=150., int maximum_iterations=0, "
        "float relative_rank_tolerance=1e-12, "
        "float absolute_rank_tolerance=0., int seed=0) -> "
        "(Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, int, int)");
}

TORCH_LIBRARY_IMPL(amfls, CPU, module) {
    module.impl("run_amfls", &amfls::run_amfls);
    module.impl("run_fixed_rbgk", &amfls::run_fixed_rbgk);
    module.impl("run_lsqr", &amfls::run_lsqr);
    module.impl("run_lsmr", &amfls::run_lsmr);
    module.impl("run_lsmb", &amfls::run_lsmb);
    module.impl("run_lsrn", &amfls::run_lsrn);
    module.impl(
        "run_randomized_block_cg", &amfls::run_randomized_block_cg);
    module.impl(
        "run_sparse_embedding_lsqr", &amfls::run_sparse_embedding_lsqr);
    module.impl("run_spir", &amfls::run_spir);
    module.impl("run_fossils", &amfls::run_fossils);
    module.impl("run_aplicur", &amfls::run_aplicur);
}

TORCH_LIBRARY_IMPL(amfls, SparseCsrCPU, module) {
    module.impl("run_amfls", &amfls::run_amfls);
    module.impl("run_fixed_rbgk", &amfls::run_fixed_rbgk);
    module.impl("run_lsqr", &amfls::run_lsqr);
    module.impl("run_lsmr", &amfls::run_lsmr);
    module.impl("run_lsmb", &amfls::run_lsmb);
    module.impl("run_lsrn", &amfls::run_lsrn);
    module.impl(
        "run_randomized_block_cg", &amfls::run_randomized_block_cg);
    module.impl(
        "run_sparse_embedding_lsqr", &amfls::run_sparse_embedding_lsqr);
    module.impl("run_spir", &amfls::run_spir);
    module.impl("run_fossils", &amfls::run_fossils);
    module.impl("run_aplicur", &amfls::run_aplicur);
}
