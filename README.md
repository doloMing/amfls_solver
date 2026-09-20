# AMFLS: Adaptive Matrix-Free Least Squares Solver

AMFLS computes ordinary and Tikhonov-regularized least-squares solutions for
large real linear systems. It is designed for applications that can return
the products $Av$ and $A^*u$ for requested vectors without storing or
factorizing the full matrix $A$.

For a linear operator $A:\mathbb{R}^n\rightarrow\mathbb{R}^m$, observed data
$b\in\mathbb{R}^m$, and a regularization parameter $\lambda\geq 0$, AMFLS
computes

$$
x_\lambda=\underset{x\in\mathrm{range}(A^*)}{\arg\min}\;
\|Ax-b\|_2^2+\lambda\|x\|_2^2.
$$

When $\lambda=0$, the output targets the minimum-norm least-squares solution
$A^\dagger b$. When $\lambda>0$, it targets the unique Tikhonov-regularized
solution. The user supplies $b$, $\lambda$, a requested tolerance, and
functions that compute $Av$ and $A^*u$. No estimate of the spectrum, rank,
conditioning, or suitable block size is required.

AMFLS begins with one basis direction. During the computation, it measures
the decrease in the error bound and the time taken by recent updates. It then
predicts whether the current width or a larger width will reach the requested
accuracy sooner. Every accepted expansion remains part of one continuing
block Golub--Kahan process.

The repository also contains the comparison methods used in the numerical
study. They include fixed-width randomized block Golub--Kahan, LSQR, LSMR,
LSMB, LSRN, sparse-embedding LSQR, SPIR, FOSSILS, randomized block CG, and
APLICUR.

The implementation is written in C++20 and uses BLAS and LAPACK. A PyTorch
CPU extension calls the same C++ algorithm implementations.

**Version:** 0.2.0 &nbsp; | &nbsp; **License:** MIT &nbsp; | &nbsp; **Language:** C++20 / PyTorch CPU

## Contents

- [1. Repository structure](#1-repository-structure)
- [2. Choosing and calling an algorithm](#2-choosing-and-calling-an-algorithm)
- [3. Inputs, outputs, counters, and diagnostics](#3-inputs-outputs-counters-and-diagnostics)
- [4. Build and deployment](#4-build-and-deployment)
- [5. C++ and PyTorch examples](#5-c-and-pytorch-examples)
- [6. Algorithm structure and pseudocode](#6-algorithm-structure-and-pseudocode)
- [7. Mathematical principles](#7-mathematical-principles)
- [8. Why AMFLS is fast and accurate](#8-why-amfls-is-fast-and-accurate)
- [9. Paper, citation, author, and license](#9-paper-citation-author-and-license)

## 1. Repository structure

```text
amfls_solver/
├── CMakeLists.txt
├── LICENSE
├── README.md
├── include/amfls/
│   ├── amfls.hpp                    # AMFLS options and C++ entry point
│   ├── matrix_operator.hpp          # matrix-free operator interface
│   ├── least_squares_result.hpp     # solution, status, counters, and trace
│   ├── fixed_rbgk.hpp               # fixed-width RBGK baseline
│   ├── lsqr.hpp, lsmr.hpp, lsmb.hpp # scalar Krylov baselines
│   ├── lsrn.hpp                     # Gaussian-preconditioned baseline
│   ├── sparse_embedding_lsqr.hpp    # sparse-embedding baseline
│   ├── spir_fossils.hpp             # SPIR and FOSSILS baselines
│   ├── randomized_block_cg.hpp      # regularized block-CG baseline
│   └── aplicur.hpp                  # explicit-matrix APLICUR baseline
├── src/algorithms/
│   ├── amfls/                       # adaptive-width AMFLS controller
│   ├── mathematics/                 # block GKB, validation, and linear algebra
│   └── */                           # independent baseline implementations
├── bindings/torch/
│   ├── torch_binding.cpp            # torch.ops.amfls entry points
│   └── result_tensors.hpp           # tensor layout for returned results
├── python/amfls/
│   ├── binding.py                   # Python functions and SolveResult
│   └── __init__.py                  # public Python names
└── experiments/
    ├── runners/                     # comparison experiment scripts
    ├── analysis/                    # result analysis scripts
    ├── configs/                     # experiment settings
    └── results/formal_100_v2/       # final experiment data
```

### Public C++ interface

An application represents $A$ by implementing the following interface.

```cpp
class MatrixOperator {
public:
    virtual ~MatrixOperator() = default;
    virtual int rows() const = 0;
    virtual int cols() const = 0;

    virtual void apply(
        const double* x, int block_cols, double* y) const = 0;

    virtual void apply_transpose(
        const double* y, int block_cols, double* x) const = 0;
};
```

For `apply`, `x` stores an $n\times s$ column-major block and `y` receives
the $m\times s$ block $Ax$. For `apply_transpose`, `y` stores an $m\times s$
block and `x` receives the $n\times s$ block $A^*y$. The operator can
represent a dense or sparse matrix, a simulation, a transform, or any other
calculation that provides these products.

The interface also lets an operator describe the relative cost of processing
a block and the norm bounds used by the stopping test. The built-in dense,
row-major dense, and CSR operators provide both. The default block-cost model
scales with the number of columns. A custom operator can override
`relative_block_product_cost` when several columns share work. It implements
`validation_error_model` to provide the norm bounds used for successful
termination.

### Internal components

The C++ core owns the complete computation. It builds the variable-width
block Golub--Kahan basis, computes each projected solution, evaluates the error
bound, selects the next width, and records work and timing data. The PyTorch
extension converts CPU tensors into built-in matrix operators and returns the
same result fields as the C++ interface.

## 2. Choosing and calling an algorithm

### 2.1 Which method should be used?

| Method | Supported problem | Main setting | Recommended role |
|---|---|---|---|
| **AMFLS** | Ordinary and regularized, square, tall, or wide | Selects its block width from observed progress and time | Default adaptive least-squares computation |
| **Fixed RBGK** | Ordinary and regularized, square, tall, or wide | Uses one fixed random block width | Controlled comparison with AMFLS |
| **LSQR, LSMR, LSMB** | Ordinary and regularized, square, tall, or wide | Add one Krylov direction per update | Scalar Krylov comparisons |
| **LSRN** | Ordinary and regularized | Builds a Gaussian preconditioner before iteration | Randomized preconditioning comparison |
| **Sparse-embedding LSQR** | Tall, full-column-rank ordinary least squares | Builds a sparse embedding and then runs LSQR | Sparse-sketch comparison |
| **SPIR and FOSSILS** | Tall ordinary least squares | Use sparse sketching, preconditioning, and refinement | High-accuracy randomized comparisons |
| **Randomized block CG** | Regularized least squares | Applies block CG to the shifted normal system | Regularized block-Krylov comparison |
| **APLICUR** | Regularized least squares with an explicit dense matrix | Uses sampled rows and columns to build a CUR preconditioner | Explicit-matrix regularized comparison |

Use **AMFLS** when a suitable block width is unknown before the computation.
It accepts the same mathematical problem for dense, sparse, streamed, and
implicit operators. The remaining methods reproduce the numerical
comparisons in the accompanying paper. Their exact domains and parameter
mappings are summarized in Sections 2.1, 2.3, and 6.3.

### 2.2 AMFLS C++ options

```cpp
amfls::AmflsOptions options;
options.regularization = 0.0;
options.tolerance = 1e-8;
options.failure_probability = 1e-6;
options.maximum_epochs = 0;
options.maximum_depth = 0;
options.maximum_auxiliary_width = 0;
options.maximum_basis_size = 0;
options.seed = 0;
options.stream = 0;
```

| Option | Valid values and meaning |
|---|---|
| `regularization` | Tikhonov parameter $\lambda\geq0$. Zero selects ordinary least squares. |
| `tolerance` | Requested relative error tolerance $\tau>0$. Section 2.4 gives its meaning for each problem. |
| `failure_probability` | Value in $(0,1)$. The current release validates this public option. Width selection and stopping are determined by the computed error bound, timing data, and work limits. |
| `maximum_epochs` | Maximum number of widening stages. Zero selects a dimension-based limit. |
| `maximum_depth` | Maximum number of completed Golub--Kahan levels. Zero permits up to $\min(m,n)$ levels. |
| `maximum_auxiliary_width` | Maximum number of random auxiliary directions. Zero permits any width that fits the basis limit. |
| `maximum_basis_size` | Maximum number of retained basis vectors. Zero selects the natural dimension limit. |
| `seed` | Random generator seed used for auxiliary directions. |
| `stream` | Independent random stream within a seed. |

The default work limits allow the basis to grow to its natural dimension.
Applications with a fixed memory or time budget can set the four maximum
values explicitly. A successful return always satisfies the requested
accuracy test. Reaching a work limit produces a distinct non-success status.

### 2.3 Baseline options

Every baseline has its own public options structure under `include/amfls/`.
All methods share `regularization`, `tolerance`, and the result type whenever
their mathematical domain includes the requested problem.

| Method | Principal method-specific options |
|---|---|
| Fixed RBGK | `auxiliary_width`, `maximum_depth`, `maximum_basis_size` |
| LSQR, LSMR, LSMB | `maximum_iterations` |
| LSRN | `oversampling`, numerical-rank tolerances, `maximum_iterations`, `sketch_block_size` |
| Sparse-embedding LSQR | embedding distortion and failure probability, sketch rows, nonzeros per embedded column, numerical-rank tolerances |
| SPIR | sketch rows, embedding nonzeros, inner-iteration limit, numerical-rank tolerances |
| FOSSILS | SPIR settings and `distortion_safety` |
| Randomized block CG | `random_block_size`, `maximum_depth` |
| APLICUR | block size, sparse-sign nonzeros, spectral probes, CUR and dynamic stopping tolerances |

The public Python functions use the same option names and default values.
APLICUR receives an explicit dense matrix because it selects individual rows
and columns. The other methods accept dense or CSR tensors through the
PyTorch interface.

### 2.4 Ordinary and regularized accuracy

For ordinary least squares, `tolerance` bounds a normwise backward error. A
value of $10^{-8}$ asks for a computed vector that is an exact solution of a
nearby least-squares problem whose relative perturbation is at most $10^{-8}$.

For positive regularization, `tolerance` bounds the relative error measured
in the energy norm associated with $A^*A+\lambda I$.

$$
\frac{\|x-x_\lambda\|_{A^*A+\lambda I}}
     {\|x_\lambda\|_{A^*A+\lambda I}}
\leq \tau.
$$

AMFLS evaluates computable upper bounds for these errors from the current
solution, $Ax-b$, and $A^*(Ax-b)+\lambda x$. These evaluations use fresh
operator calls and determine whether the returned status is `success`.

## 3. Inputs, outputs, counters, and diagnostics

### 3.1 Least-squares result

The C++ call

```cpp
amfls::LeastSquaresResult result =
    amfls::solve_amfls(matrix_operator, b.data(), options);
```

returns the following main fields.

| Field | Interpretation |
|---|---|
| `rows`, `cols` | Shape $m\times n$ of the operator. |
| `solution` | Computed vector $x\in\mathbb{R}^n$. |
| `objective` | $\|Ax-b\|_2^2+\lambda\|x\|_2^2$. |
| `residual_norm` | $\|Ax-b\|_2$. |
| `normal_residual_norm` | $\|A^*(Ax-b)+\lambda x\|_2$. |
| `solution_norm` | $\|x\|_2$. |
| `solution_energy_norm` | $\|x\|_{A^*A+\lambda I}$. |
| `backward_error_upper_bound` | Ordinary least-squares error bound used for stopping. |
| `relative_energy_error_upper_bound` | Regularized relative error bound used for stopping. |
| `objective_gap_upper_bound` | Upper bound for $F_\lambda(x)-F_\lambda(x_\lambda)$. |
| `iterations` | Number of evaluated candidate solutions. |
| `depth` | Number of completed Golub--Kahan levels. |
| `auxiliary_width` | Number of random auxiliary directions requested by the current method. |
| `basis_rank` | Number of retained basis vectors. |
| `status` | Computation outcome from the table below. |
| `stop_reason` | Specific reason for termination. |
| `statistics` | Operator work, random work, and timing data. |
| `trace` | One record for every evaluated candidate. |

The result also includes the two ordinary backward-error branches, absolute
regularized error bounds, and diagnostics for the optional regularized
correction. Section 7 summarizes their mathematical meaning.

### 3.2 Computation status

Always inspect `status` or the Python property `result.success` before using
the returned vector as a completed computation.

| Integer | C++ enum | Meaning |
|---:|---|---|
| `0` | `success` | The returned vector satisfies the requested accuracy. |
| `1` | `work_limit` | A depth or epoch limit was reached first. |
| `2` | `basis_limit` | The configured basis capacity was reached first. |
| `3` | `precision_limit` | Numerical rank loss prevents a valid continuation. |
| `4` | `numerical_breakdown` | A nonfinite or unusable numerical state ended the computation. |

### 3.3 Stop reasons

| Integer | C++ enum | Meaning |
|---:|---|---|
| `0` | `certified_optimality` | A method-specific optimality test succeeded. |
| `1` | `exact_stationarity` | The freshly evaluated gradient is zero. |
| `2` | `exhausted_search_space` | The available search space is exhausted. |
| `3` | `maximum_depth` | The depth or iteration limit was reached. |
| `4` | `maximum_epochs` | The AMFLS widening-stage limit was reached. |
| `5` | `maximum_basis` | The basis capacity was reached. |
| `6` | `precision_limit` | Numerical rank loss ended the recurrence. |
| `7` | `numerical_breakdown` | A nonfinite calculation ended the recurrence. |
| `8` | `compatible_backward_error` | The ordinary compatible-system error test succeeded. |
| `9` | `least_squares_backward_error` | The ordinary least-squares optimality test succeeded. |
| `10` | `relative_energy_error` | The regularized relative energy-error test succeeded. |

### 3.4 Work counters

`result.statistics` separates basis construction from final validation and
records vectors and calls independently.

| Field group | Contents |
|---|---|
| `a_columns`, `at_columns` | Total vector columns sent to the calculations for $Av$ and $A^*u$. A block of width $s$ contributes $s$ columns. |
| `a_block_calls`, `at_block_calls` | Number of requests made for $Av$ and $A^*u$, independent of block width. |
| `search_*` | Columns and calls used to construct a basis, sketch, or preconditioner. |
| `validation_*` | Columns and calls used to evaluate the returned solution on the original problem. |
| `sketch_*` | Work used to construct a randomized sketch or preconditioner. |
| `iterative_*` | Work used after sketch construction by the baseline iteration. |
| `gaussian_random_columns`, `gaussian_random_values` | Random data generated by AMFLS or a randomized baseline. |
| `gaussian_random_block_requests` | Number of requests made to the Gaussian generator. |

The timing fields are `total_seconds`, `a_seconds`, `at_seconds`,
`orthogonalization_seconds`, `projected_solve_seconds`, and
`validation_seconds`. The Python wrapper also reports `other_seconds` and
separates base validation from the optional regularized correction.
Validation includes its calls for $Av$ and $A^*u$, so these timing fields form
a breakdown with overlapping entries and should not be added together.

### 3.5 PyTorch return value

The registered Torch operators return

```text
(solution, metrics, counters, timings,
 trace_float, trace_int, status, stop_reason)
```

The Python functions under `python/amfls/` decode this tuple into
`SolveResult`. Normal use should access named attributes such as
`result.solution`, `result.backward_error_upper_bound`, and
`result.total_seconds`.

The raw tensor layouts are as follows.

| Item | Shape | Contents |
|---|---:|---|
| `solution` | `(n,)` | Computed least-squares vector. |
| `metrics` | `(21,)` | Objective, residual norms, error bounds, and correction diagnostics. |
| `counters` | `(36,)` | Work counts, dimensions, and correction state. |
| `timings` | `(9,)` | Total and component times. |
| `trace_float` | `(checkpoints, 29)` | Floating-point data for each evaluated candidate. |
| `trace_int` | `(checkpoints, 4)` | Status, stop reason, success flag, and correction state at each checkpoint. |
| `status` | scalar integer | Computation status. |
| `stop_reason` | scalar integer | Termination reason. |

The `metrics` indices are

```text
 0 objective                         11 relative normal residual bound
 1 residual norm                     12 energy error bound
 2 augmented residual norm           13 relative energy error bound
 3 normal residual norm              14 objective gap bound
 4 solution norm                     15 base regularized error bound
 5 solution energy norm              16 corrected regularized error bound
 6 operator norm lower bound         17 correction coefficient
 7 augmented operator lower bound    18 correction energy term
 8 compatible backward error bound   19 correction cross term
 9 least-squares backward bound      20 correction residual term
10 backward error bound
```

The `counters` indices are

```text
 0 A columns                    18 iterative A block calls
 1 A* columns                   19 iterative A* block calls
 2 A block calls                20 Gaussian columns
 3 A* block calls               21 Gaussian values
 4 search A columns             22 search A block calls
 5 search A* columns            23 search A* block calls
 6 validation A columns         24 validation A block calls
 7 validation A* columns        25 validation A* block calls
 8 candidate evaluations        26 base validation A columns
 9 completed depth              27 base validation A* columns
10 auxiliary width              28 base validation A block calls
11 basis rank                   29 base validation A* block calls
12 sketch A columns             30 correction A columns
13 sketch A* columns            31 correction A* columns
14 sketch A block calls         32 correction A block calls
15 sketch A* block calls        33 correction A* block calls
16 iterative A columns          34 Gaussian block requests
17 iterative A* columns         35 correction disposition
```

The `timings` indices are

```text
0 total                       5 validation
1 calculations of Av         6 unclassified total time
2 calculations of A*u        7 base validation
3 orthogonalization           8 regularized correction
4 projected computation
```

### 3.6 Complete PyTorch function list

The shared library registers the following functions under
`torch.ops.amfls`. The Python package exports wrappers with the same names.

| Function | Main inputs | Purpose |
|---|---|---|
| `run_amfls` | matrix, $b$, $\lambda$, tolerance, work limits, random options | Adaptive AMFLS computation |
| `run_fixed_rbgk` | matrix, $b$, fixed width, maximum depth, tolerance | Fixed-width block Golub--Kahan comparison |
| `run_lsqr` | matrix, $b$, $\lambda$, tolerance, iteration limit | LSQR comparison |
| `run_lsmr` | matrix, $b$, $\lambda$, tolerance, iteration limit | LSMR comparison |
| `run_lsmb` | matrix, $b$, $\lambda$, tolerance, iteration limit | LSMB comparison |
| `run_lsrn` | matrix, $b$, $\lambda$, oversampling and rank settings | LSRN comparison |
| `run_sparse_embedding_lsqr` | matrix, $b$, sparse embedding settings | Sparse-embedding LSQR comparison |
| `run_spir` | matrix, $b$, sparse sketch and refinement settings | SPIR comparison |
| `run_fossils` | matrix, $b$, sparse sketch and refinement settings | FOSSILS comparison |
| `run_randomized_block_cg` | matrix, $b$, positive $\lambda$, random block settings | Regularized randomized block-CG comparison |
| `run_aplicur` | explicit dense matrix, $b$, positive $\lambda$, CUR settings | Regularized APLICUR comparison |

## 4. Build and deployment

### 4.1 Requirements

- CMake 3.24 or newer
- a C++20 compiler
- an LP64 BLAS implementation
- LAPACK
- OpenMP for optional threaded C++ execution
- Python for the PyTorch extension and experiment scripts
- a CPU PyTorch installation for the PyTorch extension

The implementation uses real FP64 data and 32-bit BLAS and LAPACK integers.

### 4.2 Linux dependencies

On Debian or Ubuntu, a typical C++ build uses

```bash
sudo apt update
sudo apt install build-essential cmake libopenblas-dev liblapack-dev
```

For the PyTorch extension, activate the Python environment that contains the
desired CPU PyTorch installation before configuring CMake.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch
```

### 4.3 Build the C++ core

```bash
git clone https://github.com/doloMing/amfls_solver.git
cd amfls_solver

cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DAMFLS_BUILD_TORCH=OFF

cmake --build build --parallel
```

The main library is `build/libamfls_core.a`.

For a CMake application, add this repository as a subdirectory.

```cmake
cmake_minimum_required(VERSION 3.24)
project(my_amfls_app LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(AMFLS_BUILD_TORCH OFF CACHE BOOL "" FORCE)

add_subdirectory(external/amfls_solver)
add_executable(my_app main.cpp)
target_link_libraries(my_app PRIVATE amfls_core)
```

### 4.4 Build the PyTorch extension

Run CMake from the Python environment containing PyTorch.

```bash
cmake -S . -B build-torch \
  -DCMAKE_BUILD_TYPE=Release \
  -DAMFLS_BUILD_TORCH=ON \
  -DPython3_EXECUTABLE="$(command -v python)"

cmake --build build-torch --parallel
```

The extension is `build-torch/amfls_torch.so`. It registers Torch operators
when loaded. Add the repository's `python` directory to `PYTHONPATH` to use
the named Python interface.

```bash
export PYTHONPATH="$PWD/python${PYTHONPATH:+:$PYTHONPATH}"
```

```python
from amfls import load_library

load_library("build-torch/amfls_torch.so")
```

### 4.5 Thread control

AMFLS uses the selected BLAS implementation and uses OpenMP when it is
available. Set thread counts before starting a process when comparing running
times.

```bash
export OMP_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export MKL_NUM_THREADS=8
```

When several computations run concurrently, assign a suitable thread count
to each process so that the total does not exceed the available CPU cores.

### 4.6 Numerical model

The released implementation uses real FP64 arithmetic. The built-in dense,
row-major dense, and CSR operators provide the norm information used by the
common accuracy test. Custom operators should return a valid
`MatrixOperatorValidationErrorModel` so that the computation can attach a
finite error bound and report successful termination.

## 5. C++ and PyTorch examples

### 5.1 Matrix-free C++ example

This example computes a diagonal least-squares solution by defining the products
$Av$ and $A^*u$. It never stores a dense matrix.

```cpp
#include <algorithm>
#include <cmath>
#include <iostream>
#include <utility>
#include <vector>

#include "amfls/amfls.hpp"

class DiagonalOperator final : public amfls::MatrixOperator {
public:
    explicit DiagonalOperator(std::vector<double> diagonal)
        : diagonal_(std::move(diagonal)) {
        for (double value : diagonal_) {
            operator_norm_ = std::max(operator_norm_, std::abs(value));
        }
    }

    int rows() const override {
        return static_cast<int>(diagonal_.size());
    }

    int cols() const override {
        return static_cast<int>(diagonal_.size());
    }

    void apply(
        const double* x,
        int block_cols,
        double* y) const override {
        const int n = cols();
        for (int column = 0; column < block_cols; ++column) {
            for (int row = 0; row < n; ++row) {
                y[row + column * n] =
                    diagonal_[row] * x[row + column * n];
            }
        }
    }

    void apply_transpose(
        const double* y,
        int block_cols,
        double* x) const override {
        apply(y, block_cols, x);
    }

    amfls::MatrixOperatorValidationErrorModel validation_error_model()
        const noexcept override {
        return {operator_norm_, operator_norm_, 1, 1};
    }

private:
    std::vector<double> diagonal_;
    double operator_norm_ = 0.0;
};

int main() {
    DiagonalOperator A({100.0, 3.0, 2.0, 1.0});
    const std::vector<double> b{100.0, 6.0, 6.0, 4.0};

    amfls::AmflsOptions options;
    options.tolerance = 1e-10;
    options.seed = 7;

    const amfls::LeastSquaresResult result =
        amfls::solve_amfls(A, b.data(), options);

    if (result.status != amfls::SolverStatus::success) {
        std::cerr << "AMFLS stopped before reaching the tolerance.\n";
        return 1;
    }

    std::cout << "backward error bound: "
              << result.backward_error_upper_bound << '\n';
    std::cout << "basis rank: " << result.basis_rank << '\n';
    for (double value : result.solution) {
        std::cout << value << ' ';
    }
    std::cout << '\n';
}
```

### 5.2 Dense PyTorch example

The Python interface accepts two-dimensional CPU matrices and one-dimensional
CPU data vectors with `torch.float64` dtype. Dense matrices may be contiguous
or strided.

```python
import torch

from amfls import load_library, run_amfls

load_library("build-torch/amfls_torch.so")

torch.manual_seed(7)
A = torch.randn(1200, 200, dtype=torch.float64)
x_true = torch.randn(200, dtype=torch.float64)
b = A @ x_true

result = run_amfls(
    A,
    b,
    tolerance=1e-10,
    seed=2026,
)

if not result.success:
    raise RuntimeError(
        f"AMFLS status={result.status}, stop_reason={result.stop_reason}"
    )

relative_solution_error = (
    torch.linalg.vector_norm(result.solution - x_true)
    / torch.linalg.vector_norm(x_true)
)

print(f"relative solution error = {relative_solution_error:.3e}")
print(f"backward error bound    = {result.backward_error_upper_bound:.3e}")
print(f"selected width          = {result.auxiliary_width + 1}")
print(f"completed depth         = {result.depth}")
print(f"total time              = {result.total_seconds:.6f} s")
```

### 5.3 Regularized PyTorch example

Set `regularization` to a positive value for Tikhonov regularization.

```python
regularized = run_amfls(
    A,
    b,
    regularization=1e-4,
    tolerance=1e-10,
    seed=2026,
)

if regularized.success:
    print(
        "relative energy error bound = "
        f"{regularized.relative_energy_error_upper_bound:.3e}"
    )
```

### 5.4 Sparse CSR input

AMFLS and all matrix-operator baselines accept CPU CSR tensors. APLICUR uses
an explicit dense matrix because its algorithm samples individual entries,
rows, and columns.

```python
A_csr = A.to_sparse_csr()

sparse_result = run_amfls(
    A_csr,
    b,
    tolerance=1e-10,
    seed=2026,
)

print(sparse_result.success)
```

### 5.5 Baseline calls in PyTorch

```python
from amfls import run_fixed_rbgk, run_lsqr, run_lsrn

fixed = run_fixed_rbgk(
    A,
    b,
    auxiliary_width=8,
    maximum_depth=200,
    tolerance=1e-10,
    seed=2026,
)

lsqr = run_lsqr(
    A,
    b,
    tolerance=1e-10,
    maximum_iterations=200,
)

lsrn = run_lsrn(
    A,
    b,
    tolerance=1e-10,
    oversampling=2.0,
    seed=2026,
)

print(fixed.success, lsqr.success, lsrn.success)
```

All wrappers return `SolveResult`, so accuracy, work, and timing fields have
the same names across methods.

## 6. Algorithm structure and pseudocode

The repository contains one AMFLS algorithm. `solve_amfls` and `run_amfls`
call the same adaptive C++ core. The other public functions implement the
comparison methods.

### 6.1 AMFLS

```text
Input: dimensions m and n, data b, regularization lambda, tolerance tau,
       functions that return Av and A*u, and resource limits
Output: a least-squares vector x and its computed error bound

1. Normalize b to form the first basis vector in R^m.
2. Use A* on the current block in R^m to obtain new candidate vectors in R^n.
3. Orthogonalize these directions against the retained basis V.
4. Use A on the retained vectors in R^n to update the basis in R^m and the
   small projected matrix T = U* A V.
5. Compute the projected ordinary or regularized least-squares solution and
   form x = V y.
6. Recompute Ax-b and A*(Ax-b) + lambda x for this candidate.
7. Return x when its computed error bound is at most tau.
8. Otherwise measure the error decrease and time taken by the recent levels.
9. Compare the predicted remaining cost at the current width with the cost
   at the next larger width.
10. Continue at the current width or add random vectors in R^m to the same
    process according to this comparison.
11. Repeat from Step 2 until the tolerance or a resource limit is reached.
```

The projected matrix is small. Its columns describe the action of $A$ on the
current basis $V$, and its rows describe the corresponding basis in
$\mathbb{R}^m$. The full matrix $A$ is never needed by this procedure.

### 6.2 Width selection

AMFLS starts at width one. A larger width is considered after the method has
observed enough same-width progress to estimate the recent rate. The
controller compares the two choices over the same finite increase in basis
rank.

A width increase is accepted when all three comparisons favor it.

1. The larger width reduces the predicted number of sequential levels.
2. The operator's block-cost model predicts less total work for the products.
3. The measured operator and local computation times predict a lower total
   cost.

An accepted increase adds fresh Gaussian directions and keeps the existing
basis, projected matrix, and candidate history. Each accepted change doubles
the width unless a configured limit allows only a smaller increase.

### 6.3 Comparison methods

The baseline implementations share the matrix-operator interface, result
type, and final accuracy test. Their principal computations are as follows.

- Fixed RBGK follows the same block Golub--Kahan construction with one fixed
  auxiliary width.
- LSQR, LSMR, and LSMB advance their scalar recurrences by one direction at
  each iteration.
- LSRN forms a Gaussian sketch, builds a right preconditioner, and applies an
  iterative least-squares method to the preconditioned problem.
- Sparse-embedding LSQR, SPIR, and FOSSILS form sparse sketches before their
  iterative or refinement stages.
- Randomized block CG constructs a block Krylov space for
  $(A^*A+\lambda I)x=A^*b$.
- APLICUR samples rows and columns from an explicit dense matrix and uses the
  resulting CUR approximation as a preconditioner.

Section 2.1 lists the mathematical scope of each comparison method.

## 7. Mathematical principles

### 7.1 Projected least squares

After a completed level, AMFLS has orthonormal bases $U$ and $V$ and a small
matrix $T$ satisfying

$$
AV=UT.
$$

It computes

$$
y=\underset{y}{\arg\min}\;
\|Ty-U^*b\|_2^2+\lambda\|y\|_2^2,
\qquad x=Vy.
$$

Therefore $x$ minimizes the original objective over the current space
$\mathrm{range}(V)$. These spaces are nested, so the energy error
decreases as AMFLS completes more levels. For ordinary least squares,
$\mathrm{range}(V)\subseteq\mathrm{range}(A^*)$, which gives the
minimum-norm solution when stationarity is reached.

### 7.2 Random block expansion

The scalar Golub--Kahan process builds a Krylov space from $A^*b$. When the
controller selects a larger width, AMFLS adds a Gaussian block $\Omega$ and
extends the space with directions generated from $A^*\Omega$. With high
probability, these directions capture several leading singular-vector
components together. This can replace many dependent one-direction levels by
fewer block levels.

The convergence bound holds simultaneously for the finitely many expansions
that the adaptive controller can select. The block width may therefore depend
on the observed progress and timing without changing the stated probability
guarantee.

### 7.3 Computable stopping tests

For a candidate $x$, define

$$
r=Ax-b,
\qquad
g_\lambda=A^*r+\lambda x.
$$

For ordinary least squares, AMFLS combines a compatible-system residual test
with a least-squares stationarity test. Their minimum is an upper bound for
the normwise backward error.

For positive regularization, $g_\lambda$ provides an upper bound for the
distance to $x_\lambda$ in the energy norm. The implementation may tighten
this bound with one correction along the gradient. Dividing by a lower bound
for $\|x_\lambda\|_{A^*A+\lambda I}$ gives the reported relative error bound.

The stopping test is evaluated on the original operator at every candidate.
Thus `status == success` means that the returned error bound is no larger
than the requested tolerance.

### 7.4 Work bound

Suppose completed level $j$ processes an active block of width $a_j$, adds
$p_j$ pending random directions, and retains $q_j$ new directions. It sends
$a_j+p_j$ vectors through the calculation for $A^*u$ and $q_j$ vectors
through the calculation for $Av$. Hence the search work over $L$ levels is

$$
\sum_{j=1}^{L}(a_j+p_j+q_j)
$$

vector inputs and at most $2L$ block requests. A larger width can reduce the
number of dependent requests while increasing the number of vectors within a
request. The AMFLS controller measures both effects before changing the
width.

Complete theorem statements and proofs accompany the paper.

## 8. Why AMFLS is fast and accurate

### Reliable accuracy

AMFLS evaluates every candidate on the original least-squares problem. The
same computation supplies the error bound used for stopping and the progress
measurement used by the width controller. Ordinary problems return a
minimum-norm candidate, while regularized problems return a candidate with a
relative energy-error bound.

The paper reports 1,700 dense and sparse problem instances. They cover
square, tall, wide, rank-deficient, ill-conditioned, and regularized
matrices. AMFLS keeps the reported error below $10^{-10}$ on every instance.
The fixed-width method fails to reach this accuracy on 11.8% of the complete
test set, and the scalar Krylov methods fail on 17.6%.

### Faster execution

AMFLS selects its width from the current computation. Easy problems remain
narrow and avoid unnecessary basis work. Problems that benefit from a block
add several directions together and reduce the number of dependent levels.
The decision includes both the time used to obtain $Av$ and $A^*u$ and the
time used by orthogonalization and the projected problem.

The reported running times include every method's setup.

| Comparison | Reported result |
|---|---:|
| AMFLS relative to fixed RBGK | Lower average running time in **93.3%** of the cases in which both methods reach the requested accuracy and **28.4%** lower time on average |
| AMFLS relative to LSQR, LSMR, LSMB, and LSRN | Lower average running time in **64.3% to 93.8%** of the cases in which both methods reach the requested accuracy, with average reductions of **18.4% to 63.1%** |
| AMFLS relative to sparse-embedding LSQR, FOSSILS, and APLICUR | Lower average running time in **88.9% to 100%** of the cases in which both methods reach the requested accuracy, with average reductions of **32.4% to 82.0%** |
| AMFLS relative to randomized block CG | Lower average running time in both regularized cases and **37.3%** lower time on average |

### Moderate memory use

AMFLS retains one adaptive basis and its small projected problem. In the
reported experiments, it uses 22.6% less additional resident memory than
fixed RBGK and 61.1% to 93.9% less than the randomized block, sketching, and
preconditioning methods included in the memory comparison.

## 9. Paper, citation, author, and license

### Paper

**Adaptive Matrix-Free Least Squares Solver**<br>
Yang Tian

If AMFLS supports your research, please cite the paper.

```bibtex
@misc{Tian2026AMFLS,
  author = {Tian, Yang},
  title  = {Adaptive Matrix-Free Least Squares Solver},
  year   = {2026}
}
```

The software can be cited separately.

```bibtex
@misc{Tian2026AMFLSSoftware,
  author       = {Tian, Yang},
  title        = {Adaptive Matrix-Free Least Squares Solver},
  year         = {2026},
  howpublished = {GitHub repository},
  url          = {https://github.com/doloMing/amfls_solver}
}
```

If the algorithms or implementation are useful to you, please **star this
repository** and cite the paper. A GitHub star helps other researchers find
the project, and a citation records its use in scientific work.

### Author

**Yang Tian**<br>
Infplane Computing Technologies Ltd<br>
[tyanyang04@gmail.com](mailto:tyanyang04@gmail.com) &
[yang.tian@infplane.com](mailto:yang.tian@infplane.com)

### License

AMFLS is released under the [MIT License](LICENSE).
