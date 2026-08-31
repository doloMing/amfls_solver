"""Independent formula-level differential audit for the LSRN binding.

The reference below deliberately does not import ``amfls`` or call any core
random/SVD/Krylov helper.  It reconstructs the LSRN sketches and the zero-start
LSQR recurrence directly from the published tall and wide formulas, using
Torch only for dense linear algebra.  The counter-addressed normal stream is
also rebuilt here from its public seed/stream/index contract.
"""

from __future__ import annotations

import dataclasses
import decimal
import math
import pathlib
import sys
from typing import Callable

import torch


torch.set_num_threads(1)
torch.set_num_interop_threads(1)
torch.use_deterministic_algorithms(True)

library = pathlib.Path(sys.argv[1]).resolve()
torch.ops.load_library(str(library))

DTYPE = torch.float64
MASK64 = (1 << 64) - 1
STATUS_SUCCESS = 0
STATUS_WORK_LIMIT = 1
REASON_EXHAUSTED = 2
REASON_MAXIMUM_DEPTH = 3
REASON_COMPATIBLE = 8
REASON_LEAST_SQUARES = 9
REASON_RELATIVE_ENERGY = 10
DOUBLE_EPSILON = 2.0**-52
DOUBLE_DENORM_MIN = 2.0**-1074


def decimal_value(value: float) -> decimal.Decimal:
    return decimal.Decimal.from_float(value)


def upward_from_decimal(value: decimal.Decimal) -> float:
    if not value.is_finite() or value < 0:
        return math.inf
    if value == 0:
        return 0.0
    rounded = float(value)
    if not math.isfinite(rounded):
        return math.inf
    if decimal_value(rounded) < value:
        rounded = math.nextafter(rounded, math.inf)
    return math.nextafter(rounded, math.inf)


def downward_from_decimal(value: decimal.Decimal) -> float:
    if not value.is_finite() or value <= 0:
        return 0.0
    if value > decimal_value(sys.float_info.max):
        return sys.float_info.max
    rounded = float(value)
    if decimal_value(rounded) > value:
        rounded = math.nextafter(rounded, 0.0)
    return math.nextafter(rounded, 0.0)


def upward_add(left: float, right: float) -> float:
    if left < 0.0 or right < 0.0:
        return math.inf
    return upward_from_decimal(decimal_value(left) + decimal_value(right))


def upward_multiply(left: float, right: float) -> float:
    if (
        left < 0.0
        or right < 0.0
        or not math.isfinite(left)
        or not math.isfinite(right)
    ):
        return math.inf
    return upward_from_decimal(decimal_value(left) * decimal_value(right))


def downward_add(left: float, right: float) -> float:
    if (
        left < 0.0
        or right < 0.0
        or not math.isfinite(left)
        or not math.isfinite(right)
    ):
        return 0.0
    return downward_from_decimal(decimal_value(left) + decimal_value(right))


def downward_multiply(left: float, right: float) -> float:
    if (
        left < 0.0
        or right < 0.0
        or not math.isfinite(left)
        or not math.isfinite(right)
    ):
        return 0.0
    return downward_from_decimal(decimal_value(left) * decimal_value(right))


def upward_divide(numerator: float, denominator: float) -> float:
    if numerator == 0.0:
        return 0.0
    if (
        numerator < 0.0
        or denominator <= 0.0
        or not math.isfinite(numerator)
        or not math.isfinite(denominator)
    ):
        return math.inf
    return upward_from_decimal(
        decimal_value(numerator) / decimal_value(denominator)
    )


def downward_divide(numerator: float, denominator: float) -> float:
    if (
        numerator <= 0.0
        or denominator <= 0.0
        or not math.isfinite(numerator)
        or not math.isfinite(denominator)
    ):
        return 0.0
    return downward_from_decimal(
        decimal_value(numerator) / decimal_value(denominator)
    )


def downward_difference(left: float, right: float) -> float:
    if (
        left <= right
        or right < 0.0
        or not math.isfinite(left)
        or not math.isfinite(right)
    ):
        return 0.0
    return downward_from_decimal(decimal_value(left) - decimal_value(right))


def upward_sqrt(value: float) -> float:
    if value < 0.0 or not math.isfinite(value):
        return value if value == math.inf else math.inf
    with decimal.localcontext() as context:
        context.prec = 200
        root = context.sqrt(decimal_value(value))
    return upward_from_decimal(root)


def downward_sqrt(value: float) -> float:
    if value <= 0.0 or not math.isfinite(value):
        return 0.0
    with decimal.localcontext() as context:
        context.prec = 200
        root = context.sqrt(decimal_value(value))
    return downward_from_decimal(root)


def norm_interval(values: torch.Tensor) -> tuple[float, float]:
    flattened = [float(value) for value in values.reshape(-1)]
    if any(not math.isfinite(value) for value in flattened):
        return 0.0, math.inf
    with decimal.localcontext() as context:
        context.prec = 200
        squared = sum(
            (decimal_value(value) * decimal_value(value)
             for value in flattened),
            decimal.Decimal(0),
        )
        root = context.sqrt(squared)
    return downward_from_decimal(root), upward_from_decimal(root)


def binary64_gamma(operations: int) -> float:
    product = decimal_value(float(operations)) * decimal_value(
        0.5 * DOUBLE_EPSILON
    )
    if product >= 1:
        return math.inf
    return upward_from_decimal(product / (decimal.Decimal(1) - product))


def positive_sum_upper_bound(stored_sum: float, operations: int) -> float:
    if stored_sum < 0.0 or not math.isfinite(stored_sum) or operations < 0:
        return math.inf
    if stored_sum == 0.0:
        return 0.0
    denominator = downward_difference(
        1.0, binary64_gamma(operations)
    )
    if denominator <= 0.0:
        return math.inf
    absolute_roundoff = upward_from_decimal(
        decimal.Decimal(operations) * decimal_value(DOUBLE_DENORM_MIN)
    )
    return upward_divide(
        upward_add(stored_sum, absolute_roundoff), denominator
    )


def positive_sum_lower_bound(stored_sum: float, operations: int) -> float:
    if stored_sum <= 0.0 or not math.isfinite(stored_sum) or operations < 0:
        return 0.0
    absolute_roundoff = upward_from_decimal(
        decimal.Decimal(operations) * decimal_value(DOUBLE_DENORM_MIN)
    )
    numerator = downward_difference(stored_sum, absolute_roundoff)
    denominator = upward_add(1.0, binary64_gamma(operations))
    return downward_divide(numerator, denominator)


def binary64_fma_square(value: float, accumulated: float) -> float:
    # Decimal evaluates the exact product-plus-sum; conversion back to float
    # supplies the single binary64 rounding of std::fma.
    return float(
        decimal_value(value) * decimal_value(value) +
        decimal_value(accumulated)
    )


def underflow_radius(operations: int, output_size: int) -> float:
    if operations <= 0 or output_size <= 0:
        return 0.0
    with decimal.localcontext() as context:
        context.prec = 200
        value = (
            decimal.Decimal(operations)
            * context.sqrt(decimal.Decimal(output_size))
            * decimal_value(DOUBLE_DENORM_MIN)
        )
    return upward_from_decimal(value)


def splitmix64(value: int) -> int:
    """Pure-Python unsigned-64 SplitMix permutation."""

    value = (value + 0x9E3779B97F4A7C15) & MASK64
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & MASK64
    return (value ^ (value >> 31)) & MASK64


def open_uniform(bits: int) -> float:
    return float((bits >> 11) + 1) / 9007199254740992.0


def counter_gaussian(
    rows: int, cols: int, seed: int, stream: int
) -> torch.Tensor:
    """Rebuild columns by global counter, independently of any block split."""

    key = splitmix64(seed) ^ splitmix64(
        (stream + 0xD2B74407B1CE6E93) & MASK64
    )
    values = torch.empty((rows, cols), dtype=DTYPE)
    for col in range(cols):
        for row in range(rows):
            index = col * rows + row
            u1 = open_uniform(splitmix64(key ^ ((2 * index) & MASK64)))
            u2 = open_uniform(
                splitmix64(key ^ ((2 * index + 1) & MASK64))
            )
            values[row, col] = math.sqrt(-2.0 * math.log(u1)) * math.cos(
                2.0 * math.pi * u2
            )
    return values


def vector_norm(values: torch.Tensor) -> float:
    return float(torch.linalg.vector_norm(values))


def update_lower_bound(
    lower_bound: float, input_vector: torch.Tensor, output_vector: torch.Tensor
) -> float:
    input_norm = vector_norm(input_vector)
    output_norm = vector_norm(output_vector)
    if input_norm > 0.0 and math.isfinite(input_norm) and math.isfinite(
        output_norm
    ):
        ratio = output_norm / input_norm
        if math.isfinite(ratio):
            return max(lower_bound, ratio)
    return lower_bound


def retained_rank(
    singular_values: torch.Tensor,
    sketch_rows: int,
    sketch_cols: int,
    relative_tolerance: float,
    absolute_tolerance: float,
) -> int:
    if singular_values.numel() == 0:
        return 0
    sigma_max = float(singular_values[0])
    if not (sigma_max > 0.0) or not math.isfinite(sigma_max):
        return 0
    if relative_tolerance < 0.0:
        relative_tolerance = (
            16.0
            * torch.finfo(DTYPE).eps
            * max(sketch_rows, sketch_cols)
        )
    cutoff = max(relative_tolerance * sigma_max, absolute_tolerance)
    rank = 0
    for singular_value in singular_values:
        value = float(singular_value)
        # The strict inequality is intentional and separately exercised with
        # relative_tolerance == 1, where cutoff is exactly sigma_max.
        if not (
            value > cutoff
            and math.isfinite(value)
            and math.isfinite(1.0 / value)
        ):
            break
        rank += 1
    return rank


@dataclasses.dataclass
class ReferenceSystem:
    matrix: torch.Tensor
    right_hand_side: torch.Tensor
    regularization: float
    tall: bool
    factor: torch.Tensor
    rank: int
    sketch_size: int
    random_rows: int
    sketch_lower_bound: float

    @property
    def rows(self) -> int:
        m, n = self.matrix.shape
        if self.tall:
            return m + n if self.regularization > 0.0 else m
        return self.rank

    @property
    def cols(self) -> int:
        m, n = self.matrix.shape
        if self.tall:
            return self.rank
        return n + m if self.regularization > 0.0 else n

    def transformed_rhs(self) -> torch.Tensor:
        if self.tall:
            if self.regularization == 0.0:
                return self.right_hand_side.clone()
            return torch.cat(
                (
                    self.right_hand_side,
                    torch.zeros(self.matrix.shape[1], dtype=DTYPE),
                )
            )
        return self.factor.T @ self.right_hand_side

    def original_candidate(self, inner_solution: torch.Tensor) -> torch.Tensor:
        if self.tall:
            return self.factor @ inner_solution
        return inner_solution[: self.matrix.shape[1]].clone()

    def make_operators(
        self, lower_bound: list[float]
    ) -> tuple[
        Callable[[torch.Tensor], torch.Tensor],
        Callable[[torch.Tensor], torch.Tensor],
    ]:
        matrix = self.matrix
        m, n = matrix.shape
        root = math.sqrt(self.regularization)

        if self.tall:

            def apply(direction: torch.Tensor) -> torch.Tensor:
                original_direction = self.factor @ direction
                product = matrix @ original_direction
                lower_bound[0] = update_lower_bound(
                    lower_bound[0], original_direction, product
                )
                if self.regularization > 0.0:
                    return torch.cat((product, root * original_direction))
                return product

            def apply_transpose(direction: torch.Tensor) -> torch.Tensor:
                original_transpose = matrix.T @ direction[:m]
                lower_bound[0] = update_lower_bound(
                    lower_bound[0], direction[:m], original_transpose
                )
                if self.regularization > 0.0:
                    original_transpose = (
                        original_transpose + root * direction[m : m + n]
                    )
                return self.factor.T @ original_transpose

            return apply, apply_transpose

        def apply(direction: torch.Tensor) -> torch.Tensor:
            original_product = matrix @ direction[:n]
            lower_bound[0] = update_lower_bound(
                lower_bound[0], direction[:n], original_product
            )
            if self.regularization > 0.0:
                original_product = original_product + root * direction[n:]
            return self.factor.T @ original_product

        def apply_transpose(direction: torch.Tensor) -> torch.Tensor:
            left_direction = self.factor @ direction
            original_transpose = matrix.T @ left_direction
            lower_bound[0] = update_lower_bound(
                lower_bound[0], left_direction, original_transpose
            )
            if self.regularization > 0.0:
                return torch.cat((original_transpose, root * left_direction))
            return original_transpose

        return apply, apply_transpose


def build_reference_system(
    matrix: torch.Tensor,
    right_hand_side: torch.Tensor,
    regularization: float,
    oversampling: float,
    relative_rank_tolerance: float,
    absolute_rank_tolerance: float,
    seed: int,
    stream: int,
) -> ReferenceSystem:
    m, n = matrix.shape
    tall = m >= n
    dimension = n if tall else m
    sketch_size = max(math.ceil(oversampling * dimension), dimension + 1)
    random_rows = (
        (m + n if tall else n + m)
        if regularization > 0.0
        else (m if tall else n)
    )
    gaussian = counter_gaussian(random_rows, sketch_size, seed, stream)
    lower_bound = 0.0
    root = math.sqrt(regularization)

    if tall:
        random_top = gaussian[:m, :]
        sketch = matrix.T @ random_top
        for col in range(sketch_size):
            lower_bound = update_lower_bound(
                lower_bound, random_top[:, col], sketch[:, col]
            )
        if regularization > 0.0:
            sketch = sketch + root * gaussian[m : m + n, :]
        left_vectors, singular_values, _ = torch.linalg.svd(
            sketch, full_matrices=False
        )
        rank = retained_rank(
            singular_values,
            sketch_size,
            n,
            relative_rank_tolerance,
            absolute_rank_tolerance,
        )
        factor = left_vectors[:, :rank] / singular_values[:rank]
    else:
        random_top = gaussian[:n, :]
        sketch = matrix @ random_top
        for col in range(sketch_size):
            lower_bound = update_lower_bound(
                lower_bound, random_top[:, col], sketch[:, col]
            )
        if regularization > 0.0:
            sketch = sketch + root * gaussian[n : n + m, :]
        left_vectors, singular_values, _ = torch.linalg.svd(
            sketch, full_matrices=False
        )
        rank = retained_rank(
            singular_values,
            m,
            sketch_size,
            relative_rank_tolerance,
            absolute_rank_tolerance,
        )
        factor = left_vectors[:, :rank] / singular_values[:rank]

    return ReferenceSystem(
        matrix=matrix,
        right_hand_side=right_hand_side,
        regularization=regularization,
        tall=tall,
        factor=factor,
        rank=rank,
        sketch_size=sketch_size,
        random_rows=random_rows,
        sketch_lower_bound=lower_bound,
    )


@dataclasses.dataclass
class ReferenceCheckpoint:
    solution: torch.Tensor
    metrics: torch.Tensor
    status: int
    reason: int
    correction_attempted: bool


@dataclasses.dataclass
class ValidationEnvelope:
    model_upper: float
    model_lower: float
    apply_steps: int
    transpose_steps: int
    operator_lower: float
    residual_lower: float
    residual_upper: float
    gradient_upper: float
    solution_lower: float
    solution_upper: float
    right_hand_side_lower: float
    solution_energy_lower: float
    gradient_error: float


def matrix_error_model(
    matrix: torch.Tensor,
) -> tuple[float, float, int, int]:
    rows, cols = matrix.shape
    column_sums = [0.0] * cols
    column_square_sums = [0.0] * cols
    maximum_row_sum_upper = 0.0
    maximum_two_norm_lower = 0.0
    maximum_entry = 0.0
    for row in range(rows):
        row_sum = 0.0
        row_square_sum = 0.0
        for col in range(cols):
            magnitude = abs(float(matrix[row, col]))
            maximum_entry = max(maximum_entry, magnitude)
            row_sum += magnitude
            column_sums[col] += magnitude
            row_square_sum = binary64_fma_square(
                magnitude, row_square_sum
            )
            column_square_sums[col] = binary64_fma_square(
                magnitude, column_square_sums[col]
            )
        maximum_row_sum_upper = max(
            maximum_row_sum_upper,
            positive_sum_upper_bound(row_sum, cols),
        )
        maximum_two_norm_lower = max(
            maximum_two_norm_lower,
            downward_sqrt(
                positive_sum_lower_bound(row_square_sum, cols)
            ),
        )
    maximum_column_sum_upper = max(
        positive_sum_upper_bound(column_sum, rows)
        for column_sum in column_sums
    )
    for column_square_sum in column_square_sums:
        maximum_two_norm_lower = max(
            maximum_two_norm_lower,
            downward_sqrt(
                positive_sum_lower_bound(column_square_sum, rows)
            ),
        )
    upper = upward_multiply(
        upward_sqrt(maximum_column_sum_upper),
        upward_sqrt(maximum_row_sum_upper),
    )
    return (
        upper,
        max(maximum_entry, maximum_two_norm_lower),
        2 * cols + 2,
        2 * rows + 2,
    )


def callback_error_radius(
    model_upper: float,
    operations: int,
    input_norm_upper: float,
    output_size: int,
) -> float:
    if not math.isfinite(model_upper) or not math.isfinite(input_norm_upper):
        return math.inf
    if model_upper == 0.0 or input_norm_upper == 0.0:
        return 0.0
    main = upward_multiply(
        binary64_gamma(operations),
        upward_multiply(model_upper, input_norm_upper),
    )
    return upward_add(main, underflow_radius(operations, output_size))


def one_operation_radius(
    stored_result_norm_upper: float,
    can_underflow: bool,
    output_size: int,
) -> float:
    radius = upward_multiply(
        binary64_gamma(2), stored_result_norm_upper
    )
    if can_underflow:
        radius = upward_add(radius, underflow_radius(1, output_size))
    return radius


def enlarge_norm_interval(
    interval: tuple[float, float], radius: float
) -> tuple[float, float]:
    return (
        downward_difference(interval[0], radius),
        upward_add(interval[1], radius),
    )


def augmented_norm_interval(
    primary: tuple[float, float],
    secondary: tuple[float, float],
    regularization: float,
) -> tuple[float, float]:
    lower_squared = downward_add(
        downward_multiply(primary[0], primary[0]),
        downward_multiply(
            regularization,
            downward_multiply(secondary[0], secondary[0]),
        ),
    )
    upper_squared = upward_add(
        upward_multiply(primary[1], primary[1]),
        upward_multiply(
            regularization,
            upward_multiply(secondary[1], secondary[1]),
        ),
    )
    return downward_sqrt(lower_squared), upward_sqrt(upper_squared)


def build_validation_envelope(
    system: ReferenceSystem,
    solution: torch.Tensor,
    applied_solution: torch.Tensor,
    residual: torch.Tensor,
    transposed_residual: torch.Tensor,
    gradient: torch.Tensor,
) -> ValidationEnvelope:
    matrix = system.matrix
    regularization = system.regularization
    rows, cols = matrix.shape
    model_upper, model_lower, apply_steps, transpose_steps = (
        matrix_error_model(matrix)
    )
    b_interval = norm_interval(system.right_hand_side)
    x_interval = norm_interval(solution)
    applied_interval = norm_interval(applied_solution)
    stored_residual_interval = norm_interval(residual)
    transpose_interval = norm_interval(transposed_residual)
    stored_gradient_interval = norm_interval(gradient)

    apply_error = callback_error_radius(
        model_upper, apply_steps, x_interval[1], rows
    )
    subtraction_error = one_operation_radius(
        stored_residual_interval[1],
        applied_interval[1] > 0.0 or b_interval[1] > 0.0,
        rows,
    )
    residual_error = upward_add(apply_error, subtraction_error)
    true_residual = enlarge_norm_interval(
        stored_residual_interval, residual_error
    )

    transpose_error = callback_error_radius(
        model_upper,
        transpose_steps,
        stored_residual_interval[1],
        cols,
    )
    ridge_sum_error = one_operation_radius(
        stored_gradient_interval[1],
        regularization > 0.0 and x_interval[1] > 0.0,
        cols,
    )
    gradient_error = upward_add(
        upward_add(
            transpose_error,
            upward_multiply(model_upper, residual_error),
        ),
        ridge_sum_error,
    )
    true_gradient = enlarge_norm_interval(
        stored_gradient_interval, gradient_error
    )
    true_applied = enlarge_norm_interval(applied_interval, apply_error)

    operator_lower = model_lower
    if x_interval[1] > 0.0:
        operator_lower = max(
            operator_lower,
            downward_divide(
                downward_difference(applied_interval[0], apply_error),
                x_interval[1],
            ),
        )
    if stored_residual_interval[1] > 0.0:
        operator_lower = max(
            operator_lower,
            downward_divide(
                downward_difference(
                    transpose_interval[0], transpose_error
                ),
                stored_residual_interval[1],
            ),
        )

    solution_energy = augmented_norm_interval(
        true_applied, x_interval, regularization
    )
    assert model_lower <= operator_lower <= model_upper
    return ValidationEnvelope(
        model_upper=model_upper,
        model_lower=model_lower,
        apply_steps=apply_steps,
        transpose_steps=transpose_steps,
        operator_lower=operator_lower,
        residual_lower=true_residual[0],
        residual_upper=true_residual[1],
        gradient_upper=true_gradient[1],
        solution_lower=x_interval[0],
        solution_upper=x_interval[1],
        right_hand_side_lower=b_interval[0],
        solution_energy_lower=solution_energy[0],
        gradient_error=gradient_error,
    )


def evaluate_contract(
    base_values: list[float],
    envelope: ValidationEnvelope,
    regularization: float,
    tolerance: float,
    ridge_bound: float | None = None,
) -> tuple[list[float], int, int]:
    values = list(base_values)
    augmented_residual = augmented_norm_interval(
        (envelope.residual_lower, envelope.residual_upper),
        (envelope.solution_lower, envelope.solution_upper),
        regularization,
    )
    augmented_operator_squared = downward_add(
        downward_multiply(
            envelope.operator_lower, envelope.operator_lower
        ),
        regularization,
    )
    augmented_operator = downward_sqrt(augmented_operator_squared)
    compatible_denominator = downward_add(
        downward_multiply(
            augmented_operator, envelope.solution_lower
        ),
        envelope.right_hand_side_lower,
    )
    compatible = upward_divide(
        augmented_residual[1], compatible_denominator
    )
    least_squares_denominator = downward_multiply(
        augmented_operator, augmented_residual[0]
    )
    least_squares = upward_divide(
        envelope.gradient_upper, least_squares_denominator
    )
    backward = min(compatible, least_squares)
    normal_denominator = downward_add(
        downward_multiply(
            downward_add(
                downward_multiply(
                    envelope.operator_lower, envelope.operator_lower
                ),
                regularization,
            ),
            envelope.solution_lower,
        ),
        downward_multiply(
            envelope.operator_lower, envelope.right_hand_side_lower
        ),
    )
    relative_normal = upward_divide(
        envelope.gradient_upper, normal_denominator
    )
    if regularization > 0.0:
        base_bound = upward_divide(
            envelope.gradient_upper, downward_sqrt(regularization)
        )
        energy_bound = base_bound if ridge_bound is None else ridge_bound
        objective_gap = upward_multiply(energy_bound, energy_bound)
        relative_energy = upward_divide(
            energy_bound,
            downward_difference(
                envelope.solution_energy_lower, energy_bound
            ),
        )
        passed = relative_energy <= tolerance
    elif envelope.gradient_upper == 0.0:
        energy_bound = 0.0
        objective_gap = 0.0
        relative_energy = 0.0
        passed = True
    else:
        energy_bound = math.inf
        objective_gap = math.inf
        relative_energy = math.inf
        passed = backward <= tolerance

    values[6] = envelope.operator_lower
    values[7] = augmented_operator
    values.extend(
        (
            compatible,
            least_squares,
            backward,
            relative_normal,
            energy_bound,
            relative_energy,
            objective_gap,
        )
    )
    passed = envelope.gradient_upper == 0.0 or passed
    status = STATUS_SUCCESS if passed else STATUS_WORK_LIMIT
    if envelope.gradient_upper == 0.0:
        reason = 1
    elif status == STATUS_SUCCESS and regularization > 0.0:
        reason = REASON_RELATIVE_ENERGY
    elif status == STATUS_SUCCESS and compatible <= least_squares:
        reason = REASON_COMPATIBLE
    elif status == STATUS_SUCCESS:
        reason = REASON_LEAST_SQUARES
    else:
        reason = REASON_MAXIMUM_DEPTH
    return values, status, reason


def validate_reference_candidate(
    system: ReferenceSystem,
    solution: torch.Tensor,
    search_operator_lower_bound: float,
    tolerance: float,
) -> ReferenceCheckpoint:
    # Search/sketch quotients have no fresh callback-error audit.  Production
    # retains this argument for compatibility but reconstructs the stopping
    # lower bound solely from the matrix model and fresh validation products.
    _ = search_operator_lower_bound
    matrix = system.matrix
    regularization = system.regularization
    applied_solution = matrix @ solution
    residual = applied_solution - system.right_hand_side
    transposed_residual = matrix.T @ residual
    gradient = transposed_residual + regularization * solution
    residual_norm = vector_norm(residual)
    normal_residual_norm = vector_norm(gradient)
    solution_norm = vector_norm(solution)
    solution_energy_norm = math.hypot(
        vector_norm(applied_solution),
        math.sqrt(regularization) * solution_norm,
    )
    objective = residual_norm * residual_norm + regularization * (
        solution_norm * solution_norm
    )
    base = [
        objective,
        residual_norm,
        math.hypot(
            residual_norm,
            math.sqrt(regularization) * solution_norm,
        ),
        normal_residual_norm,
        solution_norm,
        solution_energy_norm,
        0.0,
        0.0,
    ]
    envelope = build_validation_envelope(
        system,
        solution,
        applied_solution,
        residual,
        transposed_residual,
        gradient,
    )
    values, status, reason = evaluate_contract(
        base,
        envelope,
        regularization,
        tolerance,
    )

    correction_attempted = (
        regularization > 0.0
        and status != STATUS_SUCCESS
        and envelope.solution_energy_lower > 0.0
        and envelope.gradient_upper > 0.0
        and math.isfinite(envelope.gradient_upper)
        and normal_residual_norm > 0.0
        and math.isfinite(normal_residual_norm)
        and bool(torch.isfinite(solution).all())
        and bool(torch.isfinite(gradient).all())
    )
    if correction_attempted:
        direction = gradient / normal_residual_norm
        direction_interval = norm_interval(direction)
        applied_direction = matrix @ direction
        applied_direction_interval = norm_interval(applied_direction)
        applied_direction_error = callback_error_radius(
            envelope.model_upper,
            envelope.apply_steps,
            direction_interval[1],
            matrix.shape[0],
        )
        transposed_direction = matrix.T @ applied_direction
        hessian_direction = (
            transposed_direction + regularization * direction
        )
        transpose_direction_error = callback_error_radius(
            envelope.model_upper,
            envelope.transpose_steps,
            applied_direction_interval[1],
            matrix.shape[1],
        )
        hessian_interval = norm_interval(hessian_direction)
        hessian_sum_error = one_operation_radius(
            hessian_interval[1],
            regularization > 0.0 and direction_interval[1] > 0.0,
            matrix.shape[1],
        )
        hessian_error = upward_add(
            upward_add(
                transpose_direction_error,
                upward_multiply(
                    envelope.model_upper, applied_direction_error
                ),
            ),
            hessian_sum_error,
        )
        curvature = float(torch.dot(direction, hessian_direction))
        if curvature > 0.0 and math.isfinite(curvature):
            step = normal_residual_norm / curvature
            correction_residual = gradient - step * hessian_direction
            correction_residual_interval = norm_interval(
                correction_residual
            )
            correction_axpy_error = one_operation_radius(
                correction_residual_interval[1],
                normal_residual_norm > 0.0 and hessian_interval[1] > 0.0,
                matrix.shape[1],
            )
            correction_residual_error = upward_add(
                upward_add(
                    envelope.gradient_error,
                    upward_multiply(step, hessian_error),
                ),
                correction_axpy_error,
            )
            q_upper = upward_add(
                correction_residual_interval[1],
                correction_residual_error,
            )
            applied_direction_true_upper = upward_add(
                applied_direction_interval[1], applied_direction_error
            )
            step_squared = upward_multiply(step, step)
            direction_energy_squared = upward_add(
                upward_multiply(
                    applied_direction_true_upper,
                    applied_direction_true_upper,
                ),
                upward_multiply(
                    regularization,
                    upward_multiply(
                        direction_interval[1], direction_interval[1]
                    ),
                ),
            )
            z_h_z = upward_multiply(
                step_squared, direction_energy_squared
            )
            cross = upward_multiply(
                step,
                upward_multiply(direction_interval[1], q_upper),
            )
            corrected_squared = upward_add(
                z_h_z,
                upward_add(
                    upward_multiply(2.0, cross),
                    upward_divide(
                        upward_multiply(q_upper, q_upper),
                        regularization,
                    ),
                ),
            )
            corrected_bound = upward_sqrt(corrected_squared)
            if math.isfinite(corrected_bound) and corrected_bound < values[12]:
                values, status, reason = evaluate_contract(
                    base,
                    envelope,
                    regularization,
                    tolerance,
                    corrected_bound,
                )

    return ReferenceCheckpoint(
        solution=solution.clone(),
        metrics=torch.tensor(values, dtype=DTYPE),
        status=status,
        reason=reason,
        correction_attempted=correction_attempted,
    )


@dataclasses.dataclass
class ReferenceRun:
    system: ReferenceSystem
    checkpoints: list[ReferenceCheckpoint]
    iterations: int
    iterative_transpose_products: int

    @property
    def final(self) -> ReferenceCheckpoint:
        return self.checkpoints[-1]


def is_certificate_checkpoint(iteration: int, regularization: float) -> bool:
    if iteration <= 1:
        return True
    odd_part = iteration
    while odd_part % 2 == 0:
        odd_part //= 2
    return odd_part == 1 or (regularization == 0.0 and odd_part == 3)


def predicted_certificate_crossing(
    previous: float,
    current: float,
    previous_iteration: int,
    current_iteration: int,
    tolerance: float,
) -> int | None:
    if not (
        previous > current > tolerance > 0.0
        and math.isfinite(previous)
        and math.isfinite(current)
        and math.isfinite(tolerance)
    ):
        return None
    offset = math.ceil(
        (current_iteration - previous_iteration)
        * (math.log(current) - math.log(tolerance))
        / (math.log(previous) - math.log(current))
    )
    return current_iteration + offset if offset > 0 else None


def run_reference_lsrn(
    matrix: torch.Tensor,
    right_hand_side: torch.Tensor,
    *,
    regularization: float,
    tolerance: float,
    oversampling: float,
    relative_rank_tolerance: float,
    absolute_rank_tolerance: float,
    maximum_iterations: int,
    seed: int,
    stream: int,
) -> ReferenceRun:
    system = build_reference_system(
        matrix,
        right_hand_side,
        regularization,
        oversampling,
        relative_rank_tolerance,
        absolute_rank_tolerance,
        seed,
        stream,
    )
    lower_bound = [system.sketch_lower_bound]
    inner_solution = torch.zeros(system.cols, dtype=DTYPE)
    checkpoints: list[ReferenceCheckpoint] = []
    if system.rank == 0:
        checkpoint = validate_reference_candidate(
            system,
            system.original_candidate(inner_solution),
            lower_bound[0],
            tolerance,
        )
        if checkpoint.status != STATUS_SUCCESS:
            checkpoint.reason = REASON_EXHAUSTED
        return ReferenceRun(system, [checkpoint], 0, 0)

    apply, apply_transpose = system.make_operators(lower_bound)
    transformed_rhs = system.transformed_rhs()
    beta = vector_norm(transformed_rhs)
    assert beta > 0.0 and math.isfinite(beta)
    u = transformed_rhs / beta
    v = apply_transpose(u)
    alpha = vector_norm(v)
    assert alpha > 0.0 and math.isfinite(alpha)
    v = v / alpha
    iterative_transpose_products = 1
    w = v.clone()
    rhobar = alpha
    phibar = beta

    committed_iteration = 0
    predicted_iteration: int | None = None
    previous_checkpoint_iteration: int | None = None
    previous_checkpoint: ReferenceCheckpoint | None = None
    for iteration in range(1, maximum_iterations + 1):
        next_u = apply(v) - alpha * u
        beta = vector_norm(next_u)
        if beta > 0.0:
            next_u = next_u / beta

        rho = math.hypot(rhobar, beta)
        cosine = rhobar / rho
        sine = beta / rho
        phi = cosine * phibar
        phibar = sine * phibar
        update = phi / rho
        inner_solution = inner_solution + update * w
        u = next_u
        committed_iteration = iteration

        static_checkpoint = is_certificate_checkpoint(
            iteration, regularization
        )
        checkpoint_due = (
            static_checkpoint
            or (
                predicted_iteration is not None
                and iteration >= predicted_iteration
            )
            or beta == 0.0
            or iteration == maximum_iterations
        )
        if checkpoint_due:
            checkpoint = validate_reference_candidate(
                system,
                system.original_candidate(inner_solution),
                lower_bound[0],
                tolerance,
            )
            checkpoints.append(checkpoint)
            predicted_iteration = None
            if (
                static_checkpoint
                and previous_checkpoint_iteration is not None
                and previous_checkpoint is not None
                and iteration > previous_checkpoint_iteration
            ):
                if regularization > 0.0:
                    predicted_iteration = predicted_certificate_crossing(
                        float(previous_checkpoint.metrics[13]),
                        float(checkpoint.metrics[13]),
                        previous_checkpoint_iteration,
                        iteration,
                        tolerance,
                    )
                else:
                    branch_predictions = (
                        predicted_certificate_crossing(
                            float(previous_checkpoint.metrics[index]),
                            float(checkpoint.metrics[index]),
                            previous_checkpoint_iteration,
                            iteration,
                            tolerance,
                        )
                        for index in (8, 9)
                    )
                    valid_predictions = [
                        prediction
                        for prediction in branch_predictions
                        if prediction is not None
                    ]
                    if valid_predictions:
                        predicted_iteration = min(valid_predictions)
            previous_checkpoint_iteration = iteration
            previous_checkpoint = checkpoint
            if checkpoint.status == STATUS_SUCCESS:
                break
            if beta == 0.0:
                checkpoint.reason = REASON_EXHAUSTED
                break
            if iteration == maximum_iterations:
                break

        next_v = apply_transpose(u) - beta * v
        iterative_transpose_products += 1
        alpha = vector_norm(next_v)
        if alpha > 0.0:
            next_v = next_v / alpha
        if alpha == 0.0:
            checkpoint.reason = REASON_EXHAUSTED
            break
        theta = sine * alpha
        rhobar = -cosine * alpha
        recurrence = theta / rho
        w = next_v - recurrence * w
        v = next_v

    return ReferenceRun(
        system,
        checkpoints,
        committed_iteration,
        iterative_transpose_products,
    )


def expected_counters(
    run: ReferenceRun, sketch_block_size: int
) -> torch.Tensor:
    system = run.system
    k = run.iterations
    validation_columns = len(run.checkpoints) + sum(
        int(checkpoint.correction_attempted)
        for checkpoint in run.checkpoints
    )
    effective_block = (
        min(sketch_block_size, system.sketch_size)
        if sketch_block_size > 0
        else system.sketch_size
    )
    sketch_calls = math.ceil(system.sketch_size / effective_block)
    iterative_a = k
    iterative_at = run.iterative_transpose_products
    if system.tall:
        sketch_a = 0
        sketch_at = system.sketch_size
        sketch_a_calls = 0
        sketch_at_calls = sketch_calls
    else:
        sketch_a = system.sketch_size
        sketch_at = 0
        sketch_a_calls = sketch_calls
        sketch_at_calls = 0
    search_a = sketch_a + iterative_a
    search_at = sketch_at + iterative_at
    total_a = search_a + validation_columns
    total_at = search_at + validation_columns
    total_a_calls = sketch_a_calls + iterative_a + validation_columns
    total_at_calls = sketch_at_calls + iterative_at + validation_columns
    return torch.tensor(
        [
            total_a,
            total_at,
            total_a_calls,
            total_at_calls,
            search_a,
            search_at,
            validation_columns,
            validation_columns,
            k,
            k,
            system.sketch_size,
            system.rank,
            sketch_a,
            sketch_at,
            sketch_a_calls,
            sketch_at_calls,
            iterative_a,
            iterative_at,
            iterative_a,
            iterative_at,
            system.sketch_size,
            system.random_rows * system.sketch_size,
        ],
        dtype=torch.int64,
    )


@dataclasses.dataclass(frozen=True)
class Case:
    name: str
    matrix: torch.Tensor
    right_hand_side: torch.Tensor
    regularization: float
    oversampling: float
    seed: int
    stream: int
    caps: tuple[int, ...]
    relative_rank_tolerance: float = 1e-12
    absolute_rank_tolerance: float = 0.0


def run_core(case: Case, cap: int, sketch_block_size: int) -> tuple[object, ...]:
    return torch.ops.amfls.run_lsrn(
        case.matrix,
        case.right_hand_side,
        case.regularization,
        1e-12,
        case.oversampling,
        case.relative_rank_tolerance,
        case.absolute_rank_tolerance,
        cap,
        sketch_block_size,
        case.seed,
        case.stream,
    )


def matrix_operator_reference_view(matrix: torch.Tensor) -> torch.Tensor:
    # MatrixOperator exchanges column-major blocks with the C++ algorithms.
    # This logically identical view makes Torch use the corresponding BLAS
    # reduction orientation, so the differential audit compares formulas at
    # the same FP64 storage contract even at the convergence floor.
    reference = matrix.T.contiguous().T
    assert torch.equal(reference, matrix)
    return reference


def assert_metric_close(
    actual: torch.Tensor,
    expected: torch.Tensor,
    label: str,
    unstable_indices: tuple[int, ...] = (),
) -> None:
    assert actual.shape == expected.shape, label
    finite = torch.isfinite(expected)
    for index in unstable_indices:
        finite[index] = False
    assert torch.equal(torch.isinf(actual), torch.isinf(expected)), label
    if bool(finite.any()):
        torch.testing.assert_close(
            actual[finite],
            expected[finite],
            rtol=2e-7,
            atol=3e-11,
            msg=(
                f"{label}\nactual={actual.tolist()}\n"
                f"expected={expected.tolist()}"
            ),
        )


def residual_floor_metric_indices(
    metrics: torch.Tensor, least_squares_index: int
) -> tuple[int, ...]:
    # When a consistent problem has reached the FP64 residual floor, the
    # scale-free LS backward ratio ||A^T r||/(||A|| ||r||) has the indeterminate
    # exact limit 0/0.  Its numerator and denominator norms remain audited,
    # as do this ratio at all pre-floor checkpoints.
    return (least_squares_index,) if float(metrics[1]) < 1e-12 else ()


def audit_case(case: Case) -> None:
    for cap in case.caps:
        reference = run_reference_lsrn(
            matrix_operator_reference_view(case.matrix),
            case.right_hand_side,
            regularization=case.regularization,
            tolerance=1e-12,
            oversampling=case.oversampling,
            relative_rank_tolerance=case.relative_rank_tolerance,
            absolute_rank_tolerance=case.absolute_rank_tolerance,
            maximum_iterations=cap,
            seed=case.seed,
            stream=case.stream,
        )
        block_size = 2
        (
            solution,
            metrics,
            counters,
            _,
            trace_float,
            trace_int,
            status,
            reason,
        ) = run_core(case, cap, block_size)
        trace = trace_float[:, :18]
        label = f"{case.name}, cap={cap}"
        comparison_reference = reference
        core_iterations = int(counters[8])
        if core_iterations != reference.iterations:
            # At a first crossing within the backend rounding floor, the
            # independent Torch recurrence and the OpenBLAS production
            # recurrence can place their mathematically identical candidates
            # on opposite sides of the requested tolerance. Audit that event
            # explicitly: production must schedule exactly its predicted next
            # checkpoint, and the independent strict validator must certify
            # the returned extra candidate without omitting any metric.
            assert core_iterations == reference.iterations + 1, label
            assert reference.final.status == STATUS_SUCCESS, label
            assert trace_float.shape[0] == len(reference.checkpoints) + 1
            boundary = len(reference.checkpoints) - 1
            reference_backward = float(reference.final.metrics[10])
            core_boundary_backward = float(trace_float[boundary, 15])
            assert (
                reference_backward <= 1e-12 < core_boundary_backward
            ), label
            previous_iteration = int(trace_float[boundary - 1, 1])
            boundary_iteration = int(trace_float[boundary, 1])
            predictions = [
                predicted_certificate_crossing(
                    float(trace_float[boundary - 1, metric_index]),
                    float(trace_float[boundary, metric_index]),
                    previous_iteration,
                    boundary_iteration,
                    1e-12,
                )
                for metric_index in (13, 14)
            ]
            assert core_iterations == min(
                prediction
                for prediction in predictions
                if prediction is not None
            ), label
            extra_checkpoint = validate_reference_candidate(
                reference.system, solution, 0.0, 1e-12
            )
            comparison_reference = ReferenceRun(
                reference.system,
                reference.checkpoints + [extra_checkpoint],
                core_iterations,
                core_iterations,
            )

        assert int(status) == comparison_reference.final.status, label
        assert int(reason) == comparison_reference.final.reason, label
        torch.testing.assert_close(
            solution,
            reference.final.solution,
            rtol=3e-9,
            atol=5e-11,
            msg=label + " original-coordinate candidate",
        )
        assert_metric_close(
            metrics[:15],
            comparison_reference.final.metrics,
            label + " metrics",
            residual_floor_metric_indices(
                comparison_reference.final.metrics, 9
            ),
        )
        assert torch.equal(
            counters[:22],
            expected_counters(comparison_reference, block_size),
        ), label + " counters"
        assert trace_float.shape == (
            len(comparison_reference.checkpoints), 29
        ), label
        assert trace_int.shape == (
            len(comparison_reference.checkpoints), 4
        ), label
        assert bool(torch.all((trace_int[:, 2] == 0) |
                              (trace_int[:, 2] == 1))), label
        assert trace.shape == (
            len(comparison_reference.checkpoints), 18
        ), label

        # Trace records expose every validation checkpoint's original-problem
        # metrics even though only the final original x is returned.
        for index, checkpoint in enumerate(
            comparison_reference.checkpoints
        ):
            trace_metrics = trace[index, 8:]
            expected_trace_metrics = checkpoint.metrics[
                torch.tensor([0, 1, 2, 3, 6, 8, 9, 10, 11, 13])
            ]
            assert_metric_close(
                trace_metrics,
                expected_trace_metrics,
                label + f" trace checkpoint {index}",
                residual_floor_metric_indices(expected_trace_metrics, 6),
            )


tall = torch.tensor(
    [
        [4.0, 0.2, -0.3, 0.1],
        [0.1, 3.0, 0.4, -0.2],
        [0.5, -0.4, 2.5, 0.3],
        [-0.2, 0.1, 0.6, 1.8],
        [1.1, -0.7, 0.2, 0.5],
        [-0.8, 0.3, 0.9, -0.4],
        [0.4, 1.2, -0.5, 0.7],
    ],
    dtype=DTYPE,
)
tall_b = torch.tensor([1.0, -2.0, 0.5, 3.0, -1.0, 0.7, 2.2], dtype=DTYPE)
wide = torch.tensor(
    [
        [2.5, 0.2, -0.4, 1.0, 0.3, -0.7],
        [-0.3, 1.8, 0.5, -0.2, 1.1, 0.4],
        [0.6, -0.5, 2.2, 0.7, -0.8, 1.3],
    ],
    dtype=DTYPE,
)
wide_b = torch.tensor([1.3, -0.8, 2.1], dtype=DTYPE)

rank_tall = torch.tensor(
    [
        [1.0, 0.2],
        [0.3, 1.1],
        [-0.7, 0.5],
        [1.2, -0.4],
        [0.8, 0.9],
        [-0.2, 1.4],
    ],
    dtype=DTYPE,
) @ torch.tensor(
    [[1.0, -0.4, 0.8, 0.3], [0.2, 1.1, -0.5, 0.7]], dtype=DTYPE
)
rank_tall_b = torch.tensor([1.0, -0.3, 2.0, 0.4, -1.1, 0.8], dtype=DTYPE)
rank_wide = torch.tensor(
    [[1.0, 0.3], [-0.4, 1.2], [0.7, -0.2]], dtype=DTYPE
) @ torch.tensor(
    [
        [1.0, -0.5, 0.2, 0.8, -0.3, 0.6],
        [0.4, 1.1, -0.7, 0.3, 0.9, -0.2],
    ],
    dtype=DTYPE,
)
rank_wide_b = torch.tensor([0.7, -1.4, 2.0], dtype=DTYPE)

cases = [
    Case("tall ordinary seed 11", tall, tall_b, 0.0, 1.6, 11, 3, (1, 2, 3, 6)),
    Case(
        "tall ordinary second seed",
        tall,
        tall_b,
        0.0,
        2.25,
        0x123456789ABCDEF0,
        29,
        (1, 3, 6),
    ),
    Case("wide ordinary", wide, wide_b, 0.0, 1.75, 97, 5, (1, 2, 4)),
    Case("tall ridge", tall, tall_b, 0.35, 1.6, 991, 7, (1, 2, 3, 6)),
    Case("wide ridge", wide, wide_b, 0.8, 2.25, 1237, 11, (1, 2, 4)),
    Case(
        "rank-deficient tall",
        rank_tall,
        rank_tall_b,
        0.0,
        2.0,
        7919,
        13,
        (1, 2, 4),
    ),
    Case(
        "rank-deficient wide",
        rank_wide,
        rank_wide_b,
        0.0,
        2.0,
        104729,
        17,
        (1, 2, 4),
    ),
]

for differential_case in cases:
    audit_case(differential_case)


# A cutoff equal to sigma_max must discard sigma_max: retention is ``>`` and
# not ``>=``.  Choosing relative tolerance one makes this equality exact in
# each implementation without sharing an SVD result across the boundary.
strict_cutoff_case = Case(
    "strict rank cutoff",
    tall,
    tall_b,
    0.0,
    1.6,
    65537,
    19,
    (1,),
    relative_rank_tolerance=1.0,
)
audit_case(strict_cutoff_case)
strict_raw = run_core(strict_cutoff_case, 1, 2)
assert int(strict_raw[2][11]) == 0
assert int(strict_raw[6]) == STATUS_WORK_LIMIT
assert int(strict_raw[7]) == REASON_EXHAUSTED


def assert_block_partition_contract(case: Case, cap: int) -> None:
    single = run_core(case, cap, 0)
    scalar = run_core(case, cap, 1)
    # The addressed random values are bitwise identical.  Different DGEMM
    # widths may nevertheless change last-bit reduction order, so the public
    # numerical candidate contract is checked at a backend-scale tolerance.
    torch.testing.assert_close(single[0], scalar[0], rtol=2e-11, atol=2e-12)
    assert_metric_close(single[1], scalar[1], case.name + " block metrics")
    assert single[6:8] == scalar[6:8]
    assert single[4].shape == scalar[4].shape
    torch.testing.assert_close(
        single[4][:, :6], scalar[4][:, :6], rtol=0.0, atol=0.0
    )
    for row in range(single[4].shape[0]):
        assert_metric_close(
            single[4][row, 8:],
            scalar[4][row, 8:],
            case.name + f" block trace {row}",
            residual_floor_metric_indices(single[4][row, 8:], 6),
        )

    single_counts = single[2]
    scalar_counts = scalar[2]
    tall_case = case.matrix.shape[0] >= case.matrix.shape[1]
    sketch_call_index = 15 if tall_case else 14
    total_call_index = 3 if tall_case else 2
    search_call_index = 23 if tall_case else 22
    gaussian_request_index = 34
    assert int(single_counts[sketch_call_index]) == 1
    sketch_size = int(single_counts[20])
    assert int(scalar_counts[sketch_call_index]) == sketch_size
    for changed_index in (
        total_call_index,
        search_call_index,
        gaussian_request_index,
    ):
        assert int(
            scalar_counts[changed_index] - single_counts[changed_index]
        ) == (sketch_size - 1)
    ignored = {
        sketch_call_index,
        total_call_index,
        search_call_index,
        gaussian_request_index,
    }
    for index in range(single_counts.numel()):
        if index not in ignored:
            assert int(single_counts[index]) == int(scalar_counts[index])


assert_block_partition_contract(cases[0], 3)
assert_block_partition_contract(cases[4], 2)

print("test_lsrn_differential passed")
