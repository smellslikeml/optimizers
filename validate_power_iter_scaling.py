"""
Validation harness for the power-iteration lambda-max scaling of the iterative inverse-root
solvers in Distributed Shampoo (GPU / Colab gated; also runs on CPU at small sizes).

Compares, on Shampoo-like preconditioner factors (accumulated gradient outer products):

  1. Accuracy  -- power-iteration lambda_max estimate (for several iteration counts) against
                  torch.linalg.eigvalsh and against the loose infinity-norm upper bound that
                  feeds the ridge today.
  2. Speed     -- iterations-to-converge (and wall time) of the coupled higher-order solver
                  with the default trace-based scale (`lambda_max_power_iterations=0`) versus
                  the opt-in power-iteration scale, at several iteration counts.
  3. Correctness -- the final inverse root under each scale against the eigendecomposition
                  reference, within tolerance.

The technique is credited to the DASH optimizer (Modoranu et al., IST-DASLab,
arXiv:2602.02016, MIT-licensed); see power_iteration_lambda_max in
distributed_shampoo/preconditioner/matrix_functions.py.

Usage:
    # CPU smoke test (small sizes):
    python validate_power_iter_scaling.py --device cpu --dims 128 256

    # GPU run at representative Shampoo block sizes:
    python validate_power_iter_scaling.py --device cuda --dims 1024 2048 4096 8192

Colab notebook plan (run on a GPU runtime):
    Cell 1: !pip install -q "git+https://github.com/<your-fork>/optimizers.git@<this-branch>"
    Cell 2: import torch; assert torch.cuda.is_available(); !nvidia-smi
    Cell 3: %run validate_power_iter_scaling.py --device cuda --dims 1024 2048 4096 8192 \
                --num-matrices 5
    Cell 4 (optional sweep): loop `--power-iterations` over 1..32 and plot (a) the relative
            error of the estimate vs torch.linalg.eigvalsh and (b) solver iterations, both
            against k, to pick the knee of the curve.
    Cell 5: record the printed tables into the RFC/issue coordinated with the maintainers
            (hjmshi / runame / wz337); note any dtype (fp32 vs bf16) differences.

"""

import argparse
import logging
import re
import time
from collections.abc import Callable
from fractions import Fraction

import torch
from distributed_shampoo.preconditioner import matrix_functions
from distributed_shampoo.preconditioner.matrix_functions import (
    matrix_inverse_root,
    power_iteration_lambda_max,
)
from distributed_shampoo.preconditioner.matrix_functions_types import (
    CoupledHigherOrderConfig,
    DefaultEigenConfig,
)
from torch import Tensor

_NUMBER_OF_ITERATIONS_LOG_PREFIX = "Number of iterations:"
_TERMINATION_FLAG_LOG_PREFIX = "Termination Flag:"


class _LogCapture(logging.Handler):
    """Captures debug log records emitted by the matrix-functions module."""

    def __init__(self) -> None:
        super().__init__(level=logging.DEBUG)
        self.messages: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.messages.append(record.getMessage())


def _capture_solver_logs(func: Callable[[], Tensor]) -> tuple[Tensor, int, str]:
    """Runs func() while capturing the higher-order solver's debug logs.

    Returns the function's result along with the solver's reported iteration count and
    termination flag (both logged at DEBUG level by matrix_inverse_root_higher_order).
    """
    capture = _LogCapture()
    module_logger = matrix_functions.logger
    previous_level, previous_propagate = module_logger.level, module_logger.propagate
    module_logger.setLevel(logging.DEBUG)
    module_logger.propagate = False
    module_logger.addHandler(capture)
    try:
        result = func()
    finally:
        module_logger.removeHandler(capture)
        module_logger.setLevel(previous_level)
        module_logger.propagate = previous_propagate
    iterations = 0
    termination_flag = "UNKNOWN"
    for message in capture.messages:
        if match := re.fullmatch(rf"{_NUMBER_OF_ITERATIONS_LOG_PREFIX} (\d+)", message):
            iterations = int(match.group(1))
        elif message.startswith(_TERMINATION_FLAG_LOG_PREFIX):
            # E.g. "Termination Flag: NewtonConvergenceFlag.CONVERGED" -> "CONVERGED".
            termination_flag = message.rsplit(".", maxsplit=1)[-1].strip()
    return result, iterations, termination_flag


def make_shampoo_like_factor(
    dim: int, dtype: torch.dtype, device: str, seed: int, steps_multiplier: float = 0.25
) -> Tensor:
    """Accumulated gradient outer products, as a Shampoo second-moment factor.

    With `steps = max(1, round(steps_multiplier * dim))` gradient steps the factor is
    rank-deficient early in training (steps < dim) and approaches full rank later, which
    spans the regimes the preconditioner actually sees.
    """
    generator = torch.Generator(device=device).manual_seed(seed)
    steps = max(1, round(steps_multiplier * dim))
    gradients = torch.randn((steps, dim), generator=generator, dtype=torch.float32, device=device)
    return (gradients.T @ gradients).to(dtype=dtype, device=device)


def synchronize(device: str) -> None:
    if device.startswith("cuda"):
        torch.cuda.synchronize()


def validate_accuracy(
    dims: list[int],
    power_iterations: list[int],
    dtype: torch.dtype,
    device: str,
    num_matrices: int,
) -> None:
    print("=" * 78)
    print("1. Lambda-max estimate accuracy (reference: torch.linalg.eigvalsh in float64)")
    print("=" * 78)
    estimate_columns = [f"err(k={k})" for k in power_iterations]
    print(
        f"{'dim':>6} {'seed':>4} {'lambda_max':>12} {'inf-norm':>12} {'inf/lambda':>10} "
        + " ".join(f"{column:>11}" for column in estimate_columns)
    )
    for dim in dims:
        for seed in range(num_matrices):
            A = make_shampoo_like_factor(dim, dtype, device, seed)
            lambda_max = torch.linalg.eigvalsh(A.to(torch.float64)).max().item()
            infinity_norm = torch.linalg.matrix_norm(A, torch.inf).item()
            errors = []
            for k in power_iterations:
                estimate = power_iteration_lambda_max(A, num_iterations=k).to(
                    torch.float64
                )
                errors.append(abs(estimate.item() - lambda_max) / lambda_max)
            print(
                f"{dim:>6} {seed:>4} {lambda_max:>12.4e} {infinity_norm:>12.4e} "
                f"{infinity_norm / lambda_max:>10.3f} "
                + " ".join(f"{error:>11.3e}" for error in errors)
            )
    print()


def validate_solver(
    dims: list[int],
    power_iterations: list[int],
    dtype: torch.dtype,
    device: str,
    num_matrices: int,
    root: int,
    order: int,
    rel_epsilon: float,
    abs_epsilon: float,
) -> None:
    print("=" * 78)
    print(
        f"2. Coupled higher-order solver: iterations / wall time "
        f"(root={root}, order={order}, rel_epsilon={rel_epsilon}, abs_epsilon={abs_epsilon})"
    )
    print("=" * 78)
    cells_header = [f"k={k}" for k in (0, *power_iterations)]
    print(f"{'dim':>6} {'seed':>4} " + " ".join(f"{cell:>22}" for cell in cells_header))
    for dim in dims:
        for seed in range(num_matrices):
            A = make_shampoo_like_factor(dim, dtype, device, seed)
            cells = []
            for k in (0, *power_iterations):
                config = CoupledHigherOrderConfig(
                    rel_epsilon=rel_epsilon,
                    abs_epsilon=abs_epsilon,
                    order=order,
                    lambda_max_power_iterations=k,
                )
                synchronize(device)
                begin = time.perf_counter()
                try:
                    _, iterations, termination_flag = _capture_solver_logs(
                        lambda: matrix_inverse_root(
                            A=A, root=Fraction(root), root_inv_config=config
                        )
                    )
                    synchronize(device)
                    cell = f"{iterations:>4}it {termination_flag:>9}"
                except ArithmeticError as error:
                    cell = f"FAILED({str(error)[:8]})"
                cells.append(f"{cell} {(time.perf_counter() - begin) * 1e3:>6.1f}ms")
            print(f"{dim:>6} {seed:>4} " + " ".join(f"{cell:>22}" for cell in cells))
    print()


def validate_inverse_root_agreement(
    dims: list[int],
    power_iterations: list[int],
    dtype: torch.dtype,
    device: str,
    num_matrices: int,
    root: int,
    order: int,
    rel_epsilon: float,
    abs_epsilon: float,
) -> None:
    print("=" * 78)
    print("3. Inverse-root agreement vs eigendecomposition reference (relative Frobenius error)")
    print("=" * 78)
    cells_header = [f"k={k}" for k in (0, *power_iterations)]
    print(f"{'dim':>6} {'seed':>4} " + " ".join(f"{cell:>13}" for cell in cells_header))
    for dim in dims:
        for seed in range(num_matrices):
            A = make_shampoo_like_factor(dim, dtype, device, seed)
            epsilon = rel_epsilon * torch.linalg.matrix_norm(A, torch.inf).item()
            reference = matrix_inverse_root(
                A=A, root=Fraction(root), root_inv_config=DefaultEigenConfig, epsilon=epsilon
            )
            reference_norm = torch.linalg.matrix_norm(reference).item()
            cells = []
            for k in (0, *power_iterations):
                config = CoupledHigherOrderConfig(
                    rel_epsilon=rel_epsilon,
                    abs_epsilon=abs_epsilon,
                    order=order,
                    lambda_max_power_iterations=k,
                )
                try:
                    X = matrix_inverse_root(A=A, root=Fraction(root), root_inv_config=config)
                    error = (
                        torch.linalg.matrix_norm((X - reference).to(torch.float64)).item()
                        / reference_norm
                    )
                    cells.append(f"{error:>13.3e}")
                except ArithmeticError:
                    cells.append(f"{'FAILED':>13}")
            print(f"{dim:>6} {seed:>4} " + " ".join(cells))
    print()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu", help="cpu or cuda (default: cpu)")
    parser.add_argument(
        "--dims",
        nargs="+",
        type=int,
        default=(128, 256),
        help="matrix dimensions (default: 128 256; GPU: 1024 2048 4096 8192)",
    )
    parser.add_argument(
        "--power-iterations",
        nargs="+",
        type=int,
        default=(4, 8, 16),
        help="power-iteration counts to sweep (default: 4 8 16)",
    )
    parser.add_argument(
        "--num-matrices", type=int, default=3, help="matrices (seeds) per dimension"
    )
    parser.add_argument(
        "--dtype",
        default="float32",
        choices=("float32", "bfloat16", "float64"),
        help="compute dtype for the solver and the estimate (default: float32)",
    )
    parser.add_argument("--root", type=int, default=2, help="inverse root (default: 2)")
    parser.add_argument("--order", type=int, default=3, help="solver order (default: 3)")
    parser.add_argument("--rel-epsilon", type=float, default=1e-6)
    parser.add_argument("--abs-epsilon", type=float, default=0.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dtype = getattr(torch, args.dtype)
    print(
        f"device={args.device} dtype={args.dtype} dims={list(args.dims)} "
        f"power_iterations={list(args.power_iterations)} num_matrices={args.num_matrices}"
    )
    validate_accuracy(args.dims, args.power_iterations, dtype, args.device, args.num_matrices)
    validate_solver(
        args.dims,
        args.power_iterations,
        dtype,
        args.device,
        args.num_matrices,
        root=args.root,
        order=args.order,
        rel_epsilon=args.rel_epsilon,
        abs_epsilon=args.abs_epsilon,
    )
    validate_inverse_root_agreement(
        args.dims,
        args.power_iterations,
        dtype,
        args.device,
        args.num_matrices,
        root=args.root,
        order=args.order,
        rel_epsilon=args.rel_epsilon,
        abs_epsilon=args.abs_epsilon,
    )


if __name__ == "__main__":
    main()
