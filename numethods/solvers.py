from __future__ import annotations
from operator import mul
from typing import Iterable, List, Optional
from .linalg import Matrix, Vector, forward_substitution, backward_substitution
from .exceptions import (
    NonSquareMatrixError,
    SingularMatrixError,
    NotSymmetricError,
    NotPositiveDefiniteError,
    ConvergenceError,
)
from .utils import max_abs, pivot_tolerance, relative_tolerance_reached


class LUDecomposition:
    """LU decomposition with partial pivoting: PA = LU.

    The row permutation is stored as ``perm`` (row ``k`` of ``PA`` is row
    ``perm[k]`` of ``A``); ``P`` builds the permutation matrix on demand.
    Pass ``record_steps=True`` to keep snapshots of (L, U, P) after every
    elimination step for :meth:`trace` (costs O(n^3) extra time and memory).
    """

    def __init__(self, A: Matrix, record_steps: bool = False):
        if not A.is_square():
            raise NonSquareMatrixError("A must be square")
        self.n = A.n
        self.L = Matrix.identity(self.n)
        self.U = A.copy()
        self.perm: List[int] = list(range(self.n))
        self.swaps = 0
        self.record_steps = record_steps
        self.steps: list[tuple[int, Matrix, Matrix, Matrix]] = []  # store pivot steps
        self._decompose()

    @property
    def P(self) -> Matrix:
        """Permutation matrix with PA = LU."""
        P = Matrix.zeros(self.n, self.n)
        for k, p in enumerate(self.perm):
            P.data[k][p] = 1.0
        return P

    def _decompose(self) -> None:
        n = self.n
        L, U = self.L.data, self.U.data
        tol = pivot_tolerance(max_abs(U), n)
        for k in range(n):
            pivot_row = self.U.max_abs_in_col(k, k)
            if abs(U[pivot_row][k]) <= tol:
                raise SingularMatrixError("Matrix is singular to working precision")
            if pivot_row != k:
                U[k], U[pivot_row] = U[pivot_row], U[k]
                self.perm[k], self.perm[pivot_row] = self.perm[pivot_row], self.perm[k]
                L[k][:k], L[pivot_row][:k] = L[pivot_row][:k], L[k][:k]
                self.swaps += 1
            pivot = U[k]
            pkk = pivot[k]
            for i in range(k + 1, n):
                row = U[i]
                m = row[k] / pkk
                L[i][k] = m
                if m != 0.0:
                    for j in range(k + 1, n):
                        row[j] -= m * pivot[j]
                row[k] = 0.0
            if self.record_steps:
                self.steps.append((k, self.L.copy(), self.U.copy(), self.P))

    def solve(self, b: Vector) -> Vector:
        if len(b) != self.n:
            raise ValueError("Dimension mismatch: len(b) must equal the size of A")
        Pb = Vector([b[p] for p in self.perm])
        y = forward_substitution(self.L, Pb)
        x = backward_substitution(self.U, y)
        return x

    def det(self) -> float:
        """Determinant of A: (-1)^swaps * prod(diag(U))."""
        d = -1.0 if self.swaps % 2 else 1.0
        for i in range(self.n):
            d *= self.U.data[i][i]
        return d

    def inverse(self) -> Matrix:
        """A^{-1}, computed column by column from the stored factors."""
        n = self.n
        cols = []
        for j in range(n):
            e = Vector([1.0 if i == j else 0.0 for i in range(n)])
            cols.append(self.solve(e).data)
        return Matrix._from_rows([[cols[j][i] for j in range(n)] for i in range(n)], n)

    def trace(self):
        if not self.steps:
            print("No steps recorded. Create the solver with record_steps=True.")
            return
        print("LU Decomposition Trace (steps of elimination)")
        for k, L, U, P in self.steps:
            print(f"\nStep {k}:")
            print(f"L = {L}")
            print(f"U = {U}")
            print(f"P = {P}")


class GaussJordan:
    """Gauss-Jordan elimination with partial pivoting.

    Pass ``record_steps=True`` to keep a copy of the augmented matrix after
    every column for :meth:`trace` (reset on each call to :meth:`solve`).
    """

    def __init__(self, A: Matrix, record_steps: bool = False):
        if not A.is_square():
            raise NonSquareMatrixError("A must be square")
        self.n = A.n
        self.A = A.copy()
        self.record_steps = record_steps
        self.steps: list[tuple[int, Matrix]] = []

    def solve(self, b: Vector) -> Vector:
        n = self.n
        Ab = self.A.augment(b)
        self.steps = []
        tol = pivot_tolerance(max_abs(self.A.data), n)
        for col in range(n):
            pivot = Ab.max_abs_in_col(col, col)
            if abs(Ab.data[pivot][col]) <= tol:
                raise SingularMatrixError("Matrix is singular or nearly singular")
            Ab.swap_rows(col, pivot)
            pv = Ab.data[col][col]
            Ab.data[col] = [v / pv for v in Ab.data[col]]
            pivot_row = Ab.data[col]
            for r in range(n):
                if r == col:
                    continue
                factor = Ab.data[r][col]
                if factor != 0.0:
                    Ab.data[r] = [rv - factor * cv for rv, cv in zip(Ab.data[r], pivot_row)]
            if self.record_steps:
                self.steps.append((col, Ab.copy()))
        return Vector(row[-1] for row in Ab.data)

    def trace(self):
        if not self.steps:
            print("No steps recorded. Create the solver with record_steps=True.")
            return
        print("Gauss-Jordan Trace (row reduction steps)")
        for step, Ab in self.steps:
            print(f"\nColumn {step}:")
            print(f"Augmented matrix = {Ab}")


class _StationaryIteration:
    """Shared set-up for Jacobi / Gauss-Seidel / SOR."""

    name = "Stationary iteration"

    def __init__(
        self, A: Matrix, b: Vector, tol: float = 1e-10, max_iter: int = 10_000
    ):
        if not A.is_square():
            raise NonSquareMatrixError("A must be square")
        if A.n != len(b):
            raise ValueError("Dimension mismatch")
        self.A = A.copy()
        self.b = b.copy()
        self.tol = tol
        self.max_iter = max_iter
        self.history: list[float] = []

    def _diagonal(self) -> List[float]:
        n = self.A.n
        diag = [self.A.data[i][i] for i in range(n)]
        tol = pivot_tolerance(max_abs(self.A.data), n)
        if any(abs(d) <= tol for d in diag):
            raise SingularMatrixError(f"Zero diagonal entry in {self.name}")
        return diag

    def _start(self, x0: Optional[Iterable[float]]) -> List[float]:
        n = self.A.n
        if x0 is None:
            return [0.0] * n
        x = [float(v) for v in x0]
        if len(x) != n:
            raise ValueError("x0 has the wrong length")
        return x

    def _converged(self, x: List[float]) -> bool:
        """Record the residual norm ||Ax - b||_2 and apply the relative test."""
        A, b = self.A.data, self.b.data
        res2 = 0.0
        for row, bi in zip(A, b):
            ri = sum(map(mul, row, x), 0.0) - bi
            res2 += ri * ri
        res_norm = res2**0.5
        self.history.append(res_norm)
        return relative_tolerance_reached(res_norm, sum(v * v for v in x) ** 0.5, self.tol)

    def trace(self):
        print(f"{self.name} Iteration Trace")
        print(f"{'iter':>6} | {'residual norm':>14}")
        print("-" * 26)
        for k, res in enumerate(self.history):
            print(f"{k:6d} | {res:14.6e}")


class Jacobi(_StationaryIteration):
    """Jacobi iterative method for Ax = b."""

    name = "Jacobi"

    def solve(self, x0: Vector | None = None) -> Vector:
        n = self.A.n
        A, b = self.A.data, self.b.data
        diag = self._diagonal()
        x = self._start(x0)
        self.history.clear()
        for _ in range(self.max_iter):
            x_new = [
                (b[i] - sum(A[i][j] * x[j] for j in range(n) if j != i)) / diag[i]
                for i in range(n)
            ]
            if self._converged(x_new):
                return Vector(x_new)
            x = x_new
        raise ConvergenceError("Jacobi did not converge within max_iter")


class SOR(_StationaryIteration):
    """Successive over-relaxation (SOR) for Ax = b; omega = 1 is Gauss-Seidel."""

    name = "SOR"

    def __init__(
        self,
        A: Matrix,
        b: Vector,
        omega: float = 1.5,
        tol: float = 1e-10,
        max_iter: int = 10_000,
    ):
        if not 0.0 < omega < 2.0:
            raise ValueError("SOR requires 0 < omega < 2")
        super().__init__(A, b, tol=tol, max_iter=max_iter)
        self.omega = omega

    def solve(self, x0: Vector | None = None) -> Vector:
        n = self.A.n
        A, b, w = self.A.data, self.b.data, self.omega
        diag = self._diagonal()
        x = self._start(x0)
        self.history.clear()
        for _ in range(self.max_iter):
            # In-place sweep: x[j] already holds the new value for j < i.
            for i in range(n):
                row = A[i]
                s = sum(row[j] * x[j] for j in range(n) if j != i)
                gs = (b[i] - s) / diag[i]
                x[i] = gs if w == 1.0 else (1.0 - w) * x[i] + w * gs
            if self._converged(x):
                return Vector(x)
        raise ConvergenceError(f"{self.name} did not converge within max_iter")


class GaussSeidel(SOR):
    """Gauss-Seidel iterative method for Ax = b (SOR with omega = 1)."""

    name = "Gauss-Seidel"

    def __init__(
        self, A: Matrix, b: Vector, tol: float = 1e-10, max_iter: int = 10_000
    ):
        super().__init__(A, b, omega=1.0, tol=tol, max_iter=max_iter)


class Cholesky:
    """Cholesky factorization A = L L^T for SPD matrices.

    Pass ``record_steps=True`` to keep a copy of L after every row for
    :meth:`trace`.
    """

    # Relative tolerance for the symmetry check |a_ij - a_ji| <= SYM_TOL * max|a|.
    SYM_TOL = 1e-12

    def __init__(self, A: Matrix, record_steps: bool = False):
        if not A.is_square():
            raise NonSquareMatrixError("A must be square")
        n = A.n
        scale = max_abs(A.data)
        for i in range(n):
            for j in range(i + 1, n):
                if abs(A.data[i][j] - A.data[j][i]) > self.SYM_TOL * scale:
                    raise NotSymmetricError("Matrix is not symmetric")
        self.n = n
        self.L = Matrix.zeros(n, n)
        self.record_steps = record_steps
        self.steps: list[tuple[int, Matrix]] = []
        self._decompose(A, scale)
        self._LT = self.L.transpose()

    def _decompose(self, A: Matrix, scale: float) -> None:
        n = self.n
        L = self.L.data
        tol = pivot_tolerance(scale, n)
        for i in range(n):
            for j in range(i + 1):
                s = sum(L[i][k] * L[j][k] for k in range(j))
                if i == j:
                    val = A.data[i][i] - s
                    if val <= tol:
                        raise NotPositiveDefiniteError(
                            "Matrix is not positive definite"
                        )
                    L[i][j] = val**0.5
                else:
                    L[i][j] = (A.data[i][j] - s) / L[j][j]
            if self.record_steps:
                self.steps.append((i, self.L.copy()))

    def solve(self, b: Vector) -> Vector:
        if len(b) != self.n:
            raise ValueError("Dimension mismatch: len(b) must equal the size of A")
        y = forward_substitution(self.L, b)
        x = backward_substitution(self._LT, y)
        return x

    def trace(self):
        if not self.steps:
            print("No steps recorded. Create the solver with record_steps=True.")
            return
        print("Cholesky Decomposition Trace")
        for i, L in self.steps:
            print(f"\nRow {i}:")
            print(f"L = {L}")
