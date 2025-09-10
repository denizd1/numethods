from __future__ import annotations
from .linalg import Matrix, Vector, forward_substitution, backward_substitution
from .exceptions import NonSquareMatrixError, SingularMatrixError, NotSymmetricError, NotPositiveDefiniteError, ConvergenceError

class LUDecomposition:
    """LU with partial pivoting: PA = LU"""
    def __init__(self, A: Matrix):
        if not A.is_square():
            raise NonSquareMatrixError("A must be square")
        self.n = A.n
        self.L = Matrix.identity(self.n)
        self.U = A.copy()
        self.P = Matrix.identity(self.n)
        self._decompose()

    def _decompose(self) -> None:
        n = self.n
        for k in range(n):
            pivot_row = self.U.max_abs_in_col(k, k)
            if abs(self.U.data[pivot_row][k]) < 1e-15:
                raise SingularMatrixError("Matrix is singular to working precision")
            self.U.swap_rows(k, pivot_row)
            self.P.swap_rows(k, pivot_row)
            if k > 0:
                self.L.data[k][:k], self.L.data[pivot_row][:k] = self.L.data[pivot_row][:k], self.L.data[k][:k]
            for i in range(k+1, n):
                m = self.U.data[i][k] / self.U.data[k][k]
                self.L.data[i][k] = m
                for j in range(k, n):
                    self.U.data[i][j] -= m * self.U.data[k][j]

    def solve(self, b: Vector) -> Vector:
        Pb = Vector([sum(self.P.data[i][j]*b[j] for j in range(self.n)) for i in range(self.n)])
        y = forward_substitution(self.L, Pb)
        x = backward_substitution(self.U, y)
        return x


class GaussJordan:
    def __init__(self, A: Matrix):
        if not A.is_square():
            raise NonSquareMatrixError("A must be square")
        self.n = A.n
        self.A = A.copy()

    def solve(self, b: Vector) -> Vector:
        n = self.n
        Ab = self.A.augment(b)
        for col in range(n):
            pivot = Ab.max_abs_in_col(col, col)
            if abs(Ab.data[pivot][col]) < 1e-15:
                raise SingularMatrixError("Matrix is singular or nearly singular")
            Ab.swap_rows(col, pivot)
            pv = Ab.data[col][col]
            Ab.data[col] = [v / pv for v in Ab.data[col]]
            for r in range(n):
                if r == col:
                    continue
                factor = Ab.data[r][col]
                Ab.data[r] = [rv - factor*cv for rv, cv in zip(Ab.data[r], Ab.data[col])]
        return Vector(row[-1] for row in Ab.data)


class Jacobi:
    def __init__(self, A: Matrix, b: Vector, tol: float = 1e-10, max_iter: int = 10_000):
        if not A.is_square():
            raise NonSquareMatrixError("A must be square")
        if A.n != len(b):
            raise ValueError("Dimension mismatch")
        self.A = A.copy()
        self.b = b.copy()
        self.tol = tol
        self.max_iter = max_iter

    def solve(self, x0: Vector | None = None) -> Vector:
        n = self.A.n
        x = Vector([0.0]*n) if x0 is None else x0.copy()
        for _ in range(self.max_iter):
            x_new = [0.0]*n
            for i in range(n):
                diag = self.A.data[i][i]
                if abs(diag) < 1e-15:
                    raise SingularMatrixError("Zero diagonal entry in Jacobi")
                s = sum(self.A.data[i][j]*x[j] for j in range(n) if j != i)
                x_new[i] = (self.b[i] - s) / diag
            x_new = Vector(x_new)
            if (x_new - x).norm_inf() <= self.tol * (1.0 + x_new.norm_inf()):
                return x_new
            x = x_new
        raise ConvergenceError("Jacobi did not converge within max_iter")


class GaussSeidel:
    def __init__(self, A: Matrix, b: Vector, tol: float = 1e-10, max_iter: int = 10_000):
        if not A.is_square():
            raise NonSquareMatrixError("A must be square")
        if A.n != len(b):
            raise ValueError("Dimension mismatch")
        self.A = A.copy()
        self.b = b.copy()
        self.tol = tol
        self.max_iter = max_iter

    def solve(self, x0: Vector | None = None) -> Vector:
        n = self.A.n
        x = Vector([0.0]*n) if x0 is None else x0.copy()
        for _ in range(self.max_iter):
            x_old = x.copy()
            for i in range(n):
                diag = self.A.data[i][i]
                if abs(diag) < 1e-15:
                    raise SingularMatrixError("Zero diagonal entry in Gauss-Seidel")
                s1 = sum(self.A.data[i][j]*x[j] for j in range(i))
                s2 = sum(self.A.data[i][j]*x_old[j] for j in range(i+1, n))
                x[i] = (self.b[i] - s1 - s2) / diag
            if (x - x_old).norm_inf() <= self.tol * (1.0 + x.norm_inf()):
                return x
        raise ConvergenceError("Gauss-Seidel did not converge within max_iter")


class Cholesky:
    """A = L L^T for SPD matrices."""
    def __init__(self, A: Matrix):
        if not A.is_square():
            raise NonSquareMatrixError("A must be square")
        n = A.n
        for i in range(n):
            for j in range(i+1, n):
                if abs(A.data[i][j] - A.data[j][i]) > 1e-12:
                    raise NotSymmetricError("Matrix is not symmetric")
        self.n = n
        self.L = Matrix.zeros(n, n)
        self._decompose(A.copy())

    def _decompose(self, A: Matrix) -> None:
        n = self.n
        for i in range(n):
            for j in range(i+1):
                s = sum(self.L.data[i][k]*self.L.data[j][k] for k in range(j))
                if i == j:
                    val = A.data[i][i] - s
                    if val <= 0.0:
                        raise NotPositiveDefiniteError("Matrix is not positive definite")
                    self.L.data[i][j] = val**0.5
                else:
                    self.L.data[i][j] = (A.data[i][j] - s) / self.L.data[j][j]

    def solve(self, b: Vector) -> Vector:
        y = forward_substitution(self.L, b)
        x = backward_substitution(self.L.transpose(), y)
        return x
