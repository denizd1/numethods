from __future__ import annotations
from .linalg import Matrix, Vector
from .orthogonal import QRHouseholder
from .solvers import LUDecomposition
from .exceptions import NonSquareMatrixError, ConvergenceError
from .linalg import Matrix, Vector
import math


def solve_linear(M: Matrix, b: Vector) -> Vector:
    """Solve Mx = b using LU decomposition."""
    solver = LUDecomposition(M)
    return solver.solve(b)


class PowerIteration:
    def __init__(self, A: Matrix, tol: float = 1e-10, max_iter: int = 5000):
        if not A.is_square():
            raise NonSquareMatrixError("A must be square")
        self.A, self.tol, self.max_iter = A, tol, max_iter
        self.history = []

    def solve(self, x0: Vector | None = None) -> tuple[float, Vector]:
        n = self.A.n
        x = Vector([1.0] * n) if x0 is None else x0.copy()
        lam_old = 0.0
        self.history.clear()

        for k in range(self.max_iter):
            y = self.A @ x
            nrm = y.norm2()
            if nrm == 0.0:
                raise ConvergenceError("Zero vector encountered")
            x = (1.0 / nrm) * y
            lam = (x.dot(self.A @ x)) / (x.dot(x))
            err = abs(lam - lam_old)

            self.history.append({"iter": k, "lambda": lam, "error": err})

            if err <= self.tol * (1.0 + abs(lam)):
                return lam, x
            lam_old = lam

        raise ConvergenceError("Power iteration did not converge")

    def trace(self):
        if not self.history:
            print("No iterations stored. Run .solve() first.")
            return
        print("Power Iteration Trace")
        print(f"{'iter':>6} | {'lambda':>12} | {'error':>12}")
        print("-" * 40)
        for row in self.history:
            print(f"{row['iter']:6d} | {row['lambda']:12.6e} | {row['error']:12.6e}")


class InversePowerIteration:
    def __init__(
        self, A: Matrix, shift: float = 0.0, tol: float = 1e-10, max_iter: int = 5000
    ):
        if not A.is_square():
            raise NonSquareMatrixError("A must be square")
        self.A, self.shift, self.tol, self.max_iter = A, shift, tol, max_iter
        self.history = []

    def solve(self, x0: Vector | None = None) -> tuple[float, Vector]:
        n = self.A.n
        x = Vector([1.0] * n) if x0 is None else x0.copy()
        mu_old = None
        self.history.clear()

        for k in range(self.max_iter):
            M = Matrix(
                [
                    [
                        self.A.data[i][j] - (self.shift if i == j else 0.0)
                        for j in range(n)
                    ]
                    for i in range(n)
                ]
            )
            y = solve_linear(M, x)
            nrm = y.norm2()
            if nrm == 0.0:
                raise ConvergenceError("Zero vector")
            x = (1.0 / nrm) * y
            mu = (x.dot(self.A @ x)) / (x.dot(x))
            err = abs(mu - mu_old) if mu_old is not None else float("inf")

            self.history.append({"iter": k, "mu": mu, "error": err})

            if (mu_old is not None) and err <= self.tol * (1.0 + abs(mu)):
                return mu, x
            mu_old = mu

        raise ConvergenceError("Inverse/shifted power iteration did not converge")

    def trace(self):
        if not self.history:
            print("No iterations stored. Run .solve() first.")
            return
        print("Inverse/Shifted Power Iteration Trace")
        print(f"{'iter':>6} | {'mu':>12} | {'error':>12}")
        print("-" * 40)
        for row in self.history:
            print(f"{row['iter']:6d} | {row['mu']:12.6e} | {row['error']:12.6e}")


class RayleighQuotientIteration:
    def __init__(self, A: Matrix, tol: float = 1e-12, max_iter: int = 1000):
        if not A.is_square():
            raise NonSquareMatrixError("A must be square")
        self.A, self.tol, self.max_iter = A, tol, max_iter
        self.history = []

    def solve(self, x0: Vector | None = None) -> tuple[float, Vector]:
        n = self.A.n
        x = Vector([1.0] * n) if x0 is None else x0.copy()
        x = (1.0 / x.norm2()) * x
        mu = (x.dot(self.A @ x)) / (x.dot(x))
        self.history.clear()

        for k in range(self.max_iter):
            M = Matrix(
                [
                    [self.A.data[i][j] - (mu if i == j else 0.0) for j in range(n)]
                    for i in range(n)
                ]
            )
            y = solve_linear(M, x)
            x = (1.0 / y.norm2()) * y
            mu_new = (x.dot(self.A @ x)) / (x.dot(x))
            err = abs(mu_new - mu)

            self.history.append({"iter": k, "mu": mu_new, "error": err})

            if err <= self.tol * (1.0 + abs(mu_new)):
                return mu_new, x
            mu = mu_new

        raise ConvergenceError("Rayleigh quotient iteration did not converge")

    def trace(self):
        if not self.history:
            print("No iterations stored. Run .solve() first.")
            return
        print("Rayleigh Quotient Iteration Trace")
        print(f"{'iter':>6} | {'mu':>12} | {'error':>12}")
        print("-" * 40)
        for row in self.history:
            print(f"{row['iter']:6d} | {row['mu']:12.6e} | {row['error']:12.6e}")


class QREigenvalues:
    def __init__(self, A: Matrix, tol: float = 1e-10, max_iter: int = 10000):
        if not A.is_square():
            raise NonSquareMatrixError("A must be square")
        self.A0, self.tol, self.max_iter = A.copy(), tol, max_iter

    def solve(self) -> Matrix:
        A = self.A0.copy()
        n = A.n
        for _ in range(self.max_iter):
            qr = QRHouseholder(A)
            Q, R = qr.Q, qr.R
            A = R @ Q
            off = 0.0
            for i in range(1, n):
                off += sum(abs(A.data[i][j]) for j in range(0, i))
            if off <= self.tol:
                return A
        raise ConvergenceError("QR did not converge")


class SVD:
    def __init__(self, A: Matrix, tol: float = 1e-10, max_iter: int = 10000):
        self.A, A = A, A
        self.tol, self.max_iter = tol, max_iter

    def _eig_sym(self, S: Matrix):
        n = S.n
        V = Matrix.identity(n)
        A = S.copy()
        for _ in range(self.max_iter):
            qr = QRHouseholder(A)
            Q, R = qr.Q, qr.R
            A = R @ Q
            V = V @ Q
            off = 0.0
            for i in range(1, n):
                off += sum(abs(A.data[i][j]) for j in range(0, i))
            if off <= self.tol:
                break
        return [A.data[i][i] for i in range(n)], V

    def solve(self) -> tuple[Matrix, Vector, Matrix]:
        At = self.A.transpose()
        S = At @ self.A
        eigvals, V = self._eig_sym(S)
        idx = sorted(range(len(eigvals)), key=lambda i: eigvals[i], reverse=True)
        eigvals = [eigvals[i] for i in idx]
        V = Matrix([[V.data[r][i] for i in idx] for r in range(V.m)])
        sing = [math.sqrt(ev) if ev > 0 else 0.0 for ev in eigvals]
        Ucols = []
        for j, sv in enumerate(sing):
            vj = V.col(j)
            Av = self.A @ vj
            if sv > 1e-14:
                uj = (1.0 / sv) * Av
            else:
                nrm = Av.norm2()
                uj = (1.0 / nrm) * Av if nrm > 0 else Vector([0.0] * self.A.m)
            nrm = uj.norm2()
            uj = (1.0 / nrm) * uj if nrm > 0 else uj
            Ucols.append(uj.data)
        U = Matrix([[Ucols[j][i] for j in range(len(Ucols))] for i in range(self.A.m)])

        Sigma = Vector(sing)
        return U, Sigma, V
