from __future__ import annotations
from .linalg import Matrix, Vector
from .orthogonal import QRHouseholder
from .exceptions import NonSquareMatrixError, ConvergenceError
import math


class PowerIteration:
    def __init__(self, A: Matrix, tol: float = 1e-10, max_iter: int = 50000):
        if not A.is_square():
            raise NonSquareMatrixError("A must be square")
        self.A, self.tol, self.max_iter = A, tol, max_iter

    def solve(self, x0: Vector | None = None) -> tuple[float, Vector]:
        n = self.A.n
        x = Vector([1.0] * n) if x0 is None else x0.copy()
        lam_old = 0.0
        for _ in range(self.max_iter):
            y = self.A @ x
            nrm = y.norm2()
            if nrm == 0.0:
                raise ConvergenceError("Zero vector encountered")
            x = (1.0 / nrm) * y
            lam = (x.dot(self.A @ x)) / (x.dot(x))
            if abs(lam - lam_old) <= self.tol * (1.0 + abs(lam)):
                return lam, x
            lam_old = lam
        raise ConvergenceError("Power iteration did not converge")


class InversePowerIteration:
    def __init__(
        self, A: Matrix, shift: float = 0.0, tol: float = 1e-10, max_iter: int = 50000
    ):
        if not A.is_square():
            raise NonSquareMatrixError("A must be square")
        self.A, self.shift, self.tol, self.max_iter = A, shift, tol, max_iter

    def _solve_linear(self, M: Matrix, b: Vector) -> Vector:
        n = M.n
        Ab = M.augment(b)
        for col in range(n):
            piv = max(range(col, n), key=lambda i: abs(Ab.data[i][col]))
            if abs(Ab.data[piv][col]) < 1e-15:
                raise ConvergenceError("Singular in inverse iteration")
            Ab.swap_rows(col, piv)
            for r in range(col + 1, n):
                m = Ab.data[r][col] / Ab.data[col][col]
                for c in range(col, n + 1):
                    Ab.data[r][c] -= m * Ab.data[col][c]
        x = [0.0] * n
        for i in range(n - 1, -1, -1):
            s = sum(Ab.data[i][j] * x[j] for j in range(i + 1, n))
            x[i] = (Ab.data[i][n] - s) / Ab.data[i][i]
        return Vector(x)

    def solve(self, x0: Vector | None = None) -> tuple[float, Vector]:
        n = self.A.n
        x = Vector([1.0] * n) if x0 is None else x0.copy()
        mu_old = None
        for _ in range(self.max_iter):
            M = Matrix(
                [
                    [
                        self.A.data[i][j] - (self.shift if i == j else 0.0)
                        for j in range(n)
                    ]
                    for i in range(n)
                ]
            )
            y = self._solve_linear(M, x)
            nrm = y.norm2()
            if nrm == 0.0:
                raise ConvergenceError("Zero vector")
            x = (1.0 / nrm) * y
            mu = (x.dot(self.A @ x)) / (x.dot(x))
            if (mu_old is not None) and abs(mu - mu_old) <= self.tol * (1.0 + abs(mu)):
                return mu, x
            mu_old = mu
        raise ConvergenceError("Inverse/shifted power iteration did not converge")


class RayleighQuotientIteration:
    def __init__(self, A: Matrix, tol: float = 1e-12, max_iter: int = 1000):
        if not A.is_square():
            raise NonSquareMatrixError("A must be square")
        self.A, self.tol, self.max_iter = A, tol, max_iter

    def _solve_linear(self, M: Matrix, b: Vector) -> Vector:
        n = M.n
        Ab = M.augment(b)
        for col in range(n):
            piv = max(range(col, n), key=lambda i: abs(Ab.data[i][col]))
            if abs(Ab.data[piv][col]) < 1e-15:
                raise ConvergenceError("Singular in RQI")
            Ab.swap_rows(col, piv)
            for r in range(col + 1, n):
                m = Ab.data[r][col] / Ab.data[col][col]
                for c in range(col, n + 1):
                    Ab.data[r][c] -= m * Ab.data[col][c]
        x = [0.0] * n
        for i in range(n - 1, -1, -1):
            s = sum(Ab.data[i][j] * x[j] for j in range(i + 1, n))
            x[i] = (Ab.data[i][n] - s) / Ab.data[i][i]
        return Vector(x)

    def solve(self, x0: Vector | None = None) -> tuple[float, Vector]:
        n = self.A.n
        x = Vector([1.0] * n) if x0 is None else x0.copy()
        x = (1.0 / x.norm2()) * x
        mu = (x.dot(self.A @ x)) / (x.dot(x))
        for _ in range(self.max_iter):
            M = Matrix(
                [
                    [self.A.data[i][j] - (mu if i == j else 0.0) for j in range(n)]
                    for i in range(n)
                ]
            )
            y = self._solve_linear(M, x)
            x = (1.0 / y.norm2()) * y
            mu_new = (x.dot(self.A @ x)) / (x.dot(x))
            if abs(mu_new - mu) <= self.tol * (1.0 + abs(mu_new)):
                return mu_new, x
            mu = mu_new
        raise ConvergenceError("RQI did not converge")


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
        from .linalg import Vector as Vect

        Sigma = Vect(sing)
        return U, Sigma, V
