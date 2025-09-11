from __future__ import annotations
from typing import List
from .linalg import Matrix, Vector, backward_substitution
from .exceptions import SingularMatrixError


class QRGramSchmidt:
    """Classical Gram–Schmidt orthogonalization."""

    def __init__(self, A: Matrix):
        self.m, self.n = A.shape()
        self.Q = Matrix.zeros(self.m, self.n)
        self.R = Matrix.zeros(self.n, self.n)
        self._decompose(A)

    def _decompose(self, A: Matrix) -> None:
        m, n = self.m, self.n
        Qcols: List[Vector] = []
        for j in range(n):
            v = A.col(j)
            for k in range(j):
                qk = Qcols[k]
                r = qk.dot(v)
                self.R.data[k][j] = r
                v = Vector([vi - r * qi for vi, qi in zip(v.data, qk.data)])
            norm = sum(vi * vi for vi in v.data) ** 0.5
            if abs(norm) < 1e-15:
                raise SingularMatrixError("Linearly dependent columns in Gram-Schmidt")
            self.R.data[j][j] = norm
            qj = Vector([vi / norm for vi in v.data])
            Qcols.append(qj)
            for i in range(m):
                self.Q.data[i][j] = qj[i]


class QRModifiedGramSchmidt:
    """Modified Gram–Schmidt orthogonalization."""

    def __init__(self, A: Matrix):
        self.m, self.n = A.shape()
        self.Q = Matrix.zeros(self.m, self.n)
        self.R = Matrix.zeros(self.n, self.n)
        self._decompose(A)

    def _decompose(self, A: Matrix) -> None:
        m, n = self.m, self.n
        V = [A.col(j).data for j in range(n)]
        for i in range(n):
            vi = Vector(V[i])
            norm = sum(v * v for v in vi.data) ** 0.5
            if abs(norm) < 1e-15:
                raise SingularMatrixError("Linearly dependent columns in MGS")
            self.R.data[i][i] = norm
            qi = Vector([v / norm for v in vi.data])
            for r in range(m):
                self.Q.data[r][i] = qi[r]
            for j in range(i + 1, n):
                r = qi.dot(Vector(V[j]))
                self.R.data[i][j] = r
                V[j] = [vj - r * qi_k for vj, qi_k in zip(V[j], qi.data)]


class QRHouseholder:
    """Stable QR decomposition using Householder reflectors."""

    def __init__(self, A: Matrix):
        self.m, self.n = A.shape()
        self.R = A.copy()
        self.Q = Matrix.identity(self.m)
        self._decompose()

    def _decompose(self) -> None:
        m, n = self.m, self.n
        for k in range(min(m, n)):
            x = [self.R.data[i][k] for i in range(k, m)]
            normx = sum(xi * xi for xi in x) ** 0.5
            if normx < 1e-15:
                continue
            sign = 1.0 if x[0] >= 0 else -1.0
            u1 = x[0] + sign * normx
            v = [xi / u1 if i > 0 else 1.0 for i, xi in enumerate(x)]
            normv = sum(vi * vi for vi in v) ** 0.5
            v = [vi / normv for vi in v]
            for j in range(k, n):
                s = sum(v[i] * self.R.data[k + i][j] for i in range(len(v)))
                for i in range(len(v)):
                    self.R.data[k + i][j] -= 2 * s * v[i]
            for j in range(m):
                s = sum(v[i] * self.Q.data[j][k + i] for i in range(len(v)))
                for i in range(len(v)):
                    self.Q.data[j][k + i] -= 2 * s * v[i]
        self.Q = self.Q.transpose()


class QRSolver:
    """Solve Ax=b given QR (square A)."""

    def __init__(self, qr: QRHouseholder | QRGramSchmidt | QRModifiedGramSchmidt):
        self.Q, self.R = qr.Q, qr.R

    def solve(self, b: Vector) -> Vector:
        Qtb = Vector(
            [
                sum(self.Q.data[i][j] * b[i] for i in range(self.Q.m))
                for j in range(self.Q.n)
            ]
        )
        return backward_substitution(self.R, Qtb)


class LeastSquaresSolver:
    """Solve overdetermined system Ax ≈ b in least squares sense using QR."""

    def __init__(self, A: Matrix, b: Vector):
        self.A, self.b = A, b

    def solve(self) -> Vector:
        qr = QRHouseholder(self.A)
        Q, R = qr.Q, qr.R
        # Compute Q^T b (dimension m)
        Qtb_full = [
            sum(Q.data[i][j] * self.b[i] for i in range(Q.m)) for j in range(Q.n)
        ]
        # Take only first n entries
        Qtb = Vector(Qtb_full[: self.A.n])
        # Extract leading n×n block of R
        Rtop = Matrix([R.data[i][: self.A.n] for i in range(self.A.n)])
        return backward_substitution(Rtop, Qtb)
