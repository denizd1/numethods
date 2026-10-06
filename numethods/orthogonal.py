from __future__ import annotations
from typing import List
from .linalg import Matrix, Vector, backward_substitution
from .exceptions import SingularMatrixError, DomainError
from .utils import pivot_tolerance


class QRGramSchmidt:
    """Classical Gram-Schmidt orthogonalization (thin QR: Q is m x n, R is n x n)."""

    def __init__(self, A: Matrix):
        self.m, self.n = A.shape()
        self.Q = Matrix.zeros(self.m, self.n)
        self.R = Matrix.zeros(self.n, self.n)
        self._decompose(A)

    def _decompose(self, A: Matrix) -> None:
        m, n = self.m, self.n
        Qcols: List[Vector] = []
        for j in range(n):
            a = A.col(j)
            v = a
            for k in range(j):
                qk = Qcols[k]
                r = qk.dot(a)  # classical GS projects the *original* column
                self.R.data[k][j] = r
                v = Vector([vi - r * qi for vi, qi in zip(v.data, qk.data)])
            norm = v.norm2()
            if norm <= pivot_tolerance(a.norm2(), max(m, n)):
                raise SingularMatrixError("Linearly dependent columns in Gram-Schmidt")
            self.R.data[j][j] = norm
            qj = Vector([vi / norm for vi in v.data])
            Qcols.append(qj)
            for i in range(m):
                self.Q.data[i][j] = qj[i]


class QRModifiedGramSchmidt:
    """Modified Gram-Schmidt orthogonalization (thin QR: Q is m x n, R is n x n)."""

    def __init__(self, A: Matrix):
        self.m, self.n = A.shape()
        self.Q = Matrix.zeros(self.m, self.n)
        self.R = Matrix.zeros(self.n, self.n)
        self._decompose(A)

    def _decompose(self, A: Matrix) -> None:
        m, n = self.m, self.n
        V = [A.col(j).data for j in range(n)]
        col_norms = [sum(v * v for v in col) ** 0.5 for col in V]
        for i in range(n):
            vi = Vector(V[i])
            norm = vi.norm2()
            if norm <= pivot_tolerance(col_norms[i], max(m, n)):
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
    """Stable QR decomposition using Householder reflectors (full QR: Q is m x m, R is m x n)."""

    def __init__(self, A: Matrix):
        self.m, self.n = A.shape()
        self.R = A.copy()
        self.Q = Matrix.identity(self.m)
        self._decompose()

    def _decompose(self) -> None:
        m, n = self.m, self.n
        R, Q = self.R.data, self.Q.data
        for k in range(min(m, n)):
            x = [R[i][k] for i in range(k, m)]
            normx = sum(xi * xi for xi in x) ** 0.5
            if normx == 0.0:
                continue  # column already zero below (and on) the diagonal
            sign = 1.0 if x[0] >= 0 else -1.0
            u1 = x[0] + sign * normx
            v = [xi / u1 if i > 0 else 1.0 for i, xi in enumerate(x)]
            normv = sum(vi * vi for vi in v) ** 0.5
            v = [vi / normv for vi in v]
            # R <- H R for the remaining columns; column k is set exactly below.
            for j in range(k + 1, n):
                s = sum(v[i] * R[k + i][j] for i in range(len(v)))
                for i in range(len(v)):
                    R[k + i][j] -= 2 * s * v[i]
            R[k][k] = -sign * normx
            for i in range(k + 1, m):
                R[i][k] = 0.0
            # Q <- Q H
            for j in range(m):
                s = sum(v[i] * Q[j][k + i] for i in range(len(v)))
                for i in range(len(v)):
                    Q[j][k + i] -= 2 * s * v[i]


def _qr_least_squares(Q: Matrix, R: Matrix, b: Vector) -> Vector:
    """Solve R[:n,:n] x = (Q^T b)[:n] for a thin or full QR factorization."""
    m, n = Q.m, R.n
    if len(b) != m:
        raise ValueError("Dimension mismatch: len(b) must equal the number of rows of A")
    if m < n:
        raise DomainError("QR solve needs at least as many rows as columns (m >= n)")
    Qtb = Vector([sum(Q.data[i][j] * b[i] for i in range(m)) for j in range(n)])
    Rtop = Matrix._from_rows([R.data[i][:n] for i in range(n)], n)
    try:
        return backward_substitution(Rtop, Qtb)
    except SingularMatrixError:
        raise SingularMatrixError(
            "A is rank deficient (R has a zero pivot); least squares solution is not unique"
        ) from None


class QRSolver:
    """Solve Ax = b given a QR factorization of A.

    Square A gives the exact solution; for a tall A (m > n) the result is the
    least-squares solution. Works with thin (Gram-Schmidt) and full
    (Householder) factorizations.
    """

    def __init__(self, qr: QRHouseholder | QRGramSchmidt | QRModifiedGramSchmidt):
        self.Q, self.R = qr.Q, qr.R

    def solve(self, b: Vector) -> Vector:
        return _qr_least_squares(self.Q, self.R, b)


class LeastSquaresSolver:
    """Solve overdetermined system Ax ≈ b in least squares sense using Householder QR."""

    def __init__(self, A: Matrix, b: Vector):
        if A.m < A.n:
            raise DomainError(
                f"Least squares needs m >= n (got {A.m} equations for {A.n} unknowns)"
            )
        if len(b) != A.m:
            raise ValueError("Dimension mismatch: len(b) must equal the number of rows of A")
        self.A, self.b = A, b

    def solve(self) -> Vector:
        qr = QRHouseholder(self.A)
        return _qr_least_squares(qr.Q, qr.R, self.b)
