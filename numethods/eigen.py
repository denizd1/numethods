from __future__ import annotations
import math
import random
from typing import List, Optional, Tuple, Union
from .linalg import Matrix, Vector
from .orthogonal import QRHouseholder
from .solvers import LUDecomposition
from .exceptions import NonSquareMatrixError, ConvergenceError, SingularMatrixError
from .utils import EPS, max_abs, pivot_tolerance, relative_tolerance_reached


def solve_linear(M: Matrix, b: Vector) -> Vector:
    """Solve Mx = b using LU decomposition."""
    solver = LUDecomposition(M)
    return solver.solve(b)


def _shifted(A: Matrix, shift: float) -> Matrix:
    """Return A - shift * I."""
    n = A.n
    return Matrix._from_rows(
        [[A.data[i][j] - (shift if i == j else 0.0) for j in range(n)] for i in range(n)],
        n,
    )


def _start_vector(n: int, x0: Optional[Vector]) -> Vector:
    """Normalized starting vector.

    The default is a fixed pseudo-random vector (reproducible results). A
    constant vector such as [1, ..., 1] is avoided on purpose: it is an exact
    eigenvector of many structured matrices and then hides the dominant one.
    """
    if x0 is None:
        rng = random.Random(2024)
        x = Vector([0.5 + rng.random() for _ in range(n)])
    else:
        x = Vector(x0)
        if len(x) != n:
            raise ValueError("x0 has the wrong length")
    nrm = x.norm2()
    if nrm == 0.0:
        raise ValueError("x0 must be a nonzero vector")
    return x / nrm


class PowerIteration:
    def __init__(self, A: Matrix, tol: float = 1e-10, max_iter: int = 5000):
        if not A.is_square():
            raise NonSquareMatrixError("A must be square")
        self.A, self.tol, self.max_iter = A, tol, max_iter
        self.history = []

    def solve(self, x0: Vector | None = None) -> tuple[float, Vector]:
        x = _start_vector(self.A.n, x0)
        Ax = self.A @ x
        lam_old = 0.0
        self.history.clear()

        for k in range(self.max_iter):
            nrm = Ax.norm2()
            if nrm == 0.0:
                raise ConvergenceError("Zero vector encountered (x lies in the null space of A)")
            x = (1.0 / nrm) * Ax
            Ax = self.A @ x  # one matrix-vector product per iteration, reused next step
            lam = x.dot(Ax)  # Rayleigh quotient (x has unit norm)
            err = abs(lam - lam_old)

            self.history.append({"iter": k, "lambda": lam, "error": err})

            if relative_tolerance_reached(err, lam, self.tol):
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
    """Inverse (shifted) power iteration: eigenvalue of A closest to ``shift``.

    A - shift*I is factorized once and the LU factors are reused in every
    iteration. If ``shift`` is (numerically) an eigenvalue, it is perturbed by
    a tiny amount so the factorization exists; the iteration then converges
    in one or two steps. The shift actually used is stored in
    ``effective_shift``.
    """

    def __init__(
        self, A: Matrix, shift: float = 0.0, tol: float = 1e-10, max_iter: int = 5000
    ):
        if not A.is_square():
            raise NonSquareMatrixError("A must be square")
        self.A, self.shift, self.tol, self.max_iter = A, shift, tol, max_iter
        self.effective_shift = shift
        self.history = []

    def _factorize(self) -> LUDecomposition:
        try:
            self.effective_shift = self.shift
            return LUDecomposition(_shifted(self.A, self.shift))
        except SingularMatrixError:
            delta = math.sqrt(EPS) * max(1.0, abs(self.shift), max_abs(self.A.data))
            self.effective_shift = self.shift + delta
            return LUDecomposition(_shifted(self.A, self.effective_shift))

    def solve(self, x0: Vector | None = None) -> tuple[float, Vector]:
        x = _start_vector(self.A.n, x0)
        lu = self._factorize()
        mu_old = None
        self.history.clear()

        for k in range(self.max_iter):
            y = lu.solve(x)
            nrm = y.norm2()
            if nrm == 0.0 or not math.isfinite(nrm):
                raise ConvergenceError("Inverse iteration produced an invalid vector")
            x = (1.0 / nrm) * y
            mu = x.dot(self.A @ x)  # Rayleigh quotient (x has unit norm)
            err = abs(mu - mu_old) if mu_old is not None else float("inf")

            self.history.append({"iter": k, "mu": mu, "error": err})

            if (mu_old is not None) and relative_tolerance_reached(err, mu, self.tol):
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
        x = _start_vector(self.A.n, x0)
        mu = x.dot(self.A @ x)
        self.history.clear()

        for k in range(self.max_iter):
            try:
                y = solve_linear(_shifted(self.A, mu), x)
            except SingularMatrixError:
                # A - mu I is singular to working precision: mu is an eigenvalue
                # and x its eigenvector (this is how RQI normally finishes).
                self.history.append({"iter": k, "mu": mu, "error": 0.0})
                return mu, x
            nrm = y.norm2()
            if nrm == 0.0 or not math.isfinite(nrm):
                raise ConvergenceError("Rayleigh quotient iteration produced an invalid vector")
            x = (1.0 / nrm) * y
            mu_new = x.dot(self.A @ x)  # x has unit norm
            err = abs(mu_new - mu)

            self.history.append({"iter": k, "mu": mu_new, "error": err})

            if relative_tolerance_reached(err, mu_new, self.tol):
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


# ---------------------------------------------------------------------------
# QR algorithm
# ---------------------------------------------------------------------------


def _house(vec: List[float]) -> Optional[Tuple[List[float], float, float]]:
    """Householder vector v (v[0] = 1), beta = 2 / v.v, and the image -sign*||vec||."""
    alpha = math.sqrt(sum(v * v for v in vec))
    if alpha == 0.0:
        return None
    sign = 1.0 if vec[0] >= 0 else -1.0
    u0 = vec[0] + sign * alpha
    v = [1.0] + [vi / u0 for vi in vec[1:]]
    beta = 2.0 / sum(vi * vi for vi in v)
    return v, beta, -sign * alpha


def _apply_left(H: List[List[float]], v, beta, r0: int, c0: int, c1: int) -> None:
    """Rows r0..r0+len(v)-1, columns c0..c1-1:  H <- (I - beta v v^T) H."""
    rows = [H[r0 + i] for i in range(len(v))]
    for j in range(c0, c1):
        s = beta * sum(vi * row[j] for vi, row in zip(v, rows))
        if s != 0.0:
            for vi, row in zip(v, rows):
                row[j] -= s * vi


def _apply_right(H: List[List[float]], v, beta, c0: int, r0: int, r1: int) -> None:
    """Rows r0..r1-1, columns c0..c0+len(v)-1:  H <- H (I - beta v v^T)."""
    for i in range(r0, r1):
        row = H[i]
        s = beta * sum(row[c0 + j] * vj for j, vj in enumerate(v))
        if s != 0.0:
            for j, vj in enumerate(v):
                row[c0 + j] -= s * vj


def _hessenberg(H: List[List[float]]) -> None:
    """In-place reduction to upper Hessenberg form by Householder similarities."""
    n = len(H)
    for k in range(n - 2):
        h = _house([H[i][k] for i in range(k + 1, n)])
        if h is None:
            continue
        v, beta, top = h
        _apply_left(H, v, beta, k + 1, k + 1, n)
        _apply_right(H, v, beta, k + 1, 0, n)
        H[k + 1][k] = top
        for i in range(k + 2, n):
            H[i][k] = 0.0


def _eig2(a: float, b: float, c: float, d: float) -> Tuple[complex, complex]:
    """Eigenvalues of [[a, b], [c, d]] (real pair or complex conjugate pair)."""
    half_tr = 0.5 * (a + d)
    p = 0.5 * (a - d)
    disc = p * p + b * c
    if disc >= 0.0:
        s = math.sqrt(disc)
        l1 = half_tr + (s if half_tr >= 0 else -s)  # larger magnitude, no cancellation
        det = a * d - b * c
        l2 = det / l1 if l1 != 0.0 else half_tr - (s if half_tr >= 0 else -s)
        return l1, l2
    s = math.sqrt(-disc)
    return complex(half_tr, s), complex(half_tr, -s)


def _standardize_2x2(H: List[List[float]], p: int) -> None:
    """Rotate a converged 2x2 diagonal block (rows/cols p-1, p) to upper
    triangular form when its eigenvalues are real (full Schur update)."""
    n = len(H)
    a, b = H[p - 1][p - 1], H[p - 1][p]
    c, d = H[p][p - 1], H[p][p]
    if c == 0.0:
        return
    lam, _ = _eig2(a, b, c, d)
    if isinstance(lam, complex):
        return  # complex conjugate pair: keep the 2x2 block
    v1 = (lam - d, c)
    v2 = (b, lam - a)
    v = v1 if abs(v1[0]) + abs(v1[1]) >= abs(v2[0]) + abs(v2[1]) else v2
    nrm = math.hypot(v[0], v[1])
    if nrm == 0.0:
        return
    cs, sn = v[0] / nrm, v[1] / nrm
    for j in range(p - 1, n):
        h1, h2 = H[p - 1][j], H[p][j]
        H[p - 1][j] = cs * h1 + sn * h2
        H[p][j] = -sn * h1 + cs * h2
    for i in range(p + 1):
        h1, h2 = H[i][p - 1], H[i][p]
        H[i][p - 1] = cs * h1 + sn * h2
        H[i][p] = -sn * h1 + cs * h2
    H[p][p - 1] = 0.0


def _francis_schur(H: List[List[float]], max_iter: int) -> int:
    """In-place Francis double-shift QR iteration on an upper Hessenberg matrix.

    On return H is in real Schur form (quasi upper triangular: 1x1 blocks for
    real eigenvalues, 2x2 blocks for complex conjugate pairs). Returns the
    number of QR sweeps performed.
    """
    n = len(H)
    norm = max_abs(H)
    p = n - 1
    total = 0
    its = 0
    while p >= 0:
        # Look for a negligible subdiagonal entry H[l][l-1] (deflation).
        l = p
        while l > 0:
            s = abs(H[l - 1][l - 1]) + abs(H[l][l])
            if s == 0.0:
                s = norm
            if abs(H[l][l - 1]) <= EPS * s:
                H[l][l - 1] = 0.0
                break
            l -= 1
        if l == p:  # 1x1 block converged
            p -= 1
            its = 0
            continue
        if l == p - 1:  # 2x2 block converged
            _standardize_2x2(H, p)
            p -= 2
            its = 0
            continue
        if total >= max_iter:
            raise ConvergenceError("QR algorithm did not converge within max_iter")
        its += 1
        total += 1

        # Shift polynomial (s1 + s2 = tr, s1 * s2 = det).
        if its in (10, 20):  # exceptional shift to break cycles
            ex = abs(H[p][p - 1]) + abs(H[p - 1][p - 2])
            h = 0.75 * ex + H[p][p]
            tr = 2.0 * h
            det = h * h + 0.4375 * ex * ex
        else:
            tr = H[p - 1][p - 1] + H[p][p]
            det = H[p - 1][p - 1] * H[p][p] - H[p - 1][p] * H[p][p - 1]

        # First column of (H - s1 I)(H - s2 I) restricted to the active block.
        x = H[l][l] * H[l][l] + H[l][l + 1] * H[l + 1][l] - tr * H[l][l] + det
        y = H[l + 1][l] * (H[l][l] + H[l + 1][l + 1] - tr)
        z = H[l + 1][l] * H[l + 2][l + 1]
        # Chase the bulge down the active block l..p.
        for k in range(l, p - 1):
            h = _house([x, y, z])
            if h is not None:
                v, beta, top = h
                c0 = max(l, k - 1)
                _apply_left(H, v, beta, k, c0, n)
                _apply_right(H, v, beta, k, 0, min(k + 4, p + 1))
                if k > l:
                    H[k][k - 1] = top
                    H[k + 1][k - 1] = 0.0
                    H[k + 2][k - 1] = 0.0
            x = H[k + 1][k]
            y = H[k + 2][k]
            if k < p - 2:
                z = H[k + 3][k]
        h = _house([x, y])
        if h is not None:
            v, beta, top = h
            _apply_left(H, v, beta, p - 1, p - 2, n)
            _apply_right(H, v, beta, p - 1, 0, p + 1)
            if p - 2 >= l:
                H[p - 1][p - 2] = top
                H[p][p - 2] = 0.0
    return total


def _sort_eigenvalues(vals: List[Union[float, complex]]) -> List[Union[float, complex]]:
    """Descending modulus, then descending real part, then descending imaginary part."""
    return sorted(vals, key=lambda z: (-abs(z), -z.real, -(z.imag if isinstance(z, complex) else 0.0)))


class QREigenvalues:
    """All eigenvalues of a square matrix with the QR algorithm.

    shifted=True (default): Hessenberg reduction followed by Francis
        double-shift QR steps with deflation. Each sweep costs O(n^2), the
        convergence is quadratic, and complex conjugate eigenvalue pairs are
        handled (they appear as 2x2 blocks of the real Schur form).
        The deflation test works at machine precision; ``tol`` is not used.
    shifted=False: the basic unshifted iteration A_{k+1} = R_k Q_k (educational;
        linear convergence, O(n^3) per iteration, it stops when the strictly
        lower part is below ``tol * ||A||_F`` and fails for complex
        eigenvalues or eigenvalues of equal modulus such as +1 and -1).

    ``solve()`` returns the final (quasi) upper triangular matrix,
    ``eigenvalues()`` returns the eigenvalues sorted by decreasing modulus
    (complex numbers for complex pairs).
    """

    def __init__(
        self, A: Matrix, tol: float = 1e-10, max_iter: int = 10000, shifted: bool = True
    ):
        if not A.is_square():
            raise NonSquareMatrixError("A must be square")
        self.A0, self.tol, self.max_iter = A.copy(), tol, max_iter
        self.shifted = shifted
        self.T: Optional[Matrix] = None
        self.iterations = 0

    def solve(self) -> Matrix:
        if self.shifted:
            H = [row[:] for row in self.A0.data]
            _hessenberg(H)
            self.iterations = _francis_schur(H, self.max_iter)
            self.T = Matrix._from_rows(H, self.A0.n)
            return self.T
        A = self.A0.copy()
        n = A.n
        scale = A.norm_fro()
        for k in range(self.max_iter):
            qr = QRHouseholder(A)
            A = qr.R @ qr.Q
            off = 0.0
            for i in range(1, n):
                off += sum(abs(A.data[i][j]) for j in range(0, i))
            if off <= self.tol * scale:
                self.iterations = k + 1
                self.T = A
                return A
        raise ConvergenceError(
            "Unshifted QR did not converge (complex or equal-modulus eigenvalues?); "
            "try shifted=True"
        )

    def eigenvalues(self) -> List[Union[float, complex]]:
        T = self.T if self.T is not None else self.solve()
        n = T.n
        if not self.shifted:
            return _sort_eigenvalues([T.data[i][i] for i in range(n)])
        vals: List[Union[float, complex]] = []
        i = 0
        while i < n:
            if i + 1 < n and T.data[i + 1][i] != 0.0:
                l1, l2 = _eig2(T.data[i][i], T.data[i][i + 1], T.data[i + 1][i], T.data[i + 1][i + 1])
                vals.extend([l1, l2])
                i += 2
            else:
                vals.append(T.data[i][i])
                i += 1
        return _sort_eigenvalues(vals)


# ---------------------------------------------------------------------------
# Singular value decomposition
# ---------------------------------------------------------------------------


def _jacobi_columns(W: List[List[float]], V: Optional[List[List[float]]], tol: float, max_sweeps: int) -> int:
    """One-sided (Hestenes) Jacobi: rotate column pairs of W until they are
    mutually orthogonal. W and V are stored as lists of columns. Returns the
    number of sweeps."""
    n = len(W)
    for sweep in range(1, max_sweeps + 1):
        rotated = False
        for i in range(n - 1):
            wi = W[i]
            for j in range(i + 1, n):
                wj = W[j]
                alpha = sum(a * a for a in wi)
                beta = sum(b * b for b in wj)
                gamma = sum(a * b for a, b in zip(wi, wj))
                if gamma == 0.0 or abs(gamma) <= tol * math.sqrt(alpha) * math.sqrt(beta):
                    continue
                rotated = True
                zeta = (beta - alpha) / (2.0 * gamma)
                t = (1.0 if zeta >= 0 else -1.0) / (abs(zeta) + math.sqrt(1.0 + zeta * zeta))
                c = 1.0 / math.sqrt(1.0 + t * t)
                s = c * t
                W[i], W[j] = (
                    [c * a - s * b for a, b in zip(wi, wj)],
                    [s * a + c * b for a, b in zip(wi, wj)],
                )
                wi = W[i]
                if V is not None:
                    vi, vj = V[i], V[j]
                    V[i], V[j] = (
                        [c * a - s * b for a, b in zip(vi, vj)],
                        [s * a + c * b for a, b in zip(vi, vj)],
                    )
        if not rotated:
            return sweep
    raise ConvergenceError("Jacobi SVD did not converge within max_iter sweeps")


class SVD:
    """Thin singular value decomposition A = U diag(sigma) V^T.

    Computed with one-sided Jacobi rotations applied directly to the columns
    of A (mathematically the Jacobi eigenvalue method for A^T A, but A^T A is
    never formed, so the condition number is not squared).

    Shapes (k = min(m, n)): U is m x k, sigma has k entries in decreasing
    order, V is n x k. U and V have orthonormal columns, also for
    rank-deficient A (columns for zero singular values are completed to an
    orthonormal set).

    tol: rotation threshold |a_i . a_j| <= tol ||a_i|| ||a_j||.
    max_iter: maximum number of Jacobi sweeps.
    """

    def __init__(self, A: Matrix, tol: float = 1e-12, max_iter: int = 100):
        self.A = A
        self.tol, self.max_iter = tol, max_iter
        self.sweeps = 0

    def singular_values(self) -> List[float]:
        """Singular values in decreasing order (V is not accumulated)."""
        A = self.A if self.A.m >= self.A.n else self.A.T
        W = [A.col(j).data for j in range(A.n)]
        self.sweeps = _jacobi_columns(W, None, self.tol, self.max_iter)
        return sorted((math.sqrt(sum(w * w for w in col)) for col in W), reverse=True)

    def solve(self) -> tuple[Matrix, Vector, Matrix]:
        m, n = self.A.m, self.A.n
        transposed = m < n
        A = self.A.T if transposed else self.A
        rows, k = A.m, A.n  # rows >= k
        W = [A.col(j).data for j in range(k)]
        Vc = [[1.0 if i == j else 0.0 for i in range(k)] for j in range(k)]
        self.sweeps = _jacobi_columns(W, Vc, self.tol, self.max_iter)

        sig = [math.sqrt(sum(w * w for w in col)) for col in W]
        order = sorted(range(k), key=lambda i: sig[i], reverse=True)
        sig = [sig[i] for i in order]
        W = [W[i] for i in order]
        Vc = [Vc[i] for i in order]

        thresh = pivot_tolerance(sig[0] if sig else 0.0, max(rows, k))
        Uc: List[Optional[List[float]]] = [
            [w / s for w in col] if s > thresh else None for col, s in zip(W, sig)
        ]
        self._complete_orthonormal(Uc, rows)

        U = Matrix._from_rows([[Uc[j][i] for j in range(k)] for i in range(rows)], k)
        V = Matrix._from_rows([[Vc[j][i] for j in range(k)] for i in range(k)], k)
        Sigma = Vector(sig)
        if transposed:
            return V, Sigma, U
        return U, Sigma, V

    @staticmethod
    def _complete_orthonormal(cols: List[Optional[List[float]]], m: int) -> None:
        """Replace None entries by unit vectors orthogonal to all other columns."""
        for idx, col in enumerate(cols):
            if col is not None:
                continue
            basis = [c for c in cols if c is not None]
            best, best_norm = None, -1.0
            for e in range(m):
                v = [1.0 if i == e else 0.0 for i in range(m)]
                for _ in range(2):  # Gram-Schmidt, repeated once for stability
                    for q in basis:
                        r = sum(a * b for a, b in zip(q, v))
                        v = [a - r * b for a, b in zip(v, q)]
                nv = math.sqrt(sum(a * a for a in v))
                if nv > best_norm:
                    best, best_norm = v, nv
            cols[idx] = [a / best_norm for a in best]
