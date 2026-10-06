from __future__ import annotations
import bisect
import math
from typing import List, Optional, Sequence, Tuple


def _validate_nodes(x: Sequence[float], y: Sequence[float]) -> Tuple[List[float], List[float]]:
    if len(x) != len(y):
        raise ValueError("x and y must have same length")
    if len(x) == 0:
        raise ValueError("at least one data point is required")
    xs = [float(v) for v in x]
    ys = [float(v) for v in y]
    if len(set(xs)) != len(xs):
        raise ValueError("x values must be distinct")
    return xs, ys


class NewtonInterpolation:
    """Polynomial interpolation via divided differences (Newton form).

    ``coeffs[i] = f[x_0, ..., x_i]``. ``add_point`` appends a node in O(n)
    using the last diagonal of the divided-difference table.
    """

    def __init__(self, x: List[float], y: List[float]):
        xs, ys = _validate_nodes(x, y)
        self.x: List[float] = []
        self.coeffs: List[float] = []
        self._last: List[float] = []  # _last[j] = f[x_{n-j}, ..., x_n]
        for xi, yi in zip(xs, ys):
            self._append(xi, yi)

    def _append(self, xn: float, yn: float) -> None:
        last = [yn]
        n = len(self.x)
        for j in range(1, n + 1):
            last.append((last[j - 1] - self._last[j - 1]) / (xn - self.x[n - j]))
        self.x.append(xn)
        self._last = last
        self.coeffs.append(last[-1])

    def add_point(self, x: float, y: float) -> None:
        """Add one interpolation node (O(n) update)."""
        x = float(x)
        if x in self.x:
            raise ValueError("x values must be distinct")
        self._append(x, float(y))

    def evaluate(self, t: float) -> float:
        n = len(self.x)
        result = 0.0
        for i in reversed(range(n)):
            result = result * (t - self.x[i]) + self.coeffs[i]
        return result


class LagrangeInterpolation:
    """Lagrange-form polynomial interpolation, evaluated with the (second)
    barycentric formula

        p(t) = sum_i w_i y_i / (t - x_i)  /  sum_i w_i / (t - x_i),
        w_i = 1 / prod_{j != i} (x_i - x_j).

    The weights cost O(n^2) once; each evaluation then costs O(n) and is
    numerically stable. ``add_point`` updates the weights in O(n).
    """

    def __init__(self, x: List[float], y: List[float]):
        self.x, self.y = _validate_nodes(x, y)
        n = len(self.x)
        self.weights = []
        for i in range(n):
            p = 1.0
            for j in range(n):
                if j != i:
                    p *= self.x[i] - self.x[j]
            self.weights.append(1.0 / p)

    def add_point(self, x: float, y: float) -> None:
        """Add one interpolation node (O(n) weight update)."""
        x = float(x)
        if x in self.x:
            raise ValueError("x values must be distinct")
        p = 1.0
        for i, xi in enumerate(self.x):
            self.weights[i] /= xi - x
            p *= x - xi
        self.x.append(x)
        self.y.append(float(y))
        self.weights.append(1.0 / p)

    def basis(self, i: int, t: float) -> float:
        """Value of the Lagrange basis polynomial L_i(t)."""
        L = 1.0
        for j, xj in enumerate(self.x):
            if j != i:
                L *= (t - xj) / (self.x[i] - xj)
        return L

    def evaluate(self, t: float) -> float:
        num = 0.0
        den = 0.0
        for xi, yi, wi in zip(self.x, self.y, self.weights):
            d = t - xi
            if d == 0.0:
                return yi  # t is a node
            q = wi / d
            num += q * yi
            den += q
        return num / den


def chebyshev_nodes(n: int, a: float = -1.0, b: float = 1.0) -> List[float]:
    """The n Chebyshev points of the first kind on [a, b], in increasing order:

        x_k = (a+b)/2 + (b-a)/2 * cos((2k+1) pi / (2n)),  k = 0..n-1.

    Use n = degree + 1 nodes for a polynomial of a given degree.
    """
    if n < 1:
        raise ValueError("n must be at least 1")
    mid, half = 0.5 * (a + b), 0.5 * (b - a)
    return [mid + half * math.cos((2 * k + 1) * math.pi / (2 * n)) for k in reversed(range(n))]


def _solve_tridiagonal(sub: List[float], diag: List[float], sup: List[float], rhs: List[float]) -> List[float]:
    """Thomas algorithm for a diagonally dominant tridiagonal system."""
    n = len(diag)
    c = [0.0] * n
    d = [0.0] * n
    c[0] = sup[0] / diag[0] if n > 1 else 0.0
    d[0] = rhs[0] / diag[0]
    for i in range(1, n):
        denom = diag[i] - sub[i] * c[i - 1]
        c[i] = sup[i] / denom if i < n - 1 else 0.0
        d[i] = (rhs[i] - sub[i] * d[i - 1]) / denom
    x = [0.0] * n
    x[-1] = d[-1]
    for i in range(n - 2, -1, -1):
        x[i] = d[i] - c[i] * x[i + 1]
    return x


class CubicSpline:
    """Interpolating cubic spline (C^2 piecewise cubic).

    bc="natural": S''(x_0) = S''(x_n) = 0.
    bc="clamped": S'(x_0) = fprime[0], S'(x_n) = fprime[1].

    The data are sorted by x. Outside [x_0, x_n] the end cubic pieces are
    extended. ``M`` holds the second derivatives (moments) at the nodes.
    """

    def __init__(
        self,
        x: List[float],
        y: List[float],
        bc: str = "natural",
        fprime: Optional[Tuple[float, float]] = None,
    ):
        xs, ys = _validate_nodes(x, y)
        if len(xs) < 2:
            raise ValueError("a cubic spline needs at least two points")
        if bc not in ("natural", "clamped"):
            raise ValueError("bc must be 'natural' or 'clamped'")
        if bc == "clamped" and (fprime is None or len(fprime) != 2):
            raise ValueError("bc='clamped' needs fprime=(f'(x_0), f'(x_n))")
        order = sorted(range(len(xs)), key=lambda i: xs[i])
        self.x = [xs[i] for i in order]
        self.y = [ys[i] for i in order]
        self.bc = bc
        self.M = self._moments(bc, fprime)

    def _moments(self, bc: str, fprime) -> List[float]:
        x, y = self.x, self.y
        n = len(x) - 1
        h = [x[i + 1] - x[i] for i in range(n)]
        slope = [(y[i + 1] - y[i]) / h[i] for i in range(n)]
        sub = [0.0] * (n + 1)
        diag = [0.0] * (n + 1)
        sup = [0.0] * (n + 1)
        rhs = [0.0] * (n + 1)
        for i in range(1, n):
            sub[i], diag[i], sup[i] = h[i - 1], 2.0 * (h[i - 1] + h[i]), h[i]
            rhs[i] = 6.0 * (slope[i] - slope[i - 1])
        if bc == "natural":
            diag[0] = diag[n] = 1.0
        else:
            d0, dn = float(fprime[0]), float(fprime[1])
            diag[0], sup[0], rhs[0] = 2.0 * h[0], h[0], 6.0 * (slope[0] - d0)
            sub[n], diag[n], rhs[n] = h[n - 1], 2.0 * h[n - 1], 6.0 * (dn - slope[n - 1])
        return _solve_tridiagonal(sub, diag, sup, rhs)

    def _interval(self, t: float) -> int:
        i = bisect.bisect_right(self.x, t) - 1
        return min(max(i, 0), len(self.x) - 2)

    def evaluate(self, t: float) -> float:
        i = self._interval(t)
        x0, x1 = self.x[i], self.x[i + 1]
        h = x1 - x0
        A = (x1 - t) / h
        B = (t - x0) / h
        return (
            A * self.y[i]
            + B * self.y[i + 1]
            + ((A**3 - A) * self.M[i] + (B**3 - B) * self.M[i + 1]) * h * h / 6.0
        )

    def derivative(self, t: float, order: int = 1) -> float:
        """First, second or third derivative of the spline at t."""
        i = self._interval(t)
        x0, x1 = self.x[i], self.x[i + 1]
        h = x1 - x0
        A = (x1 - t) / h
        B = (t - x0) / h
        Mi, Mj = self.M[i], self.M[i + 1]
        if order == 1:
            return (self.y[i + 1] - self.y[i]) / h - (3 * A * A - 1) * h * Mi / 6.0 + (3 * B * B - 1) * h * Mj / 6.0
        if order == 2:
            return A * Mi + B * Mj
        if order == 3:
            return (Mj - Mi) / h
        raise ValueError("order must be 1, 2 or 3")
