from __future__ import annotations
from functools import lru_cache
from numbers import Integral
from typing import Callable, List, Tuple
import math
import warnings
from .exceptions import ConvergenceError
from .utils import EPS, relative_tolerance_reached


class Quadrature:
    """Base class for numerical integration on [a, b] with n subintervals/points."""

    def __init__(self, f: Callable[[float], float], a: float, b: float, n: int = 100):
        if not isinstance(n, Integral) or n < 1:
            raise ValueError("n must be a positive integer")
        self.f = f
        self.a = a
        self.b = b
        self.n = int(n)

    def integrate(self) -> float:
        raise NotImplementedError


class Trapezoidal(Quadrature):
    """Composite trapezoidal rule."""

    def integrate(self) -> float:
        h = (self.b - self.a) / self.n
        s = 0.5 * (self.f(self.a) + self.f(self.b))
        for i in range(1, self.n):
            s += self.f(self.a + i * h)
        return h * s


class Simpson(Quadrature):
    """Composite Simpson’s rule (n must be even)."""

    def integrate(self) -> float:
        if self.n % 2 != 0:
            raise ValueError("n must be even for Simpson's rule")
        h = (self.b - self.a) / self.n
        s = self.f(self.a) + self.f(self.b)
        for i in range(1, self.n):
            coef = 4 if i % 2 == 1 else 2
            s += coef * self.f(self.a + i * h)
        return h * s / 3.0


@lru_cache(maxsize=None)
def gauss_legendre_nodes(n: int) -> Tuple[Tuple[float, ...], Tuple[float, ...]]:
    """Nodes (increasing) and weights of the n-point Gauss–Legendre rule on [-1, 1].

    The nodes are the roots of the Legendre polynomial P_n, found by Newton's
    method from Chebyshev-like initial guesses; w_i = 2 / ((1 - x_i^2) P_n'(x_i)^2).
    """
    if not isinstance(n, Integral) or n < 1:
        raise ValueError("n must be a positive integer")
    x = [0.0] * n
    w = [0.0] * n
    for i in range((n + 1) // 2):
        z = math.cos(math.pi * (i + 0.75) / (n + 0.5))
        for _ in range(100):
            p1, p2 = 1.0, 0.0
            for j in range(1, n + 1):
                p1, p2 = ((2 * j - 1) * z * p1 - (j - 1) * p2) / j, p1
            dp = n * (z * p1 - p2) / (z * z - 1.0)
            dz = p1 / dp
            z -= dz
            if abs(dz) <= 4 * EPS:
                break
        # recompute P_n'(z) at the converged node for the weight
        p1, p2 = 1.0, 0.0
        for j in range(1, n + 1):
            p1, p2 = ((2 * j - 1) * z * p1 - (j - 1) * p2) / j, p1
        dp = n * (z * p1 - p2) / (z * z - 1.0)
        x[i], x[n - 1 - i] = -z, z
        w[i] = w[n - 1 - i] = 2.0 / ((1.0 - z * z) * dp * dp)
    if n % 2 == 1:
        x[n // 2] = 0.0
    return tuple(x), tuple(w)


class GaussLegendre(Quadrature):
    """Gauss–Legendre quadrature with ``n`` points (any n >= 1).

    ``panels > 1`` gives the composite rule: [a, b] is split into ``panels``
    equal subintervals and the n-point rule is applied on each one
    (error O(h^{2n}) for smooth f).
    """

    def __init__(self, f, a, b, n=2, panels: int = 1):
        super().__init__(f, a, b, n)
        if not isinstance(panels, Integral) or panels < 1:
            raise ValueError("panels must be a positive integer")
        self.panels = int(panels)

    def integrate(self) -> float:
        nodes, weights = gauss_legendre_nodes(self.n)
        h = (self.b - self.a) / self.panels
        half = 0.5 * h
        total = 0.0
        for k in range(self.panels):
            mid = self.a + (k + 0.5) * h
            total += half * sum(w * self.f(mid + half * x) for x, w in zip(nodes, weights))
        return total


class AdaptiveSimpson:
    """Adaptive Simpson quadrature with local Richardson correction.

    An interval is accepted when |S(left) + S(right) - S(whole)| <= 15 tol;
    otherwise it is split and the tolerance is halved. If ``max_depth`` is
    reached somewhere, the result is still returned, ``max_depth_reached`` is
    set and a RuntimeWarning is issued.
    """

    def __init__(
        self,
        f: Callable[[float], float],
        a: float,
        b: float,
        tol: float = 1e-10,
        max_depth: int = 50,
    ):
        self.f, self.a, self.b = f, a, b
        self.tol, self.max_depth = tol, max_depth
        self.evaluations = 0
        self.max_depth_reached = False

    def _f(self, x: float) -> float:
        self.evaluations += 1
        return self.f(x)

    def integrate(self) -> float:
        self.evaluations = 0
        self.max_depth_reached = False
        a, b = self.a, self.b
        fa, fb = self._f(a), self._f(b)
        m = 0.5 * (a + b)
        fm = self._f(m)
        whole = (b - a) / 6.0 * (fa + 4.0 * fm + fb)
        result = self._recurse(a, b, fa, fm, fb, whole, self.tol, self.max_depth)
        if self.max_depth_reached:
            warnings.warn(
                "AdaptiveSimpson reached max_depth; the requested tolerance may not be met",
                RuntimeWarning,
                stacklevel=2,
            )
        return result

    def _recurse(self, a, b, fa, fm, fb, whole, tol, depth) -> float:
        m = 0.5 * (a + b)
        lm, rm = 0.5 * (a + m), 0.5 * (m + b)
        flm, frm = self._f(lm), self._f(rm)
        left = (m - a) / 6.0 * (fa + 4.0 * flm + fm)
        right = (b - m) / 6.0 * (fm + 4.0 * frm + fb)
        delta = left + right - whole
        if abs(delta) <= 15.0 * tol:
            return left + right + delta / 15.0
        if depth <= 0:
            self.max_depth_reached = True
            return left + right + delta / 15.0
        return self._recurse(a, m, fa, flm, fm, left, 0.5 * tol, depth - 1) + self._recurse(
            m, b, fm, frm, fb, right, 0.5 * tol, depth - 1
        )


class Romberg:
    """Romberg integration: Richardson extrapolation of the trapezoidal rule.

    R[k][0] is the trapezoidal rule with 2^k subintervals (reusing previous
    function values) and R[k][j] = (4^j R[k][j-1] - R[k-1][j-1]) / (4^j - 1).
    Stops when |R[k][k] - R[k-1][k-1]| <= tol (1 + |R[k][k]|), k >= 2.
    The full table is kept in ``table``.
    """

    def __init__(
        self,
        f: Callable[[float], float],
        a: float,
        b: float,
        tol: float = 1e-10,
        max_levels: int = 20,
    ):
        if max_levels < 2:
            raise ValueError("max_levels must be at least 2")
        self.f, self.a, self.b = f, a, b
        self.tol, self.max_levels = tol, max_levels
        self.table: List[List[float]] = []

    def integrate(self) -> float:
        f, a, b = self.f, self.a, self.b
        h = b - a
        R = [[0.5 * h * (f(a) + f(b))]]
        self.table = R
        for k in range(1, self.max_levels + 1):
            h *= 0.5
            n_new = 2 ** (k - 1)
            s = sum(f(a + (2 * i + 1) * h) for i in range(n_new))
            row = [0.5 * R[k - 1][0] + h * s]
            for j in range(1, k + 1):
                p = 4.0**j
                row.append((p * row[j - 1] - R[k - 1][j - 1]) / (p - 1.0))
            R.append(row)
            if k >= 2 and relative_tolerance_reached(row[k] - R[k - 1][k - 1], row[k], self.tol):
                return row[k]
        raise ConvergenceError("Romberg integration did not converge within max_levels")
