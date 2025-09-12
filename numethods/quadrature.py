from __future__ import annotations
from typing import Callable
import math


class Quadrature:
    """Base class for numerical integration on [a, b]."""

    def __init__(self, f: Callable[[float], float], a: float, b: float, n: int = 100):
        self.f = f
        self.a = a
        self.b = b
        self.n = n

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


class GaussLegendre(Quadrature):
    """Gauss–Legendre quadrature (supports 2- and 3-point)."""

    def __init__(self, f, a, b, n=2):
        super().__init__(f, a, b, n)

    def integrate(self) -> float:
        if self.n == 2:
            nodes = [-1 / math.sqrt(3), 1 / math.sqrt(3)]
            weights = [1, 1]
        elif self.n == 3:
            nodes = [-math.sqrt(3 / 5), 0.0, math.sqrt(3 / 5)]
            weights = [5 / 9, 8 / 9, 5 / 9]
        else:
            raise NotImplementedError("Only 2- and 3-point Gauss-Legendre supported")

        mid = 0.5 * (self.a + self.b)
        half = 0.5 * (self.b - self.a)
        return half * sum(w * self.f(mid + half * x) for x, w in zip(nodes, weights))
