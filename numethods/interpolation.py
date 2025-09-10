from __future__ import annotations
from typing import List

class NewtonInterpolation:
    """Polynomial interpolation via divided differences."""
    def __init__(self, x: List[float], y: List[float]):
        if len(x) != len(y):
            raise ValueError("x and y must have same length")
        if len(set(x)) != len(x):
            raise ValueError("x values must be distinct")
        self.x = [float(v) for v in x]
        self.coeffs = self._divided_differences(self.x, [float(v) for v in y])

    def _divided_differences(self, x: List[float], y: List[float]) -> List[float]:
        n = len(x)
        coef = y[:]
        for j in range(1, n):
            for i in range(n-1, j-1, -1):
                denom = x[i] - x[i-j]
                if abs(denom) < 1e-20:
                    raise ZeroDivisionError("Repeated x values in divided differences")
                coef[i] = (coef[i] - coef[i-1]) / denom
        return coef

    def evaluate(self, t: float) -> float:
        n = len(self.x)
        result = 0.0
        for i in reversed(range(n)):
            result = result * (t - self.x[i]) + self.coeffs[i]
        return result


class LagrangeInterpolation:
    """Lagrange-form polynomial interpolation."""
    def __init__(self, x: List[float], y: List[float]):
        if len(x) != len(y):
            raise ValueError("x and y must have same length")
        if len(set(x)) != len(x):
            raise ValueError("x values must be distinct")
        self.x = [float(v) for v in x]
        self.y = [float(v) for v in y]

    def evaluate(self, t: float) -> float:
        x, y = self.x, self.y
        n = len(x)
        total = 0.0
        for i in range(n):
            L = 1.0
            for j in range(n):
                if i == j:
                    continue
                denom = x[i] - x[j]
                if abs(denom) < 1e-20:
                    raise ZeroDivisionError("Repeated x values in Lagrange basis")
                L *= (t - x[j]) / denom
            total += y[i]*L
        return total
