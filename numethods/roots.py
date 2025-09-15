from __future__ import annotations
from typing import Callable, List, Dict, Any
from .exceptions import ConvergenceError, DomainError


class Bisection:
    def __init__(
        self,
        f: Callable[[float], float],
        a: float,
        b: float,
        tol: float = 1e-10,
        max_iter: int = 10_000,
    ):
        if a >= b:
            raise ValueError("Require a < b")
        fa, fb = f(a), f(b)
        if fa * fb > 0:
            raise DomainError("f(a) and f(b) must have opposite signs")
        self.f, self.a, self.b = f, a, b
        self.tol, self.max_iter = tol, max_iter

    def solve(self) -> float:
        a, b, f = self.a, self.b, self.f
        fa, fb = f(a), f(b)
        for _ in range(self.max_iter):
            c = 0.5 * (a + b)
            fc = f(c)
            if abs(fc) <= self.tol or 0.5 * (b - a) <= self.tol:
                return c
            if fa * fc < 0:
                b, fb = c, fc
            else:
                a, fa = c, fc
        raise ConvergenceError("Bisection did not converge")

    def trace(self) -> List[Dict[str, Any]]:
        steps = []
        a, b, f = self.a, self.b, self.f
        fa, fb = f(a), f(b)
        for k in range(self.max_iter):
            c = 0.5 * (a + b)
            fc = f(c)
            steps.append(
                {
                    "iter": k,
                    "a": a,
                    "b": b,
                    "c": c,
                    "f(a)": fa,
                    "f(b)": fb,
                    "f(c)": fc,
                    "interval": b - a,
                }
            )
            if abs(fc) <= self.tol or 0.5 * (b - a) <= self.tol:
                return steps
            if fa * fc < 0:
                b, fb = c, fc
            else:
                a, fa = c, fc
        raise ConvergenceError("Bisection did not converge")


class FixedPoint:
    def __init__(
        self,
        g: Callable[[float], float],
        x0: float,
        tol: float = 1e-10,
        max_iter: int = 10_000,
    ):
        self.g, self.x0, self.tol, self.max_iter = g, x0, tol, max_iter

    def solve(self) -> float:
        x = self.x0
        for _ in range(self.max_iter):
            x_new = self.g(x)
            if abs(x_new - x) <= self.tol * (1.0 + abs(x_new)):
                return x_new
            x = x_new
        raise ConvergenceError("Fixed-point iteration did not converge")

    def trace(self) -> List[Dict[str, Any]]:
        steps = []
        x = self.x0
        for k in range(self.max_iter):
            x_new = self.g(x)
            steps.append({"iter": k, "x": x, "x_new": x_new, "error": abs(x_new - x)})
            if abs(x_new - x) <= self.tol * (1.0 + abs(x_new)):
                return steps
            x = x_new
        raise ConvergenceError("Fixed-point iteration did not converge")


class Secant:
    def __init__(
        self,
        f: Callable[[float], float],
        x0: float,
        x1: float,
        tol: float = 1e-10,
        max_iter: int = 10_000,
    ):
        self.f, self.x0, self.x1, self.tol, self.max_iter = f, x0, x1, tol, max_iter

    def solve(self) -> float:
        x0, x1, f = self.x0, self.x1, self.f
        f0, f1 = f(x0), f(x1)
        for _ in range(self.max_iter):
            denom = f1 - f0
            if abs(denom) < 1e-20:
                raise ConvergenceError("Secant encountered nearly zero denominator")
            x2 = x1 - f1 * (x1 - x0) / denom
            if abs(x2 - x1) <= self.tol * (1.0 + abs(x2)):
                return x2
            x0, x1 = x1, x2
            f0, f1 = f1, f(x1)
        raise ConvergenceError("Secant did not converge")

    def trace(self) -> List[Dict[str, Any]]:
        steps = []
        x0, x1, f = self.x0, self.x1, self.f
        f0, f1 = f(x0), f(x1)
        for k in range(self.max_iter):
            denom = f1 - f0
            if abs(denom) < 1e-20:
                raise ConvergenceError("Secant encountered nearly zero denominator")
            x2 = x1 - f1 * (x1 - x0) / denom
            steps.append(
                {
                    "iter": k,
                    "x0": x0,
                    "x1": x1,
                    "x2": x2,
                    "f(x0)": f0,
                    "f(x1)": f1,
                    "error": abs(x2 - x1),
                }
            )
            if abs(x2 - x1) <= self.tol * (1.0 + abs(x2)):
                return steps
            x0, x1 = x1, x2
            f0, f1 = f1, f(x1)
        raise ConvergenceError("Secant did not converge")


class NewtonRoot:
    def __init__(
        self,
        f: Callable[[float], float],
        df: Callable[[float], float],
        x0: float,
        tol: float = 1e-10,
        max_iter: int = 10_000,
    ):
        self.f, self.df, self.x0, self.tol, self.max_iter = f, df, x0, tol, max_iter

    def solve(self) -> float:
        x = self.x0
        for _ in range(self.max_iter):
            dfx = self.df(x)
            if abs(dfx) < 1e-20:
                raise ConvergenceError("Derivative near zero in Newton method")
            x_new = x - self.f(x) / dfx
            if abs(x_new - x) <= self.tol * (1.0 + abs(x_new)):
                return x_new
            x = x_new
        raise ConvergenceError("Newton method did not converge")

    def trace(self) -> List[Dict[str, Any]]:
        steps = []
        x = self.x0
        for k in range(self.max_iter):
            dfx = self.df(x)
            if abs(dfx) < 1e-20:
                raise ConvergenceError("Derivative near zero in Newton method")
            x_new = x - self.f(x) / dfx
            steps.append(
                {
                    "iter": k,
                    "x": x,
                    "f(x)": self.f(x),
                    "df(x)": dfx,
                    "x_new": x_new,
                    "error": abs(x_new - x),
                }
            )
            if abs(x_new - x) <= self.tol * (1.0 + abs(x_new)):
                return steps
            x = x_new
        raise ConvergenceError("Newton method did not converge")


def print_trace(steps: List[Dict[str, Any]]):
    if not steps:
        print("No steps recorded.")
        return
    # Get headers from dict keys
    headers = list(steps[0].keys())
    # Print header
    print(" | ".join(f"{h:>10}" for h in headers))
    print("-" * (13 * len(headers)))
    # Print rows
    for row in steps:
        print(
            " | ".join(
                f"{row[h]:>10.6g}" if isinstance(row[h], (int, float)) else str(row[h])
                for h in headers
            )
        )
