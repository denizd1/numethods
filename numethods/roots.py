from __future__ import annotations
import math
from typing import Callable, List, Dict, Any, Optional, Tuple
from .exceptions import ConvergenceError, DomainError
from .utils import EPS, relative_tolerance_reached

# Every method implements its iteration once in ``_run(record)``; ``solve()``
# returns the root and ``trace()`` returns the list of recorded steps.

Steps = Optional[List[Dict[str, Any]]]


def _same_sign(u: float, v: float) -> bool:
    """True when u and v are both strictly positive or both strictly negative."""
    return (u > 0 and v > 0) or (u < 0 and v < 0)


def _check_value(v: float, where: str) -> None:
    if math.isnan(v):
        raise DomainError(f"f returned NaN at {where}")


class Bisection:
    """Bisection on a bracket [a, b] with f(a) f(b) <= 0.

    Stops when |f(c)| <= tol, when the half-interval satisfies the relative
    test (b - a)/2 <= tol (1 + |c|), or when the bracket cannot be split any
    further in floating point. An exact root at an end point is returned
    immediately.
    """

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
        _check_value(fa, "a")
        _check_value(fb, "b")
        if _same_sign(fa, fb):
            raise DomainError("f(a) and f(b) must have opposite signs")
        self.f, self.a, self.b = f, a, b
        self.fa, self.fb = fa, fb
        self.tol, self.max_iter = tol, max_iter

    def _run(self, record: bool) -> Tuple[float, Steps]:
        steps: Steps = [] if record else None
        a, b, f = self.a, self.b, self.f
        fa, fb = self.fa, self.fb
        for end, fend in ((a, fa), (b, fb)):
            if fend == 0.0:  # exact root at an end point
                if steps is not None:
                    steps.append(
                        {"iter": 0, "a": a, "b": b, "c": end, "f(a)": fa,
                         "f(b)": fb, "f(c)": fend, "interval": b - a}
                    )
                return end, steps
        for k in range(self.max_iter):
            c = 0.5 * (a + b)
            fc = f(c)
            _check_value(fc, f"x = {c}")
            if steps is not None:
                steps.append(
                    {"iter": k, "a": a, "b": b, "c": c, "f(a)": fa, "f(b)": fb,
                     "f(c)": fc, "interval": b - a}
                )
            if (
                abs(fc) <= self.tol
                or relative_tolerance_reached(0.5 * (b - a), c, self.tol)
                or c <= a
                or c >= b
            ):
                return c, steps
            if _same_sign(fa, fc):
                a, fa = c, fc
            else:
                b, fb = c, fc
        raise ConvergenceError("Bisection did not converge")

    def solve(self) -> float:
        return self._run(False)[0]

    def trace(self) -> List[Dict[str, Any]]:
        return self._run(True)[1]


class RegulaFalsi:
    """False position (regula falsi) on a bracket [a, b].

    With ``illinois=True`` (default) the Illinois modification halves the
    function value of an end point that is kept twice in a row, which avoids
    the one-sided, slow convergence of the plain method.
    """

    def __init__(
        self,
        f: Callable[[float], float],
        a: float,
        b: float,
        tol: float = 1e-10,
        max_iter: int = 10_000,
        illinois: bool = True,
    ):
        if a >= b:
            raise ValueError("Require a < b")
        fa, fb = f(a), f(b)
        _check_value(fa, "a")
        _check_value(fb, "b")
        if _same_sign(fa, fb):
            raise DomainError("f(a) and f(b) must have opposite signs")
        self.f, self.a, self.b = f, a, b
        self.fa, self.fb = fa, fb
        self.tol, self.max_iter, self.illinois = tol, max_iter, illinois

    def _run(self, record: bool) -> Tuple[float, Steps]:
        steps: Steps = [] if record else None
        a, b, f = self.a, self.b, self.f
        fa, fb = self.fa, self.fb
        if fa == 0.0:
            return a, steps
        if fb == 0.0:
            return b, steps
        side = 0
        c_old = None
        for k in range(self.max_iter):
            c = (a * fb - b * fa) / (fb - fa)
            fc = f(c)
            _check_value(fc, f"x = {c}")
            if steps is not None:
                steps.append(
                    {"iter": k, "a": a, "b": b, "c": c, "f(a)": fa, "f(b)": fb,
                     "f(c)": fc, "interval": b - a}
                )
            if (
                fc == 0.0
                or (c_old is not None and relative_tolerance_reached(c - c_old, c, self.tol))
                or relative_tolerance_reached(b - a, c, self.tol)
            ):
                return c, steps
            c_old = c
            if _same_sign(fc, fb):
                b, fb = c, fc
                if self.illinois and side == -1:
                    fa *= 0.5
                side = -1
            else:
                a, fa = c, fc
                if self.illinois and side == 1:
                    fb *= 0.5
                side = 1
        raise ConvergenceError("Regula falsi did not converge")

    def solve(self) -> float:
        return self._run(False)[0]

    def trace(self) -> List[Dict[str, Any]]:
        return self._run(True)[1]


class Brent:
    """Brent's method: inverse quadratic interpolation / secant steps
    safeguarded by bisection on a bracket [a, b] (guaranteed convergence,
    usually superlinear)."""

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
        _check_value(fa, "a")
        _check_value(fb, "b")
        if _same_sign(fa, fb):
            raise DomainError("f(a) and f(b) must have opposite signs")
        self.f, self.a, self.b = f, a, b
        self.fa, self.fb = fa, fb
        self.tol, self.max_iter = tol, max_iter

    def _run(self, record: bool) -> Tuple[float, Steps]:
        steps: Steps = [] if record else None
        f = self.f
        a, b = self.a, self.b
        fa, fb = self.fa, self.fb
        c, fc = b, fb
        d = e = b - a
        for k in range(self.max_iter):
            if _same_sign(fb, fc):
                c, fc = a, fa  # c is the other end of the bracket
                d = e = b - a
            if abs(fc) < abs(fb):  # b is always the best estimate
                a, b, c = b, c, b
                fa, fb, fc = fb, fc, fb
            tol1 = 2.0 * EPS * abs(b) + 0.5 * self.tol * (1.0 + abs(b))
            xm = 0.5 * (c - b)
            if abs(xm) <= tol1 or fb == 0.0:
                return b, steps
            step = "bisection"
            if abs(e) >= tol1 and abs(fa) > abs(fb):
                s = fb / fa
                if a == c:  # secant step
                    p = 2.0 * xm * s
                    q = 1.0 - s
                else:  # inverse quadratic interpolation
                    q = fa / fc
                    r = fb / fc
                    p = s * (2.0 * xm * q * (q - r) - (b - a) * (r - 1.0))
                    q = (q - 1.0) * (r - 1.0) * (s - 1.0)
                if p > 0:
                    q = -q
                p = abs(p)
                if 2.0 * p < min(3.0 * xm * q - abs(tol1 * q), abs(e * q)):
                    e, d = d, p / q  # accept the interpolation step
                    step = "interpolation"
                else:
                    d = xm
                    e = d
            else:
                d = xm
                e = d
            a, fa = b, fb
            b = b + d if abs(d) > tol1 else b + math.copysign(tol1, xm)
            fb = f(b)
            _check_value(fb, f"x = {b}")
            if steps is not None:
                steps.append(
                    {"iter": k, "b": b, "f(b)": fb, "c": c, "interval": abs(c - b), "step": step}
                )
        raise ConvergenceError("Brent's method did not converge")

    def solve(self) -> float:
        return self._run(False)[0]

    def trace(self) -> List[Dict[str, Any]]:
        return self._run(True)[1]


class FixedPoint:
    def __init__(
        self,
        g: Callable[[float], float],
        x0: float,
        tol: float = 1e-10,
        max_iter: int = 10_000,
    ):
        self.g, self.x0, self.tol, self.max_iter = g, x0, tol, max_iter

    def _run(self, record: bool) -> Tuple[float, Steps]:
        steps: Steps = [] if record else None
        x = self.x0
        for k in range(self.max_iter):
            x_new = self.g(x)
            if not math.isfinite(x_new):
                raise ConvergenceError("Fixed-point iteration diverged")
            if steps is not None:
                steps.append({"iter": k, "x": x, "x_new": x_new, "error": abs(x_new - x)})
            if relative_tolerance_reached(x_new - x, x_new, self.tol):
                return x_new, steps
            x = x_new
        raise ConvergenceError("Fixed-point iteration did not converge")

    def solve(self) -> float:
        return self._run(False)[0]

    def trace(self) -> List[Dict[str, Any]]:
        return self._run(True)[1]


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

    def _run(self, record: bool) -> Tuple[float, Steps]:
        steps: Steps = [] if record else None
        x0, x1, f = self.x0, self.x1, self.f
        f0, f1 = f(x0), f(x1)
        for k in range(self.max_iter):
            if f1 == 0.0:  # exact root
                return x1, steps
            denom = f1 - f0
            if denom == 0.0:
                raise ConvergenceError("Secant encountered a zero denominator f(x1) - f(x0)")
            x2 = x1 - f1 * (x1 - x0) / denom
            if not math.isfinite(x2):
                raise ConvergenceError("Secant method diverged")
            if steps is not None:
                steps.append(
                    {"iter": k, "x0": x0, "x1": x1, "x2": x2, "f(x0)": f0,
                     "f(x1)": f1, "error": abs(x2 - x1)}
                )
            if relative_tolerance_reached(x2 - x1, x2, self.tol):
                return x2, steps
            x0, x1 = x1, x2
            f0, f1 = f1, f(x1)
        raise ConvergenceError("Secant did not converge")

    def solve(self) -> float:
        return self._run(False)[0]

    def trace(self) -> List[Dict[str, Any]]:
        return self._run(True)[1]


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

    def _run(self, record: bool) -> Tuple[float, Steps]:
        steps: Steps = [] if record else None
        x = self.x0
        for k in range(self.max_iter):
            fx = self.f(x)
            if fx == 0.0:  # exact root
                return x, steps
            dfx = self.df(x)
            if dfx == 0.0:
                raise ConvergenceError("Derivative is zero in Newton method")
            x_new = x - fx / dfx
            if not math.isfinite(x_new):
                raise ConvergenceError("Newton method diverged")
            if steps is not None:
                steps.append(
                    {"iter": k, "x": x, "f(x)": fx, "df(x)": dfx, "x_new": x_new,
                     "error": abs(x_new - x)}
                )
            if relative_tolerance_reached(x_new - x, x_new, self.tol):
                return x_new, steps
            x = x_new
        raise ConvergenceError("Newton method did not converge")

    def solve(self) -> float:
        return self._run(False)[0]

    def trace(self) -> List[Dict[str, Any]]:
        return self._run(True)[1]


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
                f"{row[h]:>10.6g}" if isinstance(row[h], (int, float)) else f"{str(row[h]):>10}"
                for h in headers
            )
        )
