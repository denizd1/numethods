from __future__ import annotations
from typing import Callable, Optional
from .utils import EPS

# When h is None a step close to the optimum for each formula is used:
#     h = EPS**p * max(1, |x|),
# with p = 1/2 (forward/backward), 1/3 (central), 1/5 (4th-order central and
# Richardson) and 1/4 (second derivative). This balances truncation and
# rounding error and scales the step with |x|. An explicit h is used as given.


def _step(x: float, h: Optional[float], power: float) -> float:
    if h is None:
        h = EPS**power * max(1.0, abs(x))
        h = (x + h) - x  # make x + h exactly representable
    if not h > 0:
        raise ValueError("h must be positive")
    return h


# ----------------------------
# First derivative approximations
# ----------------------------


def ForwardDiff(f: Callable[[float], float], x: float, h: Optional[float] = None) -> float:
    """Forward finite difference approximation of f'(x) (1st order)."""
    h = _step(x, h, 1 / 2)
    return (f(x + h) - f(x)) / h


def BackwardDiff(f: Callable[[float], float], x: float, h: Optional[float] = None) -> float:
    """Backward finite difference approximation of f'(x) (1st order)."""
    h = _step(x, h, 1 / 2)
    return (f(x) - f(x - h)) / h


def CentralDiff(f: Callable[[float], float], x: float, h: Optional[float] = None) -> float:
    """Central finite difference approximation of f'(x) (2nd-order accurate)."""
    h = _step(x, h, 1 / 3)
    return (f(x + h) - f(x - h)) / (2 * h)


def CentralDiff4th(f: Callable[[float], float], x: float, h: Optional[float] = None) -> float:
    """Fourth-order accurate central difference approximation of f'(x)."""
    h = _step(x, h, 1 / 5)
    return (-f(x + 2 * h) + 8 * f(x + h) - 8 * f(x - h) + f(x - 2 * h)) / (12 * h)


# ----------------------------
# Second derivative
# ----------------------------


def SecondDerivative(f: Callable[[float], float], x: float, h: Optional[float] = None) -> float:
    """Central difference approximation of second derivative f''(x) (2nd order)."""
    h = _step(x, h, 1 / 4)
    return (f(x + h) - 2 * f(x) + f(x - h)) / (h**2)


# ----------------------------
# Richardson Extrapolation
# ----------------------------


def RichardsonExtrap(f: Callable[[float], float], x: float, h: Optional[float] = None) -> float:
    """Richardson extrapolation to improve derivative accuracy (4th order).
    Combines central-difference estimates with step h and h/2.
    """
    h = _step(x, h, 1 / 5)
    D_h = CentralDiff(f, x, h)
    D_h2 = CentralDiff(f, x, h / 2)
    return (4 * D_h2 - D_h) / 3
