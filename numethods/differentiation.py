from __future__ import annotations
from typing import Callable


# ----------------------------
# First derivative approximations
# ----------------------------


def ForwardDiff(f: Callable[[float], float], x: float, h: float = 1e-5) -> float:
    """Forward finite difference approximation of f'(x)."""
    return (f(x + h) - f(x)) / h


def BackwardDiff(f: Callable[[float], float], x: float, h: float = 1e-5) -> float:
    """Backward finite difference approximation of f'(x)."""
    return (f(x) - f(x - h)) / h


def CentralDiff(f: Callable[[float], float], x: float, h: float = 1e-5) -> float:
    """Central finite difference approximation of f'(x) (2nd-order accurate)."""
    return (f(x + h) - f(x - h)) / (2 * h)


def CentralDiff4th(f: Callable[[float], float], x: float, h: float = 1e-5) -> float:
    """Fourth-order accurate central difference approximation of f'(x)."""
    return (-f(x + 2 * h) + 8 * f(x + h) - 8 * f(x - h) + f(x - 2 * h)) / (12 * h)


# ----------------------------
# Second derivative
# ----------------------------


def SecondDerivative(f: Callable[[float], float], x: float, h: float = 1e-5) -> float:
    """Central difference approximation of second derivative f''(x)."""
    return (f(x + h) - 2 * f(x) + f(x - h)) / (h**2)


# ----------------------------
# Richardson Extrapolation
# ----------------------------


def RichardsonExtrap(f: Callable[[float], float], x: float, h: float = 1e-2) -> float:
    """Richardson extrapolation to improve derivative accuracy.
    Combines estimates with step h and h/2.
    """
    D_h = CentralDiff(f, x, h)
    D_h2 = CentralDiff(f, x, h / 2)
    return (4 * D_h2 - D_h) / 3
