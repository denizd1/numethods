from __future__ import annotations
import sys
from typing import Iterable, Sequence

# Machine epsilon for IEEE double precision (~2.22e-16).
EPS = sys.float_info.epsilon


def relative_tolerance_reached(delta: float, value: float, tol: float) -> bool:
    """Relative stopping test used throughout the package: |delta| <= tol (1 + |value|)."""
    return abs(delta) <= tol * (1.0 + abs(value))


def max_abs(rows: Iterable[Sequence[float]]) -> float:
    """Largest absolute entry of a list-of-lists (0.0 when empty)."""
    return max((abs(v) for row in rows for v in row), default=0.0)


def pivot_tolerance(scale: float, n: int) -> float:
    """Threshold below which a pivot is treated as zero.

    Uses the scale-aware rule ``n * eps * scale`` so that the singularity test
    does not depend on the units of the matrix (``1e-20 * I`` is not singular).
    """
    return max(n, 1) * EPS * scale
