from __future__ import annotations
def relative_tolerance_reached(delta: float, value: float, tol: float) -> bool:
    return abs(delta) <= tol * (1.0 + abs(value))
