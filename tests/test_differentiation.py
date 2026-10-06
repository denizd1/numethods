import math

import pytest

from numethods import (
    ForwardDiff,
    BackwardDiff,
    CentralDiff,
    CentralDiff4th,
    SecondDerivative,
    RichardsonExtrap,
)


@pytest.mark.parametrize(
    "func, tol",
    [(ForwardDiff, 1e-7), (BackwardDiff, 1e-7), (CentralDiff, 1e-10),
     (CentralDiff4th, 1e-12), (RichardsonExtrap, 1e-12)],
)
def test_default_step_is_near_optimal(func, tol):
    assert abs(func(math.exp, 1.0) - math.e) < tol * math.e
    # the default step scales with |x|
    assert abs(func(lambda x: x * x, 1e6) - 2e6) < tol * 2e6


def test_second_derivative_default_step():
    assert abs(SecondDerivative(math.exp, 1.0) - math.e) < 1e-7


def test_explicit_step_is_used_as_given():
    f = lambda x: x**3
    assert ForwardDiff(f, 1.0, 0.1) == pytest.approx((1.1**3 - 1) / 0.1)
    with pytest.raises(ValueError):
        CentralDiff(f, 1.0, 0.0)
