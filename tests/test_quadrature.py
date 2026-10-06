import math
import warnings

import pytest

from numethods import (
    Trapezoidal,
    Simpson,
    GaussLegendre,
    AdaptiveSimpson,
    Romberg,
    gauss_legendre_nodes,
)
from numethods.exceptions import ConvergenceError


def test_composite_rules_converge_with_expected_order():
    e_t = [abs(Trapezoidal(math.sin, 0, math.pi, n).integrate() - 2) for n in (16, 32)]
    e_s = [abs(Simpson(math.sin, 0, math.pi, n).integrate() - 2) for n in (16, 32)]
    assert e_t[0] / e_t[1] == pytest.approx(4.0, rel=0.01)
    assert e_s[0] / e_s[1] == pytest.approx(16.0, rel=0.01)


@pytest.mark.parametrize("n", [0, -2, 2.5])
def test_invalid_n_is_rejected(n):
    with pytest.raises(ValueError):
        Trapezoidal(math.sin, 0, 1, n)
    with pytest.raises(ValueError):
        Simpson(math.sin, 0, 1, n)


def test_simpson_needs_even_n():
    with pytest.raises(ValueError):
        Simpson(math.sin, 0, 1, 3).integrate()


def test_gauss_legendre_nodes_and_exactness():
    x, w = gauss_legendre_nodes(2)
    assert x == pytest.approx((-1 / math.sqrt(3), 1 / math.sqrt(3)))
    x, w = gauss_legendre_nodes(3)
    assert x == pytest.approx((-math.sqrt(0.6), 0.0, math.sqrt(0.6)))
    assert w == pytest.approx((5 / 9, 8 / 9, 5 / 9))
    for n in range(1, 12):
        x, w = gauss_legendre_nodes(n)
        assert sum(w) == pytest.approx(2.0)
        for deg in range(2 * n):  # exact up to degree 2n - 1
            exact = 0.0 if deg % 2 else 2.0 / (deg + 1)
            assert sum(wi * xi**deg for xi, wi in zip(x, w)) == pytest.approx(exact, abs=1e-13)


def test_composite_gauss_legendre():
    assert GaussLegendre(math.exp, 0, 1, n=8).integrate() == pytest.approx(math.e - 1, abs=1e-15)
    errs = [abs(GaussLegendre(math.sin, 0, math.pi, 2, panels=p).integrate() - 2) for p in (4, 8)]
    assert errs[0] / errs[1] == pytest.approx(16.0, rel=0.05)  # O(h^4) for 2 points
    with pytest.raises(ValueError):
        GaussLegendre(math.sin, 0, 1, n=2, panels=0)


def test_adaptive_simpson():
    q = AdaptiveSimpson(math.sqrt, 0, 1, tol=1e-10)
    assert q.integrate() == pytest.approx(2 / 3, abs=1e-9)
    assert not q.max_depth_reached
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        AdaptiveSimpson(lambda x: 1.0 / x if x else 0.0, 0, 1, max_depth=5).integrate()
    assert any(issubclass(i.category, RuntimeWarning) for i in w)


def test_romberg():
    r = Romberg(math.sin, 0, math.pi)
    assert r.integrate() == pytest.approx(2.0, abs=1e-12)
    assert len(r.table) >= 3
    with pytest.raises(ConvergenceError):
        Romberg(lambda x: abs(x - 0.3) ** 0.5, 0, 1, tol=1e-15, max_levels=4).integrate()
