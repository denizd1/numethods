import math

import pytest

from numethods import Bisection, RegulaFalsi, Brent, FixedPoint, Secant, NewtonRoot, print_trace
from numethods.exceptions import ConvergenceError, DomainError

f = lambda x: x**2 - 2
df = lambda x: 2 * x
ROOT = math.sqrt(2)


@pytest.mark.parametrize("cls", [Bisection, RegulaFalsi, Brent])
def test_bracketing_methods(cls):
    assert cls(f, 0, 2, tol=1e-12).solve() == pytest.approx(ROOT, abs=1e-11)
    assert cls(lambda x: math.cos(x) - x, 0, 1).solve() == pytest.approx(0.7390851332151607, abs=1e-9)


@pytest.mark.parametrize("cls", [Bisection, RegulaFalsi, Brent])
def test_bracketing_root_at_end_point(cls):
    assert cls(lambda x: x, 0.0, 1.0).solve() == 0.0
    assert cls(lambda x: x - 1.0, 0.0, 1.0).solve() == 1.0


@pytest.mark.parametrize("cls", [Bisection, RegulaFalsi, Brent])
def test_bracketing_input_checks(cls):
    with pytest.raises(DomainError):
        cls(f, 2, 3)
    with pytest.raises(ValueError):
        cls(f, 2, 0)


def test_bisection_relative_tolerance_for_large_roots():
    r = Bisection(lambda x: x - 1e8 - 0.3, 0, 2e8).solve()
    assert r == pytest.approx(1e8 + 0.3, rel=1e-9)


def test_illinois_is_faster_than_plain_regula_falsi():
    g = lambda x: x**10 - 1
    plain = RegulaFalsi(g, 0, 1.3, illinois=False, max_iter=100_000).trace()
    illinois = RegulaFalsi(g, 0, 1.3).trace()
    assert len(illinois) < len(plain)


def test_open_methods():
    assert NewtonRoot(f, df, 1.0).solve() == pytest.approx(ROOT)
    assert Secant(f, 0, 2).solve() == pytest.approx(ROOT)
    assert FixedPoint(lambda x: 0.5 * (x + 2 / x), 1.0).solve() == pytest.approx(ROOT)


def test_open_method_failures():
    with pytest.raises(ConvergenceError):
        FixedPoint(lambda x: 2 / x, 1.0, max_iter=100).solve()
    with pytest.raises(ConvergenceError):
        NewtonRoot(lambda x: x**2 + 1, lambda x: 2 * x, 0.0).solve()


def test_trace_matches_solve(capsys):
    for solver in (Bisection(f, 0, 2), Brent(f, 0, 2), NewtonRoot(f, df, 1.0),
                   Secant(f, 0, 2), FixedPoint(lambda x: 0.5 * (x + 2 / x), 1.0)):
        steps = solver.trace()
        assert steps
        print_trace(steps)
    assert "iter" in capsys.readouterr().out
    newton_steps = NewtonRoot(f, df, 1.0).trace()
    assert newton_steps[-1]["x_new"] == NewtonRoot(f, df, 1.0).solve()
