import math

import pytest

from numethods import (
    Vector,
    Euler,
    Heun,
    RK2,
    RK4,
    BackwardEuler,
    ODETrapezoidal,
    AdamsBashforth,
    AdamsMoulton,
    PredictorCorrector,
    RK45,
    DormandPrince,
)
from numethods.exceptions import ConvergenceError

f = lambda t, y: -2 * y + t
exact = lambda t: (2 * t - 1 + 5 * math.exp(-2 * t)) / 4

FIXED = [
    (Euler, {}, 1),
    (Heun, {}, 2),
    (RK2, {}, 2),
    (RK4, {}, 4),
    (BackwardEuler, {}, 1),
    (ODETrapezoidal, {}, 2),
    (AdamsBashforth, {"order": 2}, 2),
    (AdamsBashforth, {"order": 3}, 3),
    (AdamsMoulton, {"steps": 1}, 2),
    (AdamsMoulton, {"steps": 2}, 3),
    (PredictorCorrector, {}, 2),
]


@pytest.mark.parametrize("cls, kw, order", FIXED)
def test_convergence_order(cls, kw, order):
    errs = []
    for h in (0.05, 0.025):
        ts, ys = cls(f, 0.0, 1.0, h, **kw).solve(3.0)
        assert ts[-1] == 3.0 and len(ts) == round(3.0 / h) + 1
        errs.append(abs(ys[-1] - exact(3.0)))
    assert math.log2(errs[0] / errs[1]) == pytest.approx(order, abs=0.25)


@pytest.mark.parametrize("cls, kw, order", FIXED)
def test_short_final_step(cls, kw, order):
    ts, ys = cls(f, 0.0, 1.0, 0.07, **kw).solve(1.0)
    assert ts[-1] == 1.0
    assert ts[-1] - ts[-2] == pytest.approx(0.02)
    assert abs(ys[-1] - exact(1.0)) < 0.05


def test_solve_does_not_change_h_and_continues():
    s = RK4(lambda t, y: -y, 0.0, 1.0, 0.3)
    s.solve(1.0)
    assert s.h == 0.3 and s.t == 1.0
    ts, ys = s.solve(2.0)
    assert len(ts) - 1 == 4
    assert ys[-1] == pytest.approx(math.exp(-2.0), rel=1e-3)


def test_step_does_not_change_state():
    s = Euler(lambda t, y: -y, 0.0, 1.0, 0.1)
    assert s.step() == pytest.approx(0.9)
    assert s.t == 0.0 and s.y == 1.0


def test_backward_integration():
    ts, ys = RK4(lambda t, y: -y, 1.0, math.exp(-1.0), 0.1).solve(0.0)
    assert ts[-1] == 0.0
    assert ys[-1] == pytest.approx(1.0, rel=1e-6)


def test_multistep_bootstrap_stays_inside_interval():
    ts, ys = AdamsBashforth(f, 0.0, 1.0, 0.5, order=3).solve(0.6)
    assert ts == pytest.approx([0.0, 0.5, 0.6])
    ts, ys = AdamsMoulton(f, 0.0, 1.0, 0.5).solve(0.2)
    assert ts == [0.0, 0.2]


def test_invalid_arguments():
    with pytest.raises(ValueError):
        AdamsBashforth(f, 0.0, 1.0, 0.1, order=4)
    with pytest.raises(ValueError):
        AdamsMoulton(f, 0.0, 1.0, 0.1, steps=3)
    with pytest.raises(ValueError):
        Euler(f, 0.0, 1.0, 0.0)


ALL = [Euler, Heun, RK2, RK4, BackwardEuler, ODETrapezoidal, AdamsBashforth,
       AdamsMoulton, PredictorCorrector, RK45, DormandPrince]


@pytest.mark.parametrize("cls", ALL)
def test_systems(cls):
    osc = lambda t, y: [y[1], -y[0]]
    ts, ys = cls(osc, 0.0, [1.0, 0.0], 0.01).solve(math.pi)
    assert isinstance(ys[-1], Vector) and len(ys[-1]) == 2
    tol = 0.05 if cls in (Euler, BackwardEuler) else 1e-3
    assert ys[-1][0] == pytest.approx(-1.0, abs=tol)
    assert ys[-1][1] == pytest.approx(0.0, abs=tol)


def test_system_dimension_check():
    with pytest.raises(ValueError):
        RK4(lambda t, y: [y[0]], 0.0, [1.0, 2.0], 0.1).solve(1.0)


def test_implicit_methods_on_stiff_problem():
    lam = -100.0
    g = lambda t, y: lam * (y - math.sin(t)) + math.cos(t)
    for cls, tol in ((BackwardEuler, 1e-3), (ODETrapezoidal, 1e-5), (AdamsMoulton, 1e-5)):
        ts, ys = cls(g, 0.0, 0.0, 0.05).solve(2.0)
        assert abs(ys[-1] - math.sin(2.0)) < tol


def test_implicit_newton_failure_raises():
    s = BackwardEuler(lambda t, y: y * y, 0.0, 1.0, 2.0)  # y = 1 + 2 y^2 has no real root
    with pytest.raises(ConvergenceError):
        s.solve(2.0)


@pytest.mark.parametrize("cls", [RK45, DormandPrince])
def test_adaptive_tolerance_is_scale_invariant(cls):
    results = []
    for y0 in (1.0, 1e8):
        s = cls(lambda t, y: -y, 0.0, y0, 0.1, rtol=1e-6, atol=1e-12 * y0)
        ts, ys = s.solve(5.0)
        rel = abs(ys[-1] - y0 * math.exp(-5)) / (y0 * math.exp(-5))
        assert rel < 1e-5
        assert ts[-1] == 5.0
        results.append(len(ts))
    assert results[0] == results[1]


def test_adaptive_step_count_follows_tolerance():
    counts = []
    for rtol in (1e-3, 1e-6, 1e-9):
        ts, _ = RK45(lambda t, y: -y, 0.0, 1.0, 0.1, rtol=rtol, atol=1e-12).solve(5.0)
        counts.append(len(ts))
    assert counts[0] < counts[1] < counts[2] < 200


def test_rk45_h_min_uses_unclipped_step():
    s = RK45(lambda t, y: 1.0, 0.0, 0.0, 0.1, h_min=1e-6)
    s.t = 1.0 - 5e-7
    ts, ys = s.solve(1.0)
    assert ts[-1] == 1.0


def test_dormand_prince_fsal_saves_evaluations():
    calls = {"rkf": 0, "dp": 0}

    def make(key):
        def g(t, y):
            calls[key] += 1
            return -y + math.sin(10 * t)
        return g

    RK45(make("rkf"), 0.0, 0.0, 0.5).solve(10.0)
    DormandPrince(make("dp"), 0.0, 0.0, 0.5).solve(10.0)
    assert calls["dp"] < calls["rkf"]
