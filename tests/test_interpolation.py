import math

import pytest

from numethods import NewtonInterpolation, LagrangeInterpolation, CubicSpline, chebyshev_nodes

X = [0.0, 1.0, 3.0, 5.0]
Y = [2.0, -1.0, 4.0, 1.0]


def test_newton_and_lagrange_agree_and_interpolate():
    n, l = NewtonInterpolation(X, Y), LagrangeInterpolation(X, Y)
    for xi, yi in zip(X, Y):
        assert n.evaluate(xi) == pytest.approx(yi)
        assert l.evaluate(xi) == yi
    for t in [-1.0, 0.5, 2.2, 4.9, 6.0]:
        assert n.evaluate(t) == pytest.approx(l.evaluate(t), abs=1e-12)


def test_lagrange_basis_is_kronecker_delta():
    l = LagrangeInterpolation(X, Y)
    for i in range(len(X)):
        for j, xj in enumerate(X):
            assert l.basis(i, xj) == pytest.approx(1.0 if i == j else 0.0)


def test_add_point_matches_full_construction():
    n = NewtonInterpolation(X[:2], Y[:2])
    l = LagrangeInterpolation(X[:2], Y[:2])
    for xi, yi in zip(X[2:], Y[2:]):
        n.add_point(xi, yi)
        l.add_point(xi, yi)
    assert n.coeffs == NewtonInterpolation(X, Y).coeffs
    assert l.weights == pytest.approx(LagrangeInterpolation(X, Y).weights)
    with pytest.raises(ValueError):
        n.add_point(1.0, 0.0)


@pytest.mark.parametrize("cls", [NewtonInterpolation, LagrangeInterpolation])
def test_input_validation(cls):
    with pytest.raises(ValueError):
        cls([], [])
    with pytest.raises(ValueError):
        cls([1, 1.0], [2, 3])
    with pytest.raises(ValueError):
        cls([1, 2], [3])


def test_chebyshev_nodes():
    nodes = chebyshev_nodes(3)
    assert nodes == sorted(nodes)
    assert nodes == pytest.approx([-math.sqrt(3) / 2, 0.0, math.sqrt(3) / 2], abs=1e-15)
    assert chebyshev_nodes(1, 2.0, 4.0) == pytest.approx([3.0])


def test_cubic_spline_natural_and_clamped():
    xs = [2 * math.pi * i / 15 for i in range(16)]
    ys = [math.sin(x) for x in xs]
    nat = CubicSpline(xs, ys)
    cl = CubicSpline(xs, ys, bc="clamped", fprime=(1.0, 1.0))
    for s in (nat, cl):
        for xi, yi in zip(xs, ys):
            assert s.evaluate(xi) == pytest.approx(yi, abs=1e-14)
        err = max(abs(s.evaluate(t) - math.sin(t)) for t in [0.01 * k for k in range(629)])
        assert err < 1e-4
    assert nat.derivative(0.0, 2) == pytest.approx(0.0, abs=1e-12)
    assert cl.derivative(0.0) == pytest.approx(1.0)
    assert cl.derivative(xs[-1]) == pytest.approx(1.0)


def test_cubic_spline_reproduces_cubic_with_clamped_ends():
    p = lambda x: x**3 - 2 * x + 1
    dp = lambda x: 3 * x**2 - 2
    xs = [0.0, 0.7, 1.5, 2.0, 3.0]
    s = CubicSpline(xs, [p(x) for x in xs], bc="clamped", fprime=(dp(0.0), dp(3.0)))
    for t in [0.1, 1.0, 2.5, 2.9]:
        assert s.evaluate(t) == pytest.approx(p(t), abs=1e-12)


def test_cubic_spline_sorts_input_and_validates():
    s = CubicSpline([2.0, 0.0, 1.0], [4.0, 0.0, 1.0])
    assert s.x == [0.0, 1.0, 2.0]
    with pytest.raises(ValueError):
        CubicSpline([1.0], [1.0])
    with pytest.raises(ValueError):
        CubicSpline([0.0, 1.0], [0.0, 1.0], bc="clamped")
