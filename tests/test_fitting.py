import math
import random
import warnings

import pytest

from numethods import PolyFit, LinearFit, ExpFit, NonlinearFit
from numethods.exceptions import DomainError


def test_polyfit_recovers_exact_polynomial():
    x = [0, 0.5, 1, 1.5, 2, 3]
    y = [2 - 3 * t + 1.5 * t * t for t in x]
    p = PolyFit(x, y, 2)
    assert list(p.coeffs) == pytest.approx([2.0, -3.0, 1.5], abs=1e-12)
    assert p.evaluate(2.5) == pytest.approx(2 - 7.5 + 1.5 * 6.25)


def test_polyfit_high_degree_large_x_is_well_conditioned():
    xs = [1000.0 + i for i in range(12)]
    ys = [math.sin(x / 3) for x in xs]
    p = PolyFit(xs, ys, 9)
    assert max(abs(p.evaluate(a) - b) for a, b in zip(xs, ys)) < 1e-6


def test_polyfit_input_checks():
    with pytest.raises(DomainError):
        PolyFit([0, 1], [1, 2], 3)
    with pytest.raises(ValueError):
        PolyFit([0, 1], [1], 1)


def test_linear_and_exp_fit():
    x = [0, 1, 2, 3, 4]
    lf = LinearFit(x, [1 + 2 * t for t in x], [lambda t: 1.0, lambda t: t])
    assert list(lf.coeffs) == pytest.approx([1.0, 2.0])
    ef = ExpFit(x, [3 * math.exp(0.7 * t) for t in x])
    assert (ef.a, ef.b) == pytest.approx((3.0, 0.7))
    with pytest.raises(DomainError):
        LinearFit([1.0], [1.0], [lambda t: 1.0, lambda t: t])


def exp_model(x, p):
    return p[0] * math.exp(p[1] * x)


def test_nonlinear_fit_exact_data():
    x = [0, 1, 2, 3, 4, 5]
    y = [2 * math.exp(0.5 * t) for t in x]
    nf = NonlinearFit(exp_model, x, y, [1.0, 1.2])
    assert nf.converged
    assert nf.params == pytest.approx([2.0, 0.5], rel=1e-8)
    assert nf.lam == 1e-3  # initial damping is not overwritten
    accepted = [h[2] for h in nf.history if h[5] == "accepted"]
    assert all(b < a for a, b in zip(accepted, accepted[1:]))  # monotone decrease


def test_nonlinear_fit_noisy_sine():
    rng = random.Random(42)
    model = lambda x, p: p[0] * math.sin(p[1] * x + p[2])
    x = [i * 0.4 for i in range(16)]
    y = [model(t, [2.5, 1.3, 0.5]) + rng.gauss(0, 0.15) for t in x]
    nf = NonlinearFit(model, x, y, [2.0, 1.0, 0.0])
    assert nf.converged
    assert nf.params == pytest.approx([2.5, 1.3, 0.5], abs=0.1)


def test_nonlinear_fit_validation_and_errors():
    x, y = [0, 1, 2], [1, 2, 4]
    with pytest.raises(ValueError):
        NonlinearFit(exp_model, x, y, [1.0, 1.0], derivative_method="foo")
    with pytest.raises(DomainError):
        NonlinearFit(exp_model, [0], [1], [1.0, 1.0])

    def buggy(t, p):
        if p[0] > 1.5:
            raise TypeError("bug in user model")
        return p[0] * t

    with pytest.raises(TypeError):
        NonlinearFit(buggy, [1, 2, 3], [2, 4, 6], [1.0])


def test_nonlinear_fit_warns_when_not_converged():
    x = [0, 1, 2, 3, 4, 5]
    y = [2 * math.exp(0.5 * t) for t in x]
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        nf = NonlinearFit(exp_model, x, y, [1.0, 1.2], max_iter=2)
    assert not nf.converged
    assert any(issubclass(i.category, RuntimeWarning) for i in w)


def test_plot_helpers_accept_axes():
    pytest.importorskip("matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from numethods import plot_fit, plot_residuals

    x = [0, 1, 2, 3]
    y = [1, 3, 2, 5]
    fit = PolyFit(x, y, 1)
    fig, ax = plt.subplots()
    assert plot_fit(x, y, [fit], ax=ax) is ax
    assert plot_residuals(x, y, [fit, fit], mode="bar", ax=ax) is ax
    with pytest.raises(ValueError):
        plot_residuals(x, y, [fit], mode="pie", ax=ax)
    plt.close(fig)
