from __future__ import annotations
from typing import List, Callable, Optional
import math
import warnings
from .linalg import Matrix, Vector
from .orthogonal import LeastSquaresSolver
from .differentiation import ForwardDiff, BackwardDiff, CentralDiff, CentralDiff4th
from .exceptions import DomainError, NumericalError


class PolyFit:
    """Least squares polynomial fit of chosen degree.

    The least squares problem is solved (Householder QR) in the scaled
    variable s = (x - shift) / scale, with ``shift`` the midpoint and
    ``scale`` the half-width of the data range, so the Vandermonde matrix
    stays well conditioned even for high degrees or large x. ``coeffs`` are
    the coefficients of 1, x, x^2, ... (converted back from the scaled
    variable); ``evaluate`` uses the scaled form with Horner's rule.
    """

    def __init__(self, x: List[float], y: List[float], degree: int):
        if len(x) != len(y):
            raise ValueError("x and y must have same length")
        if degree < 0:
            raise ValueError("degree must be non-negative")
        if len(x) < degree + 1:
            raise DomainError(
                f"degree {degree} needs at least {degree + 1} data points (got {len(x)})"
            )

        self.x = [float(v) for v in x]
        self.y = [float(v) for v in y]
        self.degree = degree
        lo, hi = min(self.x), max(self.x)
        self.shift = 0.5 * (lo + hi)
        self.scale = 0.5 * (hi - lo) if hi > lo else 1.0
        self.scaled_coeffs = self._fit()
        self.coeffs = Vector(self._monomial_coeffs())

    def _fit(self) -> Vector:
        m = self.degree + 1
        s = [(xi - self.shift) / self.scale for xi in self.x]
        A = Matrix([[si**j for j in range(m)] for si in s])
        b = Vector(self.y)
        return LeastSquaresSolver(A, b).solve()

    def _monomial_coeffs(self) -> List[float]:
        """Expand sum_k a_k ((x - c)/d)^k into powers of x."""
        a, c, d = self.scaled_coeffs, self.shift, self.scale
        n = len(a)
        out = [0.0] * n
        for k in range(n):
            ak = a[k] / d**k
            for j in range(k + 1):
                out[j] += ak * math.comb(k, j) * (-c) ** (k - j)
        return out

    def evaluate(self, t: float) -> float:
        s = (t - self.shift) / self.scale
        result = 0.0
        for c in reversed(self.scaled_coeffs.data):
            result = result * s + c
        return result

    def summary(self):
        print("Polynomial Fit Coefficients")
        print("degree =", self.degree)
        print(" coeff |   value")
        print("-------------------")
        for j, c in enumerate(self.coeffs):
            print(f"  c{j:<3}| {c: .6f}")
        print()

    def trace(self):
        print("Polynomial Fit Trace (Vandermonde system)")
        print(" x | y | " + " | ".join([f"x^{j}" for j in range(self.degree + 1)]))
        print("-" * 40)
        for xi, yi in zip(self.x, self.y):
            row = " | ".join([f"{xi**j: .4f}" for j in range(self.degree + 1)])
            print(f"{xi: .4f} | {yi: .4f} | {row}")
        print()


class LinearFit:
    """Least squares fit with custom basis functions."""

    def __init__(
        self, x: List[float], y: List[float], basis: List[Callable[[float], float]]
    ):
        if len(x) != len(y):
            raise ValueError("x and y must have same length")
        if not basis:
            raise ValueError("basis must contain at least one function")
        if len(x) < len(basis):
            raise DomainError(
                f"{len(basis)} basis functions need at least {len(basis)} data points (got {len(x)})"
            )

        self.x = [float(v) for v in x]
        self.y = [float(v) for v in y]
        self.basis = basis
        self.coeffs = self._fit()

    def _fit(self):
        A = Matrix([[phi(xi) for phi in self.basis] for xi in self.x])
        b = Vector(self.y)
        return LeastSquaresSolver(A, b).solve()

    def evaluate(self, t: float) -> float:
        return sum(c * phi(t) for c, phi in zip(self.coeffs, self.basis))

    def summary(self):
        print("Linear Fit Coefficients")
        print(" basis |   value")
        print("-------------------")
        for j, c in enumerate(self.coeffs):
            print(f"  φ{j:<3}| {c: .6f}")
        print()

    def trace(self):
        print("Linear Fit Trace (design matrix)")
        print(" x | y | " + " | ".join([f"φ{j}(x)" for j in range(len(self.basis))]))
        print("-" * 40)
        for xi, yi in zip(self.x, self.y):
            row = " | ".join([f"{phi(xi): .4f}" for phi in self.basis])
            print(f"{xi: .4f} | {yi: .4f} | {row}")
        print()


class ExpFit:
    """Fit y ≈ a * exp(bx) using log transform + linear least squares."""

    def __init__(self, x: List[float], y: List[float]):
        if len(x) != len(y):
            raise ValueError("x and y must have same length")
        if len(x) < 2:
            raise DomainError("exponential fit needs at least two data points")
        if any(val <= 0 for val in y):
            raise ValueError("y values must be positive for exponential fit")

        self.x = [float(v) for v in x]
        self.y = [float(v) for v in y]
        self.a, self.b = self._fit()

    def _fit(self):
        Y = [math.log(v) for v in self.y]
        A = Matrix([[1.0, xi] for xi in self.x])
        coeffs = LeastSquaresSolver(A, Vector(Y)).solve()
        return math.exp(coeffs[0]), coeffs[1]

    def evaluate(self, t: float) -> float:
        return self.a * math.exp(self.b * t)

    def summary(self):
        print("Exponential Fit Parameters")
        print(" param |   value")
        print("-------------------")
        print(f"   a   | {self.a: .6f}")
        print(f"   b   | {self.b: .6f}")
        print()

    def trace(self):
        print("Exponential Fit Trace (log transform)")
        print(" x | y | log(y)")
        print("-------------------")
        for xi, yi in zip(self.x, self.y):
            print(f"{xi: .4f} | {yi: .4f} | {math.log(yi): .4f}")
        print()


class NonlinearFit:
    """Nonlinear least squares fitting with adaptive Levenberg–Marquardt.

    Minimizes ||r(p)||_2^2 with r_i = model(x_i, p) - y_i. Each iteration
    solves the damped Gauss–Newton problem

        min_delta || [J; sqrt(lam) I] delta + [r; 0] ||_2,

    i.e. (J^T J + lam I) delta = -J^T r, with Householder QR (J^T J is never
    formed). J is a finite-difference Jacobian (``derivative_method``). A
    step is accepted only if it lowers ||r||_2; then lam is divided by 10,
    otherwise lam is multiplied by 10 and the step is retried with the same
    Jacobian. Model errors such as math domain errors or overflow at a trial
    point count as a rejected step; other exceptions propagate.

    Stopping tests (all relative): ||r|| <= tol, step max_i |delta_i| /
    (1 + |p_i|) <= tol, or relative decrease of ||r|| <= tol. ``converged``
    records whether one was met; otherwise a RuntimeWarning is issued.

    ``history`` holds tuples (iter, params, res_norm, lam, delta, status),
    where ``params`` is the point the step starts from, ``res_norm`` is
    ||r||_2 at the trial point and status is "accepted", "rejected" or
    "step failed". ``lam`` keeps the initial damping, ``lam_final`` the last.
    """

    DERIVATIVES = {
        "forward": ForwardDiff,
        "backward": BackwardDiff,
        "central": CentralDiff,
        "central4th": CentralDiff4th,
    }

    def __init__(
        self,
        model: Callable[[float, List[float]], float],
        x: List[float],
        y: List[float],
        init_params: List[float],
        max_iter: int = 100,
        tol: float = 1e-8,
        lam: float = 1e-3,
        derivative_method: str = "central",
        verbose: bool = False,
    ):
        if len(x) != len(y):
            raise ValueError("x and y must have same length")
        if not init_params:
            raise ValueError("init_params must not be empty")
        if derivative_method not in self.DERIVATIVES:
            raise ValueError(
                f"derivative_method must be one of {sorted(self.DERIVATIVES)}"
            )
        if len(x) < len(init_params):
            raise DomainError(
                f"{len(init_params)} parameters need at least {len(init_params)} data points"
            )
        if lam <= 0:
            raise ValueError("lam must be positive")

        self.model = model
        self.x = [float(v) for v in x]
        self.y = [float(v) for v in y]
        self.params = [float(p) for p in init_params]
        self.max_iter = max_iter
        self.tol = tol
        self.lam = lam
        self.lam_final = lam
        self.derivative_method = derivative_method
        self.verbose = verbose
        self.converged = False
        # history stores tuples: (iter, params, res_norm, λ, step, status)
        self.history: List[tuple] = []
        self._fit()

    def _residuals(self, params) -> Vector:
        return Vector([self.model(xi, params) - yi for xi, yi in zip(self.x, self.y)])

    def _jacobian(self, params) -> Matrix:
        diff_method = self.DERIVATIVES[self.derivative_method]
        m, n = len(self.x), len(params)
        J = [[0.0] * n for _ in range(m)]
        for j in range(n):
            for i, xi in enumerate(self.x):

                def func(pj, xi=xi, j=j):
                    new_params = params[:]
                    new_params[j] = pj
                    return self.model(xi, new_params)

                J[i][j] = diff_method(func, params[j])
        return Matrix(J)

    @staticmethod
    def _lm_step(J: Matrix, r: Vector, lam: float) -> Vector:
        n = J.n
        root = math.sqrt(lam)
        A = Matrix(J.data + [[root if i == j else 0.0 for j in range(n)] for i in range(n)])
        b = Vector([-v for v in r.data] + [0.0] * n)
        return LeastSquaresSolver(A, b).solve()

    def _fit(self):
        params = self.params[:]
        lam = self.lam
        r = self._residuals(params)
        res_norm = r.norm2()
        J = None
        self.converged = res_norm <= self.tol

        for k in range(self.max_iter):
            if self.converged:
                break
            if J is None:
                J = self._jacobian(params)
            try:
                delta = self._lm_step(J, r, lam)
                new_params = [p + d for p, d in zip(params, delta)]
                new_r = self._residuals(new_params)
                new_norm = new_r.norm2()
                if not math.isfinite(new_norm):
                    raise ValueError("non-finite residual")
            except (NumericalError, ArithmeticError, ValueError):
                self.history.append((k, params[:], float("inf"), lam, [0.0] * len(params), "step failed"))
                if self.verbose:
                    print(f"iter {k:3d}: step failed, lam={lam:.1e}")
                lam *= 10
                if lam > 1e16:
                    break
                continue

            accepted = new_norm < res_norm
            status = "accepted" if accepted else "rejected"
            self.history.append((k, params[:], new_norm, lam, list(delta), status))
            if self.verbose:
                print(f"iter {k:3d}: res_norm={new_norm:.6e} lam={lam:.1e} {status}")

            step_small = max(abs(d) / (1.0 + abs(p)) for d, p in zip(delta, params)) <= self.tol
            if accepted:
                decrease_small = (res_norm - new_norm) <= self.tol * res_norm
                params, r, res_norm = new_params, new_r, new_norm
                J = None
                lam = max(lam / 10, 1e-12)
                if res_norm <= self.tol or step_small or decrease_small:
                    self.converged = True
            else:
                if step_small:  # no further progress possible at this resolution
                    self.converged = True
                lam *= 10
                if lam > 1e16:
                    break

        self.params = params
        self.lam_final = lam
        if not self.converged:
            warnings.warn(
                "NonlinearFit did not meet its stopping criteria; check max_iter, tol or init_params",
                RuntimeWarning,
                stacklevel=3,
            )

    def evaluate(self, t: float) -> float:
        return self.model(t, self.params)

    def summary(self):
        print("Nonlinear Fit Final Parameters")
        for j, p in enumerate(self.params):
            print(f" param{j} = {p: .6f}")
        print(f" converged = {self.converged}")
        print()

    def trace(self):
        print("Nonlinear Fit Trace (Levenberg–Marquardt)")
        header = (
            " iter | "
            + " | ".join([f"param{j}" for j in range(len(self.params))])
            + " | res_norm |    λ    | step_norm | status"
        )
        print(header)
        print("-" * len(header))
        for k, params, res_norm, lam, delta, status in self.history:
            row = " | ".join([f"{p: .6f}" for p in params])
            step_norm = max(abs(d) for d in delta) if delta else 0.0
            print(
                f"{k:5d} | {row} | {res_norm: .6e} | {lam: .1e} | {step_norm: .3e} | {status}"
            )
        print("Final params:", self.params)
        print()


# ----------------------------
# Plotting helper for curve fitting
# ----------------------------
def _pyplot():
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError(
            "matplotlib is required for plotting. Install with `pip install matplotlib`."
        ) from None
    return plt


def plot_fit(
    x: List[float],
    y: List[float],
    fit_objects: List[object],
    labels: Optional[List[str]] = None,
    true_func: Optional[Callable[[float], float]] = None,
    num_points: int = 200,
    ax=None,
    show: Optional[bool] = None,
):
    """
    Plot data points and fitted curves.
    fit_objects must implement .evaluate(t).
    Draws on ``ax`` (a new figure if None) and returns the axes;
    ``show`` defaults to True only when a new figure was created.
    """
    plt = _pyplot()
    if show is None:
        show = ax is None
    if ax is None:
        _, ax = plt.subplots()

    def linspace(a: float, b: float, n: int) -> List[float]:
        if n == 1:
            return [a]
        step = (b - a) / (n - 1)
        return [a + i * step for i in range(n)]

    ax.scatter(x, y, color="black", label="data")
    t_vals = linspace(min(x), max(x), num_points)

    if true_func:
        ax.plot(t_vals, [true_func(t) for t in t_vals], "k--", label="true function")

    for i, fit in enumerate(fit_objects):
        lbl = labels[i] if labels else f"fit{i + 1}"
        ax.plot(t_vals, [fit.evaluate(t) for t in t_vals], label=lbl)

    ax.legend()
    if show:
        plt.show()
    return ax


def plot_residuals(
    x: List[float],
    y: List[float],
    fit_objects: List[object],
    labels: Optional[List[str]] = None,
    mode: str = "line",
    ax=None,
    show: Optional[bool] = None,
):
    """
    Plot residuals (y_i - fit.evaluate(x_i)) for each fit object.
    mode: "line" (default) or "bar" for absolute residual magnitudes.
    Draws on ``ax`` (a new figure if None) and returns the axes;
    ``show`` defaults to True only when a new figure was created.
    """
    if mode not in ("line", "bar"):
        raise ValueError("mode must be 'line' or 'bar'")
    plt = _pyplot()
    if show is None:
        show = ax is None
    if ax is None:
        _, ax = plt.subplots()

    ax.axhline(0, color="black", linewidth=0.8)

    xs = sorted(set(float(v) for v in x))
    spacing = min((b - a for a, b in zip(xs, xs[1:])), default=1.0)
    width = 0.8 * spacing / max(len(fit_objects), 1)

    for i, fit in enumerate(fit_objects):
        residuals = [yi - fit.evaluate(xi) for xi, yi in zip(x, y)]
        lbl = labels[i] if labels else f"fit{i + 1}"
        if mode == "line":
            ax.plot(x, residuals, marker="o", linestyle="--", label=lbl)
        else:
            offset = (i - 0.5 * (len(fit_objects) - 1)) * width
            ax.bar(
                [xi + offset for xi in x],
                [abs(r) for r in residuals],
                width=width,
                label=lbl,
            )

    ax.set_xlabel("x")
    ax.set_ylabel("Residuals" if mode == "line" else "|Residuals|")
    ax.set_title("Curve Fitting Residuals")
    ax.legend()
    if show:
        plt.show()
    return ax
