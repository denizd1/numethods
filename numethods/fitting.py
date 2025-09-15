from __future__ import annotations
from typing import List, Callable
import math
from .linalg import Matrix, Vector
from .orthogonal import LeastSquaresSolver
from .differentiation import ForwardDiff, BackwardDiff, CentralDiff, CentralDiff4th


class PolyFit:
    """Least squares polynomial fit of chosen degree."""

    def __init__(self, x: List[float], y: List[float], degree: int):
        if len(x) != len(y):
            raise ValueError("x and y must have same length")
        if degree < 0:
            raise ValueError("degree must be non-negative")

        self.x = [float(v) for v in x]
        self.y = [float(v) for v in y]
        self.degree = degree
        self.coeffs = self._fit()

    def _fit(self):
        n = len(self.x)
        m = self.degree + 1
        A = Matrix([[self.x[i] ** j for j in range(m)] for i in range(n)])
        b = Vector(self.y)
        return LeastSquaresSolver(A, b).solve()

    def evaluate(self, t: float) -> float:
        return sum(self.coeffs[j] * (t**j) for j in range(len(self.coeffs)))

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

        self.x = [float(v) for v in x]
        self.y = [float(v) for v in y]
        self.basis = basis
        self.coeffs = self._fit()

    def _fit(self):
        n = len(self.x)
        m = len(self.basis)
        A = Matrix([[phi(self.x[i]) for phi in self.basis] for i in range(n)])
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
        if any(val <= 0 for val in y):
            raise ValueError("y values must be positive for exponential fit")

        self.x = [float(v) for v in x]
        self.y = [float(v) for v in y]
        self.a, self.b = self._fit()

    def _fit(self):
        Y = [math.log(v) for v in self.y]
        A = Matrix([[1.0, self.x[i]] for i in range(len(self.x))])
        b = Vector(Y)
        coeffs = LeastSquaresSolver(A, b).solve()
        a = math.exp(coeffs[0])
        b = coeffs[1]
        return a, b

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
    """Nonlinear least squares fitting using adaptive Levenberg–Marquardt."""

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
        verbose: bool = True,
    ):
        if len(x) != len(y):
            raise ValueError("x and y must have same length")

        self.model = model
        self.x = [float(v) for v in x]
        self.y = [float(v) for v in y]
        self.params = [float(p) for p in init_params]
        self.max_iter = max_iter
        self.tol = tol
        self.lam = lam
        self.derivative_method = derivative_method
        self.verbose = verbose
        # history stores tuples: (iter, params, res_norm, λ, step, status)
        self.history: List[tuple] = []
        self._fit()

    def _residuals(self, params):
        return Vector([self.model(xi, params) - yi for xi, yi in zip(self.x, self.y)])

    def _jacobian(self, params):
        methods = {
            "forward": ForwardDiff,
            "backward": BackwardDiff,
            "central": CentralDiff,
            "central4th": CentralDiff4th,
        }
        diff_method = methods[self.derivative_method]

        m, n = len(self.x), len(params)
        J = [[0.0] * n for _ in range(m)]
        for j in range(n):
            for i, xi in enumerate(self.x):

                def func(pj):
                    new_params = params[:]
                    new_params[j] = pj
                    return self.model(xi, new_params)

                J[i][j] = diff_method(func, params[j])
        return Matrix(J)

    def _fit(self):
        params = self.params[:]
        prev_res_norm = float("inf")

        for k in range(self.max_iter):
            r = self._residuals(params)
            J = self._jacobian(params)
            JT = J.T
            A = JT @ J
            g = JT @ r

            for i in range(len(params)):
                A.data[i][i] += self.lam

            try:
                delta = LeastSquaresSolver(A, Vector([-gi for gi in g])).solve()
                new_params = [p + d for p, d in zip(params, delta)]
                new_r = self._residuals(new_params)
                res_norm = sum(abs(val) for val in new_r)
                status = "ok"
            except Exception:
                delta = [0.0] * len(params)
                new_params = params[:]
                res_norm = float("inf")
                status = "solver fail"
                self.lam *= 10
                self.history.append((k, params[:], res_norm, self.lam, delta, status))
                continue

            # record this iteration
            self.history.append((k, params[:], res_norm, self.lam, delta, status))

            # stopping conditions
            if res_norm < self.tol:
                params = new_params
                break
            if max(abs(d) for d in delta) < self.tol:
                params = new_params
                break
            if abs(prev_res_norm - res_norm) < 1e-12:
                params = new_params
                break

            if res_norm < prev_res_norm:
                params = new_params
                prev_res_norm = res_norm
                self.lam = max(self.lam / 10, 1e-12)
            else:
                self.lam *= 10
                if self.lam > 1e12:
                    break

        self.params = params

    def evaluate(self, t: float) -> float:
        return self.model(t, self.params)

    def summary(self):
        print("Nonlinear Fit Final Parameters")
        for j, p in enumerate(self.params):
            print(f" param{j} = {p: .6f}")
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
def plot_fit(
    x: List[float],
    y: List[float],
    fit_objects: List[object],
    labels: List[str] = None,
    true_func: Callable[[float], float] = None,
    num_points: int = 200,
):
    """
    Plot data points and fitted curves.
    fit_objects must implement .evaluate(t).
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError(
            "matplotlib is required for plotting. Install with `pip install matplotlib`."
        )

    def linspace(a: float, b: float, n: int) -> List[float]:
        if n == 1:
            return [a]
        step = (b - a) / (n - 1)
        return [a + i * step for i in range(n)]

    plt.scatter(x, y, color="black", label="data")
    t_vals = linspace(min(x), max(x), num_points)

    if true_func:
        plt.plot(t_vals, [true_func(t) for t in t_vals], "k--", label="true function")

    for i, fit in enumerate(fit_objects):
        lbl = labels[i] if labels else f"fit{i + 1}"
        plt.plot(t_vals, [fit.evaluate(t) for t in t_vals], label=lbl)

    plt.legend()
    plt.show()


def plot_residuals(
    x: List[float],
    y: List[float],
    fit_objects: List[object],
    labels: List[str] = None,
    mode: str = "line",
):
    """
    Plot residuals (y_i - fit.evaluate(x_i)) for each fit object.
    mode: "line" (default) or "bar" for absolute residual magnitudes.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("matplotlib is required for plotting.")

    plt.axhline(0, color="black", linewidth=0.8)

    for i, fit in enumerate(fit_objects):
        residuals = [yi - fit.evaluate(xi) for xi, yi in zip(x, y)]
        lbl = labels[i] if labels else f"fit{i + 1}"
        if mode == "line":
            plt.plot(x, residuals, marker="o", linestyle="--", label=lbl)
        elif mode == "bar":
            plt.bar(
                [xi + i * 0.1 for xi in x],
                [abs(r) for r in residuals],
                width=0.1,
                label=lbl,
            )

    plt.xlabel("x")
    plt.ylabel("Residuals")
    plt.title("Curve Fitting Residuals")
    plt.legend()
    plt.show()
