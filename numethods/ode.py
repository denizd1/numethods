from __future__ import annotations
import math
from numbers import Real
from typing import Callable, List, Sequence, Tuple, Union
from .linalg import Matrix, Vector
from .solvers import LUDecomposition
from .exceptions import ConvergenceError, SingularMatrixError
from .utils import EPS

# A state is a float (scalar ODE) or a Vector (system of ODEs).
State = Union[float, Vector]


def _norm(v: State) -> float:
    """Max-norm of a state (absolute value for scalars)."""
    if isinstance(v, Vector):
        return v.norm_inf()
    return abs(v)


def _lincomb(y: State, h: float, terms: Sequence[Tuple[float, State]]) -> State:
    """y + h * sum(c * k) over (c, k) pairs with nonzero c."""
    acc = None
    for c, k in terms:
        if c == 0.0:
            continue
        term = c * k
        acc = term if acc is None else acc + term
    return y if acc is None else y + h * acc


def _scaled_error(err: State, y: State, y_new: State, atol: float, rtol: float) -> float:
    """max_i |err_i| / (atol + rtol * max(|y_i|, |y_new_i|))."""
    if isinstance(err, Vector):
        return max(
            (
                abs(e) / ((atol + rtol * max(abs(a), abs(b))) or EPS)
                for e, a, b in zip(err, y, y_new)
            ),
            default=0.0,
        )
    return abs(err) / ((atol + rtol * max(abs(y), abs(y_new))) or EPS)


# Gauss–Legendre 3-point rule on [-1, 1] (exact for polynomials of degree <= 5),
# used to integrate Lagrange basis polynomials for variable-step Adams weights.
_GL3 = ((-math.sqrt(0.6), 5.0 / 9.0), (0.0, 8.0 / 9.0), (math.sqrt(0.6), 5.0 / 9.0))


def _adams_weights(nodes: Sequence[float], a: float, b: float) -> List[float]:
    """w_j = integral_a^b L_j(s) ds for the Lagrange basis on ``nodes``
    (len(nodes) <= 3). For an Adams step: y(b) ≈ y(a) + sum_j w_j f(nodes[j])."""
    mid, half = 0.5 * (a + b), 0.5 * (b - a)
    weights = []
    for j, tj in enumerate(nodes):
        total = 0.0
        for x, w in _GL3:
            s = mid + half * x
            L = 1.0
            for m, tm in enumerate(nodes):
                if m != j:
                    L *= (s - tm) / (tj - tm)
            total += w * L
        weights.append(half * total)
    return weights


class ODESolver:
    """Base class for initial value problems y' = f(t, y), y(t0) = y0.

    * ``y0`` may be a float (scalar ODE) or a sequence (system of ODEs). For a
      system, ``f(t, y)`` receives a Vector and must return a sequence of the
      same length; the solution values are then Vectors.
    * ``h`` is the (positive) step size. ``solve(t_end)`` integrates from the
      current state to ``t_end`` on the grid t_k = t + k h (only the last step
      is shortened to land exactly on ``t_end``); ``t_end < t`` integrates
      backwards. The solver is left at ``t_end`` so ``solve`` can be called
      again to continue; ``h`` itself is never modified.
    """

    def __init__(
        self, f: Callable[[float, State], State], t0: float, y0: Union[float, Sequence[float]], h: float
    ):
        if not h > 0:
            raise ValueError("h must be positive (the direction is taken from t_end)")
        self.f = f
        self.t = float(t0)
        self.h = h
        if isinstance(y0, Real):
            self.y: State = float(y0)
            self.dim = None
        else:
            self.y = Vector(y0)
            self.dim = len(self.y)

    # -- right-hand side and Jacobian -------------------------------------
    def _rhs(self, t: float, y: State) -> State:
        if self.dim is None:
            return self.f(t, y)
        v = Vector(self.f(t, y))
        if len(v) != self.dim:
            raise ValueError(f"f(t, y) returned {len(v)} components, expected {self.dim}")
        return v

    def _dfdy(self, t: float, y: State, fy: State):
        """Forward-difference Jacobian df/dy (float for scalars, Matrix for systems)."""
        if self.dim is None:
            d = math.sqrt(EPS) * max(1.0, abs(y))
            return (self._rhs(t, y + d) - fy) / d
        n = self.dim
        cols = []
        for j in range(n):
            d = math.sqrt(EPS) * max(1.0, abs(y[j]))
            yp = y.copy()
            yp[j] = y[j] + d
            cols.append(((self._rhs(t, yp) - fy) / d).data)
        return Matrix._from_rows([[cols[j][i] for j in range(n)] for i in range(n)], n)

    newton_tol = 1e-10
    newton_max_iter = 50

    def _implicit_solve(self, t_new: float, base: State, ch: float, y_guess: State) -> State:
        """Solve y = base + ch * f(t_new, y) with Newton's method (finite-difference Jacobian)."""
        y = y_guess
        for _ in range(self.newton_max_iter):
            fy = self._rhs(t_new, y)
            g = y - base - ch * fy
            J = self._dfdy(t_new, y, fy)
            if self.dim is None:
                d = 1.0 - ch * J
                if d == 0.0:
                    raise ConvergenceError("Newton iteration: zero derivative in implicit step")
                dy = -g / d
            else:
                M = Matrix.identity(self.dim) - ch * J
                try:
                    dy = LUDecomposition(M).solve(-g)
                except SingularMatrixError:
                    raise ConvergenceError("Newton iteration: singular Jacobian in implicit step") from None
            y = y + dy
            if _norm(dy) <= self.newton_tol * (1.0 + _norm(y)):
                return y
        raise ConvergenceError("Newton iteration in implicit step did not converge")

    # -- time grid ---------------------------------------------------------
    def _grid(self, t_end: float) -> List[float]:
        t0 = self.t
        span = t_end - t0
        if span == 0.0:
            return [t0]
        direction = 1.0 if span > 0 else -1.0
        h = abs(self.h)
        ratio = abs(span) / h
        n = round(ratio)
        exact = n >= 1 and abs(ratio - n) <= 1e-9 * max(1.0, ratio)
        if not exact:
            n = math.floor(ratio)
        ts = [t0 + direction * i * h for i in range(n + 1)]
        if exact:
            ts[-1] = t_end
        else:
            ts.append(t_end)
        return ts

    # -- stepping ----------------------------------------------------------
    def _step(self, t: float, y: State, h: float) -> State:
        raise NotImplementedError

    def step(self) -> State:
        """One step of size h from the current state (the state is not changed)."""
        return self._step(self.t, self.y, self.h)

    def solve(self, t_end: float) -> Tuple[List[float], List[State]]:
        ts = self._grid(t_end)
        y = self.y
        ys = [y]
        for i in range(len(ts) - 1):
            y = self._step(ts[i], y, ts[i + 1] - ts[i])
            ys.append(y)
        self.t, self.y = ts[-1], y
        return ts, ys


# ------------------ Explicit Methods ------------------


class Euler(ODESolver):
    def _step(self, t, y, h):
        return y + h * self._rhs(t, y)


class Heun(ODESolver):
    def _step(self, t, y, h):
        k1 = self._rhs(t, y)
        y_predict = y + h * k1
        k2 = self._rhs(t + h, y_predict)
        return y + 0.5 * h * (k1 + k2)


class RK2(ODESolver):  # midpoint
    def _step(self, t, y, h):
        k1 = self._rhs(t, y)
        k2 = self._rhs(t + 0.5 * h, y + 0.5 * h * k1)
        return y + h * k2


class RK4(ODESolver):
    def _step(self, t, y, h):
        k1 = self._rhs(t, y)
        k2 = self._rhs(t + 0.5 * h, y + 0.5 * h * k1)
        k3 = self._rhs(t + 0.5 * h, y + 0.5 * h * k2)
        k4 = self._rhs(t + h, y + h * k3)
        return y + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


# ------------------ Implicit Methods ------------------


class BackwardEuler(ODESolver):
    """Backward Euler: y_{n+1} = y_n + h f(t_{n+1}, y_{n+1}) (Newton; raises
    ConvergenceError if the Newton iteration fails)."""

    def _step(self, t, y, h):
        return self._implicit_solve(t + h, y, h, y)


class ODETrapezoidal(ODESolver):
    """Implicit trapezoidal rule (Crank–Nicolson):
    y_{n+1} = y_n + h/2 [f(t_n, y_n) + f(t_{n+1}, y_{n+1})]."""

    def _step(self, t, y, h):
        base = y + 0.5 * h * self._rhs(t, y)  # f(t_n, y_n) evaluated once per step
        return self._implicit_solve(t + h, base, 0.5 * h, y)


# ------------------ Multistep Methods ------------------


def _uniform(ts: Sequence[float], i: int, k: int, h: float) -> bool:
    """True if the k-1 spacings before ts[i] equal the current step h."""
    return all(abs((ts[i - j] - ts[i - j - 1]) - h) <= 1e-9 * abs(h) for j in range(k - 1))


class AdamsBashforth(ODESolver):
    """k-step Adams–Bashforth (explicit), k = ``order`` in {2, 3}. Default: 2-step.

    The first k-1 steps are taken with RK4. Past values of f are cached, so
    each step costs one evaluation of f. If the final step is shortened to hit
    t_end, the coefficients are recomputed for the non-uniform spacing.
    """

    COEFFS = {2: (3.0 / 2.0, -1.0 / 2.0), 3: (23.0 / 12.0, -16.0 / 12.0, 5.0 / 12.0)}

    def __init__(self, f, t0, y0, h, order=2):
        if order not in self.COEFFS:
            raise ValueError("Only 2- and 3-step Adams-Bashforth are implemented")
        super().__init__(f, t0, y0, h)
        self.order = order

    def solve(self, t_end):
        ts = self._grid(t_end)
        k = self.order
        ys = [self.y]
        fs = [self._rhs(ts[0], self.y)]
        for i in range(len(ts) - 1):
            h = ts[i + 1] - ts[i]
            if i < k - 1:
                y_next = RK4._step(self, ts[i], ys[i], h)  # bootstrap
            elif _uniform(ts, i, k, h):
                y_next = _lincomb(ys[i], h, [(c, fs[i - j]) for j, c in enumerate(self.COEFFS[k])])
            else:
                w = _adams_weights([ts[i - j] for j in range(k)], ts[i], ts[i + 1])
                y_next = _lincomb(ys[i], 1.0, [(w[j], fs[i - j]) for j in range(k)])
            ys.append(y_next)
            if i + 1 < len(ts) - 1:
                fs.append(self._rhs(ts[i + 1], y_next))
        self.t, self.y = ts[-1], ys[-1]
        return ts, ys


class AdamsMoulton(ODESolver):
    """Implicit Adams–Moulton method.

    steps=2 (default): 2-step AM, 3rd order,
        y_{n+1} = y_n + h/12 (5 f_{n+1} + 8 f_n - f_{n-1}),
        the first step is taken with RK4.
    steps=1: 1-step AM = trapezoidal rule, 2nd order.

    The implicit equation is solved by Newton's method starting from an
    explicit Euler predictor; f values are cached between steps.
    """

    COEFFS = {1: (0.5, 0.5), 2: (5.0 / 12.0, 8.0 / 12.0, -1.0 / 12.0)}

    def __init__(self, f, t0, y0, h, steps=2):
        if steps not in self.COEFFS:
            raise ValueError("Only 1- and 2-step Adams-Moulton are implemented")
        super().__init__(f, t0, y0, h)
        self.steps = steps

    def solve(self, t_end):
        ts = self._grid(t_end)
        s = self.steps
        ys = [self.y]
        fs = [self._rhs(ts[0], self.y)]
        for i in range(len(ts) - 1):
            h = ts[i + 1] - ts[i]
            if i < s - 1:
                y_next = RK4._step(self, ts[i], ys[i], h)  # bootstrap
            else:
                if _uniform(ts, i, s, h):
                    c = self.COEFFS[s]
                    w = [h * cj for cj in c]
                else:
                    w = _adams_weights([ts[i + 1]] + [ts[i - j] for j in range(s)], ts[i], ts[i + 1])
                base = _lincomb(ys[i], 1.0, [(w[j + 1], fs[i - j]) for j in range(s)])
                guess = ys[i] + h * fs[i]
                y_next = self._implicit_solve(ts[i + 1], base, w[0], guess)
            ys.append(y_next)
            if i + 1 < len(ts) - 1:
                fs.append(self._rhs(ts[i + 1], y_next))
        self.t, self.y = ts[-1], ys[-1]
        return ts, ys


class PredictorCorrector(ODESolver):
    """AB2 predictor + trapezoidal (1-step Adams–Moulton) corrector, PECE mode:

        y*      = y_n + h (3/2 f_n - 1/2 f_{n-1})          (predict, evaluate)
        y_{n+1} = y_n + h/2 (f_n + f(t_{n+1}, y*))          (correct, evaluate)

    Two evaluations of f per step; the first step is taken with RK4.
    The stability region is that of an explicit method.
    """

    def solve(self, t_end):
        ts = self._grid(t_end)
        ys = [self.y]
        fs = [self._rhs(ts[0], self.y)]
        for i in range(len(ts) - 1):
            h = ts[i + 1] - ts[i]
            if i == 0:
                y_next = RK4._step(self, ts[0], ys[0], h)  # bootstrap
            else:
                if _uniform(ts, i, 2, h):
                    y_pred = _lincomb(ys[i], h, [(1.5, fs[i]), (-0.5, fs[i - 1])])
                else:
                    w = _adams_weights([ts[i], ts[i - 1]], ts[i], ts[i + 1])
                    y_pred = _lincomb(ys[i], 1.0, [(w[0], fs[i]), (w[1], fs[i - 1])])
                y_next = ys[i] + 0.5 * h * (fs[i] + self._rhs(ts[i + 1], y_pred))
            ys.append(y_next)
            if i + 1 < len(ts) - 1:
                fs.append(self._rhs(ts[i + 1], y_next))
        self.t, self.y = ts[-1], ys[-1]
        return ts, ys


# ------------------ Adaptive embedded Runge–Kutta ------------------


class RK45(ODESolver):
    """Runge–Kutta–Fehlberg 4(5) with adaptive step size.

    Each step is accepted when the scaled error estimate
        err = max_i |y5_i - y4_i| / (atol + rtol * max(|y_i|, |y_new_i|))
    is <= 1; the 5th-order solution is propagated (local extrapolation).
    The next step is h * clip(safety * err^(-1/5), min_factor, max_factor).
    ``h`` is the initial step size; after ``solve`` it holds the last
    proposed step, so a further ``solve`` continues smoothly.
    """

    # Fehlberg tableau
    _C = (0.0, 1 / 4, 3 / 8, 12 / 13, 1.0, 1 / 2)
    _A = (
        (),
        (1 / 4,),
        (3 / 32, 9 / 32),
        (1932 / 2197, -7200 / 2197, 7296 / 2197),
        (439 / 216, -8.0, 3680 / 513, -845 / 4104),
        (-8 / 27, 2.0, -3544 / 2565, 1859 / 4104, -11 / 40),
    )
    _B5 = (16 / 135, 0.0, 6656 / 12825, 28561 / 56430, -9 / 50, 2 / 55)
    _B4 = (25 / 216, 0.0, 1408 / 2565, 2197 / 4104, -1 / 5, 0.0)

    def __init__(
        self,
        f,
        t0,
        y0,
        h,
        rtol: float = 1e-6,
        atol: float = 1e-9,
        h_min: float = 1e-12,
        min_factor: float = 0.2,
        max_factor: float = 5.0,
        safety: float = 0.84,
        max_steps: int = 1_000_000,
    ):
        super().__init__(f, t0, y0, h)
        if rtol < 0 or atol < 0 or (rtol == 0 and atol == 0):
            raise ValueError("rtol and atol must be non-negative and not both zero")
        self.rtol = rtol
        self.atol = atol
        self.h_min = h_min
        self.min_factor = min_factor
        self.max_factor = max_factor
        self.safety = safety
        self.max_steps = max_steps
        self.rejected = 0  # number of rejected steps in the last solve

    def _embedded_step(self, t: float, y: State, h: float, k1: State):
        """Return (y_new, error_estimate, f(t+h, y_new) or None)."""
        ks = [k1]
        for c, row in zip(self._C[1:], self._A[1:]):
            ks.append(self._rhs(t + c * h, _lincomb(y, h, list(zip(row, ks)))))
        y5 = _lincomb(y, h, list(zip(self._B5, ks)))
        err = _lincomb(0.0 * y, h, [(b5 - b4, k) for b5, b4, k in zip(self._B5, self._B4, ks)])
        return y5, err, None

    def _step(self, t, y, h):
        return self._embedded_step(t, y, h, self._rhs(t, y))[0]

    def solve(self, t_end):
        t, y = self.t, self.y
        ts, ys = [t], [y]
        span = t_end - t
        direction = 1.0 if span >= 0 else -1.0
        h_abs = abs(self.h)
        fy = None
        steps = 0
        self.rejected = 0
        while True:
            remaining = abs(t_end - t)
            if remaining <= 4 * EPS * max(1.0, abs(t_end)):
                break
            if h_abs < self.h_min:
                raise ConvergenceError(f"Step size fell below h_min = {self.h_min} at t = {t}")
            steps += 1
            if steps > self.max_steps:
                raise ConvergenceError("RK45 exceeded max_steps")
            last = h_abs >= remaining
            h_step = remaining if last else h_abs
            if fy is None:
                fy = self._rhs(t, y)
            y_new, err, f_new = self._embedded_step(t, y, direction * h_step, fy)
            err_norm = _scaled_error(err, y, y_new, self.atol, self.rtol)
            if err_norm <= 1.0:
                t = t_end if last else t + direction * h_step
                y = y_new
                ts.append(t)
                ys.append(y)
                fy = f_new  # FSAL methods reuse the last stage
                factor = (
                    self.max_factor
                    if err_norm == 0.0
                    else min(self.max_factor, max(self.min_factor, self.safety * err_norm ** -0.2))
                )
                if not last:
                    h_abs = h_step * factor
            else:
                self.rejected += 1
                h_abs = h_step * max(self.min_factor, self.safety * err_norm ** -0.2)
        self.t, self.y, self.h = t, y, h_abs
        return ts, ys


class DormandPrince(RK45):
    """Dormand–Prince 5(4) (DOPRI5) with adaptive step size.

    Same step-size control as :class:`RK45`, but with the Dormand–Prince
    coefficients (smaller error constants) and the FSAL property: the last
    stage of an accepted step is the first stage of the next one, so a step
    costs six evaluations of f.
    """

    _C = (0.0, 1 / 5, 3 / 10, 4 / 5, 8 / 9, 1.0, 1.0)
    _A = (
        (),
        (1 / 5,),
        (3 / 40, 9 / 40),
        (44 / 45, -56 / 15, 32 / 9),
        (19372 / 6561, -25360 / 2187, 64448 / 6561, -212 / 729),
        (9017 / 3168, -355 / 33, 46732 / 5247, 49 / 176, -5103 / 18656),
        (35 / 384, 0.0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84),
    )
    _B5 = (35 / 384, 0.0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84, 0.0)
    _B4 = (5179 / 57600, 0.0, 7571 / 16695, 393 / 640, -92097 / 339200, 187 / 2100, 1 / 40)

    def _embedded_step(self, t, y, h, k1):
        ks = [k1]
        for c, row in zip(self._C[1:], self._A[1:]):
            ks.append(self._rhs(t + c * h, _lincomb(y, h, list(zip(row, ks)))))
        # The 7th stage is evaluated at y5 itself (row 7 of A equals B5).
        y5 = _lincomb(y, h, list(zip(self._B5, ks)))
        err = _lincomb(0.0 * y, h, [(b5 - b4, k) for b5, b4, k in zip(self._B5, self._B4, ks)])
        return y5, err, ks[-1]
