from __future__ import annotations
from typing import Callable, List, Tuple


class ODESolver:
    def __init__(
        self, f: Callable[[float, float], float], t0: float, y0: float, h: float
    ):
        self.f = f
        self.t = t0
        self.y = y0
        self.h = h

    def step(self) -> float:
        raise NotImplementedError

    def solve(self, t_end: float) -> Tuple[List[float], List[float]]:
        ts, ys = [self.t], [self.y]
        while self.t < t_end - 1e-14:
            h = min(self.h, t_end - self.t)
            self.h = h
            self.y = self.step()
            self.t += h
            ts.append(self.t)
            ys.append(self.y)
        return ts, ys


# ------------------ Explicit Methods ------------------


class Euler(ODESolver):
    def step(self):
        return self.y + self.h * self.f(self.t, self.y)


class Heun(ODESolver):
    def step(self):
        k1 = self.f(self.t, self.y)
        y_predict = self.y + self.h * k1
        k2 = self.f(self.t + self.h, y_predict)
        return self.y + 0.5 * self.h * (k1 + k2)


class RK2(ODESolver):  # midpoint
    def step(self):
        k1 = self.f(self.t, self.y)
        k2 = self.f(self.t + 0.5 * self.h, self.y + 0.5 * self.h * k1)
        return self.y + self.h * k2


class RK4(ODESolver):
    def step(self):
        k1 = self.f(self.t, self.y)
        k2 = self.f(self.t + 0.5 * self.h, self.y + 0.5 * self.h * k1)
        k3 = self.f(self.t + 0.5 * self.h, self.y + 0.5 * self.h * k2)
        k4 = self.f(self.t + self.h, self.y + self.h * k3)
        return self.y + (self.h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


# ------------------ Implicit Methods ------------------


class BackwardEuler(ODESolver):
    def step(self):
        # Newton iteration for implicit solve
        y_new = self.y  # initial guess
        for _ in range(20):
            F = y_new - self.y - self.h * self.f(self.t + self.h, y_new)
            dF = 1.0 - self.h * self._dfdy(self.t + self.h, y_new)
            if abs(dF) < 1e-14:
                break
            y_next = y_new - F / dF
            if abs(y_next - y_new) < 1e-12:
                return y_next
            y_new = y_next
        return y_new

    def _dfdy(self, t, y):
        eps = 1e-8
        return (self.f(t, y + eps) - self.f(t, y - eps)) / (2 * eps)


class ODETrapezoidal(ODESolver):
    def step(self):
        y_new = self.y  # initial guess
        for _ in range(20):
            F = (
                y_new
                - self.y
                - 0.5
                * self.h
                * (self.f(self.t, self.y) + self.f(self.t + self.h, y_new))
            )
            dF = 1.0 - 0.5 * self.h * self._dfdy(self.t + self.h, y_new)
            if abs(dF) < 1e-14:
                break
            y_next = y_new - F / dF
            if abs(y_next - y_new) < 1e-12:
                return y_next
            y_new = y_next
        return y_new

    def _dfdy(self, t, y):
        eps = 1e-8
        return (self.f(t, y + eps) - self.f(t, y - eps)) / (2 * eps)


# ------------------ Multistep Methods ------------------


class AdamsBashforth(ODESolver):
    """k-step Adams–Bashforth. Default: 2-step."""

    def __init__(self, f, t0, y0, h, order=2):
        super().__init__(f, t0, y0, h)
        self.order = order
        # Bootstrap with RK4
        rk4 = RK4(f, t0, y0, h)
        self.ts, self.ys = [t0], [y0]
        for _ in range(order - 1):
            t1 = rk4.t + h
            y1 = rk4.step()
            rk4.t, rk4.y = t1, y1
            self.ts.append(t1)
            self.ys.append(y1)

    def solve(self, t_end):
        ts, ys = self.ts[:], self.ys[:]
        while ts[-1] < t_end - 1e-14:
            h = min(self.h, t_end - ts[-1])
            f_vals = [self.f(ts[-i - 1], ys[-i - 1]) for i in range(self.order)]
            if self.order == 2:
                y_next = ys[-1] + h * (3 / 2 * f_vals[0] - 1 / 2 * f_vals[1])
            elif self.order == 3:
                y_next = ys[-1] + h * (
                    23 / 12 * f_vals[0] - 16 / 12 * f_vals[1] + 5 / 12 * f_vals[2]
                )
            else:
                raise NotImplementedError("Only 2- and 3-step AB implemented")
            t_next = ts[-1] + h
            ts.append(t_next)
            ys.append(y_next)
        return ts, ys


class AdamsMoulton(ODESolver):
    """2-step Adams–Moulton implicit method (trapezoidal)."""

    def solve(self, t_end):
        ts, ys = [self.t], [self.y]
        rk4 = RK4(self.f, self.t, self.y, self.h)
        t1, y1 = rk4.t + self.h, rk4.step()
        ts.append(t1)
        ys.append(y1)
        while ts[-1] < t_end - 1e-14:
            h = min(self.h, t_end - ts[-1])
            f_prev = self.f(ts[-1], ys[-1])
            y_guess = ys[-1] + h * f_prev  # predictor
            for _ in range(10):
                F = ys[-1] + 0.5 * h * (f_prev + self.f(ts[-1] + h, y_guess)) - y_guess
                dF = -1 - 0.5 * h * self._dfdy(ts[-1] + h, y_guess)
                y_new = y_guess - F / dF
                if abs(y_new - y_guess) < 1e-12:
                    break
                y_guess = y_new
            t_next = ts[-1] + h
            ts.append(t_next)
            ys.append(y_guess)
        return ts, ys

    def _dfdy(self, t, y):
        eps = 1e-8
        return (self.f(t, y + eps) - self.f(t, y - eps)) / (2 * eps)


class PredictorCorrector(ODESolver):
    """AB2 predictor + AM2 corrector"""

    def solve(self, t_end):
        ts, ys = [self.t], [self.y]
        rk4 = RK4(self.f, self.t, self.y, self.h)
        t1, y1 = rk4.t + self.h, rk4.step()
        ts.append(t1)
        ys.append(y1)
        while ts[-1] < t_end - 1e-14:
            h = min(self.h, t_end - ts[-1])
            f_n = self.f(ts[-1], ys[-1])
            f_nm1 = self.f(ts[-2], ys[-2])
            y_pred = ys[-1] + h * (3 / 2 * f_n - 1 / 2 * f_nm1)
            y_corr = ys[-1] + 0.5 * h * (f_n + self.f(ts[-1] + h, y_pred))
            t_next = ts[-1] + h
            ts.append(t_next)
            ys.append(y_corr)
        return ts, ys


# ------------------ Adaptive RK45 ------------------


class RK45(ODESolver):
    """Runge–Kutta–Fehlberg (4,5) adaptive step."""

    def __init__(self, f, t0, y0, h, tol=1e-6):
        super().__init__(f, t0, y0, h)
        self.tol = tol

    def solve(self, t_end):
        ts, ys = [self.t], [self.y]
        while self.t < t_end - 1e-14:
            h = min(self.h, t_end - self.t)
            y, err = self._rkf_step(self.t, self.y, h)
            if err < self.tol:
                self.t += h
                self.y = y
                ts.append(self.t)
                ys.append(self.y)
                # adapt step
                s = 0.84 * (self.tol / (err + 1e-14)) ** 0.25
                self.h = min(h * max(0.1, s), 5 * h)
            else:
                self.h = 0.5 * h
        return ts, ys

    def _rkf_step(self, t, y, h):
        f = self.f
        k1 = h * f(t, y)
        k2 = h * f(t + 0.25 * h, y + 0.25 * k1)
        k3 = h * f(t + 3 / 8 * h, y + 3 / 32 * k1 + 9 / 32 * k2)
        k4 = h * f(
            t + 12 / 13 * h, y + 1932 / 2197 * k1 - 7200 / 2197 * k2 + 7296 / 2197 * k3
        )
        k5 = h * f(
            t + h, y + 439 / 216 * k1 - 8 * k2 + 3680 / 513 * k3 - 845 / 4104 * k4
        )
        k6 = h * f(
            t + 0.5 * h,
            y
            - 8 / 27 * k1
            + 2 * k2
            - 3544 / 2565 * k3
            + 1859 / 4104 * k4
            - 11 / 40 * k5,
        )
        y4 = y + (25 / 216 * k1 + 1408 / 2565 * k3 + 2197 / 4104 * k4 - 1 / 5 * k5)
        y5 = y + (
            16 / 135 * k1
            + 6656 / 12825 * k3
            + 28561 / 56430 * k4
            - 9 / 50 * k5
            + 2 / 55 * k6
        )
        err = abs(y5 - y4)
        return y5, err
