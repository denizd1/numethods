from numethods import Matrix, Vector
from numethods.orthogonal import (
    QRHouseholder,
    QRModifiedGramSchmidt,
    LeastSquaresSolver,
)
from numethods.solvers import (
    LUDecomposition,
    GaussJordan,
    Jacobi,
    GaussSeidel,
    Cholesky,
)
from numethods.roots import Bisection, FixedPoint, Secant, NewtonRoot, print_trace
from numethods.interpolation import NewtonInterpolation, LagrangeInterpolation
from numethods.quadrature import Trapezoidal, Simpson, GaussLegendre
from numethods.eigen import (
    PowerIteration,
    InversePowerIteration,
    RayleighQuotientIteration,
    QREigenvalues,
    SVD,
)
from numethods.ode import (
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
)
from numethods.differentiation import (
    ForwardDiff,
    BackwardDiff,
    CentralDiff,
    CentralDiff4th,
    SecondDerivative,
    RichardsonExtrap,
)
from numethods.fitting import (
    LinearFit,
    NonlinearFit,
    PolyFit,
    ExpFit,
    plot_fit,
    plot_residuals,
)
import math


def run_solver(Solver, name, **kwargs):
    # Test problem: y' = -2y + t, y(0) = 1
    f = lambda t, y: -2 * y + t
    t0, y0, h, t_end = 0.0, 1.0, 0.1, 2.0
    solver = Solver(f, t0, y0, h, **kwargs)
    ts, ys = solver.solve(t_end)
    print(f"{name:20s} final y({t_end}) ≈ {ys[-1]:.6f}")
    return ts, ys


def demo_ode():
    print("Solving y' = -2y + t,  y(0)=1,  over [0, 2]")
    print("=" * 60)

    run_solver(Euler, "Euler")
    run_solver(Heun, "Heun (Improved Euler)")
    run_solver(RK2, "RK2 (Midpoint)")
    run_solver(RK4, "RK4 (Classic)")
    run_solver(BackwardEuler, "Backward Euler")
    run_solver(ODETrapezoidal, "Trapezoidal Rule")
    run_solver(AdamsBashforth, "Adams-Bashforth (2-step)", order=2)
    run_solver(AdamsMoulton, "Adams-Moulton (2-step)")
    run_solver(PredictorCorrector, "Predictor-Corrector")
    run_solver(RK45, "RK45 (adaptive)", tol=1e-6)

    print("=" * 60)
    print("All solvers finished.")


def demo_differentiation():
    f = lambda x: x**3  # f'(x) = 3x^2, f''(x) = 6x
    x0 = 2.0

    print("Forward  :", ForwardDiff(f, x0))
    print("Backward :", BackwardDiff(f, x0))
    print("Central  :", CentralDiff(f, x0))
    print("4th order:", CentralDiff4th(f, x0))
    print("Richardson:", RichardsonExtrap(f, x0))
    print("Second derivative:", SecondDerivative(f, x0))


def demo_qr():
    A = Matrix([[2, -1], [1, 2], [1, 1]])

    b = Vector([1, 2, 3])

    # Factorization
    qr = QRHouseholder(A)
    Q, R = qr.Q, qr.R
    print("Q =", Q)
    print("R =", R)

    qrm = QRModifiedGramSchmidt(A)
    Qm, Rm = qrm.Q, qrm.R
    print("Qm =", Qm)
    print("Rm =", Rm)
    print("Q^T Q =", Q.T @ Q)
    print("Qm^T Qm =", Qm.T @ Qm)
    print("A=Qm Rm =", Qm @ Rm)
    print("A=Q R =", Q @ R)

    # Solve Ax = b (least squares, since A is tall)
    x_ls = LeastSquaresSolver(A, b).solve()
    print("Least squares solution:", x_ls)


def demo_eigen():
    A = Matrix([[4, 1, 1], [1, 3, 0], [1, 0, 2]])
    lam, v = PowerIteration(A, tol=1e-12, verbose=True).solve()
    print("Power iteration:", lam, v)

    lam_min, v_min = InversePowerIteration(
        A, shift=0.0, tol=1e-12, verbose=True
    ).solve()
    print("Inverse power:", lam_min, v_min)

    lam_rqi, v_rqi = RayleighQuotientIteration(A, tol=1e-12, verbose=True).solve()
    print("RQI:", lam_rqi, v_rqi)

    Aq = QREigenvalues(A, tol=1e-12).solve()
    print("QR diag eigs:", [Aq.data[i][i] for i in range(3)])

    M = Matrix([[3, 1, 1], [-1, 3, 1], [1, 1, 3], [0, 2, 1]])
    U, S, V = SVD(M).solve()
    print("Singular values:", S)


def demo_linear_solvers():
    A = Matrix([[4, -1, 0], [-1, 4, -1], [0, -1, 3]])
    b = Vector([15, 10, 10])

    print("LU:", LUDecomposition(A).solve(b))
    print("Gauss-Jordan:", GaussJordan(A).solve(b))
    print("Cholesky:", Cholesky(A).solve(b))
    print("Jacobi:", Jacobi(A, b, tol=1e-12).solve())
    print("Gauss-Seidel:", GaussSeidel(A, b, tol=1e-12).solve())


def demo_roots():
    f = lambda x: x**2 - 2
    df = lambda x: 2 * x

    # Newton
    steps = NewtonRoot(f, df, x0=1.0).trace()
    print("Newton Method Trace (x^2 - 2):")
    print_trace(steps)

    # Secant
    steps = Secant(f, 0, 2).trace()
    print("\nSecant Method Trace (x^2 - 2):")
    print_trace(steps)

    # Bisection
    steps = Bisection(f, 0, 2).trace()
    print("\nBisection Method Trace (x^2 - 2):")
    print_trace(steps)

    # Fixed-point: solve
    g = lambda x: 0.5 * (x + 2 / x)
    steps = FixedPoint(g, 1.0).trace()
    print("\nFixed-Point Iteration Trace (x^2 - 2):")
    print_trace(steps)


def demo_interpolation():
    x = [0, 1, 2, 3]
    y = [1, 2, 0, 5]
    newt = NewtonInterpolation(x, y)
    lagr = LagrangeInterpolation(x, y)
    t = 1.5
    print("Newton interpolation at", t, "=", newt.evaluate(t))
    print("Lagrange interpolation at", t, "=", lagr.evaluate(t))


def demo_quadrature():
    f = lambda x: x**2
    I1 = Trapezoidal(f, 0, 1, n=100).integrate()
    I2 = Simpson(f, 0, 1, n=100).integrate()
    I3 = GaussLegendre(f, 0, 1, n=2).integrate()

    print("Trapezoidal integral of x^2 over [0,1]:", I1)
    print("Simpson integral of x^2 over [0,1]:", I2)
    print("Gauss-Legendre integral of x^2 over [0,1]:", I3)


def demo_fitting():
    x = [0, 1, 2, 3, 4]
    y = [1, 2.7, 7.4, 20.1, 54.6]

    # Polynomial fit (degree 2)
    poly = PolyFit(x, y, degree=2)

    # Exponential fit
    expfit = ExpFit(x, y)

    # Linear fit with a chosen basis (example: [1, x])
    basis = [lambda t: 1.0, lambda t: t]
    lin = LinearFit(x, y, basis)

    # Nonlinear exponential fit (Gauss–Newton / LM)
    def model(x, params):
        a, b = params
        return a * math.exp(b * x)

    nonlin = NonlinearFit(
        model, x, y, init_params=[1.0, 0.8], lam=1e-3, max_iter=50, verbose=True
    )

    # Plot all fits
    plot_fit(
        x,
        y,
        [poly, lin, expfit, nonlin],
        labels=["Polynomial", "Linear (1,x)", "Exponential", "Nonlinear"],
    )

    # Line plot (current style)
    plot_residuals(
        x,
        y,
        [poly, lin, expfit, nonlin],
        labels=["Polynomial", "Linear (1,x)", "Exponential", "Nonlinear"],
        mode="line",
    )
    poly.summary()
    poly.trace()
    lin.summary()
    lin.trace()
    expfit.summary()
    expfit.trace()
    nonlin.summary()
    nonlin.trace()

    # Bar chart of absolute residuals
    # plot_residuals(
    #     x,
    #     y,
    #     [poly, expfit, nonlin],
    #     labels=["Polynomial", "Exponential", "Nonlinear"],
    #     mode="bar",
    # )


if __name__ == "__main__":
    demo_qr()
    demo_eigen()
    demo_linear_solvers()
    demo_roots()
    demo_interpolation()
    demo_quadrature()
    demo_ode()
    demo_differentiation()
    demo_fitting()
