from numethods import Matrix, Vector
from numethods import (
    QRHouseholder,
    QRModifiedGramSchmidt,
    LeastSquaresSolver,
    PowerIteration,
    InversePowerIteration,
    RayleighQuotientIteration,
    QREigenvalues,
    SVD,
    LUDecomposition,
    GaussJordan,
    Cholesky,
    Jacobi,
    GaussSeidel,
    Bisection,
    Secant,
    NewtonRoot,
    FixedPoint,
    NewtonInterpolation,
    LagrangeInterpolation,
    Euler,
    Heun,
    RK2,
    RK4,
    BackwardEuler,
    Trapezoidal,
    AdamsBashforth,
    AdamsMoulton,
    PredictorCorrector,
    RK45,
)

# Test problem: y' = -2y + t, y(0) = 1
f = lambda t, y: -2 * y + t
t0, y0, h, t_end = 0.0, 1.0, 0.1, 2.0


def run_solver(Solver, name, **kwargs):
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
    run_solver(Trapezoidal, "Trapezoidal Rule")
    run_solver(AdamsBashforth, "Adams-Bashforth (2-step)", order=2)
    run_solver(AdamsMoulton, "Adams-Moulton (2-step)")
    run_solver(PredictorCorrector, "Predictor-Corrector")
    run_solver(RK45, "RK45 (adaptive)", tol=1e-6)

    print("=" * 60)
    print("All solvers finished.")


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

    # Solve Ax = b (least squares, since A is tall)
    x_ls = LeastSquaresSolver(A, b).solve()
    print("Least squares solution:", x_ls)


def demo_eigen():
    A = Matrix([[4, 1, 1], [1, 3, 0], [1, 0, 2]])
    lam, v = PowerIteration(A, tol=1e-12).solve()
    print("Power iteration:", lam, v)

    lam_min, v_min = InversePowerIteration(A, shift=0.0, tol=1e-12).solve()
    print("Inverse power:", lam_min, v_min)

    lam_rqi, v_rqi = RayleighQuotientIteration(A, tol=1e-12).solve()
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
    f = lambda x: x**3 - x - 2
    df = lambda x: 3 * x**2 - 1
    print("Bisection:", Bisection(f, 1, 2).solve())
    print("Secant:", Secant(f, 1.0, 2.0).solve())
    print("Newton root:", NewtonRoot(f, df, 1.5).solve())
    # A simple contraction for demonstration; not general-purpose
    g = lambda x: (x + 2 / x**2) / 2
    print("Fixed point (demo):", FixedPoint(g, 1.5).solve())


def demo_interpolation():
    x = [0, 1, 2, 3]
    y = [1, 2, 0, 5]
    newt = NewtonInterpolation(x, y)
    lagr = LagrangeInterpolation(x, y)
    t = 1.5
    print("Newton interpolation at", t, "=", newt.evaluate(t))
    print("Lagrange interpolation at", t, "=", lagr.evaluate(t))


if __name__ == "__main__":
    demo_qr()
    demo_eigen()
    demo_linear_solvers()
    demo_roots()
    demo_interpolation()
    demo_ode()
