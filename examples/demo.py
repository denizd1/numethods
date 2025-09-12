from numethods import *


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
