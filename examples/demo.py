from numethods import *

def demo_linear_solvers():
    A = Matrix([[4, -1, 0],
                [-1, 4, -1],
                [0, -1, 3]])
    b = Vector([15, 10, 10])

    print("LU:", LUDecomposition(A).solve(b))
    print("Gauss-Jordan:", GaussJordan(A).solve(b))
    print("Cholesky:", Cholesky(A).solve(b))
    print("Jacobi:", Jacobi(A, b, tol=1e-12).solve())
    print("Gauss-Seidel:", GaussSeidel(A, b, tol=1e-12).solve())

def demo_roots():
    f = lambda x: x**3 - x - 2
    df = lambda x: 3*x**2 - 1
    print("Bisection:", Bisection(f, 1, 2).solve())
    print("Secant:", Secant(f, 1.0, 2.0).solve())
    print("Newton root:", NewtonRoot(f, df, 1.5).solve())
    # A simple contraction for demonstration; not general-purpose
    g = lambda x: (x + 2/x**2) / 2
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
    demo_linear_solvers()
    demo_roots()
    demo_interpolation()
