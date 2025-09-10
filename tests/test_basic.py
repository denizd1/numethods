import math
from numethods import *

def approx_equal(a, b, tol=1e-8):
    return abs(a-b) <= tol*(1+abs(b))

def test_lu_gauss_seidel():
    A = Matrix([[10, -1, 2, 0],
                [-1, 11, -1, 3],
                [2, -1, 10, -1],
                [0, 3, -1, 8]])
    b = Vector([6, 25, -11, 15])
    x_lu = LUDecomposition(A).solve(b)
    x_gs = GaussSeidel(A, b, tol=1e-12).solve()
    for i in range(4):
        assert approx_equal(x_lu[i], x_gs[i])

def test_cholesky():
    A = Matrix([[4, 1, 1],
                [1, 3, 0],
                [1, 0, 2]])
    b = Vector([1, 2, 3])
    x = Cholesky(A).solve(b)
    Ax = [sum(A.data[i][j]*x[j] for j in range(3)) for i in range(3)]
    for i in range(3):
        assert approx_equal(Ax[i], b[i])

def test_roots():
    f = lambda x: x**3 - 2
    df = lambda x: 3*x**2
    r = NewtonRoot(f, df, 1.0).solve()
    assert approx_equal(r, 2**(1/3))
    b = Bisection(f, 0, 2).solve()
    assert approx_equal(b, 2**(1/3))

def test_interp():
    x = [0,1,2]
    y = [1,3,2]
    n = NewtonInterpolation(x,y)
    l = LagrangeInterpolation(x,y)
    for t in [0,0.5,1.7,2.0]:
        assert approx_equal(n.evaluate(t), l.evaluate(t), tol=1e-9)
