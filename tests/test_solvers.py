import pytest

from numethods import (
    Matrix,
    Vector,
    LUDecomposition,
    GaussJordan,
    Jacobi,
    GaussSeidel,
    SOR,
    Cholesky,
)
from numethods.exceptions import (
    NonSquareMatrixError,
    SingularMatrixError,
    NotSymmetricError,
    NotPositiveDefiniteError,
    ConvergenceError,
)

A = Matrix([[10, -1, 2, 0], [-1, 11, -1, 3], [2, -1, 10, -1], [0, 3, -1, 8]])
b = Vector([6, 25, -11, 15])
x_exact = [1.0, 2.0, -1.0, 1.0]


def assert_close(x, ref, tol=1e-9):
    assert len(x) == len(ref)
    for a, r in zip(x, ref):
        assert a == pytest.approx(r, abs=tol)


@pytest.mark.parametrize("solver", [LUDecomposition, GaussJordan])
def test_direct_solvers(solver):
    assert_close(solver(A).solve(b), x_exact, 1e-12)


def test_lu_factors_permutation_det_inverse():
    M = Matrix([[0, 2, 1], [1, 1, 1], [2, 1, 0]])
    lu = LUDecomposition(M)
    PA = lu.P @ M
    LU = lu.L @ lu.U
    for i in range(3):
        for j in range(3):
            assert PA[i, j] == pytest.approx(LU[i, j], abs=1e-14)
        for j in range(i):
            assert lu.U[i, j] == 0.0
    assert lu.det() == pytest.approx(3.0)
    inv = lu.inverse()
    I = M @ inv
    for i in range(3):
        for j in range(3):
            assert I[i, j] == pytest.approx(1.0 if i == j else 0.0, abs=1e-14)


def test_lu_record_steps_is_optional():
    assert LUDecomposition(A).steps == []
    lu = LUDecomposition(A, record_steps=True)
    assert len(lu.steps) == 4


def test_lu_singular_and_scale_invariance():
    with pytest.raises(SingularMatrixError):
        LUDecomposition(Matrix([[1, 2], [2, 4]]))
    with pytest.raises(SingularMatrixError):
        LUDecomposition(Matrix([[1, 2, 3], [4, 5, 6], [7, 8, 9]]))
    x = LUDecomposition(Matrix([[1e-20, 0], [0, 1e-20]])).solve(Vector([1e-20, 2e-20]))
    assert_close(x, [1.0, 2.0])
    with pytest.raises(NonSquareMatrixError):
        LUDecomposition(Matrix([[1, 2, 3]]))
    with pytest.raises(ValueError):
        LUDecomposition(A).solve(Vector([1, 2]))


def test_gauss_jordan_steps_reset_between_solves():
    gj = GaussJordan(Matrix([[2, 1], [1, 3]]), record_steps=True)
    gj.solve(Vector([1, 2]))
    gj.solve(Vector([3, 4]))
    assert len(gj.steps) == 2


@pytest.mark.parametrize(
    "make", [lambda: Jacobi(A, b, tol=1e-12), lambda: GaussSeidel(A, b, tol=1e-12),
             lambda: SOR(A, b, omega=1.1, tol=1e-12)]
)
def test_iterative_solvers(make):
    s = make()
    assert_close(s.solve(), x_exact, 1e-9)
    n1 = len(s.history)
    s.solve()
    assert len(s.history) == n1  # history is reset by every solve()


def test_gauss_seidel_is_sor_with_omega_one():
    x1 = GaussSeidel(A, b, tol=1e-12).solve()
    x2 = SOR(A, b, omega=1.0, tol=1e-12).solve()
    assert x1 == x2


def test_iterative_errors():
    with pytest.raises(SingularMatrixError):
        Jacobi(Matrix([[0, 1], [1, 0]]), Vector([1, 1])).solve()
    with pytest.raises(ConvergenceError):
        Jacobi(Matrix([[1, 2], [3, 1]]), Vector([1, 1]), max_iter=50).solve()
    with pytest.raises(ValueError):
        SOR(A, b, omega=2.5)
    with pytest.raises(ValueError):
        Jacobi(A, b).solve(Vector([0, 0]))


def test_cholesky():
    S = Matrix([[4, 1, 1], [1, 3, 0], [1, 0, 2]])
    ch = Cholesky(S)
    LLT = ch.L @ ch.L.T
    for i in range(3):
        for j in range(3):
            assert LLT[i, j] == pytest.approx(S[i, j], abs=1e-14)
    x = ch.solve(Vector([1, 2, 3]))
    assert_close(S @ x, [1, 2, 3], 1e-13)


def test_cholesky_symmetry_check_is_relative():
    big = Matrix([[1e10, 1e10 * (1 + 1e-15)], [1e10, 2e10]])
    Cholesky(big)  # rounding-level asymmetry is accepted
    with pytest.raises(NotSymmetricError):
        Cholesky(Matrix([[4, 1], [2, 3]]))
    with pytest.raises(NotPositiveDefiniteError):
        Cholesky(Matrix([[1, 2], [2, 1]]))
