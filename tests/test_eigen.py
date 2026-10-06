import math
import random

import pytest

from numethods import (
    Matrix,
    Vector,
    PowerIteration,
    InversePowerIteration,
    RayleighQuotientIteration,
    QREigenvalues,
    SVD,
)
from numethods.exceptions import ConvergenceError

S3 = Matrix([[6, 2, 1], [2, 3, 1], [1, 1, 1]])
B = Matrix([[2, 1, 0], [1, 3, 1], [0, 1, 4]])  # eigenvalues 3 - sqrt(3), 3, 3 + sqrt(3)


def residual(A, lam, v):
    Av = A @ v
    return max(abs(a - lam * x) for a, x in zip(Av, v))


def test_power_iteration_default_start_finds_dominant_eigenvalue():
    # [1, 1] is the eigenvector of the *smaller* eigenvalue here.
    lam, v = PowerIteration(Matrix([[2, -1], [-1, 2]])).solve()
    assert lam == pytest.approx(3.0)
    lam, v = PowerIteration(B, tol=1e-14).solve()
    assert lam == pytest.approx(3 + math.sqrt(3))
    assert residual(B, lam, v) < 1e-6


def test_power_iteration_rejects_zero_start():
    with pytest.raises(ValueError):
        PowerIteration(B).solve(Vector([0, 0, 0]))


def test_inverse_power_iteration():
    lam, v = InversePowerIteration(B, shift=0.0, tol=1e-12).solve()
    assert lam == pytest.approx(3 - math.sqrt(3))
    lam, v = InversePowerIteration(B, shift=2.9, tol=1e-12).solve()
    assert lam == pytest.approx(3.0)


def test_inverse_power_iteration_with_exact_eigenvalue_shift():
    ip = InversePowerIteration(Matrix([[2, 0], [0, 3]]), shift=2.0)
    lam, v = ip.solve()
    assert lam == pytest.approx(2.0)
    assert ip.effective_shift != 2.0


def test_rayleigh_quotient_iteration_handles_exact_convergence():
    lam, v = RayleighQuotientIteration(B).solve()
    assert residual(B, lam, v) < 1e-10
    lam, v = RayleighQuotientIteration(Matrix([[2, 0], [0, 3]])).solve(Vector([1.0, 0.0]))
    assert lam == 2.0


def test_qr_eigenvalues_symmetric():
    q = QREigenvalues(S3)
    ev = q.eigenvalues()
    assert sum(ev) == pytest.approx(10.0)
    T = q.T
    for i in range(3):
        for j in range(i):
            assert T[i, j] == 0.0
    lam_pi, _ = PowerIteration(S3, tol=1e-14).solve()
    assert ev[0] == pytest.approx(lam_pi)


@pytest.mark.parametrize(
    "M, expected",
    [
        ([[0, 1], [1, 0]], [1.0, -1.0]),
        ([[0, -1], [1, 0]], [1j, -1j]),
        ([[0, 0, 1], [1, 0, 0], [0, 1, 0]],
         [1.0, complex(-0.5, math.sqrt(3) / 2), complex(-0.5, -math.sqrt(3) / 2)]),
    ],
)
def test_qr_eigenvalues_hard_cases(M, expected):
    ev = QREigenvalues(Matrix(M)).eigenvalues()
    assert len(ev) == len(expected)
    for a, b in zip(ev, expected):
        assert abs(a - b) < 1e-12


def test_qr_eigenvalues_random_matrices_match_trace_and_determinant():
    rng = random.Random(3)
    for n in (4, 7, 10):
        M = Matrix([[rng.uniform(-1, 1) for _ in range(n)] for _ in range(n)])
        ev = QREigenvalues(M).eigenvalues()
        trace = sum(M[i, i] for i in range(n))
        assert sum(ev).real == pytest.approx(trace, abs=1e-10)
        assert abs(sum(ev).imag) < 1e-10
        from numethods import LUDecomposition

        det = 1.0
        for z in ev:
            det *= z
        assert det.real == pytest.approx(LUDecomposition(M).det(), rel=1e-8, abs=1e-12)


def test_unshifted_qr_is_kept_for_teaching():
    ev = QREigenvalues(S3, shifted=False).eigenvalues()
    assert ev == pytest.approx(QREigenvalues(S3).eigenvalues())
    with pytest.raises(ConvergenceError):
        QREigenvalues(Matrix([[0, 1], [1, 0]]), shifted=False, max_iter=100).solve()


def check_svd(A):
    U, S, V = SVD(A).solve()
    k = min(A.m, A.n)
    assert U.shape() == (A.m, k) and V.shape() == (A.n, k) and len(S) == k
    assert list(S) == sorted(S, reverse=True)
    for M in (U.T @ U, V.T @ V):
        for i in range(k):
            for j in range(k):
                assert M[i, j] == pytest.approx(1.0 if i == j else 0.0, abs=1e-12)
    for i in range(A.m):
        for j in range(A.n):
            val = sum(U[i, r] * S[r] * V[j, r] for r in range(k))
            assert val == pytest.approx(A[i, j], abs=1e-12)
    return S


def test_svd_shapes_orthogonality_reconstruction():
    check_svd(Matrix([[3, 1, 0], [1, 3, 1], [0, 1, 2], [1, 0, 1]]))  # tall
    check_svd(Matrix([[3, 1, 0, 1], [1, 3, 1, 0]]))  # wide


def test_svd_rank_deficient_has_orthonormal_u():
    S = check_svd(Matrix([[1, 2], [2, 4], [3, 6]]))
    assert S[0] == pytest.approx(math.sqrt(70))
    assert S[1] == pytest.approx(0.0, abs=1e-14)


def test_svd_small_singular_values_are_accurate():
    # A^T A would square the condition number; Jacobi works on A directly.
    eps = 1e-9
    A = Matrix([[1, 1], [eps, 0], [0, eps]])
    s = SVD(A).singular_values()
    assert s[0] == pytest.approx(math.sqrt(2 + eps**2), rel=1e-14)
    assert s[1] == pytest.approx(eps, rel=1e-6)
