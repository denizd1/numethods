import pytest

from numethods import (
    Matrix,
    Vector,
    QRGramSchmidt,
    QRModifiedGramSchmidt,
    QRHouseholder,
    QRSolver,
    LeastSquaresSolver,
)
from numethods.exceptions import DomainError, SingularMatrixError

A = Matrix([[2, -1], [1, 2], [1, 1]])


def is_identity(M, tol=1e-12):
    return all(
        abs(M[i, j] - (1.0 if i == j else 0.0)) <= tol for i in range(M.m) for j in range(M.n)
    )


@pytest.mark.parametrize("cls", [QRGramSchmidt, QRModifiedGramSchmidt, QRHouseholder])
def test_qr_factorizations(cls):
    qr = cls(A)
    assert is_identity(qr.Q.T @ qr.Q)
    QR = qr.Q @ qr.R
    for i in range(A.m):
        for j in range(A.n):
            assert QR[i, j] == pytest.approx(A[i, j], abs=1e-14)
    for i in range(qr.R.m):
        for j in range(min(i, qr.R.n)):
            assert qr.R[i, j] == 0.0  # exactly upper triangular


def test_classical_gram_schmidt_uses_original_columns():
    # For CGS, R[k][j] = q_k . a_j with the *original* column a_j.
    qr = QRGramSchmidt(A)
    q0 = qr.Q.col(0)
    assert qr.R[0, 1] == pytest.approx(q0.dot(A.col(1)), abs=0)


@pytest.mark.parametrize("cls", [QRGramSchmidt, QRModifiedGramSchmidt])
def test_gram_schmidt_detects_dependent_columns(cls):
    with pytest.raises(SingularMatrixError):
        cls(Matrix([[1, 2], [2, 4], [3, 6]]))


def test_least_squares_and_qr_solver_on_tall_systems():
    b = Vector([1, 2, 3])
    x_ls = LeastSquaresSolver(A, b).solve()
    assert list(x_ls) == pytest.approx([36 / 35, 29 / 35])
    for cls in (QRGramSchmidt, QRModifiedGramSchmidt, QRHouseholder):
        assert list(QRSolver(cls(A)).solve(b)) == pytest.approx([36 / 35, 29 / 35])


def test_least_squares_input_checks():
    with pytest.raises(DomainError):
        LeastSquaresSolver(Matrix([[1, 2, 3]]), Vector([1]))
    with pytest.raises(ValueError):
        LeastSquaresSolver(A, Vector([1, 2]))
    with pytest.raises(SingularMatrixError):
        LeastSquaresSolver(Matrix([[1, 2], [2, 4], [3, 6]]), Vector([1, 2, 3])).solve()
