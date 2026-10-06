import math

import pytest

from numethods import Matrix, Vector, forward_substitution, backward_substitution
from numethods.exceptions import NonSquareMatrixError, SingularMatrixError


def test_vector_operators():
    u, v = Vector([1, 2, 3]), Vector([3, 2, 1])
    assert u + v == Vector([4, 4, 4])
    assert u - v == Vector([-2, 0, 2])
    assert -u == Vector([-1, -2, -3])
    assert 2 * u == u * 2 == Vector([2, 4, 6])
    assert u / 2 == Vector([0.5, 1, 1.5])
    assert list(u) == [1.0, 2.0, 3.0]
    assert u.dot(v) == 10.0
    assert u != v


def test_vector_norms():
    v = Vector([3, -4, 5])
    assert v.norm() == v.norm1() == 12.0
    assert v.norm2() == pytest.approx(math.sqrt(50))
    assert v.norm_inf() == 5.0


def test_vector_dot_dimension_mismatch_raises_value_error():
    with pytest.raises(ValueError):
        Vector([1, 2]).dot(Vector([1]))


def test_matrix_indexing_and_rows():
    A = Matrix([[1, 2], [3, 4]])
    assert A[0, 1] == 2.0
    assert A[1] == Vector([3, 4])
    A[0] = [5, 6]
    A[1, 1] = 7
    assert A == Matrix([[5, 6], [3, 7]])
    assert [row for row in A] == [Vector([5, 6]), Vector([3, 7])]
    with pytest.raises(ValueError):
        A[0] = [1, 2, 3]


def test_matrix_arithmetic():
    A = Matrix([[1, 2], [3, 4]])
    B = Matrix([[2, 0], [1, 2]])
    assert A @ B == Matrix([[4, 4], [10, 8]])
    assert A @ Vector([1, -1]) == Vector([-1, -1])
    assert A + B == Matrix([[3, 2], [4, 6]])
    assert A - B == Matrix([[-1, 2], [2, 2]])
    assert -A == Matrix([[-1, -2], [-3, -4]])
    assert 2 * A == A * 2 == Matrix([[2, 4], [6, 8]])
    assert A / 2 == Matrix([[0.5, 1], [1.5, 2]])
    assert A.T == Matrix([[1, 3], [2, 4]])
    with pytest.raises(ValueError):
        A @ Matrix([[1, 2, 3]])
    with pytest.raises(TypeError):
        A * A


def test_zeros_keeps_shape_for_empty_matrices():
    assert Matrix.zeros(0, 3).shape() == (0, 3)
    assert Matrix.zeros(0, 3).T.shape() == (3, 0)
    assert (Matrix.zeros(2, 0) @ Matrix.zeros(0, 3)) == Matrix.zeros(2, 3)


def test_norms_and_condition_number():
    A = Matrix([[2, -1], [-1, 2]])  # eigenvalues 1 and 3, eigenvector of 1 is [1, 1]
    assert A.norm2() == pytest.approx(3.0)
    assert A.condition_number("2") == pytest.approx(3.0)
    assert A.condition_number("1") == pytest.approx(3.0)
    assert A.condition_number("inf") == pytest.approx(3.0)
    B = Matrix([[1, -2], [3, 4]])
    assert B.norm() == B.norm1() == 6.0
    assert B.norm_inf() == 7.0
    assert B.norm_fro() == pytest.approx(math.sqrt(30))
    with pytest.raises(SingularMatrixError):
        Matrix([[1, 2], [2, 4]]).condition_number("2")
    with pytest.raises(NonSquareMatrixError):
        Matrix([[1, 2, 3]]).condition_number()


def test_inverse():
    A = Matrix([[4, 7], [2, 6]])
    Ainv = A.inverse()
    I = A @ Ainv
    for i in range(2):
        for j in range(2):
            assert I[i, j] == pytest.approx(1.0 if i == j else 0.0, abs=1e-14)


def test_substitution_is_scale_invariant():
    tiny = Matrix([[1e-20, 0.0], [1e-20, 1e-20]])
    x = forward_substitution(tiny, Vector([1e-20, 2e-20]))
    assert x == Vector([1.0, 1.0])
    with pytest.raises(SingularMatrixError):
        backward_substitution(Matrix([[1.0, 1.0], [0.0, 0.0]]), Vector([1, 1]))


def test_substitution_checks_rhs_length():
    with pytest.raises(ValueError):
        forward_substitution(Matrix([[1.0, 0.0], [1.0, 1.0]]), Vector([1.0]))
