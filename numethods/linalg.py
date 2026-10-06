from __future__ import annotations
from numbers import Integral, Real
from operator import mul
from typing import Iterable, Iterator, List, Optional, Tuple, Union
from .exceptions import NonSquareMatrixError, SingularMatrixError
from .utils import max_abs, pivot_tolerance

Number = float  # We'll use float throughout


class Vector:
    """Dense real vector stored as a list of floats."""

    def __init__(self, data: Iterable[Number]):
        self.data = [float(x) for x in data]

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, i: int) -> Number:
        return self.data[i]

    def __setitem__(self, i: int, value: Number) -> None:
        self.data[i] = float(value)

    def __iter__(self) -> Iterator[Number]:
        return iter(self.data)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Vector):
            return NotImplemented
        return self.data == other.data

    __hash__ = None  # mutable container

    def copy(self) -> "Vector":
        return Vector(self.data[:])

    def norm(self) -> Number:
        """1-norm: sum of absolute values (same as ``norm1``)."""
        return sum(abs(x) for x in self.data)

    def norm1(self) -> Number:
        """1-norm: sum of absolute values."""
        return self.norm()

    def norm_inf(self) -> Number:
        """Infinity norm: largest absolute value."""
        return max(abs(x) for x in self.data) if self.data else 0.0

    def norm2(self) -> Number:
        """Euclidean (2-) norm."""
        return sum(x * x for x in self.data) ** 0.5

    def __add__(self, other: "Vector") -> "Vector":
        if len(self) != len(other):
            raise ValueError("Vector dimensions must match for addition")
        return Vector([a + b for a, b in zip(self.data, other.data)])

    def __sub__(self, other: "Vector") -> "Vector":
        if len(self) != len(other):
            raise ValueError("Vector dimensions must match for subtraction")
        return Vector([a - b for a, b in zip(self.data, other.data)])

    def __neg__(self) -> "Vector":
        return Vector(-x for x in self.data)

    def __mul__(self, scalar: Number) -> "Vector":
        if not isinstance(scalar, Real):
            return NotImplemented
        return Vector(scalar * x for x in self.data)

    __rmul__ = __mul__

    def __truediv__(self, scalar: Number) -> "Vector":
        if not isinstance(scalar, Real):
            return NotImplemented
        return Vector(x / scalar for x in self.data)

    def dot(self, other: "Vector") -> Number:
        if len(self) != len(other):
            raise ValueError("Vector dimensions must match for dot product")
        return sum(map(mul, self.data, other.data), 0.0)

    def __repr__(self):
        return f"Vector({self.data})"


class Matrix:
    """Dense real matrix stored as a list of row lists."""

    def __init__(self, rows: List[Iterable[Number]]):
        data = [list(map(float, row)) for row in rows]
        if not data:
            self.m, self.n = 0, 0
        else:
            n = len(data[0])
            for r in data:
                if len(r) != n:
                    raise ValueError("All rows must have the same length")
            self.m, self.n = len(data), n
        self.data = data

    @classmethod
    def _from_rows(cls, data: List[List[float]], n: Optional[int] = None) -> "Matrix":
        """Wrap freshly built float rows without copying or re-validating them.

        ``n`` keeps the column count of matrices that have no rows.
        """
        M = cls.__new__(cls)
        M.data = data
        M.m = len(data)
        M.n = len(data[0]) if data else (n or 0)
        return M

    @staticmethod
    def zeros(m: int, n: int) -> "Matrix":
        return Matrix._from_rows([[0.0] * n for _ in range(m)], n)

    @staticmethod
    def identity(n: int) -> "Matrix":
        A = Matrix.zeros(n, n)
        for i in range(n):
            A.data[i][i] = 1.0
        return A

    def copy(self) -> "Matrix":
        return Matrix._from_rows([row[:] for row in self.data], self.n)

    def shape(self) -> Tuple[int, int]:
        return self.m, self.n

    def __getitem__(self, idx):
        """``A[i, j]`` returns an entry; ``A[i]`` returns a copy of row ``i`` as a Vector."""
        if isinstance(idx, tuple):
            i, j = idx
            return self.data[i][j]
        if isinstance(idx, Integral):
            return Vector(self.data[idx])
        raise TypeError("Matrix indices must be A[i, j] (entry) or A[i] (row copy)")

    def __setitem__(self, idx, value):
        """``A[i, j] = v`` sets an entry; ``A[i] = row`` replaces row ``i``."""
        if isinstance(idx, tuple):
            i, j = idx
            self.data[i][j] = float(value)
            return
        if isinstance(idx, Integral):
            row = [float(v) for v in value]
            if len(row) != self.n:
                raise ValueError("Row length must match the number of columns")
            self.data[idx] = row
            return
        raise TypeError("Matrix indices must be A[i, j] (entry) or A[i] (row)")

    def __iter__(self) -> Iterator[Vector]:
        """Iterate over copies of the rows."""
        return (Vector(row) for row in self.data)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Matrix):
            return NotImplemented
        return self.shape() == other.shape() and self.data == other.data

    __hash__ = None  # mutable container

    def row(self, i: int) -> Vector:
        return Vector(self.data[i][:])

    def col(self, j: int) -> Vector:
        return Vector(self.data[i][j] for i in range(self.m))

    def norm(self) -> float:
        """Matrix 1-norm: max column sum (same as ``norm1``)."""
        if self.m == 0 or self.n == 0:
            return 0.0
        return max(
            sum(abs(self.data[i][j]) for i in range(self.m)) for j in range(self.n)
        )

    def norm1(self) -> float:
        """Matrix 1-norm: max column sum."""
        return self.norm()

    def norm_inf(self) -> float:
        """Matrix infinity norm: max row sum."""
        if self.m == 0 or self.n == 0:
            return 0.0
        return max(sum(abs(v) for v in row) for row in self.data)

    def norm2(self, tol: float = 1e-10, max_iter: int = 5000) -> float:
        """Spectral norm ||A||_2 = largest singular value (one-sided Jacobi SVD).

        ``tol`` and ``max_iter`` are forwarded to :class:`numethods.eigen.SVD`.
        """
        # lazy import, avoids circular import
        from .eigen import SVD

        if self.m == 0 or self.n == 0:
            return 0.0
        sigma = SVD(self, tol=tol, max_iter=max_iter).singular_values()
        return sigma[0]

    def norm_fro(self) -> Number:
        return (
            sum(self.data[i][j] ** 2 for i in range(self.m) for j in range(self.n))
            ** 0.5
        )

    def inverse(self) -> "Matrix":
        """Compute A^{-1} using LU decomposition with partial pivoting."""
        # lazy import, avoids circular import
        from .solvers import LUDecomposition

        if not self.is_square():
            raise NonSquareMatrixError("Inverse requires square matrix")
        return LUDecomposition(self).inverse()

    def condition_number(self, norm: str = "2") -> float:
        """
        Compute condition number κ(A) = ||A|| * ||A^{-1}||.
        norm: "1", "inf", or "2" (the 2-norm version uses σ_max / σ_min).
        Raises SingularMatrixError when A is singular to working precision.
        """
        if not self.is_square():
            raise NonSquareMatrixError("Condition number requires square matrix")

        if norm == "1":
            return self.norm() * self.inverse().norm()
        if norm == "inf":
            return self.norm_inf() * self.inverse().norm_inf()
        if norm == "2":
            # lazy import, avoids circular import
            from .eigen import SVD

            if self.n == 0:
                return 0.0
            sigma = SVD(self).singular_values()
            if sigma[-1] <= pivot_tolerance(sigma[0], self.n):
                raise SingularMatrixError("Matrix is singular to working precision")
            return sigma[0] / sigma[-1]
        raise ValueError("norm must be '1', 'inf', or '2'")

    def __add__(self, other: "Matrix") -> "Matrix":
        if not isinstance(other, Matrix):
            raise TypeError("Can only add Matrix with Matrix")
        if self.m != other.m or self.n != other.n:
            raise ValueError("Matrix dimensions must match for addition")
        return Matrix._from_rows(
            [[a + b for a, b in zip(ra, rb)] for ra, rb in zip(self.data, other.data)],
            self.n,
        )

    def __sub__(self, other: "Matrix") -> "Matrix":
        if not isinstance(other, Matrix):
            raise TypeError("Can only subtract Matrix with Matrix")
        if self.m != other.m or self.n != other.n:
            raise ValueError("Matrix dimensions must match for subtraction")
        return Matrix._from_rows(
            [[a - b for a, b in zip(ra, rb)] for ra, rb in zip(self.data, other.data)],
            self.n,
        )

    def __neg__(self) -> "Matrix":
        return Matrix._from_rows([[-v for v in row] for row in self.data], self.n)

    def transpose(self) -> "Matrix":
        if self.m == 0:
            return Matrix.zeros(self.n, 0)
        return Matrix._from_rows([list(col) for col in zip(*self.data)], self.m)

    T = property(transpose)

    def __matmul__(self, other: Union["Matrix", "Vector"]):
        if isinstance(other, Matrix):
            if self.n != other.m:
                raise ValueError(
                    f"Shape mismatch in A @ B: {self.shape()} @ {other.shape()}"
                )
            # Iterate over the columns of B as tuples (zip(*B)) instead of
            # indexing B[k][j] in the inner loop.
            cols = list(zip(*other.data)) if other.m else [()] * other.n
            return Matrix._from_rows(
                [[sum(map(mul, row, col), 0.0) for col in cols] for row in self.data],
                other.n,
            )
        elif isinstance(other, Vector):
            if self.n != len(other):
                raise ValueError(
                    f"Shape mismatch in A @ x: {self.shape()} @ ({len(other)},)"
                )
            x = other.data
            return Vector([sum(map(mul, row, x), 0.0) for row in self.data])
        else:
            raise TypeError("Unsupported operand for @ (expected Matrix or Vector)")

    def __mul__(self, s):
        if isinstance(s, Real):
            return Matrix._from_rows([[v * s for v in row] for row in self.data], self.n)
        raise TypeError("Use @ for matrix multiply; * is scalar")

    __rmul__ = __mul__

    def __truediv__(self, s):
        if isinstance(s, Real):
            return Matrix._from_rows([[v / s for v in row] for row in self.data], self.n)
        raise TypeError("Matrix can only be divided by a scalar")

    def is_square(self) -> bool:
        return self.m == self.n

    def augment(self, b: Vector) -> "Matrix":
        if self.m != len(b):
            raise ValueError("Dimension mismatch for augmentation")
        return Matrix([self.data[i] + [b[i]] for i in range(self.m)])

    def max_abs_in_col(self, col: int, start_row: int = 0) -> int:
        max_i = start_row
        max_val = abs(self.data[start_row][col])
        for i in range(start_row + 1, self.m):
            v = abs(self.data[i][col])
            if v > max_val:
                max_val, max_i = v, i
        return max_i

    def swap_rows(self, i: int, j: int) -> None:
        if i != j:
            self.data[i], self.data[j] = self.data[j], self.data[i]

    def __repr__(self):
        return f"Matrix({self.data})"


def forward_substitution(L: Matrix, b: Vector) -> Vector:
    """Solve Lx = b for x using forward substitution"""
    if not L.is_square():
        raise NonSquareMatrixError("L must be square")
    n = L.n
    if len(b) != n:
        raise ValueError("Dimension mismatch: len(b) must equal the size of L")
    tol = pivot_tolerance(max_abs(L.data), n)
    x = [0.0] * n
    for i in range(n):
        row = L.data[i]
        s = sum(row[j] * x[j] for j in range(i))
        if abs(row[i]) <= tol:
            raise SingularMatrixError("Zero pivot in forward substitution")
        x[i] = (b[i] - s) / row[i]
    return Vector(x)


def backward_substitution(U: Matrix, b: Vector) -> Vector:
    """Solve Ux = b for x using backward substitution"""
    if not U.is_square():
        raise NonSquareMatrixError("U must be square")
    n = U.n
    if len(b) != n:
        raise ValueError("Dimension mismatch: len(b) must equal the size of U")
    tol = pivot_tolerance(max_abs(U.data), n)
    x = [0.0] * n
    for i in reversed(range(n)):
        row = U.data[i]
        s = sum(row[j] * x[j] for j in range(i + 1, n))
        if abs(row[i]) <= tol:
            raise SingularMatrixError("Zero pivot in backward substitution")
        x[i] = (b[i] - s) / row[i]
    return Vector(x)
