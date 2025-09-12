from __future__ import annotations
from typing import Iterable, Tuple, List, Union
from .exceptions import NonSquareMatrixError, SingularMatrixError

Number = float  # We'll use float throughout


class Vector:
    def __init__(self, data: Iterable[Number]):
        self.data = [float(x) for x in data]

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, i: int) -> Number:
        return self.data[i]

    def __setitem__(self, i: int, value: Number) -> None:
        self.data[i] = float(value)

    def copy(self) -> "Vector":
        return Vector(self.data[:])

    def norm_inf(self) -> Number:
        return max(abs(x) for x in self.data) if self.data else 0.0

    def norm2(self) -> Number:
        return sum(x * x for x in self.data) ** 0.5

    def __add__(self, other: "Vector") -> "Vector":
        assert len(self) == len(other)
        return Vector(a + b for a, b in zip(self.data, other.data))

    def __sub__(self, other: "Vector") -> "Vector":
        assert len(self) == len(other)
        return Vector(a - b for a, b in zip(self.data, other.data))

    def __mul__(self, scalar: Number) -> "Vector":
        return Vector(scalar * x for x in self.data)

    __rmul__ = __mul__

    def dot(self, other: "Vector") -> Number:
        assert len(self) == len(other)
        return sum(a * b for a, b in zip(self.data, other.data))

    def __repr__(self):
        return f"Vector({self.data})"


class Matrix:
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

    @staticmethod
    def zeros(m: int, n: int) -> "Matrix":
        return Matrix([[0.0] * n for _ in range(m)])

    @staticmethod
    def identity(n: int) -> "Matrix":
        A = Matrix.zeros(n, n)
        for i in range(n):
            A.data[i][i] = 1.0
        return A

    def copy(self) -> "Matrix":
        return Matrix([row[:] for row in self.data])

    def shape(self) -> Tuple[int, int]:
        return self.m, self.n

    def __getitem__(self, idx):
        i, j = idx
        return self.data[i][j]

    def __setitem__(self, idx, value):
        i, j = idx
        self.data[i][j] = float(value)

    def row(self, i: int) -> Vector:
        return Vector(self.data[i][:])

    def col(self, j: int) -> Vector:
        return Vector(self.data[i][j] for i in range(self.m))

    def transpose(self) -> "Matrix":
        return Matrix([[self.data[i][j] for i in range(self.m)] for j in range(self.n)])

    T = property(transpose)

    def __matmul__(self, other: Union["Matrix", "Vector"]):
        if isinstance(other, Matrix):
            if self.n != other.m:
                raise ValueError("dims")
            return Matrix(
                [
                    [
                        sum(self.data[i][k] * other.data[k][j] for k in range(self.n))
                        for j in range(other.n)
                    ]
                    for i in range(self.m)
                ]
            )
        elif isinstance(other, Vector):
            if self.n != len(other):
                raise ValueError("dims")
            return Vector(
                [
                    sum(self.data[i][k] * other[k] for k in range(self.n))
                    for i in range(self.m)
                ]
            )
        else:
            raise TypeError("Unsupported @")

    def __mul__(self, s):
        if isinstance(s, (int, float)):
            return Matrix([[v * s for v in row] for row in self.data])
        raise TypeError("Use @ for matrix multiply; * is scalar")

    __rmul__ = __mul__

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
    if not L.is_square():
        raise NonSquareMatrixError("L must be square")
    n = L.n
    x = [0.0] * n
    for i in range(n):
        s = sum(L.data[i][j] * x[j] for j in range(i))
        if abs(L.data[i][i]) < 1e-15:
            raise SingularMatrixError("Zero pivot in forward substitution")
        x[i] = (b[i] - s) / L.data[i][i]
    return Vector(x)


def backward_substitution(U: Matrix, b: Vector) -> Vector:
    if not U.is_square():
        raise NonSquareMatrixError("U must be square")
    n = U.n
    x = [0.0] * n
    for i in reversed(range(n)):
        s = sum(U.data[i][j] * x[j] for j in range(i + 1, n))
        if abs(U.data[i][i]) < 1e-15:
            raise SingularMatrixError("Zero pivot in backward substitution")
        x[i] = (b[i] - s) / U.data[i][i]
    return Vector(x)
