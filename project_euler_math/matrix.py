from __future__ import annotations

from abc import ABC, abstractmethod
from itertools import product
from math import prod
from numbers import Integral
from operator import (
    lt,
    le,
    eq,
    ne,
    ge,
    gt,
    add,
    sub,
    mul,
    truediv,
    floordiv,
    mod,
    lshift,
    rshift,
    and_,
    xor,
    or_,
    neg,
    pos,
    invert,
)
from typing import TypeVar, Iterator, Generic, Iterable, Callable, Sequence

T = TypeVar("T")
R = TypeVar("R")

_VectorKey = int | slice
_MatrixKey = tuple[int | slice, int | slice]


def _zero(n):
    return 0 * n


def _one(n):
    return n**0


class BaseVector(ABC, Generic[T]):
    """
    Vector base class. The data layout is unspecified.
    """

    @property
    @abstractmethod
    def shape(self) -> tuple[int]:
        raise NotImplementedError

    def __len__(self) -> int:
        return self.shape[0]

    @classmethod
    @abstractmethod
    def from_function(cls, f: Callable[[int], T], n: int) -> BaseVector[T]:
        raise NotImplementedError

    @classmethod
    def zeros(cls, n: int) -> BaseVector[int]:
        return cls.full(n, 0)

    @classmethod
    def ones(cls, n: int) -> BaseVector[int]:
        return cls.full(n, 1)

    @classmethod
    def full(cls, n: int, x: T) -> BaseVector[T]:
        return cls.from_function(lambda i: x, n)

    @classmethod
    @abstractmethod
    def stack(cls, tup: Sequence[BaseVector[T]]) -> BaseVector[T]:
        raise NotImplementedError

    @abstractmethod
    def __getitem__(self, key: _VectorKey) -> T | BaseVector[T]:
        raise NotImplementedError

    @abstractmethod
    def __setitem__(self, key: _VectorKey, value) -> None:
        raise NotImplementedError

    @abstractmethod
    def __iter__(self) -> Iterator[T]:
        raise NotImplementedError

    def __contains__(self, item) -> bool:
        return item in iter(self)

    @abstractmethod
    def copy(self):
        raise NotImplementedError

    @abstractmethod
    def as_list(self) -> list[T]:
        raise NotImplementedError

    @abstractmethod
    def _left_op(self, other, op):
        raise NotImplementedError

    @abstractmethod
    def _right_op(self, other, op):
        raise NotImplementedError

    def __lt__(self, other: T | BaseVector[T]) -> BaseVector[bool]:
        return self._left_op(other, lt)

    def __le__(self, other: T | BaseVector[T]) -> BaseVector[bool]:
        return self._left_op(other, le)

    def __eq__(self, other: T | BaseVector[T]) -> BaseVector[bool]:
        return self._left_op(other, eq)

    def __ne__(self, other: T | BaseVector[T]) -> BaseVector[bool]:
        return self._left_op(other, ne)

    def __ge__(self, other: T | BaseVector[T]) -> BaseVector[bool]:
        return self._left_op(other, ge)

    def __gt__(self, other: T | BaseVector[T]) -> BaseVector[bool]:
        return self._left_op(other, gt)

    def __add__(self, other: T | BaseVector[T]) -> BaseVector[T]:
        return self._left_op(other, add)

    def __sub__(self, other: T | BaseVector[T]) -> BaseVector[T]:
        return self._left_op(other, sub)

    def __mul__(self, other: T | BaseVector[T]) -> BaseVector[T]:
        return self._left_op(other, mul)

    def __truediv__(self, other: T | BaseVector[T]) -> BaseVector[T]:
        return self._left_op(other, truediv)

    def __floordiv__(self, other: T | BaseVector[T]) -> BaseVector[T]:
        return self._left_op(other, floordiv)

    def __mod__(self, other: T | BaseVector[T]) -> BaseVector[T]:
        return self._left_op(other, mod)

    def __pow__(self, power: T | BaseVector[T]) -> BaseVector[T]:
        return self._left_op(power, pow)

    def __lshift__(self, other: T | BaseVector[T]) -> BaseVector[T]:
        return self._left_op(other, lshift)

    def __rshift__(self, other: T | BaseVector[T]) -> BaseVector[T]:
        return self._left_op(other, rshift)

    def __and__(self, other: T | BaseVector[T]) -> BaseVector[T]:
        return self._left_op(other, and_)

    def __xor__(self, other: T | BaseVector[T]) -> BaseVector[T]:
        return self._left_op(other, xor)

    def __or__(self, other: T | BaseVector[T]) -> BaseVector[T]:
        return self._left_op(other, or_)

    def __radd__(self, other: T) -> BaseVector[T]:
        return self._right_op(other, add)

    def __rsub__(self, other: T) -> BaseVector[T]:
        return self._right_op(other, sub)

    def __rmul__(self, other: T) -> BaseVector[T]:
        return self._right_op(other, mul)

    def __rtruediv__(self, other: T) -> BaseVector[T]:
        return self._right_op(other, truediv)

    def __rfloordiv__(self, other: T) -> BaseVector[T]:
        return self._right_op(other, floordiv)

    def __rmod__(self, other: T) -> BaseVector[T]:
        return self._right_op(other, mod)

    def __rpow__(self, power: T) -> BaseVector[T]:
        return self._right_op(power, pow)

    def __rlshift__(self, other: T) -> BaseVector[T]:
        return self._right_op(other, lshift)

    def __rrshift__(self, other: T) -> BaseVector[T]:
        return self._right_op(other, rshift)

    def __rand__(self, other: T) -> BaseVector[T]:
        return self._right_op(other, and_)

    def __rxor__(self, other: T) -> BaseVector[T]:
        return self._right_op(other, xor)

    def __ror__(self, other: T) -> BaseVector[T]:
        return self._right_op(other, or_)

    @abstractmethod
    def map(self, f: Callable[[T], R]) -> BaseVector[R]:
        raise NotImplementedError

    def __neg__(self) -> BaseVector[T]:
        return self.map(neg)

    def __pos__(self) -> BaseVector[T]:
        return self.map(pos)

    def __abs__(self) -> BaseVector[T]:
        return self.map(abs)

    def __invert__(self) -> BaseVector[T]:
        return self.map(invert)

    @abstractmethod
    def __matmul__(self, other: BaseMatrix[T]) -> BaseVector[T]:
        raise NotImplementedError

    @abstractmethod
    def __rmatmul__(self, other: BaseMatrix[T]) -> BaseVector[T]:
        raise NotImplementedError

    def __str__(self) -> str:
        return "[" + " ".join(map(repr, self)) + "]"

    def __repr__(self) -> str:
        return type(self).__name__ + f"({self.as_list()!r})"


class Vector(BaseVector[T]):
    """
    Vector class backed by all entries in a list.
    """

    _vec: list[T]

    __slots__ = "_vec"

    @property
    def shape(self):
        return (len(self._vec),)

    @classmethod
    def _create(cls, vec):
        self = cls.__new__(cls)
        self._vec = vec
        return self

    def __init__(self, vec: Iterable[T]) -> None:
        try:
            vec = list(vec)
        except TypeError:
            raise TypeError("vec must be an iterable")

        self._vec = vec

    @classmethod
    def from_function(cls, f, n):
        vec = [f(i) for i in range(n)]
        return cls._create(vec)

    @classmethod
    def stack(cls, tup):
        if not tup:
            raise ValueError("at least one vector is required")

        stack = [x for vec in tup for x in vec]
        return cls._create(stack)

    def __getitem__(self, key):
        if isinstance(key, (int, slice)):
            return self._vec[key]
        elif isinstance(key, Sequence):
            vec = self._vec
            selected = [vec[k] for k in key]
            return self._create(selected)
        else:
            raise TypeError

    def __setitem__(self, key, value):
        if isinstance(key, (int, slice)):
            self._vec[key] = value
        elif isinstance(key, Sequence):
            value = self.__class__(value)
            vec = self._vec
            for k, x in zip(key, value):
                vec[k] = x
        else:
            raise TypeError

    def __iter__(self):
        return iter(self._vec)

    def copy(self):
        return self._create(list(self._vec))

    def as_list(self):
        return self._vec

    def _left_op(self, other, op):
        if isinstance(other, Vector):
            if len(self) != len(other):
                raise ValueError(
                    "vector shapes do not match: " "{len(self)}, {len(other)}"
                )
            return self._create(list(map(op, self, other)))

        else:
            return self._create([op(x, other) for x in self])

    def _right_op(self, other, op):
        return self._create([op(other, x) for x in self])

    def map(self, f):
        return self._create([f(x) for x in self])

    def __matmul__(self, other):
        if isinstance(other, BaseVector):
            if len(self) != len(other):
                raise ValueError(
                    "vector shapes are not compatible: " f"{len(self)}, {len(other)}"
                )

            return sum(map(mul, self._vec, other))

        elif isinstance(other, BaseMatrix):
            if len(self) != other.nrows:
                raise ValueError(
                    "vector and matrix shapes are not compatible: "
                    f"{len(self)}, {other.shape}"
                )

            cols = [other.col_list(j) for j in range(other.ncols)]

            def entry(j):
                return sum(map(mul, self._vec, cols[j]))

            return self.from_function(entry, len(self))

        else:
            return NotImplemented

    def __rmatmul__(self, other):
        if other.ncols != len(self):
            raise ValueError(
                "matrix and vector shapes are not compatible: "
                f"{other.shape}, {len(self)}"
            )

        else:
            rows = [other.row_list(i) for i in range(other.nrows)]

            def entry(i):
                return sum(map(mul, rows[i], self._vec))

            return self.from_function(entry, len(self))


class BaseMatrix(ABC, Generic[T]):
    """
    Matrix base class. The data layout is unspecified.
    """

    @property
    @abstractmethod
    def shape(self) -> tuple[int, int]:
        raise NotImplementedError

    @property
    def nrows(self) -> int:
        return self.shape[0]

    @property
    def ncols(self) -> int:
        return self.shape[1]

    def __len__(self) -> int:
        return self.nrows * self.ncols

    def is_square(self) -> bool:
        return self.ncols == self.nrows

    @classmethod
    @abstractmethod
    def from_function(
        cls, function: Callable[[int, int], T], shape: tuple[int, int]
    ) -> BaseMatrix[T]:
        raise NotImplementedError

    @classmethod
    def zeros(cls, shape: tuple[int, int]) -> BaseMatrix[int]:
        return cls.full(shape, 0)

    @classmethod
    def ones(cls, shape: tuple[int, int]) -> BaseMatrix[int]:
        return cls.full(shape, 1)

    @classmethod
    def full(cls, shape: tuple[int, int], x: T) -> BaseMatrix[T]:
        return cls.from_function(lambda i, j: x, shape)

    @classmethod
    def identity(cls, n: int) -> BaseMatrix[int]:
        return cls.from_function(lambda i, j: 1 if i == j else 0, (n, n))

    @classmethod
    @abstractmethod
    def hstack(cls, tup: Sequence[BaseMatrix[T]]) -> BaseMatrix[T]:
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def vstack(cls, tup: Sequence[BaseMatrix[T]]) -> BaseMatrix[T]:
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def kronecker(cls, a: BaseMatrix[T], b: BaseMatrix[T]) -> BaseMatrix[T]:
        raise NotImplementedError

    @abstractmethod
    def __getitem__(self, key: _MatrixKey) -> T | BaseVector[T] | BaseMatrix[T]:
        raise NotImplementedError

    @abstractmethod
    def __setitem__(self, key: _MatrixKey, value) -> None:
        raise NotImplementedError

    @abstractmethod
    def __iter__(self) -> Iterator[T]:
        raise NotImplementedError

    def __contains__(self, item) -> bool:
        return item in iter(self)

    @abstractmethod
    def copy(self):
        raise NotImplementedError

    @abstractmethod
    def row_list(self, i: int) -> list[T]:
        raise NotImplementedError

    def row(self, i: int) -> BaseVector[T]:
        return Vector(self.row_list(i))

    @abstractmethod
    def col_list(self, j: int) -> list[T]:
        raise NotImplementedError

    def col(self, j: int) -> BaseVector[T]:
        return Vector(self.col_list(j))

    @abstractmethod
    def diag_list(self) -> list[T]:
        raise NotImplementedError

    def diag(self) -> BaseVector[T]:
        return Vector(self.diag_list())

    @abstractmethod
    def tril(self, k: int = 0) -> BaseMatrix[T]:
        raise NotImplementedError

    @abstractmethod
    def triu(self, k: int = 0) -> BaseMatrix[T]:
        raise NotImplementedError

    def as_list(self) -> list[list[T]]:
        return [self.row_list(i) for i in range(self.nrows)]

    @abstractmethod
    def transpose(self) -> BaseMatrix[T]:
        raise NotImplementedError

    @abstractmethod
    def _left_op(self, other, op):
        raise NotImplementedError

    @abstractmethod
    def _right_op(self, other, op):
        raise NotImplementedError

    def __lt__(self, other: T | BaseMatrix[T]) -> BaseMatrix[bool]:
        return self._left_op(other, lt)

    def __le__(self, other: T | BaseMatrix[T]) -> BaseMatrix[bool]:
        return self._left_op(other, le)

    def __eq__(self, other: T | BaseMatrix[T]) -> BaseMatrix[bool]:
        return self._left_op(other, eq)

    def __ne__(self, other: T | BaseMatrix[T]) -> BaseMatrix[bool]:
        return self._left_op(other, ne)

    def __ge__(self, other: T | BaseMatrix[T]) -> BaseMatrix[bool]:
        return self._left_op(other, ge)

    def __gt__(self, other: T | BaseMatrix[T]) -> BaseMatrix[bool]:
        return self._left_op(other, gt)

    def __add__(self, other: T | BaseMatrix[T]) -> BaseMatrix[T]:
        return self._left_op(other, add)

    def __sub__(self, other: T | BaseMatrix[T]) -> BaseMatrix[T]:
        return self._left_op(other, sub)

    def __mul__(self, other: T | BaseMatrix[T]) -> BaseMatrix[T]:
        return self._left_op(other, mul)

    def __truediv__(self, other: T | BaseMatrix[T]) -> BaseMatrix[T]:
        return self._left_op(other, truediv)

    def __floordiv__(self, other: T | BaseMatrix[T]) -> BaseMatrix[T]:
        return self._left_op(other, floordiv)

    def __mod__(self, other: T | BaseMatrix[T]) -> BaseMatrix[T]:
        return self._left_op(other, mod)

    def __pow__(self, power: T | BaseMatrix[T]) -> BaseMatrix[T]:
        return self._left_op(power, pow)

    def __lshift__(self, other: T | BaseMatrix[T]) -> BaseMatrix[T]:
        return self._left_op(other, lshift)

    def __rshift__(self, other: T | BaseMatrix[T]) -> BaseMatrix[T]:
        return self._left_op(other, rshift)

    def __and__(self, other: T | BaseMatrix[T]) -> BaseMatrix[T]:
        return self._left_op(other, and_)

    def __xor__(self, other: T | BaseMatrix[T]) -> BaseMatrix[T]:
        return self._left_op(other, xor)

    def __or__(self, other: T | BaseMatrix[T]) -> BaseMatrix[T]:
        return self._left_op(other, or_)

    def __radd__(self, other: T) -> BaseMatrix[T]:
        return self._right_op(other, add)

    def __rsub__(self, other: T) -> BaseMatrix[T]:
        return self._right_op(other, sub)

    def __rmul__(self, other: T) -> BaseMatrix[T]:
        return self._right_op(other, mul)

    def __rtruediv__(self, other: T) -> BaseMatrix[T]:
        return self._right_op(other, truediv)

    def __rfloordiv__(self, other: T) -> BaseMatrix[T]:
        return self._right_op(other, floordiv)

    def __rmod__(self, other: T) -> BaseMatrix[T]:
        return self._right_op(other, mod)

    def __rpow__(self, power: T) -> BaseMatrix[T]:
        return self._right_op(power, pow)

    def __rlshift__(self, other: T) -> BaseMatrix[T]:
        return self._right_op(other, lshift)

    def __rrshift__(self, other: T) -> BaseMatrix[T]:
        return self._right_op(other, rshift)

    def __rand__(self, other: T) -> BaseMatrix[T]:
        return self._right_op(other, and_)

    def __rxor__(self, other: T) -> BaseMatrix[T]:
        return self._right_op(other, xor)

    def __ror__(self, other: T) -> BaseMatrix[T]:
        return self._right_op(other, or_)

    @abstractmethod
    def map(self, f: Callable[[T], R]) -> BaseMatrix[R]:
        raise NotImplementedError

    def __neg__(self) -> BaseMatrix[T]:
        return self.map(neg)

    def __pos__(self) -> BaseMatrix[T]:
        return self.map(pos)

    def __abs__(self) -> BaseMatrix[T]:
        return self.map(abs)

    def __invert__(self) -> BaseMatrix[T]:
        return self.map(invert)

    @abstractmethod
    def __matmul__(self, other: BaseMatrix[T]) -> BaseMatrix[T]:
        raise NotImplementedError

    def matrix_power(self, power: Integral) -> BaseMatrix[T]:
        if isinstance(power, Integral):
            if power < 0:
                raise ValueError("power must be non-negative")

            if not self.is_square():
                raise NonSquareMatrixError

            if not self:
                return self.copy()

            one = _one(self[0, 0])
            zero = _zero(self[0, 0])
            ans = self.from_function(
                lambda i, j: one if i == j else zero, (self.nrows, self.nrows)
            )
            pow2 = self
            while power:
                power, r = divmod(power, 2)
                if r:
                    ans @= pow2
                pow2 @= pow2
            return ans

        else:
            return NotImplemented

    def solve(self, other):
        return LUDecomposition.decompose(self).solve(other)

    def inv(self):
        return LUDecomposition.decompose(self).inv()

    def det(self):
        try:
            return LUDecomposition.decompose(self).det()
        except SingularMatrixError:
            return _zero(self[0, 0])

    def __str__(self) -> str:
        row_strs = []
        for row in self.as_list():
            row_strs.append("[" + " ".join(map(repr, row)) + "]")
        return "[" + "\n ".join(row_strs) + "]"

    def __repr__(self) -> str:
        return type(self).__name__ + f"({self.as_list()!r})"


class Matrix(BaseMatrix[T]):
    """
    Matrix class backed by a single list of all entries arranged in row-major
    order.
    """

    _mat: list[list[T]]
    _shape: tuple[int, int]

    vector_class = Vector

    __slots__ = ("_mat", "_shape")

    @property
    def shape(self):
        return self._shape

    @classmethod
    def _create(cls, mat, shape):
        self = cls.__new__(cls)
        self._mat = mat
        self._shape = shape
        return self

    @classmethod
    def _create_vector(cls, vec):
        # noinspection PyProtectedMember
        return cls.vector_class._create(vec)

    def __init__(self, mat: BaseMatrix | Sequence[Sequence[T]]) -> None:
        if isinstance(mat, BaseMatrix):
            m, n = mat.shape
            self._mat = [mat[i, j] for i in range(m) for j in range(n)]
            self._shape = m, n

        else:
            try:
                self._mat = [x for row in mat for x in row]
                m = len(mat)
                n = len(mat[0]) if mat else 0
                self._shape = (m, n)
            except TypeError:
                raise TypeError("mat must be an matrix or a sequence of " "sequences")

            for i, row in enumerate(mat):
                if len(row) != n:
                    raise ValueError(f"row {i} of mat does not have length {n}")

    @classmethod
    def from_function(cls, f, shape):
        m, n = shape
        mat = [f(i, j) for i in range(m) for j in range(n)]
        return cls._create(mat, shape)

    @classmethod
    def hstack(cls, tup):
        if not tup:
            raise ValueError("at least one matrix is required")

        nrows = tup[0].nrows
        for mat in tup:
            if mat.ncols != nrows:
                raise ValueError("matrices must all have the same" "number of rows")

        ncols = sum(mat.ncols for mat in tup)
        hstack = [x for i in range(nrows) for mat in tup for x in mat.row_list(i)]
        return cls._create(hstack, (nrows, ncols))

    @classmethod
    def vstack(cls, tup):
        if not tup:
            raise ValueError("at least one matrix is required")

        ncols = tup[0].ncols
        for mat in tup:
            if mat.ncols != ncols:
                raise ValueError("matrices must all have the same" "number of columns")

        nrows = sum(mat.nrows for mat in tup)
        vstack = [x for mat in tup for x in mat]
        return cls._create(vstack, (nrows, ncols))

    @classmethod
    def kronecker(cls, a, b):
        arows = [a.row_list(i) for i in range(a.nrows)]
        brows = [b.row_list(i) for i in range(b.nrows)]
        kronecker = []
        for arow in arows:
            for brow in brows:
                kronecker.extend([x * y for x in arow for y in brow])
        return cls._create(kronecker, (a.nrows * b.nrows, a.ncols * b.ncols))

    def __getitem__(self, key):
        self._check_key(key)

        i, j = key
        if isinstance(i, int) and isinstance(j, int):
            i = self._resolve_i(i)
            j = self._resolve_j(j)
            return self._mat[i * self.ncols + j]

        else:
            i_list = self._resolve_i_list(i)
            j_list = self._resolve_j_list(j)

            mat = self._mat
            ncols = self.ncols
            selected = [mat[ii * ncols + jj] for ii in i_list for jj in j_list]

            if isinstance(i, int) or isinstance(j, int):
                return self._create_vector(selected)
            else:
                shape = len(i_list), len(j_list)
                return self._create(selected, shape)

    def __setitem__(self, key, value):
        self._check_key(key)

        i, j = key
        if isinstance(i, int) and isinstance(j, int):
            i = self._resolve_i(i)
            j = self._resolve_j(j)
            self._mat[i * self.ncols + j] = value

        else:
            i_list = self._resolve_i_list(i)
            j_list = self._resolve_j_list(j)

            if isinstance(i, int) or isinstance(j, int):
                value = self.vector_class(value)
            else:
                shape = len(i_list), len(j_list)
                value = self.__class__(value)
                if value.shape != shape:
                    raise ValueError

            mat = self._mat
            ncols = self.ncols
            for (ii, jj), x in zip(product(i_list, j_list), value):
                mat[ii * ncols + jj] = x

    @staticmethod
    def _check_key(key):
        if not isinstance(key, tuple):
            raise TypeError("indices must be tuple, not " + type(key).__name__)
        if not len(key) == 2:
            raise ValueError(f"indices have length 2")

    def _resolve_i(self, i):
        nrows = self.nrows
        if 0 <= i < nrows:
            return i
        elif -nrows <= i:
            return nrows + i
        else:
            raise IndexError("index i out of range")

    def _resolve_j(self, j):
        ncols = self.ncols
        if 0 <= j < ncols:
            return j
        elif -ncols <= j:
            return ncols + j
        else:
            raise IndexError("index j out of range")

    def _resolve_i_list(self, i):
        if isinstance(i, int):
            return [self._resolve_i(i)]
        elif isinstance(i, slice):
            return range(self.nrows)[i]
        elif isinstance(i, Sequence):
            return i
        else:
            raise TypeError

    def _resolve_j_list(self, j):
        if isinstance(j, int):
            return [self._resolve_j(j)]
        elif isinstance(j, slice):
            return range(self.ncols)[j]
        elif isinstance(j, Sequence):
            return j
        else:
            raise TypeError

    def __iter__(self):
        return iter(self._mat)

    def copy(self):
        return self._create(list(self._mat), self._shape)

    def row_list(self, i):
        ncols = self.ncols
        return self._mat[ncols * i : ncols * (i + 1)]

    def col_list(self, j):
        ncols = self.ncols
        return self._mat[j::ncols]

    def diag_list(self):
        ncols = self.ncols
        return self._mat[:: ncols + 1]

    def tril(self, k=0):
        if not self:
            return self.copy()

        mat = self._mat
        ncols = self.ncols

        zero = _zero(self[0, 0])
        tril = [zero] * len(self)
        for i in range(self.nrows):
            start = ncols * i
            end = max(start, start + i + 1 + k)
            tril[start:end] = mat[start:end]
        return self._create(tril, self.shape)

    def triu(self, k=0):
        if not self:
            return self.copy()

        mat = self._mat
        ncols = self.ncols

        zero = _zero(self[0, 0])
        triu = [zero] * len(self)
        for i in range(self.nrows):
            row_start = ncols * i
            start = max(row_start, row_start + i + k)
            end = row_start + ncols
            triu[start:end] = mat[start:end]
        return self._create(triu, self.shape)

    def transpose(self):
        mat = self._mat
        nrows = self.nrows
        ncols = self.ncols
        mat = [x for j in range(ncols) for x in mat[j::ncols]]
        return self._create(mat, (ncols, nrows))

    def _left_op(self, other, op):
        if isinstance(other, BaseMatrix):
            if self.shape != other.shape:
                raise ValueError(
                    "matrix shapes do not match: " f"{self.shape}, {other.shape}"
                )
            mat = list(map(op, self, other))
            return self._create(mat, self.shape)

        else:
            mat = [op(x, other) for x in self]
            return self._create(mat, self.shape)

    def _right_op(self, other, op):
        mat = [op(other, x) for x in self]
        return self._create(mat, self.shape)

    def map(self, f):
        mat = [f(x) for x in self]
        return self._create(mat, self.shape)

    def __matmul__(self, other):
        if isinstance(other, BaseMatrix):
            if self.ncols != other.nrows:
                raise ValueError(
                    "matrix shapes are not compatible: " f"{self.shape}, {other.shape}"
                )

            rows = [self.row_list(i) for i in range(self.nrows)]
            cols = [other.col_list(j) for j in range(other.ncols)]

            def entry(i, j):
                return sum(map(mul, rows[i], cols[j]))

            return self.from_function(entry, (self.nrows, other.ncols))

        else:
            return NotImplemented


class LUDecomposition(Generic[T]):

    _lu: BaseMatrix[T]
    _idx: list[int]

    @property
    def lu(self) -> BaseMatrix[T]:
        return self._lu

    @property
    def l(self) -> BaseMatrix[T]:
        l = self._lu.tril()
        one = _one(l[0, 0])
        for i in range(min(l.nrows, l.ncols)):
            l[i, i] = one
        return l

    @property
    def u(self) -> BaseMatrix[T]:
        return self._lu.triu()

    @property
    def idx(self) -> list[int]:
        return self._idx

    def __init__(self, lu: BaseMatrix[T], idx: list[int]) -> None:
        self._lu = lu
        self._idx = idx

    @classmethod
    def decompose(cls, a: BaseMatrix[T]) -> LUDecomposition[T]:
        if not a.is_square():
            raise NonSquareMatrixError

        lu = a.copy()
        n = lu.nrows
        idx = list(range(n))

        for k in range(n):
            if not lu[k, k]:
                for i in range(k + 1, n):
                    if lu[i, k]:
                        rowi = lu[i, :]
                        lu[i, :] = lu[k, :]
                        lu[k, :] = rowi
                        idx[k] = i
                        break
                else:
                    raise SingularMatrixError

            pivot = lu[k, k]
            for i in range(k + 1, n):
                alpha = lu[i, k] = lu[i, k] / pivot
                for j in range(k + 1, n):
                    lu[i, j] -= alpha * lu[k, j]

        return cls(lu, idx)

    def reconstruct(self) -> BaseMatrix[T]:
        return self.l @ self.u

    def solve(self, b: BaseVector[T] | BaseMatrix[T]) -> BaseVector[T] | BaseMatrix[T]:

        lu = self._lu

        x = b.copy()

        if isinstance(x, BaseVector):
            if not lu.ncols == len(x):
                raise ValueError(
                    "matrix and vector shapes are not compatible: "
                    f"{lu.shape}, {len(x.shape)}"
                )
            return self._solve(x)

        elif isinstance(x, BaseMatrix):
            if not lu.ncols == x.nrows:
                raise ValueError(
                    "matrix shapes are not compatible: " f"{lu.shape}, {x.shape}"
                )
            for j in range(x.ncols):
                x[:, j] = self._solve(x[:, j])
            return x

    def _solve(self, b):
        lu = self._lu
        n = lu.nrows
        idx = self._idx

        x = b.copy()

        for i, xip in enumerate(x):
            ip = idx[i]
            xi = x[ip]
            x[ip] = xip
            for j, xj in enumerate(x[:i]):
                xi -= lu[i, j] * xj
            x[i] = xi

        for i, xi in zip(range(n - 1, -1, -1), reversed(x)):
            for j, xj in enumerate(x[i + 1 : :], i + 1):
                xi -= lu[i, j] * xj
            x[i] = xi / lu[i, i]

        return x

    def inv(self) -> BaseMatrix[T]:
        lu = self._lu
        return self.solve(lu.identity(lu.nrows))

    def det(self) -> T:
        lu = self._lu
        idx = self._idx
        sign = prod(-1 for i, j in enumerate(idx) if i != j)
        return sign * prod(lu.diag_list())

    def __repr__(self) -> str:
        return type(self).__name__ + f"({self._lu!r}, {self._idx!r})"


class MatrixError(Exception):
    pass


class NonSquareMatrixError(MatrixError):
    def __init__(self) -> None:
        super().__init__("matrix must be square")


class SingularMatrixError(MatrixError):
    def __init__(self) -> None:
        super().__init__("matrix must be non-singular")
