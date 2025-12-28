from __future__ import annotations

from math import sqrt
from numbers import Complex

_SQRT3 = sqrt(3)


class Eisenstein(Complex):
    """
    An Eisenstein integer. Eisenstein integers are complex numbers of the form
    ``a + bw`` where ``a`` and ``b`` are integers and ``w`` is a cube root of
    unity.
    """

    _x: int
    _y: int

    __slots__ = ("_x", "_y")

    @property
    def x(self) -> int:
        return self._x

    @property
    def y(self) -> int:
        return self._y

    @property
    def real(self) -> float:
        return self.x - self.y / 2

    @property
    def imag(self) -> float:
        return self.y * _SQRT3 / 2

    def __init__(self, x: int | Eisenstein = 0, y: int | Eisenstein = 0) -> None:
        self._x = 0
        self._y = 0

        if isinstance(x, int):
            self._x += x
        elif isinstance(x, Eisenstein):
            self._x += x.x
            self._y += x.y
        else:
            raise ValueError(f"x is not int or Eisenstein: {x}")

        if isinstance(y, int):
            self._y += y
        elif isinstance(y, Eisenstein):
            self._x -= y.y
            self._y += y.x - y.y
        else:
            raise ValueError(f"y is not int or Eisenstein: {y}")

    def __eq__(self, other) -> bool:
        if isinstance(other, int):
            return (self.x, self.y) == (other, 0)
        elif isinstance(other, Eisenstein):
            return (self.x, self.y) == (other.x, other.y)
        else:
            return NotImplemented

    def __hash__(self) -> int:
        return hash((self.x, self.y))

    def __bool__(self) -> bool:
        return bool(self._x) or bool(self._y)

    def conjugate(self) -> Eisenstein:
        return Eisenstein(self.x - self.y, -self.y)

    def norm(self) -> int:
        return self.x * self.x - self.x * self.y + self.y * self.y

    def __add__(self, other: int | Eisenstein) -> Eisenstein:
        if isinstance(other, Eisenstein):
            return Eisenstein(self.x + other.x, self.y + other.y)
        elif isinstance(other, int):
            # noinspection PyTypeChecker
            return Eisenstein(self.x + other, self.y)
        else:
            return NotImplemented

    def __sub__(self, other: int | Eisenstein) -> Eisenstein:
        if isinstance(other, Eisenstein):
            return Eisenstein(self.x - other.x, self.y - other.y)
        elif isinstance(other, int):
            # noinspection PyTypeChecker
            return Eisenstein(self.x - other, self.y)
        else:
            return NotImplemented

    def __mul__(self, other: int | Eisenstein) -> Eisenstein:
        if isinstance(other, Eisenstein):
            x1 = self.x
            y1 = self.y
            x2 = other.x
            y2 = other.y
            return Eisenstein(x1 * x2 - y1 * y2, x1 * y2 + y1 * x2 - y1 * y2)
        elif isinstance(other, int):
            return Eisenstein(self.x * other, self.y * other)
        else:
            return NotImplemented

    def __truediv__(self, other: complex) -> complex:
        if isinstance(other, complex):
            return complex(self) / other
        else:
            return NotImplemented

    def __div__(self, other: complex) -> complex:
        return self.__truediv__(other)

    def __floordiv__(self, other: int | Eisenstein) -> Eisenstein:
        if isinstance(other, Eisenstein):
            x1 = self.x
            y1 = self.y
            x2 = other.x
            y2 = other.y
            x = x1 * x2 + y1 * y2 - x1 * y2
            y = -x1 * y2 + y1 * x2
            d = other.norm()
            roundup = (d - 1) // 2
            return Eisenstein((x + roundup) // d, (y + roundup) // d)

        elif isinstance(other, int):
            roundup = (other - 1) // 2
            return Eisenstein((self.x + roundup) // other, (self.y + roundup) // other)

        else:
            return NotImplemented

    def __mod__(self, other: int | Eisenstein) -> Eisenstein:
        div = self.__floordiv__(other)
        if div is NotImplemented:
            return NotImplemented
        else:
            return self - div * other

    def __divmod__(self, other: int | Eisenstein) -> tuple[Eisenstein, Eisenstein]:
        div = self.__floordiv__(other)
        if div is NotImplemented:
            return NotImplemented
        else:
            return div, self - div * other

    def __pow__(self, power: int) -> Eisenstein:
        if isinstance(power, int):
            if power < 0:
                raise ValueError("power must be non-negative")

            ans = Eisenstein(1)
            pow2 = self
            while power:
                power, r = divmod(power, 2)
                if r:
                    ans *= pow2
                pow2 *= pow2
            return ans

        else:
            return NotImplemented

    def __radd__(self, other: int | complex) -> Eisenstein | complex:
        if isinstance(other, int):
            # noinspection PyTypeChecker
            return Eisenstein(other + self.x, self.y)
        elif isinstance(other, complex):
            return other + complex(self)
        else:
            return NotImplemented

    def __rsub__(self, other: int | complex) -> Eisenstein | complex:
        if isinstance(other, int):
            return Eisenstein(other - self.x, -self.y)
        elif isinstance(other, complex):
            return complex(other) - complex(self)
        else:
            return NotImplemented

    def __rmul__(self, other: int | complex) -> Eisenstein | complex:
        if isinstance(other, int):
            return Eisenstein(other * self.x, other * self.y)
        elif isinstance(other, complex):
            return complex(other) * complex(self)
        else:
            return NotImplemented

    def __rtruediv__(self, other: complex) -> complex:
        if isinstance(other, complex):
            return complex(other) / complex(self)
        else:
            return NotImplemented

    def __rdiv__(self, other: complex) -> complex:
        return self.__rtruediv__(other)

    def __rdivmod__(self, other: int) -> tuple[Eisenstein, Eisenstein]:
        if isinstance(other, int):
            # noinspection PyTypeChecker
            return divmod(Eisenstein(other), self)
        else:
            return NotImplemented

    def __rfloordiv__(self, other: int) -> Eisenstein:
        if isinstance(other, int):
            # noinspection PyTypeChecker
            return Eisenstein(other) // self
        else:
            return NotImplemented

    def __rmod__(self, other: int) -> Eisenstein:
        if isinstance(other, int):
            # noinspection PyTypeChecker
            return Eisenstein(other) % self
        else:
            return NotImplemented

    def __rpow__(self, base: complex) -> complex:
        if isinstance(base, complex):
            return complex(base) ** complex(self)
        else:
            return NotImplemented

    def __neg__(self) -> Eisenstein:
        return Eisenstein(-self.x, -self.y)

    def __pos__(self) -> Eisenstein:
        return Eisenstein(+self.x, +self.y)

    def __abs__(self) -> float:
        return abs(complex(self))

    def __complex__(self) -> complex:
        return complex(self.x - self.y / 2, self.y * _SQRT3 / 2)

    def __repr__(self) -> str:
        if self.x == 0:
            return f"{self.y}w"
        # noinspection PyTypeChecker
        sep = "+" if self.y >= 0 else ""
        return f"({self.x}{sep}{self.y}w)"


W = Eisenstein(0, 1)
WBAR = Eisenstein(-1, -1)
