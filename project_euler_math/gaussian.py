from __future__ import annotations

from numbers import Integral, Complex
from typing import Tuple


class Gaussian(Complex):
    """
    A Gaussian integer. Gaussian integers are complex numbers of the form
    ``a + bi`` where ``a`` and ``b`` are integers.
    """

    _x: Integral
    _y: Integral

    __slots__ = ('_x', '_y')

    @property
    def x(self) -> Integral:
        return self._x

    @property
    def y(self) -> Integral:
        return self._y

    @property
    def real(self) -> Integral:
        return self._x

    @property
    def imag(self) -> Integral:
        return self._y

    def __init__(
            self,
            x: Integral | Gaussian = 0,
            y: Integral | Gaussian = 0) -> None:

        self._x = 0
        self._y = 0

        if isinstance(x, Integral):
            self._x += x
        elif isinstance(x, Gaussian):
            self._x += x.x
            self._y += x.y
        else:
            raise ValueError(f'x is not Integral or Gaussian: {x}')

        if isinstance(y, Integral):
            self._y += y
        elif isinstance(y, Gaussian):
            self._x -= y.y
            self._y += y.x
        else:
            raise ValueError(f'y is not Integral or Gaussian: {y}')

    def __eq__(self, other) -> bool:
        if isinstance(other, Integral):
            return (self.x, self.y) == (other, 0)
        elif isinstance(other, Gaussian):
            return (self.x, self.y) == (other.x, other.y)
        return NotImplemented

    def __hash__(self) -> int:
        return hash((self.x, self.y))

    def __bool__(self) -> bool:
        return bool(self._x) or bool(self._y)

    def conjugate(self) -> Gaussian:
        return Gaussian(self.x, -self.y)

    def norm(self) -> Integral:
        return self.x * self.x + self.y * self.y

    def __add__(self, other: Integral | Gaussian) -> Gaussian:
        if isinstance(other, Gaussian):
            return Gaussian(self.x + other.x, self.y + other.y)
        elif isinstance(other, Integral):
            return Gaussian(self.x + other, self.y)
        else:
            return NotImplemented

    def __sub__(self, other: Integral | Gaussian) -> Gaussian:
        if isinstance(other, Gaussian):
            return Gaussian(self.x - other.x, self.y - other.y)
        elif isinstance(other, Integral):
            return Gaussian(self.x - other, self.y)
        else:
            return NotImplemented

    def __mul__(self, other: Integral | Gaussian) -> Gaussian:
        if isinstance(other, Gaussian):
            x1 = self.x
            y1 = self.y
            x2 = other.x
            y2 = other.y
            return Gaussian(x1 * x2 - y1 * y2, x1 * y2 + y1 * x2)
        elif isinstance(other, Integral):
            return Gaussian(self.x * other, self.y * other)
        else:
            return NotImplemented

    def __truediv__(self, other: Complex) -> Complex:
        if isinstance(other, Complex):
            return complex(self) / other
        else:
            return NotImplemented

    def __div__(self, other: Complex) -> Complex:
        return self.__truediv__(other)

    def __floordiv__(self, other: Integral | Gaussian) -> Gaussian:
        if isinstance(other, Gaussian):
            x1 = self.x
            y1 = self.y
            x2 = other.x
            y2 = other.y
            x = x1 * x2 + y1 * y2
            y = -x1 * y2 + y1 * x2
            d = other.norm()
            roundup = (d-1) // 2
            return Gaussian((x + roundup) // d, (y + roundup) // d)

        if isinstance(other, Integral):
            roundup = (other - 1) // 2
            return Gaussian(
                (self.x + roundup) // other, (self.y + roundup) // other)

        else:
            return NotImplemented

    def __mod__(self, other: Integral | Gaussian) -> Gaussian:
        div = self.__floordiv__(other)
        if div is NotImplemented:
            return NotImplemented
        else:
            return self - div * other

    def __divmod__(self, other: Integral | Gaussian)\
            -> Tuple[Gaussian, Gaussian]:
        div = self.__floordiv__(other)
        if div is NotImplemented:
            return NotImplemented
        else:
            return div, self - div * other

    def __pow__(self, power: Integral) -> Gaussian:
        if isinstance(power, Integral):
            if power < 0:
                raise ValueError("power must be non-negative")

            ans = Gaussian(1)
            pow2 = self
            while power:
                power, r = divmod(power, 2)
                if r:
                    ans *= pow2
                pow2 *= pow2
            return ans

        else:
            return NotImplemented

    def __radd__(self, other: Integral | Complex) -> Gaussian | complex:
        if isinstance(other, Integral):
            return Gaussian(other + self.x, self.y)
        elif isinstance(other, Complex):
            return complex(other) + complex(self)
        else:
            return NotImplemented

    def __rsub__(self, other: Integral | Complex) -> Gaussian | complex:
        if isinstance(other, Integral):
            return Gaussian(other - self.x, -self.y)
        elif isinstance(other, Complex):
            return complex(other) - complex(self)
        else:
            return NotImplemented

    def __rmul__(self, other: Integral | Complex) -> Gaussian | complex:
        if isinstance(other, Integral):
            return Gaussian(other * self.x, other * self.y)
        elif isinstance(other, Complex):
            return complex(other) * complex(self)
        else:
            return NotImplemented

    def __rtruediv__(self, other: Complex) -> complex:
        if isinstance(other, Complex):
            return complex(other) / complex(self)
        else:
            return NotImplemented

    def __rdiv__(self, other: Complex) -> complex:
        return self.__rtruediv__(other)

    def __rdivmod__(self, other: Integral) -> Tuple[Gaussian, Gaussian]:
        if isinstance(other, Integral):
            # noinspection PyTypeChecker
            return divmod(Gaussian(other), self)
        else:
            return NotImplemented

    def __rfloordiv__(self, other: Integral) -> Gaussian:
        if isinstance(other, Integral):
            # noinspection PyTypeChecker
            return Gaussian(other) // self
        else:
            return NotImplemented

    def __rmod__(self, other: Integral) -> Gaussian:
        if isinstance(other, Integral):
            # noinspection PyTypeChecker
            return Gaussian(other) % self
        else:
            return NotImplemented

    def __rpow__(self, base: Complex) -> complex:
        if isinstance(base, Complex):
            return complex(base) ** complex(self)
        else:
            return NotImplemented

    def __neg__(self) -> Gaussian:
        return Gaussian(-self.x, -self.y)

    def __pos__(self) -> Gaussian:
        return Gaussian(+self.x, +self.y)

    def __abs__(self) -> float:
        return abs(complex(self))

    def __complex__(self) -> complex:
        # noinspection PyTypeChecker
        return complex(self.x, self.y)

    def __repr__(self) -> str:
        if self.x == 0:
            return f'{self.y}j'
        # noinspection PyTypeChecker
        sep = '+' if self.y >= 0 else ''
        return f'({self.x}{sep}{self.y}j)'


J = Gaussian(0, 1)
JBAR = Gaussian(0, -1)
