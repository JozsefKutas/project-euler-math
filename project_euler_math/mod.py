from __future__ import annotations

from numbers import Integral
from operator import add, sub, mul, neg, pos
from typing import TypeVar, Generic, Union, Callable, Optional

from project_euler_math.ntheory import mod_inverse, gcd

T = TypeVar('T')


class Mod(Generic[T]):
    """
    A residue class modulo a specified modulus.
    """

    _n: T
    _mod: T

    __slots__ = ('_n', '_mod')

    @property
    def mod(self) -> T:
        return self._mod

    def lift(self) -> T:
        return self._n

    def __init__(self, n: T, mod: T) -> None:
        self._n = n % mod
        self._mod = mod

    def __eq__(self, other) -> bool:
        if isinstance(other, Mod):
            return (self._n, self.mod) == (other._n, other.mod)
        return NotImplemented

    def __hash__(self) -> int:
        return hash((self._n, self.mod))

    def __bool__(self) -> bool:
        return bool(self._n)

    def _left_op(self, other, op):
        if isinstance(other, Mod):
            if self.mod != other.mod:
                raise ValueError(
                    f'modulos do not match: {self.mod}, {other.mod}')
            return Mod(op(self._n, other._n), self.mod)
        else:
            return Mod(op(self._n, other), self.mod)

    def __add__(self, other: Union[T, Mod[T]]) -> Mod[T]:
        return self._left_op(other, add)

    def __sub__(self, other: Union[T, Mod[T]]) -> Mod[T]:
        return self._left_op(other, sub)

    def __mul__(self, other: Union[T, Mod[T]]) -> Mod[T]:
        return self._left_op(other, mul)

    def __truediv__(self, other: Union[T, Mod[T]]) -> Mod[T]:
        if isinstance(other, Mod):
            if self.mod != other.mod:
                raise ValueError(
                    f'modulos do not match: {self.mod}, {other.mod}')
            return self * other.invert()
        else:
            return self * Mod(other, self.mod).invert()

    def __mod__(self, other: Union[T, Mod[T]]) -> Mod[T]:
        if isinstance(other, Mod):
            if self.mod != other.mod:
                raise ValueError(
                    f'modulos do not match: {self.mod}, {other.mod}')
            return Mod(self._n % gcd(other._n, self.mod), self.mod)
        else:
            return Mod(self._n % gcd(other, self.mod), self.mod)

    def __pow__(self, power: Integral, modulo: Optional[T] = None) -> Mod[T]:
        if isinstance(power, Integral):
            if power < 0:
                return self.invert() ** (-power)
            if modulo:
                m = gcd(modulo, self.mod)
                return Mod(pow(self._n, int(power), m), self.mod)
            else:
                return Mod(self._n ** int(power), self.mod)
        else:
            return NotImplemented

    def _right_op(self, other, op):
        return Mod(op(other, self._n), self.mod)

    def __radd__(self, other: T) -> Mod[T]:
        return self._right_op(other, add)

    def __rsub__(self, other: T) -> Mod[T]:
        return self._right_op(other, sub)

    def __rmul__(self, other: T) -> Mod[T]:
        return self._right_op(other, mul)

    def __rtruediv__(self, other: T) -> Mod[T]:
        return other * self.invert()

    def __rmod__(self, other: T) -> Mod[T]:
        return Mod(other % gcd(self._n, self.mod), self.mod)

    def _map(self, op: Callable[[T], T]) -> Mod[T]:
        return Mod(op(self._n), self.mod)

    def __neg__(self) -> Mod[T]:
        return self._map(neg)

    def __pos__(self) -> Mod[T]:
        return self._map(pos)

    def invert(self) -> Mod[T]:
        inv = mod_inverse(self._n, self.mod)
        if inv is None:
            raise ValueError
        return Mod(inv, self.mod)

    def __str__(self) -> str:
        return f'{self._n!r} (mod {self.mod!r})'

    def __repr__(self) -> str:
        return type(self).__name__ + f'({self._n!r},{self.mod!r})'
