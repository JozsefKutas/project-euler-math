from __future__ import annotations

from math import inf
from numbers import Integral
from functools import reduce
from typing import Union, List, TypeVar, Generic, Iterable, Tuple, Callable

T = TypeVar('T')
R = TypeVar('R')

_PolynomialKey = Union[int, slice]


class Polynomial(Generic[T]):
    """A polynomial in a single variable."""

    _coeffs: List[T]

    __slots__ = '_coeffs'

    @property
    def coeffs(self) -> List[T]:
        return self._coeffs

    @property
    def leading_coeff(self) -> Union[T, int]:
        return self._coeffs[-1] if self._coeffs else 0

    @property
    def degree(self) -> Union[int, float]:
        length = len(self._coeffs)
        return length - 1 if length > 0 else -inf

    @classmethod
    def _create(cls, coeffs):
        self = cls.__new__(cls)
        self._coeffs = coeffs
        while self._coeffs and not self._coeffs[-1]:
            self._coeffs.pop()
        return self

    def __init__(self, coeffs: Union[None, Polynomial[T], Iterable[T]] = None) -> None:
        if coeffs is None:
            self._coeffs = []

        elif isinstance(coeffs, Polynomial):
            self._coeffs = list(coeffs.coeffs)

        else:
            self._coeffs = list(coeffs)
            while self._coeffs and not self._coeffs[-1]:
                self._coeffs.pop()

    def __eq__(self, other) -> bool:
        if isinstance(other, Polynomial):
            return self._coeffs == other._coeffs
        elif self.degree == 0:
            return self._coeffs[0] == other
        elif self.degree < 0:
            return 0 == other

    def __bool__(self) -> bool:
        return bool(self._coeffs)

    def __getitem__(self, key: _PolynomialKey) -> Union[T, Polynomial[T]]:
        return self._coeffs[key]

    def __setitem__(self, key: _PolynomialKey, value: Union[T, Polynomial[T]]) -> None:
        self._coeffs[key] = value

    def __add__(self, other: Union[T, Polynomial[T]]) -> Polynomial[T]:
        if isinstance(other, Polynomial):
            if self.degree < other.degree:
                p, q = self, other
            else:
                p, q = other, self
            coeffs = list(q._coeffs)
            for i, a in enumerate(p._coeffs):
                coeffs[i] += a
            return self._create(coeffs)

        else:
            return self + self._create([other])

    def __sub__(self, other: Union[T, Polynomial[T]]) -> Polynomial[T]:
        return self + (-other)

    def __mul__(self, other: Union[T, Polynomial[T]]) -> Polynomial[T]:
        if isinstance(other, Polynomial):
            degree = self.degree + other.degree
            if degree < 0:
                return self._create([])

            zero = _zero(self.leading_coeff or other.leading_coeff)
            coeffs = [zero] * (degree + 1)
            for i, a in enumerate(self._coeffs):
                for j, b in enumerate(other._coeffs, i):
                    coeffs[j] += a*b
            return self._create(coeffs)

        else:
            return self._create([a * other for a in self._coeffs])

    def __truediv__(self, other: Union[T, Polynomial[T]]) -> Polynomial[T]:
        if isinstance(other, Polynomial):
            raise TypeError
        else:
            return self._create([a / other for a in self._coeffs])

    def __floordiv__(self, other: Polynomial[T]) -> Polynomial[T]:
        dm = self.__divmod__(other)
        if dm is NotImplemented:
            return NotImplemented
        else:
            return dm[0]

    def __mod__(self, other: Polynomial[T]) -> Polynomial[T]:
        dm = self.__divmod__(other)
        if dm is NotImplemented:
            return NotImplemented
        else:
            return dm[1]

    def __divmod__(self, other: Polynomial[T]) -> Tuple[Polynomial[T], Polynomial[T]]:
        if isinstance(other, Polynomial):
            if self.degree < other.degree:
                return self._create([]), self

            lead = other.leading_coeff
            monic = [a / lead for a in other._coeffs]
            rem = [a / lead for a in self._coeffs]
            div = []
            for _ in range(len(rem) - len(monic) + 1):
                a = rem[-1]
                for i in range(-1, -len(monic)-1, -1):
                    rem[i] -= a * monic[i]
                div.append(a)
                rem.pop()
            return self._create(div[::-1]), self._create(rem)

        else:
            return NotImplemented

    def __pow__(self, power: Integral) -> Polynomial[T]:
        if isinstance(power, Integral):
            if power < 0:
                raise ValueError("power must be non-negative")

            one = _one(self.leading_coeff)
            ans = Polynomial([one])
            pow2 = self
            while power:
                power, r = divmod(power, 2)
                if r:
                    ans *= pow2
                pow2 *= pow2
            return ans

        else:
            return NotImplemented

    def __radd__(self, other: T) -> Polynomial[T]:
        return self._create([other]) + self

    def __rsub__(self, other: T) -> Polynomial[T]:
        return other + (-self)

    def __rmul__(self, other: T) -> Polynomial[T]:
        return self._create([other * a for a in self._coeffs])

    def __rfloordiv__(self, other: T) -> Polynomial[T]:
        return self._create([other]) // self

    def __rmod__(self, other: T) -> Polynomial[T]:
        return self._create([other]) % self

    def __rdivmod__(self, other: T) -> Tuple[Polynomial[T], Polynomial[T]]:
        # noinspection PyTypeChecker
        return divmod(self._create([other]), self)

    def __neg__(self) -> Polynomial[T]:
        return self._create([-a for a in self._coeffs])

    def __pos__(self) -> Polynomial[T]:
        return self._create([+a for a in self._coeffs])

    def map(self, f: Callable[[T], R]) -> Polynomial[R]:
        return self._create([f(a) for a in self])

    def __call__(self, x: T) -> T:
        return reduce(lambda a, b: a*x + b, reversed(self._coeffs), 0)

    def __str__(self) -> str:
        monomial_strs = []
        for i, a in enumerate(self._coeffs):
            if a:
                if i == 0:
                    monomial_str = f'{a!r}'
                elif i == 1:
                    monomial_str = f'{a!r}*x'
                else:
                    monomial_str = f'{a!r}*x**{i}'
                monomial_strs.append(monomial_str)
        return ' + '.join(monomial_strs)

    def __repr__(self) -> str:
        return type(self).__name__ + f'({self._coeffs!r})'


def _zero(n):
    return 0 * n


def _one(n):
    return n ** 0
