from __future__ import annotations

from numbers import Rational, Integral
from typing import Generic, TypeVar, Tuple

from project_euler_math.eisenstein import Eisenstein
from project_euler_math.gaussian import Gaussian
from project_euler_math.ntheory import gcd
from project_euler_math.polynomial import Polynomial

E = TypeVar("E", Integral, Gaussian, Eisenstein, Polynomial)


class Fraction(Generic[E]):
    """
    A fraction composed of a pair of elements of a Bezout domain.
    """

    _p: E
    _q: E

    __slots__ = ("_p", "_q")

    @property
    def p(self) -> E:
        return self._p

    @property
    def q(self) -> E:
        return self._q

    @property
    def numerator(self) -> E:
        return self._p

    @property
    def denominator(self) -> E:
        return self._q

    def __init__(
        self, p: E | Fraction[E] | Rational = 0, q: E | Fraction[E] | Rational = 1
    ) -> None:

        if isinstance(p, (Fraction, Rational)):
            if isinstance(q, (Fraction, Rational)):
                self._p = p.numerator * q.denominator
                self._q = p.denominator * q.numerator
            else:
                self._p = p.numerator
                self._q = p.denominator * q

        else:
            if isinstance(q, (Fraction, Rational)):
                self._p = p * q.denominator
                self._q = q.numerator
            else:
                self._p = p
                self._q = q

        d = gcd(self._p, self._q)
        self._p //= d
        self._q //= d

    def __eq__(self, other) -> bool:
        if isinstance(other, (Fraction, Rational)):
            return self._p * other.denominator == other.numerator * self._q
        return self._p == other.numerator * self._q

    def __hash__(self) -> int:
        return hash((self._p, self._q))

    def __bool__(self) -> bool:
        return bool(self._p)

    def __add__(self, other: E | Fraction[E] | Rational) -> Fraction[E]:
        if isinstance(other, (Fraction, Rational)):
            return Fraction(
                self._p * other.denominator + other.numerator * self._q,
                self._q * other.denominator,
            )
        else:
            return Fraction(self._p + other.numerator * self._q, self._q)

    def __sub__(self, other: E | Fraction[E] | Rational) -> Fraction[E]:
        if isinstance(other, (Fraction, Rational)):
            return Fraction(
                self._p * other.denominator - other.numerator * self._q,
                self._q * other.denominator,
            )
        else:
            return Fraction(self._p - other.numerator * self._q, self._q)

    def __mul__(self, other: E | Fraction[E] | Rational) -> Fraction[E]:
        if isinstance(other, (Fraction, Rational)):
            return Fraction(self._p * other.numerator, self._q * other.denominator)
        else:
            return Fraction(self._p * other.numerator, self._q)

    def __truediv__(self, other: E | Fraction[E] | Rational) -> Fraction[E]:
        if isinstance(other, (Fraction, Rational)):
            return Fraction(self._p * other.denominator, self._q * other.numerator)
        else:
            return Fraction(self._p, self._q * other.numerator)

    def __floordiv__(self, other: E | Fraction[E] | Rational) -> Fraction[E]:
        if isinstance(other, (Fraction, Rational)):
            return (self._p * other.denominator) // (other.numerator * self._q)
        else:
            return self._p // (other.numerator * self._q)

    def __mod__(self, other: E | Fraction[E] | Rational) -> Fraction[E]:
        if isinstance(other, (Fraction, Rational)):
            return Fraction(
                (self._p * other.denominator) % (other.numerator * self._q),
                self._q * other.denominator,
            )
        else:
            return Fraction(self._p % (other.numerator * self._q), self._q)

    def __divmod__(
        self, other: E | Fraction[E] | Rational
    ) -> Tuple[Fraction[E], Fraction[E]]:
        if isinstance(other, (Fraction, Rational)):
            div, rem = divmod(self._p * other.denominator, other.numerator * self._q)
            return div, Fraction(rem, self._q * other.denominator)
        else:
            div, rem = divmod(self._p, other.numerator * self._q)
            return div, Fraction(rem, self._q)

    def __pow__(self, power: Integral) -> Fraction[E]:
        return Fraction(self._p**power, self._q**power)

    def __radd__(self, other: E | Rational) -> Fraction[E]:
        if isinstance(other, Rational):
            return Fraction(
                other.numerator * self._q + self._p * other.denominator,
                other.denominator * self._q,
            )
        else:
            return Fraction(other.numerator * self._q + self._p, self._q)

    def __rsub__(self, other: E | Rational) -> Fraction[E]:
        if isinstance(other, Rational):
            return Fraction(
                other.numerator * self._q - self._p * other.denominator,
                other.denominator * self._q,
            )
        else:
            return Fraction(other.numerator * self._q - self._p, self._q)

    def __rmul__(self, other: E | Rational) -> Fraction[E]:
        if isinstance(other, Rational):
            return Fraction(other.numerator * self._p, other.denominator * self._q)
        else:
            return Fraction(other.numerator * self._p, self._q)

    def __rtruediv__(self, other: E | Rational) -> Fraction[E]:
        if isinstance(other, Rational):
            return Fraction(other.numerator * self._q, other.denominator * self._p)
        else:
            return Fraction(other * self._q, self._p)

    def __rfloordiv__(self, other: E | Rational) -> Fraction[E]:
        if isinstance(other, Rational):
            return (other.numerator * self._q) // (self._p * other.denominator)
        else:
            return (other * self._q) // self._p

    def __rmod__(self, other: E | Rational) -> Fraction[E]:
        if isinstance(other, Rational):
            return Fraction(
                (other.numerator * self._q) % (self._p * other.denominator),
                other.denominator * self._q,
            )
        else:
            return Fraction((other * self._q) % self._p, self._q)

    def __rdivmod__(self, other: E | Rational) -> Tuple[Fraction[E], Fraction[E]]:
        if isinstance(other, Rational):
            div, rem = divmod(other.numerator * self._q, self._p * other.denominator)
            return div, Fraction(rem, other.denominator * self._q)
        else:
            div, rem = divmod(other * self._q, self._p)
            return div, Fraction(rem, self._q)

    def __neg__(self) -> Fraction[E]:
        return Fraction(-self._p, self._q)

    def __pos__(self) -> Fraction[E]:
        return Fraction(+self._p, self._q)

    def __abs__(self) -> Fraction[E]:
        return Fraction(abs(self._p), abs(self._q))

    def __int__(self) -> int:
        return self._p // self._q if self._p >= 0 else -(-self._p // self._q)

    def __float__(self) -> Fraction[E]:
        return self._p / self._q

    def __str__(self) -> str:
        return f"{self.p!r}/{self.q!r}"

    def __repr__(self) -> str:
        return type(self).__name__ + f"({self.p!r}, {self.q!r})"
