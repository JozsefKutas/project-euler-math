from __future__ import annotations

from itertools import chain, cycle
from math import prod, gcd, isqrt
from typing import Optional, Sequence, Tuple, Iterator, List

from project_euler_math.ntheory import factorisation


class ContinuedFraction:
    """
    A periodic continued fraction. A number has a periodic continued fraction
    expansion if and only if it is a quadratic irrational.
    """

    _initial: List[int]
    _repeating: List[int]

    __slots__ = ("_initial", "_repeating")

    @property
    def initial(self) -> List[int]:
        """The coefficients for the initial non-repeating part of the continued
        fraction."""
        return self._initial

    @property
    def repeating(self) -> List[int]:
        """The coefficients for the repeating part of the continued fraction."""
        return self._repeating

    def __init__(
        self,
        initial: ContinuedFraction | Sequence[int],
        repeating: Optional[Sequence[int]] = None,
    ) -> None:

        if isinstance(initial, ContinuedFraction):
            if repeating is None:
                self._initial = list(initial.initial)
                self._repeating = list(initial.repeating)
            else:
                raise ValueError

        elif isinstance(initial, Sequence):
            if repeating is None:
                repeating = []
            self._initial = list(initial)
            self._repeating = list(repeating)

        else:
            raise ValueError

    @classmethod
    def from_quadratic(cls, p: int = 0, q: int = 1, d: int = 0) -> ContinuedFraction:
        """Return the periodic continued fraction representation of the
        quadratic irrational (`p` + sqrt(`d`)) / `q`."""

        # write d_old = d_new * k ** 2, where d_new is square-free
        if d != 0:
            fact = factorisation(d)
            d = prod(pr for pr, e in fact.items() if e % 2 == 1)
            k = prod(pr ** (e // 2) for pr, e in fact.items())
        else:
            d = 1
            k = 0

        partial_quotients = []

        # d == 1 iff the input is a rational number
        # in this case the continued fraction terminates
        if d == 1:
            p += k

            while q != 0:
                a, rem = divmod(p, q)
                partial_quotients.append(a)
                p, q = q, rem

            self = ContinuedFraction(partial_quotients)

        else:
            complete_quotients = {}
            complete_quotient = (p, k, q)
            i = 0

            # compute complete quotients until the sequence repeats
            while complete_quotient not in complete_quotients:
                complete_quotients[complete_quotient] = i
                partial_quotient = (p + isqrt(k * k * d)) // q
                partial_quotients.append(partial_quotient)

                # compute next triple
                rem = p - partial_quotient * q
                new_p = q * rem
                new_k = -q * k
                new_q = rem * rem - k * k * d

                # remove common factors and make sure p is positive
                common_div = gcd(new_p, gcd(new_k, new_q))
                sgn = 1 if new_p >= 0 else -1
                p = new_p // common_div * sgn
                k = new_k // common_div * sgn
                q = new_q // common_div * sgn

                complete_quotient = (p, k, q)
                i += 1

            index = complete_quotients[complete_quotient]
            self = ContinuedFraction(
                partial_quotients[:index], partial_quotients[index:]
            )

        return cls(self)

    def to_quadratic(self):
        raise NotImplementedError

    def convergents(self) -> Iterator[Tuple[int, int]]:
        """Generate rational approximations to the continued fraction."""

        partial_quotients = chain(self.initial, cycle(self.repeating))

        p, p_old = next(partial_quotients), 1
        q, q_old = 1, 0
        yield p, q

        for a in partial_quotients:
            p, p_old = a * p + p_old, p
            q, q_old = a * q + q_old, q
            yield p, q

    def __eq__(self, other):
        if isinstance(other, ContinuedFraction):
            return (self._initial, self._repeating) == (
                other._initial,
                other._repeating,
            )
        return NotImplemented

    def __hash__(self):
        return hash((self._initial, self._repeating))

    def __repr__(self) -> str:
        return type(self).__name__ + f"({self._initial!r},{self.repeating!r})"

    def __str__(self) -> str:
        lis = [repr(x) for x in self.initial]
        if self.repeating:
            lis.append("(" + ", ".join(map(repr, self.repeating)) + ")")
        return "[" + ", ".join(lis) + "]"
