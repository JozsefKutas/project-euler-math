from __future__ import annotations

import re
from collections import namedtuple
from decimal import Decimal
from fractions import Fraction

RepeatingDecimalTuple = namedtuple(
    "RepeatingDecimalTuple", "sign initial exponent repetend"
)


class RepeatingDecimal:
    """A repeating decimal."""

    _exponent: int
    _initial: str
    _repetend: str | None
    _sign: int

    __slots__ = ("_exponent", "_initial", "_repetend", "_sign")

    @property
    def exponent(self) -> int:
        return self._exponent

    @property
    def initial(self) -> str:
        return self._initial

    @property
    def repetend(self) -> str | None:
        return self._repetend

    @property
    def sign(self):
        return self._sign

    def __init__(self, value: str | RepeatingDecimal | int | Decimal = "0") -> None:
        if isinstance(value, str):
            m = _parser(value.strip().replace("_", ""))
            if m is None:
                raise ValueError(f"Invalid literal for RepeatingDecimal: {value!r}")

            intpart = m.group("int")
            fracpart = m.group("frac") or ""
            repetend = m.group("repetend")
            self._exponent = -len(fracpart)
            self._initial = (intpart + fracpart).lstrip("0")
            self._repetend = repetend
            self._sign = 1 if m.group("sign") == "-" else 0

        elif isinstance(value, RepeatingDecimal):
            self._exponent = value._exponent
            self._initial = value._initial
            self._repetend = value._repetend
            self._sign = value._sign

        elif isinstance(value, int):
            self._exponent = 0
            self._initial = str(value)
            self._repetend = None
            self._sign = 0

        elif isinstance(value, Decimal):
            tup = value.as_tuple()
            self._exponent = tup.exponent
            self._initial = "".join(map(str, tup.digits))
            self._repetend = None
            self._sign = tup.sign

        elif isinstance(value, (list, tuple)):
            if len(value) != 4:
                raise ValueError(
                    "Invalid tuple size in creation of "
                    "RepeatingDecimal from list or tuple. The "
                    "list or tuple should have exactly four "
                    "elements."
                )

            if not (isinstance(value[0], int) and value[0] in (0, 1)):
                raise ValueError(
                    "Invalid sign. The first value in the tuple "
                    "should be an integer; either 0 for a "
                    "positive number or 1 for a negative number."
                )
            self._sign = value[0]

            initial = []
            for d in value[1]:
                if isinstance(d, int) and 0 <= d <= 9:
                    # skip leading zeros
                    if initial or d != 0:
                        initial.append(d)
                else:
                    raise ValueError(
                        "The second value in the tuple must be "
                        "composed of integers in the range 0 "
                        "through 9."
                    )

            self._initial = "".join(map(str, initial))

            if not isinstance(value[2], int):
                raise ValueError("The third value in the tuple must be an " "integer.")
            self._exponent = value[2]

            if not all(isinstance(d, int) and 0 <= d <= 9 for d in value[3]):
                raise ValueError(
                    "The fourth value in the tuple must be "
                    "composed of integers in the range 0 through "
                    "9."
                )

            self._repetend = None

        else:
            raise ValueError

    def as_tuple(self) -> RepeatingDecimalTuple:
        return RepeatingDecimalTuple(
            self._sign,
            tuple(map(int, self._initial)),
            self._exponent,
            tuple(map(int, self._repetend)) if self._repetend is not None else None,
        )

    @classmethod
    def _from_state(cls, exp, initial, repetend, sign) -> RepeatingDecimal:
        self = object.__new__(cls)
        self._exponent = exp
        self._initial = initial
        self._repetend = repetend
        self._sign = sign
        return self

    @classmethod
    def from_fraction(cls, fraction: Fraction = Fraction(0, 1)) -> RepeatingDecimal:
        sign = 0 if fraction >= 0 else 1
        p = abs(fraction.numerator)
        q = abs(fraction.denominator)

        # remove factors of 2 and 5 from denominator
        exp = 0
        while q % 10 == 0:
            q //= 10
            exp -= 1
        while q % 2 == 0:
            q //= 2
            p *= 5
            exp -= 1
        while q % 5 == 0:
            q //= 5
            p *= 2
            exp -= 1

        # noinspection PyTypeChecker
        n, p = divmod(p, q)
        initial = str(n) if n else ""

        if p == 0:
            repetend = None

        else:
            # find the fraction a/b equivalent to p/q such that b = 10^i-1 and
            # such that a and b are minimal
            pow10 = 10
            period = 1
            while pow10 % q != 1:
                pow10 *= 10
                period += 1
            a = p * ((pow10 - 1) // q)
            repetend = str(a).zfill(period)

        return cls._from_state(exp, initial, repetend, sign)

    def to_fraction(self) -> Fraction:
        q1 = 10**-self._exponent
        initial = Fraction(int(self._initial) if self._initial else 0, q1)
        if self._repetend:
            q2 = (10 ** len(self._repetend) - 1) * q1
            repetend = Fraction(int(self._repetend), q2)
            fraction = initial + repetend
        else:
            fraction = initial
        return -fraction if self._sign else fraction

    def digit(self, i):
        i -= self._exponent
        if i < 0:
            i = -i - 1
            repetend = self._repetend
            if repetend is not None:
                i %= len(repetend)
                return int(repetend[i])
            else:
                return 0
        else:
            initial = self._initial
            if i < len(initial):
                return self._initial[len(initial) - i - 1]
            else:
                return 0

    def __eq__(self, other):
        if isinstance(other, RepeatingDecimal):
            return (self._exponent, self._initial, self._repetend, self._sign) == (
                other._exponent,
                other._initial,
                other._repetend,
                other._sign,
            )
        return NotImplemented

    def __hash__(self):
        return hash((self._exponent, self._initial, self._repetend, self._sign))

    def __repr__(self) -> str:
        return type(self).__name__ + f"('{self!s}')"

    def __str__(self) -> str:
        sign = "-" if self._sign else ""
        initial = self._initial.zfill(-self._exponent + 1)
        i = len(initial) + self._exponent
        nonrepetend = sign + initial[:i] + "." + initial[i:]
        if self._repetend:
            repetend = "(" + self._repetend + ")"
            return nonrepetend + repetend
        else:
            return nonrepetend.rstrip(".")


_parser = re.compile(
    r"""
    (?P<sign>[-+])?
    (?P<int>\d*)
    (
        (\.(?P<frac>\d*))
        (\((?P<repetend>\d*)\))?
    )?
""",
    re.VERBOSE | re.IGNORECASE,
).fullmatch
