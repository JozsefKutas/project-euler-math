import string

ALPHABET = string.digits + string.ascii_letters


class NumeralSystem:
    """
    A numeral system used to represent integers.
    """

    _symbols: str
    _base: int
    _chunksize: int
    _pow_base: int
    _cache: list[str]
    _reverse: dict[str, int]

    __slots__ = ("_symbols", "_base", "_chunksize", "_pow_base", "_cache", "_reverse")

    @property
    def symbols(self) -> str:
        return self._symbols

    @property
    def base(self) -> int:
        return self._base

    @property
    def cache_pow(self) -> int:
        return self._chunksize

    def __init__(
        self, symbols: str | None = None, base: int | None = None, chunksize: int = 1
    ) -> None:

        if symbols is None and base is None:
            raise ValueError

        if base is None:
            if not isinstance(symbols, str):
                raise ValueError
            base = len(symbols)

        elif symbols is None:
            if not isinstance(base, int) or base > len(ALPHABET):
                raise ValueError
            symbols = ALPHABET[:base]

        cache = self._build_cache(symbols, chunksize)
        self._symbols = symbols
        self._base = base
        self._chunksize = chunksize
        self._pow_base = base**chunksize
        self._cache = cache
        self._reverse = {s: i for i, s in enumerate(cache)}

    @staticmethod
    def _build_cache(symbols: str, cache_pow: int) -> list[str]:
        cache = [""]
        for _ in range(cache_pow):
            cache = [s + c for s in symbols for c in cache]
        return cache

    def represent(self, n: int) -> str:
        if n == 0:
            return self.symbols[0]
        elif n < 0:
            return "-" + self.represent(-n)

        cache = self._cache
        pow_base = self._pow_base
        representation = []
        while n:
            n, d = divmod(n, pow_base)
            representation.append(cache[d])
        return "".join(reversed(representation)).lstrip("0")

    def int_value(self, representation: str) -> int:
        representation = representation.strip()

        if representation[0] == "-":
            return -self.int_value(representation[1:])

        reverse = self._reverse
        cache_pow = self._chunksize
        pow_base = self._pow_base
        n = 0
        width = (len(representation) + cache_pow - 1) // cache_pow * cache_pow
        representation = representation.zfill(width)
        for i in range(0, len(representation), cache_pow):
            n *= pow_base
            s = representation[i : i + cache_pow]
            n += reverse[s]
        return n

    def __repr__(self) -> str:
        return type(self).__name__ + f"({self._symbols!r})"
