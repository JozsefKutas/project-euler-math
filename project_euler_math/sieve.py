from bisect import bisect
from itertools import compress
from typing import Union, Iterator, List

from project_euler_math.ntheory import primes_list


class PrimeSieve:
    """A prime sieve using Eratosthenes's algorithm."""

    _end: int
    _primes_list: List[int]

    __slots__ = ('_end', '_primes_list')

    EXPAND_FACTOR = 2

    def __init__(self, end: int = 1_000_000) -> None:
        self._end = end
        self._primes_list = primes_list(end)

    def prime(self, key: Union[int, slice]) -> int:
        if isinstance(key, int):
            if key < 0:
                raise IndexError("indices must be non-negative")
            self._extend_to_prime(key)
            return self._primes_list[key]

        else:
            raise TypeError("indices must be integers, not "
                            + type(key).__name__)

    def __getitem__(self, key: Union[int, slice]) -> int:
        return self.prime(key)

    def _extend_to_n(self, n: int) -> None:
        if n < self._end:
            return

        n = max(n, int(self._end * self.EXPAND_FACTOR))

        while self._end ** 2 <= n:
            self._do_extend_to_n(self._end ** 2)
        self._do_extend_to_n(n)

    def _do_extend_to_n(self, n: int) -> None:
        ext_size = n + 1 - len(self._primes_list)
        primality = [True] * ext_size
        for p in self._primes_list:
            start_index = max(p*p - self._end, -self._end % p)
            if start_index >= ext_size:
                break
            for i in range(start_index, ext_size, p):
                primality[i] = False
        primes = list(compress(range(self._end, n + 1), primality))

        self._end = n + 1
        self._primes_list += primes

    def _extend_to_prime(self, i: int) -> None:
        while i >= len(self._primes_list):
            self._extend_to_n(int(self._end * self.EXPAND_FACTOR))

    def _extend(self) -> None:
        self._extend_to_n(self._end)

    def primes(self) -> Iterator[int]:
        yield from self._primes_list
        while True:
            start = len(self._primes_list)
            self._extend()
            yield from self._primes_list[start:]

    def composites(self) -> Iterator[int]:
        prevp = 1
        for p in self.primes():
            yield from range(prevp+1, p)
            prevp = p

    def __iter__(self) -> Iterator[int]:
        return self.primes()

    def prime_count(self, n: int) -> int:
        self._extend_to_n(n)
        return bisect(self._primes_list, n)
