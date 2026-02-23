from collections import deque
from collections.abc import Iterable
from itertools import chain, combinations, islice
from typing import Callable, Iterator


def sliding_window[T](iterable: Iterable[T], n: int) -> Iterator[tuple[T, ...]]:
    """Collect data into overlapping fixed-length chunks or blocks."""
    # taken from https://docs.python.org/3/library/itertools.html
    # sliding_window('ABCDEFG', 3) → ABC BCD CDE DEF EFG
    iterator = iter(iterable)
    window = deque(islice(iterator, n - 1), maxlen=n)
    for x in iterator:
        window.append(x)
        yield tuple(window)


def rotations[T](iterable: Iterable[T]) -> Iterator[tuple[T, ...]]:
    """Generate rotations of a finite iterable."""
    tup = tuple(iterable)
    for i in range(len(tup)):
        yield tup[i:] + tup[:i]


def powerset[T](
        iterable: Iterable[T], nonempty: bool = False, proper: bool = False
) -> Iterator[tuple[T, ...]]:
    """Return the powerset of a finite iterable."""
    tup = tuple(iterable)
    rng = range(1 if nonempty else 0, len(tup) + (1 if proper else 0))
    return chain.from_iterable(combinations(tup, r) for r in rng)


def groupby[T, K, V](
        iterable: Iterable[T],
        key: Callable[[T], K],
        downstream: Callable[[list[T]], V] | None = None,
) -> dict[K, list[T]] | dict[K, V]:
    """Return a dictionary containing the elements of an iterable grouped by a
    key function."""
    gb: dict[K, list[T]] = {}
    for x in iterable:
        gb.setdefault(key(x), []).append(x)
    return {k: downstream(v) for k, v in gb.items()} if downstream else gb


class Memo[T]:
    """Memoize values in an iterable in an expanding list."""

    _iterator: Iterator[T]
    _list: list[T]

    __slots__ = ("_iterator", "_list")

    def __init__(self, iterable: Iterable[T]):
        self._iterator = iter(iterable)
        self._list = []

    def __getitem__(self, key: int) -> T:
        if key < 0:
            raise IndexError("indices must be non-negative")

        it = self._iterator
        lis = self._list
        for i in range(len(lis), key + 1):
            lis.append(next(it))
        return lis[key]
