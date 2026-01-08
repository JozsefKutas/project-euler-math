from collections.abc import Iterable
from itertools import chain, combinations
from typing import Callable


def rotations[T](iterable: Iterable[T]) -> Iterable[tuple[T, ...]]:
    """Generate rotations of a finite iterable."""
    tup = tuple(iterable)
    for i in range(len(tup)):
        yield tup[i:] + tup[:i]


def powerset[T](
    iterable: Iterable[T], nonempty: bool = False
) -> Iterable[tuple[T, ...]]:
    """Return the powerset of a finite iterable."""
    tup = tuple(iterable)
    rng = range(1 if nonempty else 0, len(tup) + 1)
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
