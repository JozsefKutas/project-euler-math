from itertools import chain, combinations, islice
from collections import defaultdict
from typing import (
    Optional, Iterable, Sequence, List, MutableMapping, Callable, TypeVar)


T = TypeVar('T')
K = TypeVar('K')
V = TypeVar('V')


def chunks(iterable: Iterable[T], chunk_size: int) -> Iterable[List[T]]:
    """Return an iterable split into chunks."""
    it = iter(iterable)
    for x in it:
        yield [x] + list(islice(it, chunk_size-1))


def interleave(*iterables: Iterable[T]) -> Iterable[T]:
    """Return an iterable split into chunks."""
    yield from chain.from_iterable(zip(*iterables))


def rotations(iterable: Iterable[T]) -> Iterable[Sequence[T]]:
    """Generate rotations of `sequence`."""
    tup = tuple(iterable)
    for i in range(len(tup)):
        yield tup[i:] + tup[:i]


def powerset(iterable: Iterable[T], nonempty: bool = False)\
        -> Iterable[Sequence[T]]:
    """Return the powerset of a finite iterable."""
    tup = tuple(iterable)
    rng = range(1 if nonempty else 0, len(tup)+1)
    return chain.from_iterable(combinations(tup, r) for r in rng)


def groupby(iterable: Iterable[T], key: Callable[[T], K],
            downstream: Optional[Callable[[Sequence[T]], V]] = None)\
        -> MutableMapping[K, V]:
    """Return a defaultdict, containing the elements of an iterable grouped by a
    key function."""
    gb = defaultdict(list)
    for x in iterable:
        gb[key(x)].append(x)
    return {k: downstream(v) for k, v in gb.items()} if downstream else gb
