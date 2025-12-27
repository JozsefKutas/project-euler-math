from itertools import accumulate, count, product
from math import prod, factorial, isqrt, comb
from typing import Iterator, Sequence, List

from project_euler_math.matrix import Matrix, Vector


def multinomial(*k: int) -> int:
    """Return the binomial coefficient ``n C k1,...,km`` where
    ``n = k1 + ... + km``."""
    if any(ki < 0 for ki in k):
        return 0
    return factorial(sum(k)) // prod(factorial(ki) for ki in k)


def stirling2_ord(n: int, k: int) -> int:
    """Return the Stirling number of the second kind ``{n k}``."""
    return sum((-1) ** (k - i) * comb(k, i) * i**n for i in range(k + 1))


def stirling2(n: int, k: int) -> int:
    """Return the Stirling number of the second kind ``{n k}``."""
    return stirling2_ord(n, k) // factorial(k)


def bell(n: int) -> int:
    """Return the `n`-th Bell number."""
    return bell_list(n + 1)[-1]


def bell_list(end: int) -> List[int]:
    """Return a list of length `end`, the i-th element of which is the i-th Bell
    number."""
    combs = [0] * end
    combs[0] = 1

    bells = [1] * end
    for i in range(2, end):
        for j in range(i - 1, 0, -1):
            combs[j] += combs[j - 1]
            bells[i] += combs[j] * bells[j]
    return bells


def partition(n: int) -> int:
    """Return the partition number of a non-negative integer `n`."""
    return partition_list(n + 1)[-1]


def partition_list(end: int) -> List[int]:
    """Return a list of length `end`, the i-th element of which is the partition
    number of i."""
    partitions = [0] * end
    partitions[0] = 1
    for i in range(1, end):
        for j, s in product(range(1, i + 1), (-1, 1)):
            p = j * (3 * j + s) // 2
            if p > i:
                break
            partitions[i] += (-1) ** (j - 1) * partitions[i - p]
    return partitions


def partition_sequence(n: int, seq: Sequence[int]) -> int:
    """Return the number of ways `n` can be partitioned (ignoring order)
    into partitions of sizes specified in `seq`."""
    return partition_sequence_list(n + 1, seq)[-1]


def partition_sequence_list(end: int, seq: Sequence[int]) -> List[int]:
    """Return a list of length `end`, the i-th element of which is the number of
    ways i can be partitioned (ignoring order) into partitions of sizes
    specified in `seq`."""
    partitions = [0] * end
    partitions[0] = 1
    for a in seq:
        for i in range(a, end):
            partitions[i] += partitions[i - a]
    return partitions


def fibonacci(i: int, a: int = 0, b: int = 1) -> int:
    """Return the `i`-th Fibonacci number."""
    if i < 0:
        return fibonacci(-i)

    mat = Matrix([[0, 1], [1, 1]])
    vec = Vector([a, b])
    return (mat.matrix_power(i) @ vec)[0]


def fibonacci_numbers(a: int = 0, b: int = 1) -> Iterator[int]:
    """Generate Fibonacci numbers."""
    x, y = a, b
    while True:
        yield x
        x, y = y, x + y


def kgon(i: int, k: int) -> int:
    """Return the `i`-th `k`-gonal number."""
    return ((k - 2) * i**2 - (k - 4) * i) // 2


def kgon_numbers(k: int) -> Iterator[int]:
    """Generate `k`-gonal numbers."""
    yield from accumulate(count(), lambda y, i: y + (k - 2) * (i - 1) + 1)


def is_kgon(n: int, k: int) -> bool:
    """Return True is `n` is a k-gon, False otherwise."""
    d = (k - 4) * (k - 4) + 8 * (k - 2) * n
    root_d = isqrt(d)
    return root_d * root_d == d and ((k - 4) + root_d) % (2 * (k - 2)) == 0
