from itertools import islice

from project_euler_math.combinatorics import (
    bell_list, partition_list, fibonacci_numbers, kgon_numbers)


def test_bell_list():
    assert bell_list(10) == [1, 1, 2, 5, 15, 52, 203, 877, 4140, 21147]


def test_partition_list():
    assert partition_list(10) == [1, 1, 2, 3, 5, 7, 11, 15, 22, 30]


def test_fibonacci_numbers():
    assert list(islice(fibonacci_numbers(), 10)) == [0, 1, 1, 2, 3, 5, 8, 13, 21, 34]


def test_kgon_numbers():
    assert list(islice(kgon_numbers(3), 10)) == [0, 1, 3, 6, 10, 15, 21, 28, 36, 45]
    assert list(islice(kgon_numbers(4), 10)) == [0, 1, 4, 9, 16, 25, 36, 49, 64, 81]
