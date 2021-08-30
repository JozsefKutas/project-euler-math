from itertools import islice

from project_euler_math.continuedfraction import ContinuedFraction


def test_from_quadratic():
    assert ContinuedFraction.from_quadratic(0) == ContinuedFraction([0], [])
    assert ContinuedFraction.from_quadratic(2) == ContinuedFraction([2], [])
    assert ContinuedFraction.from_quadratic(1, 2) == ContinuedFraction([0, 2], [])
    assert ContinuedFraction.from_quadratic(13, 8) == ContinuedFraction([1, 1, 1, 1, 2], [])
    assert ContinuedFraction.from_quadratic(0, 1, 2) == ContinuedFraction([1], [2])
    assert ContinuedFraction.from_quadratic(0, 1, 3) == ContinuedFraction([1], [1, 2])
    assert ContinuedFraction.from_quadratic(0, 1, 5) == ContinuedFraction([2], [4])
    assert ContinuedFraction.from_quadratic(0, 1, 7) == ContinuedFraction([2], [1, 1, 1, 4])
    assert ContinuedFraction.from_quadratic(1, 2, 5) == ContinuedFraction([], [1])


def test_convergents():
    assert list(ContinuedFraction.from_quadratic(2).convergents()) == [(2, 1)]
    assert list(ContinuedFraction.from_quadratic(1, 2).convergents()) == [(0, 1), (1, 2)]
    # Pell's equation
    assert (list(islice(ContinuedFraction([1], [2]).convergents(), 5))
            == [(1, 1), (3, 2), (7, 5), (17, 12), (41, 29)])
    # Fibonacci sequence
    assert (list(islice(ContinuedFraction([], [1]).convergents(), 5))
            == [(1, 1), (2, 1), (3, 2), (5, 3), (8, 5)])
