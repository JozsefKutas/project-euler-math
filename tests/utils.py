from _pytest.python_api import approx


def assert_iterable_equal(actual, expected):
    for a, e in zip(actual, expected):
        assert a == e


def assert_iterable_almost_equal(actual, expected, **kwargs):
    for a, e in zip(actual, expected):
        assert a == approx(e, **kwargs)