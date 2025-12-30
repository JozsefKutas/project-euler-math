from pytest import approx

from project_euler_math.polynomial import Polynomial


def test_call():
    p = Polynomial([-1, 0, 1])
    assert p(-1) == 0
    assert p(0) == -1
    assert p(1) == 0

    q = Polynomial([1, 0, 1])
    assert q(-1.0j) == approx(0)
    assert q(0) == approx(1)
    assert q(1.0j) == approx(0)
