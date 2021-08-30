from project_euler_math.polynomial import Polynomial


from pytest import approx


def test_call():
    p = Polynomial([-1, 0, 1])
    assert p(-1) == 0
    assert p(0) == -1
    assert p(1) == 0

    q = Polynomial([1, 0, 1])
    assert q(-1.j) == approx(0)
    assert q(0) == approx(1)
    assert q(1.j) == approx(0)
