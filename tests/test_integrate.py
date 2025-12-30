from pytest import approx

from project_euler_math.integrate import quad
from project_euler_math.polynomial import Polynomial


def test_quad():
    p = Polynomial([0, 0, 1])
    assert quad(p, 0.0, 1.0) == approx(1.0 / 3.0)
