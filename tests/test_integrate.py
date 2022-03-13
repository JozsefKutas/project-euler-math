from project_euler_math.polynomial import Polynomial
from project_euler_math.integrate import quad

from pytest import approx


def test_quad():
    p = Polynomial([0, 0, 1])
    assert quad(p, 0., 1.) == approx(1. / 3.)
