from pytest import approx

from project_euler_math.polynomial import Polynomial
from project_euler_math.roots import newton_raphson, secant, bisect


def test_newton_raphson():
    p = Polynomial([-1, 0, 1])
    pprime = Polynomial([0, 2])
    assert newton_raphson(p, pprime, 10.0) in (approx(1.0), approx(-1.0))


def test_secant():
    p = Polynomial([-1, 0, 1])
    assert secant(p, 10.0) in (approx(1.0), approx(-1.0))


def test_bisect():
    p = Polynomial([-1, 0, 1])
    assert bisect(p, 0.0, 10) == approx(1.0)
