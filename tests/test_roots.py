from project_euler_math.polynomial import Polynomial
from project_euler_math.roots import newton_raphson, secant, bisect

from pytest import approx


def test_newton_raphson():
    p = Polynomial([-1, 0, 1])
    pprime = Polynomial([0, 2])
    assert newton_raphson(p, pprime, 10.) in (approx(1.), approx(-1.))


def test_secant():
    p = Polynomial([-1, 0, 1])
    assert secant(p, 10.) in (approx(1.), approx(-1.))


def test_bisect():
    p = Polynomial([-1, 0, 1])
    assert bisect(p, 0., 10) == approx(1.)
