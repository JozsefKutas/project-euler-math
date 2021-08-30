from project_euler_math.matrix import Vector, Matrix
from project_euler_math.optimisation import nelder_mead
from tests.utils import assert_iterable_almost_equal


def test_nelder_mead():
    def f(x):
        y = x + Vector([-1, 1])
        res = y @ Matrix([[1, 1], [0, 1]]) @ y
        return res

    assert_iterable_almost_equal(nelder_mead(f, Vector([0, 0])), Vector([1, -1]))
