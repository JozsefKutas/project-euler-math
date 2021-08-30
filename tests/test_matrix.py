from fractions import Fraction

import pytest

from project_euler_math.matrix import Vector, Matrix, SingularMatrixError
from tests.utils import assert_iterable_equal, assert_iterable_almost_equal


def test_matmul():
    a = Matrix([[0, 1], [2, 3]])
    b = Matrix([[4, 3], [2, 1]])
    x = Vector([1, -1])

    assert_iterable_equal(Matrix.identity(2) @ a, a)
    assert_iterable_equal(a @ Matrix.identity(2), a)
    assert_iterable_equal(a @ b, Matrix([[2, 1], [14, 9]]))
    assert_iterable_equal(b @ a, Matrix([[6, 13], [2, 5]]))
    assert_iterable_equal(a @ x, Vector([-1, -1]))
    assert_iterable_equal(x @ a, Vector([-2, -2]))


def test_solve():
    a = Matrix([[0, 1], [2, 3]])
    x = Vector([1, -1])

    assert_iterable_equal(Matrix.identity(2).solve(x), x)
    assert_iterable_equal(a.solve(x), Vector([-2, 1]))


def test_inv():
    assert_iterable_equal(Matrix.identity(3).inv(), Matrix.identity(3))
    with pytest.raises(SingularMatrixError):
        Matrix.ones((3, 3)).inv()
    assert_iterable_equal(Matrix([[0, 1], [2, 3]]).map(Fraction).inv(),
                          Matrix([[-Fraction(3, 2), Fraction(1, 2)], [1, 0]]))
    assert_iterable_almost_equal(Matrix([[0, 1], [2, 3]]).inv(),
                                 Matrix([[-1.5, 0.5], [1, 0]]))


def test_det():
    assert Matrix.identity(3).det() == 1
    assert Matrix.ones((3, 3)).det() == 0
    assert Matrix([[0, 1], [2, 3]]).det() == -2
