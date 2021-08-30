from fractions import Fraction

from project_euler_math.repeatingdecimal import RepeatingDecimal


def test_from_fraction():
    assert RepeatingDecimal.from_fraction(Fraction(0)) == RepeatingDecimal('0')
    assert RepeatingDecimal.from_fraction(Fraction(1, 2)) == RepeatingDecimal('0.5')
    assert RepeatingDecimal.from_fraction(Fraction(2, 3)) == RepeatingDecimal('0.(6)')
    assert RepeatingDecimal.from_fraction(Fraction(1, 7)) == RepeatingDecimal('0.(142857)')
    assert RepeatingDecimal.from_fraction(Fraction(200, 3)) == RepeatingDecimal('66.(6)')
    assert RepeatingDecimal.from_fraction(Fraction(1, 700)) == RepeatingDecimal('0.00(142857)')
    assert RepeatingDecimal.from_fraction(-Fraction(1, 2)) == RepeatingDecimal('-0.5')


def test_to_fraction():
    for m in range(-100, 100):
        for n in range(1, 100):
            p = Fraction(m, n)
            assert RepeatingDecimal.from_fraction(p).to_fraction() == p
