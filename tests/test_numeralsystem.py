from project_euler_math.numeralsystem import NumeralSystem


def test_represent_binary():
    ns = NumeralSystem('01')
    for n in range(-100, 100):
        assert ns.represent(n) == format(n, 'b')


def test_represent_octal():
    ns = NumeralSystem('01234567')
    for n in range(-100, 100):
        assert ns.represent(n) == format(n, 'o')


def test_represent_hex():
    ns = NumeralSystem('0123456789abcdef')
    for n in range(-100, 100):
        assert ns.represent(n) == format(n, 'x')


def test_int_value_binary():
    ns = NumeralSystem('01')
    for n in range(-100, 100):
        assert ns.int_value(format(n, 'b')) == n


def test_int_value_octal():
    ns = NumeralSystem('01234567')
    for n in range(-100, 100):
        assert ns.int_value(format(n, 'o')) == n


def test_int_value_hex():
    ns = NumeralSystem('0123456789abcdef')
    for n in range(-100, 100):
        assert ns.int_value(format(n, 'x')) == n
