from typing import Callable


def quad(f: Callable[..., float],
         a: float, b: float, args: tuple = (),
         tol: float = 1e-12, maxiter: int = 20):
    """Calculates the integral of `f` between `a` and `b` using the trapezium
    rule."""

    delta = b - a
    integral = (f(a, *args) + f(b, *args)) / 2. * delta

    for _ in range(maxiter):
        x = a + delta / 2.
        total = 0.
        while x < b:
            total += f(x, *args)
            x += delta
        integral, old = (integral + total * delta) / 2., integral

        if abs(integral - old) < tol * abs(old) or (integral == old == 0.):
            return integral

        delta /= 2.

    raise ValueError
