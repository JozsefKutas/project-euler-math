from typing import Callable


def newton_raphson(
    f: Callable[..., float],
    fprime: Callable[..., float],
    x0: float,
    args: tuple = (),
    tol: float = 1e-8,
    maxiter: int = 50,
) -> float:
    """Searches for the root of `f` using the Newton-Raphson algorithm."""

    for _ in range(maxiter):
        y0 = f(x0, *args)
        if y0 == 0.0:
            return x0

        yprime0 = fprime(x0, *args)
        x1 = x0 - y0 / yprime0
        if abs(x1 - x0) < tol:
            return x1

        x0 = x1

    raise RootError


def secant(
    f: Callable[..., float],
    x0: float,
    x1: float | None = None,
    args: tuple = (),
    tol: float = 1e-8,
    maxiter: int = 50,
) -> float:
    """Searches for the root of `f` using the secant algorithm."""

    if x1 is None:
        eps = 1e-4
        x1 = x0 * (1.0 + eps) + (eps if x0 >= 0.0 else -eps)

    y0 = f(x0, *args)
    y1 = f(x1, *args)
    if y0 * y1 >= 0.0:
        msg = f"function sign is same at x0 and x1: f({x0}) = {y0}, f({x1}) = {y1}"
        raise ValueError(msg)

    if abs(y0) < abs(y1):
        x1, x0 = x0, x1
        y1, y0 = y0, y1

    for _ in range(maxiter):
        y1 = f(x1, *args)
        if abs(x1 - x0) < tol or y1 == 0.0:
            return x1
        x1, x0 = x1 - y1 * (x1 - x0) / (y1 - y0), x1
        y0 = y1

    raise RootError


def bisect(
    f: Callable[..., float],
    x0: float,
    x1: float,
    args: tuple = (),
    tol: float = 1e-8,
    maxiter: int = 100,
) -> float:
    """Searches for the root of `f` using bisection search."""

    y0 = f(x0, *args)
    y1 = f(x1, *args)
    if y0 * y1 >= 0.0:
        msg = f"function sign is same at x0 and x1: f({x0}) = {y0}, f({x1}) = {y1}"
        raise ValueError(msg)

    for _ in range(maxiter):
        x2 = (x0 + x1) / 2.0
        y2 = f(x2, *args)
        if y0 * y2 < 0.0:
            x1, y1 = x2, y2
        else:
            x0, y0 = x2, y2

        if abs(x0 - x1) < tol or y2 == 0.0:
            return x2

    raise RootError


class RootError(Exception):
    def __init__(self) -> None:
        super().__init__("root not found")
