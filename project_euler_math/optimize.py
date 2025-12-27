from typing import Callable

from project_euler_math.matrix import Vector


def nelder_mead(
    f: Callable[..., float],
    x0: Vector,
    deltas: Vector | None = None,
    scale=1.0,
    args: tuple = (),
    ftol: float = 1e-12,
    maxiter: int = 1000,
) -> None:
    """Searches for the minimum value of `f` using the Nelder-Mead algorithm."""

    reflect_factor = -1.0
    expand_factor = -2.0
    contract_factor = 0.5
    shrink_factor = 0.5

    eps = 1e-10
    n = len(x0)
    simplex = [(x0, f(x0, *args))]
    if deltas:
        for i in range(n):
            x = x0 + deltas[i]
            simplex.append((x, f(x, *args)))
    else:
        for i in range(n):
            ej = Vector([scale if i == j else 0.0 for j in range(n)])
            x = x0 + ej
            simplex.append((x, f(x, *args)))

    xsum = sum(p[0] for p in simplex)

    for _ in range(maxiter):
        simplex.sort(key=lambda p: p[1])
        xlo, ylo = simplex[0]
        xhi, yhi = simplex[-1]
        xnhi, ynhi = simplex[-2]
        if 2.0 * abs(yhi - ylo) / (abs(yhi) + abs(ylo) + eps) < ftol:
            return xlo

        x = _nelder_mead_transform(xhi, xsum, reflect_factor)
        y = f(x, *args)
        if y <= ylo:
            x = _nelder_mead_transform(xhi, xsum, expand_factor)
            y = f(x, *args)
        elif y >= ynhi:
            yold = y
            x = _nelder_mead_transform(xhi, xsum, contract_factor)
            y = f(x, *args)
            if y >= yold:
                b = 1.0 - shrink_factor
                for i, (x, y) in enumerate(simplex[1:], 1):
                    x = shrink_factor * x + b * xlo
                    simplex[i] = (x, f(x, *args))
                xsum = sum(p[0] for p in simplex)
                continue

        xsum += x - xhi
        simplex[-1] = (x, y)

    raise ValueError


def _nelder_mead_transform(x, xsum, factor):
    n = len(x)
    a = (1.0 - factor) / n
    b = factor - a
    return a * xsum + b * x
