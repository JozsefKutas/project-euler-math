import typing
from collections import Counter
from fractions import Fraction
from itertools import compress, count, accumulate
from math import prod, inf, isqrt, gcd as mathgcd, lcm as mathlcm
from numbers import Integral
from random import Random
from typing import Sequence, List, Mapping, Optional, Iterator, Callable, TypeVar

from project_euler_math.eisenstein import Eisenstein
from project_euler_math.gaussian import Gaussian
from project_euler_math.polynomial import Polynomial

PRIME_FACTORS_END = 10000


E = TypeVar("E", Integral, Gaussian, Eisenstein, Polynomial)


def inrt(n: int, k: int) -> int:
    """Return the integer part of the `k`-th root of a non-negative integer
    `n`."""
    if n < 0:
        raise ValueError
    if k <= 0:
        raise ValueError

    # Newton-Raphson
    if n == 0:
        return 0
    e = (int.bit_length(n) - 1) // k
    m = 1 << e
    delta = inf
    while True:
        m, m_old = ((k - 1) * m + n // m ** (k - 1)) // k, m
        delta, delta_old = abs(m - m_old), delta
        if delta >= delta_old:
            break
    while m**k < n:
        m += 1
    while m**k > n:
        m -= 1
    return m


def _is_square_mod(mod: int):
    ans = [False for _ in range(mod)]
    for i in range((mod + 1) // 2):
        ans[(i * i) % mod] = True
    return ans


_is_square63 = _is_square_mod(63)
_is_square64 = _is_square_mod(64)
_is_square65 = _is_square_mod(65)


def is_square(n: int) -> bool:
    """Return True is `n` is a square, False otherwise."""
    return (
        _is_square64[n & 0b111111]
        and _is_square63[n % 63]
        and _is_square65[n % 65]
        and isqrt(n) ** 2 == n
    )


def gcd(m: E, n: E) -> E:
    """Return the non-negative greatest common divisor of integers `m` and
    `n`."""
    while n != 0:
        m, n = n, m % n
    return m


def lcm(m: E, n: E) -> E:
    """Return the non-negative lowest common multiple of integers `m` and
    `n`."""
    if m == n == 0:
        return 0
    return m * n // gcd(m, n)


def bezout(m: E, n: E) -> Sequence[E]:
    """Return the tuple ``(d, s, t)``, where ``d`` is the non-negative highest
    common factor of integers `m` and `n`, and ``d = s*m + t*n``."""
    d_old, d = m, n
    s_old, s = 1, 0
    t_old, t = 0, 1
    while d != 0:
        q, rem = divmod(d_old, d)
        d_old, d = d, rem
        s_old, s = s, s_old - q * s
        t_old, t = t, t_old - q * t
    return d_old, s_old, t_old


def mod_inverse(a: E, mod: E) -> Optional[E]:
    """Return the inverse of `a` modulo `mod`. If no inverse exists (`a` and
    `mod` are not coprime), returns None."""
    # similar to bezout, but don't need to calculate t
    d_old, d = a, mod
    s_old, s = 1, 0
    while d != 0:
        q, rem = divmod(d_old, d)
        d_old, d = d, rem
        s_old, s = s, s_old - q * s
    if d_old % mod != 1:
        return None
    return s_old % mod


def crt(residues: Sequence[E], mods: Sequence[E]) -> E:
    """Compute the solution to the Chinese remainder theorem problem. `mods`
    should contain pairwise coprime numbers."""
    residue = residues[0]
    mod = mods[0]
    for r, m in zip(residues[1:], mods[1:]):
        residue = crt2(residue, r, mod, m)
        mod *= m
    return residue


def crt2(a: E, b: E, m: E, n: E) -> E:
    """Compute the solution to the two number Chinese remainder theorem problem.
    `m` and `n` must be coprime."""
    inv = mod_inverse(m, n)
    if inv is None:
        raise ValueError
    return (a + (b - a) * m * inv) % (m * n)


def crt2_noncoprime(a: E, b: E, m: E, n: E) -> Optional[E]:
    """Compute a solution to the two number Chinese remainder theorem problem.
    `m` and `n` need not be coprime. If there exists no solution, returns
    None."""
    d, s, t = bezout(m, n)
    q, r = divmod(b - a, d)
    if r != 0:
        return None
    return (a + q * m * s) % ((m * n) // d)


def farey_sequence(n: int) -> List[Fraction]:
    """Return the Farey sequence of order `n`."""
    seq = [Fraction(0), Fraction(1)]
    for q in range(2, n + 1):
        for p in range(1, q):
            if mathgcd(p, q) == 1:
                seq.append(Fraction(p, q))
    seq.sort()
    return seq


def left_farey(n: int, x: Fraction) -> Fraction:
    """Return the fraction to the left of `x` in the Farey sequence of order
    `n`."""
    _, s, t = bezout(x.denominator, x.numerator)
    k = (n - t) // x.denominator
    s -= k * x.numerator
    t += k * x.denominator
    return Fraction(-s, t)


def right_farey(n: int, x: Fraction) -> Fraction:
    """Return the fraction to the right of `x` in the Farey sequence of order
    `n`."""
    _, s, t = bezout(x.denominator, x.numerator)
    k = (n + t) // x.denominator
    s += k * x.numerator
    t -= k * x.denominator
    return Fraction(s, -t)


def jacobi(a: int, m: int) -> int:
    """Return the Jacobi symbol of `a` over `m`."""
    a %= m
    ans = 1
    while a:
        two_nonresidue = m % 8 in (3, 5)
        while a % 2 == 0:
            a //= 2
            if two_nonresidue:
                ans *= -1
        a, m = m, a
        if a % 4 == m % 4 == 3:
            ans *= -1
        a %= m
    return ans if m == 1 else 0


def tonelli_shanks(a: int, p: int, seed=42) -> int:
    """Return a square root of `a` mod `p`, if `a` is a quadratic residue."""
    a %= p
    if p % 4 == 3:
        x = pow(a, (p + 1) // 4, p)
    elif p % 8 == 5:
        x = pow(a, (p + 3) // 8, p)
        if pow(x, 2, p) != a:
            x = (x * pow(2, (p - 1) // 4, p)) % p
    else:
        d = _quadratic_nonresidue(p, seed)
        s = 0
        t = p - 1
        while t % 2 == 0:
            s += 1
            t //= 2
        at = pow(a, t, p)
        dt = pow(d, t, p)
        m = 0
        pow2 = 1 << s - 1
        for i in range(s):
            if pow(at * pow(dt, m, p), pow2, p) == p - 1:
                m += 1 << i
            pow2 //= 2
        x = (pow(a, (t + 1) // 2, p) * pow(dt, m // 2, p)) % p
    return x


def _quadratic_nonresidue(p: int, seed) -> int:
    rand = Random(seed)
    while True:
        d = rand.randrange(2, p)
        if jacobi(d, p) == -1:
            return d


def is_prime(n: int, prime_factors: Optional[Sequence[int]] = None) -> bool:
    """Return True if the positive integer `n` is prime, False otherwise.
    First tries trial division using small primes, then the Miller-Rabin
    test with bases from Jaeschke (1993) and Sorenson and Webster (2015)
    to deterministically test for primality."""
    if -1 <= n <= 1:
        return False
    elif n < -1:
        return is_prime(-n, prime_factors)

    prime_factors = prime_factors or _prime_factors
    if n < len(prime_factors):
        return prime_factors[n] == n

    # try trial division
    if n % 2 == 0 or n % 3 == 0 or n % 5 == 0 or n % 7 == 0:
        return False

    # try deterministic Miller-Rabin
    if n < 1_373_653:
        return miller_rabin(n, (2, 3))
    elif n < 25_326_001:
        return miller_rabin(n, (2, 3, 5))
    elif n < 3_215_031_751:
        return miller_rabin(n, (2, 3, 5, 7))
    elif n < 2_152_302_898_747:
        return miller_rabin(n, (2, 3, 5, 7, 11))
    elif n < 3_474_749_660_383:
        return miller_rabin(n, (2, 3, 5, 7, 11, 13))
    elif n < 341_550_071_728_321:
        return miller_rabin(n, (2, 3, 5, 7, 11, 13, 17))
    elif n < 3_825_123_056_546_413_051:
        return miller_rabin(n, (2, 3, 5, 7, 11, 13, 17, 19, 23))
    elif n < 318_665_857_834_031_151_167_461:
        return miller_rabin(n, (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37))
    elif n < 3_317_044_064_679_887_385_961_981:
        return miller_rabin(n, (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41))

    raise ValueError(f"n out of range: {n}")


def miller_rabin(n: int, bases: Sequence[int] = None) -> bool:
    """Performs the Miller-Rabin primality test on `n` using the specified
    bases."""
    s = 0
    t = n - 1
    while t % 2 == 0:
        s += 1
        t //= 2
    return not any(_miller_rabin_witness(n, a, s, t) for a in bases)


def _miller_rabin_witness(n: int, a: int, s: int, t: int) -> bool:
    a %= n
    if a < 2:
        return False

    b = pow(a, t, n)
    if b == n - 1 or b == 1:
        return False
    for _ in range(s):
        b = pow(b, 2, n)
        if b == n - 1:
            return False
    return True


def pseudoprime_generator(limit: int, seed: int = 42) -> Iterator[int]:
    """Generate random pseudoprimes ``q`` in the range ``[2, limit)``."""
    rand = Random(seed)
    while True:
        n = rand.randrange(2, limit)
        if is_prime(n):
            yield n


def padic_val(n: int, p: int) -> int | float:
    """Return the `p`-adic valuation of `n`."""
    if n == 0:
        return inf
    e = 0
    while n % p == 0:
        n //= p
        e += 1
    return e


_wheel = [
    x - 11
    for x in range(11, 11 + 210)
    if x % 2 != 0 and x % 3 != 0 and x % 5 != 0 and x % 7 != 0
]


def factorisation(
    n: int,
    trial_div_limit: int = 10000,
    pollard_rho_args: Optional[Mapping[int, int]] = None,
    prime_factors: Optional[Sequence[int]] = None,
) -> typing.Counter[int]:
    """Return the prime factorisation of `n`. First tries trail division using
    the wheel factorisation approach up to the specified limit, then uses
    Pollard's rho heuristic for larger factors."""

    if n == 0:
        raise ValueError
    elif -1 <= n <= 1:
        return Counter()
    elif n < -1:
        ans = factorisation(-n, trial_div_limit, pollard_rho_args, prime_factors)
        ans[-1] = 1
        return ans

    ans = Counter()

    prime_factors = prime_factors or _prime_factors
    if n < len(prime_factors):
        while n > 1:
            p = prime_factors[n]
            n //= p
            e = 1
            while n % p == 0:
                n //= p
                e += 1
            ans[p] = e
        return ans

    # wheel factorisation using the basis 2, 3, 5, 7
    for p in (2, 3, 5, 7):
        if n % p == 0:
            n //= p
            e = 1
            while n % p == 0:
                n //= p
                e += 1
            ans[p] = e

            if n == 1:
                return ans

    sqrtn = isqrt(n)

    for base in count(11, 210):
        if base >= trial_div_limit:
            break

        for shift in _wheel:
            pseudop = base + shift
            if pseudop > sqrtn:
                ans[n] = 1
                return ans

            if n % pseudop == 0:
                n //= pseudop
                e = 1
                while n % pseudop == 0:
                    n //= pseudop
                    e += 1
                ans[pseudop] = e

                if n == 1:
                    return ans

                sqrtn = isqrt(n)

    # use Pollard's rho heuristic for large factors
    components = [n]
    pollard_rho_args = pollard_rho_args or {}
    while components:
        n = components.pop()
        if is_prime(n):
            ans[n] += 1
        else:
            m = pollard_rho(n, **pollard_rho_args)
            components.append(m)
            components.append(n // m)

    return ans


def pollard_rho(n: int, a=1, seed=42, k=1, f: Callable[[int], int] = None) -> int:
    """Find a proper factor of `n` using Pollard's rho heuristic. Floyd's
    tortoise and hare algorithm is used to detect cycles."""

    f = f or (lambda x: (x * x + a) % n)
    rand = Random(seed)

    while True:
        u = v = rand.randrange(n)
        while True:
            prod_umv = 1
            for i in range(k):
                u = f(u)
                v = f(f(v))
                prod_umv = (prod_umv * (u - v)) % n
            g = mathgcd(prod_umv, n)
            if g != 1:
                if g != n:
                    return g
                break


def divisors(n: int, fact: Optional[Mapping[int, int]] = None) -> List[int]:
    """Return a list of all divisors of `n`."""
    fact = fact or factorisation(n)
    divs = [1]
    for p, e in fact.items():
        divs_ppow = divs
        for _ in range(e):
            divs_ppow = [d * p for d in divs_ppow]
            divs.extend(divs_ppow)
    return divs


def factorisations(
    n: int, fact: Optional[Mapping[int, int]] = None
) -> List[Sequence[int]]:
    """Return a list of all factorisations of `n`."""
    fact = fact or factorisation(n)

    div_divs = {1: [1]}
    for p, e in fact.items():
        div_divs_copy = dict(div_divs)
        for d, divs_d in div_divs_copy.items():
            ppow = 1
            divs = divs_d
            for _ in range(e):
                ppow *= p
                divs = divs + [dd * ppow for dd in divs_d]
                div_divs[d * ppow] = divs

    for divs in div_divs.values():
        divs.sort()

    def generator(m, max_d):
        if m == 1:
            yield ()
        else:
            for d in div_divs[m][1:]:
                if d > max_d:
                    break
                for f in generator(m // d, d):
                    yield f + (d,)

    yield from generator(n, n)


def omega(n: int, fact: Optional[Mapping[int, int]] = None) -> int:
    """Return the omega of `n`. This is the number of distinct prime factors of
    `n`."""
    fact = fact or factorisation(n)
    return len(fact)


def ndiv(n: int, fact: Optional[Mapping[int, int]] = None) -> int:
    """Return the number of divisors of `n`."""
    fact = fact or factorisation(n)
    return prod(e + 1 for e in fact.values())


def sigma(n: int, k: int = 1, fact: Optional[Mapping[int, int]] = None) -> int:
    """Return the sigma of `n`. This is the sum of the divisors of n, raised to
    the power `k`."""
    if k == 0:
        return ndiv(n, fact)
    if n == 1:
        return 1
    fact = fact or factorisation(n)
    return prod((p ** ((e + 1) * k) - 1) // (p**k - 1) for p, e in fact.items())


def totient(n: int, fact: Optional[Mapping[int, int]] = None) -> int:
    """Return the Euler totient of `n`."""
    fact = fact or factorisation(n)
    return prod(p ** (e - 1) * (p - 1) for p, e in fact.items())


def mobius(n: int, fact: Optional[Mapping[int, int]] = None) -> int:
    """Return the value of the Mobius function of `n`."""
    if n == 1:
        return 1
    fact = fact or factorisation(n)
    return 0 if fact.most_common(1)[0][1] > 1 else (-1) ** len(fact)


def rad(n: int, fact: Optional[Mapping[int, int]] = None) -> int:
    """Return the radical of `n`."""
    fact = fact or factorisation(n)
    return prod(fact)


def sum_squares(n: int, fact: Optional[Mapping[int, int]] = None) -> int:
    """Return the number of ways `n` can be written as a sum of two squares."""
    fact = fact or factorisation(n)
    ans = 1
    for p, e in fact.items():
        if p % 4 == 1:
            ans *= e + 1
        elif p % 4 == 3 and e % 2 == 1:
            return 0
    return 4 * ans


def group_order(
    a: int,
    mod: int,
    phi: Optional[int] = None,
    phi_fact: Optional[Mapping[int, int]] = None,
) -> int:
    """Return the order of `a` in the group of integers mod `mod`."""
    if mathgcd(a, mod) != 1:
        raise ValueError

    phi = phi or totient(mod)
    phi_fact = phi_fact or factorisation(phi)
    ans = phi
    for p, e in phi_fact.items():
        ans //= p**e
        apow = pow(a, ans, mod)
        while apow != 1:
            apow = pow(apow, p, mod)
            ans *= p
    return ans


def is_primitive_root(
    a: int, p: int, phi_fact: Optional[Mapping[int, int]] = None
) -> bool:
    """Return True if `a` is a primitive root in the group of integers mod `p`,
    False otherwise."""
    phi_fact = phi_fact or factorisation(p - 1)
    return a % p != 0 and all(pow(a, (p - 1) // q, p) != 1 for q in phi_fact)


def primitive_root(p: int, phi_fact: Optional[Mapping[int, int]] = None) -> int:
    """Return a primitive root in the group of integers mod `p`."""
    phi_fact = phi_fact or factorisation(p - 1)
    for a in range(1, p):
        if is_primitive_root(a, p, phi_fact):
            return a


def fibonacci_period(n: int, fact: Optional[Mapping[int, int]] = None) -> int:
    """Return a period (not necessarily the minimal period) of the Fibonacci
    sequence modulo `n`."""
    fact = fact or factorisation(n)
    ans = 1
    for p, e in fact.items():
        if p == 2:
            period = 3
        elif p == 5:
            period = 20
        elif p % 5 in (1, 4):
            period = p - 1
        else:
            period = 2 * (p + 1)
        ans = mathlcm(ans, period * p ** (e - 1))
    return ans


def primality_list(end: int) -> List[bool]:
    """Return a list of length `end`, the i-th element of which is True if i is
    prime, and False otherwise."""
    if end <= 1:
        return [False] * end

    ans = [True] * end
    ans[0] = ans[1] = False
    for i in range(2, isqrt(end) + 1):
        if ans[i]:
            for j in range(i**2, end, i):
                ans[j] = False
    return ans


def primes_list(end: int) -> List[int]:
    """Return a list of primes up to but not including `end`."""
    return list(compress(range(end), primality_list(end)))


def prime_count_list(end: int) -> List[int]:
    """Return a list of length `end`, the i-th element of which is the number of
    primes less than or equal to i."""
    return list(accumulate(map(int, primality_list(end))))


def prime_factor_list(end: int) -> List[int]:
    """Return a list of length `end`, the i-th element of which is the smallest
    prime factor of i."""
    ans = list(range(end))
    for i in range(2, isqrt(end) + 1):
        if ans[i] == i:
            for j in range(i, end, i):
                if ans[j] == j:
                    ans[j] = i
    return ans


def max_prime_factor_list(end: int) -> List[int]:
    """Return a list of length `end`, the i-th element of which is the largest
    prime factor of i."""
    ans = prime_factor_list(end)
    for i in range(2, end):
        p = ans[i]
        ans[i] = max(p, ans[i // p])
    return ans


_prime_factors = prime_factor_list(PRIME_FACTORS_END)


def omega_list(end: int, prime_factors: Optional[List[int]] = None) -> List[int]:
    """Return a list of length `end`, the i-th element of which is the omega of
    i. This is the number of distinct prime factors of i."""
    ans = _prepare_prime_factors(prime_factors, end)

    if end > 1:
        ans[1] = 0
    for i in range(2, end):
        p = ans[i]
        j = i // p
        ans[i] = ans[j] if j % p == 0 else 1 + ans[j]
    return ans


def ndiv_list(end: int, prime_factors: Optional[List[int]] = None) -> List[int]:
    """Return a list of length `end`, the i-th element of which is the number of
    divisors of i."""
    ans = _prepare_prime_factors(prime_factors, end)

    for i in range(2, end):
        p = ans[i]
        e = 1
        j = i // p
        while j % p == 0:
            j //= p
            e += 1
        ans[i] = (e + 1) * ans[j]
    return ans


def sigma_list(
    end: int, k: int = 1, prime_factors: Optional[List[int]] = None
) -> List[int]:
    """Return a list of length `end`, the i-th element of which is the sigma of
    i. This is the sum of the divisors of i, raised to the power `k`."""
    if k == 0:
        return ndiv_list(end, prime_factors)

    ans = _prepare_prime_factors(prime_factors, end)

    for i in range(2, end):
        p = ans[i]
        pk = p**k
        pek = pk
        j = i // p
        while j % p == 0:
            j //= p
            pek *= pk
        ans[i] = (pek * pk - 1) // (pk - 1) * ans[j]
    return ans


def totient_list(end: int, prime_factors: Optional[List[int]] = None) -> List[int]:
    """Return a list of length `end`, the i-th element of which is the Euler
    totient of i."""
    ans = _prepare_prime_factors(prime_factors, end)

    for i in range(2, end):
        p = ans[i]
        j = i // p
        ans[i] = ans[j] * p if j % p == 0 else ans[j] * (p - 1)
    return ans


def mobius_list(end: int, prime_factors: Optional[List[int]] = None) -> List[int]:
    """Return a list of length `end`, the i-th element of which is the value of
    the Mobius function of i."""
    ans = _prepare_prime_factors(prime_factors, end)

    for i in range(2, end):
        p = ans[i]
        j = i // p
        ans[i] = 0 if j % p == 0 else -ans[j]
    return ans


def rad_list(end: int, prime_factors: Optional[List[int]] = None) -> List[int]:
    """Return a list of length `end`, the i-th element of which is the radical
    of i."""
    ans = _prepare_prime_factors(prime_factors, end)

    for i in range(2, end):
        p = ans[i]
        j = i // p
        ans[i] = ans[j] if j % p == 0 else p * ans[j]
    return ans


def sum_squares_list(end: int, prime_factors: Optional[List[int]] = None) -> List[int]:
    """Return a list of length `end`, the i-th element of which is the number of
    ways i can be written as a sum of two squares."""
    ans = _prepare_prime_factors(prime_factors, end)

    if end > 1:
        ans[1] = 4
    for i in range(2, end):
        p = ans[i]
        j = i // p
        if p % 2 == 0:
            ans[i] = ans[j]
        else:
            e = 1
            while j % p == 0:
                j //= p
                e += 1
            if p % 4 == 1:
                ans[i] = (e + 1) * ans[j]
            elif p % 4 == 3:
                ans[i] = ans[j] if e % 2 == 0 else 0
            else:
                raise AssertionError
    return ans


def _prepare_prime_factors(prime_factors, end):
    if prime_factors:
        if len(prime_factors) < end:
            raise ValueError
        return prime_factors[:end]
    else:
        return prime_factor_list(end)


def primality_iter(segment: int = 100_000) -> Iterator[int]:
    """Return an iterator, the i-th element of which is True if i is prime, and
    False otherwise. Uses a segmented sieve algorithm with segment size equal to
    the `segment` parameter."""
    primes = primes_list(segment)

    primality = [False] * segment
    for p in primes:
        primality[p] = True
    yield from primality

    n = segment
    while n < segment**2:
        primality[:] = [True] * segment
        for p in primes:
            start_index = max(p * p - n, -n % p)
            if start_index >= segment:
                break
            for i in range(start_index, segment, p):
                primality[i] = False
        yield from primality
        n += segment


def primes_iter(segment: int = 100_000) -> Iterator[int]:
    """Generate primes. Uses a segmented sieve algorithm with segment size equal
    to the `segment` parameter."""
    yield from compress(count(0), primality_iter(segment))


def prime_count_iter(segment: int = 100_000) -> Iterator[int]:
    """Return an iterator, the i-th element of which is the number of primes
    less than or equal to i. Uses a segmented sieve algorithm with segment size
    equal to the `segment` parameter."""
    yield from accumulate(map(int, primality_iter(segment)))


class PrimeSieve:
    """A expanding prime sieve using Eratosthenes's algorithm."""

    _end: int
    _primes_list: List[int]

    __slots__ = ("_end", "_primes_list")

    EXPAND_FACTOR = 1.5

    def __init__(self, end: int = 1_000_000) -> None:
        self._end = end
        self._primes_list = primes_list(end)

    def prime(self, key: int | slice) -> int:
        if isinstance(key, int):
            if key < 0:
                raise IndexError("indices must be non-negative")
            self._extend_to_prime(key)
            return self._primes_list[key]

        else:
            raise TypeError("indices must be integers, not " + type(key).__name__)

    def __getitem__(self, key: int | slice) -> int:
        return self.prime(key)

    def _extend_to_n(self, n: int) -> None:
        if n < self._end:
            return

        n = max(n, int(self._end * self.EXPAND_FACTOR))

        while self._end**2 <= n:
            self._do_extend_to_n(self._end**2)
        self._do_extend_to_n(n)

    def _do_extend_to_n(self, n: int) -> None:
        ext_size = n + 1 - len(self._primes_list)
        primality = [True] * ext_size
        for p in self._primes_list:
            start_index = max(p * p - self._end, -self._end % p)
            if start_index >= ext_size:
                break
            for i in range(start_index, ext_size, p):
                primality[i] = False
        primes = list(compress(range(self._end, n + 1), primality))

        self._end = n + 1
        self._primes_list += primes

    def _extend_to_prime(self, i: int) -> None:
        while i >= len(self._primes_list):
            self._extend_to_n(int(self._end * self.EXPAND_FACTOR))

    def _extend(self) -> None:
        self._extend_to_n(self._end)

    def primes(self) -> Iterator[int]:
        yield from self._primes_list
        while True:
            start = len(self._primes_list)
            self._extend()
            yield from self._primes_list[start:]

    def __iter__(self) -> Iterator[int]:
        return self.primes()


def prime_count(n: int) -> int:
    """Return the number of primes less than or equal to `n`. The algorithm is
    simple variant of the Meissel–Lehmer algorithm."""
    if n == 0:
        return 0

    sqrtn = isqrt(n)
    ndsqrtn = n // sqrtn
    primes = primes_list(sqrtn + 1)

    small = [0] + [x - 1 for x in range(1, sqrtn + 1)]
    big = [0] + [n // d - 1 for d in range(1, ndsqrtn + 1)]

    for p in primes:
        k = small[p - 1]
        for d in range(1, ndsqrtn // p + 1):
            big[d] -= big[d * p] - k
        for d in range(ndsqrtn // p + 1, min(ndsqrtn, n // (p * p)) + 1):
            x = n // d
            big[d] -= small[x // p] - k
        for x in range(sqrtn, p * p - 1, -1):
            small[x] -= small[x // p] - k

    return big[1]


def prime_sum(n: int) -> int:
    """Return the sum of the primes less than or equal to `n`. The algorithm is
    simple variant of the Meissel–Lehmer algorithm."""
    if n == 0:
        return 0

    sqrtn = isqrt(n)
    ndsqrtn = n // sqrtn
    primes = primes_list(sqrtn + 1)

    small = [0] + [(x + 1) * (x + 2) // 2 - 1 for x in range(sqrtn + 1)]
    big = [0] + [(n // d) * (n // d + 1) // 2 - 1 for d in range(1, ndsqrtn + 1)]

    for p in primes:
        k = small[p - 1]
        for d in range(1, ndsqrtn // p + 1):
            big[d] -= p * (big[d * p] - k)
        for d in range(ndsqrtn // p + 1, min(ndsqrtn, n // (p * p)) + 1):
            x = n // d
            big[d] -= p * (small[x // p] - k)
        for x in range(sqrtn, p * p - 1, -1):
            small[x] -= p * (small[x // p] - k)

    return big[1]


def totient_sum(n: int) -> int:
    """Return the sum of the Euler totients of 1 to `n`."""
    if n <= 0:
        return 0

    sqrtn = isqrt(n)
    ndsqrtn = n // sqrtn

    small = [0] * (sqrtn + 1)
    big = [0] * (ndsqrtn + 1)

    for x in range(1, sqrtn + 1):
        sqrtx = isqrt(x)
        ans = x * (x + 1) // 2
        for i in range(2, sqrtx + 1):
            ans -= small[x // i]
        for i in range(1, min(sqrtx + 1, x // sqrtx)):
            ans -= small[i] * (x // i - x // (i + 1))
        small[x] = ans

    for d in range(ndsqrtn, 0, -1):
        x = n // d
        sqrtx = isqrt(x)
        ans = x * (x + 1) // 2
        for i in range(2, min(sqrtx, ndsqrtn // d) + 1):
            ans -= big[d * i]
        for i in range(ndsqrtn // d + 1, sqrtx + 1):
            ans -= small[x // i]
        for i in range(1, min(sqrtx + 1, x // sqrtx)):
            ans -= small[i] * (x // i - x // (i + 1))
        big[d] = ans

    return big[1]


def pythagorean_triples(bound: int, bound_perim: bool = False, primitive: bool = False):
    """Generate Pythagorean triples up to the specified (inclusive) bound."""

    if bound_perim:
        for n in range(1, isqrt(bound // 4) + 1):
            # if n is odd, m should be even, and vice versa
            for m in range(n + 1, (isqrt(n * n + 2 * bound) - n) // 2 + 1, 2):
                if gcd(m, n) == 1:
                    a = m * m - n * n
                    b = 2 * m * n
                    c = m * m + n * n
                    if primitive:
                        yield a, b, c
                    else:
                        for k in range(1, bound // (a + b + c) + 1):
                            yield k * a, k * b, k * c

    else:
        for n in range(1, isqrt(bound // 2) + 1):
            # if n is odd, m should be even, and vice versa
            for m in range(n + 1, isqrt(bound - n * n) + 1, 2):
                if gcd(m, n) == 1:
                    a = m * m - n * n
                    b = 2 * m * n
                    c = m * m + n * n
                    if primitive:
                        yield a, b, c
                    else:
                        for k in range(1, bound // c + 1):
                            yield k * a, k * b, k * c


def eisenstein120_triples(
    bound: int, bound_perim: bool = False, primitive: bool = False
):
    """Generate Eisentein triples with a 120 degree angle up to the specified
    (inclusive) bound. See: http://www.geocities.ws/fredlb37/node9.html"""

    if bound_perim:
        for n in range(1, isqrt(bound // 6) + 1):
            for m in range(n + 1, (isqrt(n * n + 8 * bound) - 3 * n) // 4 + 1):
                if (m - n) % 3 != 0 and gcd(m, n) == 1:
                    a = m * m - n * n
                    b = 2 * m * n + n * n
                    c = m * m + n * n + m * n
                    if primitive:
                        yield a, b, c
                    else:
                        for k in range(1, bound // (a + b + c) + 1):
                            yield k * a, k * b, k * c

    else:
        for n in range(1, isqrt(bound // 3) + 1):
            for m in range(n + 1, (isqrt(4 * bound - 3 * n * n) - n) // 2 + 1):
                if (m - n) % 3 != 0 and gcd(m, n) == 1:
                    a = m * m - n * n
                    b = 2 * m * n + n * n
                    c = m * m + n * n + m * n
                    if primitive:
                        yield a, b, c
                    else:
                        for k in range(1, bound // c + 1):
                            yield k * a, k * b, k * c


def eisenstein60_triples(
    bound: int, bound_perim: bool = False, primitive: bool = False
):
    """Generate Eisentein triples with a 60 degree angle up to the specified
    (inclusive) bound. See: http://www.geocities.ws/fredlb37/node9.html"""

    if bound_perim:
        if primitive:
            yield 1, 1, 1
        else:
            for k in range(1, bound // 3 + 1):
                yield k, k, k
        for a, b, c in eisenstein120_triples(bound, True, primitive):
            if a + 2 * b + c <= bound:
                yield a + b, b, c
            if 2 * a + b + c <= bound:
                yield a, a + b, c

    else:
        if primitive:
            yield 1, 1, 1
        else:
            for k in range(1, bound + 1):
                yield k, k, k
        for a, b, c in eisenstein120_triples(bound, False, primitive):
            yield a + b, b, c
            yield a, a + b, c
