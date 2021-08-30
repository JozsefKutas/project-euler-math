from collections import Counter
from itertools import product

import pytest

from project_euler_math.ntheory import (
    inrt, is_square, gcd, mathgcd, lcm, mathlcm, bezout, mod_inverse,
    crt2, crt2_noncoprime, farey_sequence, left_farey, right_farey,
    jacobi, tonelli_shanks, is_prime, divisors, factorisation,
    omega, omega_list, ndiv_list, ndiv, sigma, sigma_list,
    totient, mobius, rad, sum_squares,
    primes_list, primality_list,
    totient_list, mobius_list, rad_list, sum_squares_list,
    prime_count, prime_sum, totient_sum)


def test_inrt():
    for n in range(1, 100):
        for k in range(2, 10):
            assert inrt(n**k - 1, k) == n - 1
            assert inrt(n**k, k) == n
            assert inrt(n**k + 1, k) == n


def test_is_square():
    assert is_square(0)
    assert is_square(1)
    assert not is_square(2)
    assert not is_square(14_640)
    assert is_square(14_641)
    assert not is_square(14_642)


def test_gcd():
    for m, n in product(range(100), repeat=2):
        assert gcd(m, n) == mathgcd(m, n)


def test_lcm():
    for m, n in product(range(100), repeat=2):
        assert lcm(m, n) == mathlcm(m, n)


def test_bezout():
    for m, n in product(range(100), repeat=2):
        d, s, t = bezout(m, n)
        assert d == gcd(m, n) == s*m + t*n


def test_mod_inverse():
    for mod in range(2, 100):
        for a in range(100):
            if gcd(a, mod) == 1:
                assert (mod_inverse(a, mod) * a) % mod == 1
            else:
                assert mod_inverse(a, mod) is None


def test_crt2():
    for m, n in product(range(2, 50), repeat=2):
        if gcd(m, n) == 1:
            for a, b in product(range(m), range(n)):
                x = crt2(a, b, m, n)
                assert x < m*n and x % m == a and x % n == b


def test_crt2_noncoprime():
    for m, n in product(range(2, 50), repeat=2):
        d = gcd(m, n)
        for a, b in product(range(m), range(n)):
            x = crt2_noncoprime(a, b, m, n)
            if (b - a) % d == 0:
                assert x < m*n and x % m == a and x % n == b
            else:
                assert x is None


def test_farey_left():
    n = 100
    seq = farey_sequence(n)
    for i in range(1, len(seq)):
        assert left_farey(n, seq[i]) == seq[i-1]


def test_farey_right():
    n = 100
    seq = farey_sequence(n)
    for i in range(len(seq)-1):
        assert right_farey(n, seq[i]) == seq[i+1]


def test_jacobi():
    primes = primes_list(100)[1:]
    for p in primes:
        is_quad_res = [False] * p
        for i in range(1, (p-1)//2+1):
            is_quad_res[pow(i, 2, p)] = True
        assert jacobi(0, p) == 0
        for i in range(1, p):
            assert jacobi(i, p) == (1 if is_quad_res[i] else -1)


def test_tonelli_shanks():
    primes = primes_list(100)[1:]
    for p in primes:
        for a in range(100):
            b = tonelli_shanks(a*a, p)
            assert b == a % p or b == -a % p


def test_is_prime():
    assert not is_prime(0)
    assert not is_prime(1)
    assert is_prime(2)
    assert is_prime(3)
    assert not is_prime(4)
    assert is_prime(5)
    assert not is_prime(6)
    assert is_prime(103)
    assert not is_prime(105)
    assert is_prime(107)


def test_factorisation():
    with pytest.raises(ValueError):
        factorisation(0)
    assert factorisation(1) == Counter()
    assert factorisation(101) == Counter({101: 1})
    assert factorisation(2_310) == Counter(
        {2: 1, 3: 1, 5: 1, 7: 1, 11: 1})
    assert factorisation(65_536) == Counter({2: 16})
    assert factorisation(1_000_000) == Counter({2: 6, 5: 6})


def test_divisors():
    for i in range(1, 100):
        divs = [d for d in range(1, 100) if i % d == 0]
        assert sorted(divisors(i)) == divs


def test_ndiv():
    for i in range(1, 10_000):
        assert ndiv(i) == len(divisors(i))


def test_sigma():
    for k in range(10):
        for i in range(1, 10_000):
            assert sigma(i, k) == sum(d**k for d in divisors(i))


def test_primality_list():
    primality = primality_list(10_000)
    for i, is_prime_i in enumerate(primality):
        assert is_prime(i) == is_prime_i


def test_primes_list():
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
    assert primes_list(50) == primes


def test_omega_list():
    omegas = omega_list(10_000)
    for i in range(1, 10_000):
        assert omega(i) == omegas[i]


def test_ndiv_list():
    ndivs = ndiv_list(10_000)
    for i in range(1, 10_000):
        assert ndiv(i) == ndivs[i]


def test_sigma_list():
    for k in range(10):
        sigmas = sigma_list(10_000, k)
        for i in range(1, 10_000):
            assert sigma(i, k) == sigmas[i]


def test_totient_list():
    totients = totient_list(10_000)
    for i in range(1, 10_000):
        assert totient(i) == totients[i]


def test_mobius_list():
    mobiuses = mobius_list(10_000)
    for i in range(1, 10_000):
        assert mobius(i) == mobiuses[i]


def test_rad_list():
    rads = rad_list(10_000)
    for i in range(1, 10_000):
        assert rad(i) == rads[i]


def test_sum_squares_list():
    sum_squares_lis = sum_squares_list(10_000)
    for i in range(1, 10_000):
        assert sum_squares(i) == sum_squares_lis[i]


def test_prime_count():
    primality = primality_list(10_001)
    count = 0
    for i, b in enumerate(primality):
        count += 1 if b else 0
        assert prime_count(i) == count


def test_prime_sum():
    primality = primality_list(10_001)
    total = 0
    for i, b in enumerate(primality):
        total += i if b else 0
        assert prime_sum(i) == total


def test_totient_sum():
    totients = totient_list(10_001)
    total = 0
    for i, phi in enumerate(totients):
        total += phi
        assert totient_sum(i) == total
