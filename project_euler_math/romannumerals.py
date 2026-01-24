_ROMAN2INT_LOOKUP = {
    "M": 1000,
    "CM": 900,
    "D": 500,
    "CD": 400,
    "C": 100,
    "XC": 90,
    "L": 50,
    "XL": 40,
    "X": 10,
    "IX": 9,
    "V": 5,
    "IV": 4,
    "I": 1,
}


def roman2int(roman: str) -> int:
    """Computes the integer value of a roman numeral string."""
    i = 0
    ans = 0
    for s, m in _ROMAN2INT_LOOKUP.items():
        k = len(s)
        while roman[i : i + k] == s:
            i += k
            ans += m
    return ans


def int2roman(n: int) -> str:
    """Returns the minimal roman numeral representation of a non-negative
    integer."""
    ans = []
    for s, m in _ROMAN2INT_LOOKUP.items():
        q, n = divmod(n, m)
        ans.append(q * s)
    return "".join(ans)
