"""
Utility functions for integers.
"""

import math
import numpy as np

import flint

try:
    import pymqs
except ImportError:
    pymqs = None


def smallprimes(B: int) -> list[int]:
    l = np.ones(B, dtype=np.uint8)
    l[0:2] = 0
    for i in range(math.isqrt(B) + 1):
        if l[i] == 0:
            continue
        l[i * i :: i] = 0
    return [int(_i) for _i in l.nonzero()[0]]


def factor(n: int | flint.fmpz) -> list[tuple[int, int]]:
    if pymqs is not None:
        facs = pymqs.factor(abs(int(n)))
        facd = {}
        for f in facs:
            facd[f] = facd.get(f, 0) + 1
        return sorted(facd.items())
    else:
        return [(int(l), int(e)) for l, e in flint.fmpz(n).factor()]


def product(zs: list[int]) -> int:
    if len(zs) == 0:
        return 1
    elif len(zs) == 1:
        return zs[0]
    else:
        return product(zs[: len(zs) // 2]) * product(zs[len(zs) // 2 :])


def factor_smooth(n: int | flint.fmpz, bits: int) -> list[tuple[int, int]]:
    """
    Compute a partial factorization to obtain factors under bit length.
    """
    if pymqs is not None and n.bit_length() < 512:
        facs = pymqs.factor_smooth(abs(int(n)), bits)
        # assert product(facs) == abs(int(n)), n
        facd = {}
        for f in facs:
            facd[f] = facd.get(f, 0) + 1
        return sorted(facd.items())
    else:
        return [(int(l), int(e)) for l, e in flint.fmpz(n).factor_smooth(bits)]


def valuation(x: int, p: int) -> int:
    if x == 0:
        # An approximation of infinity.
        return 0xFFFFFFFF
    v = 0
    while x % p == 0:
        v += 1
        x = x // p
    return v
