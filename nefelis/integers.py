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
