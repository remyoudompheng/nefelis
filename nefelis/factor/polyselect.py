"""
Polynomial selection for factoring

There are some constraints on polynomials because
factors and norms of algebraic numbers must be known exactly.

The implementation is not optimized and simply selects a reasonably acceptable
polynomial.

The leading coefficient of f is:
    2x3x5x7 times a product of primes (size N/d)

The leading coefficient of g is a product of primes (size N/d)
"""

import bisect
import logging
import math
import random

import flint


# from nefelis.integers import smallprimes

logger = logging.getLogger("poly")


def polyselect(N: int, deg: int) -> tuple[list[int], list[int]]:
    def product(f):
        p = 1
        for fi in f:
            p *= fi
        return p

    def select(target, factors, nfacs):
        t = target ** (1 / nfacs)
        idx = bisect.bisect_left(factors, int(t))
        facs = random.sample(factors[idx - 20 : idx + 20], nfacs - 1)
        tgt = target / product(facs)
        f0 = min(factors, key=lambda f: abs(f - tgt))
        return product(facs) * f0

    # Select a random number x=u/v where u,v = O(N^(1/(deg+1)))
    # Use LLL to find a polynomial f such that f(x)=0
    # We want the leading coefficient of f to be a multiple of 210
    target = float(N) ** (1 / (deg + 1))
    logger.debug(f"target = {target} ({int(math.log2(target))} bits)")

    best = None
    bestnorm = 1e9

    smallvectors = [
        (1, 0, 0),
        (0, 1, 0),
        (0, 0, 1),
        (1, 1, 0),
        (1, -1, 0),
        (1, 0, 1),
        (1, 0, -1),
        (0, 1, 1),
        (0, 1, -1),
        (1, 1, 1),
        (1, 1, -1),
        (1, -1, 1),
        (1, -1, -1),
    ]
    smalls = [v + (0,) * (deg - 2) for v in smallvectors]

    for iter in range(1000):
        u = random.randint(int(target * 2), int(target * 3))
        v = random.randint(int(target * 2), int(target * 3))
        # print(u)
        # print(v)
        x = u * pow(v, -1, N)
        g = [-u, v]

        # ps = smallprimes(2 * int(target ** (1 / k)))
        # aa = select(target / 210, ps, k)
        # print(210 * aa, factor(210 * aa))

        M = flint.fmpz_mat(deg + 1, deg + 1)
        M[0, 0] = N
        for i in range(1, deg):
            M[i, i - 1] = x
            M[i, i] = -1
        M[deg, deg - 1] = 210 * x
        M[deg, deg] = -210
        ML = M.lll()
        for s in smalls:
            f = (flint.fmpz_mat([s]) * ML).entries()
            if any(x == 0 for x in f):
                continue
            norm = max(math.log2(abs(float(fi))) for fi in f)
            if norm < bestnorm:
                best, bestnorm = (f, g), norm
                logger.info(f"[iter {iter}] norm={norm:.1f} {f}")

    assert best is not None
    f, g = best
    f = [int(fi) for fi in f]
    return f, g


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.DEBUG)
    # A nice semiprime number
    N = 10**100 + 111
    deg = 5
    f, g = polyselect(N, deg)
    print(f"{f =}")
    print(f"{g =}")
    v, u = g
    resfg = sum(fi * u ** (deg - i) * (-v) ** i for i, fi in enumerate(f))
    assert resfg % N == 0
    print(resfg // N)
