"""
Automatic selection of polynomial for Special NFS
"""

import logging
import math
import time

import flint
from polyselect import lemma21
from nefelis import polys

logger = logging.getLogger("poly")


def snfs_select(N, radius):
    best, best_score = None, 1e9
    bound = max(10, int(0.1 * 2 ** (N.bit_length() / 50)))
    t0 = time.monotonic()

    # If N is very close to an exact power b^n, also use powers of b
    base = None
    for n in range(3, 100):
        r = nth_root(N, n)
        if abs(N - r**n) < N**0.6:
            base = r
            logger.info(f"N is very close to {base}^{n}")

    def fractions():
        nonlocal N, base
        if base is not None:
            for k in range(1, 100):
                if base ** (2 * k) > N:
                    break
                yield 1, base**k
                yield base**k, 1
        for denom in range(1, bound):
            for numer in range(1, bound):
                yield numer, denom
        for denom in range(bound, bound * bound):
            yield 1, denom
        for numer in range(bound, bound * bound):
            yield numer, 1

    for degree in (2, 3, 4, 5):
        if N.bit_length() > 400 and degree == 2:
            continue
        for numer, denom in fractions():
            if math.gcd(numer, denom) != 1:
                continue
            # check if N = A/B * (b^n + small polynomial)
            r = nth_root((denom * N) // numer, degree)
            poly = lemma21(denom * N, 1, r, degree, numer)
            if all(abs(coef) < r**0.4 for coef in poly):
                assert sum(poly[i] * r**i for i in range(degree + 1)) == denom * N
                f = poly
                g = [-r, 1]
                norm = math.log2(math.sqrt(polys.l2norm(f)))
                normg = math.log2(math.sqrt(polys.l2norm(g)))
                alpha = polys.alpha(polys.discriminant(f), f)
                score = norm + normg + alpha + (degree + 1) * radius
                if score < best_score + 2.5:
                    logger.info(
                        f"{f=} {g=} size={norm:.3f}+{normg:.3f} α={alpha:.3f} score={score:.3f}"
                    )
                    special = True
                    for p in range(2**64 - 1, 2**64 - 1000, -2):
                        if not flint.fmpz(p).is_probable_prime():
                            continue
                        _, pfacs = flint.nmod_poly(f, p).factor()
                        if len(pfacs) == 1:
                            special = False
                    if special:
                        logger.error(
                            f"Polynomial {f} is reducible or has special Galois group"
                        )
                        continue
                if score < best_score:
                    best, best_score = (f, g), score

    dt = time.monotonic() - t0
    logger.info(f"SNFS polynomial selection done in {dt:.3f}s")
    return best


def nth_root(x, n):
    """
    Returns r such that r^n is closest to x.

    >>> nth_root(10**900 + 9 * 10**800, 9) == 10**100 + 1
    True
    """
    r = int(round(x ** (1.0 / n)))
    r2 = r
    if r > 2**50:
        # r is possibly not exact
        for _ in range(10):
            r2 = r + (x - r**n) // (n * r ** (n - 1))
            if abs(r2 - r) < 10:
                break
            r = r2

    while r**n < x and x > (r**n + (r + 1) ** n) // 2:
        r += 1
    while r**n > x and x < (r**n + (r - 1) ** n) // 2:
        r -= 1
    assert abs(r - r2) < 100  # should never fail
    return r


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    # snfs_select(10**126+361)
    # fibonacci(432) + 79
    # snfs_select(857384416798688205322146546398461722061337599142249183459012869301255270820177904036306063)
    snfs_select(2**599 - 1, 25)
