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
import time
from multiprocessing import Pool, Value

import flint

from nefelis.polys import l2norm, alpha, discriminant, bad_ideals

# from nefelis.integers import smallprimes
from nefelis.integers import product

logger = logging.getLogger("poly")


def select(target, factors, nfacs):
    t = target ** (1 / nfacs)
    idx = bisect.bisect_left(factors, int(t))
    facs = random.sample(factors[idx - 20 : idx + 20], nfacs - 1)
    tgt = target / product(facs)
    f0 = min(factors, key=lambda f: abs(f - tgt))
    return product(facs) * f0


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


class Polyselect:
    def __init__(self, N: int, deg: int, target: int, best: Value):
        self.N = N
        self.deg = deg
        self.best: Value = best
        self.target = target
        self.smalls = [v + (0,) * (deg - 2) for v in smallvectors]

    def process(self):
        N, deg, target = self.N, self.deg, self.target
        smalls = self.smalls

        u = random.randint(int(target * 2), int(target * 3))
        v = random.randint(int(target * 2), int(target * 3))
        if (d := math.gcd(u, v)) != 1:
            u, v = u // d, v // d
        # print(u)
        # print(v)
        x = u * pow(v, -1, N)
        g = [-u, v]
        normg = math.log2(float(u) ** 2 + float(v) ** 2) / 2

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
        best, bestnorm = None, self.best.value
        for s in smalls:
            f = (flint.fmpz_mat([s]) * ML).entries()
            if any(x == 0 for x in f):
                continue
            resfg = sum(fi * v ** (deg - i) * u**i for i, fi in enumerate(f))
            assert resfg % N == 0
            if resfg == 0:
                continue
            norm = math.log2(l2norm(f)) / 2.0
            af = alpha(discriminant(f), f)
            score = norm + normg + af
            if score < self.best.value:
                fpoly = flint.fmpz_poly(f)
                fcont, ffacs = fpoly.factor()
                if abs(fcont) != 1:
                    continue
                if len(ffacs) != 1:
                    logger.warning(f"Skipping reducible polynomial f {f}")
                    continue

                if bads := bad_ideals([int(fi) for fi in f]):
                    logger.warning(
                        f"Skipping interesting polynomial f {f} with bad primes {bads}"
                    )
                    continue

                with self.best.get_lock():
                    self.best.value = min(self.best.value, score)

                if score < bestnorm:
                    best, bestnorm = (f, g), score
                    logger.info(
                        f"Found α(f)={af:.3f} normf={norm:.1f} normg={normg:.1f} f={f} g={g}"
                    )

        if best is None:
            return None
        return best, bestnorm


WORKER = None


def worker_init(N: int, deg: int, target: int, global_best: Value):
    global WORKER
    WORKER = Polyselect(N, deg, target, global_best)


def worker_do(args):
    return WORKER.process()


def polyselect(N: int, deg: int) -> tuple[list[int], list[int]]:
    # Select a random number x=u/v where u,v = O(N^(1/(deg+1)))
    # Use LLL to find a polynomial f such that f(x)=0
    # We want the leading coefficient of f to be a multiple of 210
    target = float(N) ** (1 / (deg + 1))
    logger.debug(f"target = {target} ({int(math.log2(target))} bits)")

    bestnorm = Value("d", 1e9)
    samples = max(100, int(2 ** (0.05 * N.bit_length())))

    t0 = time.monotonic()
    pool = Pool(initializer=worker_init, initargs=(N, deg, target, bestnorm))
    best_fg = None
    score_fg = 1e9
    for item in pool.imap_unordered(worker_do, range(samples), chunksize=32):
        if item is None:
            continue
        fg, score = item
        if score < score_fg:
            best_fg, score_fg = fg, score

    dt = time.monotonic() - t0
    logging.info(f"Tried {samples} polynomials of degree {deg} in {dt:.3f}s")

    assert best_fg is not None
    f, g = best_fg
    f = [int(fi) for fi in f]
    return f, g


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.DEBUG)
    # A nice semiprime number
    N = 10**100 + 111
    deg = 4
    f, g = polyselect(N, deg)
    print(f"{f =}")
    print(f"{g =}")
    v, u = g
    resfg = sum(fi * u ** (deg - i) * (-v) ** i for i, fi in enumerate(f))
    assert resfg % N == 0
    print(resfg // N)
