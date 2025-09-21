import argparse
import logging
import math
import time
from multiprocessing import Pool

import flint

from nefelis import polys

logger = logging.getLogger("polyselect")


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
    def __init__(self, N, l):
        # Baseline score: we should always do better
        self.best = math.log2(N) / 3.0
        self.Zmod = flint.fmpz_mod_ctx(N)
        self.N = N
        self.l = l

    def process(self, D: int, a: int, b: int, c: int, d: int, global_best: float = 1e9):
        """
        Tries pairs of polynomials with f=ax³+bx²+cx+d.
        """
        N = self.N
        # We assume that D is not a square mod N and -3D is a square
        # Roots are computed by Cardano formula:
        #   r = -1/3a (b + C + Δ0/C)
        # where Δ1 = 2b³-9abc+27a²d
        #       C³ = (Δ1 ± 3a sqrt(-3Δ))/2
        #
        # See <https://en.wikipedia.org/wiki/Cubic_equation>
        rD = flint.fmpz_mod(-3 * D, self.Zmod).sqrt()
        d0 = b * b - 3 * a * c
        d1 = 2 * b**3 - 9 * a * b * c + 27 * a * a * d
        # C^3 = (Δ1 ± 3a sqrt(-3Δ))/2
        C3 = (d1 + 3 * a * rD) / 2
        if C3 == 0:
            # d1 == (d1 - 3 * a * rD) / 2
            C3 = flint.fmpz_mod(d1, self.Zmod)
        C = C3 ** ((2 * N - 1) // 3)
        # assert C**3 == C3 and C != 0
        r = (b + C + d0 / C) / (-3 * a)
        # print(a, b, c, d, r)
        assert a * r**3 + b * r**2 + c * r + d == 0, (a, b, c, d, int(r))

        f = [a, b, c, d]
        # Norm for degree 3 polynomials
        fsize = float(5 * (a * a + d * d) + 2 * (a * c + b * d) + b * b + c * c) / 8.0
        fbits = math.log2(fsize) / 2
        af = polys.alpha3(D, a, b, c, d)
        m = flint.fmpz_mat([[N, 0, 0], [int(r), -1, 0], [0, int(r), -1]]).lll()
        g = None
        for s in smallvectors:
            w, v, u = (flint.fmpz_mat([s]) * m).entries()
            w, v, u = int(w), int(v), int(u)
            if u.bit_length() + v.bit_length() + w.bit_length() < N.bit_length() // 2:
                # the polynomial was not irreducible over Q
                return None
            # We want to avoid real quadratic fields (positive discriminants)
            Dg = v * v - 4 * u * w
            if Dg >= 0:
                continue
            ag = polys.alpha2(Dg, u, v, w)
            gsize = float(3 * (u * u + w * w) + 2 * u * w + v * v) / 6.0
            gbits = math.log2(gsize) / 2
            # Using typical parameters, the smoothness probability
            # is ~30% less sensitive to the lognorm of f than to the lognorm of g.
            score = gbits + ag + 0.7 * (fbits + af)
            if score < self.best and score < global_best:
                logger.debug(
                    f"GOOD! {D=} {f} "
                    f"fsize={math.log2(fsize) / 2:.1f} a(f)={af:.2f} normf={fbits:.2f} "
                    f"a(g)={ag:.2f} normg={gbits:.2f} score {score:.2f}"
                )
                # FIXME: handle bad primes to avoid this
                if bads := polys.bad_ideals([d, c, b, a]):
                    logger.warning(
                        f"Skipping interesting polynomial f {f} with bad primes {bads}"
                    )
                    continue
                if badg := polys.bad_ideals([w, v, u]):
                    logger.warning(
                        f"Skipping interesting polynomial g {[w, v, u]} with bad primes {badg}"
                    )
                    continue
                self.best = score
                g = [u, v, w]
                # Check number of real roots
                roots_f = flint.fmpz_poly(f).complex_roots()
                assert sum(1 for r, _ in roots_f if r.imag == 0) == 1
                roots_g = flint.fmpz_poly(g).complex_roots()
                assert sum(1 for r, _ in roots_g if r.imag == 0) == 0
                assert u * r * r + v * r + w == 0

        if g is None:
            return None
        return f, g, self.best


WORKER = None


def worker_init(N: int, l: int):
    global WORKER
    WORKER = Polyselect(N, l)


def worker_do(args):
    return WORKER.process(*args)


def polyselect(N: int, bound: int | None = None):
    """
    Select a good cubic polynomial for discrete logarithm modulo N,
    with coefficient smaller than given bound.
    """
    if bound is None:
        # Empirical formula to have a small cost compared to sieve/linalg
        bound = max(3, int(2.5 * 2 ** (0.01 * N.bit_length())))

    assert N % 3 == 2

    l = N // 2
    counter = 0
    best = 1e9
    best_fg = None

    def irreducibles():
        nonlocal counter
        # Assume that a > 0, 0 < d <= a
        for a in range(1, bound):
            # logging.debug(f"Trying a={a}")
            for b in range(-bound, bound):
                for c in range(-bound, bound):
                    for d in range(1, a + 1):
                        if (a, b) < (d, c):
                            # Only consider poly >= poly.reverse() to account for symmetry
                            continue
                        ac, bd, bb, cc, ad = a * c, b * d, b * b, c * c, a * d
                        D = (
                            18 * ac * bd
                            + bb * cc
                            - 4 * bb * bd
                            - 4 * ac * cc
                            - 27 * ad * ad
                        )
                        # We want only 1 real root to keep the group of units small
                        if D >= 0:
                            continue
                        # We want a root modulo l, the easiest way is to
                        # have jacobi(D, l)==-1
                        if flint.fmpz(D).jacobi(l) != -1:
                            continue
                        # We want a root modulo N, they are easier to compute
                        # if D is not a square
                        if flint.fmpz(D).jacobi(N) != -1:
                            continue
                        # Polynomial must have content=1
                        if math.gcd(math.gcd(a, b), math.gcd(c, d)) != 1:
                            continue

                        counter += 1
                        yield D, a, b, c, d, best

    logging.info(f"Starting polynomial selection with degree 3 and bound {bound}")
    logging.info(f"Base score {N.bit_length() // 3}")

    pool = Pool(initializer=worker_init, initargs=(N, l))

    t0 = time.monotonic()
    for item in pool.imap_unordered(worker_do, irreducibles(), chunksize=32):
        if item is None:
            continue
        f, g, score = item
        if score < best:
            logging.info(f"Found polynomials {f=} {g=} {score=:.2f}")
            best = score
            best_fg = f, g

    dt = time.monotonic() - t0
    logging.info(f"Scanned {counter} polynomials of degree 3 in {dt:.3f}s")
    return best_fg


def polyselect_g(N: int, f: list[int], r: int) -> list[int] | None:
    """
    Select a quadratic polynomial for a fixed polynomial f.
    """
    m = flint.fmpz_mat([[N, 0, 0], [int(r), -1, 0], [0, int(r), -1]]).lll()
    g = None
    best = 1e9
    for s in smallvectors:
        w, v, u = (flint.fmpz_mat([s]) * m).entries()
        w, v, u = int(w), int(v), int(u)
        if u.bit_length() + v.bit_length() + w.bit_length() < N.bit_length() // 2:
            # the polynomial was not irreducible over Q
            return None
        # We want to avoid real quadratic fields (positive discriminants)
        Dg = v * v - 4 * u * w
        if Dg >= 0:
            continue
        ag = polys.alpha2(Dg, u, v, w)
        gsize = float(3 * (u * u + w * w) + 2 * u * w + v * v) / 6.0
        gbits = math.log2(gsize) / 2
        score = gbits + ag
        if score < best:
            logger.debug(f"a(g)={ag:.2f} normg={gbits:.2f} score {score:.2f}")
            # FIXME: handle bad primes to avoid this
            if badg := polys.bad_ideals([w, v, u]):
                logger.warning(
                    f"Skipping interesting polynomial g {[w, v, u]} with bad primes {badg}"
                )
                continue
            best = score
            g = [int(w), int(v), int(u)]
            assert (u * r * r + v * r + w) % N == 0
    if g is None:
        raise ValueError("failed to find a g polynomial")
    return g


def main():
    argp = argparse.ArgumentParser()
    argp.add_argument("N", type=int)
    argp.add_argument("bound", nargs="?", type=int)
    args = argp.parse_args()

    f, g = polyselect(args.N, args.bound)
    print("f", f)
    print("g", g)


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)
    main()
