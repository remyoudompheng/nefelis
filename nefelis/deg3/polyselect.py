import argparse
import logging
import math
import numpy
import time
from multiprocessing import Pool

import flint

logger = logging.getLogger("polyselect")

# fmt:off
SMALLPRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]
# fmt:on

SQUARES = set((l, i * i % l) for l in SMALLPRIMES for i in range(l))

FIELDS = {
    l: numpy.array(
        [[1, x, x * x % l, x * x * x % l] for x in range(l)], dtype=numpy.int32
    )
    for l in SMALLPRIMES
}


def alpha(D, a, b, c):
    """
    Returns the α of polynomial ax²+bx+c (following Cado-NFS notations).
    However for convenience, we express it in base 2 to interpret it
    as a number of bits.

    α(x^2-x+1)=1.389885781339105 (E.sage script in CADO-NFS)
    >>> 1.38988 / math.log(2) < alpha(-3, 1, -1, 1) < 1.38989 / math.log(2)
    True

    α(6*x^2-2*x+5)=0.30843573575972727
    >>> 0.30843 / math.log(2) < alpha(-116, 6, -2, 5) < 0.30844 / math.log(2)
    True
    """
    # for l in SMALLPRIMES:
    #    print(l, math.log(l) * (1 / (l - 1) - avgval(D, a, b, c, l) * l / (l + 1)))
    return sum(
        math.log2(l) * (1 / (l - 1) - avgval(D, a, b, c, l) * l / (l + 1))
        for l in SMALLPRIMES
    )


def avgval(D, a, b, c, l):
    """
    Average (projective) valuation of homogeneous polynomial ax²+bxy+cy²
    """
    if D % l != 0:
        return nroots(D, a, b, c, l) / (l - 1)
    # Discriminant is zero
    if l <= 5:
        if a % l == 0:
            a, c = c, a
        val = 1 / l
        roots = [x for x in range(l) if (a * x * x + b * x + c) % l == 0]
        for k in range(1, 5):
            li = l**k
            roots_lift = [
                x
                for r in roots
                for x in range(r, r + l * li, li)
                if (a * x * x + b * x + c) % (l * li) == 0
            ]
            count = len(roots_lift)
            if count == 0:
                break
            val += count / (l * li)
            roots = roots_lift
        return val
    else:
        if a % l == 0:
            a, c = c, a
        val = 1 / l
        root = next(x for x in range(l) if (a * x * x + b * x + c) % l == 0)
        count2 = sum(
            1
            for x in range(root, root + l * l, l)
            if (a * x * x + b * x + c) % (l * l) == 0
        )
        # FIXME
        return val + count2 / (l * (l - 1))


def nroots(D, a, b, c, l):
    if D % l == 0:
        return 1
    if l == 2:
        return 0 if a & b & c & 1 == 1 else 2
    if (l, D % l) in SQUARES:
        return 2
    else:
        return 0


# Cubic polynomials


def alpha3(D, a, b, c, d):
    """
    alpha(Zx([2,2,3,1]), 100) / log(2.0) = 2.09120380807321
    >>> 2.0 < alpha3(-104, 2, 2, 3, 1)) < 2.2
    True
    """
    poly = numpy.array([d, c, b, a], dtype=numpy.int32)
    return sum(
        math.log2(l) * (1 / (l - 1) - avgval3(D, a, b, c, d, l, poly) * l / (l + 1))
        for l in SMALLPRIMES
    )


def avgval3(D, a, b, c, d, l, poly):
    if D % l != 0 and l > 3:
        return nroots3(poly, l) / (l - 1)

    # Discriminant is zero: polynomial is necessarily split.
    if l <= 5:
        if a % l == 0:
            a, b, c, d = d, c, b, a
            poly = poly[::-1]
        fp = FIELDS[l]
        vals = fp @ poly
        roots = [int(x) for x in (vals % l == 0).nonzero()[0]]
        val = len(roots) / l
        for k in range(1, 5):
            li = l**k
            roots_lift = [
                x
                for r in roots
                for x in range(r, r + l * li, li)
                if (a * x * x * x + b * x * x + c * x + d) % (l * li) == 0
            ]
            count = len(roots_lift)
            if count == 0:
                break
            val += count / (l * li)
            roots = roots_lift
        return val
    else:
        if a % l == 0:
            a, b, c, d = d, c, b, a
            poly = poly[::-1]
        fp = FIELDS[l]
        vals = fp @ poly
        roots = [int(x) for x in (vals % l == 0).nonzero()[0]]
        val = len(roots) / l
        count2 = sum(
            1
            for root in roots
            for x in range(root, root + l * l, l)
            if ((a * x * x + b * x + c) * x + d) % (l * l) == 0
        )
        # FIXME
        return val + count2 / (l * (l - 1))


def nroots3(poly, l):
    fp = FIELDS[l]
    vals = fp @ poly
    return l - numpy.count_nonzero(vals % l)


smalls = [
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

    def process(self, D: int, a: int, b: int, c: int, d: int):
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
        af = alpha3(D, a, b, c, d)
        m = flint.fmpz_mat([[N, 0, 0], [int(r), -1, 0], [0, int(r), -1]]).lll()
        g = None
        for s in smalls:
            w, v, u = (flint.fmpz_mat([s]) * m).entries()
            w, v, u = int(w), int(v), int(u)
            if u.bit_length() + v.bit_length() + w.bit_length() < N.bit_length() // 2:
                # the polynomial was not irreducible over Q
                return None
            # We want to avoid real quadratic fields (positive discriminants)
            Dg = v * v - 4 * u * w
            if Dg >= 0:
                continue
            ag = alpha(Dg, u, v, w)
            gsize = float(3 * (u * u + w * w) + 2 * u * w + v * v) / 6.0
            gbits = math.log2(gsize) / 2
            # Using typical parameters, the smoothness probability
            # is ~30% less sensitive to the lognorm of f than to the lognorm of g.
            score = gbits + ag + 0.7 * (fbits + af)
            if score < self.best:
                self.best = score
                logging.debug(
                    f"GOOD! {D=} {f} "
                    f"fsize={math.log2(fsize) / 2:.1f} a(f)={af:.2f} normf={fbits:.2f} "
                    f"a(g)={ag:.2f} normg={gbits:.2f} score {score:.2f}"
                )
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

    def irreducibles():
        nonlocal counter
        # Assume that a > 0, 0 < d <= a
        for a in range(1, bound):
            #logging.debug(f"Trying a={a}")
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

                        counter += 1
                        yield D, a, b, c, d

    logging.info(f"Starting polynomial selection with degree 3 and bound {bound}")
    logging.info(f"Base score {N.bit_length() // 3}")

    pool = Pool(initializer=worker_init, initargs=(N, l))
    best = 1e9
    best_fg = None

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
