import argparse
import logging
import math
from multiprocessing import Pool

import flint

# fmt:off
SMALLPRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]
# fmt:on

SQUARES = set((l, i * i % l) for l in SMALLPRIMES for i in range(l))


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


class Polyselect:
    def __init__(self, N):
        self.best = 1e9
        self.N = N

    def process(self, D: int, a: int, b: int, c: int, rD: int):
        """
        Tries pairs of polynomials with f=ax²+bx+c.
        `rD` is a precomputed square root of D modulo N.
        """
        N = self.N
        alph = alpha(D, a, b, c)
        ainv = pow(2 * a, -1, N)
        rs = [(-b - rD) * ainv % N, (-b + rD) * ainv % N]
        f = [a, b, c]
        fsize = (3 * (a * a + c * c) + 2 * a * c + b * b) / 6
        fbits = math.log2(fsize) / 2 + alph

        g = None
        for r in rs:
            assert (a * r * r + b * r + c) % N == 0
            m = flint.fmpz_mat([[0, N], [-1, int(r)]]).lll()
            u1, v1, u2, v2 = m.entries()
            for u, v in [
                (u1, v1),
                (u2, v2),
                (u1 + u2, v1 + v2),
                (u1 - u2, v1 - v2),
            ]:
                # assert (u * r + v) % N == 0
                gsize = math.log2(u * u + v * v) / 2
                # Using typical parameters, the smoothness probability
                # is 2x less sensitive to the lognorm of f than to the lognorm of g.
                score = gsize + 0.5 * fbits
                if score < self.best:
                    self.best = score
                    logging.debug(
                        f"GOOD! {D=} {f} "
                        f"fsize={math.log2(fsize) / 2:.1f} a(f)={alph:.2f} normf={fbits:.2f} "
                        f"g {gsize:.2f} score {score:.2f}"
                    )
                    assert (a * v**2 - b * u * v + c * u**2) % N == 0
                    g = v, u

        if g is None:
            return None
        return (c, b, a), g, self.best


WORKER = None


def worker_init(N: int):
    global WORKER
    WORKER = Polyselect(N)


def worker_do(args):
    return WORKER.process(*args)


def polyselect(N: int, bound: int | None = None):
    """
    Select a good quadratic polynomial for discrete logarithm modulo N,
    with coefficient smaller than given bound.
    """
    if bound is None:
        # Empirical formula to have a small cost compared to sieve/linalg
        bound = int(3 * 1.6 ** (N.bit_length() / 40))

    def irreducibles():
        squarefree = set(range(-4 * bound * bound, 0))
        for q in range(2, 2 * bound):
            for qq in range(q * q, 4 * bound * bound, q * q):
                squarefree.discard(-qq)

        sqrtD = {}
        # Assume that a > 0, b >= 0, |c| <= a
        for a in range(1, bound):
            for b in range(0, bound):
                for c in range(-a, a + 1):
                    D = b * b - 4 * a * c
                    if D not in squarefree:
                        continue
                    if c == 0:
                        continue
                    if math.gcd(a, math.gcd(b, c)) != 1:
                        continue

                    if D not in sqrtD:
                        if flint.fmpz(D).jacobi(N) == 1:
                            sqrt = int(flint.fmpz_mod(D, flint.fmpz_mod_ctx(N)).sqrt())
                        else:
                            sqrt = None
                        sqrtD[D] = sqrt
                    rD = sqrtD[D]
                    if rD is None:
                        continue  # no root mod N

                    yield D, a, b, c, rD

    logging.info(f"Starting polynomial selection with degree 2 and bound {bound}")
    pool = Pool(initializer=worker_init, initargs=(N,))
    best = 1e9
    best_fg = None
    for item in pool.imap_unordered(worker_do, irreducibles(), chunksize=32):
        if item is None:
            continue
        f, g, score = item
        if score < best:
            logging.info(f"Found polynomials {f=} {g=} {score=:.2f}")
            best = score
            best_fg = f, g

    return best_fg


def main():
    # 280 bits => bound 100
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
