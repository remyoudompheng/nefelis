import argparse
import logging
import math
from multiprocessing import Pool

import flint
from nefelis import polys

logger = logging.getLogger("poly")


class Polyselect:
    def __init__(self, N):
        self.best = 1e9
        self.N = N

    def process(self, D: int, a: int, b: int, c: int, rD: int, global_best: float):
        """
        Tries pairs of polynomials with f=ax²+bx+c.
        `rD` is a precomputed square root of D modulo N.
        """
        self.best = min(self.best, global_best)

        N = self.N
        alph = polys.alpha2(D, a, b, c)
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
                    logger.debug(
                        f"GOOD! {D=} {f} "
                        f"fsize={math.log2(fsize) / 2:.1f} a(f)={alph:.2f} normf={fbits:.2f} "
                        f"g {gsize:.2f} score {score:.2f}"
                    )
                    if bads := polys.bad_ideals(f):
                        logger.warning(
                            f"Skipping interesting polynomial f {f} with bad primes {bads}"
                        )
                        continue
                    assert (a * v**2 - b * u * v + c * u**2) % N == 0
                    g = int(v), int(u)

        if g is None:
            return None
        return (c, b, a), g, self.best


WORKER = None


def worker_init(N: int):
    global WORKER
    WORKER = Polyselect(N)


def worker_do(args):
    return WORKER.process(*args)


def polyselect(
    N: int, bound: int | None = None
) -> tuple[tuple[int, int, int], tuple[int, int]]:
    """
    Select a good quadratic polynomial for discrete logarithm modulo N,
    with coefficient smaller than given bound.
    """
    if bound is None:
        # Empirical formula to have a small cost compared to sieve/linalg
        bound = int(3 * 1.6 ** (N.bit_length() / 40))

    best = 1e9

    def irreducibles():
        squares = frozenset(i * i for i in range(1, 4 * bound))

        sqrtD = {}
        # Assume that a > 0, b >= 0, |c| <= a
        for a in range(1, bound):
            for b in range(0, bound):
                for c in range(-a, a + 1):
                    D = b * b - 4 * a * c
                    if D in squares:
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

                    yield D, a, b, c, rD, best

    logger.info(f"Starting polynomial selection with degree 2 and bound {bound}")
    pool = Pool(initializer=worker_init, initargs=(N,))
    best_fg = None
    for item in pool.imap_unordered(worker_do, irreducibles(), chunksize=32):
        if item is None:
            continue
        f, g, score = item
        if score < best:
            logger.info(f"Found polynomials {f=} {g=} {score=:.2f}")
            best = score
            best_fg = f, g

    if best_fg is None:
        raise ValueError("Internal error: unable to select any polynomial")

    return best_fg


def main():
    import nefelis.logging

    # 280 bits => bound 100
    argp = argparse.ArgumentParser()
    argp.add_argument("-v", action="store_true")
    argp.add_argument("N", type=int)
    argp.add_argument("bound", nargs="?", type=int)
    args = argp.parse_args()

    nefelis.logging.setup(logging.DEBUG if args.v else logging.INFO)

    f, g = polyselect(args.N, args.bound)
    print("f", f)
    print("g", g)


if __name__ == "__main__":
    main()
