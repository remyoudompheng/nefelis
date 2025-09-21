import argparse
import logging
import math
import time

import flint

from nefelis import integers
from nefelis import polys

logger = logging.getLogger("poly")


def polyselect(N, bound=None) -> tuple[list, list, int, list[list]]:
    """
    Select polynomials according to the conjugate method and return:
      f: a irreducible integer polynomial of degree 4
      g: a irreducible integer polynomial of degree 2
      D: a positive discriminant
      gj: a polynomial over Z[sqrt(D)] represented as a list of pairs
    """
    if bound is None:
        if N.bit_length() <= 150:
            bound = 2
        else:
            bound = 3

    # First, select a real quadratic field where N splits.
    # We also want ell to split (for large factors ell of (N-1))
    ells = [_l for _l, _ in integers.factor(N - 1) if _l.bit_length() > 64]
    logger.info(f"Looking for a real quadratic field with split {ells}")
    for D in integers.smallprimes(10000):
        if flint.fmpz(D).jacobi(N) != 1:
            continue
        if all(flint.fmpz(D).jacobi(ell) == 1 for ell in ells):
            break
    else:
        # Should never happen for reasonable size inputs
        raise ArithmeticError("N has a no square root for any small -p")

    # Iterate over small polynomials
    logger.info(f"Selecting quadratic field K=Q(sqrt(-{D}))")

    j = int(flint.fmpz_mod(D, flint.fmpz_mod_ctx(N)).sqrt())

    def small_gj(j: int, bound: int):
        """
        Let j = sqrt(-c)
        We choose g = x^2 + (a + bj) x + (c + dj)
        """
        for x0 in range(-bound, bound + 1):
            for x1 in range(-bound, bound + 1):
                # for x2 in range(-bound, bound+1):
                for x2 in [1]:
                    for y0 in range(-bound, bound + 1):
                        if x0**2 - D * y0**2 < 0:
                            # polynomial will have a real root
                            continue
                        for y1 in range(-bound, bound + 1):
                            # for y2 in range(-bound, bound+1):
                            for y2 in [0]:
                                A = x2 + y2 * j
                                B = x1 + y1 * j
                                C = x0 + y0 * j
                                disc = B * B - 4 * A * C
                                if flint.fmpz(disc).jacobi(N) == -1:
                                    yield (x0, x1, x2), (y0, y1, y2)

    # When reducing the matrix, only 2 rows are small.
    smallv = [(1, 0, 0), (0, 1, 0), (1, 1, 0), (1, -1, 0)]

    m = flint.fmpz_mat([[0, N], [1, int(j)]]).lll()
    jden, jnum = [int(_) for _ in m.table()[0]]
    jden2, jnum2 = [int(_) for _ in m.table()[1]]
    assert jnum * pow(jden, -1, N) % N == j
    assert (jnum**2 - D * jden**2) % N == 0

    best = 1e9
    counter = 0
    f, g, gj = None, None, None

    t0 = time.monotonic()
    for xs, ys in small_gj(j, bound):
        counter += 1
        # Scale x+yj to get O(sqrt(N)) coefficients (matrix will be almost reduced)
        zs = [xi * jden + yi * jnum for xi, yi in zip(xs, ys)]
        zs2 = [xi * jden2 + yi * jnum2 for xi, yi in zip(xs, ys)]
        mat = flint.fmpz_mat([zs, zs2, [N, 0, 0]]).lll()
        for v in smallv:
            g0, g1, g2 = (flint.fmpz_mat([v]) * mat).entries()
            g0, g1, g2 = int(g0), int(g1), int(g2)
            # We prefer negative discriminants
            Dg = g1**2 - 4 * g0 * g2
            if Dg >= 0:
                continue
            ag = polys.alpha2(Dg, g2, g1, g0)
            gsize = (3 * (g0 * g0 + g2 * g2) + 2 * g0 * g2 + g1 * g1) / 6
            gbits = math.log2(gsize) / 2
            if gbits < N.bit_length() / 3:
                # Most probably f is not irreducible
                continue
            score = gbits + ag
            if score < best:
                f1 = flint.fmpz_poly(list(xs))
                f2 = flint.fmpz_poly(list(ys))
                ff = f1**2 - D * f2**2
                if ff[4] < 0:
                    ff = -ff  # normalize sign
                roots_f = flint.fmpz_poly(ff).complex_roots()
                if any(_r.imag == 0 for _r, _ in roots_f):
                    logger.warning(f"Ignoring good polynomial {ff} with a real root")
                    continue
                logger.info(f"GOOD! {f} g {gbits:.2f} score {score:.2f}")
                best = score
                f = ff
                if g2 < 0:
                    g0, g1, g2 = -g0, -g1, -g2
                g = [g0, g1, g2]
                gj = list(zip(xs, ys))

    f = [int(fi) for fi in f]
    # Check that g divides f modulo N
    ZnX = flint.fmpz_mod_poly_ctx(N)
    print(f, g)
    fn = flint.fmpz_mod_poly(f, ZnX)
    gn = flint.fmpz_mod_poly(g, ZnX)
    assert len(gn.roots()) == 0
    assert fn % gn == 0

    dt = time.monotonic() - t0
    logging.info(f"Scanned {counter} polynomials of degree 4 in {dt:.3f}s")

    return f, g, D, gj


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
