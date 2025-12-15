import argparse
import logging
import math
import time

import flint

from nefelis.cadocompat import poly_str
from nefelis import integers
from nefelis import polys

logger = logging.getLogger("poly")

# FIXME: make it parallel


def polyselect(N, bound=None) -> tuple[list, list, int, list[list]]:
    """
    Select polynomials according to the conjugate method and return:
      f: a irreducible integer polynomial of degree 4
      g: a irreducible integer polynomial of degree 2
      D: a positive discriminant
      gj: a polynomial over Z[sqrt(D)] represented as a list of pairs
    """
    if bound is None:
        bound = 5 + int((N.bit_length() / 50) ** 2)

    # First, select a real quadratic field where N splits.
    for D in integers.smallprimes(10000):
        if flint.fmpz(D).jacobi(N) == 1:
            break
    else:
        # Should never happen for reasonable size inputs
        raise ArithmeticError("N has a no square root for any small -p")

    # Iterate over small polynomials
    logger.info(f"Selecting quadratic field K=Q(sqrt({D})) and bound {bound}")

    j = int(flint.fmpz_mod(D, flint.fmpz_mod_ctx(N)).sqrt())

    def small_gj(j: int, bound: int):
        """
        Let j = sqrt(-c)
        We choose g = x^2 + (a + bj) x + (c + dj)
        """
        sqrtD = math.sqrt(D)
        for y0 in range(-bound, bound + 1):
            # We want x0 ± sqrt(D) y0 >= 0 to avoid real roots.
            xmin = int(math.ceil(abs(y0) * sqrtD))
            for x0 in range(xmin, max(xmin, bound) + 1):
                for x1 in range(-bound, bound + 1):
                    # for x2 in range(-bound, bound+1):
                    for x2 in [1]:
                        for y1 in range(-bound, bound + 1):
                            # FIXME: when y1==0, f = gx^2 - D l^2 behaves badly modulo l, why?
                            if y1 == 0:
                                continue
                            # for y2 in range(-bound, bound+1):
                            for y2 in [0]:
                                A = x2 + y2 * j
                                B = x1 + y1 * j
                                C = x0 + y0 * j
                                disc = B * B - 4 * A * C
                                if flint.fmpz(disc).jacobi(N) == -1:
                                    yield (x0, x1, x2), (y0, y1, y2)

    # When reducing the matrix, only 2 rows are small.
    smallv = [(1, 0), (0, 1), (1, 1), (1, -1), (2, 1), (2, -1), (1, 2), (1, -2)]

    best = 1e9
    counter = 0
    f, g, gj = None, None, None

    t0 = time.monotonic()
    for xs, ys in small_gj(j, bound):
        counter += 1

        f1 = flint.fmpz_poly(list(xs))
        f2 = flint.fmpz_poly(list(ys))
        ff = f1**2 - D * f2**2
        if ff[4] < 0:
            ff = -ff  # normalize sign
        roots_f = flint.fmpz_poly(ff).complex_roots()
        if any(_r.imag == 0 for _r, _ in roots_f):
            # logger.warning(f"Ignoring good polynomial {ff} with a real root")
            continue
        ff_l = [int(fi) for fi in list(ff)]
        Df = polys.discriminant(ff_l)
        if Df == 0:
            continue
        if bads := polys.bad_ideals(ff_l):
            # logger.warning(
            #    f"Skipping interesting polynomial {ff} with bad primes {bads}"
            # )
            continue
        fsize = polys.l2norm(ff_l)
        af = polys.alpha4(Df, ff_l)
        fbits = math.log2(fsize) / 2

        # Scale x+yj to get O(sqrt(N)) coefficients (matrix will be almost reduced)
        zs = [xi + yi * j for xi, yi in zip(xs, ys)]
        mat = flint.fmpz_mat([zs, [N, 0, 0], [0, N, 0], [0, 0, N]]).lll()
        v0, v1 = mat.table()[:2]
        for v in smallv:
            g0, g1, g2 = [int(v[0] * x0 + v[1] * x1) for x0, x1 in zip(v0, v1)]
            # We prefer negative discriminants
            Dg = g1**2 - 4 * g0 * g2
            if Dg >= 0:
                continue
            gsize = polys.l2norm([g0, g1, g2])
            gbits = math.log2(gsize) / 2
            if gbits < N.bit_length() / 3:
                # Most probably f is not irreducible
                continue
            if bads := polys.bad_ideals([g0, g1, g2]):
                continue
            ag = polys.alpha2(Dg, g2, g1, g0)
            score = gbits + ag + fbits + af
            if score < best:
                logger.info(
                    f"GOOD! ({poly_str(xs)})²{-D:+}({poly_str(ys)})²={ff} |f|={fbits:.2f} α(f)={af:.2f} |g|={gbits:.2f} α(g)={ag:.2f} score {score:.2f}"
                )
                best = score
                f = ff_l
                if g2 < 0:
                    g0, g1, g2 = -g0, -g1, -g2
                g = [g0, g1, g2]
                gj = list(zip(xs, ys))

    # Check that g divides f modulo N
    ZnX = flint.fmpz_mod_poly_ctx(N)
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

    f, g, D, gj = polyselect(args.N, args.bound)
    print(f"{D = }")
    print(f"f = {poly_str(f)}")
    print(f"g = {poly_str(g)}")
    gx = [x for x, y in gj]
    gy = [y for x, y in gj]
    print(f"gj = {poly_str(gx)} + sqrt(D) ({poly_str(gy)})")


if __name__ == "__main__":
    import nefelis.logging

    nefelis.logging.setup(logging.DEBUG)
    main()
