"""
CPU implementation of lattice sieve using Numpy.

This implementation is meant to be a reference for testing
with a focus on readibility instead of optimization.
However it uses a few Numpy features to achieve decent speed.
"""

import math
import time

import numpy as np
import flint

DEBUG = False


def sieve(u, v, primes, roots, q, xmax, ymax, threshold):
    """
    Perform a lattice sieve for a linear polynomial ux+vy
    and a given special-q prime.

    For prime l=primes[i], the root r=roots[i] is such that
    ur+v=0 modulo l (and ux+vy=0 mod l is equivalent to x-ry=0)
    The convention r=l is used to denote infinity (when u%l=0)
    then ux+vy=0 mod l is equivalent to y=0 mod l).

    The sieve region is rectangle [-xmax, xmax] × [0, ymax]
    """
    if DEBUG:
        for l, r in zip(primes, roots):
            if r < l:
                assert (u * r + v) % l == 0, (l, r)
            else:
                assert u % l == 0

    # Compute basis for the q-lattice
    # In original lattice we want to sieve x/y=r mod l
    # In the sublattice with x=aX+bY y=cX+dY
    # the equation is X/Y = (dr-b)/(a-cr) mod l
    qr = (-v * pow(u, -1, q)) % q
    qred = flint.fmpz_mat([[q, 0], [qr, 1]]).lll()
    a, c, b, d = qred.entries()

    S = np.zeros((2 * xmax, ymax), dtype=np.uint8)
    # FIXME: support larger q
    for l, rorig in zip(primes, roots):
        rx, ry = (1, 0) if rorig == l else (rorig, 1)
        rnum = d * rx - b * ry
        rden = a * ry - c * rx
        if rden % l == 0:
            r = l
        else:
            r = rnum * pow(rden, -1, l) % l

        llog = (l - l // 3).bit_length()
        hits = 0
        if r == l:
            S[:, 0::l] += llog
            hits += S[:, 0::l].size
        elif l < xmax:
            for y in range(0, ymax):
                x = (xmax + r * y) % l
                if DEBUG:
                    assert (x - xmax - r * y) % l == 0
                    xx = a * (x - xmax) + b * y
                    yy = c * (x - xmax) + d * y
                    if rorig == l:
                        assert yy % l == 0
                    else:
                        assert (xx - rorig * yy) % l == 0
                    hits += S[x::l, y].size
                S[x::l, y] += llog
        else:
            lred = flint.fmpz_mat([[l, 0], [r, 1]]).lll()
            # Compute square such that lred * square contains the sieve region
            lx, ly = lred.table()
            linv = flint.fmpq_mat(lred).inv()
            xs = (
                -xmax * linv[0, 0],
                xmax * linv[0, 0],
                -xmax * linv[0, 0] + ymax * linv[1, 0],
                xmax * linv[0, 0] + ymax * linv[1, 0],
            )
            ys = (
                -xmax * linv[0, 1],
                xmax * linv[0, 1],
                -xmax * linv[0, 1] + ymax * linv[1, 1],
                xmax * linv[0, 1] + ymax * linv[1, 1],
            )
            amin, amax = min(xs).floor(), max(xs).ceil()
            bmin, bmax = min(ys).floor(), max(ys).ceil()
            for i in range(amin, amax + 1):
                for j in range(bmin, bmax + 1):
                    xij = i * lx[0] + j * ly[0]
                    yij = i * lx[1] + j * ly[1]
                    if -xmax <= xij < xmax and 0 <= yij < ymax:
                        if DEBUG:
                            assert (xij - r * yij) % l == 0
                            xx = a * xij + b * yij
                            yy = c * xij + d * yij
                            if rorig == l:
                                assert yy % l == 0
                            else:
                                assert (xx - rorig * yy) % l == 0
                            hits += 1
                        S[xmax + xij, yij] += llog
        if DEBUG:
            print(f"{l=} {hits=} expected~={S.size // l}")

    xs, ys = (S >= threshold).nonzero()
    reports = []
    a, c, b, d = qred.entries()
    for x, y in zip(xs, ys):
        x, y = int(x - xmax), int(y)
        reports.append((a * x + b * y, c * x + d * y))
    return reports


if __name__ == "__main__":
    from nefelis.integers import smallprimes

    p = 1000000000000000000000000000000000000000000000000000000000000000000000270907
    u = -2423783356717505418397281193017644615
    v = 33044377492552689242111796852708437154
    ls = smallprimes(300_000)
    rs = [(-v * pow(u, -1, l)) % l if u % l else l for l in ls]

    t0 = time.monotonic()
    WIDTH = 1024
    AREA = 2 * WIDTH**2
    reports = sieve(u, v, ls, rs, 1000003, WIDTH, WIDTH, 90)
    t = time.monotonic() - t0
    print(f"Sieved {AREA} in {t:.3f}s")

    for x, y in reports:
        if math.gcd(x, y) > 1:
            continue
        value = u * x + v * y
        print(f"{x}+{y}i", flint.fmpz(value).factor())
