from typing import Iterator
import math

import flint
import numpy as np

from .backends.kompute.sieve import Siever, LineSiever, LineSiever2
from .integers import smallprimes

DEBUG_ESTIMATOR = False


def factor_base(poly: list[int], B1: int) -> tuple[list, list]:
    if len(poly) == 2:
        # degree 1 polynomial
        v, u = poly
        ls = smallprimes(B1)
        rs = [(-v * pow(u, -1, l)) % l if u % l else l for l in ls]
        return ls, rs

    lc = poly[-1]
    ls, rs = [], []
    for l in smallprimes(B1):
        roots = flint.nmod_poly(poly, l).roots()
        for r, _ in roots:
            ls.append(l)
            rs.append(int(r))
        if lc % l == 0:
            ls.append(l)
            rs.append(l)
    return ls, rs


def gen_specialq(qmin: int, poly: list) -> Iterator[tuple[int, int]]:
    A = poly[-1]
    qlo = qmin
    while True:
        qs = [q for q in smallprimes(2 * qlo) if q >= qlo and A % q != 0]
        for q in qs:
            for r, _ in flint.nmod_poly(poly, q).roots():
                yield int(q), int(r)
        qlo *= 2


"""
Estimators for progress of the sieve.

The sieve can be described as the convergence of the set of seen primes
to the entire factor base: we expect it to follow an exponential law

Primes(t) = FB * (1 - exp(-φ(Relations(t))))

Numerical experiments from sieve logs show that φ is usually a power law
with exponent smaller than 1. Usually:

Primes(t) = FB * (1 - exp(-(Relations(t)/F)^α))

where F has the same order of magnitude as FB and α is an exponent between 0.5 and 0.9.

Usually each sieve side will have its own exponent α.
"""


def eta(
    Bf: int | float, Bg: int | float, target: float, stats_l: list[tuple[int, int, int]]
):
    """
    Given the 2 factor base bounds and historical statistics of the sieve
    progress, estimate the time when target excess is reached.
    """

    pif = Bf / math.log(Bf)
    pig = Bg / math.log(Bg)
    # Ignore initial values to stabilize fit
    # log(-log(1 - primes/fb)) = α log(Relations) + β
    stats = np.array(stats_l[len(stats_l) // 4 :], dtype=np.float32)
    if stats.shape[0] < 6:
        return None

    A = np.vstack([np.log(stats[:, 0]), np.ones(stats.shape[0])]).T
    Vf = -np.log(1.0 - np.minimum(stats[:, 1], pif - 1) / pif)
    Vg = -np.log(1.0 - np.minimum(stats[:, 2], pig - 1) / pig)
    af, bf = np.linalg.lstsq(A, np.log(Vf), rcond=0)[0]
    ag, bg = np.linalg.lstsq(A, np.log(Vg), rcond=0)[0]
    if not af or not ag:
        return None
    cf = math.exp(-bf / af)
    cg = math.exp(-bg / ag)

    if DEBUG_ESTIMATOR:
        Ef = pif * (1 - np.exp(-((stats[:, 0] / cf) ** af)))
        Eg = pig * (1 - np.exp(-((stats[:, 0] / cg) ** ag)))
        print("exponents", af, ag)
        print("constants", cf / pif, cg / pig)
        print(stats[:, 1], Ef)
        print(stats[:, 2], Eg)

    # Rescale to match last value
    r = float(stats[-1, 0])
    ef = pif * (1 - np.exp(-((r / cf) ** af)))
    eg = pig * (1 - np.exp(-((r / cg) ** ag)))
    scale_f = stats[-1, 1] / ef
    scale_g = stats[-1, 2] / eg

    # Find target (larger than current relation count)
    kmin, kmax = 1.0, 1e6
    while kmax / kmin > 1 + 1e-9:
        k = (kmin + kmax) / 2.0
        r = float(stats[-1, 0]) * k
        ef = pif * (1 - np.exp(-((r / cf) ** af))) * scale_f
        eg = pig * (1 - np.exp(-((r / cg) ** ag))) * scale_g
        if r > ef + eg + target:
            kmax = k
        else:
            kmin = k

    k = (kmin + kmax) / 2.0
    return k


__all__ = [Siever, LineSiever, LineSiever2, eta]


def benchmark():
    """
    Run sieves with a specific synthetic parameter set.
    """
    import logging
    import time

    from nefelis.integers import smallprimes
    from nefelis.vulkan import gpu_cores

    logging.getLogger().setLevel(level=logging.INFO)

    # We choose 2 linear polynomials for this benchmark (should not affect performance)
    u1, v1 = 2**256 + 1, 3**192 + 1
    u2, v2 = 5**128 + 1, 7**96 + 1
    cores = gpu_cores()

    primes = smallprimes(20_000_000)
    froots = [pow(u1, -1, l) * -v1 % l if u1 % l else l for l in primes]
    groots = [pow(u2, -1, l) * -v2 % l if u2 % l else l for l in primes]

    def runbench1(B1f: int, B1g: int, thr1: int, thr2: int, I: int):
        nonlocal u1, v1, u2, v2
        ls = []
        rs = []
        ls2 = []
        rs2 = []
        for p, r1, r2 in zip(primes, froots, groots):
            if p < B1f:
                ls.append(p)
                rs.append(r1)
            if p < B1g:
                ls2.append(p)
                rs2.append(r2)

        # prewarm GPU
        S = Siever([v1, u1], ls, rs, thr1, I, [v2, u2], ls2, rs2, thr2, outsize=2**24)
        idx1, idx2 = 10, 10 + cores
        for _ in range(2):
            for q, qr in zip(ls[idx1:idx2], rs[idx1:idx2]):
                xys = S.sieve(q, qr)

        t0 = time.monotonic()
        total = 0
        for q, qr in zip(ls[idx1:idx2], rs[idx1:idx2]):
            xys = S.sieve(q, qr)
            total += len(xys)
        dt = time.monotonic() - t0

        # FIXME: should be 2 ** (2 * I + 1)
        area = 2 ** (2 * I) * cores
        speed = area / dt
        logging.info(
            f"Siever      B1={B1f // 1000}k/{B1g // 1000}k thresholds {thr1}/{thr2} "
            f"shape {2 ** (I - 10)}k x {2 ** (I + 1)}: {speed * 1e-9:.2f}G/s, {total} results"
        )

    def runbench2(B1f: int, B1g: int, thr1: int, thr2: int, W: int, H: int):
        nonlocal u1, v1, u2, v2
        ls = []
        rs = []
        ls2 = []
        rs2 = []
        for p, r1, r2 in zip(primes, froots, groots):
            if p < B1f:
                ls.append(p)
                rs.append(r1)
            if p < B1g:
                ls2.append(p)
                rs2.append(r2)

        # prewarm GPU
        S = LineSiever2([v1, u1], [v2, u2], ls, rs, thr1, ls2, rs2, thr2, W, H, True)
        idx1, idx2 = 10, 10 + cores
        for _ in range(2):
            for q, qr in zip(ls[idx1:idx2], rs[idx1:idx2]):
                _, xys = S.sieve(q, qr)

        t0 = time.monotonic()
        total = 0
        for q, qr in zip(ls[idx1:idx2], rs[idx1:idx2]):
            _, xys = S.sieve(q, qr)
            total += len(xys)
        dt = time.monotonic() - t0

        area = 2 * W * H * cores
        speed = area / dt
        logging.info(
            f"LineSiever2 B1={B1f // 1000}k/{B1g // 1000}k thresholds {thr1}/{thr2} shape {2 * W >> 10}k x {H}: {speed * 1e-9:.2f}G/s, {total} results"
        )

    # Special shapes (deg3)
    runbench1(2000_000, 300_000, 120, 30, 14)
    runbench2(2000_000, 300_000, 120, 30, 16384, 16384)

    # Small shapes
    runbench1(500_000, 200_000, 60, 70, 14)
    runbench1(1000_000, 500_000, 80, 80, 14)
    runbench1(5000_000, 3000_000, 100, 90, 14)
    runbench1(15000_000, 10000_000, 120, 100, 14)

    # Small shapes
    runbench2(500_000, 200_000, 60, 70, 16384, 16384)
    runbench2(1000_000, 500_000, 80, 80, 16384, 16384)
    runbench2(5000_000, 3000_000, 100, 90, 16384, 16384)
    runbench2(15000_000, 10000_000, 120, 100, 16384, 16384)

    # Larger skew
    runbench2(500_000, 200_000, 60, 70, 160 << 10, 1638)
    runbench2(1000_000, 500_000, 80, 80, 160 << 10, 1638)
    runbench2(5000_000, 3000_000, 100, 90, 160 << 10, 1638)
    runbench2(15000_000, 10000_000, 120, 100, 160 << 10, 1638)

    # Extreme skew
    runbench2(500_000, 200_000, 60, 70, 1 << 20, 256)
    runbench2(1000_000, 500_000, 80, 80, 1 << 20, 256)
    runbench2(5000_000, 3000_000, 100, 90, 1 << 20, 256)
    runbench2(15000_000, 10000_000, 120, 100, 1 << 20, 256)


if __name__ == "__main__":
    benchmark()
