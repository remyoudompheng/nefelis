from typing import Iterator
import math

import flint
import numpy as np

from .backends.kompute.sieve import Siever, LineSiever, LineSiever2
from .integers import smallprimes

DEBUG_ESTIMATOR = False


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
    Vf = -np.log(1.0 - stats[:, 1] / pif)
    Vg = -np.log(1.0 - stats[:, 2] / pig)
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
