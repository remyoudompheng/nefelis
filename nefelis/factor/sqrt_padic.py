"""
Square root of algebraic numbers using the "direct" approach

This method computes the coefficients of the target algebraic
number and the coefficients of its square root using p-adic
approximation modulo powers of a suitably chosen prime p.

Suitable p are such that polynomial f is irreducible modulo p,
so that the square root in the residue field GF(p^d) is unique
up to sign.

Standard lifting modulo p^k is then used to obtain a square root
in the unramified extension Qp[x]/f(x), which coincides with
the actual square root in Q[x]/f(x) when known up to a sufficiently
large precision.
"""

import logging
import math
import time

import flint
from nefelis_flint_extras import fmpz_mod_ctx_composite

logger = logging.getLogger("sqrt")

DEBUG_CHECK_NEWTON = False


def sqrt(f: list[int], xys: list[tuple[int, int]], size_hint: int) -> list[int]:
    """
    Compute a square root of the product (xi + z*yi) in field Kf=Q[z]/f(z)

    The square root is computed as sum(ai * (Az)^i) / A^len(xys) where ai
    are large integers.
    """
    A = f[-1]
    degf = len(f) - 1
    fmonic = [A ** (degf - 1 - i) * f[i] for i in range(degf)] + [1]
    # 0. Select a word-sized inert prime and the desired p-adic precision
    # We should not restrict the residue class of p (some special polynomials
    # may never have inert primes in some residue classes).
    for p in range(2**64 - 5, 2**63, -2):
        if not flint.fmpz(p).is_probable_prime():
            continue
        _a, facs = flint.nmod_poly(f, p).factor()
        if len(facs) == 1 and facs[0][1] == 1:
            # f is irreducible mod p
            break
    assert flint.fmpz(p).is_probable_prime()
    N = int(math.ceil(size_hint / math.log2(p)))
    logger.debug(f"Using p-adic arithmetic with p={hex(p)} and precision O(p^{N})")

    pN = flint.fmpz(p) ** N
    ZpX = flint.fmpz_mod_poly_ctx(p)
    ZpNX = flint.fmpz_mod_poly_ctx(fmpz_mod_ctx_composite(pN))
    fpoly = flint.fmpz_poly(fmonic)
    fpN = ZpNX(fmonic)
    Fpd = flint.fq_default_ctx(p, degf, var="i", modulus=ZpX(fmonic))

    # 1. Compute the full product of (x + z y) as a polynomial S in Q[z]/f(z)
    # Actually we only need to know it modulo p^N for large enough N
    t0 = time.monotonic()

    def product_tree(items: list[flint.fmpz_poly]):
        if len(items) <= 16:
            acc = items[0]
            for it in items[1:]:
                acc *= it
            return acc
        else:
            res = (
                product_tree(items[: len(items) // 2])
                * product_tree(items[len(items) // 2 :])
                % fpoly
            )
            # Don't reduce eagerly modulo pN to avoid huge integers
            if abs(res[0]) > pN:
                res = flint.fmpz_poly(
                    [fi % pN if fi > 0 else -(abs(fi) % pN) for fi in res]
                )
            return res

    terms = []
    # Precompute product in Z[X] modulo f, it has huge coefficients
    for x, y in xys:
        terms.append(flint.fmpz_poly([A * x, y]))
    SN = ZpNX(list(product_tree(terms)))
    S = [flint.fmpz(int(si)) for si in SN]
    dt = time.monotonic() - t0
    logger.info(f"Computed S(z) = product(Ax+yz) in {dt:.3f}s")
    assert len(S) == degf

    # 2. Reduce to the residue field GF(p^d) and compute a square root Tp=sqrt(Sp)
    Sp = Fpd(S)
    Tp = Sp.sqrt()
    logger.debug(f"Initialized square root in GF(p^{degf}): {Tp}")

    # 3. Lift p-adically to obtain T=sqrt(S)
    t1 = time.monotonic()
    k = 1
    # Invariant:
    # Tk is the square root of S modulo p^k
    # Tinvk is the inverse of 2Tk
    # It is faster to lift the inverse modulo p than recompute XGCD of polynomials each time.
    Tk: list[flint.fmpz] = Tp.to_list()
    Tinvk: list[flint.fmpz] = (1 / (2 * Tp)).to_list()
    while k < N:
        k = min(2 * k, N)
        ZpkX = flint.fmpz_mod_poly_ctx(fmpz_mod_ctx_composite(flint.fmpz(p) ** k))
        # Newton iteration in Zp[x]/f(x):
        # Tinv => 2 * Tinv - 2T Tinv^2
        # T => T + (S-T²)/2T = T/2 + S/2T
        Sk = ZpkX(S)
        fk = ZpkX(fmonic)
        T = ZpkX(Tk)
        if DEBUG_CHECK_NEWTON:
            rem = list(Sk - T**2 % fk)
            for ri in rem:
                assert int(ri) % p ** (k // 2) == 0
        Tinv = ZpkX(Tinvk)
        Tinv = Tinv * (2 - 2 * T * Tinv) % fk
        Tadj = (Sk * Tinv - T / 2) % fk
        T += Tadj
        if DEBUG_CHECK_NEWTON:
            assert (2 * T * Tinv) % fk == 1
            assert T * T % fk == Sk
        Tk = [flint.fmpz(int(ti)) for ti in list(T)]
        # Adjust inverse to match new T
        # 1/(2T+2Tadj) = Tinv - 2Tadj/(2T)^2
        Tinv = (Tinv - 2 * Tadj * Tinv * Tinv) % fk
        # assert (2 * T * Tinv) % fk == 1
        Tinvk = [flint.fmpz(int(ti)) for ti in list(Tinv)]

    dt = time.monotonic() - t1
    logger.info(f"Computed T(z) = sqrt(S(z)) in {dt:.3f}s")

    T = ZpNX(Tk)
    assert T * T % fpN == SN
    pN = p**N
    result = []
    for ti in T:
        ti = int(ti)
        if ti > pN // 2:
            ti = ti - pN
        result.append(ti)
    return result
