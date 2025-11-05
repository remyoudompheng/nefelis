"""
Square root of algebraic numbers using real/complex embeddings

This is used to estimate the size of square root coefficients,
and to provide a toy implementation with suboptimal (quadratic) performance.
"""

import logging

import flint


logger = logging.getLogger("sqrt")


def sqrt(f: list[int], xys: list[tuple[int, int]], approx=64) -> list[flint.arb]:
    """
    Compute a square root of the product (xi + z*yi) in field Kf=Q[z]/f(z)

    The square root is computed as sum(ai * (Az)^i) / A^len(xys) where ai
    are real numbers with specified precision.
    """
    # Note that the root of z is not an algebraic integer.
    # Let A be the leading coefficient of f
    # Then Az is a root of the monic polynomial A^(d-1) f(z/A)
    # Also Z[Az] is some order of field Kf
    #
    # If f is chosen to have no bad primes, we know that Z[Az] is
    # a Dedekind ring, and Z[Az] is actually a maximal order.
    with flint.ctx.workprec(approx):
        return sqrt_impl(f, xys)


def sqrt_impl(f: list[int], xys: list[tuple[int, int]]) -> list[flint.arb]:
    A = f[-1]
    degf = len(f) - 1
    fmonic = [A ** (degf - 1 - i) * f[i] for i in range(degf)] + [1]
    poly = flint.fmpz_poly(fmonic)
    roots = poly.complex_roots()
    rroots = []
    croots = []
    for r, _ in roots:
        if r.imag == 0:
            rroots.append(r)
        elif croots and croots[-1].conjugate().contains(r):
            continue
        else:
            croots.append(r)
    # print("real roots", rroots)
    # print("complex roots", croots)

    fr = []
    fc = []
    for r in rroots:
        # A * (x + y z) = Ax + y * (Az)
        prod = flint.arb(1)
        for x, y in xys:
            prod *= A * x + y * r.real
        fr.append(prod)
    for r in croots:
        prod = flint.acb(1)
        for x, y in xys:
            prod *= A * x + y * r
        fc.append(prod)

    for i in range(len(fr)):
        assert fr[i].lower() > 0
        fr[i] = fr[i].sqrt()
    for i in range(len(fc)):
        fc[i] = fc[i].sqrt()

    eval_roots = rroots
    for r in croots:
        eval_roots += [r, r.conjugate()]
    for rsign, csign in signs(fr, fc):
        eval_values = list(rsign)
        for c in csign:
            eval_values += [c, c.conjugate()]
        res = flint.acb_poly.interpolate(
            eval_roots,
            eval_values,
            "newton",
        )
        # print("SIGN")

        if all(r.contains_integer() and r.rad() < 0.5 for r in res):
            # print("FOUND INTEGER!!")
            # print(res[0].real.unique_fmpz())
            for i, r in enumerate(res):
                logger.debug(f"T[{i}] = {r.real.str(condense=10)}")
            return [r.real for r in res]
    # Return some polynomial
    return [r.real for r in res]


def signs(reals, cplxs):
    """
    Iterate over sign combinations of square roots
    """
    k = len(reals) + len(cplxs) - 1
    for sgn in range(2**k):
        if reals:
            rsigns = [reals[0]]
            csigns = []
            for i in range(k):
                if i < len(reals) - 1:
                    if sgn & (1 << i):
                        rsigns.append(-reals[i + 1])
                    else:
                        rsigns.append(reals[i + 1])
                else:
                    j = i - len(reals) + 1
                    if sgn & (1 << i):
                        csigns.append(-cplxs[j])
                    else:
                        csigns.append(cplxs[j])
        else:
            rsigns = []
            csigns = [cplxs[0]]
            for i in range(k):
                if sgn & (1 << i):
                    csigns.append(-cplxs[i + 1])
                else:
                    csigns.append(cplxs[i + 1])
        yield rsigns, csigns
