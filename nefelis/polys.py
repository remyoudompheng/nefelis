"""
Various helpers about polyomials (discriminants, "bad ideals", norms)
"""

from enum import Enum, auto
import math

import flint
import numpy

from nefelis.integers import factor_smooth, valuation


def discriminant(f: list[int]) -> int:
    """
    >>> discriminant([2, 3, 4])
    -23
    >>> discriminant([4, 3, 2])
    -23
    >>> discriminant([7, 1, 9, 4])
    -36979
    >>> discriminant([7, 1, 9, 4, 2])
    1097356
    """
    if len(f) == 3:
        return f[1] ** 2 - 4 * f[0] * f[2]
    elif len(f) == 4:
        d, c, b, a = f
        ac, bd, bb, cc, ad = a * c, b * d, b * b, c * c, a * d
        D = 18 * ac * bd + bb * cc - 4 * bb * bd - 4 * ac * cc - 27 * ad * ad
        return D
    elif len(f) == 5:
        e, d, c, b, a = f
        D0 = c * c - 3 * b * d + 12 * a * e
        D1 = 2 * c**3 - 9 * b * c * d + 27 * b * b * e + 27 * a * d * d - 72 * a * c * e
        return (4 * D0**3 - D1**2) // 27
    else:
        raise NotImplementedError


def l2norm(f: list[int]) -> float:
    # FIXME: explain
    if len(f) == 2:
        u, v = f
        return float(u * u + v * v)
    elif len(f) == 3:
        u, v, w = f
        return float(3 * (u * u + w * w) + 2 * u * w + v * v) / 6.0
    elif len(f) == 4:
        a, b, c, d = f
        return float(5 * (a * a + d * d) + 2 * (a * c + b * d) + b * b + c * c) / 8.0
    elif len(f) == 5:
        a0, a1, b, c1, c0 = f
        return float(
            35 * (a0**2 + c0**2)
            + 10 * b * (a0 + c0)
            + 5 * (a1**2 + c1**2)
            + 6 * (a0 * c0 + a1 * c1)
            + 3 * b**2
        )


class BadType(Enum):
    """
    A crude classification of singularities for 1-dimensional schemes.
    """

    # This type includes regular points (type ax+b=0 and x^2+p=0)
    # but also singular points which are always transverse to lines ax+b
    # such as: x^3+p=0, x^3+p^2=0
    # In this case, there is only 1 associated prime for a given root (p,r)
    REGULAR = auto()
    # A nodal singularity is a multiple root (multiplicity k)
    # which lifts to k distinct roots modulo p^2
    # For this type of singularity, the valuation of ax+b
    # will be > 1 for at most 1 prime.
    # There are k associated primes (p,r+ai*p) and each root lifts
    # to a p-adic root.
    NODAL = auto()
    # All other singularity types (unsupported)
    COMPLEX = auto()

    def __repr__(self):
        return f"BadType.{self.name}"


def bad_ideals(f: list[int]) -> list[tuple[int, int]]:
    """
    Check f (irreducible polynomial) for bad ideals.

    Currently we only suppport roots of multiplicity 2 with
    Eisenstein criterion, and polynomials of degree 2, 3, 4

    >>> bad_ideals([4, 1, -1, 2, 1])
    []
    >>> bad_ideals([96171004722765292, 395104756580329, 4808550236136482646])
    []
    >>> bad_ideals([2, -3, 2, 1])
    []
    >>> bad_ideals([4, 0, 0, 1])
    []
    >>> bad_ideals([4, 3, -4, 1])
    []
    >>> bad_ideals([30, 36, -19, 4])
    []

    Actual bad ideals:

    >>> bad_ideals([1289893690830, 616104301861, 721392287424])
    [(11, 7, BadType.NODAL)]

    (23, 2) is not a bad prime:

    >>> bad_ideals([294521483864, -44775242482, 175176273279])
    [(2, 0, BadType.NODAL), (5, 4, BadType.NODAL)]

    Newton polygon is not enough for these ones:

    >>> bad_ideals([4, 4, -3, 1])
    []
    >>> bad_ideals([4, 0, -1, 0, 1])  # doctest: +SKIP
    []

    A very ugly ramification 1+x^3+9x^4 at prime 3:

    >>> bad_ideals([1, 0, 0, 1, 9])
    [(3, 2, BadType.COMPLEX)]
    """
    assert 3 <= len(f) <= 5
    disc = discriminant(f)
    assert disc != 0
    # There may be large bad ideals but we only care about
    # those belonging to the factor base.
    facs = factor_smooth(disc, 20)
    bad: list[tuple[int, int]] = []
    poly = flint.fmpz_poly(f)
    for l, e in facs:
        if e < 2:
            continue
        # Compute roots
        fl = flint.nmod_poly(f, l)
        rs = fl.roots()
        for r, e in rs:
            if e < 2:
                continue
            # Now (l, r) is a multiple root of f.
            if e > 3:
                # raise NotImplementedError
                bad.append((l, int(r), BadType.COMPLEX))
                continue
            # If multiplicity is 2 or 3, (l, r) is a bad ideal
            # if and only if the polynomial has a l-adic root.
            # Compute the Newton polygon at (l, r). A necessary condition
            # to have a l-adic root is that the first Newton slope <= -1
            #
            # https://en.m.wikipedia.org/wiki/Newton_polygon
            # FIXME: check roots at infinity
            poly_r = poly(flint.fmpz_poly([int(r), 1]))
            # Since f is irreducible -r is not a root
            assert poly_r[0] != 0, (l, r)
            v0 = valuation(int(poly_r[0]), l)
            assert poly_r[e] % l != 0
            if all(valuation(int(poly_r[i]), l) > v0 - i for i in range(1, e + 1)):
                # First Newton slope > -1
                continue

            # If l is very small, manually check roots modulo small powers
            # we check at most ~100 values in a first pass.
            if l < 100:
                if l == 2:
                    lk = 64
                elif l <= 5:
                    lk = l**4
                else:
                    lk = l**2
                if all(poly(i) % lk != 0 for i in range(int(r), lk, l)):
                    # No root modulo l^k
                    continue
                # Check again with a higher power
                # This is uncommon so it's not costly.
                lk *= l
                if all(poly(i) % lk != 0 for i in range(int(r), lk, l)):
                    # No root modulo l^k
                    continue

            e2 = sum(1 for a in range(0, l * l, l) if poly_r(a) % (l ** (e + 1)) == 0)
            if e2 == e:
                # Nodal singularity
                bad.append((l, int(r), BadType.NODAL))
            else:
                # Not supported
                bad.append((l, int(r), BadType.COMPLEX))

    return bad


def alpha(D, f):
    if len(f) == 3:
        return alpha2(D, f[2], f[1], f[0])
    elif len(f) == 4:
        return alpha3(D, f[3], f[2], f[1], f[0])
    elif len(f) == 5:
        return alpha4(D, f)
    else:
        raise NotImplementedError


# fmt:off
SMALLPRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113]
# fmt:on

SQUARES = set((l, i * i % l) for l in SMALLPRIMES for i in range(l))

FIELDS = {
    l: numpy.array(
        [[1, x, x * x % l, x * x * x % l, x * x * x * x % l] for x in range(l)],
        dtype=numpy.int32,
    )
    for l in SMALLPRIMES
}


def alpha2(D, a, b, c):
    """
    Returns the α of polynomial ax²+bx+c (following Cado-NFS notations).
    However for convenience, we express it in base 2 to interpret it
    as a number of bits.

    α(x^2-x+1, 100)=1.389885781339105 (E.sage script in CADO-NFS)
    α(x^2-x+1, 115)=1.435115778287525
    >>> 1.435 / math.log(2) < alpha2(-3, 1, -1, 1) < 1.436 / math.log(2)
    True

    α(6*x^2-2*x+5, 100)=0.30843573575972727
    α(6*x^2-2*x+5, 115)=0.44366895401049494
    >>> 0.4436 / math.log(2) < alpha2(-116, 6, -2, 5) < 0.4437 / math.log(2)
    True
    """
    # for l in SMALLPRIMES:
    #     print(l, math.log(l) * (1 / (l - 1) - avgval2(D, a, b, c, l) * l / (l + 1)))
    return sum(
        math.log2(l) * (1 / (l - 1) - avgval2(D, a, b, c, l) * l / (l + 1))
        for l in SMALLPRIMES
    )


def avgval2(D, a, b, c, l):
    """
    Average (projective) valuation of homogeneous polynomial ax²+bxy+cy²
    """
    if D % l != 0:
        return nroots2(D, a, b, c, l) / (l - 1)
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


def nroots2(D, a, b, c, l):
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
    alpha(Zx([2,2,3,1]), 115) / log(2.0) = 1.90953799061677
    >>> 1.90 < alpha3(-104, 2, 2, 3, 1) < 1.94
    True
    """
    if any(abs(pi) > 2**16 for pi in [a, b, c, d]):
        poly = numpy.array([d, c, b, a], dtype=object)
    else:
        poly = numpy.array([d, c, b, a], dtype=numpy.int32)
    # for l in SMALLPRIMES:
    #     print(l, math.log(l) * (1 / (l - 1) - avgval3(D, a, b, c, d, l, poly) * l / (l + 1)))
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
        vals = fp[:, :4] @ poly
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
        vals = fp[:, :4] @ poly
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
    vals = fp[:, :4] @ poly
    return l - numpy.count_nonzero(vals % l)


# Quartic polynomials


def alpha4(D, f):
    """
    alpha(Zx([5,1,-4,2,3]), 115) / log(2.0) = 0.862355010358203
    >>> 0.86 < alpha4(132325, [5,1,-4,2,3]) < 0.89
    True

    alpha(f, 115) / log2 = -1.74153247231177
    >>> f = [14210657027941395, 89584037279263219, 45122821949983494, 99446805877972590, 33124700204200920]
    >>> -1.8 < alpha4(discriminant(f), f) < -1.4
    True
    """
    # for l in SMALLPRIMES:
    #    print(l, math.log(l) * (1 / (l - 1) - avgval4(D, l, f) * l / (l + 1)))
    return sum(
        math.log2(l) * (1 / (l - 1) - avgval4(D, l, f) * l / (l + 1))
        for l in SMALLPRIMES
    )


def avgval4(D, l, poly):
    poly = numpy.array(poly, dtype=object)
    if D % l != 0 and l > 2:
        return nroots4(poly, l) / (l - 1)

    # Discriminant is zero: polynomial is necessarily split.
    e, d, c, b, a = [int(ai) for ai in poly]
    if l <= 5:
        fp = FIELDS[l]
        vals = fp[:, :5] @ poly
        roots = [int(x) for x in (vals % l == 0).nonzero()[0]]
        val = len(roots) / l
        for k in range(1, 5):
            li = l**k
            roots_lift = []
            for r in roots:
                for x in range(r, r + l * li, li):
                    fx = (a * x * x + b * x + c) * x * x + d * x + e
                    if fx % (l * li) == 0:
                        roots_lift.append(r)
            count = len(roots_lift)
            if count == 0:
                break
            val += count / (l * li)
            roots = roots_lift
        # Roots at infinity
        if a % l == 0:
            val += 1 / l
            roots = [0]
            for k in range(1, 5):
                li = l**k
                roots_lift = []
                for r in roots:
                    for x in range(r, r + l * li, li):
                        fx = (e * x * x + d * x + c) * x * x + b * x + a
                        if fx % (l * li) == 0:
                            roots_lift.append(r)
                count = len(roots_lift)
                if count == 0:
                    break
                val += count / (l * li)
                roots = roots_lift
        return val
    else:
        fp = FIELDS[l]
        vals = fp[:, :5] @ poly
        roots = [int(x) for x in (vals % l == 0).nonzero()[0]]
        val = len(roots) / l
        count2 = 0
        for r in roots:
            for x in range(r, r + l * l, l):
                fx = (a * x * x + b * x + c) * x * x + d * x + e
                if fx % (l * l) == 0:
                    count2 += 1
        if a % l == 0:
            val += 1 / l
        return val + count2 / (l * (l - 1))


def nroots4(poly, l):
    fp = FIELDS[l]
    vals = fp[:, :5] @ poly
    infty = 1 if poly[-1] % l == 0 else 0
    return l - numpy.count_nonzero(vals % l) + infty
