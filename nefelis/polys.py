"""
Various helpers about polyomials (discriminants, "bad ideals", norms)
"""

import flint

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
    [(11, 7)]

    (23, 2) is not a bad prime:

    >>> bad_ideals([294521483864, -44775242482, 175176273279])
    [(2, 0), (5, 4)]

    Newton polygon is not enough for these ones:

    >>> bad_ideals([4, 4, -3, 1])
    []
    >>> bad_ideals([4, 0, -1, 0, 1])  # doctest: +SKIP
    []
    """
    assert 3 <= len(f) <= 5
    disc = discriminant(f)
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
                bad.append((l, int(r)))
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
            assert poly_r[0] != 0
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

            # Not supported
            bad.append((l, int(r)))

    return bad
