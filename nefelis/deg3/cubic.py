"""
Factorization in the ring of integers of Q(j)/(j³-c)

This ring has class number one for c = 2,3,4,5,6,9,10,12,16,17,18
and many larger values.

For each ideal above a prime number l, we can select a generator gi
(which is any element of norm l).

Then any algebraic number has the form:
    (z) = product (li)^ei   (factorization of ideal)
    z = u^k0 product(gi^ei)

When the ideal decomposition is known, the missing exponent k0
can be determined using real number logarithms.
"""

import flint
import math


class CubicField:
    def __init__(self, f):
        # Polynomial must be x^3 - c
        assert list(f[1:]) == [0, 0, 1]
        c = -f[0]
        assert c in (2, 3, 5, 6)
        self.c = c
        # Fundamental unit (chosen to be positive)
        if c == 2:
            self.u = 2 ** (1 / 3) - 1
        elif c == 3:
            self.u = 3 ** (2 / 3) - 2
        elif c == 5:
            self.u = 2 * 5 ** (2 / 3) - 4 * 5 ** (1 / 3) + 1
        elif c == 6:
            self.u = 3 * 6 ** (2 / 3) - 6 * 6 ** (1 / 3) + 1
        assert self.u > 0

        self.j = c ** (1 / 3)

        self.gens = {}
        # Check that all primes less than 100 are principal
        for l in range(2, 100):
            if l != c and flint.fmpz(l).is_prime():
                for r, _ in flint.nmod_poly([-c, 0, 0, 1], l).roots():
                    r = int(r)
                    g = self.idealgen(l, r)
                    # print(l, r, g)
                    self.gens[(l, r)] = g

    def norm(self, x, y, z):
        """
        Compute the norm of x+yj+zj² which is also the resultant of x+yt+zt² with t³+c

        >>> K = CubicField(2)
        >>> K.norm(123, 456, 789)
        1890654183
        """
        return x**3 + self.c * y**3 + self.c**2 * z**3 - 3 * self.c * x * y * z

    def idealgen(self, l, r):
        """
        Find a generator of ideal (l, r)

        This uses LLL to find a generator with small coefficients,
        When c in
        """
        B = flint.fmpz_mat([[l, 0, 0], [-r, 1, 0], [-r * r, 0, 1]]).lll()
        # print(f"{l=} {r=}")
        # usually the first vector is already good
        for v in B.table():
            vn = self.norm(*v)
            if vn == -l:
                return tuple(-int(_) for _ in v)
            elif vn == l:
                return tuple(int(_) for _ in v)
        # Otherwise enumerate short vectors
        bound = self.c
        for x in range(0, bound):
            for y in range(-bound, bound):
                for z in range(-bound, bound):
                    v = (flint.fmpz_mat([[x, y, z]]) * B).entries()
                    vn = self.norm(*v)
                    if vn == -l:
                        return tuple(-int(_) for _ in v)
                    elif vn == l:
                        return tuple(int(_) for _ in v)

        assert False, (l, r, B)

    def factor(self, x, y, facs=None):
        """
        Factor an element x+yj into a product of units and prime elements.

        The result is a list of tuples (l, r) and exponents:
        by convention the tuple (1,0) is a fundamental unit.

        >>> K = CubicField(2)
        >>> K.factor(12345, 4567)
        [((281, 119), 1), ((49627, 26218), 1), ((148573, 24201), 1)]
        >>> K.factor(9876, 4321)
        [((2, 0), 1), ((83, 50), 1), ((6774786203, 5031309631), 1), ((1, 0), -1)]
        """
        assert math.gcd(x, y) == 1

        if facs is None:
            facs = flint.fmpz(self.norm(x, y, 0)).factor()

        res = []
        j = self.j
        zr = x + y * j
        for l, e in facs:
            r = -x * pow(y, -1, l) % l
            g = self.idealgen(l, r)
            zr /= (g[0] + g[1] * j + g[2] * j**2) ** e
            res.append(((l, r), e))
        if zr < 0:
            zr = -zr
            res.append(((-1, 0), 1))
        uexp = math.log(abs(zr)) / math.log(self.u)
        eexp = int(round(uexp))
        assert abs(uexp - eexp) < 1e-6
        assert abs(zr - self.u**eexp) < 1e-6
        if eexp:
            res.append(((1, 0), eexp))
        return res


if __name__ == "__main__":
    import tqdm

    B = 10_000_00
    ls = [l for l in range(2, B) if flint.fmpz(l).is_prime()]

    for c in (2, 3, 5, 6):
        print(f"Test field x^3+{c}")
        K = CubicField(c)
        for l in tqdm.tqdm(ls):
            if l == c:
                continue
            rs = flint.nmod_poly([-K.c, 0, 0, 1], l).roots()
            for r, _ in rs:
                K.idealgen(l, int(r))
