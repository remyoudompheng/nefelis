"""
Factorization in the multiplicative group of K = Q[x]/f where f is cubic.

The multiplicative group of K is always a free abelian group (modulo
roots of unity).

For each ideal above a prime number l, we can select a generator gi
(which is any element of norm l).

Then any algebraic number has the form:
    (z) = product (li)^ei   (factorization of ideal)
    z = u^k0 product(gi^ei)

When the ideal decomposition is known, the missing exponent k0=ulog(z)
can be determined using logarithms in a real embedding of K.

If prime ideals are not principal, we can still select a small norm
element and compute virtual real logarithms for each prime ideal,
using the real embedding of K, and perform the same computation.

In typical cases, the coefficients of f are very small (less than 100),
and the class number is also small (less than 10 in more than 90% of cases).

The unit exponent is a (non canonical) multiplicative function:
    ulog(xy) = ulog(x)+ulog(y)
"""

import logging
import math

import flint
from nefelis.integers import smallprimes, factor

logger = logging.getLogger("field")

# Small vectors (up to sign) with coprime coordinates <= 4
smallvectors = [
    (1, 0),
    (0, 1),
    (1, 1),
    (1, -1),
    (2, 1),
    (1, 2),
    (2, -1),
    (1, -2),
    (3, 1),
    (1, 3),
    (3, -1),
    (1, -3),
    (3, 2),
    (2, 3),
    (3, -2),
    (2, -3),
    (4, 1),
    (1, 4),
    (4, -1),
    (1, -4),
]


class CubicField:
    """
    Factorization into basis elements and units, without
    computing an explicit fundamental unit.

    Logarithms of basis elements are determined for
    small prime ideals at init time, and larger ideals
    are computed on-demand. Unit exponents are computed
    relative to a fundamental "logarithm".
    """

    f: list[int]
    alpha: flint.arb
    logs: dict[tuple[int, int], flint.arb]

    def __init__(self, f):
        assert len(f) == 4
        self.f = f
        self.logs = {}

        real_roots = [
            r.real for r, _ in flint.fmpz_poly(f).complex_roots() if r.imag == 0
        ]
        assert len(real_roots) == 1
        alpha = real_roots[0]
        self.alpha = alpha
        logger.info(f"Polynomial {f} has real root {alpha}")

        # Any ideal q larger than this bound contains
        # an element factoring into primes < LARGE_BOUND
        LARGE_BOUND = 1000
        ideals: list[tuple[int, int]] = []
        for l in smallprimes(LARGE_BOUND):
            for r, _ in flint.nmod_poly(f, l).roots():
                ideals.append((l, int(r)))
            if f[-1] % l == 0:
                ideals.append((l, l))
        ideals.sort()

        # Relations (l, r) => (x, y, smalls)
        # where (x,y) is a small ideal element and smalls
        # are small ideal factors.
        small_bound = max(abs(fi) for fi in f)
        large_rels = []
        for l, r in reversed(ideals):
            if r == l:
                lbasis = [[0, l], [1, 0]]
            else:
                lbasis = [[int(r), 1], [l, 0]]
            m = flint.fmpz_mat(lbasis).lll()
            best = None
            for s in smallvectors:
                x, y = (flint.fmpz_mat([s]) * m).entries()
                fxy = sum(f[i] * x**i * y ** (3 - i) for i in range(4))
                assert fxy % l == 0
                if abs(fxy) == l:
                    best = (x, y, [])
                    break
                facs = factor(abs(fxy) // l)
                rel = []
                for _l, _e in facs:
                    if y % _l == 0:
                        _r = _l
                    else:
                        _r = x * pow(y, -1, _l) % _l
                    rel += _e * [(_l, _r)]
                if best is None or best[-1][-1][0] > rel[-1][0]:
                    best = (x, y, rel)
            if best is None:
                logger.error(f"Could not express ideal {l},{r} as smaller ideals")
                raise ValueError("ideal group failure")
            large_rels.append((l, r) + best)
            if not best[-1]:
                # logger.debug(f"Found generator of {l,r} as {x},{y}")
                continue
            maxdep = best[-1][-1][0]
            # logger.debug(f"Found relation of {l,r} with primes <= {maxdep}")
            small_bound = max(small_bound, maxdep)
        logger.debug(f"Bound for class group generators {small_bound}")

        # Generate enough relations to saturate class group relations
        smallbasis = [(l, r) for l, r in ideals if l <= small_bound]
        smallbasis_idx = {b: idx for idx, b in enumerate(smallbasis)}
        basisvecs = []
        for l, r in smallbasis:
            if r == l:
                lbasis = [[0, l], [1, 0]]
            else:
                lbasis = [[int(r), 1], [l, 0]]
            a, b, c, d = flint.fmpz_mat(lbasis).lll().entries()

            def itersmall():
                for i in range(1, 50):
                    for j in range(-i + 1, i):
                        x = i * a + j * c
                        y = i * b + j * d
                        if math.gcd(x, y) == 1:
                            yield x, y
                        x = j * a + i * c
                        y = j * b + i * d
                        if math.gcd(x, y) == 1:
                            yield x, y

            smallrels = []
            for x, y in itersmall():
                fxy = sum(f[i] * x**i * y ** (3 - i) for i in range(4))
                assert fxy % l == 0
                facs = factor(fxy)
                if any(p > small_bound for p, _ in facs):
                    continue
                row = [0 for _ in smallbasis] + [1]
                for __l, __e in facs:
                    if y % __l == 0:
                        __r = __l
                    else:
                        __r = x * pow(y, -1, __l) % __l
                    row[smallbasis_idx[(__l, __r)]] = __e
                real_log = abs(x - y * alpha).log()
                smallrels.append((x**2 + y**2, real_log, row))
                if len(smallrels) > 20:
                    break
            basisvecs += smallrels
        logger.debug(
            f"Found {len(basisvecs)} relations between ideals <= {small_bound}"
        )

        # Sort relations so that small ones come first
        basisvecs.sort()
        M, T = flint.fmpz_mat([row for _, _, row in basisvecs]).lll(transform=True)
        basis = []
        basis_logs = []
        for Mrow, Trow in zip(M.table(), T.table()):
            if any(Mrow):
                z = sum(ti * zi for ti, (_, zi, _) in zip(Trow, basisvecs))
                basis.append(Mrow)
                basis_logs.append(z)
        Mbasis = flint.arb_mat(basis)
        h = abs(Mbasis.det().unique_fmpz())
        logger.debug(f"Tentative class number {h}")

        # Now deduce all logs
        smalli_logs = (
            Mbasis.inv() * flint.arb_mat([[z] for z in basis_logs])
        ).entries()
        assert len(smalli_logs) == len(smallbasis) + 1
        for b, blog in zip(smallbasis, smalli_logs):
            self.logs[b] = blog
        # CONSTANT
        self.logs[(1, 0)] = smalli_logs[-1]

        for l, r, x, y, rel in large_rels:
            if (l, r) in self.logs:
                continue
            z = abs(x - alpha * y).log()
            z -= self.logs[(1, 0)]  # constant for denominator
            for key in rel:
                z -= self.logs[key]
            self.logs[(l, r)] = z

        logger.info(f"Determined real virtual logarithms for {len(self.logs)} ideals")

        def itersmall():
            for i in range(1, 100):
                for j in range(-i + 1, i):
                    if math.gcd(i, j) == 1:
                        yield i, j
                        yield j, i

        fundamental_log = None
        smalllogs = []
        for x, y in itersmall():
            fxy = sum(f[i] * x**i * y ** (3 - i) for i in range(4))
            facs = factor(fxy)
            if all(p < LARGE_BOUND for p, _ in facs):
                z = abs(self.unit_log(x, y))
                if z.contains(0):
                    continue
                smalllogs.append(z)
        fundamental_log = min(smalllogs)
        for z in smalllogs:
            assert (z / fundamental_log).contains_integer()
        # Must coincide with abs(log(real embedding of fundamental unit))
        logger.info(
            f"Fundamental log {fundamental_log} (checked for {len(smalllogs)} values)"
        )
        self.fundamental_log = fundamental_log

    def ideal_log(self, l, r):
        if (l, r) in self.logs:
            return self.logs[(l, r)]

        f = self.f
        alpha = self.alpha
        if r == l:
            lbasis = [[0, l], [1, 0]]
        else:
            lbasis = [[int(r), 1], [l, 0]]
        m = flint.fmpz_mat(lbasis).lll()
        for _ in range(2):
            missing = []
            for s in smallvectors:
                x, y = (flint.fmpz_mat([s]) * m).entries()
                fxy = sum(f[i] * x**i * y ** (3 - i) for i in range(4))
                assert fxy % l == 0
                facs = factor(abs(fxy) // l)
                rel = []
                for _l, _e in facs:
                    if y % _l == 0:
                        _r = _l
                    else:
                        _r = x * pow(y, -1, _l) % _l
                    rel += _e * [(_l, _r)]
                if all(key in self.logs for key in rel):
                    # Found a good relation
                    z = abs(x - alpha * y).log()
                    z -= self.logs[(1, 0)]  # constant for denominator
                    for key in rel:
                        z -= self.logs[key]
                    self.logs[(l, r)] = z
                    return z
                else:
                    missing.extend(key for key in rel if key not in self.logs)
            # None of the small vectors could be factored, add a few missing
            # ideals and try again.
            missing = sorted(set(missing))
            for _l, _r in missing[:16]:
                self.ideal_log(_l, _r)
            # logger.debug(f"Added missing logs for {missing[:16]}")

        raise ValueError(f"unable to compute logarithm for ideal {l},{r}")

    def unit_log(self, x, y) -> flint.arb:
        assert math.gcd(x, y) == 1
        f = self.f
        fxy = sum(f[i] * x**i * y ** (3 - i) for i in range(4))
        facs = factor(fxy)
        z = abs(x - y * self.alpha).log()
        z -= self.logs[(1, 0)]
        for l, e in facs:
            if y % l == 0:
                r = l
            else:
                r = x * pow(y, -1, l) % l
            z -= e * self.ideal_log(l, r)
        return z

    def unit_exponent(self, x, y) -> int:
        assert self.fundamental_log is not None
        e = self.unit_log(x, y) / self.fundamental_log
        exponent = e.unique_fmpz()
        assert exponent is not None
        return int(exponent)


class CubicFieldUFD:
    """
    An ad-hoc implementation dedicated to cases where O_K
    is a UFD and an explicit fundamental unit is known.
    """

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

        >>> K = CubicField([-2, 0, 0, 1])
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

        >>> K = CubicField([-2, 0, 0, 1])
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
    import random
    import tqdm

    logging.basicConfig(level=logging.DEBUG)

    B = 1000_000
    ls = [l for l in range(2, B) if flint.fmpz(l).is_prime()]

    for c in (2, 3, 5, 6):
        print(f"Test field x^3-{c}")
        K = CubicFieldUFD([-c, 0, 0, 1])
        for l in tqdm.tqdm(ls):
            if l == c:
                continue
            rs = flint.nmod_poly([-K.c, 0, 0, 1], l).roots()
            for r, _ in rs:
                K.idealgen(l, int(r))

    f = [23, 10, -15, 9]
    print(f"Test cubic field {f}")
    # log(fundamental unit) = 78.34017514152009
    K = CubicField(f)
    for l in tqdm.tqdm(ls):
        rs = flint.nmod_poly(f, l).roots()
        for r, _ in rs:
            K.ideal_log(l, int(r))
    print("Unit exponents for random elements")
    exps = {}
    for _ in tqdm.trange(10000):
        while True:
            x = random.getrandbits(10)
            y = random.getrandbits(10)
            if math.gcd(x, y) == 1:
                break
        e = K.unit_exponent(x, y)
        assert e is not None
        exps[e] = exps.get(e, 0) + 1
    print("Statistics", exps)

    f = [16, 17, -19, 15]
    print(f"Test cubic field {f} with class number 23")
    K = CubicField(f)
    # log(fundamental unit) = 22.51813079415122
