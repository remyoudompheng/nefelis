"""
Computation of individual discrete logarithms for degree 3

Only sieve results with Schirokauer maps are supported.
Here g is a quadratic polynomial, and the smoothing step
has to return prime factors that split over g.
"""

import json
import logging
import math
import pathlib
import random
import sys
import time

import flint

from nefelis import integers
from nefelis import sieve_vk
from nefelis.deg3 import linalg

logger = logging.getLogger("dlog")


def main():
    workdir = pathlib.Path(sys.argv[1])
    arg = int(sys.argv[2])

    with open(workdir / "args.json") as fd:
        doc = json.load(fd)
        n = doc["n"]
        f = doc["f"]
        g = doc["g"]

    # FIXME
    ell = n // 2

    dlogs = {}
    zlogs = {}
    logbase = None
    with open(workdir / "dlog") as fd:
        for line in fd:
            key, val = line.split()
            val = int(val)
            dlogs[key] = val

            if key.startswith("Z_"):
                zlogs[int(key[2:])] = val
                if val == 1:
                    logbase = int(key[2:])

    zmax = max(z for z in zlogs if z.bit_length() < 32)
    logging.info(
        f"Read {len(zlogs)} logarithms of small rational primes ({min(zlogs)}..{zmax})"
    )
    logging.info(f"Read {len(dlogs) - len(zlogs)} logarithms of small algebraic primes")
    logging.info(f"Logarithm base is {logbase}")

    D = Descent(dlogs, zlogs, logbase, n, f, g, ell)
    y = D.log(arg)

    coell = (n - 1) // ell
    assert pow(logbase, coell * y, n) == pow(arg, coell, n)
    logging.info(f"Found log({arg}) mod {ell} = {y}")
    print(y)


smallvectors = [
    (_x, _y) for _x in range(10) for _y in range(-10, 10) if math.gcd(_x, _y) == 1
]


def parse_key(k):
    words = k.split("_")
    return int(words[1]), int(words[2])


class Descent:
    def __init__(self, dlogs, zlogs, logbase, n, f, g, ell):
        self.dlogs = dlogs
        self.zlogs = zlogs

        # For fast trial division, we build the product of all rational primes.
        def product(zs):
            if len(zs) == 1:
                return zs[0]
            else:
                return product(zs[: len(zs) // 2]) * product(zs[len(zs) // 2 :])

        zs = sorted(z for z in zlogs if z.bit_length() < 32)
        self.zprod = flint.fmpz(product([z ** (32 // z.bit_length()) for z in zs]))
        self.logbase = logbase
        self.n = n
        self.f = f
        self.g = g
        self.discg = flint.fmpz(g[1] ** 2 - 4 * g[0] * g[2])
        self.ell = ell
        self.sm_place = linalg.schirokauer_place(f, ell)

        nbits = n.bit_length()
        if nbits < 220:
            self.MAX_QBITS = 45
            COFACTOR_BITS = 20
        elif nbits < 250:
            self.MAX_QBITS = 48
            COFACTOR_BITS = 25
        elif nbits < 300:
            self.MAX_QBITS = 56
            COFACTOR_BITS = 30

        fls, frs = [], []
        gls, grs = [], []
        for k in dlogs:
            if k.startswith("f_"):
                l, lr = parse_key(k)
                fls.append(l)
                frs.append(lr)
            if k.startswith("g_"):
                l, lr = parse_key(k)
                gls.append(l)
                grs.append(lr)
        gsize = max(gi.bit_length() for gi in g)
        LOGWIDTH = 14
        # Size of f is 3 * (LOGWIDTH + logq / 2)
        fthreshold = 2 * LOGWIDTH
        gthreshold = gsize + 2 * LOGWIDTH - COFACTOR_BITS

        # Siever for q less than 30 bits
        self.siever = sieve_vk.Siever(
            self.g,
            gls,
            grs,
            gthreshold,
            LOGWIDTH,
            # We want f(x,y) to factor completely
            self.f,
            fls,
            frs,
            fthreshold,
        )
        # For very large q, use a much lower threshold, expecting a relation
        # with multiple moderately large primes
        self.sieverlarge = sieve_vk.Siever(
            self.g,
            gls,
            grs,
            gthreshold - 40,
            LOGWIDTH + 1,
            # We want f(x,y) to factor completely
            self.f,
            fls,
            frs,
            fthreshold + 15,
            # We want many results
            outsize=2 << 20,
        )
        # For q > 2^35
        self.sieverhuge = sieve_vk.Siever(
            self.g,
            gls,
            grs,
            gthreshold - 40,
            LOGWIDTH + 1,
            # We want f(x,y) to factor completely
            self.f,
            fls,
            frs,
            fthreshold + 30,
            # We want many results
            outsize=32 << 20,
        )

        self.badprimes = [
            _l
            for _l in range(2, 100000)
            if flint.fmpz(_l).is_prime() and _l not in zlogs and not self.splits(_l)
        ]
        self.badprod = flint.fmpz(product(self.badprimes))
        prob = 1.0
        for l in self.badprimes:
            prob *= l / (l + 1)
        logger.info(
            f"{len(self.badprimes)} non split primes < 100000, failure rate {100 * prob:.1f}%"
        )

    def _factor(self, v):
        d = int(self.zprod.gcd(v))
        facs = integers.factor(d)
        if v < 0:
            # Sign -1 is ignored in this dlog
            v = -v
        return facs, v // d

    def splits(self, l):
        return l in self.zlogs or self.discg.jacobi(l) != -1

    def sm(self, x, y):
        return linalg.schirokauer_map(x, y, self.sm_place, self.ell)

    def smooth_candidates(self, x0):
        """
        Iterates over fractions x0 = cofacs(xu/xv) * facs where xu and xv are small
        """
        n = self.n
        for i in range(1_000_000):
            if i > 0:
                # No need to take huge exponents, a 64-bit integer is enough.
                e = random.getrandbits(64)
                x = x0 * pow(self.logbase, -e, n) % n
                facs0 = [(self.logbase, e)]
            else:
                x = x0
                facs0 = []

            if x.bit_length() < n.bit_length() // 2:
                facs_, x = self._factor(x)
                facs = facs0 + facs_
                cofacs = flint.fmpz(x).factor_smooth(32)
                yield facs, cofacs, x, 1
            else:
                # Try rational reconstruction
                m = flint.fmpz_mat([[self.n, 0], [int(x), 1]]).lll()
                for su, sv in (flint.fmpz_mat(smallvectors) * m).table():
                    if self.badprod.gcd(su * sv) != 1:
                        continue
                    facs_u, xu = self._factor(int(su))
                    facs_v, xv = self._factor(int(sv))
                    cofacs = integers.factor(xu)
                    cofacs += [(_l, -_e) for _l, _e in integers.factor(xv)]
                    cofacs.sort()
                    if any(
                        flint.fmpz(_l).is_prime() and not self.splits(_l)
                        for _l, _ in cofacs
                    ):
                        continue

                    facs = facs0 + facs_u
                    facs += [(_l, -_e) for _l, _e in facs_v]
                    yield facs, cofacs, xu, xv

    def log(self, x0):
        """
        Express x as a product of elements of the factor base and small primes
        """
        best = None
        # FIXME: make iteration count configurable
        # FIXME: make bound 32bits configurable
        i = 0
        for facs, cofacs, xu, xv in self.smooth_candidates(x0):
            i += 1
            # Now x = product(facs) * xu / xv
            if best is None:
                if all(
                    flint.fmpz(_l).is_prime() and self.splits(_l) for _l, _ in cofacs
                ):
                    # factorization may be incomplete
                    logging.info(f"Decomposed {x0} as {facs} and cofactor {xu}/{xv}")
                    best = facs, cofacs
            else:
                if not cofacs or (cofacs[-1][0] < best[1][-1][0]):
                    if any(not flint.fmpz(_l).is_prime() for _l, _ in cofacs):
                        cofacs = integers.factor(xu)
                        cofacs += [(_l, -_e) for _l, _e in integers.factor(xv)]
                        cofacs.sort()

                    facs_str = "*".join(f"{_l}^{_e}" for _l, _e in facs)
                    cofacs_str = "*".join(f"{_l}^{_e}" for _l, _e in cofacs)
                    logging.info(
                        f"[i={i}] Decomposed {x0} as {facs_str} and cofactor {xu}/{xv}={cofacs_str}"
                    )
                    best = facs, cofacs

            if not cofacs:
                break
            if i > 100 and cofacs[-1][0].bit_length() < self.MAX_QBITS:
                break

        facs, cofacs = best
        # x0 = product(facs) * x
        log = sum(e * self.zlogs[f] for f, e in facs)
        coell = (self.n - 1) // self.ell
        for _l, _e in cofacs:
            assert self.splits(_l) and _l not in self.zlogs
            # We have to sieve both prime ideals above l in K(g)
            lroots = flint.nmod_poly(self.g, _l).roots()
            assert len(lroots) == 2
            z = 0
            for lr, _ in lroots:
                logging.info(f"Recurse into small prime {_l},{lr}")
                llog = self.smalllog(_l, int(lr))
                z += llog
                log += _e * llog
            # z should be log(l)
            logging.info(f"recursive log({_l}) = {z}")
            assert pow(_l, coell, self.n) == pow(self.logbase, z * coell, self.n)

        return log % self.ell

    def fg(self, x: int, y: int) -> tuple[int, int]:
        D, C, B, A = self.f
        w, v, u = self.g
        vf = (A * x + B * y) * x * x + (C * x + D * y) * y * y
        vg = u * x * x + v * x * y + w * y * y
        return vf, vg

    def smalllog(self, q: int, qr: int) -> int:
        """
        Compute logarithm of a "small" prime using sieving.
        """
        assert flint.fmpz(q).is_prime()

        t = time.monotonic()
        if q.bit_length() <= 29:
            reports = self.siever.sieve(q, qr)
        elif q.bit_length() <= 35:
            reports = self.sieverlarge.sievelarge(q, qr)
        else:
            reports = self.sieverhuge.sievelarge(q, qr)
        dt = time.monotonic() - t
        logging.info(f"Found {len(reports)} sieve results in {dt:.3}s")

        # Store: a list of factors, list of missing primes, the largest missing prime
        best: tuple[list, list, int] | None = None

        for x, y in reports:
            x, y = int(x), int(y)
            if math.gcd(x, y) != 1:
                continue
            vf, vg = self.fg(x, y)
            facf = integers.factor(abs(vf))

            good = True
            facs = [
                ("SM", self.sm(x, -y)),
                # The CONSTANT accounts for leading coefficients of f and g
                ("CONSTANT", 1),
            ]
            for _l, _e in facf:
                _r = x * pow(y, -1, _l) % _l if y % _l else _l
                if (key := f"f_{_l}_{_r}") not in self.dlogs:
                    good = False
                    break
                else:
                    facs.append((key, _e))

            if vg % q != 0:
                raise ArithmeticError(
                    f"Unexpected g(x,y) not divisible by {q=}, overflow?"
                )

            if not good:
                continue

            # z = g(z)/u = f(z)
            if good:
                facg = integers.factor(abs(vg) // q)
                for _l, _e in facg:
                    _r = x * pow(y, -1, _l) % _l if y % _l else _l
                    key = f"g_{_l}_{_r}"
                    facs.append((key, -_e))

                missing = []
                worst = 0
                for k, _ in facs:
                    if k not in self.dlogs:
                        assert k.startswith("g_")
                        _l, _lr = parse_key(k)
                        worst = max(worst, _l)
                        missing.append((_l, _lr))

                keep = (
                    best is None
                    or not missing
                    or (best[2] > q and worst < q)
                    or (len(missing) <= len(best[1]) and max(missing) < max(best[1]))
                )
                if keep:
                    facs_str = "*".join(f"{_l}^{_e}" for _l, _e in facs)
                    if not missing:
                        logging.info(f"Good report {x},{y}")
                        logging.info(f"g_{q}_{qr} = {facs_str}")
                        log = sum(e * self.dlogs[k] for k, e in facs) % self.ell
                        logging.info(f"log(g_{q}_{qr}) = {log}")
                        self.dlogs[f"g_{q}_{qr}"] = log
                        return log
                    else:
                        best = facs, missing, worst

        if best is None:
            raise ValueError(f"failed to find any relation for ideal {q},{qr}")

        facs, missing, _ = best
        logging.info(f"Best relation {q} = {facs_str}")
        logging.info(f"Best relation is missing primes {missing}")
        log = 0
        for k, e in facs:
            if k not in self.dlogs:
                assert k.startswith("g_")
                _l, _lr = parse_key(k)
                if _l > q:
                    logging.info(f"Recurse into LARGER prime {_l},{_lr}")
                else:
                    logging.info(f"Recurse into small prime {_l},{_lr}")
                log += e * self.smalllog(_l, _lr)
            else:
                log += e * self.dlogs[k]

        log %= self.ell
        logging.info(f"Recursive log(g_{q}_{qr}) = {log}")
        self.dlogs[f"g_{q}_{qr}"] = log
        return log


if __name__ == "__main__":
    logging.getLogger().setLevel(level=logging.DEBUG)
    main()
