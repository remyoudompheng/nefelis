"""
Computation of individual discrete logarithms

Since g is a linear polynomial, any prime can be used as "special q".
"""

import json
import logging
import math
import pathlib
import random
import sys
import time

import flint
import pymqs

from nefelis import sieve_vk


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
        self.zprod = flint.fmpz(product(zs))
        self.logbase = logbase
        self.n = n
        self.f = f
        self.g = g
        self.ell = ell

        nbits = n.bit_length()
        if nbits < 250:
            self.MAX_QBITS = 48
            THRESHOLD = 20
        elif nbits < 300:
            self.MAX_QBITS = 56
            THRESHOLD = 25

        v, u = g
        rs = [(-v * pow(u, -1, l)) % l if u % l else l for l in zs]
        self.siever = sieve_vk.Siever(
            self.g, zs, rs, self.n.bit_length() // 2 - THRESHOLD, 14
        )

    def _factor(self, v):
        d0 = int(self.zprod.gcd(v))
        d = d0
        facs = []
        while d > 1:
            v //= d
            facs += flint.fmpz(d).factor()
            d = math.gcd(d0, v)
        return facs, v

    def log(self, x0):
        """
        Express x as a product of elements of the factor base and small primes
        """
        best = None
        n = self.n
        # FIXME: make iteration count configurable
        # FIXME: make bound 32bits configurable
        for i in range(1_000_000):
            if i > 0:
                e = random.randrange(n)
                x = x0 * pow(self.logbase, -e, n) % n
                facs = [(self.logbase, e)]
            else:
                x = x0
                facs = []

            if x.bit_length() < n.bit_length() // 2:
                facs_, x = self._factor(x)
                facs += facs_
                xu, xv = x, 1
            else:
                # Try rational reconstruction
                m = flint.fmpz_mat([[self.n, 0], [int(x), 1]]).lll()
                u, v = m.table()[0]
                # print(f"x={u}/{v}")
                facs_u, xu = self._factor(int(u))
                facs_v, xv = self._factor(int(v))
                facs += facs_u
                facs += [(_l, -_e) for _l, _e in facs_v]

            # Now x = product(facs) * xu / xv
            if best is None:
                logging.info(f"Decomposed {x0} as {facs} and cofactor {xu}/{xv}")
                cofacs = flint.fmpz(xu).factor_smooth(32)
                cofacs += [(_l, -_e) for _l, _e in flint.fmpz(xv).factor_smooth(32)]
                if all(flint.fmpz(_l).is_prime() for _l, _ in cofacs):
                    # factorization may be incomplete
                    best = facs, cofacs
            else:
                cofacs = flint.fmpz(xu).factor_smooth(32)
                cofacs += [(_l, -_e) for _l, _e in flint.fmpz(xv).factor_smooth(32)]
                cofacs.sort()
                if not cofacs or (cofacs[-1][0] < best[1][-1][0]):
                    if any(not flint.fmpz(_l).is_prime() for _l, _ in cofacs):
                        cofacs = flint.fmpz(xu).factor()
                        cofacs += [(_l, -_e) for _l, _e in flint.fmpz(xv).factor()]
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
        for _l, _e in cofacs:
            logging.info(f"Recurse into small prime {_l}")
            llog = self.smalllog(_l)
            log += _e * llog

        return log % self.ell

    def fg(self, x, y):
        C, B, A = self.f
        v, u = self.g
        vf = A * x * x + B * x * y + C * y * y
        vg = u * x + v * y
        return vf, vg

    def smalllog(self, q):
        """
        Compute logarithm of a "small" prime using sieving.
        """
        assert flint.fmpz(q).is_prime()
        coell = (self.n - 1) // self.ell
        v, u = self.g
        qr = -v * pow(u, -1, q) % q
        t = time.monotonic()
        if q.bit_length() <= 30:
            reports = self.siever.sieve(q, qr)
        else:
            reports = self.siever.sievelarge(q, qr)
        dt = time.monotonic() - t
        logging.info(f"Found {len(reports)} sieve results in {dt:.3}s")

        # Store: a list of factors, list of missing primes
        best = None

        for x, y in reports:
            x, y = int(x), int(y)
            if math.gcd(x, y) != 1:
                continue
            vf, vg = self.fg(x, y)
            facf = pymqs.factor(abs(vf))
            good = True
            facs = []
            for _l in facf:
                _r = x * pow(y, -1, _l) % _l if y % _l else _l
                if (key := f"f_{_l}_{_r}") not in self.dlogs:
                    good = False
                    break
                else:
                    facs.append((key, 1))
            facg = pymqs.factor(abs(vg))
            # z = g(z)/u = f(z)
            if good:
                for _l in facg:
                    if _l != q:
                        facs.append((f"Z_{_l}", -1))
                # The CONSTANT accounts for leading coefficients of f and g
                facs.append(("CONSTANT", 1))

                missing = [_l for _l in facg if _l not in self.zlogs and _l != q]
                keep = (
                    best is None
                    or not missing
                    or (any(_m > q for _m in best[1]) and all(_m < q for _m in missing))
                    or (len(missing) <= len(best[1]) and max(missing) < max(best[1]))
                )
                if keep:
                    facs_str = "*".join(f"{_l}^{_e}" for _l, _e in facs)
                    if not missing:
                        logging.info(f"Good report {x},{y}")
                        logging.info(f"{q} = {facs_str}")
                        log = sum(e * self.dlogs[k] for k, e in facs)
                        logging.info(f"log({q}) = {log}")
                        assert pow(q, coell, self.n) == pow(
                            self.logbase, log * coell, self.n
                        )
                        return log
                    else:
                        best = facs, missing

        facs, missing = best
        logging.info(f"Best relation {q} = {facs_str}")
        logging.info(f"Best relation is missing primes {missing}")
        log = 0
        for k, e in facs:
            if k not in self.dlogs:
                assert k.startswith("Z_")
                _l = int(k[2:])
                if _l > q:
                    logging.info(f"Recurse into LARGER prime {_l}")
                else:
                    logging.info(f"Recurse into small prime {_l}")
                log += e * self.smalllog(_l)
            else:
                log += e * self.dlogs[k]

        log %= self.ell
        logging.info(f"recursive log({q}) = {log}")
        assert pow(q, coell, self.n) == pow(self.logbase, log * coell, self.n)
        self.zlogs[q] = log
        self.dlogs[f"Z_{q}"] = log
        return log


if __name__ == "__main__":
    logging.getLogger().setLevel(level=logging.DEBUG)
    main()
