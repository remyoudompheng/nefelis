"""
Computation of individual discrete logarithms in GF(p^2)

Currently only computations in the norm 1 subgroup (of order p+1)
modulo the largest prime factor are implemented.
"""

import argparse
import json
import logging
import math
import pathlib
import random
import time
from concurrent.futures import ThreadPoolExecutor

import flint

from nefelis import integers
from nefelis import sieve_vk

logger = logging.getLogger("dlog")

# Recompute known logarithms for double checks
DEBUG_SMALL_LOGS = False

# Check intermediate logarithms (cheap)
DEBUG_EXTRA_CHECKS = True

DEBUG_TIMINGS = True


def main():
    argp = argparse.ArgumentParser()
    argp.add_argument(
        "-v",
        action="store_true",
        help="Verbose logging",
    )
    argp.add_argument("WORKDIR")
    argp.add_argument("ARGS", nargs="+", help="list of pairs x,y (for elements x+iy)")
    args = argp.parse_args()

    if args.v:
        logging.getLogger().setLevel(level=logging.DEBUG)
    else:
        logging.getLogger().setLevel(level=logging.INFO)

    main_impl(args)


def main_impl(args):
    workdir = pathlib.Path(args.WORKDIR)
    args = args.ARGS

    with open(workdir / "args.json") as fd:
        doc = json.load(fd)
        n = doc["n"]
        f = doc["f"]
        g = doc["g"]
        conway = doc["conway"]

        ZnX = flint.fmpz_mod_poly_ctx(n)
        Fp2 = flint.fq_default_ctx(n, 2, var="i", modulus=ZnX(conway))
        z = Fp2(doc["z"])

    ell = integers.factor(n + 1)[-1][0]
    logger.info(f"Computing logarithms modulo {ell}")

    dlogs = {}
    with open(workdir / f"subgroup.{ell}" / "gen") as fd:
        gx, gy = fd.read().strip().split(",")
        gen = Fp2([int(gx), int(gy)])
    with open(workdir / f"subgroup.{ell}" / "dlog") as fd:
        for line in fd:
            key, val = line.split()
            val = int(val)
            dlogs[key] = val

    logger.info(f"Read {len(dlogs)} logarithms of small algebraic primes")

    D = Descent(dlogs, gen, n, f, g, z, ell)
    for arg in args:
        x, y = arg.split(",")
        xy = Fp2([int(x), int(y)])
        logger.info(f"Input argument {xy}")
        xyproj = xy**D.projexp
        assert xyproj**ell == 1
        logger.info(f"In order l subgroup: {xyproj}")

        dl = D.log(xy)
        assert gen**dl == xyproj
        logger.info(f"Found log({x}+{y}*i) mod {ell} = {dl}")
        print(dl)


smallvectors = [
    (_x, _y) for _x in range(10) for _y in range(-10, 10) if math.gcd(_x, _y) == 1
]


def keyg(l, r) -> str:
    if isinstance(l, str):
        return l
    return f"g_{l}_{r}"


def parse_key(k):
    words = k.split("_")
    return int(words[1]), int(words[2])


PARAMS = [
    # nbits: modulus size (in bits)
    # initial threshold during rational reconstruction sieve (as a fraction of n)
    # MAX_QBITS: max prime size in initial decomposition (must be under 64 bits)
    # COFACTOR_BITS: cofactor size during descent
    # COFACTOR_BITS2: cofactor size for large q descent
    (100, 0.7, 40, 10, 20),
    (120, 0.65, 50, 15, 40),
    (140, 0.6, 60, 15, 50),
    (160, 0.50, 60, 15, 50),
]


class Descent:
    def __init__(self, dlogs, logbase, n, f, g, z, ell):
        self.dlogs = dlogs
        self.pending = set()

        self.logbase = logbase
        self.n = n
        self.f = f
        self.g = g
        self.z = z
        self.discg = flint.fmpz(g[1] ** 2 - 4 * g[0] * g[2])
        self.ell = ell
        coell = (n + 1) // ell
        self.projexp = (n - 1) * coell * pow((n - 1) * coell, -1, ell)
        assert (self.projexp**2 - self.projexp) % (n * n - 1) == 0
        self.Fp = flint.fmpz_mod_ctx(n)

        nbits = n.bit_length()
        _, ratio, self.MAX_QBITS, COFACTOR_BITS, COFACTOR_BITS2 = min(
            PARAMS, key=lambda p: abs(p[0] - nbits)
        )
        self.zthreshold = int(ratio * nbits)

        self.dlogs["GEN"] = 1
        if "CONSTANT" not in self.dlogs:
            self.dlogs["CONSTANT"] = 0

        fls, frs = [], []
        gls, grs = [], []
        self.glogs = set()
        for k in dlogs:
            if k.startswith("f_"):
                l, lr = parse_key(k)
                fls.append(l)
                frs.append(lr)
                # assert sum(fi * lr**i for i, fi in enumerate(f)) % l == 0
            if k.startswith("g_"):
                l, lr = parse_key(k)
                gls.append(l)
                grs.append(lr)
                self.glogs.add((l, lr))
        self.g_primes = list(zip(gls, grs))
        # Also add more small primes to make sieving easier
        self.g_primes_extended = self.g_primes.copy()
        glextra = integers.smallprimes(max(gls))
        g_set = set(gls)
        extra = 0
        for l in glextra:
            if l not in g_set:
                for r, _ in flint.nmod_poly(g, l).roots():
                    self.g_primes_extended.append((l, int(r)))
                    gls.append(l)
                    grs.append(int(r))
                    extra += 1
        logger.info(f"Added {extra} small Kg primes with unknown logs for sieving")
        self.g_primes.sort()
        self.g_primes_extended.sort()

        # MAX_FBITS: allowed size for special-q on the other side (Kf)
        self.MAX_FBITS = max(max(fls), max(gls)).bit_length() + 5

        gsize = max(gi.bit_length() for gi in g)
        LOGWIDTH = 14
        # Size of f is 4 * (LOGWIDTH + logq / 2)
        fthreshold = 4 * LOGWIDTH
        gthreshold = gsize + 2 * LOGWIDTH - COFACTOR_BITS
        gthreshold2 = gsize + 2 * LOGWIDTH - COFACTOR_BITS2

        logger.info(f"Prime ideals f:{len(fls)} g:{len(gls)}")
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
            (gthreshold + gthreshold2) // 2,
            LOGWIDTH,
            # We want f(x,y) to factor completely
            self.f,
            fls,
            frs,
            fthreshold + 10,
            # We want many results
            outsize=2 << 20,
        )
        # For q > 2^35
        self.sieverhuge = sieve_vk.Siever(
            self.g,
            gls,
            grs,
            gthreshold2,
            LOGWIDTH,
            # We want f(x,y) to factor completely
            self.f,
            fls,
            frs,
            fthreshold + 20,
            # We want many results
            outsize=8 << 20,
        )

    def _rootsg(self, l):
        rs = flint.nmod_poly(self.g, l).roots()
        if len(rs) == 1:
            return rs[0][0], rs[0][0]
        else:
            return rs[0][0], rs[1][0]

    factor_stats = 0.0
    factor_stats_decay = 0.8

    def _factorg(self, x, y):
        """
        Factor x - y * z into a product of ideals and a cofactor
        """
        # Non coprime x,y is possible if the GCD is
        # only divisible by split primes.
        ideals = []
        cofacs = []

        d = math.gcd(x, y)
        if d > 1:
            # FIXME: This is broken currently
            return None, None
        x //= d
        y //= d
        for l, e in integers.factor(d):
            # print("GCD", l)
            if self.discg.jacobi(l) == -1:
                # Found a non-split prime
                return None, None
            r1, r2 = self._rootsg(l)
            ideals.append(((l, r1), e))
            ideals.append(((l, r2), e))

        t0 = time.monotonic()
        facs = integers.factor_smooth(self.norm(x, y), self.MAX_QBITS - 5)
        if DEBUG_TIMINGS:
            dt = time.monotonic() - t0
            self.factor_stats *= self.factor_stats_decay
            self.factor_stats += dt

        for l, e in facs:
            if not flint.fmpz(l).is_prime():
                return None, None
            l, e = int(l), int(e)
            r = x * pow(y, -1, l) % l if y % l else l
            if (l, r) in self.glogs:
                ideals.append(((l, r), e))
            else:
                cofacs.append(((l, r), e))
        return ideals, cofacs

    def convert_kg(self, xy) -> tuple[int, int]:
        """
        Convert a GF(p^2) element x+iy to coordinates x+yz where z
        is the common root of f and g.
        """
        x, y = xy.to_list()
        z0, z1 = self.z.to_list()
        yy = self.Fp(y) / self.Fp(z1)
        xx = x - yy * z0
        assert xx + yy * self.z == xy
        return int(xx), int(yy)

    def norm(self, a, b):
        a, b = int(a), int(b)
        C, B, A = self.g
        return A * a * a + B * a * b + C * b * b

    def smooth_candidates(self, x0):
        """
        Decompose a GF(p^2) element into a quotient of elements
        with small coefficients.
        """
        n = self.n
        z = self.z
        factorpool = ThreadPoolExecutor(8)

        def factor_rs(args):
            Rx, Ry, Sx, Sy = args
            facs_u, xu = self._factorg(Rx, Ry)
            facs_v, xv = self._factorg(Sx, Sy)
            return facs_u, xu, facs_v, xv, args

        for i in range(1_000_000):
            if i > 0:
                # No need to take huge exponents, a 64-bit integer is enough.
                e = random.getrandbits(64)
                x = x0 * self.logbase**-e
                facs0 = [(("GEN", None), e)]
            else:
                x = x0
                facs0 = []

            a, b = self.convert_kg(x)
            if max(a.bit_length(), b.bit_length()) < n.bit_length() // 2:
                facs_, x = self._factor(x)
                facs = facs0 + facs_
                cofacs = flint.fmpz(x).factor_smooth(32)
                yield facs, cofacs, x, 1
            else:
                # Try rational reconstruction
                # To match the convention of polynomial g, we consider
                # elements x - y * z with norm g(x,y)
                #
                # By construction thay always split into good K(g) primes
                #
                # All coefficients have size sqrt(n) so the norm is O(n^1.5)
                a1, b1 = self.convert_kg(x * z)
                m = flint.fmpz_mat(
                    [
                        [int(a1), int(-b1), 0, -1],
                        [int(a), int(-b), 1, 0],
                        [0, self.n, 0, 0],
                        [self.n, 0, 0, 0],
                    ]
                ).lll()
                # print(m)
                a1, b1, c1, d1 = m.table()[0]
                a2, b2, c2, d2 = m.table()[1]
                assert (a1 - b1 * z) / (c1 - d1 * z) == x
                assert (a2 - b2 * z) / (c2 - d2 * z) == x
                # Now we have written x = R1 / S1 = R2 / S2
                # Use sieve to find (R1 x + R2 y, S1 x + S2 y) smooth
                # Smoothness should be understood in field K(g)
                ls, lr1, lr2 = [], [], []
                for l, lr in sorted(self.g_primes_extended):
                    # R1 x + R2 y = (a1 x + a2 y) - (b1 x + b2 y) z
                    # This is the same transformation as sieving q-lattices
                    # x/y = (b2 r - a2)/(a1 - r b1) mod l
                    ls.append(l)
                    if lr == l:
                        r1 = b2 * pow(-b1, -1, l) % l if b1 % l else l
                        r2 = d2 * pow(-d1, -1, l) % l if d1 % l else l
                        lr1.append(r1)
                        lr2.append(r2)
                        continue
                    if (a1 - lr * b1) % l == 0:
                        r1 = l
                    else:
                        r1 = (b2 * lr - a2) * pow(a1 - lr * b1, -1, l) % l
                    if (c1 - lr * d1) % l == 0:
                        r2 = l
                    else:
                        r2 = (d2 * lr - c2) * pow(c1 - lr * d1, -1, l) % l
                    lr1.append(r1)
                    lr2.append(r2)

                LOGWIDTH = 14
                sieve = sieve_vk.Siever(
                    self.g,
                    ls,
                    lr1,
                    self.zthreshold,
                    LOGWIDTH,
                    # We want f(x,y) to factor completely
                    self.g,
                    ls,
                    lr2,
                    self.zthreshold,
                    # We want many results
                    outsize=2 << 20,
                )

                n_fractions = 0
                for q, qr in [(1, 0)] + list(zip(ls[:40], lr1[:40])):
                    if q == qr:
                        continue
                    reports = sieve.sieve(q, qr)
                    logger.debug(f"Vector sieve {q},{qr}: {len(reports)} reports")
                    rs = []
                    for sx, sy in reports:
                        if math.gcd(sx, sy) != 1:
                            continue
                        n_fractions += 1
                        Rx, Ry = (a1 * sx + a2 * sy), (b1 * sx + b2 * sy)
                        Sx, Sy = (c1 * sx + c2 * sy), (d1 * sx + d2 * sy)
                        assert (Rx - Ry * z) / (Sx - Sy * z) == x
                        # print(Rx, Ry)
                        # print(Sx, Sy)
                        rs.append((Rx, Ry, Sx, Sy))

                    for facs_u, xu, facs_v, xv, rs in factorpool.map(factor_rs, rs):
                        if facs_u is None or facs_v is None:
                            continue

                        facs = facs0 + facs_u
                        facs += [(_l, -_e) for _l, _e in facs_v]

                        cofacs = xu
                        cofacs += [(_l, -_e) for _l, _e in xv]
                        cofacs.sort()

                        # print("R", Rx, Ry)
                        # print("S", Sx, Sy)
                        yield facs, cofacs, rs

                    if DEBUG_TIMINGS:
                        avg = self.factor_stats * (1 - self.factor_stats_decay)
                        logger.debug(f"Average factor duration {avg * 1000:.3f}ms")

    def log(self, x0):
        """
        Express x as a product of elements of the factor base and small primes
        """
        best = None
        # FIXME: make iteration count configurable
        # FIXME: make bound 32bits configurable
        logger.info(
            f"Trying to decompose {x0} into small primes ({self.MAX_QBITS} bits)"
        )
        t0 = time.monotonic()
        for facs, cofacs, RS in self.smooth_candidates(x0):
            # Now x = product(facs) * xu / xv
            if best is None:
                facs_str = "*".join(f"{keyg(*_l)}^{_e}" for _l, _e in facs)
                cofacs_str = "*".join(f"{keyg(*_l)}^{_e}" for _l, _e in cofacs)
                logger.info(f"Found relation as {facs_str} and cofactor {cofacs_str}")
                best = facs, cofacs, RS
            else:
                if not cofacs or (cofacs[-1][0] < best[1][-1][0]):
                    facs_str = "*".join(f"{keyg(*_l)}^{_e}" for _l, _e in facs)
                    cofacs_str = "*".join(f"{keyg(*_l)}^{_e}" for _l, _e in cofacs)
                    logger.info(
                        f"Found relation as {facs_str} and cofactor {cofacs_str}"
                    )
                    best = facs, cofacs, RS

            if best is not None:
                max_cofac = best[1][-1][0][0]
                if max_cofac.bit_length() <= self.MAX_QBITS:
                    break

        dt = time.monotonic() - t0
        logger.info(f"Smoothing finished in {dt:.2f}s")
        facs, cofacs, RS = best
        # x0 = product(facs) * x
        log = sum(e * self.dlogs[keyg(*f)] for f, e in facs)
        # coell = (self.n - 1) // self.ell
        for (_l, _lr), _e in cofacs:
            logger.debug(f"Recurse into small prime {_l},{_lr}")
            llog = self.smalllog(_l, _lr)
            log += _e * llog

        if DEBUG_EXTRA_CHECKS:
            # Check logarithms of numerator and denominator
            Rx, Ry, Sx, Sy = RS

            facs_u, xu = self._factorg(Rx, Ry)
            facs_v, xv = self._factorg(Sx, Sy)

            Rlog = self.dlogs["CONSTANT"]
            for pg, e in facs_u + xu:
                Rlog += e * self.dlogs[keyg(*pg)]
            assert (Rx - Ry * self.z) ** self.projexp == self.logbase**Rlog

            Slog = self.dlogs["CONSTANT"]
            for pg, e in facs_v + xv:
                Slog += e * self.dlogs[keyg(*pg)]
            assert (Sx - Sy * self.z) ** self.projexp == self.logbase**Slog

        return log % self.ell

    def fg(self, x: int, y: int) -> tuple[int, int]:
        f = self.f
        w, v, u = self.g
        vf = abs(
            x * x * (f[4] * x * x + f[3] * x * y + f[2] * y * y)
            + y * y * y * (f[1] * x + f[0] * y)
        )
        vg = u * x * x + v * x * y + w * y * y
        return vf, vg

    def smalllog(self, q: int, qr: int, side: str = "g") -> int:
        """
        Compute logarithm of a "small" prime using sieving.
        """
        assert flint.fmpz(q).is_prime()
        key = f"{side}_{q}_{qr}"
        if key in self.dlogs:
            logger.info(f"Using known log({key}) = {self.dlogs[key]}")
            return self.dlogs[key]

        # If a single sieve is not enough, use larger q-lattices
        if q == qr:
            # Don't try composite lattices with a root at infinity
            q2s = [(1, 0)]
        elif q.bit_length() < 30:
            q2s = [(1, 0)] + self.g_primes[:100]
        else:
            q2s = self.g_primes[:500]
        for qq, qqr in q2s:
            if qqr == qq:
                continue  # it's complicated
            t = time.monotonic()
            if (qq, qqr) != (1, 0):
                qcomp = q * qq
                qrcomp = (qqr * q * pow(q, -1, qq) + qr * qq * pow(qq, -1, q)) % qcomp
                assert qrcomp % q == qr and qrcomp % qq == qqr, (qrcomp, q, qr, qq, qqr)
            else:
                qcomp, qrcomp = q, qr

            if qcomp.bit_length() <= 29:
                reports = self.siever.sieve(qcomp, qrcomp)
            elif qcomp.bit_length() <= 35:
                reports = self.sieverlarge.sievelarge(qcomp, qrcomp)
            else:
                reports = self.sieverhuge.sievelarge(qcomp, qrcomp)
            dt = time.monotonic() - t
            logger.debug(f"Found {len(reports)} sieve results (q={q}*{qq}) in {dt:.3}s")

            # Store: a list of factors, list of missing primes, the largest missing prime
            best: tuple[list, list, int] | None = None

            z = self.z
            n_good = 0
            for x, y in reports:
                x, y = int(x), int(y)
                if math.gcd(x, y) != 1:
                    continue
                vf, vg = self.fg(x, y)
                facf = integers.factor_smooth(
                    abs(vf) // q if side == "f" else abs(vf), self.MAX_FBITS - 5
                )

                good = True
                # The relation is (see linalg.py) is
                # product(facf) = CONSTANT * product(facg except q) * q
                facs = [
                    ("CONSTANT", -1),
                ]
                dlogf = 0
                missingf = False
                for _l, _e in facf:
                    if not flint.fmpz(_l).is_probable_prime():
                        good = False
                        break
                    _r = x * pow(y, -1, _l) % _l if y % _l else _l
                    if (
                        key := f"f_{_l}_{_r}"
                    ) not in self.dlogs and _l.bit_length() > self.MAX_FBITS:
                        good = False
                        break
                    else:
                        facs.append((key, _e))
                        if key not in self.dlogs:
                            missingf = True
                            continue
                        dlogf += _e * self.dlogs[key]

                if side == "g":
                    if vg % q != 0:
                        raise ArithmeticError(
                            f"Unexpected g(x,y) not divisible by {q=}, overflow?"
                        )

                if not good:
                    continue

                # Since f(x,y) is completely factored, the logarithm of x-yz is known
                if DEBUG_EXTRA_CHECKS and side == "g" and not missingf:
                    xy1 = (x - y * z) ** self.projexp
                    assert xy1 == self.logbase**dlogf

                # z = g(z)/u = f(z)
                facg = integers.factor(abs(vg) // q if side == "g" else abs(vg))
                for _l, _e in facg:
                    _r = x * pow(y, -1, _l) % _l if y % _l else _l
                    key = f"g_{_l}_{_r}"
                    facs.append((key, -_e))

                missing = []
                worst = 0
                for k, _ in facs:
                    if k not in self.dlogs:
                        _l, _lr = parse_key(k)
                        worst = max(worst, _l)
                        missing.append((_l, _lr, k[0]))
                if any(ideal in self.pending for ideal in missing):
                    continue
                if worst.bit_length() > self.MAX_QBITS:
                    continue

                # Both f-primes and g-primes are good
                n_good += 1
                keep = (
                    best is None
                    or not missing
                    or (best[2] > q and worst < q)
                    or (len(missing) <= len(best[1]) and max(missing) < max(best[1]))
                    # We don't like missing f-primes
                    or (
                        any(k[2] == "f" for k in best[1])
                        and all(k[2] == "g" for k in missing)
                    )
                )
                if not keep:
                    continue
                if side == "f":
                    # Flip sign
                    facs = [(_l, -_e) for _l, _e in facs]
                facs_str = "*".join(f"{_l}^{_e}" for _l, _e in facs)
                if not missing:
                    logger.debug(f"Good report {x},{y}")
                    logger.debug(f"{side}_{q}_{qr} = {facs_str}")
                    log = sum(e * self.dlogs[k] for k, e in facs) % self.ell
                    logger.info(f"log({side}_{q}_{qr}) = {log}")
                    self.dlogs[f"{side}_{q}_{qr}"] = log
                    return log
                else:
                    logger.debug(f"Good report {x},{y} worst={worst}")
                    best = facs, missing, worst

            logger.debug(f"{n_good} reports with no large missing primes")
            if best is not None:
                break

        if best is None:
            raise ValueError(f"failed to find any relation for ideal {q},{qr}")

        facs, missing, _ = best
        key = f"{side}_{q}_{qr}"
        logger.debug(f"Best relation {key} = {facs_str}")
        logger.debug(f"Best relation is missing primes {missing}")
        log = 0
        for k, e in facs:
            if k in self.dlogs:
                if DEBUG_SMALL_LOGS and k.startswith("g_"):
                    _l, _lr = parse_key(k)
                    xxx = self.smalllog(_l, _lr)
                    assert xxx == self.dlogs[k]
                log += e * self.dlogs[k]

        missing_str = "*".join(f"{_l}^{_e}" for _l, _e in facs if _l not in self.dlogs)
        logger.info(f"log({key}) = {log} + log({missing_str})")
        for k, e in facs:
            if k not in self.dlogs:
                self.pending.add((q, qr))
                assert k.startswith(("f_", "g_"))
                _l, _lr = parse_key(k)
                if _l > q:
                    logger.debug(f"Recurse into LARGER prime {k}")
                else:
                    logger.debug(f"Recurse into small prime {k}")
                log += e * self.smalllog(_l, _lr, side=k[0])

        log %= self.ell
        logger.info(f"Recursive log({key}) = {log}")
        self.dlogs[key] = log
        return log


if __name__ == "__main__":
    main()
