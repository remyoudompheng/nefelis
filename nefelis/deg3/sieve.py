"""
Sieve for the polynomial x^3-2

Notations:
    N a prime modulus
    f(x) = x^3 - 2
    g(x) = A x^2 + B x + C

We assume that B^2 - 4AC < 0 (A > 0 and C > 0)
"""

import argparse
import itertools
import json
import logging
import math
import multiprocessing
import os
import pathlib
import time

import flint

try:
    import pymqs
except ImportError:
    pymqs = None

from nefelis import sieve_vk
from nefelis.deg3.polyselect import polyselect, polyselect_g

logger = logging.getLogger("sieve")


class Factorer:
    def __init__(self, f, g, B2f, B2g):
        self.f = f
        self.g = g
        self.B2f = B2f
        self.B2g = B2g

    def factor_fg(self, q, chunk):
        res = []
        f = self.f
        C, B, A = self.g
        for x, y in chunk:
            x, y = int(x), int(y)
            if math.gcd(x, y) != 1:
                continue
            # value = u * x + v * y
            # print(f"{x}+{y}i", flint.fmpz(value).factor())
            # Output in Cado format
            # x,y:(factors of g(x) in hex):(factors of f(x) in hex)
            vf = abs(x * x * (f[3] * x + f[2] * y) + y * y * (f[1] * x + f[0] * y))
            vg = abs(A * x * x + B * x * y + C * y * y) // q

            facf = []
            if pymqs is not None:
                facf += pymqs.factor(int(vf))
            else:
                for _l, _e in flint.fmpz(vf).factor():
                    facf += _e * [int(_l)]
            if any(_f.bit_length() > self.B2f for _f in facf):
                continue

            facg = [q]
            if pymqs is not None:
                facg += pymqs.factor(int(vg))
            else:
                for _l, _e in flint.fmpz(vg).factor():
                    facg += _e * [int(_l)]

            if any(_l.bit_length() > self.B2g for _l in facg):
                continue

            idealf = [x * pow(y, -1, _l) % _l if y % _l else _l for _l in facf]
            idealg = [x * pow(y, -1, _l) % _l if y % _l else _l for _l in facg]
            res.append((x, y, facf, facg, idealf, idealg))

        return res


FACTORER = None


def factorer_init(f, g, B2f, B2g):
    global FACTORER
    FACTORER = Factorer(f, g, B2f, B2g)


def factorer_task(args):
    q, chunk = args
    return FACTORER.factor_fg(q, chunk)


SIEVER = None


def worker_init(g, ls, rs, threshold, I, f, ls2, rs2, threshold2):
    global SIEVER
    SIEVER = sieve_vk.Siever(g, ls, rs, threshold, I, f, ls2, rs2, threshold2)


def worker_task(args):
    q, qr = args
    t = time.monotonic()
    reports = SIEVER.sieve(q, qr)
    return q, qr, time.monotonic() - t, reports

PARAMS = [
    # bitsize, B1g, B2g, B2f, cofactor bits, I=logwidth, qmin
    (120, 10_000, 16, 14, 16, 12, 20),
    (140, 20_000, 16, 15, 17, 13, 30),
    (160, 30_000, 17, 15, 18, 14, 100),
    (180, 50_000, 17, 16, 19, 14, 500),
    (200, 100_000, 18, 16, 20, 14, 3000),
    (220, 150_000, 18, 17, 20, 14, 6000),
    (240, 200_000, 19, 17, 25, 14, 10000),
    (260, 300_000, 20, 18, 25, 14, 20000),
    (280, 400_000, 20, 19, 25, 14, 40000),
    (300, 600_000, 21, 19, 28, 14, 70000),
    (320, 1_000_000, 22, 20, 30, 14, 150000),
    (340, 2_000_000, 23, 22, 35, 14, 300000),
    (360, 3_000_000, 23, 22, 35, 14, 600000),
    (380, 4_000_000, 24, 22, 37, 14, 1000_000),
    (400, 5_000_000, 24, 23, 40, 14, 2000_000),
    # (420, 7_000_000, 25, 23, 45, 14, 4000_000)
    # 2 large primes
    (420, 5_000_000, 25, 23, 65, 14, 4000_000),
    (440, 6_000_000, 25, 24, 65, 14, 10_000_000),
]

# Parameters for GPU factor
PARAMS2 = [
    # bitsize, B1f, thrF
    (200, 0, 0),
    (240, 60_000, 30),
    (260, 60_000, 20),
    (280, 100_000, 21),
    (300, 150_000, 22),
    (320, 200_000, 22),
    (340, 300_000, 22),
    (360, 300_000, 23),
    (380, 400_000, 24),
    (400, 1_000_000, 30),
    # (420, 1_000_000, 32),
    # Fast variants (2 large primes on f side)
    (420, 600_000, 30),
    (440, 1000_000, 32),
]


# In no-SM mode polynomial is less optimal and linear algebra is faster
# so we can enlarge the factor base.
PARAMS_NOSM = [
    # bitsize, B1g, B2g, B2f, cofactor bits, I=logwidth, qmin
    (120, 10000, 16, 14, 16, 12, 20),
    (140, 30_000, 17, 16, 17, 13, 30),
    (160, 50_000, 18, 17, 18, 14, 100),
    (180, 70_000, 19, 17, 19, 14, 500),
    (200, 100_000, 19, 18, 20, 14, 3000),
    (220, 150_000, 20, 18, 20, 14, 6000),
    (240, 200_000, 20, 19, 25, 14, 10000),
    (260, 300_000, 20, 19, 25, 14, 20000),
    (280, 500_000, 21, 20, 30, 14, 40000),
    (300, 600_000, 22, 20, 30, 14, 70000),
    (320, 1_000_000, 23, 21, 30, 14, 150000),
    (340, 2_000_000, 23, 22, 35, 14, 300000),
    (360, 3_000_000, 24, 22, 37, 14, 600000),
    (380, 4_000_000, 24, 23, 45, 14, 900_000),
]


def get_params(N, nosm=False):
    return min(
        PARAMS_NOSM if nosm else PARAMS, key=lambda p: abs(p[0] - N.bit_length())
    )[1:]


def get_params2(N):
    return min(PARAMS2, key=lambda p: abs(p[0] - N.bit_length()))[1:]


def main():
    argp = argparse.ArgumentParser()
    argp.add_argument(
        "--nosm",
        action="store_true",
        help="Choose simple polynomials to avoid Schirokauer maps",
    )
    argp.add_argument(
        "--nogpufactor", action="store_true", help="Don't perform trial division on GPU"
    )
    argp.add_argument("--ncpu", type=int, help="CPU threads for factoring")
    argp.add_argument("N", type=int)
    argp.add_argument("WORKDIR")
    args = argp.parse_args()

    logging.getLogger().setLevel(level=logging.DEBUG)
    main_impl(args)


def main_impl(args):
    N = args.N
    datadir = pathlib.Path(args.WORKDIR)
    datadir.mkdir(exist_ok=True)

    ell = N // 2  # FIXME: support user ell

    assert N % 3 != 1
    assert flint.fmpz(N).is_prime()
    assert flint.fmpz(ell).is_prime()

    B1g, B2g, B2f, COFACTOR_BITS, I, qmin = get_params(N, nosm=args.nosm)
    B1f, thr2 = 0, 0
    if not args.nogpufactor:
        B1f, thr2 = get_params2(N)
    logging.info(
        f"Sieving with B1={B1g / 1000:.0f}k,{B1f / 1000:.0f}k log(B2)={B2g},{B2f} q={qmin}.. {COFACTOR_BITS} cofactor bits"
    )

    if args.nosm:
        r = pow(2, (2 * N - 1) // 3, N)
        assert (r**3 - 2) % N == 0
        f = [-2, 0, 0, 1]
        g = polyselect_g(N, f, r)
    else:
        f, g = polyselect(N)
        C, B, A = g
        for r, _ in flint.fmpz_mod_poly(f, flint.fmpz_mod_poly_ctx(N)).roots():
            if A * r * r + B * r + C == 0:
                break
        assert A * r * r + B * r + C == 0
        r = int(r)

    logger.info(f"f = {f[3]}*x^3+{f[2]}*x^2+{f[1]}*x+{f[0]}")
    C, B, A = g
    assert (A * r * r + B * r + C) % N == 0
    logger.info(f"g = {A} x² + {B} x + {C}")
    logger.info(f"Root r = {r}")

    ls, rs = [], []
    for _l in sieve_vk.smallprimes(B1g):
        _rs = flint.nmod_poly(g, _l).roots()
        for _r, _ in _rs:
            ls.append(_l)
            rs.append(_r)
    qs = []
    for _q in sieve_vk.smallprimes(10 * qmin):
        if _q >= qmin and A % _q != 0:
            for _r, _ in flint.nmod_poly(g, _q).roots():
                qs.append((int(_q), int(_r)))

    LOGAREA = qs[-1][0].bit_length() + 2 * I
    # We sieve g(x) which has size log2(N)/3 + 2 log2(x) but has a known factor q
    gsize = max(_gi.bit_length() for _gi in g)
    THRESHOLD = gsize + 2 * LOGAREA // 2 - qs[-1][0].bit_length() - COFACTOR_BITS

    ls2, rs2 = None, None
    if B1f > 0:
        ls2, rs2 = [], []
        for _l in sieve_vk.smallprimes(B1f):
            _rs = flint.nmod_poly(f, _l).roots()
            for _r, _ in _rs:
                ls2.append(_l)
                rs2.append(_r)

    sievepool = multiprocessing.Pool(
        1,
        initializer=worker_init,
        initargs=(g, ls, rs, THRESHOLD, I, f, ls2, rs2, thr2),
    )
    factorpool = multiprocessing.Pool(
        args.ncpu or os.cpu_count(),
        initializer=factorer_init,
        initargs=(f, g, B2f, B2g),
    )

    with open(datadir / "args.json", "w") as w:
        json.dump(
            {
                "n": N,
                "f": f,
                "g": g,
                "z": int(r),
            },
            w,
        )
    AREA = 2 ** (2 * I + 1)
    seen = set()
    relf = open(datadir / "relations.sieve", "w", buffering=1)
    total = 0
    duplicates = 0

    t0 = time.monotonic()
    last_log = 0
    total_area = 0
    total_q = 0
    seenf = set()
    seeng = set()

    with (sievepool, factorpool):
        for q, qr, dt, reports in sievepool.imap(worker_task, qs):
            nrels = 0
            print(f"# q={q} r={qr}", file=relf)

            batchsize = 64
            chunks = ((q, chunk) for chunk in itertools.batched(reports, batchsize))
            for reschunk in factorpool.imap_unordered(factorer_task, chunks):
                for x, y, facf, facg, idealf, idealg in reschunk:
                    # Normalize sign
                    if y < 0:
                        x, y = -x, -y
                    xy = x + (y << 32)
                    if xy in seen:
                        duplicates += 1
                        continue
                    # facg will omit q
                    str_facf = ",".join(f"{_l:x}" for _l in facf)
                    str_facg = ",".join(f"{_l:x}" for _l in facg)
                    for _l, _r in zip(facf, idealf):
                        seenf.add((_l << 32) | _r)
                    for _l, _r in zip(facg, idealg):
                        seeng.add((_l << 32) | _r)
                    seen.add(xy)
                    relf.write(f"{x},{y}:{str_facg}:{str_facf}\n")
                    nrels += 1
            if nrels:
                seeng.add((q << 32) | qr)
            print(f"# Found {nrels} relations for {q=} (time {dt:.3f}s)", file=relf)
            total_q += 1
            total += nrels
            total_area += AREA
            elapsed = time.monotonic() - t0
            gcount = len(seeng)
            fcount = len(seenf)
            if elapsed < 2 or elapsed > last_log + 1:
                # Don't log too often.
                last_log = elapsed
                logger.info(
                    f"Sieved q={q} r={qr:<8} area {total_area / 1e9:.0f}G in {dt:.3f}s (speed {total_area / elapsed / 1e9:.3f}G/s): "
                    f"{nrels}/{len(reports)} relations, {gcount}/{fcount} Kg/Kf primes, total {total}"
                )
            if total > 1.1 * (fcount + gcount):
                logging.info("Enough relations")
                break

    # CADO-NFS requires that the relation file ends with \n
    # Print statistics in CADO-compatible format
    # elapsed_per_q = elapsed / total_q
    rels_per_q = total / total_q
    rels_per_t = total / elapsed
    relf.write(f"# Total elapsed time {elapsed:.3f}s\n")
    relf.write(
        f"# Total {total} reports [{1 / rels_per_t:.3g}s/r, {rels_per_q:.3f}r/sq] in {elapsed:.2f} elapsed s\n"
    )
    logger.info(f"{total} relations {duplicates} duplicates in {elapsed:.3f}s")
    relf.close()


if __name__ == "__main__":
    main()
