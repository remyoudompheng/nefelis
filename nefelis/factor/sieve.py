"""
Sieve for factoring

The sieve is done on the algebraic side (polynomial f).
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

from nefelis.sieve import Siever
from nefelis.integers import factor, smallprimes
from nefelis.factor.polyselect import polyselect

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
        df = len(f) - 1
        v, u = self.g
        for x, y in chunk:
            x, y = int(x), int(y)
            if math.gcd(x, y) != 1:
                continue
            # value = u * x + v * y
            # print(f"{x}+{y}i", flint.fmpz(value).factor())
            # Output in Cado format
            # x,y:(factors of g(x) in hex):(factors of f(x) in hex)
            vf = abs(sum(fi * x**i * y ** (df - i) for i, fi in enumerate(f)))
            assert vf % q == 0
            vf //= q
            vg = abs(u * x + v * y)

            facf = [q]
            for _l, _e in factor(vf):
                facf += _e * [int(_l)]
            if any(_f.bit_length() > self.B2f for _f in facf):
                continue

            facg = []
            for _l, _e in factor(vg):
                facg += _e * [int(_l)]
            if any(_l.bit_length() > self.B2g for _l in facg):
                continue

            idealf = [x * pow(y, -1, _l) % _l if y % _l else _l for _l in facf]
            res.append((x, y, facf, facg, idealf))

        return res


FACTORER = None


def factorer_init(f, g, B2f, B2g):
    global FACTORER
    FACTORER = Factorer(f, g, B2f, B2g)


def factorer_task(args):
    q, chunk = args
    return FACTORER.factor_fg(q, chunk)


SIEVER = None


def worker_init(*args):
    global SIEVER
    SIEVER = Siever(*args)


def worker_task(args):
    q, qr = args
    t = time.monotonic()
    reports = SIEVER.sieve(q, qr)
    return q, qr, time.monotonic() - t, reports


PARAMS = [
    # bitsize, B1f, B2f, B2g, cofactor bits, I=logwidth, qmin
    (60, 500, 11, 10, 5, 11, 10),
    (80, 1000, 12, 11, 10, 12, 10),
    (100, 3000, 13, 12, 10, 13, 20),
    (120, 10000, 14, 12, 20, 13, 30),
    (140, 20000, 15, 14, 20, 13, 20),
    (160, 30000, 16, 15, 20, 13, 100),
    (180, 50000, 17, 16, 20, 13, 1500),
    (200, 100000, 18, 17, 25, 14, 1000),
    (220, 150000, 19, 18, 30, 14, 6000),
    (240, 300000, 21, 19, 30, 14, 20000),
    (260, 500000, 21, 20, 30, 14, 50000),
    (280, 1000_000, 21, 20, 35, 14, 100000),
    (300, 1500_000, 22, 21, 40, 14, 250000),
    # (320, 2000_000, 23, 21, 45, 14, 500000),
    # (340, 3000_000, 23, 22, 55, 14, 1000000),
    # 2 large primes
    # (300, 1000_000, 22, 21, 55, 14, 250000),
    (320, 1500_000, 23, 21, 60, 14, 700000),
    (340, 2500_000, 23, 22, 65, 14, 1500000),
]

# Parameters for trial division
PARAMS2 = [
    # bitsize, B1f, thrF
    (200, 0, 0),
    (220, 100_000, 22),
    (240, 150_000, 24),
    (260, 300_000, 24),
    (280, 400_000, 26),
    (300, 500_000, 28),
    (320, 600_000, 30),
    (340, 800_000, 30),
    # (360, 300_000, 23),
    # (380, 400_000, 24),
    # (400, 1_000_000, 30),
]


def get_params(N):
    return min(PARAMS, key=lambda p: abs(p[0] - N.bit_length()))[1:]


def get_params2(N):
    return min(PARAMS2, key=lambda p: abs(p[0] - N.bit_length()))[1:]


def main():
    argp = argparse.ArgumentParser()
    argp.add_argument("--ncpu", type=int, help="CPU threads for factoring")
    argp.add_argument(
        "--nogpufactor", action="store_true", help="Don't perform trial division on GPU"
    )
    argp.add_argument("N", type=int)
    argp.add_argument("WORKDIR")
    args = argp.parse_args()

    logging.getLogger().setLevel(level=logging.DEBUG)
    main_impl(args)


def main_impl(args):
    N = args.N
    datadir = pathlib.Path(args.WORKDIR)
    datadir.mkdir(exist_ok=True)

    B1f, B2f, B2g, COFACTOR_BITS, I, qmin = get_params(N)
    B1g, thr2 = 0, 0
    if not args.nogpufactor:
        B1g, thr2 = get_params2(N)
    logger.info(
        f"Sieving with B1={B1f / 1000:.0f}k,{B1g / 1000:.0f}k log(B2)={B2f},{B2g} q={qmin}.. {COFACTOR_BITS} cofactor bits"
    )

    if N.bit_length() < 130:
        degree = 3
    else:
        degree = 4
    f, g = polyselect(N, degree)
    v, u = g

    r = v * pow(-u, -1, N) % N
    assert (u * r + v) % N == 0
    assert sum(fi * r**i for i, fi in enumerate(f)) % N == 0
    fpoly = flint.fmpz_poly(f)
    logger.info(f"f = {fpoly}")
    logger.info(f"{u = } size {u.bit_length()}")
    logger.info(f"{v = } size {v.bit_length()}")

    # For factoring, we sieve using f-primes
    ls, rs = [], []
    for _l in smallprimes(B1f):
        _rs = flint.nmod_poly(f, _l).roots()
        for _r, _ in _rs:
            ls.append(_l)
            rs.append(int(_r))

    qs, qrs = [], []
    for _l, _r in zip(ls, rs):
        if qmin <= _l <= 10 * qmin:
            qs.append(_l)
            qrs.append(_r)

    ls2, rs2 = None, None
    if B1g:
        ls2, rs2 = [], []
        for _l in smallprimes(B1g):
            _r = (-v * pow(u, -1, _l)) % _l if u % _l else _l
            ls2.append(_l)
            rs2.append(_r)

    degf = len(f) - 1
    sizef = max(fi.bit_length() for fi in f)
    LOGAREA = qs[-1].bit_length() + 2 * I
    THRESHOLD = sizef + degf * (LOGAREA // 2) - qs[-1].bit_length() - COFACTOR_BITS

    sievepool = multiprocessing.Pool(
        1,
        initializer=worker_init,
        initargs=(f, ls, rs, THRESHOLD, I, g, ls2, rs2, thr2),
    )
    factorpool = multiprocessing.Pool(
        args.ncpu or os.cpu_count(),
        initializer=factorer_init,
        initargs=(f, g, B2f, B2g),
    )

    with open(datadir / "args.json", "w") as w:
        z = int((-v * pow(u, -1, N)) % N)
        json.dump(
            {
                "n": N,
                "f": f,
                "g": g,
                "z": z,
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

    with sievepool, factorpool:
        for q, qr, dt, reports in sievepool.imap(worker_task, list(zip(qs, qrs))):
            nrels = 0
            print(f"# q={q} r={qr}", file=relf)

            batchsize = 64
            chunks = ((q, chunk) for chunk in itertools.batched(reports, batchsize))
            for reschunk in factorpool.imap_unordered(factorer_task, chunks):
                for x, y, facf, facg, idealf in reschunk:
                    # Normalize sign
                    if y < 0:
                        x, y = -x, -y
                    xy = x + (y << 32)
                    if xy in seen:
                        duplicates += 1
                        continue
                    str_facf = ",".join(f"{_l:x}" for _l in facf)
                    str_facg = ",".join(f"{_l:x}" for _l in facg)
                    for _l, _r in zip(facf, idealf):
                        seenf.add((_l << 32) | _r)
                    seeng.update(facg)
                    seen.add(xy)
                    relf.write(f"{x},{y}:{str_facg}:{str_facf}\n")
                    nrels += 1
            print(f"# Found {nrels} relations for {q=} (time {dt:.3f}s)", file=relf)
            total_q += 1
            total += nrels
            total_area += AREA
            elapsed = time.monotonic() - t0
            fcount = len(seenf)
            gcount = len(seeng)
            if elapsed < 2 or elapsed > last_log + 1:
                # Don't log too often.
                last_log = elapsed
                logger.info(
                    f"Sieved q={q:<8} r={qr:<8} area {total_area / 1e9:.0f}G in {dt:.3f}s (speed {total_area / elapsed / 1e9:.3f}G/s): "
                    f"{nrels}/{len(reports)} relations, {fcount}/{gcount} K/Q primes, total {total}"
                )
            if total > 1.1 * (fcount + gcount):
                logger.info("Enough relations")
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
