"""
Sieve for the Gaussian Integer Method

Notations:
    N a prime modulus
    f(x) = x^2 + D
    g(x) = u x + v
where D is a small positive integer
      -v/u is a square of root -D modulo N
      u, v have size ~sqrt(N)

The GPU takes care of the lattice sieve for a given special q:
    - the sieve region is split into linear intervals of size 16384
    - for each prime in the (rational) factor base, produce sieve hits in a large array
    - perform partial radix sort to match sieve hits with intervals
    - sieve each small interval and produce reports

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
from nefelis.deg2.polyselect import polyselect

logger = logging.getLogger("sieve")


class Factorer:
    def __init__(self, f, g, B2f, B2g):
        self.f = f
        self.g = g
        self.B2f = B2f
        self.B2g = B2g

    def factor_fg(self, q, chunk):
        res = []
        C, B, A = self.f
        v, u = self.g
        for x, y in chunk:
            x, y = int(x), int(y)
            if math.gcd(x, y) != 1:
                continue
            # value = u * x + v * y
            # print(f"{x}+{y}i", flint.fmpz(value).factor())
            # Output in Cado format
            # x,y:(factors of g(x) in hex):(factors of f(x) in hex)
            vf = abs(A * x * x + B * x * y + C * y * y)
            vg = abs((u * x + v * y) // q)

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


def worker_init(*args):
    global SIEVER
    SIEVER = sieve_vk.Siever(*args)


def worker_task(args):
    q, qr = args
    t = time.monotonic()
    reports = SIEVER.sieve(q, qr)
    return q, time.monotonic() - t, reports


PARAMS = [
    # bitsize, B1g, B2g, B2f, cofactor bits, I=logwidth, qmin
    (120, 4000, 13, 12, 20, 13, 10),
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

    ell = N // 2  # FIXME: support user ell

    assert flint.fmpz(N).is_prime()
    assert flint.fmpz(ell).is_prime()

    B1g, B2g, B2f, COFACTOR_BITS, I, qmin = get_params(N)
    B1f, thr2 = 0, 0
    if not args.nogpufactor:
        B1f, thr2 = get_params2(N)
    logger.info(
        f"Sieving with B1={B1g / 1000:.0f}k,{B1f / 1000:.0f}k log(B2)={B2g},{B2f} q={qmin}.. {COFACTOR_BITS} cofactor bits"
    )

    f, g = polyselect(N)
    C, B, A = f
    v, u = g

    r = v * pow(-u, -1, N) % N
    assert (u * r + v) % N == 0
    assert (A * r * r + B * r + C) % N == 0
    logger.info(f"f = {A} x^2 + {B} x + {C}")
    logger.info(f"{u = } size {u.bit_length()}")
    logger.info(f"{v = } size {v.bit_length()}")

    ls = sieve_vk.smallprimes(B1g)
    rs = [(-v * pow(u, -1, l)) % l if u % l else l for l in ls]
    qs = [_q for _q in sieve_vk.smallprimes(10 * qmin) if _q >= qmin and u % _q != 0]
    qrs = [-v * pow(u, -1, q) % q for q in qs]

    f = [C, B, A]
    g = [int(v), int(u)]
    ls2, rs2 = None, None
    if B1f:
        ls2, rs2 = [], []
        for _l in sieve_vk.smallprimes(B1f):
            _rs = flint.nmod_poly(f, _l).roots()
            for _r, _ in _rs:
                ls2.append(_l)
                rs2.append(_r)
            if f[2] % _l == 0:
                ls2.append(_l)
                rs2.append(_r)

    LOGAREA = qs[-1].bit_length() + 2 * I
    THRESHOLD = N.bit_length() // 2 + LOGAREA // 2 - qs[-1].bit_length() - COFACTOR_BITS

    sievepool = multiprocessing.Pool(
        1,
        initializer=worker_init,
        initargs=([v, u], ls, rs, THRESHOLD, I, f, ls2, rs2, thr2),
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
        for q, dt, reports in sievepool.imap(worker_task, list(zip(qs, qrs))):
            nrels = 0
            print(f"# q={q}", file=relf)

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
                    seenf.update(facf)
                    seeng.update(facg)
                    seen.add(xy)
                    relf.write(f"{x},{y}:{str_facg}:{str_facf}\n")
                    nrels += 1
            print(f"# Found {nrels} relations for {q=} (time {dt:.3f}s)", file=relf)
            total_q += 1
            total += nrels
            total_area += AREA
            elapsed = time.monotonic() - t0
            Qcount = len(seenf | seeng)
            Kcount = len(seenf)
            if elapsed < 2 or elapsed > last_log + 1:
                # Don't log too often.
                last_log = elapsed
                logger.info(
                    f"Sieved q={q:<8} area {total_area / 1e9:.0f}G in {dt:.3f}s (speed {total_area / elapsed / 1e9:.3f}G/s): "
                    f"{nrels}/{len(reports)} relations, {Qcount}/{Kcount} Q/K primes, total {total}"
                )
            if total > 1.1 * Qcount + Kcount:
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
