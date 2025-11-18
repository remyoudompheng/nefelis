"""
Sieve for the polynomial x^3-2

Notations:
    N a prime modulus
    f(x) = x^3 - 2
    g(x) = A x^2 + B x + C

We assume that B^2 - 4AC < 0 (A > 0 and C > 0)
"""

from typing import Iterator

import argparse
import concurrent.futures
from concurrent.futures import ProcessPoolExecutor
import json
import logging
import math
from multiprocessing import current_process
import os
import pathlib
import time

import flint

from nefelis.polys import estimate_size
from nefelis.sieve import Siever, gen_specialq, eta as sieve_eta
from nefelis.integers import factor_smooth, smallprimes
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
            for _l, _e in factor_smooth(vf, self.B2f):
                facf += _e * [int(_l)]
            if any(_f.bit_length() > self.B2f for _f in facf):
                continue

            facg = [q]
            for _l, _e in factor_smooth(vg, self.B2g):
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


def worker_init(
    gpu_ids: list[int] | None, g, ls, rs, threshold, I, f, ls2, rs2, threshold2
):
    global SIEVER
    if gpu_ids is None:
        gpu_idx = 0
    else:
        proc = current_process()
        gpu_idx = gpu_ids[proc._identity[-1] % len(gpu_ids)]
    SIEVER = Siever(g, ls, rs, threshold, I, f, ls2, rs2, threshold2, gpu_idx=gpu_idx)


def worker_task(args):
    q, qr = args
    t = time.monotonic()
    reports = SIEVER.sieve(q, qr)
    return q, qr, time.monotonic() - t, reports


PARAMS = [
    # bitsize, B1g, B2g, B2f, cofactor bits, I=logwidth, qmin
    (100, 5_000, 14, 14, 14, 11, 20),
    (120, 5_000, 14, 14, 15, 12, 20),
    (140, 15_000, 16, 15, 16, 13, 30),
    (160, 30_000, 17, 15, 17, 14, 100),
    (180, 40_000, 17, 16, 17, 14, 500),
    (200, 50_000, 17, 16, 17, 14, 1000),
    (220, 100_000, 18, 17, 18, 14, 3000),
    (240, 200_000, 19, 17, 19, 14, 5000),
    (260, 300_000, 19, 18, 19, 14, 10000),
    (280, 400_000, 20, 19, 20, 14, 40000),
    (300, 600_000, 20, 20, 20, 14, 70000),
    (320, 1_000_000, 22, 20, 23, 14, 150000),
    (340, 2_000_000, 23, 22, 24, 14, 300000),
    (360, 3_000_000, 23, 22, 25, 14, 500000),
    (380, 4_000_000, 24, 22, 26, 14, 800_000),
    (400, 5_000_000, 24, 23, 26, 14, 1000_000),
    # (420, 7_000_000, 25, 23, 25, 14, 4000_000)
    # 2 large primes
    (420, 5_000_000, 25, 23, 50, 14, 3000_000),
    (440, 6_000_000, 25, 24, 50, 14, 5000_000),
]

# Parameters for GPU factor
PARAMS2 = [
    # bitsize, B1f, thrF
    (140, 0, 0),
    (160, 8_000, 20),
    (180, 15_000, 25),
    (200, 30_000, 30),
    (220, 40_000, 30),
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


def get_params(N):
    return min(PARAMS, key=lambda p: abs(p[0] - N.bit_length()))[1:]


def get_params2(N):
    return min(PARAMS2, key=lambda p: abs(p[0] - N.bit_length()))[1:]


def main():
    argp = argparse.ArgumentParser()
    argp.add_argument(
        "--nogpufactor", action="store_true", help="Don't perform trial division on GPU"
    )

    def gpu_id(s: str) -> list[int]:
        return [int(x) for x in s.split(",")]

    argp.add_argument("--gpu", type=gpu_id, help="List of GPU devices to be used")
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

    B1g, B2g, B2f, COFACTOR_BITS, I, qmin = get_params(N)
    B1f, thr2 = 0, 0
    if not args.nogpufactor:
        B1f, thr2 = get_params2(N)
    logging.info(
        f"Sieving with B1={B1g / 1000:.0f}k,{B1f / 1000:.0f}k log(B2)={B2g},{B2f} q={qmin}.. {COFACTOR_BITS} cofactor bits"
    )

    if False:
        # Old hardcoded polynomial for nosm=True
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
    for _l in smallprimes(B1g):
        _rs = flint.nmod_poly(g, _l).roots()
        for _r, _ in _rs:
            ls.append(_l)
            rs.append(_r)

    # We sieve g(x) which has size log2(N)/3 + 2 log2(x) but has a known factor q
    radius = math.sqrt(qmin) * 2**I
    gsize = estimate_size(g, radius, radius)
    THRESHOLD = int(gsize) - qmin.bit_length() - COFACTOR_BITS

    ls2, rs2 = None, None
    if B1f > 0:
        ls2, rs2 = [], []
        for _l in smallprimes(B1f):
            _rs = flint.nmod_poly(f, _l).roots()
            for _r, _ in _rs:
                ls2.append(_l)
                rs2.append(_r)

    sievepool = ProcessPoolExecutor(
        1 if args.gpu is None else len(args.gpu),
        initializer=worker_init,
        initargs=(args.gpu, g, ls, rs, THRESHOLD, I, f, ls2, rs2, thr2),
    )
    factorpool = ProcessPoolExecutor(
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

    sieve_args: Iterator[tuple[int, int]] = gen_specialq(qmin, g)
    sieve_stats: list[tuple[int, int, int]] = []
    MAX_SIEVE_QUEUE = 64
    with sievepool, factorpool:
        sieve_jobs = []
        factor_jobs = []
        while True:
            while len(sieve_jobs) < MAX_SIEVE_QUEUE:
                sieve_jobs.append(sievepool.submit(worker_task, next(sieve_args)))
            # Always wait for at least 1 sieve
            concurrent.futures.wait(
                sieve_jobs, return_when=concurrent.futures.FIRST_COMPLETED
            )
            sieve_pending = []
            for sfut in sieve_jobs:
                if not sfut.done():
                    sieve_pending.append(sfut)
                    continue
                q, qr, dt, reports = sfut.result()
                fut = factorpool.submit(factorer_task, (q, reports))
                factor_jobs.append((q, qr, dt, len(reports), fut))
            # Throttle if factoring is late
            concurrent.futures.wait(
                [f for _, _, _, _, f in factor_jobs[:-MAX_SIEVE_QUEUE]]
            )

            sieve_jobs = sieve_pending
            remaining = []
            for item in factor_jobs:
                q, qr, dt, nreports, fut = item
                nrels = 0
                reschunk = fut.result()
                print(f"# q={q} r={qr}", file=relf)

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
                if total_q < 10 or elapsed > last_log + 1:
                    # Don't log too often.
                    last_log = elapsed
                    logger.info(
                        f"Sieved q={q} r={qr:<8} area {total_area / 1e9:.0f}G in {dt:.3f}s (speed {total_area / elapsed / 1e9:.3f}G/s): "
                        f"{nrels}/{nreports} relations, {gcount}/{fcount} Kg/Kf primes, total {total}"
                    )

                    sieve_stats.append((total, gcount, fcount))
                    if total and len(sieve_stats) % 30 == 10:
                        boundg = max(seeng) >> 32
                        boundf = max(seenf) >> 32
                        eta = sieve_eta(boundg, boundf, total // 10, sieve_stats)
                        logger.info(
                            f"Requiring {int(eta * total)} relations ({100 / eta:.1f}% done)"
                        )

                if total > 1.1 * (fcount + gcount):
                    break

            factor_jobs = remaining
            if len(factor_jobs) > 100:
                logger.warning(f"{len(factor_jobs)} jobs waiting for cofactorization!")

            if total > 1.1 * (fcount + gcount):
                logger.info("Enough relations")
                [j.cancel() for j in sieve_jobs]
                [tup[-1].cancel() for tup in factor_jobs]
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
