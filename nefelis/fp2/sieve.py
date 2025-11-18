"""
Sieve for the GF(p²) discrete logarithm

Currently only degree 2 polynomials are implemented using the
conjugate method.

The method is as follows:

F = (x^2 + A x + B)(x^2 + Ā x + B) where A, Ā are conjugate albegraic numbers
  = x^4 + x^3 Tr(A) + (2B+Norm(A))x^2 + x Tr(A) + B²
G = x^2 + A x + B
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

from nefelis import integers
from nefelis.sieve import Siever
from nefelis.fp2 import polyselect

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
            vf = abs(
                x * x * (f[4] * x * x + f[3] * x * y + f[2] * y * y)
                + y * y * y * (f[1] * x + f[0] * y)
            )
            vg = abs(A * x * x + B * x * y + C * y * y) // q

            facf = []
            for _l, _e in integers.factor_smooth(vf, self.B2f):
                facf += _e * [int(_l)]
            if any(_f.bit_length() > self.B2f for _f in facf):
                continue

            facg = [q]
            for _l, _e in integers.factor_smooth(vg, self.B2g):
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


def worker_init(gpu_ids: list[int] | None, *args):
    global SIEVER
    if gpu_ids is None:
        gpu_idx = 0
    else:
        proc = current_process()
        gpu_idx = gpu_ids[proc._identity[-1] % len(gpu_ids)]
    SIEVER = Siever(*args, gpu_idx=gpu_idx)


def worker_task(args):
    q, qr = args
    t = time.monotonic()
    reports = SIEVER.sieve(q, qr)
    return q, qr, time.monotonic() - t, reports


# The parameters are similar to deg3 for 2x larger N
PARAMS = [
    # bitsize, B1g, B2g, B2f, cofactor bits, I=logwidth, qmin
    (60, 10_000, 17, 16, 10, 13, 5_000),
    (80, 30_000, 18, 17, 15, 13, 5_000),
    (100, 50_000, 19, 18, 19, 14, 15_000),
    (120, 100_000, 20, 18, 20, 14, 15_000),
    (140, 200_000, 20, 19, 20, 14, 15_000),
    (150, 600_000, 21, 19, 20, 14, 15_000),
    (160, 1000_000, 21, 20, 20, 14, 30_000),
    (200, 3000_000, 23, 21, 20, 14, 100_000),
    # 2 large primes
    (240, 5000_000, 24, 22, 45, 14, 300_000),
]

# Parameters for GPU factor
PARAMS2 = [
    # bitsize, B1f, thrF
    (80, 40_000, 30),
    (140, 100_000, 30),
    (160, 200_000, 30),
    (200, 400_000, 30),
    (240, 600_000, 32),
]


def get_params(N):
    return min(PARAMS, key=lambda p: abs(p[0] - N.bit_length()))[1:]


def get_params2(N):
    return min(PARAMS2, key=lambda p: abs(p[0] - N.bit_length()))[1:]


def conway_poly(p):
    """
    Compute a Conway polynomial for field GF(p^2)

    >>> conway_poly(2)
    [1, -1, 1]
    >>> conway_poly(63361)
    [37, -17, 1]
    >>> conway_poly(65537)
    [3, -1, 1]
    >>> conway_poly(94009)
    [13, -13, 1]
    """
    # The root of the Conway polynomial must map to the root of the
    # Conway polynomial of degree 1 under the norm map.
    l0 = integers.factor(p - 1)
    for c in range(1, p):
        if any(pow(c, (p - 1) // l, p) == 1 for l, _ in l0):
            continue
        logging.debug(f"Conway: {c} is primitive modulo p")
        break
    # The leading coefficient is always 1
    l1 = integers.factor(p + 1)
    Fp2 = flint.fq_default_ctx(p, 2)
    Fp2X = flint.fq_default_poly_ctx(Fp2)
    for b in range(p):
        r = Fp2X([c, -b, 1]).roots()[0][0]
        if r ** (p - 1) == 1:
            continue
        if any(r ** ((p * p - 1) // l) == 1 for l, _ in l1):
            continue
        logging.debug(f"Conway: x²-{b}*x+{c} is the Conway polynomial")
        break

    return [c, -b, 1]


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


# FIXME: manage to remove this flag
DEBUG_IGNORE_CONJUGATES = True


def main_impl(args):
    N = args.N
    datadir = pathlib.Path(args.WORKDIR)
    datadir.mkdir(exist_ok=True)

    assert flint.fmpz(N).is_prime()

    B1g, B2g, B2f, COFACTOR_BITS, I, qmin = get_params(N)
    B1f, thr2 = 0, 0
    if not args.nogpufactor:
        B1f, thr2 = get_params2(N)
    logging.info(
        f"Sieving with B1={B1g / 1000:.0f}k,{B1f / 1000:.0f}k log(B2)={B2g},{B2f} q={qmin}.. {COFACTOR_BITS} cofactor bits"
    )

    f, g, D, gj = polyselect.polyselect(N)
    # Check roots
    ZnX = flint.fmpz_mod_poly_ctx(N)
    fn = flint.fmpz_mod_poly(f, ZnX)
    gn = flint.fmpz_mod_poly(g, ZnX)
    assert len(gn.roots()) == 0
    assert fn % gn == 0

    C, B, A = g
    assert B * B - 4 * A * C < 0

    conway = conway_poly(N)
    Fp2 = flint.fq_default_ctx(N, 2, var="i", modulus=ZnX(conway))
    Fp2X = flint.fq_default_poly_ctx(Fp2)

    logger.info(f"f = {f[4]}*x^4+{f[3]}*x^3+{f[2]}*x^2+{f[1]}*x+{f[0]}")
    logger.info(f"g = {A}*x^2 + {B}*x + {C}")
    roots_g = Fp2X(g).roots()
    z = roots_g[0][0]
    if roots_g[1][0].to_list()[1] < z.to_list()[1]:
        z = roots_g[1][0]
    logger.info(f"GF(p^2) is represented by GF(p)[x] / {flint.fmpz_poly(conway)}")
    logger.info(f"Root of f,g is z = {z}")

    ls, rs = [], []
    for _l in integers.smallprimes(B1g):
        _rs = flint.nmod_poly(g, _l).roots()
        for _r, _ in _rs:
            ls.append(_l)
            rs.append(_r)

    # NOTE: due to the Galois symmetry of K(g),
    # we don't sieve both an ideal and its conjugate
    # (they will give "mirrored" relations).
    qs = []
    for _q in integers.smallprimes(10 * qmin):
        if _q >= qmin and A % _q != 0:
            for _r, _ in flint.nmod_poly(g, _q).roots():
                qs.append((int(_q), int(_r)))
                break  # Only one root per q

    LOGAREA = qs[-1][0].bit_length() + 2 * I
    # We sieve g(x) which has size log2(N)/3 + 2 log2(x) but has a known factor q
    gsize = max(_gi.bit_length() for _gi in g)
    THRESHOLD = gsize + 2 * LOGAREA // 2 - qs[-1][0].bit_length() - COFACTOR_BITS

    ls2, rs2 = None, None
    if B1f > 0:
        ls2, rs2 = [], []
        for _l in integers.smallprimes(B1f):
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
                "D": D,
                "gj": gj,
                "conway": conway,
                "z": [int(_zi) for _zi in z.to_list()],
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

    # Remember seen ideals: due to the conjugacy symmetry
    # we only record 1 ideal per prime norm for g
    # we only record 1 ideal per prime norm for f
    # (actually, 1/4 of primes have 2 ideals up to conjugacy and 1/4 primes have 0)
    seenf = set()
    seeng = set()

    sieve_args: Iterator[tuple[int, int]] = iter(qs)
    MAX_SIEVE_QUEUE = 64
    enough_rels = False
    with sievepool, factorpool:
        sieve_jobs = []
        factor_jobs = []
        while not enough_rels:
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
            sieve_jobs = sieve_pending

            # Throttle if factoring is late
            factor_pending = [t[-1] for t in factor_jobs]
            if len(factor_pending) > MAX_SIEVE_QUEUE:
                _c = len(factor_pending)
                for _ in concurrent.futures.as_completed(factor_pending):
                    _c -= 1
                    if _c < MAX_SIEVE_QUEUE:
                        break
            else:
                concurrent.futures.wait(
                    factor_pending, return_when=concurrent.futures.FIRST_COMPLETED
                )

            remaining = []
            for item in factor_jobs:
                q, qr, dt, nreports, fut = item
                if not fut.done():
                    remaining.append(item)
                    continue
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
                    seenf.update(facf)
                    seeng.update(facg)
                    seen.add(xy)
                    relf.write(f"{x},{y}:{str_facg}:{str_facf}\n")
                    nrels += 1
                if nrels:
                    seeng.add(q)
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
                # Correct fcount because there are 4/3 primes per norm on average
                if DEBUG_IGNORE_CONJUGATES:
                    if total > 1.2 * (2.66 * fcount + 2 * gcount):
                        enough_rels = True
                        break
                else:
                    if total > 1.7 * (1.33 * fcount + gcount):
                        enough_rels = True
                        break
            factor_jobs = remaining

        logger.info("Enough relations")
        [j.cancel() for j in sieve_jobs]
        [tup[-1].cancel() for tup in factor_jobs]

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
