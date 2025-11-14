"""
Sieve for factoring

The sieve is done on the algebraic side (polynomial f).
"""

from typing import Iterator

import argparse
import concurrent.futures
from concurrent.futures import ProcessPoolExecutor
import json
import logging
import math
import os
import pathlib
import time

import flint
import numpy as np

from nefelis.skewpoly import skewness
from nefelis.sieve import eta as sieve_eta, LineSiever2
from nefelis.integers import factor_smooth, smallprimes

from nefelis.factor.polyselect import polyselect
from nefelis.factor.polyselect_snfs import snfs_select

logger = logging.getLogger("sieve")


class Factorer:
    def __init__(self, f, g, B2f, B2g):
        self.f = f
        self.g = g
        self.B2f = B2f
        self.B2g = B2g

    def factor_fg(self, q, qmat, chunk):
        res = []
        f, g = self.f, self.g
        df = len(f) - 1
        dg = len(g) - 1
        a, b, c, d = int(qmat[0, 0]), int(qmat[0, 1]), int(qmat[1, 0]), int(qmat[1, 1])
        assert chunk.shape[1] == 2
        for idx in range(chunk.shape[0]):
            x, y = int(chunk[idx, 0]), int(chunk[idx, 1])
            x, y = a * x + b * y, c * x + d * y
            if math.gcd(x, y) != 1:
                continue
            # value = u * x + v * y
            # print(f"{x}+{y}i", flint.fmpz(value).factor())
            # Output in Cado format
            # x,y:(factors of g(x) in hex):(factors of f(x) in hex)
            vf = abs(sum(fi * x**i * y ** (df - i) for i, fi in enumerate(f)))
            assert vf % q == 0
            vf //= q
            vg = abs(sum(gi * x**i * y ** (dg - i) for i, gi in enumerate(g)))

            facg = []
            for _l, _e in factor_smooth(vg, self.B2g):
                facg += _e * [int(_l)]
            if any(_l.bit_length() > self.B2g for _l in facg):
                continue

            # TODO: move f first when using 2 large primes?
            facf = [q]
            for _l, _e in factor_smooth(vf, self.B2f):
                facf += _e * [int(_l)]
            if any(_f.bit_length() > self.B2f for _f in facf):
                continue

            if dg == 1:
                ideals = [x * pow(y, -1, _l) % _l if y % _l else _l for _l in facf]
            else:
                ideals = [x * pow(y, -1, _l) % _l if y % _l else _l for _l in facg]
            res.append((x, y, facf, facg, ideals))

        return res


FACTORER = None


def factorer_init(f, g, B2f, B2g):
    global FACTORER
    FACTORER = Factorer(f, g, B2f, B2g)


def factorer_task(args):
    q, qmat, chunk = args
    return FACTORER.factor_fg(q, qmat, chunk)


SIEVER = None


def worker_init(*args):
    global SIEVER
    SIEVER = LineSiever2(*args)


def worker_task(args):
    q, qr = args
    t = time.monotonic()
    qmat, reports = SIEVER.sieve(q, qr)
    return q, qr, time.monotonic() - t, qmat, reports


# Parameters for high CPU or low GPU (gpu.cores/cpu.cores < 2)
PARAMS = [
    # bitsize, degree, B1f, B2f, B2g, cofactor bits, cofactor2 bits, A=logarea, qmin
    # FIXME: calibrate parameters
    (60, 3, 2000, 1000, 12, 11, 5, 5, 23, 20),
    (80, 3, 2000, 1000, 12, 11, 10, 10, 24, 20),
    (100, 3, 5000, 3000, 13, 12, 13, 12, 24, 50),
    (120, 3, 5000, 5000, 13, 13, 13, 13, 24, 100),
    (140, 3, 20000, 10000, 15, 14, 15, 14, 25, 150),
    (160, 3, 30000, 20000, 16, 15, 16, 15, 25, 300),
    (180, 3, 50000, 50000, 17, 17, 17, 17, 25, 500),
    (200, 3, 80000, 80000, 18, 18, 19, 19, 26, 500),
    # (220, 3, 150000, 150000, 19, 19, 20, 20, 26, 2000),
    # (240, 3, 300000, 150000, 20, 19, 25, 25, 26, 5000),
    # Degree 4 (skew is usually larger than q)
    # (200, 4, 80_000, 50_000, 18, 17, 18, 17, 26, 500),
    (220, 4, 200_000, 100_000, 20, 18, 20, 19, 26, 2000),
    (240, 4, 300_000, 150_000, 20, 19, 21, 20, 26, 5000),
    (260, 4, 500_000, 250_000, 21, 20, 23, 22, 26, 8000),  # 1 large prime + 2 bits
    (280, 4, 800_000, 400_000, 22, 21, 24, 23, 26, 10000),
    (300, 4, 1500_000, 800_000, 23, 22, 26, 25, 26, 20000),  # 1 large prime + 3 bits
    (320, 4, 2500_000, 1500_000, 24, 23, 28, 27, 27, 20000),  # 1 large prime + 4 bits
    # TODO
    (340, 4, 3500_000, 2000_000, 25, 24, 29, 28, 27, 40000),
    (360, 4, 5000_000, 3000_000, 26, 25, 31, 30, 28, 40000),  # 1 large prime + 5 bits
    # (380, 4, 6000_000, 4000_000, 27, 26, 32, 31, 28, 100000),
    # 2 large primes
    (380, 4, 4000_000, 4000_000, 27, 26, 52, 28, 29, 50000),  # 2/1 large primes
    (400, 4, 5000_000, 5000_000, 28, 27, 55, 29, 29, 100000),  # 2 large primes
    (420, 4, 7000_000, 6000_000, 29, 27, 56, 30, 29, 200000),  # 2 large primes
    (440, 5, 8000_000, 7000_000, 29, 28, 57, 31, 29, 300000),
    # Degree 5 polynomials
    # (300, 5, 2000_000, 800_000, 25, 24, 40, 25, 28, 10000),
    # (320, 5, 2000_000, 1000_000, 26, 24, 45, 30, 28, 10000),
    # (380, 5, 5000_000, 2000_000, 27, 26, 50, 35, 29, 50000),
    # (400, 5, 7000_000, 3000_000, 28, 27, 55, 35, 29, 100000),
    # (420, 5, 8000_000, 4000_000, 29, 27, 55, 35, 29, 200000),
]

# These parameters are optimized for "Cunningham" numbers
# (rational multiples of b^n+a for small a,b).
# They may need adjustments for other SNFS-compatible numbers (e.g. fibonacci(a)+b)
PARAMS_SNFS = [
    # bitsize, degree, B1f, B2f, B2g, cofactor bits, cofactor2 bits, A=logarea, qmin
    (100, 3, 1_000, 1_000, 12, 11, 5, 5, 23, 20),
    (125, 3, 2_000, 2_000, 14, 14, 10, 10, 24, 50),
    (150, 3, 5_000, 5_000, 15, 15, 15, 15, 25, 50),
    (175, 3, 15_000, 10_000, 17, 16, 16, 16, 26, 50),
    (200, 3, 20_000, 20_000, 17, 17, 17, 17, 26, 100),
    (225, 3, 60_000, 40_000, 18, 17, 18, 17, 26, 200),
    (250, 3, 100_000, 80_000, 19, 18, 18, 18, 26, 400),
    (300, 4, 250_000, 150_000, 20, 20, 20, 20, 27, 1000),
    (350, 4, 400_000, 200_000, 21, 20, 22, 21, 28, 2000),
    (400, 4, 1000_000, 600_000, 23, 22, 25, 24, 28, 5000),
    (450, 5, 2500_000, 1500_000, 24, 23, 27, 26, 28, 10000),
    # 2 large primes
    (500, 5, 4000_000, 3000_000, 26, 25, 50, 25, 29, 20000),
    (550, 5, 6000_000, 5000_000, 28, 27, 52, 28, 29, 50000),  # 2 large primes
    (600, 5, 9000_000, 7000_000, 29, 28, 55, 30, 29, 100000),  # 2 large primes
    (650, 5, 10_000_000, 9_000_000, 30, 29, 58, 32, 29, 200000),  # 2 large primes
    # FIXME: implement degree 6 for sizes above 750 bits
]


def get_params(N, snfs=False):
    if snfs:
        return min(PARAMS_SNFS, key=lambda p: abs(p[0] - N.bit_length()))[1:]
    else:
        return min(PARAMS, key=lambda p: abs(p[0] - N.bit_length()))[1:]


def main():
    argp = argparse.ArgumentParser()
    argp.add_argument("--snfs", action="store_true", help="Use Special NFS")
    argp.add_argument(
        "--parambias", type=int, default=0, help="Use params from x bits larger numbers"
    )
    argp.add_argument("--ncpu", type=int, help="CPU threads for factoring")
    argp.add_argument(
        "--nogpufactor", action="store_true", help="Don't perform trial division on GPU"
    )
    argp.add_argument("N", type=int)
    argp.add_argument("WORKDIR")
    args = argp.parse_args()

    logging.getLogger().setLevel(level=logging.DEBUG)
    main_impl(args)


def estimate_size(f, W, H):
    logs = []
    df = len(f) - 1
    for x in range(-10, 11):
        fval = sum(fi * (x * W / 10.0) ** i * H ** (df - i) for i, fi in enumerate(f))
        logs.append(math.log2(abs(fval)))
    return int(sum(logs) / len(logs))


def main_impl(args):
    N = args.N
    datadir = pathlib.Path(args.WORKDIR)
    datadir.mkdir(exist_ok=True)

    Nparams = N
    if bias := args.parambias:
        if bias > 0:
            Nparams <<= bias
        else:
            Nparams >>= bias
    degree, B1f, B1g, B2f, B2g, COFACTOR_BITS, COFACTOR_BITS2, logA, qmin = get_params(
        Nparams, snfs=args.snfs
    )
    logger.info(
        f"Sieving with B1={B1f / 1000:.0f}k,{B1g / 1000:.0f}k log(B2)={B2f},{B2g} q={qmin}.. A={logA} {COFACTOR_BITS}/{COFACTOR_BITS2} cofactor bits"
    )

    if args.snfs:
        radius = 0.5 * (qmin.bit_length() + 1 + logA)
        f, g = snfs_select(N, radius)
        degree = len(f) - 1
        skew = 1.0  # skewness(f)
    else:
        f, g = polyselect(N, degree)
        skew = skewness(f)

    v, u = g
    r = v * pow(-u, -1, N) % N
    assert (u * r + v) % N == 0
    assert sum(fi * r**i for i, fi in enumerate(f)) % N == 0
    fpoly = flint.fmpz_poly(f)
    gpoly = flint.fmpz_poly(g)
    logger.info(f"f = {fpoly}")
    logger.info(f"g = {gpoly}")

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

    # For factoring, we sieve using f-primes (g-primes for SNFS)
    # The rational polynomial is LARGER when using SNFS
    sieve_rational = False
    if args.snfs:
        qW, qH = math.sqrt(qmin << logA), math.sqrt(qmin << logA)
        sizef = estimate_size(f, qW, qH)
        sizeg = estimate_size(g, qW, qH)
        if sizeg > sizef + qmin.bit_length() // 2:
            f, g = g, f
            sizef, sizeg = sizeg, sizef
            sieve_rational = True
            logger.info("Using special-q with polynomial g")

    # Now polynomial f is using special-q
    ls, rs = [], []
    for _l in smallprimes(B1f):
        _rs = flint.nmod_poly(f, _l).roots()
        for _r, _ in _rs:
            ls.append(_l)
            rs.append(int(_r))
        if f[-1] % _l == 0:
            ls.append(_l)
            rs.append(_l)

    ls2, rs2 = [], []
    for _l in smallprimes(B1g):
        _rs = flint.nmod_poly(g, _l).roots()
        for _r, _ in _rs:
            ls2.append(_l)
            rs2.append(int(_r))
        if g[-1] % _l == 0:
            ls2.append(_l)
            rs2.append(_l)

    qs, qrs = [], []
    for _l, _r in zip(ls, rs):
        if qmin <= _l <= 30 * qmin:
            qs.append(_l)
            qrs.append(_r)

    # Choose rectangle size for area 2 ^ (2I + 1)
    if args.snfs:
        skew2 = skew
    else:
        skew2 = skew / (1.5 * qmin)
    H = int(math.sqrt(2 ** (logA - 1) / skew2))
    W = (
        max(1, int(round(2 ** (logA - 1) / H / LineSiever2.SEGMENT_SIZE)))
    ) * LineSiever2.SEGMENT_SIZE
    logger.info(
        f"Sieve rectangle size {2 * W >> 10}k x {H} (skewness: {W / H:.3g}, area: {(2 * W * H) >> 20}M)"
    )

    if not args.snfs:
        # Size was not estimated above.
        sizef = estimate_size(f, qmin * W, H)
        sizeg = estimate_size(g, qmin * W, H)
    THRESHOLD = sizef - qmin.bit_length() - COFACTOR_BITS
    THRESHOLD2 = sizeg - COFACTOR_BITS2

    reduce_q = True if args.snfs else False
    sievepool = ProcessPoolExecutor(
        1,
        initializer=worker_init,
        # Parameters for LineSiever
        initargs=(f, g, ls, rs, THRESHOLD, ls2, rs2, THRESHOLD2, W, H, reduce_q),
    )
    factorpool = ProcessPoolExecutor(
        args.ncpu or os.cpu_count(),
        initializer=factorer_init,
        initargs=(f, g, B2f, B2g),
    )

    AREA = 2 * W * H
    seen = set()
    relf = open(datadir / "relations.sieve", "w", buffering=1)
    total = 0
    duplicates = 0

    t0 = time.monotonic()
    last_log = 0
    total_area = 0
    total_q = 0
    seenK = set()
    seenQ = set()
    Kcount, Qcount = 0, 0

    # Singleton live tracking
    # For each relation: number of singleton primes (less than 255)
    orphans = np.zeros(1 << 20, dtype=np.uint8)
    # For each singleton prime, index of relation containing it
    orphanK = {}
    orphanQ = {}

    excess1, excess2 = -9999, -9999
    sieve_args: Iterator[tuple[int, int]] = zip(qs, qrs)
    # Full sieve statistics
    sieve_stats1 = []
    # Sieve statistics without orphans
    sieve_stats2 = []
    MAX_SIEVE_QUEUE = 64
    MIN_EXCESS = 64
    with sievepool, factorpool:
        sieve_jobs = []
        factor_jobs = []
        while True:
            while len(sieve_jobs) < MAX_SIEVE_QUEUE:
                sieve_jobs.append(sievepool.submit(worker_task, next(sieve_args)))
            # Always wait for at least 1 sieve
            _ = sieve_jobs[0].result()
            sieve_pending = []
            for sfut in sieve_jobs:
                if not sfut.done():
                    sieve_pending.append(sfut)
                    continue
                q, qr, dt, qmat, reports = sfut.result()
                fut = factorpool.submit(factorer_task, (q, qmat, reports))
                factor_jobs.append((q, qr, dt, len(reports), fut))
            # Throttle if factoring is late
            concurrent.futures.wait(
                [f for _, _, _, _, f in factor_jobs[:-MAX_SIEVE_QUEUE]]
            )

            sieve_jobs = sieve_pending
            remaining = []
            for item in factor_jobs:
                q, qr, dt, nreports, fut = item
                if not fut.done():
                    remaining.append(item)
                    continue

                nrels = 0
                reschunk = fut.result()
                print(f"# q={q} r={qr}", file=relf)
                for x, y, facf, facg, ideals in reschunk:
                    # Normalize sign
                    if y < 0:
                        x, y = -x, -y
                    xy = x + (y << 32)
                    if xy in seen:
                        duplicates += 1
                        continue
                    str_facf = ",".join(f"{_l:x}" for _l in facf)
                    str_facg = ",".join(f"{_l:x}" for _l in facg)
                    if len(g) == 2:
                        facK, facZ = facf, facg
                    else:
                        facK, facZ = facg, facf
                    for _l in facZ:
                        if _l not in seenQ:
                            orphans[total] += 1
                            orphanQ[_l] = total
                            seenQ.add(_l)
                        else:
                            if (idx := orphanQ.pop(_l, None)) is not None:
                                orphans[idx] -= 1
                    for _l, _r in zip(facK, ideals):
                        _lr = (_l << 32) | _r
                        if _lr not in seenK:
                            # new singleton
                            orphans[total] += 1
                            orphanK[_lr] = total
                            seenK.add(_lr)
                        else:
                            if (idx := orphanK.pop(_lr, None)) is not None:
                                orphans[idx] -= 1
                    total += 1
                    seen.add(xy)
                    if sieve_rational:
                        relf.write(f"{x},{y}:{str_facf}:{str_facg}\n")
                    else:
                        relf.write(f"{x},{y}:{str_facg}:{str_facf}\n")
                    nrels += 1
                print(f"# Found {nrels} relations for {q=} (time {dt:.3f}s)", file=relf)
                total_q += 1
                total_area += AREA
                elapsed = time.monotonic() - t0
                Kcount = len(seenK)
                Qcount = len(seenQ)
                if total_q < 10 or elapsed > last_log + 1:
                    # Don't log too often.
                    n_orphans = np.count_nonzero(orphans)
                    if total > len(orphans) / 2:
                        orphans.resize((len(orphans) * 2,))
                    excess1 = total - Kcount - Qcount
                    excess2 = (total - n_orphans) - (
                        Kcount + Qcount - len(orphanK) - len(orphanQ)
                    )
                    last_log = elapsed
                    logger.info(
                        f"Sieved q={q:<8} r={qr:<8} area {total_area / 1e9:.0f}G in {dt:.3f}s "
                        + f"(speed {total_area / elapsed / 1e9:.3f}G/s shader {AREA / dt / 1e9:.3f}G/s): "
                        + f"{nrels}/{nreports} relations, {Kcount}/{Qcount} K/Q primes, total {total}"
                        + f" excess {excess1}/{excess2}"
                    )

                    # Store stats for estimation
                    # We compute stats after 1st singleton purge (without orphans).
                    if total > n_orphans:
                        sieve_stats1.append(
                            (
                                total,
                                Kcount,
                                Qcount,
                            )
                        )
                        sieve_stats2.append(
                            (
                                total - n_orphans,
                                Kcount - len(orphanK),
                                Qcount - len(orphanQ),
                            )
                        )

                    if len(sieve_stats1) % 30 == 10:
                        boundK = max(seenK) >> 32
                        boundZ = max(seenQ)
                        eta1 = sieve_eta(
                            boundK,
                            boundZ,
                            0,
                            sieve_stats1,
                        )
                        eta2 = sieve_eta(
                            boundK,
                            boundZ,
                            max(MIN_EXCESS, min(10000, total / 20)),
                            sieve_stats2,
                        )
                        if eta1 is None or (eta2 is not None and eta1 > eta2):
                            eta1 = eta2
                        if eta1 is not None:
                            logger.info(
                                f"Requiring {int(eta1 * total)} relations ({100 / eta1:.1f}% done)"
                            )

                    if max(excess1, excess2) > max(MIN_EXCESS, min(10000, total / 20)):
                        break

            factor_jobs = remaining
            if len(factor_jobs) > 100:
                logger.warning(f"{len(factor_jobs)} jobs waiting for cofactorization!")

            # Compute excess with/without singletons
            excess1 = total - Kcount - Qcount
            if max(excess1, excess2) > max(MIN_EXCESS, min(10000, total / 20)):
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
    singles = np.count_nonzero(orphans)
    logger.info(
        f"{total} relations {duplicates} duplicates {singles} orphans for q={qmin}..{q} in {elapsed:.3f}s"
    )
    relf.close()


if __name__ == "__main__":
    main()
