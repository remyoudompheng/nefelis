"""
Sieve for factoring

The sieve is done on the algebraic side (polynomial f).
"""

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import logging
import math
import multiprocessing
import os
import pathlib
import time

import numpy as np
import flint

from nefelis.skewpoly import skewness
from nefelis.sieve import Siever, LineSiever
from nefelis.integers import factor, smallprimes

# from nefelis.factor.polyselect_basem import polyselect
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
    SIEVER = LineSiever(*args)


def worker_task(args):
    q, qr = args
    t = time.monotonic()
    reports = SIEVER.sieve(q, qr)
    return q, qr, time.monotonic() - t, reports


PARAMS = [
    # bitsize, degree, B1f, B2f, B2g, cofactor bits, A=logarea, qmin
    # FIXME: calibrate parameters
    # Use bounds for a single large prime (avoid extreme CPU pressure)
    (60, 3, 2000, 12, 10, 15, 23, 20),
    (80, 3, 3000, 12, 11, 15, 24, 20),
    (100, 3, 5000, 13, 12, 15, 24, 50),
    (120, 3, 10000, 15, 13, 15, 24, 300),
    (140, 3, 30000, 16, 14, 20, 25, 300),
    (160, 3, 80000, 18, 17, 20, 25, 500),
    (180, 3, 80000, 19, 16, 20, 26, 2000),
    (200, 3, 100000, 19, 17, 20, 26, 3000),
    (220, 3, 200_000, 20, 18, 20, 26, 6000),
    # Degree 4
    (240, 4, 400_000, 21, 19, 20, 27, 8000),
    (260, 4, 800_000, 21, 19, 25, 27, 10000),
    # NFS can be useful starting from these sizes
    (280, 4, 1500_000, 22, 21, 30, 28, 10000),
    (300, 4, 2500_000, 23, 22, 35, 28, 20000),
    (320, 4, 3000_000, 24, 22, 30, 29, 50000),
    # 2 large primes?
    (340, 4, 2000_000, 24, 23, 65, 29, 400000),
]

# Parameters for trial division (threshold must be <128)
PARAMS2 = [
    # bitsize, B1f, thrF
    (60, 0, 0),
    (80, 1000, 20),
    (100, 2000, 25),
    (120, 5_000, 30),
    (140, 10_000, 30),
    (160, 20_000, 35),
    (180, 25_000, 35),
    (200, 30_000, 35),
    (220, 40_000, 40),
    # Degree 4
    (240, 100_000, 35),
    (260, 200_000, 35),
    (280, 300_000, 35),
    (300, 300_000, 40),
    (320, 500_000, 40),
    (340, 500_000, 45),
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

    degree, B1f, B2f, B2g, COFACTOR_BITS, logA, qmin = get_params(N)
    B1g, thr2 = 0, 0
    if not args.nogpufactor:
        B1g, thr2 = get_params2(N)
    logger.info(
        f"Sieving with B1={B1f / 1000:.0f}k,{B1g / 1000:.0f}k log(B2)={B2f},{B2g} q={qmin}.. {COFACTOR_BITS} cofactor bits"
    )

    f, g = polyselect(N, degree)
    skew = skewness(f)
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
    if degf == 4:
        sizef = f[2].bit_length()
    else:
        sizef = f[1].bit_length() - int(skew).bit_length() // 2
    LOGAREA = qs[-1].bit_length() + logA + 1
    THRESHOLD = sizef + degf * (LOGAREA // 2) - qs[-1].bit_length() - COFACTOR_BITS

    # Choose rectangle size for area 2^(2I+1)
    skew2 = skew / (1.5 * qmin)
    H = int(math.sqrt(2 ** (logA - 1) / skew2))
    W = (int(H * skew2 / LineSiever.SEGMENT_SIZE) + 1) * LineSiever.SEGMENT_SIZE
    logger.info(
        f"Sieve rectangle size {2 * W >> 10}k x {H} (skewness: {W / H:.3g}, area: {(2 * W * H) >> 20}M)"
    )

    sievepool = multiprocessing.Pool(
        1,
        initializer=worker_init,
        # Parameters for LineSiever
        initargs=(f, ls, rs, THRESHOLD, W, H, g, ls2, rs2, thr2),
    )
    factorpool = ProcessPoolExecutor(
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
    AREA = 2 * W * H
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
    fcount, gcount = 0, 0

    # Singleton live tracking
    # For each relation: number of singleton primes (less than 255)
    orphans = np.zeros(1 << 20, dtype=np.uint8)
    # For each singleton prime, index of relation containing it
    orphanf = {}
    orphang = {}

    excess1, excess2 = -9999, -9999
    with sievepool, factorpool:
        factor_jobs = []
        for q, qr, dt, reports in sievepool.imap(worker_task, list(zip(qs, qrs))):
            fut = factorpool.submit(factorer_task, (q, reports))
            factor_jobs.append((q, qr, dt, fut))
            remaining = []
            for item in factor_jobs:
                q, qr, dt, fut = item
                if not fut.done():
                    remaining.append(item)
                    continue

                nrels = 0
                reschunk = fut.result()
                print(f"# q={q} r={qr}", file=relf)
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
                        _lr = (_l << 32) | _r
                        if _lr not in seenf:
                            # new singleton
                            orphans[total] += 1
                            orphanf[_lr] = total
                            seenf.add(_lr)
                        else:
                            if (idx := orphanf.pop(_lr, None)) is not None:
                                orphans[idx] -= 1
                    for _l in facg:
                        if _l not in seeng:
                            orphans[total] += 1
                            orphang[_l] = total
                            seeng.add(_l)
                        else:
                            if (idx := orphang.pop(_l, None)) is not None:
                                orphans[idx] -= 1
                    total += 1
                    seen.add(xy)
                    relf.write(f"{x},{y}:{str_facg}:{str_facf}\n")
                    nrels += 1
                print(f"# Found {nrels} relations for {q=} (time {dt:.3f}s)", file=relf)
                total_q += 1
                total_area += AREA
                elapsed = time.monotonic() - t0
                fcount = len(seenf)
                gcount = len(seeng)
                if elapsed < 2 or elapsed > last_log + 1:
                    # Don't log too often.
                    n_orphans = np.count_nonzero(orphans)
                    if total > len(orphans) / 2:
                        orphans.resize((len(orphans) * 2,))
                    excess1 = total - fcount - gcount
                    excess2 = (total - n_orphans) - (
                        fcount + gcount - len(orphanf) - len(orphang)
                    )
                    last_log = elapsed
                    logger.info(
                        f"Sieved q={q:<8} r={qr:<8} area {total_area / 1e9:.0f}G in {dt:.3f}s "
                        + f"(speed {total_area / elapsed / 1e9:.3f}G/s shader {AREA / dt / 1e9:.3f}G/s): "
                        + f"{nrels}/{len(reports)} relations, {fcount}/{gcount} K/Q primes, total {total}"
                        + f" excess {excess1}/{excess2}"
                    )

            factor_jobs = remaining
            if len(factor_jobs) > 100:
                logger.warning(f"{len(factor_jobs)} jobs waiting for cofactorization!")

            # Compute excess with/without singletons
            excess1 = total - fcount - gcount
            if max(excess1, excess2) > max(64, min(10000, total / 20)):
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
    singles = np.count_nonzero(orphans)
    logger.info(
        f"{total} relations {duplicates} duplicates {singles} orphans in {elapsed:.3f}s"
    )
    relf.close()


if __name__ == "__main__":
    main()
