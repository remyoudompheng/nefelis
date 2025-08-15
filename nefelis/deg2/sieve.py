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
import json
import logging
import math
import multiprocessing
import multiprocessing.dummy
import pathlib
import time

import flint

try:
    import pymqs
except ImportError:
    pymqs = None

from nefelis import sieve_vk


def factor_fg(args):
    x, y, vf, vg, B2f, B2g = args
    facf = []
    if pymqs is not None:
        facf = pymqs.factor(int(vf))
    else:
        for _l, _e in flint.fmpz(vf).factor():
            facf += _e * [int(_l)]
    if any(_f.bit_length() > B2f for _f in facf):
        return None

    facg = []
    if pymqs is not None:
        facg = pymqs.factor(int(vg))
    else:
        for _l, _e in flint.fmpz(vg).factor():
            facg += _e * [int(_l)]
    if any(_l.bit_length() > B2g for _l in facg):
        return None

    return x, y, facf, facg


SIEVER = None


def worker_init(poly, ls, rs, threshold, I):
    global SIEVER
    SIEVER = sieve_vk.Siever(poly, ls, rs, threshold, I)


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
    (320, 4000_000, 23, 21, 45, 14, 500000),
    # 2 large primes
    (340, 3000_000, 23, 22, 55, 14, 800000),
]


def get_params(N):
    return min(PARAMS, key=lambda p: abs(p[0] - N.bit_length()))[1:]


def main():
    argp = argparse.ArgumentParser()
    argp.add_argument("N", type=int)
    argp.add_argument("OUTDIR")
    args = argp.parse_args()

    logging.getLogger().setLevel(level=logging.DEBUG)

    N = args.N
    datadir = pathlib.Path(args.OUTDIR)
    datadir.mkdir(exist_ok=True)

    ell = N // 2  # FIXME: support user ell

    assert flint.fmpz(N).is_prime()
    assert flint.fmpz(ell).is_prime()

    B1, B2g, B2f, COFACTOR_BITS, I, qmin = get_params(N)

    r = flint.fmpz_mod(-2, flint.fmpz_mod_ctx(N)).sqrt()
    m = flint.fmpz_mat([[0, N], [-1, int(r)]]).lll()
    u, v = m.table()[0]
    print(u * r + v)
    assert u * r + v == 0
    print(f"{u = }")
    print(f"{v = }")

    ls = sieve_vk.smallprimes(B1)
    rs = [(-v * pow(u, -1, l)) % l if u % l else l for l in ls]
    qs = [_q for _q in sieve_vk.smallprimes(10 * qmin) if _q >= qmin and u % _q != 0]
    qrs = [-v * pow(u, -1, q) % q for q in qs]

    LOGAREA = qs[-1].bit_length() + 2 * I
    THRESHOLD = N.bit_length() // 2 + LOGAREA // 2 - qs[-1].bit_length() - COFACTOR_BITS

    sievepool = multiprocessing.Pool(
        1, initializer=worker_init, initargs=([v, u], ls, rs, THRESHOLD, I)
    )
    factorpool = multiprocessing.Pool(8)

    with open(datadir / "args.json", "w") as w:
        z = int((-v * pow(u, -1, N)) % N)
        json.dump(
            {
                "n": N,
                "f": [2, 0, 1],  # FIXME
                "g": [int(v), int(u)],
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
    total_area = 0
    total_q = 0
    seenf = set()
    seeng = set()
    for q, dt, reports in sievepool.imap(worker_task, list(zip(qs, qrs))):
        nrels = 0
        print(f"# q={q}", file=relf)
        values = []
        for x, y in reports:
            if math.gcd(x, y) != 1:
                continue
            # value = u * x + v * y
            # print(f"{x}+{y}i", flint.fmpz(value).factor())
            # Output in Cado format
            # x,y:(factors of g(x) in hex):(factors of f(x) in hex)
            vf = abs(x * x + 2 * y * y)
            vg = abs((u * x + v * y) // q)
            values.append((x, y, vf, vg, B2f, B2g))

        for item in factorpool.imap(factor_fg, values, chunksize=32):
            if item is None:
                continue
            x, y, facf, facg = item
            # Normalize sign
            if y < 0:
                x, y = -x, -y
            if (x, y) in seen:
                duplicates += 1
                continue
            str_facf = ",".join(f"{_l:x}" for _l in facf)
            str_facg = ",".join(f"{_l:x}" for _l in facg + [q])
            seenf.update(facf)
            seeng.update(facg)
            seen.add((x, y))
            relf.write(f"{x},{y}:{str_facg}:{str_facf}\n")
            nrels += 1
        print(f"# Found {nrels} relations for {q=} (time {dt:.3f}s)", file=relf)
        total_q += 1
        total += nrels
        total_area += AREA
        elapsed = time.monotonic() - t0
        Qcount = len(seenf | seeng)
        Kcount = len(seenf)
        print(
            f"Sieved q={q} area {AREA} in {dt:.3f}s (speed {total_area / elapsed / 1e9:.3f}G/s): {len(reports)} reports {nrels} relations, {Qcount}/{Kcount} Q/K primes, total {total}"
        )
        if total > 1.1 * Qcount + Kcount:
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
    print(total, "relations", duplicates, f"duplicates in {elapsed:.3f}s")
    relf.close()


if __name__ == "__main__":
    main()
