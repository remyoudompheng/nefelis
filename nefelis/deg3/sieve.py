"""
Sieve for the polynomial x^3-2

Notations:
    N a prime modulus
    f(x) = x^3 - 2
    g(x) = A x^2 + B x + C

We assume that B^2 - 4AC < 0 (A > 0 and C > 0)
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
        return x, y, None, None

    facg = []
    if pymqs is not None:
        facg = pymqs.factor(int(vg))
    else:
        for _l, _e in flint.fmpz(vg).factor():
            facg += _e * [int(_l)]
    return x, y, facf, facg


SIEVER = None


def worker_init(g, ls, rs, threshold, I):
    global SIEVER
    SIEVER = sieve_vk.Siever(g, ls, rs, threshold, I)


def worker_task(args):
    q, qr = args
    t = time.monotonic()
    reports = SIEVER.sieve(q, qr)
    return q, qr, time.monotonic() - t, reports


PARAMS = [
    # bitsize, B1g, B2g, B2f, cofactor bits, I=logwidth, qmin
    (120, 20000, 16, 16, 16, 15, 20),
    (160, 50_000, 18, 17, 18, 15, 100),
    (200, 100_000, 19, 18, 20, 15, 3000),
    (220, 150_000, 20, 18, 20, 15, 6000),
    (240, 200_000, 20, 19, 25, 15, 10000),
    (260, 300_000, 21, 19, 30, 15, 20000),
    (280, 500_000, 21, 20, 30, 15, 40000),
    (300, 600_000, 22, 20, 30, 15, 70000),
]


def get_params(N):
    return min(PARAMS, key=lambda p: abs(p[0] - N.bit_length()))[1:]


def main():
    argp = argparse.ArgumentParser()
    argp.add_argument("N", type=int)
    argp.add_argument("OUTDIR")
    args = argp.parse_args()

    N = args.N
    datadir = pathlib.Path(args.OUTDIR)
    datadir.mkdir(exist_ok=True)

    ell = N // 2  # FIXME: support user ell

    assert N % 3 != 1
    assert flint.fmpz(N).is_prime()
    assert flint.fmpz(ell).is_prime()

    B1, B2g, B2f, COFACTOR_BITS, I, qmin = get_params(N)

    r = pow(2, (2 * N - 1) // 3, N)
    assert (r**3 - 2) % N == 0
    m = flint.fmpz_mat([[N, 0, 0], [int(r), -1, 0], [int(r * r), 0, -1]]).lll()
    print(m)
    g = None
    for row in m.table():
        if row[1] ** 2 < 4 * row[0] * row[2]:
            g = [int(_) for _ in row]
            break
    if g is None:
        raise ValueError("could not generate suitable polynomial")

    C, B, A = g
    assert (A * r * r + B * r + C) % N == 0
    print(f"g = {A} x² + {B} x + {C}")

    ls, rs = [], []
    for _l in sieve_vk.smallprimes(B1):
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
    THRESHOLD = (
        N.bit_length() // 3 + 2 * LOGAREA // 2 - qs[-1][0].bit_length() - COFACTOR_BITS
    )

    sievepool = multiprocessing.Pool(
        1, initializer=worker_init, initargs=(g, ls, rs, THRESHOLD, I)
    )
    factorpool = multiprocessing.Pool(8)

    with open(datadir / "args.json", "w") as w:
        json.dump(
            {
                "n": N,
                "f": [-2, 0, 0, 1],
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
    total_area = 0
    total_q = 0
    seenf = set()
    seeng = set()
    for q, qr, dt, reports in sievepool.imap(worker_task, qs):
        nrels = 0
        print(f"# q={q} r={qr}", file=relf)
        values = []
        for x, y in reports:
            if math.gcd(x, y) != 1:
                continue
            # value = u * x + v * y
            # print(f"{x}+{y}i", flint.fmpz(value).factor())
            # Output in Cado format
            # x,y:(factors of g(x) in hex):(factors of f(x) in hex)
            vf = abs(x * x * x - 2 * y * y * y)
            vg = abs(A * x * x + B * x * y + C * y * y)
            values.append((x, y, vf, vg, B2f, B2g))

        for x, y, facf, facg in factorpool.imap(factor_fg, values, chunksize=32):
            # Normalize sign
            if y < 0:
                x, y = -x, -y
            if (x, y) in seen:
                duplicates += 1
                continue
            # Ignore too large primes
            if facf is None:
                continue
            if any(_l.bit_length() > B2g for _l in facg):
                continue
            str_facf = ",".join(f"{_l:x}" for _l in facf)
            str_facg = ",".join(f"{_l:x}" for _l in facg)
            for _l in facf:
                _r = x * pow(y, -1, _l) % _l
                # assert (r**3 + 2) % _l == 0, (x, y, _l, r)
                seenf.add((_l, _r))
            for _l in facg:
                if A % _l == 0:
                    continue
                _r = x * pow(y, -1, _l) % _l
                seeng.add((_l, _r))
            seen.add((x, y))
            relf.write(f"{x},{y}:{str_facg}:{str_facf}\n")
            nrels += 1
        print(f"# Found {nrels} relations for {q=} (time {dt:.3f}s)", file=relf)
        total_q += 1
        total += nrels
        total_area += AREA
        elapsed = time.monotonic() - t0
        gcount = len(seeng)
        fcount = len(seenf)
        print(
            f"Sieved q={q} r={qr} area {AREA} in {dt:.3f}s (speed {total_area / elapsed / 1e9:.3f}G/s): {len(reports)} reports {nrels} relations, {gcount}/{fcount} Kg/Kf primes, total {total}"
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
    print(total, "relations", duplicates, f"duplicates in {elapsed:.3f}s")
    relf.close()


if __name__ == "__main__":
    main()
