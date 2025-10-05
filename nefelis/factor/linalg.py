"""
Linear algebra for the factorization

Relations are collected into a matrix M (where rows are relations).

The block Wiedemann algorithm is used on sequence U M^k V
with linear generator P(X) which is a matrix polynomial.

The left kernel is obtained as K = sum(P[i] U M^i)
"""

import argparse
import json
import logging
import pathlib
import time

import flint

from nefelis import filter_gf2 as filter
from nefelis.linalg_gf2 import SpMV
from nefelis.factor import sqrt_arb

logger = logging.getLogger("linalg")
logsqrt = logging.getLogger("sqrt")


def read_relations(filepath: str | pathlib.Path):
    with open(filepath) as f:
        for line in f:
            if line.startswith("#"):
                continue
            xy, facg_, facf_ = line.strip().split(":")
            x, _, y = xy.partition(",")
            facg = [int(w, 16) for w in facg_.split(",")]
            facf = [int(w, 16) for w in facf_.split(",")]
            yield int(x), int(y), facg, facf


def main():
    argp = argparse.ArgumentParser()
    argp.add_argument(
        "--blockw", type=int, help="Use Block Wiedemann with size m=ARG n=1"
    )
    argp.add_argument("WORKDIR")
    args = argp.parse_args()
    main_impl(args)


def main_impl(args):
    workdir = pathlib.Path(args.WORKDIR)
    with open(workdir / "args.json") as f:
        doc = json.load(f)
        n = doc["n"]
        z = doc["z"]
        f = doc["f"]
        g = doc["g"]
        # Check polynomials
        assert sum(fi * z**i for i, fi in enumerate(f)) % n == 0
        assert sum(gi * z**i for i, gi in enumerate(g)) % n == 0

    Zn = flint.fmpz_mod_ctx(n)

    # Prepare quadratic characters
    chis = []
    for l in range(2**60, 2**60 + 2000):
        if not flint.fmpz(l).is_prime():
            continue
        fmodl = flint.nmod_poly(f, l)
        for r, e in fmodl.roots():
            if e == 1:
                chis.append((f"q_{l}{r}", l, int(r)))
    assert len(chis) >= 16
    chis = chis[:16]
    logger.info(f"Prepared {len(chis)} quadratic characters")
    # Prepare more quadratic characters for sanity checks
    testchis = []
    for l in range(2**61, 2**61 + 5000):
        if not flint.fmpz(l).is_prime():
            continue
        fmodl = flint.nmod_poly(f, l)
        for r, e in fmodl.roots():
            if e == 1:
                testchis.append((l, int(r)))
    assert len(testchis) >= 16
    logger.info(f"Prepared {len(testchis)} quadratic characters for validation")

    # When reading relations, we compute exact integer exponents
    # (necessary for the sqrt part?) but the kernel part will be modulo 2.

    rels = []
    zrels = {}
    seen_xy = set()
    duplicates = 0

    t0 = time.monotonic()
    for x, y, facg, facf in read_relations(workdir / "relations.sieve"):
        if (x, y) in seen_xy:
            duplicates += 1
            continue
        seen_xy.add((x, y))
        rel = {"CONSTANT": 1}
        # Marker to recognize filtering output
        rel[f"K_{x}_{y}"] = 1
        for _l in facf:
            _r = x * pow(y, -1, _l) % _l if y % _l else _l
            key = f"f_{_l}_{_r}"
            rel[key] = rel.get(key, 0) + 1
        for key, _l, _r in chis:
            if flint.fmpz(x - _r * y).jacobi(_l) == -1:
                rel[key] = 1
        for _l in facg:
            key = f"Z_{_l}"
            rel[key] = rel.get(key, 0) - 1
        zrels[(x, -y)] = facg
        # FIXME: add quadratic characters here
        rels.append(rel)

    dt = time.monotonic() - t0
    logger.info(
        f"Computed factorizations and characters for {len(rels)} relations in {dt:.1f}s"
    )

    if duplicates:
        logger.info(f"{duplicates} duplicate results in input file, ignored")

    rels2, _ = filter.prune(rels, pathlib.Path(workdir))
    rels3 = filter.filter(rels2, pathlib.Path(workdir))
    # Truncate to obtain a square matrix
    dim = len(set(key for r in rels3 for key in r))
    rels3 = rels3[:dim]

    M = SpMV(rels3)
    dim = M.dim
    kers = M.left_kernel()

    zn = Zn(z)
    A = f[-1]
    for ki in kers:
        s = set()
        for j in range(dim):
            if ki[j]:
                r = M.rels[M.rowidx[j]]
                s.symmetric_difference_update([l for l in r if l.startswith("K_")])

        xys = []
        for key in s:
            prefix, x, y = key.split("_")
            assert prefix == "K"
            xys.append((int(x), -int(y)))

        is_probably_square = True
        for _l, _r in testchis:
            residue = flint.nmod(1, _l)
            for x, y in xys:
                residue *= x + _r * y
            if flint.fmpz(int(residue)).jacobi(_l) == -1:
                logger.debug(
                    f"Candidate product of {len(s)} elements is not a square at place {_l},{_r}"
                )
                is_probably_square = False
                break

        if is_probably_square:
            logger.debug(
                f"Candidate square: product of {len(s)} elements validated at {len(testchis)} places"
            )
            rt = sqrt_arb.sqrt(f, xys)
            bits = int(max(ai.abs_upper().log_base(2) for ai in rt).ceil().fmpz())
            logsqrt.debug(f"Need {bits} precision bits")
            rt = sqrt_arb.sqrt(f, xys, int(1.1 * bits + 64))
            # Evaluate in Z/nZ
            sqrt = 0
            for i, r in enumerate(rt):
                sqrt += Zn(r.real.unique_fmpz()) * (A * zn) ** i
            sqrt /= Zn(A) ** (len(xys) // 2)
            # print(sqrt**2)

            candidate = Zn(1)
            for x, y in xys:
                candidate *= x + zn * y
            # print(candidate)

            # Compute square root on the rational side
            factors = {}
            for x, y in xys:
                for _l in zrels[(x, y)]:
                    factors[_l] = factors.get(_l, 0) + 1
            sqrtz = Zn(1)
            for p, e in factors.items():
                assert e & 1 == 0
                sqrtz *= Zn(p) ** (e // 2)
            sqrtz /= Zn(g[-1]) ** (len(xys) // 2)
            # print(sqrt, sqrtz)
            assert candidate == sqrt**2
            assert candidate == sqrtz**2

            d1 = flint.fmpz(n).gcd(int(sqrt - sqrtz))
            d2 = flint.fmpz(n).gcd(int(sqrt + sqrtz))
            if d1 > 1 and d1 < n:
                logsqrt.info(f"Found factor {d1}")
            elif d2 > 1 and d2 < n:
                logsqrt.info(f"Found factor {d2}")
            else:
                logsqrt.info("No factor from this square")


if __name__ == "__main__":
    logging.getLogger().setLevel(level=logging.DEBUG)
    main()
