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
import random
import time

import flint
import numpy as np

from nefelis import filter_disk
from nefelis import filter_gf2 as filter
from nefelis import integers
from nefelis.linalg_gf2 import SpMV, SpMV_COO, SpMV_COO2
from nefelis.factor import sqrt_arb
from nefelis.factor import sqrt_padic

logger = logging.getLogger("linalg")
logsqrt = logging.getLogger("sqrt")

DEBUG_USE_REALCOMPLEX_SQRT = False


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
    argp.add_argument(
        "--bench",
        action="store_true",
    )

    argp.add_argument("WORKDIR")
    args = argp.parse_args()

    if args.bench:
        bench(args)
    else:
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

    assert len(g) == 2

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

    t0 = time.monotonic()
    rels, xy_elems, zrels = prune_load(workdir, chis)
    dt = time.monotonic() - t0
    logger.info(
        f"Computed factorizations and characters for {len(rels)} relations in {dt:.1f}s"
    )

    rels = filter.filter(rels, pathlib.Path(workdir))

    M = SpMV_COO2(rels)
    facs = [n]
    while True:
        kers = M.left_kernel()
        logger.info(f"Found {len(kers)} left kernel elements")
        sqrt_start = time.monotonic()
        facs = factor_with_kernels(n, f, g, z, xy_elems, zrels, M, kers, facs)
        sqrt_dt = time.monotonic() - sqrt_start
        logsqrt.info(f"Sqrt step done in {sqrt_dt:.3f}s")
        if all(
            flint.fmpz(_f).is_prime() or _f.bit_length() < n.bit_length() / 2
            for _f in facs
        ):
            break

        logger.info(f"Not enough ({len(kers)}) kernel elements, retrying")

    if any(not flint.fmpz(_f).is_prime() for _f in facs):
        facprimes = [_f for _f in facs if flint.fmpz(_f).is_prime()]
        faccomps = [_f for _f in facs if not flint.fmpz(_f).is_prime()]
        logsqrt.info(f"Found prime factors {facprimes}")
        logsqrt.info(f"Found composite factors {faccomps}")


def prune_load(workdir, chis: list[tuple[str, int, int]]):
    """
    Load relations as array of basis indices.

    Each relation contains a unique negative index used to
    track original relations during Gauss elimination.
    """
    # When reading relations, we compute exact integer exponents
    # (necessary for the sqrt part?) but the kernel part will be modulo 2.

    rels: list[set[int]] = []
    zrels: dict = {}
    seen_xy = set()
    duplicates = 0

    if not (workdir / "relations.sieve.pruned").is_file():
        filter_disk.prune(str(workdir / "relations.sieve"))

    # For factoring, we need to find a square in the number field,
    # but we don't need to know which ideals are involved except for the rational side
    # (saved in zrels dictionary)
    #
    # By convention we save:
    # - negative integer (-1, ...) for the (x,y) element index
    # - positive integers for the ideal/character basis
    seen_basis: dict[str, int] = {"CONSTANT": 0}
    name_basis: list[str] = ["CONSTANT"]
    xy_elems: list[tuple] = [None]

    for key, _, _ in chis:
        seen_basis[key] = len(name_basis)
        name_basis.append(key)

    for x, y, facg, facf in read_relations(workdir / "relations.sieve.pruned"):
        if (x, y) in seen_xy:
            duplicates += 1
            continue
        seen_xy.add((x, y))
        xy_elems.append((x, y))
        # Index of (x,y) to reconstruct value from kernel vector
        rel = {0, 1 - len(xy_elems)}
        for _l in facf:
            _r = x * pow(y, -1, _l) % _l if y % _l else _l
            key = f"f_{_l}_{_r}"
            idx = seen_basis.get(key)
            if idx is None:
                idx = len(name_basis)
                seen_basis[key] = idx
                name_basis.append(key)
            if idx in rel:
                rel.remove(idx)
            else:
                rel.add(idx)
        for key, _l, _r in chis:
            if flint.fmpz(x - _r * y).jacobi(_l) == -1:
                rel.add(seen_basis[key])
        for _l in facg:
            key = f"Z_{_l}"
            idx = seen_basis.get(key)
            if idx is None:
                idx = len(name_basis)
                seen_basis[key] = idx
                name_basis.append(key)
            if idx in rel:
                rel.remove(idx)
            else:
                rel.add(idx)
        zrels[(x, -y)] = np.array(facg, dtype=np.uint64)
        rels.append(np.array(list(rel), dtype=np.int32))

    if duplicates:
        logger.info(f"{duplicates} duplicate results in input file, ignored")

    return rels, xy_elems, zrels


def factor_with_kernels(
    n: int, f, g, z, xy_elems, zrels, M, kers, facs: list[int] = None
) -> list:
    Zn = flint.fmpz_mod_ctx(n)
    if facs is None:
        facs = [n]

    # Prepare a few quadratic characters for sanity checks
    # Kernel elements are already squares modulo ~16 characters,
    # no need to check too many of them (doing this in Python is slow).
    testchis = []
    for l in range(2**61, 2**61 + 5000):
        if not flint.fmpz(l).is_prime():
            continue
        fmodl = flint.nmod_poly(f, l)
        for r, e in fmodl.roots():
            if e == 1:
                testchis.append((l, int(r)))
        if len(testchis) >= 8:
            break
    assert len(testchis) >= 8

    # Precompute character values once.
    characters = {}
    for key, xy in enumerate(xy_elems):
        if xy is None:
            continue
        x, y = xy
        chars = [flint.fmpz(x - _r * y).jacobi(_l) < 0 for _l, _r in testchis]
        characters[key] = np.array(chars, dtype=np.uint8)
    del x, y, chars
    logger.info(f"Prepared {len(testchis)} quadratic characters for validation")

    dim = M.dim
    zn = Zn(z)
    assert len(f) > 2
    A = f[-1]

    if facs is None:
        facs = [int(n)]
    for ki in kers:
        # Collect product terms for this kernel element.
        all_ridx = []
        for j in range(dim):
            if ki[j]:
                r = M.rels[M.rowidx[j]]
                all_ridx.append(r[r < 0])
        ridxs, rcounts = np.unique(np.concatenate(all_ridx), return_counts=True)
        ridxs = -ridxs[(rcounts & 1) == 1]

        xys = []
        for ridx in ridxs:
            x, y = xy_elems[ridx]
            xys.append((x, -y))

        char_k = sum(characters[ridx] for ridx in ridxs) & 1
        is_probably_square = True
        for i, (_l, _r) in enumerate(testchis):
            if char_k[i] & 1 == 1:
                logger.debug(
                    f"Candidate product of {len(s)} elements is not a square at place {_l},{_r}"
                )
                is_probably_square = False
                break
        del char_k, all_ridx, x, y

        if is_probably_square:
            logger.debug(
                f"Candidate square: product of {len(ridxs)} elements validated at {len(testchis)} places"
            )
            rt = sqrt_arb.sqrt(f, xys)
            bits = int(max(ai.abs_upper().log_base(2) for ai in rt).ceil().fmpz())
            logsqrt.debug(f"Need {bits} precision bits")
            if DEBUG_USE_REALCOMPLEX_SQRT:
                rt = sqrt_arb.sqrt(f, xys, int(1.1 * bits + 64))
                # Evaluate in Z/nZ
                sqrt = 0
                for i, r in enumerate(rt):
                    sqrt += Zn(r.real.unique_fmpz()) * (A * zn) ** i
                sqrt /= Zn(A) ** (len(xys) // 2)
            else:
                rt = sqrt_padic.sqrt(f, xys, int(1.1 * bits + 64))
                sqrt = 0
                for i, r in enumerate(rt):
                    sqrt += Zn(r) * (A * zn) ** i
                sqrt /= Zn(A) ** (len(xys) // 2)

            # print(sqrt**2)

            # Uncomment to check square roots modulo n.
            # candidate = Zn(1)
            # for x, y in xys:
            #     candidate *= x + zn * y
            # print(candidate)

            # Compute square root on the rational side
            factors = {}
            for x, y in xys:
                for _l in map(int, zrels[(x, y)]):
                    factors[_l] = factors.get(_l, 0) + 1
            sqrtz = Zn(1)
            for p, e in factors.items():
                assert e & 1 == 0
                sqrtz *= Zn(p) ** (e // 2)
            sqrtz /= Zn(g[-1]) ** (len(xys) // 2)
            # print(sqrt, sqrtz)
            # assert candidate == sqrt**2
            # assert candidate == sqrtz**2
            assert sqrt**2 == sqrtz**2

            d1 = flint.fmpz(n).gcd(int(sqrt - sqrtz))
            d2 = flint.fmpz(n).gcd(int(sqrt + sqrtz))
            divisors: list[flint.fmpz] = []
            # For convenience, we eliminate small factors here
            # to avoid looping for tiny factors.
            tiny_size = max(8, min(32, n.bit_length() // 20))
            for d in [d1, d2]:
                if d == 1 or d == n:
                    continue
                if d.is_prime():
                    divisors.append(d)
                else:
                    for _l, _e in integers.factor_smooth(d, tiny_size):
                        divisors += _e * [flint.fmpz(_l)]
            for d in divisors:
                facsplit = []
                for _f in facs:
                    if (f1 := d.gcd(_f)) not in (1, _f):
                        logsqrt.info(f"Found factor {f1}")
                        facsplit += [int(f1), int(_f // f1)]
                    else:
                        facsplit.append(_f)
                facs = facsplit
            if not divisors:
                logsqrt.info("No factor from this square")

            if all(flint.fmpz(_f).is_prime() for _f in facs):
                facs.sort()
                logsqrt.info(f"Found prime factors {facs}")
                break

    return facs


def bench(args):
    workdir = pathlib.Path(args.WORKDIR)
    rows = []
    with open(workdir / "relations.filtered", encoding="ascii") as f:
        for line in f:
            r = np.array(line.split(), dtype=np.int32)
            rows.append(r)
    logger.info(f"Loaded existing matrix with {len(rows)} rows")

    import nefelis.backends.kompute.linalg_gf2

    nefelis.backends.kompute.linalg_gf2.DEBUG_NO_SORT_ROWS = True
    nefelis.backends.kompute.linalg_gf2.DEBUG_NO_PADDING = True

    ITERS = 1000

    logger.info(f"Benchmark with SpMV ({ITERS} iterations)")
    # make deterministic
    random.seed(0)
    t0 = time.monotonic()
    M = SpMV(rows)
    logger.info(f"Built matrix in {time.monotonic() - t0:.1f}s")
    dt1, gpu_dt = M.benchmark(32, ITERS, False)
    logger.info(f"forward {dt1:.3f}ms/matmul GPU {gpu_dt:.3f}ms/matmul")
    dt2, gpu_dt = M.benchmark(32, ITERS, True)
    logger.info(f"backward {dt2:.3f}ms/matmul GPU {gpu_dt:.3f}ms/matmul")
    logger.info(
        f"Throughput forward {1000 / dt1:.1f} matmul/s, transpose {1000 / dt2:.1f} matmul/s"
    )

    logger.info(f"Benchmark with SpMV_COO ({ITERS} iterations)")
    # make deterministic
    random.seed(0)
    t0 = time.monotonic()
    M = SpMV_COO(rows)
    logger.info(f"Built matrix in {time.monotonic() - t0:.1f}s")
    dt1, gpu_dt = M.benchmark(32, ITERS, False)
    logger.info(f"forward {dt1:.3f}ms/matmul GPU {gpu_dt:.3f}ms/matmul")
    dt2, gpu_dt = M.benchmark(32, ITERS, True)
    logger.info(f"backward {dt2:.3f}ms/matmul GPU {gpu_dt:.3f}ms/matmul")
    logger.info(
        f"Throughput forward {1000 / dt1:.1f} matmul/s, transpose {1000 / dt2:.1f} matmul/s"
    )

    logger.info(f"Benchmark with SpMV_COO2 ({ITERS} iterations)")
    # make deterministic
    random.seed(0)
    t0 = time.monotonic()
    M = SpMV_COO2(rows)
    logger.info(f"Built matrix in {time.monotonic() - t0:.1f}s")
    dt1, gpu_dt = M.benchmark(32, ITERS, False)
    logger.info(f"forward {dt1:.3f}ms/matmul GPU {gpu_dt:.3f}ms/matmul")
    dt2, gpu_dt = M.benchmark(32, ITERS, True)
    logger.info(f"backward {dt2:.3f}ms/matmul GPU {gpu_dt:.3f}ms/matmul")
    logger.info(
        f"Throughput forward {1000 / dt1:.1f} matmul/s, transpose {1000 / dt2:.1f} matmul/s"
    )


if __name__ == "__main__":
    import nefelis.logging

    nefelis.logging.setup(logging.DEBUG)
    main()
