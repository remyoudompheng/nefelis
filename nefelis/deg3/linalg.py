"""
Linear algebra for degree 3 Joux-Lercier

Without Schirokauer maps:

The input is a list of relations:
    u^a product(ci^ai) = product(qi^bi)
where ci are prime ideals in a cubic field
      u is a fundamental unit
      qi are prime ideals of a quadratic field

In this subdirectory, we focus on the case where the cubic
field has class number 1 and only 1 real root (the unit group
has rank 1). In this case, we can replace the Schirokauer map
by an explicit factorization of algebraic numbers.

With Schirokauer maps:

We assume that the cubic polynomial has a unit group of rank 1
and a root modulo l, and the root modulo l is used to define
the Schirokauer map.
"""

import argparse
import json
import logging
import math
import pathlib
import random
import time

import flint

from nefelis import filter
from nefelis.deg3.cubic import CubicField
from nefelis.linalg import SpMV

DEBUG_RELS = False

logger = logging.getLogger("linalg")


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


def schirokauer_place(f, ell):
    """
    Returns the root of f modulo l^2
    """
    rs = flint.fmpz_mod_poly(f, flint.fmpz_mod_poly_ctx(ell)).roots()
    assert len(rs) == 1

    # Hensel lift the root of f modulo ell^2
    # r -> r - f(r)/f'(r)
    sm_place = flint.fmpz_mod_ctx(ell**2)
    r = flint.fmpz_mod(int(rs[0][0]), sm_place)
    fr = sum(fi * r**i for i, fi in enumerate(f))
    dfr = sum(i * fi * r ** (i - 1) for i, fi in enumerate(f) if i > 0)
    r -= fr / dfr
    assert sum(fi * r**i for i, fi in enumerate(f)) == 0
    return r


def schirokauer_map(x, y, r, ell):
    """
    Compute the Schirokauer map for algebraic element x+yω
    at place (l,r)
    """
    zl = x + y * r
    zl1 = int(zl ** (ell - 1))
    assert zl1 % ell == 1
    return zl1 // ell


def main():
    argp = argparse.ArgumentParser()
    argp.add_argument(
        "--nosm",
        action="store_true",
        help="Choose simple polynomials to avoid Schirokauer maps",
    )
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

    ell = n // 2
    sm_root = None
    if args.nosm:
        Kf = CubicField(f)
    else:
        Kf = None
        sm_root = schirokauer_place(f, ell)
        logger.info(f"Schirokauer map will use root {sm_root} mod l^2")

    rels = []
    seen_xy = set()
    duplicates = 0

    t0 = time.monotonic()
    for x, y, facg, facf in read_relations(workdir / "relations.sieve"):
        if (x, y) in seen_xy:
            duplicates += 1
            continue
        seen_xy.add((x, y))

        # z = factor(f(z), Kf) = factor(g(z), Kg) / leading(g)
        rel = {"CONSTANT": 1}
        for _l in facf:
            _r = x * pow(y, -1, _l) % _l if y % _l else _l
            key = f"f_{_l}_{_r}"
            rel[key] = rel.get(key, 0) + 1
        for _l in facg:
            _r = x * pow(y, -1, _l) % _l if y % _l else _l
            key = f"g_{_l}_{_r}"
            rel[key] = rel.get(key, 0) - 1
        if sm_root is None:
            # Without Schirokauer map: determine directly unit exponent
            rel["f_1_0"] = Kf.unit_exponent(x, y)
        else:
            # With Schirokauer map: augment prime ideals with SM value
            # Corresponding algebraic integer is x-yω
            v_sm = schirokauer_map(x, -y, sm_root, ell)
            rel["SM"] = v_sm

        rels.append(rel)
    dt = time.monotonic() - t0
    if sm_root is None:
        logger.info(f"Computed factorizations for {len(rels)} relations in {dt:.1f}s")
    else:
        logger.info(
            f"Computed factorizations and SM maps for {len(rels)} relations in {dt:.1f}s"
        )

    if duplicates:
        logger.info(f"{duplicates} duplicate results in input file, ignored")

    rels2, _ = filter.prune(rels, pathlib.Path(workdir))
    rels3 = filter.filter(rels2, pathlib.Path(workdir))
    # Truncate to obtain a square matrix
    dim = len(set(key for r in rels3 for key in r))
    rels3 = rels3[:dim]

    ell = n // 2  # FIXME
    M = SpMV(rels3, ell)
    basis = M.basis
    dim = M.dim
    poly = M.wiedemann_big(ell, blockm=args.blockw)
    logger.info(f"Computed characteristic poly {poly[:3]}...{poly[-3:]}")

    poly = [ai * pow(poly[-1], -1, ell) % ell for ai in poly]
    assert any(ai != 0 for ai in poly)
    assert len(poly) <= dim + 1 and poly[0] == 0, (dim, len(poly), poly[0])

    i0 = next(i for i, ai in enumerate(poly) if ai)
    logger.info(f"Polynomial is divisible by X^{i0}")
    wi = [random.randrange(ell) for _ in range(dim)]
    poly_k = poly[i0:]
    ker = M.polyeval(wi, ell, poly_k)
    assert any(k for k in ker)

    # Validate result
    prime_idx = {l: idx for idx, l in enumerate(basis)}
    for r in rels3:
        assert sum(e * ker[prime_idx[l]] for l, e in r.items()) % ell == 0
    logger.info("Checked element of matrix right kernel")

    assert len(basis) == len(ker)

    # Build dlog database
    dlog = {l: v for l, v in zip(basis, ker)}
    # print(dlog)

    added, nremoved = 0, 0
    with open(workdir / "relations.removed") as fd:
        for line in fd:
            nremoved += 1
            l, _, facs = line.partition("=")
            l = l.strip()
            rel = {p: int(e) for p, _, e in (_f.partition("^") for _f in facs.split())}
            # Relations from Gaussian elimination are naturally ordered
            # in a triangular shape.
            if all(_l in dlog for _l in rel):
                v = sum(_e * dlog[_l] for _l, _e in rel.items())
                dlog[l] = v % ell
                added += 1
            else:
                pass
                # logger.debug(f"incomplete relation for {l}")

    logger.info(f"{added} logarithms deduced from {nremoved} removed relations")
    logger.info(f"{len(dlog)} primes have known virtual logarithms")

    logger.info("Collecting relations from full sieve results")
    extra = rels.copy()
    for iter in range(2, 5):
        logger.info(f"Running pass {iter} for {len(extra)} remaining relations")
        remaining = []
        for rel in extra:
            news = [l for l in rel if l not in dlog]
            if len(news) == 0:
                continue
            if len(news) == 1:
                l = news[0]
                v = sum(_e * dlog[_p] for _p, _e in rel.items() if _p != l)
                dlog[l] = v * pow(-rel[l], -1, ell) % ell
            remaining.append(rel)
        if len(remaining) == len(extra):
            break
        extra = remaining
        logger.info(f"{len(dlog)} primes have known coordinates")

    f_primes = {}
    g_primes = {}
    for key in dlog:
        if key in ("CONSTANT", "SM"):
            continue
        _l = int(key.split("_")[1])
        if key.startswith("f"):
            f_primes.setdefault(_l, []).append(key)
        elif key.startswith("g"):
            g_primes.setdefault(_l, []).append(key)

    gen = None
    for l in sorted(set(f_primes) | set(g_primes)):
        # FIXME: also handle ramified primes!
        if len(f_primes.get(l, [])) == 3:
            # Beware l is not just the product of 3 primes.
            roots = flint.nmod_poly(f, l).roots()
            if sm_root is None:
                # FIXME: implement check for new CubicField
                assert len(roots) == 3
                continue
                # e = Kf.unit_exponent(l, 0)
                # dlog_f = sum(dlog[k] for k in f_primes[l]) + e * dlog["f_1_0"]
            else:
                assert len(roots) == 3 or (len(roots) == 2 and f[3] % l == 0)
                l_sm = schirokauer_map(l, 0, sm_root, ell)
                dlog_f = sum(dlog[k] for k in f_primes[l]) + l_sm * dlog["SM"]

            dlog[f"Z_{l}"] = dlog_f % ell
            if gen is None:
                gen = f"Z_{l}"
        if len(g_primes.get(l, [])) == 2:
            dlog_g = sum(dlog[k] for k in g_primes[l]) % ell
            if f"Z_{l}" in dlog:
                assert dlog[f"Z_{l}"] == dlog_g, (dlog[f"Z_{l}"], dlog_g)
            dlog[f"Z_{l}"] = dlog_g
            if gen is None:
                gen = f"Z_{l}"

    # Normalize kernel vector
    ginv = pow(dlog[gen], -1, ell)
    for k in dlog:
        dlog[k] = dlog[k] * ginv % ell
    logger.info(f"Computed logarithms in base {gen}")

    # Check rational primes
    checked = 0
    grat = int(gen[2:])
    for k in dlog:
        if k.startswith("Z"):
            krat = int(k[2:])
            if pow(grat, dlog[k], n) not in (krat, n - krat):
                logger.error(f"WRONG VALUE dlog({k}) = {dlog[k]}")
                dlog[k] = None
                continue
            checked += 1
    logger.info(f"Checked logarithms for {checked} rational primes")

    def key_l(k: str):
        for w in k.split("_"):
            if w.isdigit():
                return int(w)
        return 0

    dlogs = [(key_l(k), k, v) for k, v in dlog.items()]
    with open(workdir / "dlog", "w") as w:
        for _, k, v in sorted(dlogs):
            w.write(f"{k} {v}\n")


if __name__ == "__main__":
    logging.getLogger().setLevel(level=logging.DEBUG)
    main()
