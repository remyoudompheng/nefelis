"""
Linear algebra for degree 3 Joux-Lercier (without Schirokauer maps)

The input is a list of relations:
    u^a product(ci^ai) = product(qi^bi)
where ci are prime ideals in a cubic field
      u is a fundamental unit
      qi are prime ideals of a quadratic field

In this subdirectory, we focus on the case where the cubic
field has class number 1 and only 1 real root (the unit group
has rank 1). In this case, we can replace the Schirokauer map
by an explicit factorization of algebraic numbers.
"""

import argparse
import json
import logging
import math
import pathlib
import random

import flint

from nefelis import filter
from nefelis.linalg_impl import SpMV
from nefelis.deg3.cubic import CubicField

DEBUG_RELS = False


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

    rels = []
    seen_xy = set()
    duplicates = 0
    Kf = CubicField(-f[0])
    A = g[2]
    for x, y, facg, facf in read_relations(workdir / "relations.sieve"):
        if (x, y) in seen_xy:
            duplicates += 1
            continue
        seen_xy.add((x, y))

        rel = {}
        # z = factor(f(z), Kf) = factor(g(z), Kg) / leading(g)
        relf = {}
        for _l in facf:
            relf[_l] = relf.get(_l, 0) + 1
        for (_l, _r), _e in Kf.factor(-x, y, list(relf.items())):
            if _l == -1:
                # FIXME: why is it broken?
                continue
            rel[f"f_{_l}_{_r}"] = _e
        # If g(x,y) = Ax²+Bxy+Cy² is smooth and g(ω,1)=0
        # let z=x-yω, Norm(z)=g(x,y)/A and Norm(Ax-yAω)=Ag(x,y)

        for _l in facg:
            if A % _l == 0:
                # If l divides A, consider the algebraic integer
                # z = Ax - y (Aω)
                # The ideal is determined by Ax/y mod l
                # if val(Ax/y)=0, this is the ideal of the nonzero root
                # if val(Ax/y)>0, this is the ideal at infinity
                # if val(Ax/y)<0, this is impossible if x,y are coprime
                if y % _l == 0:
                    _r = 1  # FIXME
                else:
                    _r = _l
            else:
                # If l is coprime to A, the valuation of l is z is just e.
                # The correct ideal above l is determined by -x/y mod l.
                _r = _l if y % _l == 0 else -x * pow(y, -1, _l) % _l
            key = f"g_{_l}_{_r}"
            rel.setdefault(key, 0)
            rel[key] = rel.get(key, 0) - 1
        if rel is None:
            # print("SKIP")
            continue
        rel[f"g_{A}"] = 1
        if DEBUG_RELS and any(A % _l == 0 for _l in facg):
            print(x, y)
            print(facf)
            print(facg)
            print(rel)
        rels.append(rel)

    if duplicates:
        logging.info(f"{duplicates} duplicate results in input file, ignored")

    rels2, _ = filter.prune(rels, pathlib.Path(workdir))
    rels3 = filter.filter(rels2, pathlib.Path(workdir))
    # Truncate to obtain a square matrix
    dim = len(set(key for r in rels3 for key in r))
    rels3 = rels3[:dim]

    M = SpMV(rels3)
    basis = M.basis
    dim = M.dim
    ell = n // 2
    poly = M.wiedemann_big(ell, blockm=args.blockw or 1)
    print("Computed characteristic poly", poly[:10], "...", poly[-10:])

    poly = [ai * pow(poly[-1], -1, ell) % ell for ai in poly]
    assert any(ai != 0 for ai in poly)
    assert len(poly) <= dim + 1 and poly[0] == 0, (dim, len(poly), poly[0])

    i0 = next(i for i, ai in enumerate(poly) if ai)
    logging.info(f"Polynomial is divisible by X^{i0}")
    wi = [random.randrange(ell) for _ in range(dim)]
    poly_k = poly[i0:]
    ker = M.polyeval(wi, ell, poly_k)
    assert any(k for k in ker)

    # Validate result
    prime_idx = {l: idx for idx, l in enumerate(basis)}
    for r in rels3:
        assert sum(e * ker[prime_idx[l]] for l, e in r.items()) % ell == 0

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
                # logging.debug(f"incomplete relation for {l}")

    logging.info(f"{added} logarithms deduced from {nremoved} removed relations")
    logging.info(f"{len(dlog)} primes have known virtual logarithms")

    logging.info("Collecting relations from full sieve results")
    extra = rels.copy()
    for iter in range(2, 5):
        logging.info(f"Running pass {iter} for {len(extra)} remaining relations")
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
        logging.info(f"{len(dlog)} primes have known coordinates")

    f_primes = {}
    g_primes = {}
    for key in dlog:
        _l = int(key.split("_")[1])
        if key.startswith("f"):
            f_primes.setdefault(_l, []).append(key)
        elif key.startswith("g"):
            g_primes.setdefault(_l, []).append(key)

    gen = None
    for l in sorted(set(f_primes) | set(g_primes)):
        if len(f_primes.get(l, [])) == 3:
            # Beware l is not just the product of 3 primes.
            roots = flint.nmod_poly(f, l).roots()
            assert len(roots) == 3
            z = l
            for r, _ in roots:
                rx, ry, rz = Kf.idealgen(l, int(r))
                z /= rx + ry * Kf.j + rz * Kf.j**2
            e = int(round(math.log(z) / math.log(Kf.u)))
            dlog[f"Z_{l}"] = sum(dlog[k] for k in f_primes[l]) + e * dlog["f_1_0"]
            if gen is None:
                gen = f"Z_{l}"
        if len(g_primes.get(l, [])) == 2:
            dlog[f"Z_{l}"] = sum(dlog[k] for k in g_primes[l])
            if gen is None:
                gen = f"Z_{l}"

    # Normalize kernel vector
    ginv = pow(dlog[gen], -1, ell)
    for k in dlog:
        dlog[k] = dlog[k] * ginv % ell
    logging.info(f"Computed logarithms in base {gen}")

    # Check rational primes
    checked = 0
    grat = int(gen[2:])
    for k in dlog:
        if k.startswith("Z"):
            krat = int(k[2:])
            if pow(grat, dlog[k], n) not in (krat, n - krat):
                logging.error(f"WRONG VALUE dlog({k}) = {dlog[k]}")
                dlog[k] = None
                continue
            checked += 1
    logging.info(f"Checked logarithms for {checked} rational primes")

    dlogs = [(k, v) for k, v in dlog.items()]
    with open(workdir / "dlog", "w") as w:
        for row in sorted(dlogs):
            w.write(" ".join(str(v) for v in row) + "\n")


if __name__ == "__main__":
    logging.getLogger().setLevel(level=logging.DEBUG)
    main()
