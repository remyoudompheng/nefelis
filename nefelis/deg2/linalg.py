"""
Linear algebra for the Gaussian Integer method.

The input is a list of relations:
    product(pi^ai) = product(li^bi)
where pi is a list of rational primes
    li is a list of ideals of a quadratic field

For a complex quadratic number field, there are no
Schirokauer maps, and non-principal ideals become principal
when taking h-th powers where h is the class number.

If the ell argument is larger than h and larger than
the order of the unit group (2, 4, 6), then we can ignore
units, and process non-principal ideals as if they were principal.

The basis will be as follows:
    - all integers pi and all prime norms |li|
    - all "positive" ideals li (represented by integer -li)

Negative ideals conjugate(li) will be represented as |li|/li
"""

import json
import logging
import pathlib
import random
import sys

import flint
import numpy as np

from nefelis import filter
from nefelis.linalg_impl import SpMV


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
    workdir = pathlib.Path(sys.argv[1])

    with open(workdir / "args.json") as f:
        doc = json.load(f)
        n = doc["n"]
        z = doc["z"]
        f = doc["f"]
        g = doc["g"]
        # Check polynomials
        assert sum(fi * z**i for i, fi in enumerate(f)) % n == 0
        assert sum(gi * z**i for i, gi in enumerate(g)) % n == 0

    # A ideal of K is positive if it corresponds to the first root of f given by FLINT.
    # Otherwise it is negative.
    roots_f = {}

    rels = []
    seen_xy = set()
    duplicates = 0
    for x, y, facg, facf in read_relations(workdir / "relations.sieve"):
        if (x, y) in seen_xy:
            duplicates += 1
            continue
        seen_xy.add((x, y))
        rel = {}
        # Sign convention: f(x) / g(x) == 1
        # z zbar = f(z)
        # x+yω is divisible by ω-r (norm l) iff x+yr=0 mod l
        for _l in facf:
            if _l not in roots_f:
                roots_f[_l] = flint.nmod_poly(f, _l).roots()[0][0]
            r = roots_f[_l]
            if x + y * r == 0:
                rel[-_l] = rel.get(-_l, 0) + 1
            else:
                assert x - y * r == 0
                rel[_l] = rel.get(_l, 0) + 1
                rel[-_l] = rel.get(-_l, 0) - 1
        # u z = g(z)
        u = g[1]
        rel[u] = 1
        for _l in facg:
            rel[_l] = rel.get(_l, 0) - 1
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
    poly = M.wiedemann_big(ell)
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

    # Normalize kernel vector
    idx0, k0 = next(
        (idx, ki) for idx, ki in enumerate(ker) if ki != 0 and basis[idx] > 0
    )
    k0inv = pow(k0, -1, ell)
    for i, ki in enumerate(ker):
        ker[i] = ki * k0inv % ell
    logging.info(f"Computing logarithms in base {basis[idx0]}")
    gen = basis[idx0]

    # Validate result
    prime_idx = {l: idx for idx, l in enumerate(basis)}
    for r in rels3:
        assert sum(e * ker[prime_idx[l]] for l, e in r.items()) % ell == 0

    assert len(basis) == len(ker)

    # Build dlog database
    dlog = {l: v for l, v in zip(basis, ker)}
    # print(dlog)

    added, nremoved = 0, 0
    with open(workdir / "relations.removed") as f:
        for line in f:
            nremoved += 1
            l, _, facs = line.partition("=")
            rel = {
                int(p): int(e) for p, _, e in (f.partition("^") for f in facs.split())
            }
            # Relations from Gaussian elimination are naturally ordered
            # in a triangular shape.
            if all(_l in dlog for _l in rel):
                v = sum(_e * dlog[_l] for _l, _e in rel.items())
                dlog[int(l)] = v % ell
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

    # Output result in a format similar to CADO
    # Columns:
    # prime, 0/1 (g/f), root, dlog
    dlogs = []
    coell = (n - 1) // ell
    for l, v in list(dlog.items()):
        if l > 0 or l == g[1]:
            # rational prime
            if g[1] % l == 0:
                r = l
            else:
                r = -g[0] * pow(g[1], -1, l) % l
            dlogs.append((l, 0, r, v))
            # Check logarithm
            # assert pow(gen, v * coell, n) == pow(l, coell, n), f"{l} != ± {gen}^{v}"
            if pow(gen, v * coell, n) != pow(l, coell, n):
                logging.error(f"FAIL {l} != ± {gen}^{v}")
                del dlog[l]
        else:
            l = abs(l)
            r = int(roots_f[l])
            dlogs.append((l, 1, r, v))
            if l in dlog:
                r2 = l - r
                v0 = dlog[l]
                dlogs.append((l, 1, r2, (v0 - v) % ell))
            # FIXME: check logarithms for non-rational ideals

    with open(workdir / "dlog", "w") as w:
        for row in sorted(dlogs):
            w.write(" ".join(str(v) for v in row) + "\n")


if __name__ == "__main__":
    logging.getLogger().setLevel(level=logging.DEBUG)
    main()
