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

import argparse
import json
import logging
import pathlib
import random

import flint

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


def idealgen(f: list[int], l: int, r: int):
    """
    Try to find a generator of ideal (l,r) in number field Q[x]/(f)

    Returns (x,y) and a list of extra ideals p[i]^e[i]
    such that (x+yω) = (l,r) * prod(p[i]^e[i])
    """
    # r-ω belongs to the ideal, but this is really Ar-ω where ω is integral.
    if r == l:
        m = flint.fmpz_mat([[0, l], [1, 0]]).lll()
    else:
        m = flint.fmpz_mat([[l, 0], [int(r), 1]]).lll()
    u1, v1, u2, v2 = m.entries()
    c, b, a = f
    best = None
    for u, v in [(u1, v1), (u2, v2), (u1 + u2, v1 + v2), (u1 - u2, v1 - v2)]:
        if u == 0 or v == 0:
            continue
        norm = a * u**2 + b * u * v + c * v**2
        if best is None or norm < best[0]:
            best = (norm, (u, v))
    if best is None:
        raise ValueError("failed to find a generator")

    norm, (u, v) = best
    # print(f"Ideal ({l},{r}) => found generator {u}-{v}ω of norm {norm//l}l")
    k = best[0] // l
    # Note that f(x,y) is the norm of x-yω
    ideals = []
    for _l, _e in flint.fmpz(k).factor():
        r = u * pow(v, -1, _l) % _l if v % _l else _l
        if v % _l:
            assert (a * r * r + b * r + c) % _l == 0
        ideals.append((_l, r, _e))
    return u, -v, ideals


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

    # A ideal of K is positive if it corresponds to the first root of f given by FLINT.
    # Otherwise it is negative.
    roots_f = {}

    rels = []
    seen_xy = set()
    duplicates = 0

    assert f[1] ** 2 - 4 * f[0] * f[2] < 0
    # FIXME: this script uses keys +l for rational ideals
    # and -l for split ideals of the quadratic fields,
    # which is extremely confusing.
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
                roots_f[_l] = [int(_r) for _r, _ in flint.nmod_poly(f, _l).roots()]
                if f[2] % _l == 0:
                    # infinity is the main root
                    roots_f[_l] = [_l] + roots_f[_l]
            r0 = roots_f[_l][0]

            if y % _l:
                r = x * pow(y, -1, _l) % _l
                assert (f[2] * r**2 + f[1] * r + f[0]) % _l == 0
            else:
                r = _l
                assert f[2] % _l == 0

            if r == r0:
                rel[-_l] = rel.get(-_l, 0) + 1
            else:
                assert r == roots_f[_l][1]
                rel[_l] = rel.get(_l, 0) + 1
                rel[-_l] = rel.get(-_l, 0) - 1
        # Apply leading coefficient of f
        # for _l, _e in afacs:
        #    rel[-_l] = rel.get(-_l, 0) - _e

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

    ell = n // 2  # FIXME
    M = SpMV(rels3, ell)
    basis = M.basis
    dim = M.dim
    poly = M.wiedemann_big(ell, blockm=args.blockw or 1)
    print("Computed characteristic poly", poly[:10], "...", poly[-10:])

    poly = [ai * pow(poly[-1], -1, ell) % ell for ai in poly]
    assert any(ai != 0 for ai in poly)
    assert len(poly) <= dim + 1 and poly[0] == 0, (dim, len(poly), poly[0])

    i0 = next(i for i, ai in enumerate(poly) if ai)
    logging.info(f"Polynomial of degree {len(poly) - 1} is divisible by X^{i0}")
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

    bad_indexes = set()
    if len(poly) < dim + 1:
        logging.info("Computing another kernel vector, solution may not be unique")
        wi = [random.randrange(ell) for _ in range(dim)]
        poly_k = poly[i0:]
        ker2 = M.polyeval(wi, ell, poly_k)
        assert any(k for k in ker2)

        k1inv = pow(ker2[idx0], -1, ell)
        for i in range(len(ker2)):
            ker2[i] = ker2[i] * k1inv % ell
            if ker2[i] != ker[i]:
                logging.info(f"Ambiguous logarithm for prime {basis[i]}")
                bad_indexes.add(i)

    # Validate result
    prime_idx = {l: idx for idx, l in enumerate(basis)}
    for r in rels3:
        if any(prime_idx[l] in bad_indexes for l in r):
            logging.error("Skip check for bad relation {r}")
            continue
        assert sum(e * ker[prime_idx[l]] for l, e in r.items()) % ell == 0

    assert len(basis) == len(ker)

    # Build dlog database
    dlog = {l: v for l, v in zip(basis, ker) if prime_idx[l] not in bad_indexes}
    # print(dlog)

    added, nremoved = 0, 0
    with open(workdir / "relations.removed") as fd:
        for line in fd:
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
    z_checked = 0
    k_checked = 0

    # When considering algebraic integer, the leading coefficient of f
    # must be cancelled using the "roots at infinity".
    Ainvlog = 0
    if f[2] != 1:
        for _l, _e in flint.fmpz(f[2]).factor():
            Ainvlog -= _e * dlog[-_l]

    for l, v in list(dlog.items()):
        # The key u=g[1] actually represents u/A
        if l == g[1]:
            if pow(gen, (v - Ainvlog) * coell, n) != pow(l, coell, n):
                logging.error(f"FAIL {l} != ± {gen}^{v}")
                raise ValueError("wrong logarithm of constant coefficients")

            dlogs.append((l, "CONSTANT", r, v))
        elif l > 0:
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
                raise ValueError(f"Incorrect logarithm of rational prime {l}")
            else:
                # print(f"Checked rational {l} {v}")
                z_checked += 1
        else:
            l = abs(l)
            for r in roots_f[l]:
                r = int(r)
                # Find an explicit generator of this (with some extra small ideals)
                # Note that l * ideals is a factorization of root(A) (xl + yl ω)

                xl, yl, ideals = idealgen(f, l, r)
                if r == roots_f[l][0]:
                    vv = v
                else:
                    if l not in dlog:
                        # logging.warning(f"Cannot check ideal {(l, r)}")
                        break
                    vv = dlog[l] - v

                logz = vv + Ainvlog
                for _l, _r, _e in ideals:
                    r0 = roots_f.get(_l)
                    if r0 is None or _l not in dlog or -_l not in dlog:
                        logging.warning(f"missing small ideal {_l},{_r} in factor base")
                        logz = None
                        break
                    if _r == r0[0]:
                        logz += _e * (dlog[-_l])
                    else:
                        logz += _e * (dlog[_l] - dlog[-_l])
                if logz is None:
                    continue

                if logz is None:
                    logging.warning(
                        f"Unable to check logarithm of {xl}+{yl}i in ideal ({l},{r})"
                    )
                else:
                    zz = xl + z * yl
                    if pow(gen, logz * coell, n) != pow(zz, coell, n):
                        logging.error(f"FAIL {xl}+{yl}*ω != ± {gen}^{logz}")
                        raise
                    # logging.debug(f"Checked algebraic {xl}+{yl}*ω == ± {gen}^{logz}")
                    k_checked += 1

                dlogs.append((l, 1, r, vv))

    logging.info(
        f"Successfully checked logarithms for {z_checked} rational and {k_checked} algebraic ideals"
    )

    with open(workdir / "dlog2", "w") as w:
        for row in sorted(dlogs):
            w.write(" ".join(str(v) for v in row) + "\n")

    # Same with string keys
    with open(workdir / "dlog", "w") as w:
        # rational primes first
        for l, pol_idx, r, v in sorted(dlogs):
            if pol_idx == 0:
                w.write(f"Z_{l} {v}\n")
        for l, pol_idx, r, v in sorted(dlogs):
            if pol_idx == 1:
                w.write(f"f_{l}_{r} {v}\n")
        for l, pol_idx, r, v in sorted(dlogs):
            if pol_idx == "CONSTANT":
                w.write(f"CONSTANT {v}\n")


if __name__ == "__main__":
    logging.getLogger().setLevel(level=logging.DEBUG)
    main()
