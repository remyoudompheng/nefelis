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

from typing import Iterator

import argparse
import json
import logging
import pathlib
import random
import time

import flint

from nefelis import filter
from nefelis.linalg_impl import SpMV

DEBUG_RELS = False

# If enabled, ignore relations between conjugate ideals.
# This can be used to validate the optimization.
DEBUG_IGNORE_CONJUGATES = True

logger = logging.getLogger("linalg")


def read_relations(
    filepath: str | pathlib.Path,
) -> Iterator[tuple[int, int, list[int], list[int]]]:
    with open(filepath) as f:
        for line in f:
            if line.startswith("#"):
                continue
            xy, facg_, facf_ = line.strip().split(":")
            x, _, y = xy.partition(",")
            facg = [int(w, 16) for w in facg_.split(",")]
            facf = [int(w, 16) for w in facf_.split(",")]
            yield int(x), int(y), facg, facf


def schirokauer_place(f, ell, root=None):
    """
    Returns the root of f modulo l^2
    """
    if root is None:
        rs = flint.fmpz_mod_poly(f, flint.fmpz_mod_poly_ctx(ell)).roots()
        assert len(rs) >= 1
        root = rs[0][0]

    # Hensel lift the root of f modulo ell^2
    # r -> r - f(r)/f'(r)
    sm_place = flint.fmpz_mod_ctx(ell**2)
    r = flint.fmpz_mod(int(root), sm_place)
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
        "--blockw", default=1, type=int, help="Use Block Wiedemann with size m=ARG n=1"
    )
    argp.add_argument("WORKDIR")
    args = argp.parse_args()
    main_impl(args)


def main_impl(args):
    workdir = pathlib.Path(args.WORKDIR)

    with open(workdir / "args.json") as f:
        doc = json.load(f)
        n = doc["n"]

    ell0 = int(flint.fmpz(n - 1).factor()[-1][0])
    ell1 = int(flint.fmpz(n + 1).factor()[-1][0])
    process(workdir, doc, ell0, args.blockw)
    process(workdir, doc, ell1, args.blockw)


class Field:
    """
    A simplified representation of an algebraic number field.
    It stores a list of prime ideals with string identifiers.
    """

    # Conjugation automorphism: (l, r1) -> (l, r2)
    conjugates: dict[tuple[int, int], tuple[int, int]]

    def __init__(self, poly, D, gj, prefix: str):
        # To represent the conjugation automorphism, we need to know
        # the polynomial gj = A X^2 + B X + C
        # where A, B, C are elements of Z[sqrt(-D)]
        self.poly = poly
        self.gj = gj
        self.D = D
        self.roots = {}
        self.names = {}
        self.conjugates = {}
        self.prefix = prefix

    def key(self, l, r) -> str:
        return f"{self.prefix}_{l}_{r}"

    def canonical_conj(self, l: int, r: int) -> tuple[str, int]:
        """
        Returns a canonical choice of either (l, r) or its conjugate (l, rbar)
        and a coefficient -1 if the choice is the conjugate of the caller's
        ideal.

        The choice is the root represented by the smallest nonnegative integer.
        """
        _, r2 = self.conjugate(l, r)
        if r2 < r:
            return self.key(l, r2), -1

        return self.key(l, r), 1

    def conjugate(self, l: int, r: int) -> tuple[int, int] | None:
        """
        >>> K = Field([27, 0, 26, 0, 6], 7, [[-1, 1], [0, 0], [-1, 2]], "f")
        >>> K.conjugate(3, 0)
        (3, 0)
        """
        if (l, r) not in self.conjugates:
            if len(self.poly) == 3:
                lroots = flint.nmod_poly(self.poly, l).roots()
                # g has degree 2, conjugation is easy
                if len(lroots) == 1 and lroots[0][1] == 2:
                    # double root
                    r1 = lroots[0][0]
                    self.conjugates[(l, int(r1))] = (l, int(r1))
                elif len(lroots) == 2:
                    (r1, _), (r2, _) = lroots
                    self.conjugates[(l, int(r1))] = (l, int(r2))
                    self.conjugates[(l, int(r2))] = (l, int(r1))
                elif len(lroots) == 1 and self.poly[2] % l == 0:
                    # 2 roots, 1 is infinity
                    r1 = lroots[0][0]
                    self.conjugates[(l, int(r1))] = (l, l)
                    self.conjugates[(l, l)] = (l, int(r1))
                elif len(lroots) == 0 and self.poly[2] % l == 0:
                    # 1 root at infinity
                    self.conjugates[(l, l)] = (l, l)

            else:
                # compute conjugacy maps: D is necessarily a square modulo l
                # r is a root of either gj(j) or gj(jbar)
                xC, yC = self.gj[0]
                xB, yB = self.gj[1]
                xA, yA = self.gj[2]
                gx = xA * r**2 + xB * r + xC
                gy = yA * r**2 + yB * r + yC
                # the conjugation is trace(r) == -B/A
                j = flint.nmod(self.D, l).sqrt()
                if gx + j * gy != 0:
                    j = -j
                    assert gx + j * gy == 0
                zA, zB, _zC = xA + j * yA, xB + j * yB, xC + j * yC
                tr = -zB / zA
                rbar = tr - r
                self.conjugates[(l, int(rbar))] = (l, int(r))
                self.conjugates[(l, int(r))] = (l, int(rbar))

        return self.conjugates[(l, r)]


def process(workdir, args, ell: int, blockw: int = 1):
    """
    Perform the linear algebra step modulo a prime ell.

    A dedicated working directory is created for each prime.
    """
    # Does it divide the order of Fp* or the norm 1 subgroup of Fp²* ?
    n = args["n"]
    f = args["f"]
    g = args["g"]
    D = args["D"]
    gj = args["gj"]
    conway = args["conway"]

    ZnX = flint.fmpz_mod_poly_ctx(n)
    Fp2 = flint.fq_default_ctx(n, 2, var="i", modulus=ZnX(conway))
    Fp2X = flint.fq_default_poly_ctx(Fp2)
    z = Fp2(args["z"])
    assert Fp2X(f)(z) == 0
    assert Fp2X(g)(z) == 0

    IS_FP = n % ell == 1

    rels = []
    seen_xy = set()
    duplicates = 0

    t0 = time.monotonic()
    Kf = Field(f, D, gj, "f")
    Kg = Field(g, D, gj, "g")

    assert D > 0

    sm_root = None
    if IS_FP:
        sm_root = None
        logger.info(f"Logarithms modulo {ell=} (in GF(p))")
        # The Schirokauer map is given by sqrt(D) mod ell^2
        modell2 = flint.fmpz_mod_ctx(ell**2)
        sm_root = modell2(int(flint.fmpz_mod(D, flint.fmpz_mod_ctx(ell)).sqrt()))
        sm_root -= (sm_root**2 - D) / (2 * sm_root)
        assert sm_root**2 == D
        logger.info(f"Schirokauer map will use root {sm_root} mod l^2")
        # image of gj in polynomials modulo l^2
        gell = [_x + sm_root * _y for _x, _y in gj]
    else:
        logger.info(f"Logarithms modulo {ell=} (in norm 1 subgroup of GF(p²))")
        logger.info("Schirokauer map is ignored")

    for x, y, facg, facf in read_relations(workdir / "relations.sieve"):
        if (x, y) in seen_xy:
            duplicates += 1
            continue

        seen_xy.add((x, y))

        # z = factor(f(z), Kf) = factor(g(z), Kg) / leading(g)
        # For the norm 1 subgroup, z/zbar makes the constant disappear
        if IS_FP:
            rel = {"CONSTANT": 1}
        else:
            # Constant is on the g side, set coefficient -1 for consistency
            rel = {"CONSTANT": -1}
        for _l in facf:
            _r = x * pow(y, -1, _l) % _l if y % _l else _l
            key0 = f"f_{_l}_{_r}"
            if not DEBUG_IGNORE_CONJUGATES:
                # logarithms are conjugation-invariant
                key, e = Kf.canonical_conj(_l, int(_r))
            else:
                _, _r2 = Kf.conjugate(_l, int(_r))
                key, e = key0, 1
            if IS_FP:
                # logarithms are conjugation-invariant
                e = 1
            # print("relation key",key)
            rel[key] = rel.get(key, 0) + e
        for _l in facg:
            _r = x * pow(y, -1, _l) % _l if y % _l else _l
            key0 = f"g_{_l}_{_r}"
            if not DEBUG_IGNORE_CONJUGATES:
                key, e = Kg.canonical_conj(_l, int(_r))
            else:
                _, _r2 = Kg.conjugate(_l, int(_r))
                key, e = key0, 1
            if IS_FP:
                # logarithms are conjugation-invariant
                e = 1
            rel[key] = rel.get(key, 0) - e
        # Corresponding algebraic integer is x-yω
        if IS_FP:
            # Compute the norm of x-yω in the quadratic subfield
            vf = gell[0] * y**2 + gell[1] * x * y + gell[2] * x**2
            # Then apply Schirokauer map
            sm1 = int(vf ** (ell - 1))
            assert int(sm1) % ell == 1
            rel["SM"] = sm1 // ell

        rels.append(rel)

    dt = time.monotonic() - t0
    logger.info(
        f"Computed factorizations and SM maps for {len(rels)} relations in {dt:.1f}s"
    )

    if duplicates:
        logger.info(f"{duplicates} duplicate results in input file, ignored")

    subdir = pathlib.Path(workdir) / f"subgroup.{ell}"
    subdir.mkdir(exist_ok=True)
    rels2, _ = filter.prune(rels, subdir)
    rels3 = filter.filter(rels2, subdir)

    M = SpMV(rels3, ell)
    basis = M.basis
    dim = M.dim
    assert dim >= len(basis)

    poly = M.wiedemann_big(ell, blockm=blockw)
    logger.info(f"Computed characteristic poly {poly[:3]}...{poly[-3:]}")

    poly = [ai * pow(poly[-1], -1, ell) % ell for ai in poly]
    assert any(ai != 0 for ai in poly)
    assert len(poly) <= dim + 1 and poly[0] == 0, (dim, len(poly), poly[0])

    i0 = next(i for i, ai in enumerate(poly) if ai)
    logger.info(f"Polynomial (degree {len(poly) - 1}) is divisible by X^{i0}")

    poly_k = poly[1:]
    wi = [random.randrange(ell) for _ in range(dim)]
    ker = M.polyeval(wi, ell, poly_k)
    assert any(k for k in ker)

    if any(k for k in ker[len(basis) :]):
        kdim = len(set(ker[len(basis) :]))
        logger.info(f"Kernel is inaccurate (extra dimensions {kdim})")
        # FIXME: this will usually fail

    # Validate result
    prime_idx = {l: idx for idx, l in enumerate(basis)}
    for i, ridx in enumerate(M.rowidx):
        r = rels3[ridx]
        assert sum(e * ker[prime_idx[l]] for l, e in r.items()) % ell == 0, (
            f"row {i} (rel {ridx})"
        )

    logger.info("Checked element of matrix right kernel")

    assert len(basis) <= len(ker), (len(basis), dim, len(ker))

    # Build dlog database
    dlog: dict[str, int] = {l: v for l, v in zip(basis, ker)}

    added, nremoved = 0, 0
    with open(subdir / "relations.removed") as fd:
        for line in fd:
            nremoved += 1
            key, _, facs = line.partition("=")
            key = key.strip()
            rel = {p: int(e) for p, _, e in (_f.partition("^") for _f in facs.split())}
            # Relations from Gaussian elimination are naturally ordered
            # in a triangular shape.
            if all(_l in dlog for _l in rel):
                v = sum(_e * dlog[_l] for _l, _e in rel.items())
                dlog[key] = v % ell
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

    def key_l(k: str) -> int:
        for w in k.split("_"):
            if w.isdigit():
                return int(w)
        return 0

    dlogs = [(key_l(k), k, v) for k, v in dlog.items()]
    with open(subdir / "dlog.tmp", "w") as w:
        for _, k, v in sorted(dlogs):
            w.write(f"{k} {v}\n")

    # Check relations again
    for rel in rels:
        if all(k in dlog for k in rel):
            v = sum(_e * dlog[_p] for _p, _e in rel.items())
            assert v % ell == 0, (v, rel)

    f_primes: dict[int, list[str]] = {}
    g_primes: dict[int, list[str]] = {}
    for key in dlog:
        if key in ("CONSTANT", "SM"):
            continue
        _l = int(key.split("_")[1])
        if key.startswith("f"):
            f_primes.setdefault(_l, []).append(key)
        elif key.startswith("g"):
            g_primes.setdefault(_l, []).append(key)

    # Check the logs of conjugates
    sign = 1 if IS_FP else -1
    n_conj = 0
    for (l, r1), (_, r2) in Kf.conjugates.items():
        k1 = f"f_{l}_{r1}"
        k2 = f"f_{l}_{r2}"
        if k1 in dlog and k2 in dlog:
            assert dlog[k1] % ell == (sign * dlog[k2]) % ell, (
                k1,
                k2,
                dlog[k1],
                dlog[k2],
            )
            n_conj += 1
    for (l, r1), (_, r2) in Kg.conjugates.items():
        k1 = f"g_{l}_{r1}"
        k2 = f"g_{l}_{r2}"
        if k1 in dlog and k2 in dlog:
            assert dlog[k1] % ell == (sign * dlog[k2]) % ell, (k1, k2)
            n_conj += 1
    logger.info(f"Checked {n_conj} conjugate ideal pairs")

    if True:
        # Fill logs of conjugates
        sign = 1 if IS_FP else -1
        n_conj = 0
        for l, r1 in sorted(Kf.conjugates):
            k1 = f"f_{l}_{r1}"
            if k1 in dlog:
                continue
            _, r2 = Kf.conjugate(l, r1)
            k2 = f"f_{l}_{r2}"
            if k2 in dlog:
                dlog[k1] = (sign * dlog[k2]) % ell
                n_conj += 1
        for l, r1 in sorted(Kg.conjugates):
            k1 = f"g_{l}_{r1}"
            if k1 in dlog:
                continue
            _, r2 = Kg.conjugate(l, r1)
            k2 = f"g_{l}_{r2}"
            if k2 in dlog:
                dlog[k1] = (sign * dlog[k2]) % ell
                n_conj += 1
        logger.info(f"Added {n_conj} conjugate ideals")

    if IS_FP:
        gen = None

        # We can easily use norms of g to check results
        for l in sorted(g_primes):
            keys = g_primes[l]
            if len(keys) == 2:
                assert dlog[keys[0]] == dlog[keys[1]]
            # l = (l,r)(l,rbar) = (l,r)^2
            dlog_l = 2 * dlog[keys[0]]
            dlog[f"Z_{l}"] = dlog_l % ell
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
    else:
        for l in sorted(g_primes):
            keys = g_primes[l]
            if len(keys) == 2:
                assert (dlog[keys[0]] + dlog[keys[1]]) % ell == 0, keys

        coell = (n + 1) // ell
        gen = None
        checked = 0
        # An exponent realizing the projection on the order ell subgroup
        projexp = (n - 1) * coell * pow((n - 1) * coell, -1, ell)
        assert projexp % ell == 1
        for x, y, facg, _ in read_relations(workdir / "relations.sieve"):
            logxy = dlog["CONSTANT"]
            facs = []
            try:
                for _l in facg:
                    _r = x * pow(y, -1, _l) % _l if y % _l else _l
                    key = f"g_{_l}_{_r}"
                    logxy += dlog[key]
                    facs.append((key, e))
            except KeyError:
                # print("skip")
                continue
            assert (
                g[2] * (x - y * z) ** (n + 1)
                == g[2] * x**2 + g[1] * x * y + g[0] * y**2
            )
            # continue
            xy1 = (x - y * z) ** projexp
            assert xy1**ell == 1
            if xy1 == 1:
                assert logxy % ell == 0
            elif gen is None:
                # print(xy1, logxy)
                gen = xy1 ** int(pow(logxy, -1, ell))
                assert xy1 == gen**logxy
                assert gen**projexp == gen
            else:
                assert xy1 == gen**logxy
            checked += 1
        logger.info(f"Checked logarithms for {checked} norm 1 elements")
        logger.info(f"Logarithm base is {gen}")
        # Write generator to file
        with open(subdir / "gen", "w") as w:
            genx, geny = gen.to_list()
            w.write(f"{genx},{geny}\n")

    dlogs = [(key_l(k), k, v) for k, v in dlog.items()]
    with open(subdir / "dlog", "w") as w:
        for _, k, v in sorted(dlogs):
            w.write(f"{k} {v}\n")


if __name__ == "__main__":
    logging.getLogger().setLevel(level=logging.DEBUG)
    main()
