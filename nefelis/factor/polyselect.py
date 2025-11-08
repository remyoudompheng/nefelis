"""
Selection of polynomials

Use Kleinjung2008 algorithm as described in
https://homepages.loria.fr/EThome/teaching/2022-cse-291-14/slides/cse-291-14-lecture-09.pdf
"""

import argparse
import logging
import math
from multiprocessing import Pool, Value
import time

import flint
import numpy as np

from nefelis import integers
from nefelis import polys
from nefelis import skewpoly

logger = logging.getLogger("poly")


class Polyselect:
    def __init__(self, N: int, deg: int, pmax: int, best: Value, bestsize: Value):
        self.N = N
        self.deg = deg
        self.pmax = pmax
        self.best: Value = best
        self.bestsize: Value = bestsize

    def process(self, ad: int):
        res = find_raw(self.N, self.deg, ad, self.pmax, self.best)

        best = None, None, 1e9
        for score, f, g in res:
            if score < self.bestsize.value:
                with self.bestsize.get_lock():
                    self.bestsize.value = min(self.bestsize.value, score)
            if score > self.bestsize.value + 3.0:
                continue

            skew = skewpoly.skewness(f)
            f, g, skew = size_optimize(f, g, skew)
            f = root_optimize(f, g, skew)
            if f is None:
                # sometimes all polynomials are bad
                continue
            skew = skewpoly.skewness(f)
            norm = math.log2(math.sqrt(skewpoly.l2norm(f, skew)))
            normg = math.log2(math.sqrt(skewpoly.l2norm(g, skew)))
            alpha = polys.alpha(polys.discriminant(f), f)
            score = norm + GSCORE * normg + alpha
            if score < best[2]:
                best = (f, g, score)
            if score < self.best.value:
                with self.best.get_lock():
                    self.best.value = min(self.best.value, score)
                logger.info(
                    f"f={f} g={g} log2(norm) {norm:.3f} "
                    + f"α {alpha:.3f} score {score:.3f} skew {skew:.0f}"
                )

        if best[0] is None:
            return None
        f, g, score = best
        return f, g


WORKER = None


def worker_init(N: int, d: int, pmax: int, global_best: Value, global_bestsize: Value):
    global WORKER
    WORKER = Polyselect(N, d, pmax, global_best, global_bestsize)


def worker_do(ad):
    return WORKER.process(ad)


# Ratio of norm(g) in score
GSCORE = 1.0


PARAMS3 = [
    # (Nbits, admin, admax, adstride, PMAX)
    (100, 6, 60, 6, 300),
    (120, 6, 60, 6, 500),
    (140, 6, 60, 6, 1_000),
    (160, 6, 60, 6, 4_000),
    (180, 6, 72, 6, 10_000),
    (200, 6, 96, 6, 15_000),
    (220, 6, 120, 6, 20_000),
]

# Target normf is N^(1/6)
PARAMS4 = [
    # (Nbits, admin, admax, adstride, PMAX)
    (240, 6, 60, 6, 10_000),
    (260, 6, 60, 6, 20_000),
    (280, 12, 72, 6, 60_000),
    (300, 24, 92, 6, 100_000),
    (320, 24, 120, 6, 200_000),
    (340, 24, 240, 6, 300_000),
    # (360, 30, 2000, 30, 500_000),
    (360, 3000, 4000, 30, 500_000),
]


def polyselect(N: int, deg: int) -> tuple[list[int], list[int]]:
    best = Value("d", 1e9)
    bestsize = Value("d", 1e9)

    if deg == 3:
        PARAMS = PARAMS3
    else:
        PARAMS = PARAMS4
    _, admin, admax, adstride, pmax = min(
        PARAMS, key=lambda t: abs(t[0] - N.bit_length())
    )
    logger.info(
        f"Start polynomial selection for ad=[{admin}:{admax}:{adstride}] and {pmax=}"
    )
    pool = Pool(initializer=worker_init, initargs=(N, deg, pmax, best, bestsize))
    best_fg = None
    score_fg = 1e9
    t0 = time.monotonic()
    for item in pool.imap_unordered(worker_do, range(admin, admax + 1, adstride)):
        if item is None:
            continue
        f, g = item
        skew = skewpoly.skewness(f)
        normf = math.log2(math.sqrt(skewpoly.l2norm(f, skew)))
        normg = math.log2(math.sqrt(skewpoly.l2norm(g, skew)))
        alpha = polys.alpha(polys.discriminant(f), f)
        score = normf + GSCORE * normg + alpha
        if score < score_fg:
            logger.info(
                f"Found α(f)={alpha:.3f} normf={normf:.1f} normg={normg:.1f} f={f} g={g} score={score:.3f}"
            )
            best_fg, score_fg = (f, g), score

    dt = time.monotonic() - t0
    logger.info(f"Completed polynomial selection in {dt:.3f}s")
    return best_fg


def lemma21(N, v, u, d, ad):
    # Decompose N as ad u^4 + ... + v^4
    # Res(f, vx-u) = N
    ri = (N - ad * u**d) // v
    f = [ad]
    for i in range(d - 1, -1, -1):
        ui = u**i
        ti = (-ri * pow(v, -1, ui)) % ui
        if 2 * ti > ui:
            ti -= ui
        ai = (ri + ti * v) // ui
        # ri = (ri - ai * u**i)/v
        ri = -ti
        f.append(ai)
    f.reverse()
    return f


def roots4(N, p) -> list[int]:
    """
    Compute 4-th roots of N/ad modulo p^2
    """
    try:
        r = flint.nmod(N, p).sqrt()
    except Exception:
        return []
    rs = []
    try:
        r2 = r.sqrt()
        rs.extend([int(r2), int(-r2)])
    except Exception:
        pass
    try:
        r2 = (-r).sqrt()
        rs.extend([int(r2), int(-r2)])
    except Exception:
        pass
    lifts = []
    p2 = p * p
    for r in rs:
        if r == 0:
            continue
        r = r - (r**4 - N) * pow(4 * r**3, -1, p2)
        lifts.append(r % p2)
    return lifts


def roots_nth(N, d, p) -> list[int]:
    if d == 4:
        return roots4(N, p)
    if p % d == 1:
        return []
    inv = pow(d, -1, p - 1)
    r = pow(N, inv, p)
    assert (r**d - N) % p == 0
    p2 = p * p
    r2 = r - (r**d - N) * pow(d * r ** (d - 1), -1, p2)
    return [r2 % p2]


def size_optimize(f, g, s: float):
    # Optimal translation should be between [-2s, 2s]
    n0 = skewpoly.l2norm(f, s)
    fpoly = flint.fmpz_poly(f)
    t1, t2 = int(-2 * s), int(2 * s)
    f1 = [float(f) for f in fpoly(flint.fmpz_poly([t1, 1]))]
    f2 = [float(f) for f in fpoly(flint.fmpz_poly([t2, 1]))]
    n1 = skewpoly.l2norm(f1, skewpoly.skewness(f1))
    n2 = skewpoly.l2norm(f2, skewpoly.skewness(f2))
    while t2 - t1 > 100:
        m1 = (2 * t1 + t2) // 3
        m2 = (t1 + 2 * t2) // 3
        f1 = [float(f) for f in fpoly(flint.fmpz_poly([m1, 1]))]
        f2 = [float(f) for f in fpoly(flint.fmpz_poly([m2, 1]))]
        nn1 = skewpoly.l2norm(f1, skewpoly.skewness(f1))
        nn2 = skewpoly.l2norm(f2, skewpoly.skewness(f2))
        if nn1 < nn2:
            t2, n2 = m2, nn2
        else:
            t1, n1 = m1, nn1
    t = (t1 + t2) // 2
    f = [int(fi) for fi in fpoly(flint.fmpz_poly([t, 1]))]
    g = [g[0] + t * g[1], g[1]]
    nopt = skewpoly.l2norm(f, skew := skewpoly.skewness(f))
    # logger.debug(f"translate t={t} {n0}=>{nopt} ratio {n0 / nopt:.3f}")
    return f, g, skew


def root_optimize(f, g, s: float):
    # Look for range of k such that norm(f+kg) ~ norm(f)
    sup = max(abs(f[2]) * s**2, abs(f[1]) * s, abs(f[0]))
    bound = int(min(8e6, sup / 4 / abs(g[0])))
    if bound < 16:
        # Nothing to do
        return f
    # The interval size is O(skew) and f[0] << sup
    S = np.zeros(2 * bound, dtype=np.float32)
    # print(bound, f[0] / abs(g[0]))
    discs = []
    for i in range(200):
        fi = [f[0] + (i - bound) * g[0], f[1] + (i - bound) * g[1]] + f[2:]
        discs.append(polys.discriminant(fi))
    for l in polys.SMALLPRIMES:
        if l <= 5:
            ll = l**3
        else:
            ll = l
        alpha_l = []
        for i in range(-bound, -bound + ll):
            fi = [f[0] + i * g[0], f[1] + i * g[1]] + f[2:]
            if len(f) == 4:
                fi_arr = np.array(fi, dtype=object)
                v = polys.avgval3(
                    discs[i + bound], fi[0], fi[1], fi[2], fi[3], l, fi_arr
                )
            elif len(f) == 5:
                v = polys.avgval4(discs[i + bound], l, fi)
            elif len(f) == 6:
                v = polys.avgval5(discs[i + bound], l, fi)
            else:
                raise NotImplementedError
            alpha_l.append(v)
        alpha_l = np.array(alpha_l, dtype=np.float32) * math.log2(l) * l / (l + 1)
        S += np.tile(alpha_l, (2 * bound) // ll + 1)[: len(S)]
    best = (S > (np.max(S) - 0.5)).nonzero()[0] - bound
    assert len(best) > 0
    bestf = None
    bestalpha = 0
    for idx in best:
        fi = [f[0] + int(idx) * g[0], f[1] + int(idx) * g[1]] + f[2:]
        # Norm is unchanged
        # skew = skewpoly.skewness(fi)
        # norm = math.log2(math.sqrt(skewpoly.l2norm(fi, skew)))
        alpha = polys.alpha(polys.discriminant(fi), fi)
        if alpha < bestalpha:
            # Usually there are always bad ideals: we want only mild singularities
            if bads := polys.bad_ideals([int(c) for c in fi]):
                # if any(typ == polys.BadType.COMPLEX for _, _, typ in bads):
                # logger.warning(
                #    f"Skipping interesting polynomial f {fi} with bad primes {bads}"
                # )
                continue
            bestf, bestalpha = fi, alpha

    return bestf


def find_raw(N, d: int, ad, pmax: int, global_best: Value):
    # Find u (small) such that N=v^4 mod u^2 with v very close to N^1/4 and v >> u
    # then N = v^4 + B v^2 u^2 + C v u^3 + D u^4
    # and B ~= (N - v^4) / v^2 u^2
    # if v = N^1/4 + eps, B ~ 4 N^1/4 eps / N^1/2 u^2

    NN = d**d * ad ** (d - 1) * N
    # FIXME: choose according to size of N
    BOUND = pmax**2
    # Generate "special-q"
    qrs = []
    # Select auxiliary q to reduce skew
    for q in range(503, 700, 4):
        if not flint.fmpz(q).is_probable_prime():
            continue
        if d % q == 0 or NN % q == 0:
            continue
        for qr in roots_nth(NN, d, q):
            assert (qr**d - NN) % (q * q) == 0
            qrs.append((q, qr))

    # Prepare roots
    prs = []
    primes = integers.smallprimes(pmax)
    for p in primes[len(primes) // 16 :]:
        if p <= qrs[-1][0]:
            continue
        rs = roots_nth(NN, d, p)
        if rs:
            prs.append((p, rs))
    del p, q, qr

    Nroot = int(NN ** (1 / d))
    S = np.zeros(1 << 20, dtype=np.int64)
    count = 0
    for q, qr in qrs:
        # print("try", q)
        q2 = q * q
        v0 = Nroot - q2 * BOUND // 2
        v0 += qr - v0 % q2
        # assert (N - ad * v0**4) % q2 == 0
        # we are looking for v0 + kq
        for p, pr in prs:
            if p == q:
                continue
            p2 = p * p
            q2inv = pow(q2, -1, p2)
            roots = [(_r - v0) * q2inv % p2 for _r in pr]
            # assert all((v0 + q2 * r) ** d - NN) % p2 == 0 for r in roots)
            for r in roots:
                assert ((v0 + q2 * r) ** d - NN) % p2 == 0
                vals = np.arange(
                    v0 + q2 * r - Nroot,
                    v0 + q2 * BOUND - Nroot,
                    q2 * p2,
                    dtype=np.int64,
                )
                if len(vals) == 0:
                    continue
                if count + len(vals) >= len(S):
                    S = np.resize(S, len(S) * 2)
                S[count : count + len(vals)] = vals
                count += len(vals)
    S = S[:count]

    # Find duplicates and small values of dv
    S.sort()
    dvs = S[(S[:-1] == S[1:]).nonzero()[0]]
    good = []
    for dv in dvs:
        # Compute polynomials
        vv = Nroot + int(dv)
        facs = integers.factor_smooth(abs(NN - vv**d), pmax.bit_length())
        u = 1
        for l, e in facs:
            if e & 1 == 0 and (d * ad % l) != 0:
                u *= l ** (e // 2)
        if u < pmax:
            continue

        assert (vv**d - NN) % u**2 == 0
        # Convert to integral coefficients v=V/(d ad)
        # v = d ad V + k u
        k = vv * pow(u, -1, d * ad) % (d * ad)
        v = (vv - k * u) // (d * ad)
        if math.gcd(u, v) != 1:
            continue
        f = lemma21(N, u, v, d, ad)
        g = [-v, u]
        assert f[-1] == ad
        if sum(fi * v**i * u ** (d - i) for i, fi in enumerate(f)) % N != 0:
            # FIXME: should not happen!
            logger.error(f"invalid polynomial {f=} {g=}")
            continue
        skew = skewpoly.skewness(f)
        norm = math.log2(math.sqrt(skewpoly.l2norm(f, skew)))
        normg = math.log2(math.sqrt(skewpoly.l2norm(g, skew)))
        size = norm + GSCORE * normg
        good.append((size, f, g))

    if not good:
        return []
    good.sort()
    best_score = good[0][0]
    results = []
    for score, f, g in good:
        if score - 2.0 > best_score:
            break
        results.append((score, f, g))
    return results


def main():
    argp = argparse.ArgumentParser()
    argp.add_argument("-v", action="store_true")
    argp.add_argument("N", type=int)
    argp.add_argument("DEGREE", type=int)
    args = argp.parse_args()

    logging.getLogger().setLevel(logging.INFO)
    logger.setLevel(logging.INFO)
    if args.v:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.setLevel(logging.DEBUG)
    f, g = polyselect(args.N, args.DEGREE)
    print("f", f)
    print("g", g)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logging.getLogger().setLevel(logging.INFO)
    main()
