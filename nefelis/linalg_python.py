"""
Pure Python implementation of linear algebra.
"""

import logging
import random
import time

from nefelis import lingen


class Matrix:
    def __init__(self, rels):
        primes = set()
        for r in rels:
            primes.update(r)
        self.dim = len(rels)
        self.basis = sorted(primes)
        self.index = {p: i for i, p in enumerate(self.basis)}
        self.rows = [[(self.index[p], e) for p, e in r.items()] for r in rels]

    def charpoly(self, ell: int):
        v = [random.randrange(ell) for _ in self.basis]
        seq = [v[0]]
        w = v.copy()

        t0 = time.monotonic()
        t_print = t0
        for i in range(1, 2 * len(self.basis) + 64):
            ww = [sum(e * w[lidx] for lidx, e in r) % ell for r in self.rows]
            seq.append(ww[0])
            w = ww

            if (t1 := time.monotonic()) > t_print + 10.0:
                # print progress every 10 seconds
                elapsed = t1 - t0
                speed = i / elapsed
                logging.info(
                    f"{i} matrix muls done in {elapsed:.1f}s ({speed:.1f} SpMV/s)"
                )
                t_print = t1

        t1 = time.monotonic()
        poly = lingen.lingen([seq], self.dim, ell)
        lingen_dt = time.monotonic() - t1
        logging.info(f"Lingen completed in {lingen_dt:.3f}s (N={self.dim} m=1 n=1)")

        return [int(ai) for ai in list(poly)]

    def wiedemann_big(self, ell: int):
        return self.charpoly(ell)

    def polyeval(self, v, ell: int, poly: list[int]) -> list[int]:
        result = [poly[0] * vi for vi in v]
        w = v.copy()

        t0 = time.monotonic()
        t_print = t0
        for k, ak in enumerate(poly):
            if k == 0:
                continue
            ww = [sum(e * w[lidx] for lidx, e in r) % ell for r in self.rows]
            w = ww
            for i, wi in enumerate(w):
                result[i] += ak * wi

            if (t1 := time.monotonic()) > t_print + 10.0:
                # print progress every 10 seconds
                elapsed = t1 - t0
                speed = i / elapsed
                logging.info(
                    f"{i} matrix muls done in {elapsed:.1f}s ({speed:.1f} SpMV/s)"
                )
                t_print = t1

        return [zi % ell for zi in result]
