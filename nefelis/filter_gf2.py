"""
Variant of merge/filter for linear algebra modulo 2.

For factoring, we don't need to save purged/deleted relations,
they will not be used.

The convention is to represent relations as sets of integers:
* nonnegative integers represent the ideal/character basis elements
* negative integers represent the original relation indices (for sqrt step)
"""

import logging
import pathlib
import time

import numpy as np

logger = logging.getLogger("filter")

DEBUG_MERGE = False


def filter(rels: list, datadir: pathlib.Path | None):
    dim = max(np.max(r) for r in rels)
    logger.info(f"Max column index {dim}")

    t0 = time.monotonic()
    Ds = [2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 18, 20]
    for d in Ds:
        rels = merge(rels, d, dim + 1)
        if sum(len(r) for r in rels) > len(rels) * 100:
            break
    rels = clear_excess(rels, dim + 1)
    logger.info(f"Filtering/merge done in {time.monotonic() - t0:.1f}s")

    if datadir is not None:
        with open(datadir / "relations.filtered", "w") as w:
            for r in rels:
                line = " ".join(str(l) for l in sorted(r))
                w.write(line)
                w.write("\n")
            logger.info(f"{len(rels)} relations written to {w.name}")

    return rels


def merge(rels, d, imax: int):
    dense_limit = 100

    t0 = time.monotonic()

    # Compute stats
    counts = np.zeros(imax, dtype=np.uint32)
    for r in rels:
        counts[r[r >= 0]] += 1

    # Find mergeable primes
    md = ((0 < counts) & (counts <= d)).nonzero()[0]
    sets = [None] * imax
    for i in md:
        sets[i] = []
    for ridx, r in enumerate(rels):
        rpos = r[r >= 0]
        pivots = rpos[counts[rpos] <= d]
        for p in pivots:
            sets[p].append(ridx)

    if DEBUG_MERGE:
        for i in range(len(md)):
            assert counts[md[i]] == len(sets[i]), (md[i], counts[md[i]], sets[i])

    if len(md) == 0:
        return

    nr = len(rels)
    nc = np.count_nonzero(counts)
    avgw = np.sum(counts) / len(rels)
    assert nr > nc

    def weight(r):
        return np.count_nonzero(counts[r[r >= 0]] < dense_limit)

    pivots = 0
    # Track modified relations, we skip a pivot if any row is modified.
    pivotedp = np.zeros(imax + 1, dtype=np.uint8)
    md = sorted(md, key=lambda i: counts[i])
    for p in md:
        # Beware: relations are modified during iteration
        # We must skip p if it may appear in a relation outside sets[p]
        # This happens if p appears in a relation used as a pivot.
        if pivotedp[p]:
            continue
        rs = sets[p]
        rs = [ridx for ridx in rs if rels[ridx] is not None]
        if not rs:
            continue
        rs.sort(key=lambda ridx: weight(rels[ridx]))
        pividx = rs[0]
        piv = rels[pividx]
        assert np.isin(p, piv)
        rels[pividx] = None
        for ridx in rs[1:]:
            r = rels[ridx]
            if r is None:
                continue
            rp = np.setxor1d(r, piv, assume_unique=True)
            # If relation becomes empty, remove it
            rels[ridx] = rp if len(rp) > 0 else None
        pivotedp[piv[piv >= 0]] = 1
        pivots += 1

    dt = time.monotonic() - t0
    logger.info(
        f"{d}-merge: {nc} columns {nr} rows excess={nr - nc} weight={avgw:.3f} pivots={pivots}/{len(md)} dt={dt:.1f}s"
    )

    rels = [_r for _r in rels if _r is not None]
    return rels


def clear_excess(rels, imax: int):
    # Compute stats
    counts = np.zeros(imax, dtype=np.uint32)
    for r in rels:
        counts[r[r >= 0]] += 1

    nr = len(rels)
    nc = np.count_nonzero(counts)
    assert nr > nc

    MIN_EXCESS = 512
    excess = nr - nc
    if excess <= MIN_EXCESS:
        return rels

    # Remove heavy-weighted relations first
    scores = []
    for ridx, r in enumerate(rels):
        rpos = r[r >= 0]
        score = np.count_nonzero(counts[rpos] < 100)
        scores.append((score, ridx))
    scores.sort(reverse=True)

    to_remove = excess - MIN_EXCESS
    to_purge = scores[:to_remove]
    for _, ridx in to_purge:
        rels[ridx] = None

    logger.info(
        f"Purged {to_remove} relations with score {to_purge[-1][0]}..{to_purge[0][0]}"
    )
    return [r for r in rels if r is not None]
