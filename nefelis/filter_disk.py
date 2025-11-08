"""
Pruning implementation using disk files to save memory.

This implementation can use multiple threads to compute statistics.
"""

import itertools
import logging
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

from networkx import Graph, connected_components

logger = logging.getLogger("filter")

CHUNKS = 8


def prune(filepath: str):
    f = filepath
    t0 = time.monotonic()
    singles(f, f + ".pruned.tmp")
    singles(f + ".pruned.tmp", f + ".pruned.singles")
    cliques(f + ".pruned.singles", f + ".pruned")
    cliques(f + ".pruned", f + ".pruned.tmp")
    cliques(f + ".pruned.tmp", f + ".pruned", aggressive=True)
    os.remove(f + ".pruned.tmp")
    dt = time.monotonic() - t0
    logger.info(f"Pruning completed in {dt:.3f}s")


def process_chunk(f, off1, off2):
    seenf = dict()
    seeng = dict()
    fd = open(f, encoding="ascii")
    fd.seek(off1)
    while True:
        idx = fd.tell()
        if idx >= off2:
            break
        line = fd.readline()
        if line.startswith("#"):
            continue
        xy, facf, facg = line.strip().split(":")
        x, _, y = xy.partition(",")
        x, y = int(x), int(y)
        for w in facf.split(","):
            if len(w) <= 4:
                continue
            if w in seenf:
                seenf[w] = None
            else:
                seenf[w] = idx
        # g is always the algebraic side
        for w in facg.split(","):
            if len(w) <= 4:
                continue
            # compute more precise primes
            l = int(w, 16)
            if y % l == 0:
                r = l
            else:
                r = x * pow(y, -1, l) % l
            w = l * l + r
            if w in seeng:
                seeng[w] = None
            else:
                seeng[w] = idx
    return seenf, seeng


def singles(f, fnew):
    chunks = []
    sz = os.stat(f).st_size
    fd = open(f)
    for i in range(1, CHUNKS):
        # Use slightly unbalanced chunks to
        fd.seek(sz * (i * CHUNKS + i * i) // (2 * CHUNKS * CHUNKS))
        fd.readline()
        chunks.append(fd.tell())
    chunks.append(sz)
    # print(f"step {step} chunksize {chunks}")

    with ProcessPoolExecutor() as pool:
        jobs = []
        for i in range(CHUNKS):
            off1 = 0 if i == 0 else chunks[i - 1]
            off2 = chunks[i]
            jobs.append(pool.submit(process_chunk, f, off1, off2))

        seenf = dict()
        seeng = dict()
        for i, j in enumerate(jobs):
            fdict, gdict = j.result()
            jobs[i] = None
            del j
            # print(f"chunk {i} done")
            if i == 0:
                seenf, seeng = fdict, gdict
                continue
            for l, idx in fdict.items():
                if l in seenf:
                    seenf[l] = None
                else:
                    seenf[l] = idx
            for l, idx in gdict.items():
                if l in seeng:
                    seeng[l] = None
                else:
                    seeng[l] = idx

    singletons = set(
        idx
        for idx in itertools.chain(seenf.values(), seeng.values())
        if idx is not None
    )
    del seenf, seeng
    logger.info(f"Removing {len(singletons)} relations with singleton primes")

    with open(f, "rb") as fd, open(fnew, "wb") as w:
        while True:
            off = fd.tell()
            l = fd.readline()
            if not l:
                break
            if l.startswith(b"#"):
                continue
            if off in singletons:
                continue
            w.write(l)


# Cliques removal


def process_chunk2(f, off1, off2):
    seenf = dict()
    seeng = dict()
    lengths = {}
    fd = open(f, encoding="ascii")
    fd.seek(off1)
    while True:
        idx = fd.tell()
        if idx >= off2:
            break
        line = fd.readline()
        if line.startswith("#"):
            continue
        xy, facf, facg = line.strip().split(":")
        x, _, y = xy.partition(",")
        x, y = int(x), int(y)
        lengths[idx] = facf.count(",") + facg.count(",")
        for w in facf.split(","):
            if w in seenf:
                t = seenf[w]
                if t is not None and len(t) == 1:
                    seenf[w] = (t[0], idx)
                else:
                    seenf[w] = None
            else:
                seenf[w] = (idx,)

        # g is always the algebraic side
        for w in facg.split(","):
            # compute more precise primes
            l = int(w, 16)
            if y % l == 0:
                r = l
            else:
                r = x * pow(y, -1, l) % l
            w = l * l + r

            if w in seeng:
                t = seeng[w]
                if t is not None and len(t) == 1:
                    seeng[w] = (t[0], idx)
                else:
                    seeng[w] = None
            else:
                seeng[w] = (idx,)

    return seenf, seeng, lengths


def cliques(f, fnew, aggressive=False):
    chunks = []
    sz = os.stat(f).st_size
    fd = open(f)
    for i in range(1, CHUNKS):
        fd.seek(sz * (i * CHUNKS + i * i) // (2 * CHUNKS * CHUNKS))
        fd.readline()
        chunks.append(fd.tell())
    chunks.append(sz)
    # print(f"cliques chunksize {chunks}")

    with ProcessPoolExecutor() as pool:
        jobs = []
        for i in range(CHUNKS):
            off1 = 0 if i == 0 else chunks[i - 1]
            off2 = chunks[i]
            jobs.append(pool.submit(process_chunk2, f, off1, off2))

        seenf = dict()
        seeng = dict()
        lengths = {}
        for i, j in enumerate(jobs):
            fdict, gdict, ldict = j.result()
            jobs[i] = None
            del j
            # print(f"chunk {i} done")
            if i == 0:
                seenf, seeng, lengths = fdict, gdict, ldict
                continue
            lengths.update(ldict)
            for l, idxs in fdict.items():
                if l in seenf and idxs is not None:
                    t = seenf[l]
                    if t is None or len(t) + len(idxs) > 2:
                        seenf[l] = None
                    else:
                        seenf[l] = t + idxs
                else:
                    seenf[l] = idxs
            for l, idxs in gdict.items():
                if l in seeng and idxs is not None:
                    t = seeng[l]
                    if t is None or len(t) + len(idxs) > 2:
                        seeng[l] = None
                    else:
                        seeng[l] = t + idxs
                else:
                    seeng[l] = idxs

    # Find singletons and cliques
    pruned = set()
    g = Graph()
    for idxs in itertools.chain(seenf.values(), seeng.values()):
        match idxs:
            case (idx,):
                pruned.add(idx)
            case (i1, i2):
                g.add_edge(i1, i2)

    def score(clique):
        return sum(lengths[idx] for idx in clique)

    n_singles = len(pruned)
    excess = len(lengths) - len(seenf) - len(seeng)
    # print("excess", excess)
    cliques = list(connected_components(g))
    cliques.sort(key=score, reverse=True)
    # print(len(pruned), "singletons")
    # print(len(cliques))
    max_removed = (excess - 200) // 2
    if aggressive:
        max_removed = max(max_removed, excess - 50000)
    for c in cliques[:max_removed]:
        pruned.update(c)
    r2 = sum(len(c) for c in cliques[:max_removed])
    # print(r2, "relations in", len(cliques[:max_removed]), "cliques")
    # print("pruning", len(pruned), "relations")
    logger.info(
        f"Pruning {len(pruned)} relations ({n_singles} singletons"
        + f" and {r2} in {len(cliques[:max_removed])} cliques)"
    )

    with open(f) as fd, open(fnew, "w") as w:
        while True:
            off = fd.tell()
            l = fd.readline()
            if not l:
                break
            if l.startswith("#"):
                continue
            if off in pruned:
                continue
            w.write(l)


def main():
    fbase = sys.argv[1]
    f = fbase

    prune(f, f + ".tmp")
    prune(f + ".tmp", f + ".tmp2")
    cliques(f + ".tmp2", f + ".test")
    cliques(f + ".test", f + ".test2")
    cliques(f + ".test2", f + ".test3", aggressive=True)


if __name__ == "__main__":
    main()
