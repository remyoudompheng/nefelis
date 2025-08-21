"""
Computation of linear generator for 1 or several sequences.

This file implements the linear generator step of the Block Wiedemann algorithm.
Due to the large memory requirement of the implementation, and the quadratic
complexity (w.r.t. the block size), it is recommended to use small values
of m especially when CPU/GPU resources are unbalanced.

Basic support for multithreading is implemented by computing products
of polynomial matrices in parallel. It requires custom FLINT bindings
to avoid the Python GIL. Due to implementation details in FLINT
this may greatly increase memory usage.

Approximate memory usage for a 384-bit modulus and N=200000:
* for m=1 about 1GiB (FLINT implementation based on HGCD)
* for m=2 about 1-4GiB (this implementation, multithreaded)
* for m=3 about 2-5GiB (this implementation, multithreaded)
* for m=4 about 3-7GiB (this implementation, multithreaded)

The implementation follows the conventions of:

Emmanuel Thomé
Fast computation of linear generators for matrix sequences and application to the block Wiedemann algorithm.
ISSAC '01: Proceedings of the 2001 international symposium on Symbolic and algebraic computation, Jul 2001, London, Ontario, Canada. pp.323-331,
https://inria.hal.science/inria-00517999
"""

from concurrent.futures import ThreadPoolExecutor
import numpy
import random

import flint
import nefelis_flint_extras


def lingen(sequences, N: int, l: int):
    """
    Compute a (simultaneous) linear generating polynomial for input sequences.

    The polynomial sum a[i] X^i is such that:
       sum a[i] seq[k+i] = 0 for all k

    N: the expected degree of polynomial (matrix dimension in Block Wiedemann)
    l: the (prime) modulus
    """
    m = len(sequences)
    if m == 1:
        seq = sequences[0]
        assert len(seq) > 2 * N + 1
        ctx = flint.fmpz_mod_poly_ctx(l)
        return ctx.minpoly(seq)

    assert all(len(s) > N + N // m + 16 for s in sequences)

    ZpX = flint.fmpz_mod_poly_ctx(l)
    M = numpy.zeros((m, 1), dtype=object)
    for i, seq in enumerate(sequences):
        M[i, 0] = flint.fmpz_mod_poly(seq, ZpX)

    F = numpy.zeros((1, 1 + m), dtype=object)
    for i in range(m):
        f_rnd = [random.randrange(l) for _ in range(m)]
        F[0, i] = flint.fmpz_mod_poly(f_rnd, ZpX)
    F[0, m] = flint.fmpz_mod_poly(m * [0] + [1], ZpX)

    E = M @ F
    for i, j in numpy.ndindex(*E.shape):
        E[i, j] = E[i, j].right_shift(m)

    delta = tuple(m for _ in range(m + 1))
    EXTRA_ITERS = 4

    with ThreadPoolExecutor() as pool:
        P = mslgdc(E, delta, N + N // m + EXTRA_ITERS, ZpX, pool=pool)

    FP = F @ P
    minpoly = FP[0, 0]
    assert 0 < minpoly.degree() <= N
    # Adjust result to match FLINT minpoly convention
    # (reversed and monic)
    minpoly = minpoly.reverse()
    minpoly /= minpoly[minpoly.degree()]
    return minpoly


def lingen_step(E, delta, ZpX, P=None):
    # Advance by 1 degree using Gauss elimination (ALGO1 in [Thomé])
    # E is modified in place and divided by X
    # P is optionally modified in place
    m, mn = E.shape
    if P is None:
        P = numpy.array(
            [
                [flint.fmpz_mod_poly([1 if i == j else 0], ZpX) for j in range(mn)]
                for i in range(mn)
            ],
            dtype=object,
        )
    # Sort columns
    delta = list(iter(delta))
    for i in range(mn):
        argmin = min(range(i, mn), key=lambda idx: (delta[idx], idx))
        if i < argmin:
            delta[i], delta[argmin] = delta[argmin], delta[i]
            # Fancy numpy syntax for swapping
            P[:, [i, argmin]] = P[:, [argmin, i]]
            E[:, [i, argmin]] = E[:, [argmin, i]]
    # Eliminate without increasing degree (col[j] is cancelled using col[j0<j])
    nonzero = set()
    for i in range(m):
        # Find first nonzero element of row i
        try:
            j0 = next(j for j in range(mn) if E[i, j][0] and j not in nonzero)
        except StopIteration:
            continue
        nonzero.add(j0)
        # Use it to eliminate next columns
        for j in range(j0 + 1, mn):
            k = E[i, j][0] / E[i, j0][0]
            E[:, j] -= k * E[:, j0]
            P[:, j] -= k * P[:, j0]
    if False:
        for i, j in numpy.ndindex(*E.shape):
            if E[i, j][0]:
                assert j in nonzero
    # Now each row has 1 nonzero coefficient
    for j in nonzero:
        delta[j] += 1
        for i in range(mn):
            P[i, j] = P[i, j].left_shift(1)
    # Combine multiplication by X and division by X
    for j in range(mn):
        if j not in nonzero:
            for i in range(m):
                assert E[i, j][0] == 0
                E[i, j] = E[i, j].right_shift(1)
    # debug("P", P)
    return P, delta


def matmul_ij(ctx, A, B, i, j, b):
    n = A.shape[1]
    if b is None:
        res = 0
        for k in range(n):
            res += nefelis_flint_extras.nogil_fmpz_mod_poly_mul(A[i, k], B[k, j], ctx)
    else:
        res = 0
        for k in range(n):
            res += nefelis_flint_extras.nogil_fmpz_mod_poly_mullow(
                A[i, k], B[k, j], b, ctx
            ).right_shift(b // 2)
    return res


def mslgdc(E, delta, b, ZpX, depth=0, pool=None):
    """
    Returns a matrix P operating on columns (right multiplication)
    such that E * P is zero up to degree b and E[x^b] has rank m
    and delta*P is a valid profile for new polynomial degrees.

    E: a numpy.array with coefficients flint.fmpz_mod_poly
       The caller should assume that E will be destroyed or modified.
    """
    m, mn = E.shape

    p = ZpX.modulus()
    if p.bit_length() > 128:
        MSLGDC_THRESHOLD = 4
        PARALLEL_THRESHOLD = 200
    else:
        MSLGDC_THRESHOLD = 16
        PARALLEL_THRESHOLD = 100

    if b <= MSLGDC_THRESHOLD:
        P = None
        for _ in range(b):
            P, delta = lingen_step(E, delta, ZpX, P=P)
        return P

    # Handle the low degree side
    EL = numpy.zeros((m, mn), dtype=object)
    bhalf = b // 2
    for i in range(m):
        for j in range(mn):
            EL[i, j] = E[i, j].truncate(bhalf)
    PL = mslgdc(EL, delta, b // 2, ZpX, depth=depth + 1, pool=pool)
    del EL

    # Compute right part
    ER = numpy.zeros((m, mn), dtype=object)
    # (E @ PL).truncate(b).right_shift(b // 2)
    if pool and b > PARALLEL_THRESHOLD:
        for i in range(m):
            for j in range(mn):
                ER[i, j] = pool.submit(matmul_ij, ZpX, E, PL, i, j, b)
        for i in range(m):
            for j in range(mn):
                ER[i, j] = ER[i, j].result()
    else:
        for i in range(m):
            for j in range(mn):
                eij = 0
                for k in range(mn):
                    eij += E[i, k].mul_low(PL[k, j], b).right_shift(b // 2)
                ER[i, j] = eij
    delta_R = [
        max(delta[i] + PL[i, j].degree() for i in range(mn) if PL[i, j])
        for j in range(mn)
    ]

    # Handle high half
    assert b > 2
    # We don't need E anymore
    for i in range(m):
        for j in range(mn):
            E[i, j] = None

    PR = mslgdc(ER, delta_R, b - b // 2, ZpX, depth=depth + 1, pool=pool)
    # We don't need ER anymore
    del ER

    if pool and b > PARALLEL_THRESHOLD:
        P = numpy.zeros((mn, mn), dtype=object)
        for i in range(mn):
            for j in range(mn):
                P[i, j] = pool.submit(matmul_ij, ZpX, PL, PR, i, j, None)
        for i in range(mn):
            for j in range(mn):
                P[i, j] = P[i, j].result()
    else:
        P = PL @ PR

    return P


if __name__ == "__main__":
    import time

    p = 1000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000140299
    N = 20000
    random.seed(42)
    taps = [random.randrange(p) for _ in range(min(200, N))]

    # A single sequence of size 2N+O(1) for reference
    seqlong = [random.randrange(p) for _ in range(N)]
    for _ in range(N + 64):
        t = -sum(a * seqlong[-N + i] for i, a in enumerate(taps)) % p
        seqlong.append(t)

    t0 = time.monotonic()
    refpoly = lingen([seqlong], N, p)
    dt = time.monotonic() - t0
    print(f"lingen for {N=} m=1 p={p.bit_length()}b in {dt:.3f}s")
    assert refpoly.degree() == N
    refpoly = list(refpoly)
    for i in range(N):
        assert int(refpoly[i]) == (taps[i] if i < len(taps) else 0)
    assert refpoly[N] == 1

    for k in range(200, 10000, 37):
        assert sum(refpoly[i] * seqlong[k + i] for i in range(N + 1)) == 0

    for m in (2, 3, 4):
        # m sequences of size N+N/m+O(1) for testing
        seqs = []
        for _ in range(m):
            seq = [random.randrange(p) for _ in range(N)]
            for _ in range(N // m + 64):
                t = -sum(a * seq[-N + i] for i, a in enumerate(taps)) % p
                seq.append(t)
            seqs.append(seq)

        t0 = time.monotonic()
        minpoly = lingen(seqs, N, p)
        dt = time.monotonic() - t0
        print(f"lingen for {N=} {m=} p={p.bit_length()}b in {dt:.3f}s")
        for i in range(N + 1):
            assert int(refpoly[i]) == int(minpoly[i])
