import logging
import random
import time

import flint
import kp
import numpy as np

from nefelis import lingen
from nefelis.vulkan import shader, stamp_period

DEBUG_NO_SORT_ROWS = False


def berlekamp_massey(seq: list[int], l: int):
    ctx = flint.fmpz_mod_poly_ctx(l)
    poly = ctx.minpoly(seq)
    return [int(coef) for coef in poly]


class SpMV:
    """
    A CSR encoded matrix with:
    - a block of dense columns (int8 coefficients)
    - an array of sparse positive rows (int16 columns indices with +1 coefficient)
    - an array of sparse negative rows (int16 columns indices with -1 coefficient)

    To support larger matrices, an index 0xffff can be inserted in sparse rows
    to explain that following indices belong to another block of size 0xffff
    """

    def __init__(self, rels: list[dict], ell: int, gpu_idx=0):
        """
        Build internal representation of a sparse matrix where
        input rows are given as dictionaries.

        Dictionary keys are opaque labels corresponding to matrix columns.
        """
        weight = sum(len(r) for r in rels)
        basis, dense, densebig, plus, minus = to_sparse_matrix(rels)
        dim, dense_n = dense.shape
        assert (dim * dense_n) % 4 == 0
        self.defines = {"N": dim, "DENSE_N": dense_n}

        self.basis = basis
        self.dim = dim
        self.densebig = densebig

        BLEN = (ell.bit_length() + 8 + 31) // 32
        maxout = ell * ell * (dim + 1)
        ALEN = max(2 * BLEN, (maxout.bit_length() + 4 + 31) // 32)
        self.defines |= {"ALEN": ALEN, "BLEN": BLEN}

        # Prepare tensors
        mgr = kp.Manager(gpu_idx)
        if dense.size == 0:
            dense = np.zeros(1, dtype=np.uint32)
        xd = mgr.tensor_t(dense.flatten().view(np.uint32))
        if densebig:
            # SM vectors are stored in Montgomery form.
            R = (1 << (32 * BLEN)) % ell
            dbig = np.zeros((dim, BLEN), dtype=np.uint32)
            for i in range(dim):
                dbig[i, :] = to_array(densebig[i] * R % ell, BLEN)
            xdbig = mgr.tensor_t(dbig.flatten())
            self.defines["SM"] = 1
        else:
            xdbig = mgr.tensor_t(np.zeros(1, dtype=np.uint32))  # dummy

        pwords = to_uvec(ell, BLEN)
        assert pwords[-2] > 2**16
        pinvwords = to_uvec(pow(-ell, -1, 2 ** (32 * BLEN)), BLEN)
        xell = mgr.tensor_t(np.array(pwords + pinvwords, dtype=np.uint32))

        # Encode rows
        def encode_row(row):
            "Encode row when dimension is large"
            nonlocal dense_n, dim
            if dim <= 0xFFFF:
                return row
            enc = []
            base = 0
            for x in row:
                while x >= base + 0xFFFF:
                    enc.append(0xFFFF)
                    base += 0xFFFF
                assert 0 <= x - base < 0xFFFF
                enc.append(x - base)
            return enc

        enc_plus = [encode_row(l) for l in plus]
        enc_minus = [encode_row(l) for l in minus]

        rowlen_plus = [len(l) for l in enc_plus]
        rowlen_minus = [len(l) for l in enc_minus]
        aidx_plus = np.cumsum(
            np.array([0] + rowlen_plus, dtype=np.uint32), dtype=np.uint32
        )
        aidx_minus = np.cumsum(
            np.array([0] + rowlen_minus, dtype=np.uint32), dtype=np.uint32
        )
        size_plus = int(aidx_plus[-1])
        size_minus = int(aidx_minus[-1])
        aplus = np.zeros(size_plus + (size_plus & 1), dtype=np.uint16)
        aminus = np.zeros(size_minus + (size_minus & 1), dtype=np.uint16)
        for i, l in enumerate(enc_plus):
            aplus[aidx_plus[i] : aidx_plus[i + 1]] = l
        for i, l in enumerate(enc_minus):
            aminus[aidx_minus[i] : aidx_minus[i + 1]] = l
        # Kompute wants uint32, cast arrays to make it happy
        xplus = mgr.tensor_t(aplus.view(np.uint32))
        xminus = mgr.tensor_t(aminus.view(np.uint32))
        xidxp = mgr.tensor_t(aidx_plus)
        xidxm = mgr.tensor_t(aidx_minus)

        self.mgr = mgr
        self.tensors = [xd, xdbig, xplus, xminus, xidxp, xidxm, xell]
        bitsize = 32 * sum(t.size() for t in self.tensors)
        logging.debug(f"Matrix format using {bitsize / weight:.1f} bits/coefficient")
        logging.debug(f"Using bigint arithmetic with {BLEN=} {ALEN=}")
        self.flops = 2 * dim * dense_n + size_plus + size_minus
        self.weight = weight
        logging.debug(
            f"{self.flops} FLOPS per matrix multiplication (original weight {weight})"
        )

    def shaders(self, defines):
        if defines.get("POLYEVAL"):
            # FIXME: polyeval does not like spmv_bigint2
            return [
                (name, shader(name, defines))
                for name in ("spmv_bigint", "spmv_bigint2", "spmv_bigint3")
            ]
        return [
            (name, shader(name, defines))
            for name in ("spmv_bigint", "spmv_bigint2", "spmv_bigint3")
        ]

    def _run(self, tensors, defines, ITERS, BATCHSIZE, N_WG):
        mgr = self.mgr
        algos = [
            (name, mgr.algorithm(tensors, kernel, workgroup=(N_WG, 1, 1)))
            for name, kernel in self.shaders(defines)
        ]
        best_algo = None
        algostats = [[] for _ in algos]

        mgr.sequence().record(kp.OpTensorSyncDevice(tensors)).eval()

        t0 = time.monotonic()
        t_print = t0
        gpu_ticks = 0.0
        count = 0
        for i in range(ITERS // BATCHSIZE + 1):
            if i * BATCHSIZE >= ITERS:
                break
            # Matrix multiplication is very fast so we launch multiple
            # iterations per batch.
            if best_algo is not None:
                algo_idx = best_algo
            elif i < 4:
                # warmup
                algo_idx = 0
            elif 4 <= i < 4 + 8 * len(algos):
                algo_idx = (i - 4) // 8
            else:
                algo_times = [min(s) * stamp_period() / BATCHSIZE for s in algostats]
                algo_times_show = {
                    _name: round(_t) * 1e-6 for (_name, _), _t in zip(algos, algo_times)
                }
                logging.info(f"Matmul shader performance (ms/matmul) {algo_times_show}")
                best_algo = min(range(len(algo_times)), key=lambda idx: algo_times[idx])
                logging.info(f"Selecting shader {algos[best_algo][0]}")
                algo_idx = best_algo

            seq = mgr.sequence(total_timestamps=2 * BATCHSIZE)
            for _ in range(min(BATCHSIZE, ITERS - i * BATCHSIZE)):
                seq.record(kp.OpAlgoDispatch(algos[algo_idx][1]))
                count += 1
            seq.eval()

            stamps = seq.get_timestamps()
            if 4 <= i < 4 + 8 * len(algos):
                algostats[algo_idx].append(stamps[-1] - stamps[0])
            gpu_ticks += stamps[-1] - stamps[0]
            if (t1 := time.monotonic()) > t_print + 10.0:
                # print progress every 10 seconds
                elapsed = t1 - t0
                gpu_dt = gpu_ticks * stamp_period() * 1e-9
                speed = BATCHSIZE * (i + 1) / gpu_dt
                logging.info(
                    f"{BATCHSIZE * (i + 1)} matrix muls done in {elapsed:.1f}s ({speed:.1f} SpMV/s, GPU time {gpu_dt:.1f}s)"
                )
                t_print = t1

        assert count == ITERS
        return gpu_ticks

    def wiedemann_big(self, l: int, blockm=1) -> list[int]:
        """
        Perform Wiedemann algorithm for a single big modulus

        If blockm > 1, a block Wiedemann algorithm (with 1 vector but multiple sequences)
        is used.
        """
        WGSIZE = 128
        mgr = self.mgr
        dim = self.dim
        N_WG = (dim + WGSIZE - 1) // WGSIZE

        BATCHSIZE = 32

        ITERS = dim + dim // blockm + 320
        ITERS = (ITERS // BATCHSIZE + 2) * BATCHSIZE

        # FIXME: use actual norm
        BLEN = self.defines["BLEN"]

        defines = self.defines | {"BLOCKM": blockm}

        # Tensor holding M^k V and M^(k+1) V
        v = np.zeros((2, dim, BLEN), dtype=np.uint32)
        for i in range(dim):
            v[0, i, :] = to_uvec(random.randrange(l), BLEN)
        xv = mgr.tensor_t(v.flatten())
        xiter = mgr.tensor_t(np.zeros(N_WG, dtype=np.uint32))
        # Output sequence out[k] = S M^k V
        xout = mgr.tensor_t(np.zeros(ITERS * blockm * BLEN, dtype=np.uint32))

        tensors = self.tensors + [xv, xiter, xout]

        logging.info(f"Computing a Krylov sequence of length {ITERS}")

        t0 = time.monotonic()
        self.ell = l
        gpu_ticks = self._run(tensors, defines, ITERS, BATCHSIZE, N_WG)

        mgr.sequence().record(kp.OpTensorSyncLocal([xout])).eval()
        vout = xout.data().reshape((ITERS, blockm, BLEN))
        sequences = []
        for j in range(blockm):
            seq = [from_uvec(v[0, j, :])] + [
                from_uvec(vout[i, j, :]) for i in range(ITERS)
            ]
            sequences.append(seq)

        dt = time.monotonic() - t0
        gpu_dt = gpu_ticks * stamp_period() * 1e-9
        flops = self.flops * ITERS / gpu_dt
        speed = ITERS / gpu_dt

        t1 = time.monotonic()
        poly = lingen.lingen(sequences, dim, l)
        assert len(poly) <= dim + 1, len(poly)

        dt = time.monotonic() - t0
        lingen_dt = time.monotonic() - t1
        logging.info(f"Lingen completed in {lingen_dt:.3f}s (N={dim} m={blockm} n=1)")
        logging.info(
            f"Wiedemann completed in {dt:.3f}s (GPU {gpu_dt:.3}s, {flops / 1e9:.2f} GFLOPS, {speed:.1f} SpMV/s)"
        )

        return [int(ai) for ai in list(poly)]

    def polyeval(self, v, l: int, poly: list[int]) -> list[int]:
        """
        Compute poly(M)*v mod l
        """
        WGSIZE = 128
        mgr = self.mgr
        dim = self.dim
        N_WG = (dim + WGSIZE - 1) // WGSIZE
        BATCHSIZE = 16

        # FIXME: use actual norm
        ALEN, BLEN = self.defines["ALEN"], self.defines["BLEN"]

        defines = self.defines | {"POLYEVAL": 1}

        # Tensor holding M^k V and M^(k+1) V
        av = np.zeros((2, dim, BLEN), dtype=np.uint32)
        for i in range(dim):
            av[0, i, :] = to_uvec(v[i], BLEN)
        xv = mgr.tensor_t(av.flatten())
        xiter = mgr.tensor_t(np.zeros(N_WG, dtype=np.uint32))
        vpoly = np.zeros((len(poly), BLEN), dtype=np.uint32)
        for k, ak in enumerate(poly):
            vpoly[k, :] = to_uvec(ak, BLEN)
        xpoly = mgr.tensor_t(vpoly.flatten())
        # Output sequence out[k] = S M^k V, initialize with a0 * v
        vout = np.zeros((dim, ALEN), dtype=np.uint32)
        for i, vi in enumerate(v):
            vout[i, :] = to_uvec(poly[0] * vi, ALEN)
        xout = mgr.tensor_t(vout.flatten())

        tensors = self.tensors + [xv, xiter, xpoly, xout]

        t0 = time.monotonic()
        gpu_ticks = self._run(tensors, defines, len(poly) - 1, BATCHSIZE, N_WG)
        dt = time.monotonic() - t0

        gpu_dt = gpu_ticks * stamp_period() * 1e-9
        flops = self.flops * len(poly) / gpu_dt
        speed = len(poly) / gpu_dt

        mgr.sequence().record(kp.OpTensorSyncLocal([xout])).eval()
        vout = xout.data().reshape((dim, ALEN))
        dt = time.monotonic() - t0
        logging.info(
            f"Polyeval completed in {dt:.3f}s (GPU {gpu_dt:.3}s, {flops / 1e9:.2f} GFLOPS, {speed:.1f} SpMV/s)"
        )
        return [from_uvec(vout[i, :]) % l for i in range(dim)]


def to_sparse_matrix(rels):
    """
    Converts a list of relations into a representation suitable
    for sparse matrix kernels.

    The matrix rows may correspond to an unspecified permutation
    of input relations.
    """
    stats = {}
    for r in rels:
        for p in r:
            if r[p]:
                stats[p] = stats.get(p, 0) + abs(r[p])
    # Find densest columns above 33% fill ratio
    dense_counts = []
    dense_big = []
    for p, count in stats.items():
        if p == "SM":
            dense_big.append(p)
            continue
        if count > len(rels) // 3:
            dense_counts.append((count, p))
    dense_counts.sort()
    dense_p = sorted([p for _, p in dense_counts[len(dense_counts) % 4 :]])
    assert len(dense_p) % 4 == 0

    dense_weight = sum(stats[p] for p in dense_p) / float(len(rels))
    logging.debug(f"Dense columns for {len(dense_p)} primes {dense_p}")
    logging.debug(f"Dense big columns for {dense_big}")
    logging.info(
        f"Dense block has {len(dense_p)} columns, average weight {dense_weight:.1f} per row"
    )
    dense_set = frozenset(dense_p + dense_big)
    sparse_weight = sum(
        sum(abs(e) for p, e in r.items() if p not in dense_set) for r in rels
    ) / float(len(rels))
    logging.info(f"Sparse block has avg weight {sparse_weight:.1f} per row")

    # To reduce divergence, we sort rows by the number of ±signs in the sparse part.
    sign_rels = []
    for r in rels:
        nplus, nminus = 0, 0
        for _p, _e in r.items():
            if _p not in dense_set:
                if _e > 0:
                    nplus += 1
                else:
                    nminus += 1
        sign_rels.append((nplus, nminus, r))
    if not DEBUG_NO_SORT_ROWS:
        sign_rels.sort(key=lambda t: t[:2])
    # print([(x, y) for x, y, z in sign_rels])
    rels = [_r for _, _, _r in sign_rels]

    # Dense coefficients must fit in int8 type
    for r in rels:
        for p in dense_p:
            if p in r:
                assert abs(r[p]) < 127

    dense = np.zeros((len(rels), len(dense_p)), dtype=np.int8)
    for i, r in enumerate(rels):
        dense[i, :] = [r.get(p, 0) for p in dense_p]
    dense_norm = max(np.sum(np.abs(dense[i, :])) for i in range(len(rels)))
    logging.info(f"Dense block has max row norm {dense_norm}")

    matbig = []
    for k in dense_big:
        rowbig = [r.get(k, 0) for r in rels]
        matbig = rowbig  # FIXME for multiple SM

    primes = dense_p + dense_big + sorted(p for p in stats if p not in dense_set)

    prime_idx = {p: idx for idx, p in enumerate(primes)}
    plus = []
    minus = []
    for r in rels:
        row_p, row_m = [], []
        for p, e in r.items():
            if p in dense_set:
                continue
            idx = prime_idx[p]
            if e > 0:
                row_p.extend(e * [idx])
            else:
                row_m.extend(-e * [idx])
        row_p.sort()
        row_m.sort()
        plus.append(row_p)
        minus.append(row_m)
    return primes, dense, matbig, plus, minus


def to_uvec(x: int, length: int):
    assert x.bit_length() <= 32 * length
    return [(x >> (32 * i)) & 0xFFFFFFFF for i in range(length)]


def from_uvec(words: list) -> int:
    return sum(int(x) << (32 * i) for i, x in enumerate(words))


def from_array(array) -> int:
    return int.from_bytes(array.tobytes(), "little")


def to_array(x: int, length: int):
    return np.frombuffer(x.to_bytes(4 * length, "little"), dtype=np.uint32)
