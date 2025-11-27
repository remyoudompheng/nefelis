from typing import Iterable
import logging
import random
import time

import kp
import numpy as np
import numpy.typing as npt

from nefelis import lingen_gf2
from nefelis.vulkan import shader, stamp_period, gpu_cores

# Avoid row reordering when constructing SpMV matrices.
DEBUG_NO_SORT_ROWS = False
# Don't add padding in extra columns
DEBUG_NO_PADDING = False
# Sanity check for matrix representation
DEBUG_CHECK_ENCODING = False
# Extra checks for returns kernel elements
DEBUG_CHECK_KERNEL = False
# Print output vectors after benchmark (SpMV performance may decrease).
DEBUG_BENCHMARK_OUTPUT = False

logger = logging.getLogger("linalg")


class SpMV:
    """
    A CSR encoded matrix with:
    - a block of dense columns (int8 coefficients)
    - an array of sparse positive rows (int16 columns indices with +1 coefficient)
    - an array of sparse negative rows (int16 columns indices with -1 coefficient)

    To support larger matrices, an index 0xffff can be inserted in sparse rows
    to explain that following indices belong to another block of size 0xffff
    """

    ROWS_PER_WG = 128

    def __init__(self, rels: list[npt.NDArray[np.int32]], gpu_idx=0):
        """
        Build internal representation of a sparse matrix where
        input rows are given as dictionaries.

        Dictionary keys are opaque labels corresponding to matrix columns.
        """
        weight = sum(len(r) for r in rels)
        basis, rowidx, dense, sparserows = to_sparse_matrix(
            rels,
            sort_rows=not DEBUG_NO_SORT_ROWS,
            # If we check benchmark output, we need to shuffle columns in the same way.
            # This may degrade performance.
            shuffle_cols=DEBUG_BENCHMARK_OUTPUT,
        )
        dim, dense_n = dense.shape
        self.defines = {"N": dim, "DENSE_N": dense_n}

        self.rels = rels
        # For debugging
        if DEBUG_CHECK_ENCODING:
            self.dense = dense
            self.sparserows = sparserows

            # FIXME

            keyidx = {k: idx for idx, k in enumerate(basis)}
            for ridx in range(dim):
                realidx = rowidx[ridx]
                l1 = sorted(
                    [keyidx[_l] for _l in self.rels[realidx] if not _l.startswith("K_")]
                )
                l2 = [
                    j for j in range(self.dense.shape[1]) if self.dense[ridx, j]
                ] + self.sparserows[ridx]
                assert l1 == l2

        self.basis = basis
        self.rowidx = rowidx
        self.dim = dim

        # Prepare tensors
        mgr = kp.Manager(gpu_idx)
        if dense.size == 0:
            dense = np.zeros(1, dtype=np.uint32)
        xd = mgr.tensor_t(dense.flatten().view(np.uint32))

        # Encode rows
        def encode_row(row):
            "Encode row when dimension is large"
            nonlocal dim
            if dim <= 0xFFFF:
                return row
            enc = []
            base = 0
            row.sort()
            for x in map(int, row):
                while x >= base + 0xFFFF:
                    enc.append(0xFFFF)
                    base += 0xFFFF
                assert 0 <= x - base < 0xFFFF, (x, base)
                enc.append(x - base)
            return enc

        enc = [encode_row(l) for l in sparserows]

        rowlen = [len(l) for l in enc]
        aidx = np.cumsum(np.array([0] + rowlen, dtype=np.uint32), dtype=np.uint32)
        size_plus = int(aidx[-1])
        aplus = np.zeros(size_plus + (size_plus & 1), dtype=np.uint16)
        for i, l in enumerate(enc):
            aplus[aidx[i] : aidx[i + 1]] = l
        # Kompute wants uint32, cast arrays to make it happy
        xplus = mgr.tensor_t(aplus.view(np.uint32))
        xidxp = mgr.tensor_t(aidx)

        self.mgr = mgr
        self.tensors = [xd, xplus, xidxp]
        self.tensors_transposed = self.tensors
        bitsize = 32 * sum(t.size() for t in self.tensors)
        logger.debug(
            f"Matrix format using {bitsize / weight:.1f} bits/coefficient ({bitsize >> 25}MiB)"
        )
        self.flops = dim * dense_n * 32 + size_plus
        self.weight = weight
        logger.debug(
            f"{self.flops} bit ops per matrix multiplication (original weight {weight})"
        )

    def shaders(self, defines, transpose: bool):
        if transpose:
            return [shader("spmv_gf2_transpose1", defines)]
        else:
            return [shader("spmv_gf2", defines)]

    def _run(
        self,
        tensors,
        defines,
        ITERS: int,
        BATCHSIZE: int,
        N_WG: int,
        transpose: bool = False,
    ):
        mgr = self.mgr
        kernels = self.shaders(defines, transpose)
        algos = [
            mgr.algorithm(tensors, kernel, workgroup=(N_WG, 1, 1)) for kernel in kernels
        ]

        mgr.sequence().record(kp.OpTensorSyncDevice(tensors)).eval()

        t0 = time.monotonic()
        t_print = t0
        gpu_ticks = 0.0
        count = 0
        for i in range(ITERS // BATCHSIZE + 1):
            if i * BATCHSIZE >= ITERS:
                break

            seq = mgr.sequence(total_timestamps=2 * BATCHSIZE)
            for _ in range(min(BATCHSIZE, ITERS - i * BATCHSIZE)):
                for a in algos:
                    seq.record(kp.OpAlgoDispatch(a))
                count += 1
            seq.eval()

            stamps = seq.get_timestamps()
            gpu_ticks += stamps[-1] - stamps[0]
            if (t1 := time.monotonic()) > t_print + 10.0:
                # print progress every 10 seconds
                elapsed = t1 - t0
                gpu_dt = gpu_ticks * stamp_period() * 1e-9
                speed = BATCHSIZE * (i + 1) / gpu_dt
                logger.info(
                    f"{BATCHSIZE * (i + 1)} matrix muls done in {elapsed:.1f}s ({speed:.1f} SpMV/s, GPU time {gpu_dt:.1f}s)"
                )
                t_print = t1

        assert count == ITERS
        return gpu_ticks

    def right_kernel(self):
        """
        Compute a list of right-kernel elements: non-zero vectors Vi
        such that M*Vi == 0.
        """
        # We want large m to obtain many kernel elements,
        # but the CPU cost is high.
        if self.dim < 16384:
            m = 16
        else:
            m = 32
        # If sum(ak X^k) is a matrix generator of the sequence W0 M^k V0
        # then sum(M^(d-k) V0 ak) is a possible right kernel element.
        poly, v = self.block_wiedemann(m)
        poly1 = [poly[len(poly) - i - 1] for i in range(len(poly))]
        w = self.polyeval(poly1, v, m)
        w = w[:, 0]
        if DEBUG_CHECK_KERNEL:
            poly1 = [poly1[0] * 0] + poly1
            wx = self.polyeval(poly1, v, m)
            wx = wx[:, 0]

        for i in range(m):
            wi = list(((w >> i) & 1).flatten())
            print(f"w[{i}]", "".join(str(b) for b in wi))
            keyidx = {k: idx for idx, k in enumerate(self.basis)}
            if DEBUG_CHECK_KERNEL:
                wxi = list(((wx >> i) & 1).flatten())
                print(f"wx[{i}]", "".join(str(b) for b in wxi))
                ok = True
                for r in self.rels:
                    rw = sum(wi[keyidx[_l]] for _l in r if not _l.startswith("K_")) & 1
                    print(rw, end="")
                    ok = ok and rw == 0
                print("")
                print("\nkernel ok", ok)
        # M w = 0
        # poly1 = [poly1[0] * 0] + poly1
        # assert self.polyeval(poly1, v, m) == 0

    def _is_valid_outidx(self, m: int, outidx: int):
        # No constraint on outidx
        return True

    def left_kernel(self):
        if self.dim < 16384:
            m = 16
        else:
            m = 32
        # If sum(ak X^k) is a matrix generator of the sequence W0 M^k V0
        # then sum(ak W0 M^k) is a possible left kernel element.
        #
        # W0 is not random and is always a zero-padded identity matrix.
        # W0 is shifted by a random amount of columns.
        outidx = random.randrange(0, self.dim - m)
        while not self._is_valid_outidx(m, outidx):
            outidx = random.randrange(0, self.dim - m)
        poly, v = self.block_wiedemann(m, outidx=outidx, left=True)
        # FIXME: assumes m <= 32
        w0 = np.zeros((self.dim, (m + 31) // 32), dtype=np.uint32)
        for i in range(m):
            iout = outidx + i
            w0[iout, i // 32] = 1 << (i % 32)
        poly1 = [poly[len(poly) - i - 1] for i in range(len(poly))]
        logger.debug(f"Linear generator has degree {len(poly) - 1}")

        # Usually the degree 0 matrix has many zero rows, try dividing by X
        shifted = 0
        for i in range(m):
            if np.all(poly1[0][i, :] == 0):
                shifted += 1
                for j in range(1, len(poly1)):
                    poly1[j - 1][i, :] = poly1[j][i, :]
        logger.info(f"Shifted {shifted} rows with polynomials divisible by X")

        kers = []
        seen = set()

        w = self.polyeval(poly1, w0, m, transpose=True)
        polyx = [np.zeros((m, m), dtype=np.uint8), np.identity(m, dtype=np.uint8)]
        wx = self.polyeval(polyx, w, m, transpose=True)
        wx2 = self.polyeval(polyx, wx, m, transpose=True)

        dups = 0
        for i in range(m):
            wi = list(((w[:, 0] >> i) & 1).flatten())
            wxi = list(((wx[:, 0] >> i) & 1).flatten())
            wx2i = list(((wx2[:, 0] >> i) & 1).flatten())
            if sum(wxi) != 0 and sum(wx2i) == 0:
                logger.debug(f"w[{i}] is not in kernel, shifting to wx[{i}]")
                wi, wxi = wxi, wx2i
            if sum(wi) > 0 and sum(wxi) == 0:
                wstr = "".join(str(b) for b in wi)
                if wstr in seen:
                    dups += 1
                    continue
                kers.append(wi)
                seen.add(wstr)
                # print(f"wx[{i}]", "".join(str(b) for b in wxi))
                if DEBUG_CHECK_KERNEL:
                    print("found left kernel")
                    print(f"w[{i}]", wstr)
                    mw = {}
                    for j in range(self.dim):
                        if wi[j]:
                            r = self.rels[self.rowidx[j]]
                            for _l in r:
                                mw[_l] = mw.get(_l, 0) + 1
                    assert all(
                        coef & 1 == 0
                        for _l, coef in mw.items()
                        if not _l.startswith("K_")
                    )
                    # print(mw)
        logger.debug(f"Skipped {dups} duplicate kernel elements")
        return kers

    def benchmark(self, m: int, iters: int, polyeval: bool):
        """
        Runs a given number of iteration of the standard or transposed
        kernels with random inputs. The random generator seed is fixed
        to zero to obtain reproducible results.
        """
        mgr = self.mgr
        dim = self.dim
        N_WG = (dim + self.ROWS_PER_WG - 1) // self.ROWS_PER_WG

        if dim > 1000_000:
            BATCHSIZE = 4
        elif dim > 400_000:
            BATCHSIZE = 8
        else:
            BATCHSIZE = 16
        K = (m + 31) // 32

        defines = self.defines | {"M": m, "K": K}
        if polyeval:
            defines |= {"POLYEVAL": 1}
        else:
            defines |= {"OUTIDX": 0}

        # Random initial vector
        rng = np.random.default_rng(seed=0)
        v = np.zeros((2, dim, K), dtype=np.uint32)
        v[0, :, :] = rng.integers(2**32, size=(dim, K), dtype=np.uint32)

        # Tensors
        xv = mgr.tensor_t(v.flatten())
        xiter = mgr.tensor_t(np.zeros(N_WG, dtype=np.uint32))

        if polyeval:
            vpoly = rng.integers(2**32, size=(iters, m, K), dtype=np.uint32)
            xpoly = mgr.tensor_t(vpoly.flatten())
            xout = mgr.tensor_t(np.zeros(dim * K, dtype=np.uint32))
            tensors = self.tensors_transposed + [xv, xiter, xpoly, xout]
        else:
            xout = mgr.tensor_t(np.zeros(iters * K * m, dtype=np.uint32))
            tensors = self.tensors + [xv, xiter, xout]

        t0 = time.monotonic()
        gpu_ticks = self._run(
            tensors, defines, iters, BATCHSIZE, N_WG, transpose=polyeval
        )
        dt = time.monotonic() - t0
        gpu_dt = gpu_ticks * stamp_period() * 1e-9

        mgr.sequence().record(kp.OpTensorSyncLocal([xv, xout])).eval()

        if DEBUG_BENCHMARK_OUTPUT:
            if not polyeval:
                vv = xv.data().reshape((2, dim, K))
                print(vv[iters & 1, :, :].flatten())
                vout = xout.data().reshape((iters, m, K))
                print(vout.flatten())
            else:
                vv = xv.data().reshape((2, dim, K))
                print(vv[iters & 1, :, :].flatten())
                vout = xout.data().reshape((dim, K))
                print(vout.flatten())

        return dt, gpu_dt

    def block_wiedemann(
        self, m=32, outidx=0, left=False
    ) -> tuple[list[npt.NDArray], npt.NDArray]:
        """
        Compute a matrix linear generator using block Wiedemann algorithm

        The sequence is W M^k V where V is a random vector and W is
        the zero-extended identity matrix.

        The result is a right linear generator: sum ak X^k such that
           sum(W M^(d-k) V ak) = 0 for large enough d.

        If argument `left` is True, a left generator is returned:
           sum(ak W M^(d-k) V) = 0 for large enough d.

        The polynomial is returns as a list of numpy matrices with 0/1 coefficients.
        """
        mgr = self.mgr
        dim = self.dim
        N_WG = (dim + self.ROWS_PER_WG - 1) // self.ROWS_PER_WG

        if dim > 1000_000:
            BATCHSIZE = 4
        elif dim > 400_000:
            BATCHSIZE = 8
        else:
            BATCHSIZE = 16
        K = (m + 31) // 32

        ITERS = dim // m + dim // m + 32
        ITERS = (ITERS // BATCHSIZE + 2) * BATCHSIZE

        defines = self.defines | {"M": m, "K": K, "OUTIDX": outidx}

        # Tensor holding M^k V and M^(k+1) V
        v = np.zeros((2, dim, K), dtype=np.uint32)
        for i in range(dim):
            for j in range(K):
                v[0, i, j] = random.getrandbits(32)
        xv = mgr.tensor_t(v.flatten())
        xiter = mgr.tensor_t(np.zeros(N_WG, dtype=np.uint32))
        # Output sequence out[k] = S M^k V
        xout = mgr.tensor_t(np.zeros(ITERS * K * m, dtype=np.uint32))

        tensors = self.tensors + [xv, xiter, xout]

        logger.info(f"Computing a Krylov sequence of length {ITERS}")

        t0 = time.monotonic()
        gpu_ticks = self._run(tensors, defines, ITERS, BATCHSIZE, N_WG)

        mgr.sequence().record(kp.OpTensorSyncLocal([xout])).eval()
        vout = xout.data().reshape((ITERS, m, K))

        dt = time.monotonic() - t0
        gpu_dt = gpu_ticks * stamp_period() * 1e-9
        speed = ITERS / dt
        logger.info(
            f"Krylov completed in {dt:.3f}s (GPU {gpu_dt:.3f}s, {ITERS} iterations, {speed:.3f} iter/s)"
        )

        # vout[i, j, :] is the j-th row of M^i V
        mats = [[] for _ in range(m)]
        if left:
            # mat[k, j] = vout[:, j, k]
            for k in range(m):
                for j in range(m):
                    seqij = [int(b) for b in (vout[:, j, k // 32] >> (k % 32)) & 1]
                    # print(j, k, "".join(str(b) for b in seqij))
                    mats[k].append(seqij)
        else:
            # mat[j, k] = vout[:, j, k]
            for j in range(m):
                for k in range(m):
                    seqij = [int(b) for b in (vout[:, j, k // 32] >> (k % 32)) & 1]
                    # print(j, k, "".join(str(b) for b in seqij))
                    mats[j].append(seqij)

        t1 = time.monotonic()
        poly = lingen_gf2.lingen_mat(mats, dim)

        if left:
            # Transpose polynomial again
            poly = [ak.T for ak in poly]

        dt = time.monotonic() - t0
        lingen_dt = time.monotonic() - t1
        logger.info(f"Lingen completed in {lingen_dt:.3f}s (N={dim} m=n={m})")

        return poly, v[0, :, :]

    def polyeval(
        self, poly: list[npt.NDArray], v0: npt.NDArray, m: int, transpose: bool = False
    ) -> npt.NDArray:
        """
        Compute sum(M^k v0 ak) where poly = sum(ak X^k)
        or the transposed variant sum(ak v0 M^k)
        """
        mgr = self.mgr
        dim = self.dim
        N_WG = (dim + self.ROWS_PER_WG - 1) // self.ROWS_PER_WG

        if dim > 1000_000:
            BATCHSIZE = 4
        elif dim > 400_000:
            BATCHSIZE = 8
        else:
            BATCHSIZE = 16
        K = (m + 31) // 32

        defines = self.defines | {"M": m, "K": K, "POLYEVAL": 1}

        # Tensor holding M^k V and M^(k+1) V
        v = np.zeros((2, dim, K), dtype=np.uint32)
        v[0, :, :] = v0

        xv = mgr.tensor_t(v.flatten())
        xiter = mgr.tensor_t(np.zeros(N_WG, dtype=np.uint32))
        vpoly = np.zeros((len(poly), m, K), dtype=np.uint32)
        pow2 = np.array([1 << i for i in range(m)], dtype=np.uint32)
        for k, ak in enumerate(poly):
            assert ak.shape == (m, m)
            vpoly[k, :, :] = (ak @ pow2).reshape(m, K)
        xpoly = mgr.tensor_t(vpoly.flatten())
        # Output accumulator out = sum(M^k V ak)
        vout = np.zeros((dim, K), dtype=np.uint32)
        xout = mgr.tensor_t(vout.flatten())

        if transpose:
            tensors = self.tensors_transposed + [xv, xiter, xpoly, xout]
        else:
            tensors = self.tensors + [xv, xiter, xpoly, xout]

        t0 = time.monotonic()
        gpu_ticks = self._run(tensors, defines, len(poly), BATCHSIZE, N_WG, transpose)
        dt = time.monotonic() - t0

        gpu_dt = gpu_ticks * stamp_period() * 1e-9
        flops = self.flops * len(poly) * m / gpu_dt
        speed = len(poly) * m / gpu_dt

        mgr.sequence().record(kp.OpTensorSyncLocal([xout])).eval()
        vout = xout.data().reshape((dim, K))
        dt = time.monotonic() - t0
        logger.info(
            f"Polyeval completed in {dt:.3f}s (GPU {gpu_dt:.3f}s, {flops / 1e9:.2f} GFLOPS, {speed:.1f} SpMV/s)"
        )
        return vout


class SpMV_COO(SpMV):
    """
    A block-COO encoded matrix where each sparse coefficient is encoded
    as a packed tuple of (b,32-b) bit coordinates.

    The matrix is split into stripes of 2^b rows and must have
    at most 2^(32-b) columns.

    The shaders take input vectors/polynomials in the same descriptors as the SpMV
    class so most methods can be reused identically.
    """

    # True for SpMV_COO2 (initializer is shared)
    NEED_COL_STRIPES = False

    def __init__(self, rels: list[npt.NDArray[np.int32]], gpu_idx=0):
        """
        Build internal representation of a sparse matrix where
        input rows are given as dictionaries.

        Dictionary keys are opaque labels corresponding to matrix columns.
        """
        weight = sum(len(r) for r in rels)
        basis, rowidx, dense, sparserows = to_sparse_matrix(
            rels, sort_rows=False, shuffle_cols=True
        )
        dim, dense_n = dense.shape

        self.rels = rels
        # For debugging
        if DEBUG_CHECK_ENCODING:
            self.dense = dense
            self.sparserows = sparserows

        BM = 128
        # We don't want to use more than 8kB of local memory.
        while BM < 2048:
            # The encoding must fit in 32 bits
            if dim * 2 * BM > 2**32:
                break
            # We want at least 4 workgroups per GPU cores
            if dim / (2 * BM) < 4 * gpu_cores():
                break
            BM *= 2
        assert 128 <= BM <= 2048
        assert dim <= 2**32 / BM

        self.defines = {"N": dim, "DENSE_N": dense_n, "BM": BM}

        self.ROWS_PER_WG = BM
        self.basis = basis
        self.rowidx = rowidx
        self.dim = dim

        # Prepare tensors
        mgr = kp.Manager(gpu_idx)
        if dense.size == 0:
            dense = np.zeros(1, dtype=np.uint32)
        xd = mgr.tensor_t(dense.flatten().view(np.uint32))

        # Encode rows
        def encode_stripe(rows):
            "Encode row when dimension is large"
            chunks = []
            for i, row in enumerate(rows):
                row = row.astype(np.uint32)
                row = row * BM + (i % BM)
                chunks.append(row)
            enc = np.concatenate(chunks)
            enc.sort()
            return enc

        enc = [encode_stripe(sparserows[i : i + BM]) for i in range(0, dim, BM)]

        rowlen = [len(l) for l in enc]
        # logger.debug(f"Stripe sizes {rowlen}")
        # logger.debug(f"Stripe cols sizes {[len(c) for c in enc_cols]}")
        aidx = np.cumsum(np.array([0] + rowlen, dtype=np.uint32), dtype=np.uint32)
        arows = np.zeros(sum(rowlen), dtype=np.uint32)
        for i, l in enumerate(enc):
            arows[aidx[i] : aidx[i + 1]] = l
        xrows = mgr.tensor_t(arows)
        xidx = mgr.tensor_t(aidx)

        if self.NEED_COL_STRIPES:
            # Encode columns
            # For this we build a large array with all coefficients
            all_cols = []
            for i, row in enumerate(sparserows):
                lo = (row & (BM - 1)).astype(np.uint64)
                hi = (row // BM).astype(np.uint64)
                all_cols.append(lo + (i * BM) + (hi << 32))
            all_cols = np.concatenate(all_cols)
            all_cols.sort()

            idxc = np.searchsorted(
                all_cols, np.arange(0, (dim // BM + 2) << 32, 1 << 32, dtype=np.uint64)
            )
            for j, jidx in zip(range(dim // BM + 2), idxc):
                if j == 0:
                    assert jidx == 0
                elif jidx == len(all_cols):
                    assert (int(all_cols[-1]) >> 32) < j
                else:
                    assert (int(all_cols[jidx]) >> 32) >= j
                    assert (int(all_cols[jidx - 1]) >> 32) < j
            assert idxc[-1] == len(all_cols)

            aidxc = np.array(idxc, dtype=np.uint32)
            # Truncate to low 32 bits.
            acols = all_cols.astype(np.uint32)
            xcols = mgr.tensor_t(acols)
            xidxc = mgr.tensor_t(aidxc)
            self.tensors_columns = [xd, xcols, xidxc]

        self.mgr = mgr
        self.tensors = [xd, xrows, xidx]
        self.tensors_transposed = self.tensors
        bitsize = 32 * sum(t.size() for t in self.tensors)
        logger.debug(
            f"Matrix format using {bitsize / weight:.1f} bits/coefficient ({bitsize >> 25}MiB)"
        )
        if self.NEED_COL_STRIPES:
            bitsize = 32 * sum(t.size() for t in self.tensors_columns)
            logger.debug(
                f"Transposed matrix format using {bitsize / weight:.1f} bits/coefficient ({bitsize >> 25}MiB)"
            )

        self.flops = dim * dense_n * 32 + sum(rowlen)
        self.weight = weight
        logger.debug(
            f"{self.flops} bit ops per matrix multiplication (original weight {weight})"
        )

    def shaders(self, defines, transpose: bool):
        if transpose:
            return [shader("spmv_gf2_blockcoo_transpose", defines)]
        else:
            return [shader("spmv_gf2_blockcoo", defines)]

    def _is_valid_outidx(self, m: int, outidx: int):
        return outidx % self.ROWS_PER_WG < self.ROWS_PER_WG - m


class SpMV_COO2(SpMV_COO):
    """
    Variant of SpMV_COO where the transposed matrix is encoded
    as stripes of columns.
    """

    NEED_COL_STRIPES = True

    def __init__(self, *args, **kwargs):
        super(SpMV_COO2, self).__init__(*args, **kwargs)
        self.tensors_transposed = self.tensors_columns

    def shaders(self, defines, transpose: bool):
        if transpose:
            return [
                shader("spmv_gf2_blockcoo_clear2", defines),
                shader("spmv_gf2_blockcoo_transpose2", defines),
            ]
        else:
            return [shader("spmv_gf2_blockcoo", defines)]


DENSE_ALIGN = 32


def to_sparse_matrix(
    rels: list[npt.NDArray[np.int32]],
    sort_rows=False,
    shuffle_cols=False,
) -> tuple[list[int], list[int], npt.NDArray[np.uint32], list[npt.NDArray[np.uint32]]]:
    """
    Converts a list of relations into a representation suitable
    for sparse matrix kernels.

    The matrix rows may correspond to an unspecified permutation
    of input relations.

    Returns:
    - primes: the list of column labels
    - rowidx: the list of relations indices used for each row
    - dense: a dense tensor (bit packed) for the first n columns
    - sparse: a CSR representation of other columns
    """
    pmax = max(np.max(r) for r in rels)
    stats_table = np.zeros(pmax + 1, dtype=np.uint32)
    for r in rels:
        stats_table[r[r >= 0]] += 1
    stats = {int(p): int(stats_table[p]) for p in stats_table.nonzero()[0]}
    # Ignore negative entries (unused in matrix)
    rels = [r[r >= 0] for r in rels]
    # Find densest columns
    counts = sorted((count, p) for p, count in stats.items())
    dense_n = sum(1 for count, _ in counts if count > len(rels) // 4)
    if dense_n % DENSE_ALIGN:
        dense_n += DENSE_ALIGN - dense_n % DENSE_ALIGN
    dense_p = [p for _, p in counts[-dense_n:]]
    assert len(dense_p) % DENSE_ALIGN == 0

    dense_set = frozenset(dense_p)
    if shuffle_cols:
        sparse_p = [p for p in stats if p not in dense_set]
        random.shuffle(sparse_p)
    else:
        sparse_p = sorted(p for p in stats if p not in dense_set)
    primes = dense_p + sparse_p

    prime_idx = np.zeros(pmax + 1, dtype=np.uint32)
    for idx, p in enumerate(primes):
        prime_idx[p] = idx

    dense_weight = np.sum(stats_table[dense_p]) / float(len(rels))
    logger.debug(f"Dense columns for {len(dense_p)} primes")
    logger.info(
        f"Dense block has {len(dense_p)} columns, average weight {dense_weight:.1f} per row"
    )
    sparse_weight = sum(
        np.count_nonzero(prime_idx[r] >= len(dense_p)) for r in rels
    ) / float(len(rels))
    logger.info(f"Sparse block has avg weight {sparse_weight:.1f} per row")
    logger.info(f"Largest sparse column has weight {counts[-dense_n - 1][0]}")

    # To reduce divergence, we sort rows by length
    if sort_rows:
        size_rels = []
        for ridx, r in enumerate(rels):
            rsize = sum(1 for _p in r if _p not in dense_set)
            size_rels.append((-rsize, ridx))
            size_rels.sort()
        rowidx = [ridx for _, ridx in size_rels]
        rels = [rels[i] for i in rowidx]
    else:
        rowidx = list(range(len(rels)))

    # Encode dense part
    dense = np.zeros((len(rels), len(dense_p) // 32), dtype=np.uint32)
    dense_w = dense.shape[1]
    for i, r in enumerate(rels):
        r_dense = prime_idx[r]
        r_dense = r_dense[r_dense < len(dense_p)]
        r_bits = sum(1 << int(j) for j in r_dense)
        dense[i, :] = to_uvec(r_bits, dense_w)

    # Encode sparse part
    padding = {}
    if not DEBUG_NO_PADDING:
        # For extra columns, add padding elements (weight 4)
        for j in range(len(primes), len(rels) - 64):
            ii = [random.randrange(len(rels)) for _ in range(4)]
            for i in ii:
                padding.setdefault(i, []).append(j)

    sparse = []
    for ridx, r in enumerate(rels):
        row_idx = prime_idx[r]
        sparse_r = row_idx[row_idx >= len(dense_p)]
        row = sparse_r
        if ridx in padding:
            pad = np.array(padding[ridx], dtype=np.uint32)
            row = np.concatenate([row, pad])
        # no need to sort
        # np.sort(row)
        sparse.append(row)
    return primes, rowidx, dense, sparse


def to_uvec(x: int, length: int):
    assert x.bit_length() <= 32 * length
    return [(x >> (32 * i)) & 0xFFFFFFFF for i in range(length)]


def from_uvec(words: Iterable[int]) -> int:
    return sum(int(x) << (32 * i) for i, x in enumerate(words))


def from_array(array) -> int:
    return int.from_bytes(array.tobytes(), "little")


def to_array(x: int, length: int):
    return np.frombuffer(x.to_bytes(4 * length, "little"), dtype=np.uint32)
