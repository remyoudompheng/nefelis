import logging
import random
import time

import flint
import kp
import numpy as np

from nefelis.vulkan import shader, stamp_period


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

    def __init__(self, dense, plus, minus, basis, weight, gpu_idx=0):
        dim, dense_n = dense.shape
        assert dim < 65536 * dense_n
        assert (dim * dense_n) % 4 == 0
        self.defines = {"N": dim, "DENSE_N": dense_n}

        self.basis = basis
        self.dim = dim

        # Prepare tensors
        mgr = kp.Manager(gpu_idx)
        xd = mgr.tensor_t(dense.flatten().view(np.uint32))

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
        self.tensors = [xd, xplus, xminus, xidxp, xidxm]
        bitsize = 32 * sum(t.size() for t in self.tensors)
        logging.debug(f"Matrix format using {bitsize / weight:.1f} bits/coefficient")
        self.flops = 2 * dim * dense_n + size_plus + size_minus
        self.weight = weight
        logging.debug(
            f"{self.flops} FLOPS per matrix multiplication (original weight {weight})"
        )

    def wiedemann_big(self, l: int):
        "Perform Wiedemann algorithm for a single big modulus"
        WGSIZE = 128
        mgr = self.mgr
        dim = self.dim
        N_WG = (dim + WGSIZE - 1) // WGSIZE

        if dim < 1000:
            BATCHSIZE = 32
        else:
            BATCHSIZE = 64
        ITERS = (2 * dim // BATCHSIZE + 2) * BATCHSIZE

        # FIXME: use actual norm
        BLEN = (l.bit_length() + 8 + 31) // 32
        pwords = to_uvec(l, BLEN)
        assert pwords[-2] > 2**16

        defines = self.defines | {"BLEN": BLEN}
        kernel = shader("spmv_bigint", defines)

        # Tensor holding M^k V and M^(k+1) V
        v = np.zeros((2, dim, BLEN), dtype=np.uint32)
        for i in range(dim):
            v[0, i, :] = to_uvec(random.randrange(l), BLEN)
        xv = mgr.tensor_t(v.flatten())
        xiter = mgr.tensor_t(np.zeros(N_WG, dtype=np.uint32))
        # Output sequence out[k] = S M^k V
        xout = mgr.tensor_t(np.zeros(ITERS * BLEN, dtype=np.uint32))
        xmod = mgr.tensor_t(np.array(pwords, dtype=np.uint32))

        tensors = self.tensors + [xv, xiter, xmod, xout]
        algo = mgr.algorithm(tensors, kernel, workgroup=(N_WG, 1, 1))
        mgr.sequence().record(kp.OpTensorSyncDevice(tensors)).eval()

        seq0 = from_uvec(v[0, 0, :])
        t0 = time.monotonic()
        gpu_ticks = 0.0
        for i in range(0, ITERS, BATCHSIZE):
            # Matrix multiplication is very fast so we launch multiple
            # iterations per batch.
            seq = mgr.sequence(total_timestamps=2 * BATCHSIZE)
            for _ in range(BATCHSIZE):
                seq.record(kp.OpAlgoDispatch(algo))
            seq.eval()

            stamps = seq.get_timestamps()
            gpu_ticks += stamps[-1] - stamps[0]

        mgr.sequence().record(kp.OpTensorSyncLocal([xout])).eval()
        vout = xout.data().reshape((ITERS, BLEN))
        sequence = [seq0] + [from_uvec(vout[i, :]) for i in range(ITERS)]

        dt = time.monotonic() - t0
        gpu_dt = gpu_ticks * stamp_period() * 1e-9
        flops = self.flops * ITERS / gpu_dt
        speed = ITERS / gpu_dt

        poly = berlekamp_massey(sequence, l)
        assert len(poly) <= dim + 1, len(poly)

        dt = time.monotonic() - t0
        logging.info(
            f"Wiedemann completed in {dt:.3f}s (GPU {gpu_dt:.3}s, {flops / 1e9:.2f} GFLOPS, {speed:.1f} SpMV/s)"
        )

        return poly

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
        BLEN = (l.bit_length() + 8 + 31) // 32
        maxout = l * l * len(poly)
        ALEN = (maxout.bit_length() + 4 + 31) // 32
        pwords = to_uvec(l, BLEN)
        assert pwords[-2] > 2**16

        defines = self.defines | {"ALEN": ALEN, "BLEN": BLEN, "POLYEVAL": 1}
        kernel = shader("spmv_bigint", defines)

        # Tensor holding M^k V and M^(k+1) V
        av = np.zeros((2, dim, BLEN), dtype=np.uint32)
        for i in range(dim):
            av[0, i, :] = to_uvec(v[i], BLEN)
        xv = mgr.tensor_t(av.flatten())
        xiter = mgr.tensor_t(np.zeros(N_WG, dtype=np.uint32))
        xmod = mgr.tensor_t(np.array(pwords, dtype=np.uint32))
        vpoly = np.zeros((len(poly), BLEN), dtype=np.uint32)
        for k, ak in enumerate(poly):
            vpoly[k, :] = to_uvec(ak, BLEN)
        xpoly = mgr.tensor_t(vpoly.flatten())
        # Output sequence out[k] = S M^k V, initialize with a0 * v
        vout = np.zeros((dim, ALEN), dtype=np.uint32)
        for i, vi in enumerate(v):
            vout[i, :] = to_uvec(poly[0] * vi, ALEN)
        xout = mgr.tensor_t(vout.flatten())

        tensors = self.tensors + [xv, xiter, xmod, xpoly, xout]
        mgr.sequence().record(kp.OpTensorSyncDevice(tensors)).eval()
        algo = mgr.algorithm(tensors, kernel, workgroup=(N_WG, 1, 1))

        t0 = time.monotonic()
        gpu_ticks = 0.0
        count = 0
        for i in range(1, len(poly), BATCHSIZE):
            # Matrix multiplication is very fast so we launch multiple
            # iterations per batch.
            seq = mgr.sequence(total_timestamps=2 * BATCHSIZE)
            for _ in range(min(BATCHSIZE, len(poly) - i)):
                count += 1
                seq.record(kp.OpAlgoDispatch(algo))
            seq.eval()

            stamps = seq.get_timestamps()
            gpu_ticks += stamps[-1] - stamps[0]
        assert count == len(poly) - 1

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


def to_uvec(x: int, length: int):
    assert x.bit_length() <= 32 * length
    return [(x >> (32 * i)) & 0xFFFFFFFF for i in range(length)]


def from_uvec(words: list) -> int:
    return sum(int(x) << (32 * i) for i, x in enumerate(words))
