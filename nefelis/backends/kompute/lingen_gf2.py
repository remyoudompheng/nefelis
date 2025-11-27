"""
GPU implementation of lingen_gf2
"""

import time

import kp
import numpy as np

from nefelis import vulkan


def mslgdc(E, delta, b):
    B = b // 32 + 1
    M, MN = E.shape

    # Assumptions from shader
    assert M % 8 == 0 or M <= 8
    assert MN % 16 == 0 or M <= 16

    mgr = kp.Manager()
    xdelta = mgr.tensor_t(np.array(delta + [0], np.uint32))
    p = np.zeros((2, MN, MN, B), np.uint32)
    p[0, :, :, 0] = np.identity(MN)
    xp = mgr.tensor_t(p.flatten())
    e = np.zeros((2, M, MN, B), np.uint32)
    mask = (1 << (32 * B)) - 1
    for i in range(M):
        for j in range(MN):
            e[0, i, j] = to_array(E[i, j] & mask, B)
    xe = mgr.tensor_t(e.flatten())
    xpt = mgr.tensor_t(np.zeros((MN, MN), np.uint8).view(np.uint32).flatten())
    xs = mgr.tensor_t(np.zeros(MN, np.uint32))

    code = vulkan.shader("lingen_step1")
    code2 = vulkan.shader("lingen_step2")
    # HACK: workaround Kompute issue where constants type is improperly set
    spec_consts = np.array([M, MN, B], np.uint32).view(np.float32)
    alg = mgr.algorithm(
        [xdelta, xp, xe, xpt, xs], code, workgroup=(1, 1, 1), spec_consts=spec_consts
    )
    alg2 = mgr.algorithm(
        [xdelta, xp, xe, xpt, xs], code2, workgroup=(MN, MN, 1), spec_consts=spec_consts
    )

    mgr.sequence().record(kp.OpTensorSyncDevice(alg.get_tensors())).eval()
    seq = (
        mgr.sequence(total_timestamps=10)
        .record(kp.OpAlgoDispatch(alg))
        .record(kp.OpAlgoDispatch(alg2))
    )
    for _ in range(b):
        seq.eval()

    mgr.sequence().record(kp.OpTensorSyncLocal(alg.get_tensors())).eval()

    P = np.zeros((MN, MN), dtype=object)
    p = xp.data().reshape((2, MN, MN, B))
    for i in range(MN):
        for j in range(MN):
            P[i, j] = from_array(p[b % 2, i, j, :])
    return P


def test_shader():
    mgr = kp.Manager()
    M = 32
    MN = 64
    B = 256
    # Simulate N=120000 m=n=24
    M = 24
    MN = 48
    B = 315
    # Simulate N=400000 m=n=24
    M = 24
    MN = 48
    B = 1000
    # Simulate N=500000 m=n=32
    M = 32
    MN = 64
    B = 1000
    # Simulate N=500000 m=n=64
    # M = 64
    # MN = 128
    # B = 500

    delta = [i // 8 for i in range(MN)]
    for idx in (3, 7, 9, 11, 12, 13, 15, 24, 37, 50):
        if idx < MN:
            delta[idx] += 1
    print(delta)
    assert len(delta) == MN
    for i, di in enumerate(delta):
        assert di <= max(delta[i:]) + 1
    xdelta = mgr.tensor_t(np.array(delta + [0, 0, 0, 0], np.uint32))
    xp = mgr.tensor_t(np.zeros((2, MN, MN, B), np.uint32).flatten())
    xe = mgr.tensor_t(np.zeros((2, M, MN, B), np.uint32).flatten())
    xpt = mgr.tensor_t(np.zeros((MN, MN), np.uint8).view(np.uint32).flatten())
    xs = mgr.tensor_t(np.zeros(MN, np.uint32))

    code = vulkan.shader("lingen_step1")
    code2 = vulkan.shader("lingen_step2c")
    spec_consts = np.array([M, MN, B], np.uint32).view(np.float32)
    alg = mgr.algorithm(
        [xdelta, xp, xe, xpt, xs], code, workgroup=(1, 1, 1), spec_consts=spec_consts
    )
    alg2 = mgr.algorithm(
        [xdelta, xp, xe, xpt, xs], code2, workgroup=(MN, MN, 1), spec_consts=spec_consts
    )

    seq = (
        mgr.sequence(total_timestamps=10)
        .record(kp.OpTensorSyncDevice(alg.get_tensors()))
        .record(kp.OpAlgoDispatch(alg))
        .record(kp.OpAlgoDispatch(alg2))
        .record(kp.OpTensorSyncLocal(alg.get_tensors()))
    ).eval()
    ts = seq.get_timestamps()
    print(
        "timestamps",
        [
            round(vulkan.stamp_period() * (t1 - t0) * 1e-9, 6)
            for t0, t1 in zip(ts, ts[1:])
        ],
    )

    # If P and E are zero, the final matrix is the permutation matrix
    # making delta sorted.
    out_delta = xdelta.data()
    print(out_delta)
    pt = xpt.data().view(np.uint8).reshape((MN, MN))
    print(pt)
    for i in range(MN):
        (j,) = pt[i, :].nonzero()[0]
        assert pt[i, j] == 1
        assert out_delta[j] == delta[i], (i, j)

    # If E is nonzero:
    # The delta permutation is given by P[i, argmin(pij=1)]
    # Delta is incremented by shifts
    # Shift[i] is 1 iff (E @ PT)[:,i] is nonzero
    print("Nonzero E")
    xdelta.data()[:] = delta + [0, 0, 0, 0]
    print("delta before", xdelta.data())
    rng = np.random.default_rng(seed=42)
    xp.data().reshape((2, MN, MN, B))[0, :, :, 0] = np.identity(MN)
    xe.data().reshape((2, M, MN, B))[0] = rng.integers(0xFFFFFFFF, size=(M, MN, B))
    print(xe.data())
    e = np.copy(xe.data().reshape((2, M, MN, B))[0])
    print("E0", e[:, :, 0] & 1)

    seq = (
        mgr.sequence(total_timestamps=10)
        .record(kp.OpTensorSyncDevice(alg.get_tensors()))
        .record(kp.OpAlgoDispatch(alg))
        .record(kp.OpAlgoDispatch(alg2))
        .record(kp.OpTensorSyncLocal(alg.get_tensors()))
    ).eval()
    ts = seq.get_timestamps()
    print(
        "timestamps",
        [
            round(vulkan.stamp_period() * (t1 - t0) * 1e-9, 6)
            for t0, t1 in zip(ts, ts[1:])
        ],
    )

    out_delta = xdelta.data()
    out_shift = xs.data()
    out_p = xp.data().reshape((2, MN, MN, B))[1]
    out_e = xe.data().reshape((2, M, MN, B))[1]
    print("delta after", out_delta)
    assert out_delta[MN] == 1
    pt = xpt.data().view(np.uint8).reshape((MN, MN))
    print(pt)
    # print(list(zip(*pt.nonzero())))
    for i in range(MN):
        j = pt[i, :].nonzero()[0][0]
        assert pt[i, j] == 1
        assert out_delta[j] == delta[i] + out_shift[j]

    E0 = e[:, :, 0] & 1
    EP0 = (E0 @ pt) & 1
    print(EP0)
    print(np.bitwise_or.reduce(EP0, axis=0))
    assert np.all(np.bitwise_or.reduce(EP0, axis=0) == out_shift)
    print("shifts", out_shift, "sum", np.sum(xs.data()))
    assert np.sum(out_shift) <= M

    # Check output arrays
    for i in range(M):
        for j in range(MN):
            ref_eij = 0
            for k in range(MN):
                if pt[k, j] == 1:
                    eik = from_array(e[i, k])
                    ref_eij ^= eik
            out_eij = from_array(out_e[i, j])
            if out_shift[j] == 0:
                assert out_eij == ref_eij >> 1
            else:
                assert out_eij == ref_eij

    for i in range(M):
        for j in range(MN):
            pij = from_array(out_p[i, j])
            if out_shift[j] == 0:
                assert pij == pt[i, j]
            else:
                assert pij == pt[i, j] << 1

    # Benchmark
    seq = (
        mgr.sequence(total_timestamps=10)
        .record(kp.OpAlgoDispatch(alg))
        .record(kp.OpAlgoDispatch(alg2))
    )
    t0 = time.monotonic()
    for _ in range(1000):
        break
        seq.eval()
    dt = time.monotonic() - t0
    print("1000 iters in", dt)


def from_array(array) -> int:
    return int.from_bytes(array.tobytes(), "little")


def to_array(x: int, length: int):
    return np.frombuffer(x.to_bytes(4 * length, "little"), dtype=np.uint32)


if __name__ == "__main__":
    test_shader()
