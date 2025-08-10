"""
Lattice sieve using Vulkan Compute shaders
"""

import math
import time

import numpy as np
import kp
import flint

from nefelis.vulkan import shader

DEBUG_ROOTS = False
WIDTH = 16384
OUTLEN = 32768


def sieve(u, v, primes, roots, q, threshold):
    qr = (-v * pow(u, -1, q)) % q
    qred = flint.fmpz_mat([[q, 0], [qr, 1]]).lll()
    a, c, b, d = qred.entries()

    mgr = kp.Manager()
    tprimes = mgr.tensor_t(np.array(primes, dtype=np.uint32))
    troots = mgr.tensor_t(np.array(roots, dtype=np.uint32))
    tq = mgr.tensor_t(np.array([a, b, c, d], dtype=np.uint32))
    tout = mgr.tensor_t(np.zeros(2 * OUTLEN, dtype=np.int32))
    tdebug = mgr.tensor_t(np.array(roots, dtype=np.uint32))

    shader1 = shader("sieve_rat_2sieve", {"THRESHOLD": threshold, "DEBUG": 1})
    algo = mgr.algorithm(
        [tprimes, troots, tq, tout, tdebug], shader1, (2 * WIDTH, 1, 1)
    )
    seq = mgr.sequence(total_timestamps=16)
    seq.record(kp.OpTensorSyncDevice([tprimes, troots, tq]))
    seq.record(kp.OpAlgoDispatch(algo))
    seq.record(kp.OpTensorSyncLocal([tout, tdebug]))
    seq.eval()

    if DEBUG_ROOTS:
        mgr.sequence().record(kp.OpTensorSyncLocal([tdebug])).eval()
        for pidx, x in enumerate(tdebug.data()):
            p = primes[pidx]
            r = roots[pidx]
            xx = a * int(x) + b * 1337
            yy = c * int(x) + d * 1337
            if r == p:
                continue
            assert (xx - r * yy) % p == 0, (p, r)

    bout = tout.data()
    # print(bout.reshape((OUTLEN, 2)))
    outlen = min(bout[0], OUTLEN - 1)
    return [(int(bout[2 * i]), int(bout[2 * i + 1])) for i in range(1, outlen + 1)]


def smallprimes(B):
    l = np.ones(B, dtype=np.uint8)
    l[0:2] = 0
    for i in range(math.isqrt(B) + 1):
        if l[i] == 0:
            continue
        l[i * i :: i] = 0
    return [int(_i) for _i in l.nonzero()[0]]


if __name__ == "__main__":
    p = 1000000000000000000000000000000000000000000000000000000000000000000000270907
    u = -2423783356717505418397281193017644615
    v = 33044377492552689242111796852708437154
    ls = smallprimes(300_000)
    rs = [(-v * pow(u, -1, l)) % l if u % l else l for l in ls]

    t0 = time.monotonic()
    reports = sieve(u, v, ls, rs, 1000003, 90)
    t = time.monotonic() - t0
    AREA = 2 * WIDTH**2
    print(f"Sieved {AREA} in {t:.3f}s (speed {AREA / t / 1e9:.3f}G/s)")

    print(len(reports), "reports")
    for x, y in reports[:100]:
        if math.gcd(x, y) > 1:
            continue
        value = u * x + v * y
        print(f"{x}+{y}i", flint.fmpz(value).factor())
