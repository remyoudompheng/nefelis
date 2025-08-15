"""
Lattice sieve using Vulkan Compute shaders
"""

import logging
import math
import time

import numpy as np
import kp
import flint

from nefelis.vulkan import shader

DEBUG_ROOTS = False
DEBUG_TIMINGS = False
OUTLEN = 256 * 1024


class Siever:
    def __init__(self, poly, primes, roots, threshold, I=14):
        mgr = kp.Manager()
        tprimes = mgr.tensor_t(np.array(primes, dtype=np.uint32))
        troots = mgr.tensor_t(np.array(roots, dtype=np.uint32))
        tqroots = mgr.tensor_t(np.array(roots, dtype=np.uint32))
        tq = mgr.tensor_t(np.zeros(4, dtype=np.int32))
        tout = mgr.tensor_t(np.zeros(2 * OUTLEN, dtype=np.int32))

        if DEBUG_ROOTS:
            extra = {"DEBUG": 1}
        else:
            extra = {}

        WIDTH = 1 << I
        WGROWS = 32
        N_WG = 2 * WIDTH // WGROWS

        defines = {
            "THRESHOLD": threshold,
            "DEGREE": len(poly) - 1,
            "WIDTH": 1 << I,
            "LOGWIDTH": I,
            "WGROWS": WGROWS,
        }

        if primes[-1] > 4 * WIDTH:
            # Manage huge primes
            hugeidx = next(pidx for pidx, p in enumerate(primes) if p > 2 * WIDTH)
            hugep = primes[hugeidx]
            avg_bucket = WIDTH * (math.log(math.log(primes[-1]) / math.log(hugep)))
            bucket = int(1.1 * avg_bucket / 8 + 1) * 8
        
            defines |= {
                    "HUGE_PRIME": hugeidx,
                    "BUCKET_SIZE": bucket,
            }
            thuge = mgr.tensor_t(np.zeros(N_WG * bucket * (WGROWS - 1), dtype=np.uint16).view(np.uint32))

            memhuge = thuge.size() * 4
            logging.info(f"Sieving with huge primes (primes[{hugeidx}]={hugep}, bucket size {bucket}, memory {memhuge>>20}MiB)")
        else:
            thuge = mgr.tensor_t(np.zeros(1, dtype=np.int32))

        if DEBUG_ROOTS:
            defines |= {"DEBUG": 1}
            tdebug = mgr.tensor_t(np.array(roots, dtype=np.uint32))
        else:
            tdebug = mgr.tensor_t(np.zeros(1, dtype=np.int32))

        shader1 = shader("sieve_rat_1roots")
        shader2 = shader("sieve_rat_2sieve", defines)
        algo1 = mgr.algorithm(
            [tprimes, troots, tq, tqroots], shader1, (len(primes) // 256 + 1, 1, 1)
        )
        algo2 = mgr.algorithm(
            [tprimes, tqroots, tq, tout, thuge, tdebug], shader2, (N_WG, 1, 1)
        )

        # Send constants
        mgr.sequence().record(kp.OpTensorSyncDevice([tprimes, troots])).eval()

        self.mgr = mgr
        self.primes = primes
        self.roots = roots
        self.poly = poly
        self.tq = tq
        self.tout = tout
        self.algo1 = algo1
        self.algo2 = algo2
        self.defines = defines

    def sieve(self, q, qr):
        qred = flint.fmpz_mat([[q, 0], [qr, 1]]).lll()
        a, c, b, d = qred.entries()

        self.tout.data()[:2].fill(0)
        self.tq.data()[:] = [a, b, c, d]
        seq = self.mgr.sequence(total_timestamps=16)
        seq.record(kp.OpTensorSyncDevice([self.tq, self.tout]))
        seq.record(kp.OpAlgoDispatch(self.algo1))
        seq.record(kp.OpAlgoDispatch(self.algo2))
        seq.record(kp.OpTensorSyncLocal([self.tout]))
        seq.eval()

        if DEBUG_TIMINGS:
            ts = seq.get_timestamps()
            print([t1 - t0 for t0, t1 in zip(ts, ts[1:])])

        if DEBUG_ROOTS:
            tdebug = self.algo2.get_tensors()[-1]
            self.mgr.sequence().record(kp.OpTensorSyncLocal([tdebug])).eval()
            for pidx, x in enumerate(tdebug.data()):
                p = self.primes[pidx]
                r = self.roots[pidx]
                xx = a * int(x) + b * 1337
                yy = c * int(x) + d * 1337
                if r == p:
                    continue
                assert (xx - r * yy) % p == 0, (p, r)

        bout = self.tout.data()
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

    sv = Siever(u, v, ls, rs, 90)
    t0 = time.monotonic()
    reports = sv.sieve(1000003)
    t = time.monotonic() - t0
    WIDTH = 16384
    AREA = 2 * WIDTH**2
    print(f"Sieved {AREA} in {t:.3f}s (speed {AREA / t / 1e9:.3f}G/s)")

    print(len(reports), "reports")
    for x, y in reports[:100]:
        if math.gcd(x, y) > 1:
            continue
        value = u * x + v * y
        print(f"{x}+{y}i", flint.fmpz(value).factor())
