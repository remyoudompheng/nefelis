"""
Lattice sieve using Vulkan Compute shaders
"""

import logging
import math
import time

import numpy as np
import kp
import flint

from nefelis.vulkan import shader, stamp_period

DEBUG_ROOTS = False
DEBUG_TIMINGS = False
OUTLEN = 256 * 1024


class Siever:
    def __init__(
        self,
        poly,
        primes,
        roots,
        threshold,
        I=14,
        poly2=None,
        primes2=None,
        roots2=None,
        threshold2=None,
        /,
    ):
        mgr = kp.Manager()
        tprimes = mgr.tensor_t(np.array(primes, dtype=np.uint32))
        troots = mgr.tensor_t(np.array(roots, dtype=np.uint32))
        tqroots = mgr.tensor_t(np.array(roots, dtype=np.uint32))
        tq = mgr.tensor_t(np.zeros(4, dtype=np.int32))
        tout = mgr.tensor_t(np.zeros(2 * OUTLEN, dtype=np.int32))

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

        GPU_FACTOR = False
        if primes2:
            assert primes2 and roots2 and len(primes2) == len(roots2) and threshold2
            GPU_FACTOR = True
            tprimes2 = mgr.tensor_t(np.array(primes2, dtype=np.uint32))
            troots2 = mgr.tensor_t(np.array(roots2, dtype=np.uint32))
            tqroots2 = mgr.tensor_t(np.zeros(len(roots2), dtype=np.uint32))
            touttemp = mgr.tensor_t(np.zeros(4 * OUTLEN, dtype=np.int32))
            defines["DEGREE2"] = len(poly2) - 1
            defines["THRESHOLD2"] = threshold2

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
            thuge = mgr.tensor_t(
                np.zeros(N_WG * bucket * (WGROWS - 1), dtype=np.uint16).view(np.uint32)
            )

            memhuge = thuge.size() * 4
            logging.info(
                f"Sieving with huge primes (primes[{hugeidx}]={hugep}, bucket size {bucket}, memory {memhuge >> 20}MiB)"
            )
        else:
            thuge = mgr.tensor_t(np.zeros(1, dtype=np.int32))

        if DEBUG_ROOTS:
            defines |= {"DEBUG": 1}
            tdebug = mgr.tensor_t(np.array(roots, dtype=np.uint32))
        else:
            tdebug = mgr.tensor_t(np.zeros(1, dtype=np.int32))

        shader1 = shader("sieve_rat_1roots", defines)
        shader2 = shader("sieve_rat_2sieve", defines)
        tensors1 = [tprimes, troots, tq, tqroots, tout]
        if GPU_FACTOR:
            tensors1 += [touttemp, tprimes2, troots2, tqroots2]
        else:
            tensors1 += [tout]
        algo1 = mgr.algorithm(tensors1, shader1, (len(primes) // 256 + 1, 1, 1))
        algo2 = mgr.algorithm(
            [tprimes, tqroots, tq, touttemp if GPU_FACTOR else tout, thuge, tdebug],
            shader2,
            (N_WG, 1, 1),
        )
        if GPU_FACTOR:
            shader3 = shader("sieve_3factor", defines)
            algo3 = mgr.algorithm(
                [tprimes2, tqroots2, tq, touttemp, tout],
                shader3,
                (OUTLEN // 256, 1, 1),
            )
            mgr.sequence().record(kp.OpTensorSyncDevice([tprimes2, troots2])).eval()

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
        if GPU_FACTOR:
            self.algo3 = algo3
        else:
            self.algo3 = None
        self.defines = defines

    def sieve(self, q, qr):
        qred = flint.fmpz_mat([[q, 0], [qr, 1]]).lll()
        a, c, b, d = qred.entries()

        self.tq.data()[:] = [a, b, c, d]
        seq = self.mgr.sequence(total_timestamps=16)
        seq.record(kp.OpTensorSyncDevice([self.tq]))
        seq.record(kp.OpAlgoDispatch(self.algo1))
        seq.record(kp.OpAlgoDispatch(self.algo2))
        if self.algo3 is not None:
            seq.record(kp.OpAlgoDispatch(self.algo3))
        seq.record(kp.OpTensorSyncLocal([self.tout]))
        seq.eval()

        if DEBUG_TIMINGS:
            ts = seq.get_timestamps()
            print(
                [round((t1 - t0) * stamp_period() * 1e-6) for t0, t1 in zip(ts, ts[1:])]
            )

        if DEBUG_ROOTS:
            print("Q", a, b, c, d)
            tprimes = self.algo1.get_tensors()[0]
            troots = self.algo1.get_tensors()[1]
            tqroots = self.algo1.get_tensors()[3]
            self.mgr.sequence().record(
                kp.OpTensorSyncLocal([tprimes, troots, tqroots])
            ).eval()
            for p, r, qr in zip(tprimes.data(), troots.data(), tqroots.data()):
                # p, r, qr = int(p), int(r), int(qr)
                if qr == p:
                    xx, yy = a, c
                else:
                    xx = a * qr + b
                    yy = c * qr + d
                assert (xx - r * yy) % p == 0, (p, r)
            del p, r, qr
            if self.algo3:
                tprimes2 = self.algo1.get_tensors()[6]
                troots2 = self.algo1.get_tensors()[7]
                tqroots2 = self.algo1.get_tensors()[8]
                self.mgr.sequence().record(
                    kp.OpTensorSyncLocal([tprimes2, troots2, tqroots2])
                ).eval()
                qroots2 = (
                    tqroots2.data().view(np.int16).reshape((len(tprimes2.data()), 2))
                )
                for p, r, qr in zip(tprimes2.data(), troots2.data(), qroots2):
                    p, r, qrx, qry = int(p), int(r), int(qr[0]), int(qr[1])
                    # This is such that:
                    # qrx * y == qry * x mod p IFF X/Y=r mod p where X,Y = Q(x,y)
                    xx = a * qrx + b * qry
                    yy = c * qrx + d * qry
                    if r == p:
                        assert yy % p == 0
                    else:
                        assert (r * yy - xx) % p == 0, (p, r, qrx, qry)

        bout = self.tout.data()
        # print(bout.reshape((OUTLEN, 2)))
        outlen = min(bout[0], OUTLEN - 1)
        return [(int(bout[2 * i]), int(bout[2 * i + 1])) for i in range(1, outlen + 1)]

    def sievelarge(self, q, qr):
        """
        Handle coordinate change on CPU side.
        """
        qred = flint.fmpz_mat([[q, 0], [qr, 1]]).lll()
        a, c, b, d = qred.entries()

        # Since a,b,c,d are still small, shader 1 will succeed
        # without overflow
        self.tq.data()[:] = [a, b, c, d]
        seq = self.mgr.sequence()
        seq.record(kp.OpTensorSyncDevice([self.tq]))
        seq.record(kp.OpAlgoDispatch(self.algo1))
        seq.eval()

        # Hack: Inhibit change of coordinates for output
        self.tq.data()[:] = [1, 0, 0, 1]
        seq = self.mgr.sequence()
        seq.record(kp.OpTensorSyncDevice([self.tq]))
        seq.record(kp.OpAlgoDispatch(self.algo2))
        if self.algo3 is not None:
            seq.record(kp.OpAlgoDispatch(self.algo3))
        seq.record(kp.OpTensorSyncLocal([self.tout]))
        seq.eval()

        bout = self.tout.data()
        bout = bout.reshape((OUTLEN, 2))
        outlen = min(bout[0, 0], OUTLEN - 1)
        results = []
        for i in range(1, outlen + 1):
            x, y = int(bout[i, 0]), int(bout[i, 1])
            results.append((int(a * x + b * y), int(c * x + d * y)))
        return results


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
