"""
Generate deterministic, normalized example outputs
"""

import argparse
import os
import random

import flint
import numpy as np
from nefelis.integers import smallprimes, factor, factor_smooth


def main():
    argp = argparse.ArgumentParser()
    argp.add_argument(
        "TYPE",
        choices=("p", "p2", "c", "csemi"),
        help="Generate primes for GF(p), primes for GF(p^2), composites, semiprimes",
    )
    argp.add_argument(
        "SEED",
        choices=("pow10", "fibo", "e", "pi", "random"),
        help="Generate numbers using this seed family",
    )
    argp.add_argument("MINBITS", type=int)
    argp.add_argument("MAXBITS", type=int)
    args = argp.parse_args()

    # Precomputed things
    smalls.extend(smallprimes(10000))
    flint.ctx.dps = 1000

    for bits in range(args.MINBITS, args.MAXBITS + 1, 10):
        sname, s = seed(args.SEED, bits)
        p = generate(args.TYPE, s)
        print(bits, f"{sname}+{p - s}", p)


def seed(kind: str, size: int):
    if kind == "pow10":
        return f"10^{size // 10 * 3}", 10 ** (size // 10 * 3)
    elif kind == "fibo":
        n = int(1.44 * size)
        u, v = 0, 1
        for _ in range(n):
            u, v = v, u + v
        return f"fibonacci({n})", u
    elif kind == "e":
        exp10 = (size * 3) // 10
        e_round = (flint.arb(1).exp() * flint.arb(10) ** exp10).floor()
        assert e_round.is_exact()
        return f"floor(e*10^{exp10})", int(e_round.fmpz())
    elif kind == "pi":
        exp10 = (size * 3) // 10
        pi_round = (flint.arb.pi() * flint.arb(10) ** exp10).floor()
        assert pi_round.is_exact()
        return f"floor(π*10^{exp10})", int(pi_round.fmpz())
    elif kind == "random":
        return f"random({size})", random.getrandbits(size) | (1 << (size - 1))


smalls: list[int] = []


def generate(typ: str, start: int) -> int:
    if typ == "p":
        # Generate a strong prime:
        # we want start+i prime and (start+i-1)/2 prime
        arr = np.zeros(200_000, dtype=np.uint8)
        for l in smalls:
            # (start+r) is divisible by l?
            r = (-start) % l
            arr[r::l] += int(1)
            # (start+r2-1)/2 is divisible by l?
            if l > 2:
                r2 = (r + 1) % l
                arr[r2::l] += int(1)
        for idx in (arr == 0).nonzero()[0]:
            n = start + int(idx)
            if not flint.fmpz(n).is_probable_prime():
                continue
            if not flint.fmpz(n // 2).is_probable_prime():
                continue
            return n
    elif typ == "p2":
        # Try to get large factors for p-1 and p+1
        # Generate p=2q-1 such that q and (p-1)/12 are also prime
        arr = np.zeros(100_000_000, dtype=np.uint8)
        for l in smalls:
            # (start+r) is divisible by l?
            r = (-start) % l
            arr[r::l] += int(1)
            # (start+r2+1)/2 is divisible by l?
            if l > 3:
                r2 = (r - 1) % l
                arr[r2::l] += int(1)
            # (start+r2-1)/3 is divisible by l?
            if l > 3:
                r3 = (r + 1) % l
                arr[r3::l] += int(1)
        for idx in (arr == 0).nonzero()[0]:
            n = start + int(idx)
            if n % 12 != 1:
                continue
            if not flint.fmpz(n).is_probable_prime():
                continue
            if not flint.fmpz((n + 1) // 2).is_probable_prime():
                continue
            if not flint.fmpz((n - 1) // 12).is_probable_prime():
                continue
            return n
    elif typ in ("c", "csemi"):
        # Look for a composite number not divisible by any small prime.
        arr = np.zeros(10_000, dtype=np.uint8)
        for l in smalls:
            r = (-start) % l
            arr[r::l] += int(1)
        for idx in (arr == 0).nonzero()[0]:
            n = start + int(idx)
            if flint.fmpz(n).is_probable_prime():
                continue
            facs = factor_smooth(n, min(32, n.bit_length() // 10))
            if len(facs) > 1:
                continue

            if typ == "csemi":
                facs = factor(n, threads=os.cpu_count())
            else:
                # Any number which looks hard to factor
                facs = factor_smooth(n, min(63, n.bit_length() // 4))
            if len(facs) == 2 and all(f**3 > n for f, _ in facs):
                # Good semiprimes are accepted immediately
                if all(flint.fmpz(f).is_probable_prime() for f, _ in facs):
                    return n
            if len(facs) > 1:
                continue
            return n


if __name__ == "__main__":
    main()
