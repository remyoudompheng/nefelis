import random

import gmpy2
import nefelis_rust

for _ in range(10):
    x = random.getrandbits(8)
    p = random.getrandbits(8) | 1
    assert gmpy2.legendre(x, p) == nefelis_rust.legendre_symbol(x, p), (
        x,
        p,
        gmpy2.legendre(x, p),
        nefelis_rust.legendre_symbol(x, p),
    )

for _ in range(1_000_000):
    x = random.getrandbits(32)
    if random.getrandbits(1):
        x = -x
    p = random.getrandbits(32) | 1
    assert gmpy2.legendre(x, p) == nefelis_rust.legendre_symbol(x, p)
