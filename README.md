# Nefelis: Number Field Sieve implementation for GPU

Nefelis is a hobby project to reimplement the Number Field Sieve
using GPU shaders (Vulkan) and Python, with support from FLINT for less compute
intensive steps of the algorithm.

The implementation follows closely the structure of
[pyroclastic](https://github.com/remyoudompheng/pyroclastic).

It targets single systems with gaming-oriented GPUs
(notably iGPU found in AMD Ryzen APUs, or Nvidia RTX series cards).

## Implementation status

The project intentionally uses duplicated, non-generic code
for several variants of the Number Field Sieve, and makes
use of additional assumptions to help readability.

Planned variants are described in the following table. Several
parts of the implementation are suboptimal or broken.

| Identifier | Purpose | Description | Status | Comments |
| ---------- | ------- | ----------- | ------ | -------- |
| `deg2` | Discrete logarithm in GF(p) | Gaussian integer method | ✅ | Polynomial selection inspired by Joux-Lercier |
| `deg3` | Discrete logarithm in GF(p) | Joux-Lercier method with degree 3/2 polynomials | ✅ | Tries to avoid Schirokauer maps |
| `fp2`  | Discrete logarithm in GF(p²)| Conjugation method with degree 4/2 polynomials  | ⚠️ | Basic (broken) implementation |
| `fp3`  | Discrete logarithm in GF(p³)| TBD | ❌ | |
| `factor`  | Integer factorization    | General NFS for factoring | ✅ | Only degree 3/4 polynomials |
|`factor --snfs`| Integer factorization| Special NFS for factoring | ✅ | Tries to find polynomial automatically |

The project is not currently open to external contributions.

## Dependencies

The most notable project dependencies are:

* [Kompute](https://github.com/KomputeProject/kompute) for Python bindings to Vulkan
* [python-flint](https://python-flint.readthedocs.io) for computer algebra
* [yamaquasi](https://github.com/remyoudompheng/yamaquasi) for fast small number factorization

Note that Kompute 0.9 may require manual compilation and it depends
on `numpy==1.26.4` which is not packaged for Python 3.13 and above.

## Bibliography

See [bibliography file](./doc/bibliography.md)

