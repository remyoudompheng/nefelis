# Nefelis: Number Field Sieve from scratch on GPU

Nefelis is a hobby project to reimplement the Number Field Sieve
from scratch on GPU, with support from FLINT for less compute
intensive steps of the algorithm.

The implementation follows closely the structure of
[pyroclastic](https://github.com/remyoudompheng/pyroclastic).

It targets home computers with integrated GPUs (such as AMD Ryzen APUs)
or discrete GPUs (notably Nvidia RTX cards).

## Implementation status

The project intentionally uses duplicated, non-generic code
for several variants of the Number Field Sieve, and makes
use of additional assumptions to help readability.

It is work in progress, not in a usable state.

Planned variants are described in the following table.

| Identifier | Purpose | Description | Status | Comments |
| ---------- | ------- | ----------- | ------ | -------- |
| `deg2` | Discrete logarithm in GF(p) | Gaussian integer method | ✅ | Polynomial selection inspired by Joux-Lercier |
| `deg3` | Discrete logarithm in GF(p) | Joux-Lercier method with degree 3/2 polynomials | ✅ | Enforces a single Schirokauer map |
| `fp2`  | Discrete logarithm in GF(p²)| Conjugation method with degree 4/2 polynomials  | ⚠️ | Basic (broken) implementation |
| `fp3`  | Discrete logarithm in GF(p³)| TBD | ❌ | |
| `factor`  | Integer factorization    | General NFS for factoring | 🐢 | Slower than Cado-NFS with equivalent computing power |
| `factor --snfs`  | Integer factorization    | Special NFS for factoring | 🐢 | Tries to find polynomial automatically        |

The project is not currently open to external contributions.

## Dependencies

The most notable project dependencies are:

* [Kompute](https://github.com/KomputeProject/kompute) for Python bindings to Vulkan
* [python-flint](https://python-flint.readthedocs.io) for computer algebra

The program can use [yamaquasi](https://github.com/remyoudompheng/yamaquasi)
as an optional dependency (version 0.2.2 or later). Without `yamaquasi`
cofactorization will be handled by FLINT but is usually noticeably slower.

## Bibliography

See [bibliography file](./doc/bibliography.md)

