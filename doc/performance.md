# Performance

Benchmark results on standardized inputs: note that performance
is extremely sensitive to parameter choice and hardware throttling.

| Task          |   Number    | Ryzen 7840HS (Cado-NFS) | Radeon 780M iGPU (620a7e37) | RTX 3080 Ti (1340b4eb) |
| ------------- | ----------- | ----------------------- | --------------------------- | ---------------------- |
| factor (GNFS) | 10^90+187   | 133s | 131s  | 140s |
| factor (GNFS) | RSA-110     | 965s | 1087s | 895s |
| factor (GNFS) | RSA-130     |187min|337min |270min|
| factor (GNFS) | RSA-150     | | |
| factor (SNFS) | 2^401-1     | 155s | 128s  | 120s |
| factor (SNFS) | 2^499-1     | 970s | 1290s | 800s |
| factor (SNFS) | 2^599-1     |183min|277min |120min|
| dlog (JL deg3)| 10^75+21259 |      | 28s   |  35s |
| dlog (JL deg3)| 10^90+36307 |      | 230s  | 155s |
| dlog (JL deg3)| 10^105+70399|      | 51min | 22min|
| dlog (JL deg3)|10^120+138139|      | 560min|174min|
| dlog (GNFS)   | 10^75+21259 |  75s | | |
| dlog (GNFS)   | 10^90+36307 | 666s | | |
| dlog (GNFS)   | 10^105+70399|74min | | |
| dlog (GNFS)   |10^120+138139|      | | |

Cado-NFS is using commit d84ae397 (25 Sep 2025) with default parameters.
The label `dlog (GNFS)` refers to polynomial selection using skewed base-m
polynomials (same as factoring). For SNFS, polynomial and configuration
must be supplied manually (see [examples/cado](./examples/cado) for more
details).

The RTX 3080 Ti test was performed on a computer with an old Xeon CPU (Haswell series)
in `--lowcpu` mode.

Discrete logarithm time only includes sieving and linear algebra.

For reference, `pyroclastic` (using the quadratic sieve) on Radeon 780M can factor:
10^90+187 in ~75s, RSA-110 in 3000s, RSA-130 in more than 24 hours.


