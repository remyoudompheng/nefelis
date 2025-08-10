# Nefelis: Number Field Sieve from scratch on GPU

Nefelis is a hobby project to reimplement the Number Field Sieve
from scratch on GPU.

Currently, the goal is to implement only the Gaussian Integer Method
(NFS with polynomials of degree 2/1 for discrete logarithms),
which avoids polynomial selection and Schirokauer map computations.

The implementation follows closely the structure of
[pyroclastic](https://github.com/remyoudompheng/pyroclastic).

## Bibliography

Oliver Schirokauer
Discrete Logarithms and Local Units
Philosophical Transactions: Physical Sciences and Engineering
Vol. 345, No. 1676, Theory and Applications of Numbers without Large Prime Factors (Nov. 15, 1993), pp. 409-423
https://www.jstor.org/stable/54275

Antoine Joux, Reynald Lercier
Improvements to the general number field sieve for discrete logarithms in prime fields. A comparison with the gaussian integer method
Math. Comp. 72 (2003), 953-967
https://www.ams.org/journals/mcom/2003-72-242/S0025-5718-02-01482-5/

The CADO-NFS Development Team.
CADO-NFS, An Implementation of the Number Field Sieve Algorithm, Release 2.3.0, 2017
http://cado-nfs.inria.fr/
