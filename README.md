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

Antoine Joux, Reynald Lercier, Nigel P. Smart, Frederik Vercauteren
The Number Field Sieve in the Medium Prime Case
Advances in Cryptology - CRYPTO 2006, 26th Annual International Cryptology Conference
Lecture Notes in Computer Science, 4117, 326-344
https://iacr.org/archive/crypto2006/41170323/41170323.pdf

Razvan Barbulescu, Pierrick Gaudry, Aurore Guillevic, François Morain.
Improving NFS for the Discrete Logarithm Problem in Non-prime Finite Fields.
EUROCRYPT 2015, Proceedings, Part I, Apr 2015, Sofia, Bulgaria. pp.129-155,
https://inria.hal.science/hal-01112879v2/file/BGGM-Eurocrypt15.pdf

Emmanuel Thomé
Fast computation of linear generators for matrix sequences and application to the block Wiedemann algorithm
https://inria.hal.science/inria-00517999v1/document

The CADO-NFS Development Team.
CADO-NFS, An Implementation of the Number Field Sieve Algorithm, Release 2.3.0, 2017
http://cado-nfs.inria.fr/

Emmanuel Thomé
Lecture notes on the Number Field Sieve, 2022
https://cseweb.ucsd.edu/classes/wi22/cse291-14/
