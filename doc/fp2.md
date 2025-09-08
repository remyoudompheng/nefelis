# Discrete logarithms in GF(p²) using the conjugation method

This method is described in the following paper.

Razvan Barbulescu, Pierrick Gaudry, Aurore Guillevic, François Morain.
Improving NFS for the Discrete Logarithm Problem in Non-prime Finite Fields.
EUROCRYPT 2015, Proceedings, Part I, Apr 2015, Sofia, Bulgaria. pp.129-155,
https://inria.hal.science/hal-01112879v2/file/BGGM-Eurocrypt15.pdf

It selects polynomials f of degree 4 and g of degree 2, such that
g is irreducible over GF(p) and f, g have a common root in GF(p²).

The procedure is identical to the general NFS.

## Mathematical description

### Polynomial selection process

The polynomial definition corresponds to the following setting. Let $d$ be the extension
degree of the finite field we are interested in.

Let $K = \mathbb Q(j)$ be a quadratic extension of $\mathbb Q$ defined by a small polynomial,
such that $K$ is split at prime $p$.

Then define a polynomial $g_j(x)$ of degree $d$ with small coefficients in $\mathcal O_K$
such that $g_j(x)$ is irreducible over one of the residue fields of $K$ at prime $p$.

Then we define $f(x) = \text{Norm}_{K / \mathbb Q}(g_j)$, which is a polynomial
with small coefficients in $\mathbb Z$, of degree $2d$ with a residue field isomorphic
to $\mathbb F_{p^d}$.

Final polynomial $g$ is defined by a lift to integer coefficients of $g_j \mod p$,
which are coefficients of size $O(\sqrt p)$ using a rational reconstruction $j = u/v \mod p$.

In practice we select:

* $K$ as a real quadratic field
* $f$ with no real roots so that the unit group of $K_f$ has rank 1 and coincides with the unit group of $K$
* $f$ must split modulo large prime factors of $p-1$ for the Schirokauer maps

### Roots and conjugation automorphism

Since $K_f$ has a norm map to $K$, we can observe that $f$ has roots modulo a small prime $l$
if and only if $K$ is split at prime $l$. In particular, the conjugation automorphism $\pi: K_f/K$
is also well-defined modulo $l$. It can be computed using the coefficients of $g_j$.

There are at most 2 pairs of conjugate roots, corresponding to the roots of $g_j$ and
its conjugate.

### Multiplicative group of GF(p²)

To compute discrete logarithms in $\mathbb F_{p^2}$ it is useful to consider
the exact sequence with the inclusion $\mathbb F_p \subset \mathbb F_{p^2}$
and the map $z \to z/\bar z$ mapping $\mathbb F_{p^2}^\times$ to the subgroup
of elements of norm 1.

$$ 0 \to \mathbb F_p^\times \to \mathbb F_{p^2}^\times \to^{z/\bar z} \mathbb F_p^\times \to 0 $$

Then for any odd prime divisor $\ell$ of $p^2-1$, $\ell$ is either a divisor of $p-1$ or a divisor of $p+1$.

Depending on the choice of $K$, units of $K_f$ map to 1 either through the
Galois-invariant map (norm) or through the anti-invariant map
$z \mapsto z/\bar z$ in the norm 1 subgroup. For convenience, we choose the first case
(as in Barbulescu-Gaudry-Guillevic-Morain) which is easier to handle.

In the first case, we need to compute discrete logarithms in $\mathbb F_p$, which is similar
to other NFS methods. The norm map $\mathbb F_{p^2} \to \mathbb F_p$ corresponds to the
norm map $K_{f} = K_{g_j}$ to $K$ and any prime ideal of $K_f$ will have
the same discrete logarithm as its conjugate under the action of $\text{Gal}(K_f/K)$.

However, since the lift of primes to $K_f^\times$ is not canonical, this is only correct
if we have selected a Galois-equivariant Schirokauer map: this is done by composing the norm
map, with the Schirokauer map of $K = \mathbb Q(\sqrt D)$:

$$ K_f^\times \to K^\times \to K_\ell^\times \simeq \mathbb Z_\ell^\times \to \mathbb Z / \ell \mathbb Z $$

Units of $K_f$ map to units of the subfield $K$ which has residue field $\mathbb F_p$ at p.
They have no specific property: a Schirokauer map is necessary to define virtual logarithms properly,
if and only if $K$ is a real quadratic field.

In the second case, we need to compute discrete logarithms in the norm 1 subgroup of $\mathbb F_{p^2}^\times$.
The computation is unchanged if we replace all algebraic numbers $z$ by $z / \bar z$.
Any conjugate pair of prime ideals of $K_f = K_{g_j}$ (with respect to $\text{Gal}(K_f/K)$)
will have opposite discrete logarithms.

Units of $K_f$ map to norm 1 units, and they have no specific property: a Schirokauer map is necessary
again. However, the Schirokauer map given by reduction to a single $\mathbb F_\ell$ field is not equivariant
for the Galois action, and can break symmetry. To restore symmetry, we can use a pair of conjugate
Schirokauer maps and compute the anti-invariant component of the pair.
