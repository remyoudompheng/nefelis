# Degree 2 NFS and the Gaussian integer method

The Gaussian Integer Method corresponds to the special case
of discrete logarithm with the Joux-Lercier polynomial choice,
except that the small polynomial is chosen to be a degree 2
polynomial ($x^2+a$) defining a ring of integers which is a UFD.

This is only possible when one of −1, −2, −3, −7, −11, −19, −43, −67, −163
is a square modulo p.

However, more generally for any complex quadratic number field,
the group of units has rank zero and the Number Field Sieve
applies as if factorization was well defined (Schirokauer maps
are not necessary).

## Implementation details

The code currently assumes that (p-1)/2 is prime.

Polynomial selection follows the standard Joux-Lercier polynomial
method, but we restrict to quadratic polynomials with negative
discriminant.

Linear algebra does not introduce "free relations" but selects
a fixed representative for each conjugacy class of algebraic ideals.

Descent may possibly not support all number sizes.

The algorithm should complete in a reasonable time for any input
number between 64 bits and 320 bits. However, degree 2 polynomials
are suboptimal for moduli above 256 bits.

## Mathematical details

Let $\mathcal P$ be the group of principal (fractional) ideals and $U$ be the
group of units of number field $K$. There is an exact sequence:

$$ 0 \to U \to K^\times \to \mathcal P \to 0 $$

We can restrict it to the localization at prime $\ell$ (subgroup of valuation 0 at $\ell$).

$$ 0 \to U \to K^\times_\ell \to \mathcal P_\ell \to 0 $$

If $U$ is finite (this is always true for complex quadratic fields) and $\ell$
is a prime larger than $|U|$ then the following natural morphism is an
isomorphism (the logarithm of a principal fractional ideal is well-defined):

$$ \mathrm{Hom}(\mathcal P_\ell, \mathbb Z / \ell \mathbb Z) \to \mathrm{Hom}(K^\times_\ell, \mathbb Z / \ell \mathbb Z) $$

Similarly, the class group exact sequence involving the group of fractional ideals $\mathcal I$:

$$ 0 \to \mathcal P \to \mathcal I \to \mathrm{Cl}(K) \to 0 $$

can be restricted to the subgroup of $\ell$-valuation 0:

$$ 0 \to \mathcal P_\ell \to \mathcal I_\ell \to \mathrm{Cl}(K) \to 0 $$

If $\ell$ is coprime to the class number of K (extremely likely), the restriction morphism:

$$ \mathrm{Hom}(\mathcal I_\ell, \mathbb Z / \ell \mathbb Z) \to \mathrm{Hom}(\mathcal P_\ell, \mathbb Z / \ell \mathbb Z) $$

is also an isomorphism, meaning that the virtual logarithm of principal ideals can be extended in a canonical way to all ideals.
