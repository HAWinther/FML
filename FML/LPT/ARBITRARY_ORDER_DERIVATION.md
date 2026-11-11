# EdS growing-mode recursion and stage-1 reference

## Conventions

Use x(q,a) = q + Psi(q,a), g = a/a_ref and

\[
\Psi(q,a)=\sum_{n\geq1}g^n\Psi_n(q),\qquad
A^{(n)}_{ij}=\partial_{q_j}\Psi_{n,i},\qquad
\delta_L(q,a)=g\delta_{\rm ref}(q).
\]

The expansion has no factorial. Each coefficient includes its EdS amplitude.
For the unit box, Fourier waves are exp(2 pi i m.q), k = 2 pi m, derivatives
are i k and the inverse Laplacian is -1/k^2. The mean displacement is zero.
The growing seed obeys div(Psi_1) = -delta_ref and
Psi_1(k) = i k delta_ref(k)/k^2. Consequently
dPsi/dln(a) = sum n g^n Psi_n and physical peculiar velocity is a H times
that derivative after converting box lengths to the desired physical units.

## Original equations used by the independent oracle

Let F = I + A, J = det(F), and C_ij = dJ/dF_ij be the cofactor matrix, so
(F inverse)_ji = C_ij/J. Mass conservation gives 1 + delta = 1/J. Multiplying
the Eulerian divergence of the acceleration by J gives

\[
\sum_{ij}C_{ij}(\partial_t^2+2H\partial_t)A_{ij}
=\frac32H^2(J-1).
\]

For EdS, (partial_t^2 + 2H partial_t) g^n = H^2 lambda_n g^n, with
lambda_n = n(n+1/2). The coefficient equation is therefore

\[
[g^n]\left\{\sum_{ij}C_{ij}(g)
\sum_{r\geq1}\lambda_r g^r A^{(r)}_{ij}
-\frac32[J(g)-1]\right\}=0.
\]

At order n >= 2 the unknown field appears only as
(lambda_n - 3/2) div(Psi_n). Evaluating the remaining terms from orders below
n supplies its divergence. This is implemented by permutation determinants
and 2-by-2 cofactor minors in `tests/eds_reference.py`.

The initial Eulerian flow is irrotational. Its Cauchy invariants require
F^T dF/dln(a) to be symmetric. Equivalently,

\[
\nabla_q\times\Psi_n=
\frac1n\sum_{r+s=n}s\sum_l
\nabla_q\Psi_{r,l}\times\nabla_q\Psi_{s,l}.
\]

The oracle extracts these coefficients with ordered sums. Its sparse integer
wavevectors have no mesh wrapping or intermediate-band truncation.

## Compact source recursion for the planned grid engine

Define three contractions of the full displacement-gradient matrices

\[
Q(A,B)=\operatorname{tr}A\operatorname{tr}B-\operatorname{tr}(AB),
\]
\[
R(A,B,C)=\epsilon_{ijk}\epsilon_{pqr}A_{ip}B_{jq}C_{kr},
\qquad K_i(A,B)=\epsilon_{ijk}\sum_l A_{lj}B_{lk}.
\]

Q is symmetric in its arguments, R is fully symmetric, R(A,A,A) = 6 det(A),
and K(B,A) = -K(A,B). These identities apply to nonsymmetric matrices as well.
From Matsubara (2015), section IV, equations (62)-(64), in the EdS limit,

\[
d_n\equiv\nabla\cdot\Psi_n=
-\frac12\sum_{r+s=n,\ r\leq s}M_{rs}
\left[1-\frac{4rs}{(2n+3)(n-1)}\right]Q(A^{(r)},A^{(s)})
\]
\[
\hspace{1em}-\frac16\sum_{r+s+t=n,\ r\leq s\leq t}M_{rst}
\left[1-\frac{4(rs+st+tr)}{(2n+3)(n-1)}\right]
R(A^{(r)},A^{(s)},A^{(t)}),
\]
\[
c_n\equiv\nabla\times\Psi_n=
\sum_{r+s=n,\ r<s}\frac{s-r}{n}K(A^{(r)},A^{(s)}).
\]

For pairs, M is 1 for equal orders and 2 otherwise. For triples, M is 1 when
all orders are equal, 3 when exactly two are equal, and 6 when all differ.
Every partition contains positive orders. There is no transverse source at
orders 1 or 2. From order 4 onward, lower transverse displacements also enter
the scalar source through their full gradient matrices.

The first coefficients give the following implementation checkpoints. Indices
on Q, R and K denote perturbative orders, not tensor components.

| n | d_n | c_n |
|---|---|---|
| 2 | -3/14 Q_11 | 0 |
| 3 | -5/9 Q_12 - 1/18 R_111 | 1/3 K_12 |
| 4 | -7/11 Q_13 - 17/66 Q_22 - 13/66 R_112 | 1/2 K_13 |
| 5 | -9/13 Q_14 - 7/13 Q_23 - 3/13 R_113 - 5/26 R_122 | 3/5 K_14 + 1/5 K_23 |

## Helmholtz inversion and FML potential conversion

For k != 0 reconstruct the vector with

\[
\Psi_n(k)=\frac{-i k\,d_n(k)+i k\times c_n(k)}{k^2}.
\]

Taking i k dot Psi returns d_n; taking i k cross Psi returns c_n when
k dot c_n = 0. Set Psi_n(0) = 0. The periodic scalar and curl sources must
have compatible zero modes; residual checks include them rather than silently
discarding incompatible source means.

The existing FML API returns scaled potentials, with Psi_1 = grad(phi_1),
Psi_2 = grad(phi_2), and Psi_3 = grad(phi_3a + phi_3b) + curl(A_3).
To translate conventions, let H_ij = partial_i partial_j phi_1 and let
Laplacian(chi) = Q(H,H)/2. Then

- Laplacian(phi_1) = -delta_ref.
- phi_2 = -3 chi/7, so A^(2) = -3 Hessian(chi)/7.
- Laplacian(phi_3a) = -det(H)/3.
- Laplacian(phi_3b) = 5 Q(H,Hessian(chi))/21.
- Laplacian(A_3) = K(H,Hessian(chi))/7, choosing the transverse vector gauge.

These match the independent 3LPT expressions in `FML/tests/test_lpt.cpp`.
Comparisons use the displacement vector, avoiding a dependence on vector-
potential gauge. Retaining just the symmetric gradient after third order
would omit physical contributions at higher orders.

## Stage-1 validation entry points

From the repository root, run

```sh
make -C FML/LPT/tests run
```

This builds a single-rank exporter of the existing FML 3LPT potentials and runs
the Python standard-library-only oracle checks. Local FFTW paths can be passed
through the shared Makefile configuration. Running the Python test file directly
runs the mathematical checks alone and explicitly skips the compiled-library
comparison.

The comparisons cover

- All sparse coefficients through order 4 versus the independent explicit
  unsymmetrized Fourier kernels in Matsubara appendix B, equations (82)-(87).
  Summing all ordered input tuples performs the field-level symmetrization.
  The conversion includes i/n!, with an additional 1/(2 pi) for integer k.
- A four-distinct-wave output, explicitly averaging all 24 kernel permutations.
- Existing FML order-1/2/3 displacement fields, including the transverse third
  order, on both unpadded and padded paths. Every stored half-spectrum mode is
  compared, including coefficients expected to vanish.
- The original scalar equation and the full Cauchy invariants through order 5,
  amplitude homogeneity, real-field Hermitian symmetry, zero modes, zero input,
  plane-parallel exactness, and Helmholtz signs.
- A nonzero third-order transverse field whose removal changes the fourth-order
  divergence, guarding against dropping transverse feedback.

This stage establishes a small-field reference through order 5 and an independent
explicit-kernel comparison through order 4. The generic engine and supported
1/2/4-rank MPI/OpenMP validation are documented in
[`ARBITRARY_ORDER_API.md`](ARBITRARY_ORDER_API.md) and
[`ARBITRARY_ORDER_PLAN.md`](ARBITRARY_ORDER_PLAN.md). Production-scale
bandwidth/resource checks and broader convergence studies remain subsequent work.

On 11 November 2026, all eight checks passed with the compiled-library comparison.
For the Hermitian three-wave fixture, maximum coefficient errors relative to the
largest reference coefficient at each order were 0, 1.95e-16, 2.84e-16 and 5.52e-16
for orders 1 through 4, respectively. These are small-fixture floating-point
agreement results. The third-order transverse displacement has a maximum
coefficient about 9.1% of the maximum total third-order coefficient in this fixture.
The independently rebuilt existing serial `FML/tests/test_lpt` suite passed all
84 checks. The module-test runner discovers this new `LPT/tests/Makefile` through
its existing directory scan.

## Reference

[Matsubara (2015), Recursive Solutions of Lagrangian Perturbation Theory,
section IV and appendix B](https://arxiv.org/abs/1505.01481v3).
