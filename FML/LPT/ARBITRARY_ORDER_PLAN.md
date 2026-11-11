# Arbitrary-order EdS LPT implementation plan

Proposed plan for development backlog item 55, 11 November 2026.

Stage 1 is complete for the small-field reference scope. The conventions,
derived coefficient table, reference equations and validation commands are in
[`ARBITRARY_ORDER_DERIVATION.md`](ARBITRARY_ORDER_DERIVATION.md). Eight focused
checks pass, including compiled FML comparisons through order 3 on padded and
unpadded paths, independent published kernels through order 4 and equation
residuals through order 5. The existing serial LPT suite also passes all 84
checks.

Stage 2 is complete as a written design in
[`ARBITRARY_ORDER_API.md`](ARBITRARY_ORDER_API.md). It specifies explicit input
support and a required owned-memory budget, conservative work-mesh selection,
collected/streamed entry points, Fourier output ownership, EdS evaluation,
roundoff corrections, reporting and collective failure behavior. Work-mesh and
storage examples were checked against the current grid conventions. These are
APIs now implemented by stage 3.

Stage 3 completed the double-precision recursion and coefficient validation in
`ArbitraryOrder.h`. The compact pair/triple recursion, conservative full-bandwidth
mesh, explicit input checks/corrections, planner, collected/streamed outputs and
EdS evaluator are implemented. Native contract tests and sparse order-1..5
comparisons on regular, odd and high-frequency meshes pass. Stage 4 has now added
MPI/OpenMP execution and workspace alternatives; its implementation and measured
validation are summarized below. Empty slabs remain unsupported by the current
`FFTWGrid` layout contract.

## Scope and conventions

Implement a standalone 3D periodic, pressureless, single-fluid, growing-mode
Einstein-de Sitter solver parameterized by a finite requested order P. The first
validated milestone reaches order 5, with order 6 as a further reference check.
The recursion itself must accept P rather than contain a separate formula for
each order. Resource and integer limits bound accepted requests.

Use reference-epoch linear density delta_ref and g = a/a_ref. Define

    x(q,a) = q + sum(n=1..P) g^n Psi_n(q)
    dPsi/dln(a) = sum(n=1..P) n g^n Psi_n(q)

There is no factorial in this expansion. Each Psi_n includes its EdS numerical
coefficient and is homogeneous of degree n in delta_ref. In Fourier space,
Psi_1(k) = i k delta_ref(k)/k^2, with zero mean displacement. Displacements use
the grid's box-coordinate units. A caller obtains physical peculiar velocity
from a H dPsi/dln(a), with a consistent length unit conversion.

Retain transverse displacement from order 3 onward. Its contribution also enters
the full displacement-gradient tensor in every subsequent source. General
cosmology, decaying modes and SimulationCode/COLA wiring are later milestones.

## Existing components to reuse

- `DisplacementFields.h` supplies 1LPT, 2LPT and 3LPT comparison entry points,
  Fourier derivative conventions, Poisson operations and the existing curl term.
- `Dealiasing.h` and `FFTWGrid/FourierResampling.h` supply padding conventions,
  normalized spectral remapping and bounded MPI communication.
- `FML/tests/test_lpt.cpp` has independent sparse-wavevector convolutions for
  3LPT, including transverse and high-frequency intermediate-mode cases.
- `FFTWGrid` supplies slab-distributed transforms and allocation accounting.
  Existing reusable remapping plans and memory-reporting conventions should be
  used where applicable.

## Stage 1 - Derivation and independent low-order reference

Use Matsubara (2015), section IV, equations (62)-(65), for the real-space
divergence/curl recursion in the EdS growing-mode limit. Derive coefficients
with the expansion above and enumerate positive integer pairs and triples
summing to n, including their symmetry multiplicities.

1. Write the recursion in terms of A_n,ij = d Psi_n,i / d q_j. Keep all nine
   components when transverse terms are present.
2. Recover the seed, second-order coefficient and all three third-order pieces,
   translating explicitly to the existing potential conventions.
3. Reconstruct Psi_n from its divergence and curl with a Fourier Helmholtz solve.
   Verify its signs directly from i k dot Psi and i k cross Psi.
4. Build an independent sparse Fourier reference through at least order 5,
   using published Fourier kernels or coefficient extraction from the original
   Lagrangian equations. It must not call the production source contractions.
5. Compare fourth-order configurations with a published explicit kernel after
   symmetrization and factorial conversion. Include non-collinear modes giving
   a nonzero transverse field and transverse feedback into the next order.

Acceptance gate: agreement through order 3 with both the existing solver and
its independent reference, plus an independently checked order-4 result.

## Stage 2 - Numerical and API contract

Proposed entry point is `compute_eds_lpt(delta_ref, options)` in
`FML/LPT/ArbitraryOrder.h`, under `FML::COSMOLOGY::LPT`.

Options specify maximum order, explicit input spectral support, product
evaluation policy and a workspace budget. Results expose cropped Fourier
displacement vectors per order and an evaluator for the summed displacement
and dPsi/dln(a). Provide optional per-order delivery to a caller without forcing
retention of all cropped outputs. Internal lower-order state still has a cost.

Before allocation, validate order, dimensions, layouts, finite inputs, support
and budget consistently across MPI ranks. Report work mesh, retained fields,
planned transform count, owned workspace and estimated transient/FFT overhead.

Specify zero modes, Hermitian symmetry and Nyquist handling. Use the existing
strictly-sub-Nyquist resampling convention. A support cutoff must be explicit
and recorded in output metadata. Floating-point tolerances scale with field
amplitude and precision.

Acceptance gate: the written API and normalization contract supports low-order
comparisons and defines exactly which band-limited continuum solution is computed.

## Stage 3 - Serial reference engine and bandwidth policy

First implement a transparent common-work-mesh algorithm. Compute derivatives
spectrally, accumulate pair/triple contractions in real space, transform the
sources, reconstruct each vector field, and retain its required internal modes.
Crop only the externally requested outputs.

For the first exact band-limited implementation, use a conservative support
bound. If the input has componentwise integer support |m_i| <= K_i, choose an
even common work width M with M > 2 P max(K_i), rounded for the MPI layout.
All products through P then fit strictly below the work-mesh Nyquist frequency.
This is deliberately expensive and serves as a reference implementation.

Full-band inputs can make this policy require roughly P times the original
width, or P^3 times the cells per work field. Plan and reject over-budget
requests before allocating. Small grids and explicitly band-limited seeds are
the initial practical workloads.

Derive a cheaper retained-band policy separately if required for production.
Track which intermediate modes can contribute to the final output band and
prove both alias exclusion and intermediate retention for the whole recursion.
Agreement of individually padded products is insufficient if intermediate
cropping discards a later physical coupling. Validate any reduced policy against
the conservative engine and sparse convolution before enabling it by default.

Acceptance gate: serial orders 1-5 agree mode by mode with independent references,
including cases where high intermediate modes return to the output band.

## Stage 4 - Workspace, MPI and OpenMP

**Complete for supported nonempty slabs.** `ArbitraryOrder.h` now runs collectively
on 1, 2 or 4 MPI ranks, validates common options and local layouts, exchanges only
the input Hermitian boundary planes in bounded chunks, uses the distributed FFT
and reusable Fourier remapping plans, and coordinates validation, sink and
recoverable allocation failures before the next collective. OpenMP row loops cover
derivative construction, source contractions, source inversion and evaluation.
Both nine-gradient caching and a three-order displacement LRU recomputation policy
are selectable. Capacity preflight and execution report local, maximum-rank and
sum-of-rank bounds, timings, FFT/remap counts, and workspace policy. Empty slabs
are rejected because `FFTWGrid` currently requires the mesh width to be divisible
by the rank count and does not support zero-width slabs in this path.

Validation command (with local FFTW paths):

```sh
make -C FML/LPT/tests run USE_MPI=true USE_OMP=true USE_FFTW=true \
  FFTW_INCLUDE=/opt/local/include FFTW_LIB=/opt/local/lib \
  MPIRUN=/opt/local/bin/mpirun NP_LIST='1 2 4'
```

It passed 41 native API/contract checks, 9 independent Python reference checks,
and 34/80/160 MPI checks at 1/2/4 ranks. The 24 order-5 coefficient comparisons
across rank counts, 1/2 threads and both workspace policies had worst relative
error `9.63e-16`. Injected rank-local allocation, sink, malformed-input, option
mismatch and remapping failures returned collectively. A fresh-process order-8,
N=32 benchmark fixture measured the following (RSS is process `ru_maxrss`; the
four-rank `rss_max` is the largest rank and `rss_sum` sums rank maxima):

| Ranks | Policy | FFTs | Max planned bytes/rank | Max accounted bytes/rank | Max RSS/rank | Max seconds |
| ---: | :--- | ---: | ---: | ---: | ---: | ---: |
| 1 | cache | 91 | 30,664,104 | 29,217,064 | 58,232,832 | 0.130 |
| 1 | recompute | 343 | 25,664,136 | 24,217,096 | 53,264,384 | 0.344 |
| 2 | cache | 91 | 15,470,208 | 14,746,688 | 47,824,896 | 0.133 |
| 2 | recompute | 343 | 12,967,200 | 12,243,680 | 45,424,640 | 0.315 |
| 4 | cache | 91 | 8,892,032 | 8,515,008 | 42,815,488 | 0.153 |
| 4 | recompute | 343 | 7,408,544 | 7,031,520 | 41,394,176 | 0.321 |

On this small fixture, recomputation reduces the reported per-rank bound by
roughly 16-21%, while using 3.8 times as many FFTs and taking about 2.1-2.6 times
as long. These measurements establish the tradeoff, not production-scale
performance. Users should choose the policy against their memory and runtime
limits. The reported accounted bytes are capacity accounting, not RSS; FFTW,
MPI and runtime allocations explain part of the measured RSS gap.

Reuse derivative/source scratch fields. Compare retaining three displacement
components per order and recomputing derivatives with caching nine gradients;
select a bounded policy based on measured memory and FFT costs. All earlier
orders can be dependencies, so release/recomputation must follow the actual
dependency graph.

Parallelize cell contractions with OpenMP and use existing MPI FFT/remapping
interfaces. Validate supported slab layouts, including empty slabs where the
chosen mesh/backend permits them. Use collective failure paths for invalid
requests and resource limits. Measure fresh-process peak RSS and rank timings
alongside reported capacities and FFT counts.

Acceptance gate met for the supported rank/thread matrix above. Add empty-slab
support only with a corresponding FFTWGrid/backend change and separate tests.

## Stage 5 - Scientific validation and delivery

- Existing padded 1LPT/2LPT/3LPT comparisons, with transverse terms included.
- Independent order-4/5 references and an order-6 smoke/reference configuration.
- Plane-parallel exactness: terms above first order vanish before shell crossing.
- Amplitude scaling Psi_n[lambda delta] = lambda^n Psi_n[delta], time scaling
  g^n and finite-difference verification of dPsi/dln(a).
- Zero input, translations/reflections, Hermitian symmetry and zero modes.
- Coefficientwise Lagrangian equations and Cauchy-invariant checks through P.
  The invariant must use the full displacement gradient and time derivative.
- Resolution and support convergence, including deliberately alias-sensitive cases.
- Truncation studies with per-order norms and det(I + grad Psi). Report loss
  of the single-stream description at shell crossing and possible series
  nonconvergence separately from numerical error. Positive sampled Jacobians
  alone do not prove convergence or exclude crossings between cells.
- Precision, rank/thread and memory-budget failure checks.

Deliver the header, focused tests following the existing harness, a small
standalone example, and documentation of conventions, validated orders, support
policies and measured resource costs. Run existing LPT regressions as well as
the new order-4/5 checks. Add the focused tests to the appropriate CI target.

## Suggested sequence and effort

1. Derivation, normalization and independent order-4 check: 3-5 working days.
2. Conservative serial recursion and order-5 reference: 4-7 working days.
3. Bounded workspace, MPI/OpenMP and resource diagnostics: 3-5 working days.
4. Scientific convergence checks, example and documentation: 3-5 working days.

Allow roughly 3-5 focused weeks for the initial validated module. A cheaper
bandwidth policy and production-scale calibration can add substantial work.
Revisit estimates after the derivation and bandwidth gates.

## References

- Matsubara (2015), Recursive Solutions of Lagrangian Perturbation Theory,
  section IV and appendix B: https://arxiv.org/abs/1505.01481v3
- Michaux et al. (2020), Accurate initial conditions for cosmological N-body
  simulations, section 2.3: https://arxiv.org/abs/2008.09588
- Existing library conventions and padding behavior: `README.md` and
  `DisplacementFields.h` in this directory.
