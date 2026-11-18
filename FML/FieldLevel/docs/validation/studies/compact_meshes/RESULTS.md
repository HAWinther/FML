# Compact latent/product mesh validation

Recorded 18 November 2026. Bounded numerical checks pass.
[Machine-readable cases](summary.json); [configuration and scaling](../../../COMPACT_MESHES.md).

The new path samples an N8 real white field, evolves particles and forms operators
on N12, and deposits onto N24. The reference samples on N12 and retains the same
N12 evolution/operator and N24 deposition grids, cutoffs, spectrum and parameters.
Both source fields are evaluated independently from analytic Fourier waves, with
the analytically required sqrt(512/1728) white-field amplitude on the reference.
Thus smaller state size is not obtained by reducing particle/product resolution.
The initial cutoff 2.7 fundamental modes exceeds half the compact Nyquist but
respects half the product Nyquist. Smooth and PSWF (epsilon=1e-10) are both tested.

| Measured quantity | Largest error over serial/MPI layouts |
| --- | --- |
| Physical Fourier prediction, absolute | 7.09983e-17 |
| All four bias columns, absolute | 6.93889e-17 |
| Compact field finite difference, relative | 9.95681e-09 |
| All scalar finite differences, relative | 2.9467e-08 |

The conditional likelihood and scalar gradients also match the padded reference.
Tests independently check the compact Gaussian mock-noise amplitude by direct
Fourier sums, the unit-Gaussian prior and field gradient, field-metric powers,
symmetry/positivity and identity on discarded Nyquist directions. Small HMC runs
retain two draws and agree across slab/pencil layouts to the declared 2e-10
trajectory tolerance. These are integration checks, not mixing/calibration.
Catalog mode export, Hermitian layout, workspace moves and rank-local malformed
state/configuration rejection are included. No full-field gather is used by the
library; the test-only global tables permit independent comparisons.

The final formatted-source suite passes 38 serial checks and 78/166/208 checks
on 1/2/4 MPI ranks. Every small pencil process-grid orientation and compatible
unequal domains run. The existing shared-mesh Eulerian suite passes 63 serial
and 107/111/111 MPI checks. Fixed-IC cache/sampler (20 checks per rank count)
and catalog (36/57/76 checks) regressions pass at 1/2/4 ranks.
Builds use Clang 17, double precision, FFTW and bounds checks; MPI runs use Open
MPI with two OpenMP threads per rank. Exact commands, logs, source hashes and
plots are outside the library in workspace `outputs/fieldlevel-compact-meshes/`.

State storage drops from 1728 to 512 doubles per sampler field vector: a factor
3.375, or 70.4 percent less. This is **not a total-memory measurement**. The large
padded workspaces remain, and compact transfers add a compact FFT grid, a lifted
product-sized field and plans/metadata. Runtime, total RSS, production scaling
and mixing gains are not established. Nyquist/out-of-band prior-only directions
still exist: this is not a packed active-mode sampler.

No public challenge catalog was downloaded and no new posterior campaign was
run. The existing Cartesian survey-likelihood adapter is not added for this
Eulerian configuration. Packed active-mode storage, independently chosen
particle/evolution/operator resolutions, production restart, initial cubic
filtering and real-data/free-IC scientific validation remain open.
