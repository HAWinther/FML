> Historical validation record. Original commands and full raw outputs are
> preserved in the [external archive](../../../ARCHIVE.md).

# Particle-sampled PM force reference (15 November 2026)

**Particle sampling is controlled to much better precision than the remaining
force-grid density difference for this fixture.** The N96→N128 correction on M64
is 0.0424%, while M32→M64 at N128 changes the observed density by 3.7749%.
The latter still fails the unchanged 0.1-noise-SD criterion (0.1063). All
consistency, conservation, particle and timestep checks pass. The failed force
criterion is retained in [summary.json (archived)](../../../ARCHIVE.md).

This resolves the particle-load ambiguity in the earlier strong91 density
comparison within the tested band. It does **not** close its posterior-gradient
accuracy criteria: no new adjoints or inference are run. The [fixed protocol](README.md)
uses one strong realization, 122 initial directions and 256 observed real degrees.
The original force law, all previous measurements and numerical-law identity are
unchanged.

## Measured finite corrections

| Comparison | Relative observed density change | Change / noise SD per observed degree |
| --- | ---: | ---: |
| N64 vs N128, M32, 32 steps | 0.33715% | 0.009173 |
| N96 vs N128, M32, 32 steps | 0.06477% | 0.001762 |
| N64 vs N128, M64, 32 steps | 0.36435% | 0.010260 |
| N96 vs N128, M64, 32 steps | 0.04237% | 0.001193 |
| M32 vs M64, N128, 32 steps | 3.77491% | 0.106306 |
| 32 vs 64 steps, N128/M64 | 0.01193% | 0.000336 |

Particle corrections improve from N64 to N96 against the common N128 reference
on both force grids. At fixed N64/96/128, the M32→M64 relative density changes are
3.7699%/3.7901%/3.7749%. Their stability supports attributing the density sensitivity
primarily to force-grid discretization in this experiment, rather than particle
sampling. This is an inference from controlled finite comparisons, not proof of
an exact continuum solution or of a specific assignment-window error formula.

*Archived figure: Particle-sampled force reference (`pm_force_reference_validation/figures/reference.png`). See [archive guide](../../../ARCHIVE.md).*

The shell-power response also stays almost unchanged with particle refinement.
The full-mesh RMS includes unobserved modes and is shown separately from the
observed-band error. N128/M64 has eight particles per force cell; it remains a
finite reference. The conclusions are limited to these bands, amplitudes and time
nodes, and do not select an optimal replacement force kernel.

## Engineering and reproducibility

Seven native trajectories complete on four MPI ranks with the pencil backend and
one OpenMP thread per rank. Each is below the declared five-minute limit. Two
additional N128/M32/two-step runs check four-rank slab/pencil agreement: maximum
mode difference 1.39e-17 and physical initial-coefficient difference 3.36e-15.
The N64/M32 and N64/M64 density outputs reproduce the prior serial full-adjoint
campaign within 2.78e-17 in Fourier coefficients; physical initial coefficients
agree within 2.27e-15.

The extended shared driver is clean-built with Clang17/O2 serial, MPI slabs and
MPI pencils, plus Clang20/O1 serial ASan/UBSan (leak detection disabled).
FFTWGrid bounds checks are enabled. Eighteen small native test invocations cover:

- Original CLI/full-mode outputs, gradients, finite-difference table and metadata:
  **bitwise identical** under the old and new serial binaries.
- Forward-only fields/value against full mode on serial, MPI 1/2/4 slabs and
  pencils, and sanitizer: maximum absolute discrepancy 4.21e-12 (value summation
  order); forward mode emits no gradient file.
- Collective rejection of unsupported large adjoints, forward FD/MAP/mock requests,
  invalid flags/size, incompatible frozen data and inconsistent rank controls.

Forward-only mode uses the existing production prediction and normalized likelihood
value; it requests no pullback or reverse trajectory. No production headers,
optimizer or sampler are modified. Memory scales with the current IC/particle
lattice and is not a measured peak-memory guarantee. The larger N bound belongs
to this bounded validation CLI, not a new library-wide mesh restriction.

The launcher records exact commands/PIDs and handles timeouts and keyboard
interruptions. Stub timeout and interrupt tests confirm child termination; a real
two-rank MPI sleeping job also times out and both ranks terminate. The actual
science trajectories finish normally. The one nonzero campaign return is analysis
reporting the retained noise-scaled force criterion. [engineering.json (archived)](../../../ARCHIVE.md)
records numerical checks; full commands/logs and review diff are in the local
artifact. An inherited unused-capture warning in `FieldLevelBAO.h` remains.
Full-library CI and large-N sanitizer execution are not run.

## Next step

The M64 density reference now has a measured particle-refinement correction small
compared with the force-grid difference. Next evaluate explicit force/assignment
window choices against this controlled setup, then verify their adjoints and
posterior behavior. Any adopted numerical-law change must update PM checkpoint
identity. This forward-only pass leaves the prior gradient failures and expensive,
unconverged MAP diagnostics open. More realizations, wider physical regimes,
external references, production scaling, posterior coverage and PM tracer/RSD/survey
integration remain outside this slice; item #52 remains open.
