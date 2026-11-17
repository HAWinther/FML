# Slab/pencil 1LPT/2LPT — step 2 results

14 November 2026. The distributed LPT model and posterior are now generic over a
3D grid backend. This extends [step 1](LINEAR_GRID_RESULTS.md); the original slab
aliases remain available. See [the API guide](../../FRAMEWORK.md#distributed-1lpt-stage-4b-12-november-2026).
Survey observations, checkpoint layout portability and the serial BAO backend are
not ported by this step.

## Cross-layout tests

The new `test_framework_lpt_grid_mpi.cpp` passes with Clang 17, Open MPI, OpenMP
(2 threads), double-precision FFTW and FFTWGrid bounds checks. Its checks are
collective and counted once per process group.

| MPI ranks | Checks passed | Pencil process grids on initial N=8 |
| --- | --- | --- |
| 1 | 87 | 1x1 |
| 2 | 129 | 1x2, 2x1 |
| 4 | 157 | 1x4, 2x2, 4x1 |
| 6 | 95 | 2x3, 3x2, 6x1 |
| 8 | 136 | 2x4, 4x2, 8x1 |

Every process-grid case tests both Zeldovich and 2LPT evolution with distinct
initial/bias/likelihood cutoffs, nonzero quadratic/tidal/Laplacian bias, a
nonidentity Fourier response and plane-parallel RSD. Snapshots compare prediction
and mock modes, posterior potentials, field/scalar gradients, large-displacement
field gradients, all physical reconstruction fields, the frozen metric and a short
joint-HMC endpoint. Full fields/modes are assembled **only in tests**, keyed by
canonical global coordinates/wavevectors rather than backend storage indices.

The largest snapshot comparison error is **9.55e-15**, using
`abs(a-b)/max(1,abs(a),abs(b))`. Slabs are the reference where the task count divides
N=8; the 6-rank comparison uses the first pencil layout as its reference.
This is not a general bitwise reproducibility or HMC mixing/calibration claim.

Additional tests cover unequal cell-aligned N=10 / N_density=20 domains, meshes
not divisible by total task count, and an 8→12 transfer on a compatible unequal
physical domain. Kernel widths 4/6/8 pass. On 8 ranks a nontrivial 2LPT model with
initial N=6 and density N=12 runs on 4x2 pencils, exceeding the slab rank limit.
Incompatible fine-grid domain edges are rejected collectively before constructing
the fine-grid workspace. Fresh workspaces preserve both model-owned grid domains
after changes to the global default domain.

## Independent derivative and physical checks

- Analytic Fourier coefficient fields verify coarse/fine transfer values directly,
  including DC and excluded Nyquist planes. Weighted transfer/transpose dot products
  pass within 2e-13. There is no extra volume rescaling of Fourier coefficients.
- An independent sparse deposition/reverse dot test routes colliding and distant
  node requests, including a rank with no requests, within 2e-13; returned node
  cotangents match the analytic mesh field within 1e-14.
- Posterior central finite differences cover the white field and every spectrum,
  bias and noise parameter. Across all tested cases, maximum relative errors are
  3.04e-9 for the ordinary field direction and 3.05e-8 for scalar directions.
- The weighted forward/adjoint dot-product error is at most 1.61e-12. A separate
  field finite difference with the latent state amplified by 80 exercises large
  periodic/rank-crossing displacements; its maximum error is 9.22e-8. Existing
  2e-6 derivative acceptance bounds were retained.
- Two orthogonal waves u=(1,1,0), v=(0,0,1) give the independent analytic source
  `S2=A*B*cos(u.q)*cos(v.q)`. Density, all displacement components and physical
  peculiar velocities match their analytic EdS 2LPT expressions within 2e-12.
  A diagonal single wave cancels the second-order source within 1e-12.
- A direct sum over analytically displaced particles, including first/second-order
  RSD factors, differs from the finite smooth-mesh prediction by at most 6.47e-6;
  the nonzero second-order signal is 0.0252. The finite assignment/deconvolution
  approximation is retained; this comparison is not exact NUFFT accuracy.

MAP lowers the objective, reconstruction invalidates the forward cache, joint
scalar sampler states remain replicated, and one-rank malformed fields fail
collectively. The three-sample joint-HMC comparison is an execution/parity test,
not scientific nonlinear posterior validation.

## Regression and instrumentation

- ASan/UBSan builds pass the new test at 1/2 ranks (87/129 collective checks).
  External MPI/FFTW libraries are uninstrumented and MPI leak detection is disabled.
- Existing slab LPT tests pass in Clang serial and GCC 14 serial (44 checks each),
  and on MPI 1/2/4 ranks (44/80/160 reported checks).
- Existing physical reconstruction tests pass on 1/2/4 ranks (48/96/192 checks).
- Existing slab survey observation/likelihood tests pass on 1/2/4 ranks
  (57/114/228 checks). This is slab compatibility, not pencil survey validation.
- The step 1 generic linear tests pass again on 1/2/4/6/8 ranks
  (42/61/77/76/91 collective checks).
- The registered `framework-lpt-grid-mpi` Make target rebuilds and passes on 4 ranks.

Older regressions mix collective and summed rank-local checks; their reported counts
are not directly comparable across rank counts. FFTW uses a matching Open MPI build;
FFTW threading is disabled. No timing, peak-memory or strong-scaling claim is made.

## Build and limits

```sh
make -C FML/FieldLevel/tests clean
make -C FML/FieldLevel/tests framework-lpt-grid-mpi \
  USE_MPI=true USE_OMP=true MPIRUN=mpirun NP_LIST='1 2 4 6 8'
```

Set `CXX_MPI`, `FFTW_INCLUDE` and `FFTW_LIB` to matching installations. PencilGrid
requires MPI/FFTW and its implementation object. Both meshes must be even, with
density mesh at least as large as the initial mesh. The domain must align with
both; independently chosen mesh domains are not supported by this interface.
The initial cutoff remains below half the initial-grid Nyquist.

FFTWGrid retains its cached resampling plan. Pencil transfers currently rebuild
ownership/staging records through the shared Fourier-copy implementation. Assignment
retains its existing checked MPI-int limits. Production forward/adjoint operations
introduce no full-field replication. Caching pencil transfers, survey observations,
checkpoint geometry, PM/COLA and broader scientific/performance validation remain
follow-ups. The original serial BAO compatibility API remains separate.
