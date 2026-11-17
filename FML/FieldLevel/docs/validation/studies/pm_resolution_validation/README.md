> Historical validation record. Original commands and full raw outputs are
> preserved in the [external archive](../../../ARCHIVE.md).

# Bounded PM resolution and more-mode inference validation

**Spatial accuracy remains unresolved.** Four declared force/particle accuracy
criteria fail in this campaign. Time refinement, directional derivatives, a larger
joint posterior and backend/sanitizer replay pass. [RESULTS.md](RESULTS.md) records
both outcomes; [PM-S1](../../../KNOWN_ISSUES.md) tracks the accuracy follow-up. No force,
adjoint or sampler implementation is changed by this validation.

**Follow-up:** [native N>M and dense-sampling results](../pm_oversampling_validation/RESULTS.md)
now implement the previously blocked experiment. The measurements and failed
criteria below describe the original sparse-load campaign and remain unchanged.

## Matched physical initial conditions

The box has length 1, initial linear power `P=0.002*exp(log_amplitude)`, initial
cutoff 10 and observed cutoff 15. The active initial wavevectors have integer
squared norm 1 or 2: nine conjugate pairs, or 18 real directions. For one member
of each pair (first nonzero component positive), define

```text
w_N(q) = sqrt(2/N^3) sum_k [z_kc cos(2 pi k.q) + z_ks sin(2 pi k.q)].
```

The production whitening map gives physical Fourier coefficients
`sqrt(P/2)*(z_kc - i*z_ks)`, independently of N. Counter-based seed 91 and a key
from the integer wavevector fix the same coefficients across all grids. Reusing a
grid-white seed without this construction would change the physical realization.
The emitted `.initial.csv` records the 18 physical orthonormal coefficients; their
cross-resolution agreement is checked to 2e-12. Initial density is extrapolated to
a=1; evolution starts at a=0.2 and ends at a=0.6 in flat Omega_m=0.3.

All forward/gradient comparisons reuse one noisy dataset generated at N=16/M=32
with 16 steps and noise seed 92, power 1e-4. The retained observed band has 56 real
degrees of freedom. Fourier coefficients are exported on a common 8^3 rFFT index
set, covering all nonzero k<15 without a Nyquist ambiguity.

| Family | Fixed settings | Refinements | Finest comparison reference |
| --- | --- | --- | --- |
| Force mesh | N=8 particles/IC, 16 steps | M=8,16,32,64 | M=64 |
| Time steps | N=8, M=32 | 2,4,8,16,32,64 | 64 steps |
| Particle/IC lattice | M=32, 16 steps | N=4,8,16,32 | N=32 |

Each family varies one factor and has its own **finite** reference, not an exact
continuum solution. Common-band field norms include full-Hermitian weights. Raw
white-grid derivatives have resolution-dependent normalization, so gradients are
compared after projection onto the same 18 orthonormal Fourier coordinates, plus
log-amplitude/noise derivatives. Two resolutions also check field-only and mixed
field/scalar directional derivatives against independent forward-value differences.
The derivative check requires multiple refined steps to agree, not one lucky step.

Declared finest-correction criteria are relative field error <5%, relative projected
gradient error <15%, and field discrepancy <0.1 measurement-noise SD per observed
real degree of freedom. Improvement from the coarsest trial is also checked. These
bounds were set before inspecting the spatial results and are retained after failure.
CIC is piecewise differentiable; no universal convergence order is assumed.

## More active modes and full-field inference

A separate N=6/M=12, four-step same-model mock samples all **216 white coordinates**
and two nuisance parameters. Its 18 active initial directions use the matched
field above. Independent Gaussian white seed 193 supplies the 198 null coordinates,
which are orthogonally projected away from the active subspace before adding the
matched signal. Null coordinates remain independent unit Gaussians in the posterior.
This is a larger inference consistency test, not resolution-bias calibration.

Four starts use white seeds 300..303 and scales 0.5,1,1.5,2. Scalar starts are
spread within their prior bounds. Both log-amplitude and log-noise have Gaussian
prior width 0.5; amplitude is centered at 0 on [-1,1], noise at log(1e-4) on [-16,-4].
Chains use seeds 11000..11003, 256 warmup and 512 retained draws, jittered 8..16
leapfrog steps, dual averaging from step 0.08 targeting acceptance 0.9, identity
white metric, fixed scalar masses and positive-energy divergence threshold 10.
Mass adaptation is disabled. Pilot runs are separate evidence.

Diagnostics use every retained draw, including rejections: all 216 white cells,
18 active projections/squares, 198 null projections/squares, norms, potential and
both varying scalars (653 features). Declared gates are Rhat <1.05, bulk ESS >200,
tail ESS >100, finite/support-valid states and no post-warmup divergences. Exact
null first/second moments use five Monte Carlo standard errors with ESS capped at
actual sample count. These are finite-chain diagnostics, not convergence proofs.

Every eighth retained draw supplies initial/final latent density and power
intervals. Independent Gaussian measurement-noise draws use the production
Fourier variance convention. Initial power covers two supported shells; final
power/density and residuals use the observed band. Pointwise latent uncertainty
and noisy posterior predictions are distinct. One mock cannot establish coverage.

## Independent force/assignment investigation

The failed spatial criteria prompted two additional diagnostics. `assignment_probe.py`
implements NumPy CIC deposition, the full-complex continuous spectral force with
component-Nyquist/DC handling, and CIC gathering. It has no PM integration, adjoint
or inference. A longitudinal perturbation `x=q+epsilon*psi(q)` has continuum linear
force `F=epsilon*psi`, so the normalized projection `dot(F,psi)/(epsilon*dot(psi,psi))`
should approach 1 in a suitable continuum limit. Amplitudes 1e-3 and 1e-4 probe this.

`ForceProbe.cpp` extracts the native force from a tiny public-model momentum kick.
It uses EdS for analytic initial momentum and kick normalization, a0=1e-4 and two
increments 1e-8/1e-9. `force_analysis.py` checks both normalized force-response and
RMS-error statistics against NumPy and shrinking-kick consistency. This localizes
the limitation to the current discrete force/assignment setup.

The original NumPy N=64/M=32 point was unsupported by the native constructor at
that time. Native support and extended force-probe cases are now implemented;
[the separate follow-up](../pm_oversampling_validation/RESULTS.md) saves their results.
Rerunning the extended force probe produces extra N>M rows and a third plot panel;
the historical data/figures in this directory are preserved. No smoothing or
force-law correction is applied. Future numerical-law changes must update PM identity.

## Reproduce

```sh
make -C FML/FieldLevel/pm_resolution_validation pm_resolution pm_resolution_diagnostics pm_force_probe USE_MPI=false
python3 FML/FieldLevel/pm_resolution_validation/run.py --bin-dir /path/to/binaries --output /path/to/fresh/OUT --jobs 4
```

The explicit campaign requires NumPy and Matplotlib. **Analysis returns nonzero
when the scientific accuracy gates fail**, after writing the report/plots. This
campaign is not part of routine CI. Use separate/clean builds for MPI slabs or
`USE_MPI=true USE_PENCIL_GRID=true` pencils with normal FML compiler/library overrides.
The generic C++ diagnostics executable reuses the existing full-field diagnostic
source. Short replay is engineering evidence, separate from the longer serial chains.

After the campaign, reproduce the independent investigation:

```sh
python3 FML/FieldLevel/pm_resolution_validation/assignment_probe.py --initial OUT/science/data-reference.initial.csv --output OUT/force-probe
/path/to/pm_force_probe OUT/science/data-reference.initial.csv OUT/force-probe/native.csv
python3 FML/FieldLevel/pm_resolution_validation/force_analysis.py --input OUT/force-probe --output OUT/analysis
```

Native single-case and chain interfaces:

```sh
pm_resolution case PREFIX N M STEPS DATA_PATH FD_FLAG
pm_resolution chain PREFIX N M STEPS CHAIN_ID WARMUP DRAWS LEAPFROG PREDICTIVE_STRIDE
```

`DATA_PATH=-` creates a same-model mock; otherwise it reads the common Fourier data.
Controls are deliberately bounded (N<=64, 4<=M<=64, steps<=64 for the main driver).
Replicated inputs are checked before grid construction; inconsistent MPI geometry
or data paths reject collectively. `pm_force_probe` requires one rank. Canonical
field gathers are bounded validation output, not a production chain-storage design.

The force probe accepts an optional final `FORCE_WINDOW` argument (0/1), and the
independent NumPy probe accepts `--force-window 0|1`. Compensated probes add a
1e-10 kick to check refinement of the amplified sparse-load response. Default
probe steps/output remain unchanged. [The window experiment](../pm_window_validation/RESULTS.md)
retains the coarse-step failure and reports accepted refined checks separately.
