> Historical validation record. Original commands and full raw outputs are
> preserved in the [external archive](../../../ARCHIVE.md).

# PM resolution and more-mode inference results

**The planned spatial-accuracy gates fail.** Time integration, derivative consistency,
more-mode joint inference and engineering replay pass. The four failed force/particle
criteria are retained in [summary.json (archived)](../../../ARCHIVE.md), not loosened. [The guide](README.md)
records the fixed initial field, observed band, priors, controls and error definitions.
See also the open [PM-S1 issue](../../../KNOWN_ISSUES.md).

**Historical sparse-load campaign:** the subsequent [dense-sampling follow-up](../pm_oversampling_validation/RESULTS.md)
implements native N>M and passes bounded refinement criteria. Native-support
limitations below describe the original run; its measured results are preserved.

## One-factor convergence

All initial physical Fourier coefficients match across resolutions within 2e-12.
The last nonreference trial versus each family's finest reference gives:

| Family | Comparison | Field change | Projected posterior gradient change | Field change / noise SD per observed degree |
| --- | --- | ---: | ---: | ---: |
| Force mesh | M=32 vs 64, N=8, 16 steps | 21.923% | 68.496% | 0.466 |
| Time | 32 vs 64 steps, N=8/M=32 | 0.01442% | 0.08403% | 0.000305 |
| Particle lattice | N=16 vs 32, M=32, 16 steps | 16.838% | 50.947% | 0.326 |

Force-gradient changes are not monotonic over M=8,16,32; particle-field changes
also are not monotonic at the coarsest points. The finest references are finite
comparison trials, not verified continuum solutions. Time refinement converges
within its fixed spatial discretization. Coarse spatial choices remain unsuitable
for an accuracy claim at the declared noise/error tolerance.

*Archived figure: Field and projected-gradient convergence (`pm_resolution_validation/figures/convergence.png`). See [archive guide](../../../ARCHIVE.md).*

*Archived figure: Power changes across resolutions (`pm_resolution_validation/figures/resolution_power.png`). See [archive guide](../../../ARCHIVE.md).*

Field-only and mixed field/scalar finite-difference checks at N=8/M=16 and N=16/M=32
(both 16 steps) pass the declared stable-refinement criteria. Best refined scaled
directional errors range from 3.53e-11 to 3.17e-10. The adjoint differentiates the
implemented discrete target consistently; this does not remove discretization bias.

*Archived figure: Directional derivative refinements (`pm_resolution_validation/figures/gradients.png`). See [archive guide](../../../ARCHIVE.md).*

## Force/assignment limitation isolated

Independent NumPy CIC/spectral-force/gather calculations show a strong response to
force-grid/particle ratio for a tiny matched longitudinal displacement. The continuum
linear projection is 1. At perturbation amplitude 1e-4:

| Particle N | Force M | NumPy longitudinal response |
| ---: | ---: | ---: |
| 8 | 8 | 0.900314 |
| 8 | 16 | 1.322494 |
| 8 | 32 | 1.788018 |
| 8 | 64 | 3.022917 |
| 16 | 32 | 1.185309 |
| 32 | 32 | 0.993588 |
| 64 | 32 | 0.991973 (NumPy only; unsupported native geometry) |

A native tiny momentum-kick probe reproduces the NumPy response and RMS force-error
statistics within 4.55e-7 across both kick increments; all 21 cross-check/refinement
assertions pass. The native probe uses EdS to remove the analytically known initial
momentum/kick. The force kernel itself is background independent. Both perturbation
amplitudes and shrinking native kicks preserve the observed discrepancy. Full raw
statistics are in [force_summary.json (archived)](../../../ARCHIVE.md), [native_force.csv (archived)](../../../ARCHIVE.md)
and [numpy_force.csv (archived)](../../../ARCHIVE.md).

*Archived figure: Independent and native force-response comparison (`pm_resolution_validation/figures/force_response.png`). See [archive guide](../../../ARCHIVE.md).*

The matching independent calculation localizes the limitation to the discrete
force/assignment setup. Increasing force resolution at fixed sparse particle load
is not a sufficient continuum-accuracy prescription. Particle oversampling and the
assignment/force-window treatment need a separate controlled follow-up. Native
N>M support was excluded by the constructor for this original run; that restriction
is removed and tested in the linked follow-up. No force law/filter is changed here.

## More active modes: larger joint posterior

Four dispersed joint chains sample 216 white cells (18 active IC directions and
198 exact-prior null directions), spectrum amplitude and noise. Each uses 256 warmup
and 512 retained draws, jittered 8..16 leapfrog steps and target acceptance 0.9.
All 653 monitored features pass Rhat <1.05, bulk ESS >200 and tail ESS >100:

- Maximum Rhat: 1.01279; minimum bulk ESS: 666.51; minimum tail ESS: 891.76.
- No post-warmup divergences; mean acceptance per chain: 0.9028–0.9145.
- Analytic null first/second moments pass; largest discrepancy: 3.52 Monte Carlo
  standard errors. Mean null squared norm: 197.583 (expected 198).
- Direct-DFT initial normalization error: 5.60e-17.
- Posterior-mean/truth correlation: 0.9582 initially and 0.9585 in the final observed band.
- Replicated/observed chi-square means: 56.80 / 56.02 (56 observed real degrees).
  The descriptive posterior-predictive tail fraction is 0.5547.
- Joint log-amplitude mean 0.0337, central 95% [-0.5077,0.5880]; log-noise mean
  -9.1837, central 95% [-9.6069,-8.7538]. Injected values are 0 and log(1e-4).

This is a separate same-model/same-grid mock, so its reconstruction success does
not establish spatial accuracy. It is one finite-chain case, not coverage validation.
Latent density/power uncertainty and noisy posterior predictions are distinct.

*Archived figure: Larger-field reconstruction (`pm_resolution_validation/figures/reconstruction.png`). See [archive guide](../../../ARCHIVE.md).*

*Archived figure: Larger-field power predictions (`pm_resolution_validation/figures/inference_power.png`). See [archive guide](../../../ARCHIVE.md).*

*Archived figure: Joint retained traces (`pm_resolution_validation/figures/traces.png`). See [archive guide](../../../ARCHIVE.md).*

## Software validation and scope

All native simulations and chains completed. Short adaptive replays match serial
within 1.42e-9 on 1/2/4-rank slabs, four-rank pencils and one/two threads; serial
ASan/UBSan matches exactly. The N=6 inference geometry supports slab ranks 1/2 and
2x2 pencils; four-rank slab chain replay uses a separately matched N=8 reference.
Case forward/gradient replay uses N=8 everywhere. Three invalid-control and two
inconsistent-MPI-input checks reject safely before grid construction.

The first plotting pass had a variable-name collision, which was corrected without
rerunning native science. Corrected/final analysis writes all figures/results and
returns nonzero because the four scientific spatial criteria remain failed. There
are 1183 declared main checks: 1179 pass, four fail. This is an explicit scientific
campaign, not a new failing routine-CI target. Full library CI was not rerun.

Builds use Clang 17 with serial FFTW or matching Open MPI/FFTW-MPI, bounds checks and
no GSL. Sanitizers use Clang 20 with ASan/UBSan and leak detection disabled. Pilot
runs are separate. Raw native output, state/predictive traces, manifests, logs,
PNG/PDF figures and complete review diff are preserved locally under
`outputs/fml-fieldlevel-pm-resolution/`. Earlier uncommitted PM checkpoint work is
preserved; numerical PM/adjoint and sampler code are unchanged by this validation.
