# Joint amplitude/bias fixed-IC pilot — 18 November 2026

The new joint Gaussian bias proposal passed the same declared scalar mixing
gates on all three numerical configurations of the original small Gaussian
tracer-field problem. Target, observed data, physical ICs, cutoffs, noise and
bounded uniform priors are unchanged. New chain seeds were declared in
[plan_joint.json](../../../../validation/galaxy/fixed_ic/plan_joint.json), with
four chains, 256 warmup and 1024 retained draws per configuration. The original
[failed-mixing pilot](../galaxy_fixed_ic/RESULTS.md) stays intact; neither campaign
was extended or pooled with the other.

| N / deposition mesh / width | Mean alpha | SD | Alpha Rhat | Bulk ESS | Tail ESS |
|---|---:|---:|---:|---:|---:|
| 12 / 24 / 6 | 0.99600 | 0.01813 | 1.00561 | 579 | 756 |
| 12 / 36 / 6 | 0.99596 | 0.01804 | 1.00613 | 578 | 689 |
| 16 / 48 / 8 | 0.99610 | 0.01827 | 1.00572 | 649 | 724 |

Across all five parameters and configurations, maximum Rhat=1.00613, minimum
bulk ESS=578 and minimum tail ESS=689. The frozen requirements were Rhat<1.01,
bulk ESS>=400 and tail ESS>=200. All pass. At the finest setting the central
68% amplitude interval is [0.97775,1.01438], for injected alpha=1.

Each configuration used 5124 forward calls in each campaign (including warmup).
Alpha bulk ESS per forward evaluation improved by factors 2.01, 1.75 and 2.40,
respectively. The joint proposal draws the complete bias block at the proposed
amplitude, so this campaign uses zero extra conditional bias sweeps. Its MH ratio
uses both normalized Gaussian proposal densities, while keeping the original
finite prior domain and penalties. Gaussian regularization modifies the
proposal only, not the posterior. No amplitude interpolation, profiling or
analytic bias marginalization is used.

Compared to the finest-mesh posterior, the coarse/intermediate mean shifts are
-0.00513/-0.00731 posterior SD. Approximate paired length-64 batch MC errors are
0.00675/0.00856 SD. The observed shifts plus twice these MC errors are below
0.025 SD, inside the frozen 0.1 SD budget. This is a bounded three-configuration
comparison with common seeds. Its final refinement changes several numerical
settings together; it is not a universal numerical error bound or a complete
independent mesh scan.

Correctness checks include independent joint-posterior quadrature with an
unidentified coefficient that must retain its bounded uniform prior despite
Gaussian proposal regularization, normalized correlated q density/covariance,
and an independent analytic Gaussian-prior conditional solve. All 20 native
sampler/cache checks pass in serial and MPI 1/2/4 ranks. A nonlinear-model short
chain agrees across serial/pencil MPI to 5.7e-13, and the original sampler
reproduces its archived short chain byte for byte numerically.

An initial smoke comparison picked up an older MPI executable with eight
conditional sweeps while serial used zero. It was discarded as a build-settings
mismatch. The final comparison was made after both builds completed, with matching
settings. No scientific campaign was removed or substituted.

This remains one synthetic Gaussian tracer field with 56 active real IC
coefficients and 124 real likelihood degrees of freedom. It is not a discrete
galaxy-catalog result, coverage/SBC study, N-body robustness result or free-IC
mixing validation. Exact challenge IC/spectrum/cosmology inputs, NUFFT catalog
integration, compact latent/product meshes and production restart remain open.
See the [sampler guide](../../../FIXED_IC.md),
[machine-readable outcome](summary.json), and [separate-campaign comparison](comparison.json).
Raw chains, exact executable, manifests and generated figures are kept outside
the library; the maintained driver can regenerate plots without rerunning chains.
