# First scientific FieldLevel project: LEFTfield-style galaxy inference

Started 17 November 2026. The reusable Eulerian operators are now composed with unbiased LPT matter, an explicit fixed-noise likelihood and fixed/free-field prior policies. The separate joint-proposal synthetic fixed-IC pilot passes its declared scalar mixing and bounded numerical-refinement gates; the original failed pilot remains preserved. Catalog ingestion and the real-data amplitude result remain open. No galaxy-catalog posterior has been computed. See [current implementation](EULERIAN_BIAS.md) and [data audit](leftfield/catalog-manifest.json).

## Scientific objective

Deliver an FML analysis of periodic galaxy catalogs that infers sigma8 while marginalizing the initial density field and galaxy-bias parameters. Produce an explicitly labeled reproduction configuration and a separate robustness configuration. Keep other cosmological parameters fixed initially. This is an application of the general FieldLevel framework, with reusable additions to that framework.

The first reviewable scientific deliverable should be a **one-box, fixed-IC sigma8 posterior with Eulerian bias and numerical-convergence plots**. Full free-IC sampling follows only after this passes. Continue the perturbative LPT route for this project; additional PM accuracy campaigns are not prerequisites for it.

## Reference specification to freeze

The primary reference is [2405.02252v2, sections 5.3 and 6.3](https://arxiv.org/html/2405.02252v2), the current version behind the supplied PDF. Its real-space analysis uses z=1 periodic boxes, 2LPT, filtered Eulerian bias, and alpha=sigma8/sigma8_fid with other cosmological parameters fixed. The baseline uses four operators, a cubic likelihood cutoff kmax=0.1 h/Mpc with Lambda=1.4 kmax, fixed Poisson stochasticity, and bounded uniform priors. It first fixes ICs in box 1 and then samples ICs for the ten boxes independently. Extended modeling adds cubic bias and scale-dependent stochasticity. These configurations must remain distinct; the model assumptions affect reported precision.

Use [2403.03220v3, Appendix A](https://arxiv.org/html/2403.03220v3#S5) as a supporting implementation reference, not an interchangeable challenge configuration: its filters, priors and bias marginalization differ. Consult [2409.10937v2](https://arxiv.org/html/2409.10937v2) for numerical-convergence methodology. No numerical error bars or claimed information gain are targets to tune toward.

The [official challenge repository](https://github.com/ANSalcedo/Beyond2ptMock) provides the public catalogs. Its README describes ten real-space boxes of side 2000 Mpc/h, with number density about 4.5e-4 (h/Mpc)^3. Public availability of the exact box-1 IC file, non-amplitude cosmology and reference linear spectrum has **not** been verified. Those are initial data dependencies. Do not assume the organizers' special IC release is included with the catalogs, and do not use the companion paper's halo datasets as substitutes for these galaxy catalogs.

## Verified FML starting point

| Component | Present | Project-specific work |
|---|---|---|
| Spectrum and priors | `Framework.h`: P(k) callback, explicit d ln P derivatives, bounded Gaussian/uniform scalar priors | Add an amplitude-only spectrum provider and import the actual reference spectrum |
| LPT | `DistributedLPT.h`: differentiable ZA/EdS-like 2LPT, grid-templated execution | Expose an unbiased evolved-matter component and separate latent/product/output resolutions |
| Tracers | Current LPT model deposits **advected Lagrangian weights** | Add **Eulerian** operators after evolving/filtering matter; coefficient names alone do not make the models equivalent |
| Filtering | Separate initial/bias/likelihood cutoffs, currently radial selections in the relevant paths | Add shared, explicit spherical/cubic selection descriptors and apply them consistently |
| Gaussian target | Fourier normalization and Hermitian weights; log-noise normalization | Fixed-noise/free-parameter layout and later a k-dependent covariance with derivatives |
| Posterior plumbing | Generic posterior; current Gaussian path uses `p.back()` for noise | Make noise configuration explicit instead of disguising a fixed parameter as an almost-zero-width prior |
| NUFFT | `NUFFT.h` has grid-templated particle type-1/type-2 operations and position-force machinery | Catalog adapter, reusable workspaces and an independently verified FieldLevel position adjoint |
| MPI | Slab/pencil frontends and distributed model contracts | Preserve both in each new component; avoid replicated full fields or active-coefficient all-gathers |
| Checkpoints | Generic framework has trajectory-boundary checkpoint/restart | Extend to the selected blocked field/scalar sampler and application identities; recent PM pilot does not expose mid-chain restart |
| High-order LPT | `ArbitraryOrder.h` supplies forward machinery | Do not assume it supplies an inference adjoint; it is not a prerequisite for cubic **bias** with 2LPT |

Source pointers: [Framework.h](/Users/hansw/Claude/FML/FML/FieldLevel/Framework.h), [DistributedLPT.h](/Users/hansw/Claude/FML/FML/FieldLevel/DistributedLPT.h), [Frontend.h](/Users/hansw/Claude/FML/FML/FieldLevel/Frontend.h), [Checkpoint.h](/Users/hansw/Claude/FML/FML/FieldLevel/Checkpoint.h), [NUFFT.h](/Users/hansw/Claude/FML/FML/NUFFT/NUFFT.h), [ArbitraryOrder.h](/Users/hansw/Claude/FML/FML/LPT/ArbitraryOrder.h).

The three recent PM mocks establish bounded same-model evidence for 122 active coefficients. They do not validate a galaxy model, full-band inference, or this catalog analysis. Their dense fixed metric and hard-coded two-scalar target are validation prototypes, not production components for this project.

## Mathematical contracts

Use normalized Fourier-series coefficients internally. For a white field s and a fixed reference spectrum at the analysis epoch, construct delta_L=alpha L_fid s and set P_L(k;alpha)=alpha^2 P_fid(k). The existing spectrum interface then uses d ln P/d alpha=2/alpha.

If retaining the earlier log-power coordinate A, **A=2 ln alpha**. A uniform prior in alpha is not uniform or Gaussian in A: include the coordinate Jacobian. In whitened coordinates the unit Gaussian field prior requires no extra power-spectrum determinant. A physical-field parameterization needs its own field-measure factor; do not reuse the 122-mode Jacobian with a new active dimension.

Proposed pipeline:

```
white ICs -> reference-spectrum/amplitude map -> initial filter
          -> padded 2LPT -> displaced matter -> NUFFT density
          -> Eulerian matter filter -> Eulerian operator columns
          -> linear combination with bias coefficients -> likelihood selection
          -> normalized Fourier likelihood + declared priors
```

Start with operator columns delta, centered delta^2, centered K_ij K_ij and laplacian(delta). Compute them from the evolved filtered matter field. Specify the b2 convention, tidal normalization, physical derivative units and all mean subtractions. Copying prior endpoints before matching these conventions is unsafe. The third-order option must include the full chosen cubic basis, including its nonlocal tidal operator, with verified reverse paths; adding delta^3 alone is insufficient. Supporting operator definitions and alternative bias bases are documented in [the companion's Appendix A](https://arxiv.org/html/2403.03220v3).

For independent retained modes, express the target as

```
U = 0.5 sum_k w_k [ |d_k-m_k|^2 / C_k + log(C_k) ]
    + 0.5 ||s||^2 + scalar prior penalties,
C_k = P_epsilon(k) / V.
```

Handle the zero mode, Hermitian boundary planes and Nyquist exclusions explicitly. The baseline Poisson power is derived from the actual catalog number density and transform normalization. Never subtract shot noise from the observed density field. For weighted catalogs, derive the effective shot-noise normalization separately.

For the extended noise option, preserve the distinction between **noise standard deviation** and **noise power**: squaring a k-dependent amplitude changes the covariance. Register positivity/support over all retained modes and differentiate both the residual weighting and the log determinant.

## Implementation sequence and gates

### 0. Analysis contract and data audit

- Pin paper/configuration versions, catalog identifiers/checksums, coordinate origin, periodic wrapping, redshift, Fourier sign/normalization, spectrum/growth convention and sigma8 reporting epoch.
- Obtain the box-1 ICs and required non-amplitude background parameters, if available. If not, use an independently generated known-IC validation dataset for the fixed-IC milestone; label that as methodological validation rather than reproduction of their box-1 result.
- Record the initial filter, Eulerian operator filter and likelihood mask independently. Verify the challenge-specific intermediate filter prescription; do not import the companion's different cutoff ratios without checking.
- Freeze the prior table, including units of higher-derivative coefficients. Audit whether the reference spectrum and growth approximation match the simulation's neutrino/transfer convention.
- Stream catalog ingestion and compute density modes once, with delta_k=Ngal^-1 sum exp(-ik.x) for nonzero unweighted modes in our convention. Compare a small catalog to direct summation; compare power and counts with supplied diagnostics if available.

**Gate:** one manifest and a reproducible data vector. No large catalog download or contacts with authors are performed as part of this planning task.

### 1. Minimal Eulerian forward model and adjoint

- Factor reusable unbiased matter evolution out of the current LPT path while preserving existing Lagrangian-bias behavior as a separate model choice.
- Add grid-generic Eulerian operator evaluation and reverse operations through filtering, products, mean subtraction, tidal projections and matter deposition.
- Introduce explicit immutable filter/noise/parameter-layout descriptors. Preserve current default interfaces behind compatibility adapters.
- Begin with conservative resolved meshes. Test against direct particle Fourier sums and small direct convolutions, analytic single-/two-mode inputs, field/scalar directional derivatives and adjoint inner products.
- Run serial and slab/pencil 1/2/4-rank comparisons from this first component increment.

**Gate:** correct second-order forward/adjoint and baseline fixed-noise target on small meshes. This is the recommended first coding increment.

### 2. Resolution, padding and reusable NUFFT execution

- Separate IC sampling, LPT source/product, displaced-particle and final operator meshes. Existing half-Nyquist restrictions cannot just be removed.
- Implement normalized resampling and its adjoint, plus sufficiently padded polynomial products. The [numerical LEFTfield study](https://arxiv.org/html/2409.10937v2) motivates starting resolutions, but does not replace convergence checks on our implementation.
- Test particle-lattice density, NUFFT tolerance, intermediate filters and all relevant meshes independently at fixed physical cutoffs. Verify positions as well as density/parameter derivatives; a type-2 transform alone is not a position derivative.
- Cache FFT plans, assignment workspaces and kernels within each chain. Preserve bounded communication and slab/pencil ownership for every grid.

**Gate:** numerical changes are small in noise-weighted residuals and produce a predeclared subdominant shift in inferred amplitude on a fixed-IC test. Proposed numerical budget: less than 0.1 posterior SD in the amplitude mean, with Monte Carlo uncertainty accounted for. This is our acceptance target, not a result already achieved.

### 3. First scientific deliverable: fixed-IC one-box analysis

- Infer amplitude and the four baseline bias coefficients while holding the supplied IC realization fixed. Keep the reference Poisson noise fixed in this compatibility configuration.
- Cache operator columns whenever the evolved matter field is unchanged. Bias updates should use their small Gram matrix rather than recompute the forward model.
- Use known-parameter synthetic checks first, then the actual catalog. Inspect bias/amplitude degeneracies and stability under numerical refinement.
- Report a fully marginalized amplitude posterior; a coordinate-dependent joint MAP is an initializer/diagnostic, not the cosmological estimate.

**Deliverable:** amplitude/bias corner plot, traces, data/model/residual fields, residual spectra and a numerical-refinement plot for one box.

### 4. Model robustness while ICs are still fixed

- Add complete cubic Eulerian bias and scale-dependent noise as independent switches, retaining the compatibility configuration unchanged.
- Scan kmax and the three filter roles, separating numerical convergence from changes to the EFT model. Test wrong/simplified models as negative controls on independently generated higher-resolution/N-body tracer mocks.
- Document any informative noise prior and test sensitivity to it; do not silently impose a lower bound to cure poor sampling.
- Retain explicit bias sampling for the challenge's bounded uniform-prior configuration. Analytic Gaussian-prior marginalization is a separate option. A profiled best fit omits determinant/prior-volume terms and is not a substitute for marginalization.

**Gate:** choose and freeze the science configuration and scale-cut rule using validation data. The extension is not validated merely because its interval contains truth or becomes wider.

### 5. Scalable sampler, outputs and restart

- Replace the small dense metric with an FFT-diagonal or otherwise O(D)-memory preconditioner, tested against small exact-reference problems.
- Use field HMC with cached scalar updates. Compare whitened, physical and optional interweaved parameterizations under the same target and actual forward/adjoint cost before choosing a production kernel. Do not assume the PM pilot's cheapest amplitude conditional implies the fastest large-field mixing.
- Extend existing checkpoint machinery to the complete block state: field/scalars, HMC/adaptation state, scalar RNG streams, frozen metric and progress. Rebuild deterministic caches with identity checks.
- Support a wall-time limit and graceful checkpoint at a complete block/trajectory boundary. Test separate-process interruption/restart, including warmup, and reproduce the uninterrupted result at the supported same layout. Changed rank-count/backend restart remains a separate feature.
- Stream scalar diagnostics and predetermined field diagnostics at every retained draw; save full fields/predictive snapshots sparsely under an explicit output budget. Do not thin diagnostic sequences. At this scale it is impractical to copy the 122-coordinate pilot's full-storage strategy blindly.

**Gate:** restart demonstrated before multi-hour chains; no dense D-by-D object, repeated full-grid allocation per force evaluation, or unbounded full-field replication.

### 6. Free-IC one-box posterior

- Validate on smaller independently periodic synthetic volumes, then on low-cutoff pilots in the actual full challenge box. Cropping a periodic catalog does not produce a smaller periodic simulation.
- Increase resolution at fixed physical cutoffs when testing numerical convergence. A lower-cutoff pilot changes the statistical target and is not evidence for final-cutoff mixing.
- Benchmark one full-sized forward/adjoint pair and an entire HMC block before setting production resources. Roughly 0.7 million field degrees of freedom require a different resource plan from the 122-coordinate pilot.
- Start dispersed chains without access to the true field. Monitor amplitude/bias/noise marginals, energy/divergences, selected Fourier coefficients, band powers and fixed field projections. Predeclare production diagnostics; do not claim every field coordinate is validated from a few projections.
- Proposed science gate: rank Rhat<1.01 for the reported scalar parameters, bulk ESS>400, tail ESS>200, no unexplained retained divergences, and amplitude-mean Monte Carlo error below 0.05 posterior SD. Field-diagnostic criteria and run budgets must be frozen before the final run; no extension-until-pass policy.

**Deliverable:** one converged, checkpointable free-IC amplitude posterior with reconstruction and predictive plots, plus numerical/model-sensitivity evidence.

### 7. Ten-box application and final comparison

- Run each eligible box independently with the frozen configuration, preserving unsuccessful runs and declared budgets.
- Report per-box posteriors and scatter, and reproduce the reference's stated ensemble reporting convention. Averaging marginalized posteriors is not the same as multiplying likelihoods or pooling chains into one convergence calculation.
- Keep baseline/extended model comparisons visible. Larger nuisance freedom can change both central values and uncertainty; a narrow interval is not itself a scientific success.
- Any power-spectrum/bispectrum information comparison must match volume, Fourier modes, noise assumptions and nuisance priors. Treat that as a follow-up rather than a prerequisite for the first galaxy-field result.

**Final plots:** per-box amplitude intervals, bias/amplitude corner plots, scale-cut and operator-order stability, initial-field reconstruction/uncertainty, posterior predictive density/residuals and mode-dependent residual spectra. Distinguish same-model calibration from robustness to N-body/HOD mismatch. A ten-box recovery study is useful but not a precise 95% coverage measurement.

## Scale, resources and priorities

Our own dimensional estimate for L=2000 Mpc/h and a cubic Lambda=0.14 h/Mpc band is of order 7e5 independent real nonzero-mode degrees of freedom. The exact count depends on strict boundary/Nyquist conventions. A dense double-precision square matrix would require roughly 4 TB before factorization/workspace. It is ruled out for the production metric.

A minimally resolved latent grid is around 90--96 cells per side; product/evolution grids must be larger and independently converged. Applying the current half-Nyquist restriction by enlarging the *latent* grid alone would waste most sampled dimensions. Separate compact latent storage from padded numerical workspaces. Fixed-basis linear dependence tests, buffer lifetime accounting and cheap scalar updates are higher priorities than adding gravity orders initially.

This is a multi-week development/validation project, not a small follow-on to the laptop PM pilot. Production runtime and hardware requests should be based on measured final-cutoff forward/gradient/block timings and mixing, not on extrapolating the toy wall time. Parallelize independent chains/boxes as well as distributed FFTs once memory/runtime measurements support it.

The immediate order is **contract/data audit -> differentiable Eulerian model -> fixed-IC result and convergence -> robust model/sampler/restart -> free ICs -> ensemble**. RSD, lightcones, survey selection and full cosmological-shape inference remain later applications. The new evolution/tracer/filter/noise/sampler components should be reusable for them.

## Decisions still requiring project information

1. Exact catalog/IC/spectrum files and permission/access to any non-public validation products.
2. Whether an external LEFTfield forward-output comparison is available; it is valuable but not assumed.
3. Available cluster/node memory and intended production wall-time budget, to be set after profiling.
4. Whether the first publication-level target is challenge compatibility, a more flexible amplitude analysis, or a later full-cosmology extension. Recommendation: deliver compatibility first, but assess precision with the robustness configuration.

The first operator component is now implemented (status below). No galaxy-catalog sampling, catalog download or external message has been made. The preceding folder cleanup was committed separately.

## Initial implementation status (17 November 2026)

- Done: periodic catalog Git-tree audit pinned in `leftfield/catalog-manifest.json`; no full catalog downloaded.
- Done: grid-generic Eulerian density, centered density squared, centered tidal squared and physical Laplacian, with reverse derivatives in `EulerianBias.h`.
- Done: explicit spherical/cubic matter-band geometry and conservative product-resolution guard. These bands do not yet replace framework mode enumeration.
- Done: analytic field and per-column derivative tests, serial and slab/pencil 1/2/4-rank validation.
- Done: [composed Eulerian LPT model](EULERIAN_LPT.md), amplitude-only spectrum, fixed-noise target with explicit scalar layout and conditional/free-field prior policies. Small Gaussian tracer-field mocks validate the full field/scalar reverse path; they are not discrete galaxy mocks or posterior calibration.
- Done: reusable exact linear-bias likelihood cache and amplitude/bias sampler, independent quadrature/MPI tests, and a [three-mesh fixed-IC pilot](validation/studies/galaxy_fixed_ic/RESULTS.md). Its bulk ESS fails the frozen gate; numerical mean-shift uncertainty does not yet certify the 0.1 SD refinement budget.
- Done: [joint amplitude/bias proposal and separate campaign](validation/studies/galaxy_fixed_ic_joint/RESULTS.md), preserving the original failed pilot. All three small configurations pass scalar mixing and their bounded refinement check, with 1.75–2.40 times the alpha ESS per forward call.
- Done: [streaming periodic catalog/NUFFT observation adapter](PERIODIC_CATALOG.md), weighted/unweighted density and Poisson normalization, internal owner routing, same-layout model mode export, and CSV/HDF5 file conversion. Small direct-sum and slab/pencil MPI verification is recorded separately.
- Done: [PSWF NUFFT matter deposition and matched position adjoint](NUFFT_MATTER.md), with independent particle derivative tests and complete 1LPT/2LPT field/scalar checks on slabs and pencils. The smooth assignment remains the default; stencil-boundary approximation and production tolerance convergence are explicit.
- Done: [compact latent and padded product meshes](COMPACT_MESHES.md), with isometric white-field transfer and its Euclidean transpose, compact observation mode indices and short MPI HMC checks against a fixed padded reference. This reduces sampler field coordinates; total RSS and production mixing are not measured.
- Done: [Eulerian HMC checkpoint/restart adapter](EULERIAN_RESTART.md), fresh-process exact joint/field-only continuation and indexed draw output replay on small slab/pencil fixtures. Existing checkpoint formats retain compatibility.
- Done: [bounded synthetic numerical/resource screen](NUMERICAL_SCREEN.md) at frozen physical cutoffs, with all 17 serial/MPI cases passing their declared screen gates. This does not establish posterior-mean convergence or real-catalog fidelity.
- Done: [FL-7 posterior-aware field metric](EULERIAN_METRIC.md), independent linear-limit/compact/MPI/restart checks and a fixed-budget old/new field-only probe with improved projection diagnostics. Its maximum corrected Rhat=1.023 still needs longer mixing validation.
- Next: declare the larger joint field/scalar pilot and its fixed budgets/gates. Exact catalog IC/spectrum inputs and initial cubic filtering remain open; the real-data and free-IC scientific milestones are not complete.
- Open: exact challenge IC/spectrum/cosmology inputs and intermediate-filter/prior-unit details. Fixed-IC synthetic validation can proceed independently.

The milestones above remain gates. The small forward/adjoint and fixed-noise target checks are implemented. The improved synthetic fixed-IC pilot passes its bounded scalar-mixing/refinement checks. The catalog-to-mode data path is now implemented on small fixtures; application to the actual catalog, catalog-specific numerical/model validation and the real-data/free-IC deliverables remain open.
