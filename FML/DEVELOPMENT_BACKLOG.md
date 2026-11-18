# FML remaining-work backlog

Updated 14 November 2026 (items 25–26, 28, 32, 34–35, 50–52 and 55), including completed Engineering items 1–7, 10, 14, 17 and 19–21, and tested initial implementations of items 25–32, 34–38, 49 and 54–55. This consolidates the original review and subsequent work in this chat. The user intends to work through the full backlog over time. These initial implementations and their remaining production-validation work are recorded below; KD-tree work remains deferred. Effort estimates are relative; performance gains need measurements. Remaining new science entries are proposals from the original review and need a focused capability check before implementation.

## Completed in this sequence

- Pencil decomposition throughout (item 43, the pencil part): the generic grid interface, `PencilGrid` and the complex grids `CFFTWGrid` / `CPencilGrid`, the library, SimulationCode, SnapshotAnalyzer and the distributed FieldLevel models on pencils (`make USE_PENCIL_GRID=true`). See `PencilDomain/README.md`.
- Domain-filtered binary Gadget reservation.
- Streamed baryonification application.
- Streamed PGD force interpolation/application.
- NUFFT count-safe chunked redistribution.
- Parallel CosmicWeb loops and optional byte classification labels.
- Sparse canonical polyspectrum result storage.
- Pruning inactive non-Gaussian random-field kernels and transforms.
- Reusable FFTLog plans, aligned buffers, and coefficient caching.
- Multi-slab spherical-void particle exchange, without the slab-width-triggered all-particle fallback.
- Multi-slab spherical-overdensity measurements with serialized selective exchange and adaptive centering coverage.
- Explicit multigrid convergence, iteration-exhaustion and nonfinite status, with final residual and V-cycle count.
- Deterministic min-heap pair-count rank assignment, preserving rank ties and each rank's cell order.
- Neighbor-aware pair-count candidate-work estimates, including asymmetric cross-catalogues and auto-pair ownership.
- MPI count audit: widened migration counters, checked collective layouts/catalogues and serialization products, bounded reductions and forced-chunk regressions.
- Faster particle migration with direct neighbor/binary slab lookup, collective no-traffic/neighbor routing and a preserved general fallback.
- Reusable Fourier-resampling runs with values-only bounded messages, precomputed peer rounds and shared 3LPT crop routing.
- Hessian and eigensystem output allocation without redundant owned-field copies, preserving halo/padding contents.
- Polyspectrum evaluation-storage component reports, checked cross-rank totals and an explicit opaque-FFT workspace allowance.
- Reproducible FFT/copy/clone/resampling/migration/polyspectrum performance cases with scoped counters, OS peak RSS and machine-readable provenance.
- Rank-normalized split/folded Rhat and bulk/tail ESS with standalone diagnostics, explicit invalid-chain handling and shared Metropolis/HMC adapters.
- One capability matrix and a shared C++17 build/API guide with linked regression examples.
- A serial Python analysis package with owned 3D catalogues, kNN/CDF and spherical-count summaries, NumPy-owned results and explicit exception behavior.
- Survey Fourier multipoles (0, 2, 4), weighted noise, physical geometry and MPI catalogue routing.
- Survey window multipoles (0–8), model mixing matrices, output-band averaging and global IC modeling.
- Local/fixed-LOS survey displacement solving and RecIso/RecSym data/random reconstruction.
- HOD central/satellite mocks with halo-ID-keyed Philox streams, NFW/particle satellites and velocity/concentration controls.
- Batched kNN distances and streamed CDFs with explicit self-exclusion and distributed query samples.
- Counts/density PDFs, factorial moments and explicit Poisson self-counting summaries for unit-weight tracers.
- Initial flat-GR Born source-bin convergence, periodic flat-sky shear, Limber spectra and projected galaxy–matter profiles with explicit units and domain checks.
- Seeded equal-mass glass loads with repulsive softened P3M relaxation, residual status, versioned sharded IO and irregular-position LPT/transfer-provider adapters.
- Initial periodic complex-field Schrödinger–Poisson evolution, prescribed-expansion factors, energy diagnostics, source/force hooks and repartitionable checksummed checkpoints.

- Generic-order growing-mode EdS LPT with transverse fields, conservative bandwidth/resource planning, collected/streamed outputs and small-field reference validation through order 6.

These completions do not cover all the follow-ups below. Other ongoing SimulationCode/transfer-function edits are outside this backlog update.

## Survey workflow status

Items 25–27 have initial implementations in `Survey/SurveyPower.h`, `Survey/SurveyWindow.h` and `Survey/SurveyReconstruction.h`. The standalone example and tests are in `Survey/workflow_example/` and `Survey/tests/`. Read `Survey/README.md` for conventions and limits. Independent DFT, explicit random-pair, analytic Gaussian convolution and manufactured displacement tests pass in serial, float/OpenMP, sanitizer and MPI builds. Item 25 now also has an external pypower raw-multipole comparison with matched normalization/CIC conventions, an independent unpainted-catalogue DFT mesh-convergence experiment and pre-FFT MPI option agreement checks; see `Survey/power_validation/README.md` and `RESULTS.md`. Item 26 now also has every-column independent angular/Bessel matrix validation, a pypower smooth-model comparison, global-IC checks and continuous anisotropic Gaussian mesh/padding/binning/support experiments; the latter exposed and fixed radial quadrature crossing Q-interpolation knots. See `Survey/window_validation/README.md` and `RESULTS.md`. Small-scale high-order lattice errors remain visible and require target-mask convergence checks. Production mock recovery, external pyrecon comparisons and application-specific mask/padding/binning/noise calibration remain follow-up validation work.

## Control-variate ensemble status

Item 32 now has `EnsembleStatistics::ControlMean`, which attaches phase provenance directly to streaming independent cheap rows and carries it through MPI reductions. Its `estimate()` returns the existing uncertainty-aware `MeanEstimate`; the low-level moments-plus-keys API remains compatible. `EnsembleStatistics/matched_validation/` connects Philox phases, existing 2LPT initial conditions, ZA/2LPT controls, EdS particle-mesh evolution and matched raw spectra to held-out diagonal/full fits. NumPy independently checks coefficients, corrected means, residual covariance and the shared cheap-mean uncertainty floor. A separate 2048-experiment Gaussian test measures mean bias, covariance calibration and approximate interval coverage. The coarse PM integration shows mean-variance reduction at two output times but is not mesh-converged; no production speedup or precision N-body calibration is claimed. Read its `README.md` and `RESULTS.md` for measured costs, conditioning, numerical checks and remaining work. Converged cosmological/COLA calibration and specialized target-covariance control variates remain extensions.

## HOD and spatial-statistics status

Items 28–30 have initial implementations in `HOD/HOD.h` and `CountsInCells/SpatialStatistics.h`. Tests and the HOD-to-statistics example are under those module folders. Both build without FFTW/GSL. HOD supports conditional/independent central-satellite occupations, prepared halo/member data, NFW or particle satellites and stable identity pairs. Spatial summaries use the existing KD-tree with replicated ordered tracers and distributed query samples; they do not implement the deferred KD-tree optimizations. Unit-weight counts, rectangular nonperiodic full-cell windows and periodic spherical cells are supported. Item 28 now also has explicit marginal satellite means and a Heaviside decorated-HOD assembly-bias prescription conserving population central/satellite first moments at fixed mass when supplied percentiles have the declared conditional population fractions. Conditional compatibility, independent zero-central satellite models, independent Decimal and original Halotools pointwise-kernel comparisons, and rank/thread replay are documented in `HOD/ASSEMBLY_RESULTS.md`. Calibrated occupation fits, empirical conditional-percentile calibration, cosmological assembly-bias clustering/lensing validation, weighted estimators, arbitrary-mask boundary corrections and distributed tracer searches remain extensions. See each module README for the exact conventions.

## Engineering work, suggested order

Items 1–7, 10, 14, 17 and 19–21 are complete. Items 8 and 12 are waiting at the user's request. Other pending Engineering items are 9, 11, 13, 15–16 and 18. See `FriendsOfFriends/SphericalOverdensity.md` for the multi-slab exchange contract and regression coverage, and `MultigridSolver/TerminationStatus.md` for solve-result semantics. See `PairCounting/RankAssignment.md` for deterministic scheduling and neighbor-aware workload estimates, and `Global/MPICounts.md` for the count-audit scope and limits. See `MPIParticles/Migration.md` for automatic migration routing and its general fallback. See `FFTWGrid/FourierResampling.md` for reusable Fourier-routing geometry and bounded payload buffers. See `CAPABILITIES.md`, `BUILDING.md` and `PythonAnalysis/README.md` for current capabilities and the selected Python analysis API.

| # | Task | Concrete scope | Effort |
|---|---|---|---|
| 1 | ~~Multi-slab spherical-overdensity measurements~~ (completed) | Removed the slab-width restriction; preserved particle serialization, periodic/unwrapped distances, centering, velocities, masses and truncation flags. | Medium |
| 2 | ~~Explicit multigrid termination status~~ (completed) | Distinguished convergence, maximum iterations and nonfinite failure; returned final residual and iteration count. | Small–medium |
| 3 | ~~Faster pair-count rank assignment~~ (completed) | Replaced repeated rank-load scans with a deterministic min-heap, retaining tie-breaking behavior. | Small |
| 4 | ~~Better pair-count workload estimates~~ (completed) | Used neighboring candidate occupancy rather than only same-cell occupancy, including cross-catalogues and auto-pair ownership. | Medium |
| 5 | ~~Finish the MPI count audit~~ (completed) | Widened/checked migration counters; validated sheet and catalogue layouts before narrowing; checked serialization products and bounded array reductions. | Medium |
| 6 | ~~Faster particle migration~~ (completed) | Added neighbor/binary owner lookup and collective no-traffic/neighbor routes, retaining general routing for long jumps. | Medium–large |
| 7 | ~~Reduce Fourier-resampling communication overhead~~ (completed) | Added reusable contiguous runs, values-only bounded buffers and precomputed peer rounds; reuse the crop plan across 3LPT fields. | Medium–large |
| 8 | Distributed pair-count catalogue backend | Owned particles plus ghosts, explicit auto/cross-pair ownership, and no double counting; retain replicated APIs for small catalogues. | Large |
| 9 | Stream binary Gadget field blocks | Bound the keep mask and field-block storage; preserve endian, mixed-species, format-1/2 and MASS-block behavior. | Medium–large |
| 10 | ~~Remove redundant Hessian/tensor copies~~ (completed) | Shared empty-layout allocation preserves halos/padding while Hessian/eigensystem outputs avoid full overwritten-field copies. | Medium |
| 11 | Flat generic cell storage | Offset/index-based replacement or alternate API for vector-per-cell ParticlesInBoxes; preserve compatibility through views. | Medium–large |
| 12 | Improve polyspectrum shell reuse | Deterministic, cache-aware configuration ordering to reduce shell rebuilds under a bounded cache. | Medium |
| 13 | Tile polyspectrum integration | Accumulate small compatible groups of configurations over spatial tiles; validate changed floating-point summation. | Medium–large |
| 14 | ~~Report total workspace memory~~ (completed) | Reported input/results/metadata, shells, padding and remapping capacities plus explicit FFT allowance; kept cache budgets distinct from evaluation estimates and RSS. | Medium |
| 15 | Extend explicit workspace reuse | Apply consistent reuse/release conventions to LPT, PGD, NUFFT, Hessians and polyspectra, building on FFTLog. | Incremental |
| 16 | Stream MCMC samples and moments | Batched output, online means/variances/covariances, optional sample retention, checkpoint/resume and rejected-step multiplicities. Preserve exact-quantile requirements. | Medium–large |
| 17 | ~~Improve MCMC convergence diagnostics~~ (completed) | Added standalone rank-normalized split/folded Rhat and bulk/tail ESS, sampler/HMC adapters, explicit degeneracy statuses and independent ArviZ reference tests; legacy stopping remains distinct from modern post-run diagnostics. | Medium |
| 18 | Consolidate MPI transport contracts | Reuse checked chunking/routing across modules; standardize collective failures, instrument bytes, and reduce per-peer send staging where measurements justify it. | Incremental |
| 19 | ~~Add reproducible performance cases~~ (completed) | Added six validated cases with opt-in FFT/grid/transport counters, fresh-process RSS/rank timing reports and source/build/binary/configuration provenance. Extend workloads as production needs arise. | Incremental |
| 20 | ~~Update capability documentation~~ (completed) | Added the capability matrix, shared C++17 build/API guide and updated root entry points with regression/example links. | Small–medium |
| 21 | ~~Stabilize a small Python analysis interface~~ (completed) | Added serial owned Catalogue/kNN/CDF/count-PDF workflows with explicit array/exception contracts, wheel/source packaging and installed-package tests. | Medium–large |

## Deferred at the user's request

| # | Task | Scope |
|---|---|---|
| 22 | Reusable KD-tree query workspaces | Per-thread traversal/result buffers, array-based queries, visitor APIs and stable-order baryonification integration. |
| 23 | Support-radius-aware halo searches | Store maximum halo support per tree node and prune irrelevant halo spheres without changing periodic or summation semantics. |
| 24 | Lower KD-tree construction peak memory | Reusable build scratch or in-place permutation instead of simultaneous coordinate/index/permutation copies. |

Items 22–23 were explicitly deferred. Item 24 is grouped here as related KD-tree work; revisit the group only when the user wants to resume it. Science using the existing KD-tree need not wait for these optimizations.

## New LSS capabilities, suggested science order

| # | Capability | First useful scope |
|---|---|---|
| 25 | Survey Fourier multipoles (implemented; external raw-estimator validation added) | FKP data/random fields, fixed/local-LOS 0/2/4 multipoles, normalization and declared weighted/CIC noise. pypower raw-multipole agreement, independent catalogue-DFT mesh refinement and MPI schema checks pass; target-survey production/noise/padding calibration remains. |
| 26 | Survey-window modeling (implemented; independent validation added) | Window multipoles/mixing matrices, global IC, every-column quadrature and external pypower smooth-model checks; Gaussian mesh/padding/binning/support convergence and MPI option agreement. Target-mask calibration and small-scale high-ell convergence remain; assignment-window deconvolution stays distinct. |
| 27 | Complete survey BAO reconstruction | Data/random displacement APIs, RecIso/RecSym, local LOS, masks/selection and reconstructed statistics. |
| 28 | HOD galaxy mocks (implemented; decorated assembly-bias extension added) | Central/satellite occupations, NFW/particle phase space, ID-keyed randomness and explicit marginal-mean Heaviside assembly bias. Independent pointwise/moment checks and MPI/OpenMP replay pass; occupation fitting, empirical percentile and cosmological clustering calibration remain. |
| 29 | kNN-CDF statistics | Batched random query points, distributions of kth-neighbor distances and explicit self-exclusion conventions. |
| 30 | Counts-in-cells | Counts/density PDFs and factorial moments, with explicit window and shot-noise conventions. |
| 31 | Density-split and marked clustering (initial implementation) | Environment quantiles and cross-correlations; normalized marks and well-defined noise treatment. |
| 32 | Control variates and ensemble summaries (implemented; matched-field validation added) | Phase-tracked streaming cheap means, frozen scalar/vector fits, cross spectra and covariance; matched ZA/2LPT versus coarse PM integration, independent NumPy checks and repeated Gaussian coverage. Converged cosmological/COLA calibration and cost at fixed mean error remain. |
| 33 | 3PCF and anisotropic higher-order estimators (initial isotropic implementation) | Periodic spherical-harmonic 3PCF multipoles with distinct-triplet data/random normalization; anisotropic statistics and bispectrum multipoles remain. |
| 34 | Connected four-point statistics (initial periodic implementation) | Shell-filtered joint fourth moments with explicit Wick subtraction, dealiased quartet normalization and optional external covariance; independent-box connected power covariance and volume-scaled collapsed trispectra. General shape/diagonal-resolved and survey estimators remain extensions. |
| 35 | Faster perturbative predictions (initial implementation and extensions) | Checked FFTLog/direct one-loop terms, IR-resummed LPT/EPT, TNS and CLASS-PT-style models; shared reusable correlation-function multipoles xi_0/2/4 with explicit finite-band filters. Broader cosmological/UV/IR calibration remains. |
| 36 | Morphology statistics (initial implementation) | Excursion-set volume, surface area, curvature and Euler characteristic, with explicit voxel/boundary conventions. |
| 37 | Integrated lensing observables (initial implementation) | Added normalized source-bin kernels, streamed finite-shell Born convergence, MPI-capable periodic flat-sky shear, strict-domain Limber Ckk/Cgk and finite-Pi galaxy–matter/DeltaSigma profiles. Full-sky, mask and production/reference-code calibration remain follow-ups. |
| 38 | Halo histories and external substructure interfaces (initial implementation) | Stable particle-ID overlap merger trees and external subhalo catalogue integration. |

The scientific implementations should be checked against independent analytic or established reference results, as well as serial/MPI consistency. Existing periodic spectra, reconstruction building blocks, raw shell fourth moments, halo models and lensing/lightcone utilities should remain the starting point.

## Conditional work: measure first

| # | Task | Condition for proceeding |
|---|---|---|
| 39 | SIMD/SoA pair-count specialization | Standard weighted-count hot loops show a worthwhile bottleneck; preserve the generic callback path. |
| 40 | Multigrid coarse-level rank agglomeration (deferred) | Revisit when representative production profiles show coarse levels materially limit total solve time. The local experiment's ~12% full-solve gain did not justify its complexity; the slice-count heuristic was slower. |
| 41 | Overlap multigrid interior work and halos | Profiles justify it; preserve color-sweep communication dependencies. |
| 42 | Scoped mixed precision | Memory pressure justifies float field storage with double accumulation; validate forces, displacements, higher moments, thresholds and MG residuals. |
| 43 | GPU FFT backend (the pencil decomposition is done) | Production runs are FFT limited on GPU nodes. Pencils (`PencilGrid`, `CPencilGrid`) with deposition, particles, ghosts and indexing are in place for the library, SimulationCode and SnapshotAnalyzer; distributed FieldLevel models now also support pencils (the original serial BAO prototype stays separate). |
| 44 | DTFE output/boundary memory improvements | Profile requested-output allocation and boundary traffic; consider batched boundary fields. |
| 45 | Broader distributed snapshot IO | Benchmark binary/HDF5/Ramses read patterns and peak memory first, then choose collective or chunked improvements. |

Items 25–30 now have tested initial implementations. Next options are production validation of these workflows, resuming the engineering queue with item 1, or extending science item 33 beyond its initial isotropic 3PCF implementation. Items 2–5 are good subsequent bounded changes. The larger observational workflow is 25–27; new mock/statistics work can follow with 28–32.

Items 32 and 36 now also have tested initial implementations. `EnsembleStatistics/` provides streaming covariance, independently trained full/diagonal control variates, independent surrogate-mean uncertainty, phase provenance and a matched Fourier auto/cross-spectrum adapter. `Morphology/` provides distributed closed-voxel excursion volume, surface area, integrated mean curvature and Euler characteristic, with open/periodic boundaries and separate Crofton normalization. Both include runnable examples and analytic/independent-reference serial/MPI tests. Production follow-ups are matched ZA/2LPT/COLA versus N-body calibration and continuum Gaussian-field morphology/resolution validation; see each module's README. Item 33 now has the initial isotropic scope described below; item 34 now has the initial periodic implementation described below. Item 37 now has the initial implementation described below.

Item 31 now has an initial periodic-box implementation in `Environment/`: top-hat counts/overdensities, deterministic equal-count density quantiles and query–tracer radial cross-correlations, plus one-pass normalized marked correlations with explicit sample-mean and finite-population pair conventions. It reuses the existing KD-tree/counts-in-cells and pair-counting engines and includes direct-enumeration, analytic, serial/MPI tests and a runnable example. Density noise and finite query sampling are retained explicitly; survey masks/weights, angular estimators and production noise/resolution validation remain follow-ups. See `Environment/README.md`.

Items 35 and 38 now have tested initial implementations. `PerturbationTheory/` provides EdS matter P22/P13 using FFTLog/FFT convolution with mandatory direct-reference spot checks over an explicit output range, plus independent direct quadrature for custom scale-dependent kernels. `HaloHistory/` provides sparse particle-ID overlap candidates, deterministic merger forests, optional bounded snapshot gaps, host/subhalo relationships, a versioned neutral catalogue format and explicit HOD adaptation. Both include examples, reference tests and documented limits. Expanded/IR-resummed bias/RSD templates, broader cosmological validation, native finder-specific readers, distributed halo matching and advanced core/trajectory-based tree repairs remain follow-ups. Item 35 also has a first biased-tracer RSD template: `PerturbationTheory/LPTRedshiftSpace.h` (one-loop LPT in redshift space with resummed displacements, ported from velocileptors and validated against it and against direct integrations), and an IR-resummed one: `PerturbationTheory/EPTRedshiftSpace.h` (velocileptors' REPT, one-loop EPT with a wiggle/no-wiggle split; validated against velocileptors and the direct one-loop SPT), with the no-wiggle methods in `PerturbationTheory/NoWiggle.h`.

Item 35 now also includes `PerturbationTheory/CorrelationMultipoles.h`: a local reusable FFTLog workspace converting existing LPT/EPT, TNS and EFTofLSS multipoles to xi_0/2/4 on a common separation grid, with the correct Fourier phases, signed interpolation, explicit finite k support, endpoint taper and optional Gaussian damping. It retains supplied stochastic terms with the declared filter, so clustering-only predictions must set the noise parameters to zero explicitly. Analytic/direct quadrature, model-adapter, refinement, unit, thread and local-MPI checks are documented in `PerturbationTheory/CORRELATION_RESULTS.md`; the runnable example is `example/Correlation.cpp`. Infinite-band/contact prescriptions, target-analysis support/filter convergence, shell/window averaging and broader cosmological calibration remain separate.

## Three-point statistics status

Item 33 has an initial periodic isotropic implementation in `ThreePointCorrelation/`: KD-tree spherical-harmonic neighbor accumulations through ell=16, all eight ordered data/random moments with explicit center/neighbor exclusions, factorial catalogue normalization and an analytic periodic-shell denominator. MPI partitions primary work over replicated catalogues and reduces only moments. Tests compare against independent explicit triplet enumeration, analytic triangles, self-product/boundary cases and independent uniform mocks; an example and serial/MPI build targets are included. See [ThreePointCorrelation/README.md](ThreePointCorrelation/README.md) and its validation record. Fixed/local-LOS anisotropic estimators, bispectrum multipoles, weighted/survey edge corrections, distributed-catalogue memory, OpenMP/load balancing and cosmological/reference-code calibration remain follow-ups. This does not complete the full anisotropic scope of #33; the separate connected four-point item 34 is described below.

## Connected four-point statistics status

Item 34 now has an initial periodic implementation in `ConnectedFourPoint/`: joint known-zero-mean fourth moments, all three Wick contractions with optional external covariance, dealiased shell-integrated and ordered-quartet-normalized connected moments, and an independent-box power-covariance adapter returning `V * (Cov - Cov_G)`. The same-field Wick statistic is explicitly a biased finite-mode plug-in; an exact Gaussian Fourier-ensemble test demonstrates both its bias and the external-reference null. Tests use independent direct DFTs and explicit signed-wavevector pair/quartet enumeration, analytic non-Gaussian/Gaussian moments, physical-volume scaling and collective failure cases. See `ConnectedFourPoint/README.md`, `tests/RESULTS.md` and `TODO.md` for validation and limits. No change is made to the legacy raw fourth-moment estimator. Production cosmological/reference calibration, connected particle-noise corrections, reference uncertainty, weighted/survey estimators and shape/diagonal-resolved trispectra remain follow-ups.

## Weak-lensing workflow status

Item 37 has an initial implementation in `WeakLensing/`: normalized source planes or sampled n(z)/n(chi), precomputed finite-shell Born weights and streamed local convergence maps; an optional `FFTWGrid<2>` periodic flat-sky shear adapter; extended-Limber auto/cross projections using explicitly bounded Pmm/Pgm providers; and finite-Pi projected galaxy–matter profiles, comoving excess surface density and thin-lens critical-density conversions. See [WeakLensing/README.md](WeakLensing/README.md) for units, input ownership, Nyquist/mass-sheet treatment and numerical tests. This does not change the existing SimulationCode onion-map or CMB-remapping pipelines. Full-sky/curved geometry, masks, non-Limber and observational corrections, realistic background/power/reference-code comparisons and application-specific radial/source/power convergence remain follow-up work. The runnable example is a labeled analytic toy, not a calibrated cosmological forecast.

## Deferred multigrid rank agglomeration

Item 40 was deferred at the user's request on 5 November 2026. The experimental periodic bottom-level backend improved a local 64³/eight-rank example from 146 ms to 129 ms with one coarse rank, while the automatic two-rank slice heuristic slowed it to 153 ms. The bottom represented roughly one-fifth of total runtime, so its communication-dominated profile alone was not sufficient justification for the added transport, communicator and duplicate-operator maintenance. The uncommitted implementation and measurements were archived separately and removed from the checkout; a profiling-only patch was retained separately as well. No agglomeration backend is shipped. Reconsider after representative production measurements establish a substantial end-to-end opportunity, rather than merely a high communication fraction within a small coarse level.

## Short-range gravity extension

| # | Task | Concrete scope | Effort |
|---|---|---|---|
| 46 | Short-range gravity library: P3M and octree TreePM | New `ShortRangeGravity/` module with a consistent Gaussian long-/short-range force split, cosmological normalization, compact gravitational softening and tolerance-controlled cutoff. Implement a direct-sum reference and periodic cell-list P3M backend; validate combined PM/pair forces against independent periodic direct/Ewald results across separations, grid orientations, softening, split scales and cutoffs. Add OpenMP target loops and MPI read-only ghosts spanning any number of slabs, preserving ownership and avoiding periodic duplicates. Then add a dedicated gravity octree with mass moments, conservative opening and cutoff-aware node traversal, validated against P3M. Begin with Newtonian gravity and nonrelativistic massive particles; measure force errors, pair/tree work, ghost memory and rank scaling. | Large |
| 47 | SimulationCode short-range gravity integration and subcycling (initial common-timestep N-body/COLA implementation) | Depends on item 46. Assemble interpolated long-range PM and short-range particle forces before the existing kick and COLA frame correction; add the short-range force once after PM interlacing and retain the appropriate lightcone potential calculation. Validate standard N-body mode first, then COLA, initially using common timesteps. Add symmetric short-range subcycling between long-range kicks, deriving the cosmological kick/drift factors and COLA frame acceleration throughout the inner steps. Test timestep convergence, halo/orbit dynamics and serial/MPI agreement; define separate prescriptions before enabling screened gravity or particle neutrinos. | Large |

## Phase-space halo finding

| # | Task | Concrete scope | Effort |
|---|---|---|---|
| 48 | Phase-space halo/subhalo finder (ROCKSTAR-style) | Start with spatial FoF parent groups, then adaptive hierarchical refinement in six-dimensional position–velocity space to identify halo/subhalo seeds and assign particle membership. Specify phase-space scaling, peculiar-velocity/Hubble-flow conventions, periodic boundaries and mass definitions explicitly. Add gravitational unbinding, robust centers/bulk velocities, bound masses, Vmax and host/subhalo relationships, with adapters for the existing SO/HOD catalogue APIs. Design MPI ownership and boundary exchange without full particle replication. Validate controlled overlapping halos, major mergers, tidal stripping, resolution and rank/thread consistency against independent references and [ROCKSTAR](https://arxiv.org/abs/1110.4372). Deliver a single-snapshot finder first; add optional temporal consistency through the halo-history work in item 38. Reuse short-range gravity primitives from item 46 where appropriate for unbinding. | Very large |

Item 46 now has an initial implementation in `ShortRangeGravity/`: Gaussian/spline force kernels, a long-range PM filter and explicit FML normalization, direct and cell-list backends, a conservative monopole octree, and chunked multi-slab MPI ghosts. See `ShortRangeGravity/README.md` for contracts and validation. This is a force-library implementation; production mesh/interpolation convergence, clustered MPI performance and tree error-distribution validation remain follow-ups. Item 47 now has the initial common-timestep milestone described below; its COLA, subcycling and production-calibration scope remains open.

## SimulationCode short-range integration status

Item 47 has an initial opt-in common-timestep implementation: periodic 3D unscreened gravity with one cold species, continuous Fourier kernels and Quinn N-body evolution. GravityModel supplies generic large-finite-k limits and analytic Yukawa overrides, with a matched long filter and a resolved remainder for arbitrary Geff(k,a); the short acceleration is added once after PM/interlacing velocity scaling and before drift. Physical comoving lengths, global source-mass normalization and the existing epoch-dependent kick convention are explicit. Selective slab ghosts support cell and tree backends. Lightcone output retains its unsplit PM potential estimate. Independent off-grid Ewald/mesh tests, isolated real-step checks, full-run timestep convergence, exact tree/cell and MPI agreement, and scope rejection tests are documented in [SimulationCode/SHORT_RANGE.md](SimulationCode/SHORT_RANGE.md) and [its results](SimulationCode/SHORT_RANGE_RESULTS.md). Broader nonlinear COLA calibration, short-range subcycling, multi-species/screened prescriptions, matched particle-pair lightcone potentials and production halo/orbit/power/memory calibration remain. The milestone does not complete all of #47.

Clustered static-force validation for item 47 now has an optional 2,048-source/128-target experiment with independently softened periodic Ewald references, concentrated/overlapping/boundary/translated halo fixtures, mesh/split/softening/interlacing/tree sweeps and serial/MPI checks. It measures force distributions, radial errors and instantaneous momentum imbalance; angle 0.6 exercises real tree approximation. See [SimulationCode/CLUSTERED_FORCE.md](SimulationCode/CLUSTERED_FORCE.md). Controlled small-N orbit/time-integration validation is now also covered below; this does not complete COLA, subcycling or production calibration.

Item 47 also has controlled near-circular/eccentric pair and 16-particle cluster/merger evolution checks through the actual SimulationCode step, compared to independently converged direct Ewald/RKF45 trajectories. They verify synchronized velocity epochs, second-order Quinn refinement, canonical momentum, interlacing, exact tree/cell evolution and serial/MPI ownership changes. The eccentric fixture requires a much finer timestep than the smoother cases. See [SimulationCode/DYNAMICS.md](SimulationCode/DYNAMICS.md). Production halo/finite-opening dynamics, short-range subcycling and adaptive timestep control remain open; common-timestep COLA support is recorded below.

Common-timestep short-range forces are now enabled with the existing 1LPT/2LPT COLA frames, including scale-dependent frames, using Quinn or Tassev. The existing force/frame ordering required no formula change; the blanket COLA rejection was replaced with tested scope checks. Manufactured nonzero frames, restored physical initial/final states, timestep refinement, exact backends, table/interlacing variants, lightcone transparency and serial/MPI runs are documented in [SimulationCode/COLA_SHORT_RANGE.md](SimulationCode/COLA_SHORT_RANGE.md). Ordinary N-body remains Quinn-only with short forces; enforced LPT, screening, massive/separate species and unsupported kernels/methods remain excluded. Short-range subcycling is the next implementation milestone.

## Wave dark matter and gas dynamics

| # | Task | Concrete scope | Effort |
|---|---|---|---|
| 49 | Spectral solver for ultralight axion / fuzzy dark matter (initial implementation) | Nonrelativistic Schrödinger–Poisson evolution with full complex wavefunction storage and complex-to-complex FFTs, symmetric split-step kinetic/potential evolution and a periodic Poisson solve. Specify axion mass, wavefunction/density normalization, units and zero-mode conventions. Start in a non-expanding periodic box; then derive cosmological variables/time factors and initial conditions, and add optional CDM/particle-gravity coupling. Reuse FFT workspaces, support MPI complex-field layouts and versioned checkpoints, and measure memory/FFT costs. Validate free-wave dispersion, interference, norm conservation, static energy, a stationary soliton reference and timestep/de Broglie-resolution convergence; derive an appropriate cosmological energy diagnostic. Treat self-interactions and isolated boundaries as later explicit extensions. Reference: [PyUltraLight](https://arxiv.org/abs/1807.04037). | Very large |
| 50 | Smoothed particle hydrodynamics (SPH; initial implementation) | A conservative gas-particle formulation with kernel density estimates, adaptive smoothing lengths and consistent grad-h terms, pressure forces, energy/entropy evolution and an EOS interface. Add controlled artificial viscosity/conductivity, shock/shear switches and Courant/acceleration timestep criteria. Begin with adiabatic ideal gas. Design symmetric variable-support neighbor interactions, reusable per-thread buffers and multi-slab MPI ghosts without full particle replication; reuse existing spatial backends without resuming deferred KD-tree optimizations. Couple gas/collisionless gravity with explicit cosmological source and time conventions. Validate uniform gas, sound waves, Sod shocks, Sedov blasts, shear/contact tests and gravitational collapse, including conservation and resolution/rank/thread convergence. Share gas/EOS interfaces with item 51. Reference: [conservative SPH](https://arxiv.org/abs/1007.1245). | Very large |
| 51 | Riemann solvers and conservative finite-volume hydrodynamics (initial implementation) | Reusable Euler primitives: primitive/conserved conversions, EOS and sound-speed interfaces, Rusanov/HLLE baselines, HLLC fluxes and an exact 1D reference where applicable. Specify wave-speed estimates, entropy/vacuum treatment and positivity-preserving fallback behavior. Add limited reconstruction, second-order Godunov evolution, CFL control and periodic/outflow/reflecting boundaries, progressing from 1D verification to conservative multidimensional finite-volume evolution. Include high-Mach/dual-energy handling, MPI halos and explicit gravity/cosmological source coupling. Check common gas conventions against SPH; add thermal physics or meshless/Godunov variants only as separately validated extensions. Validate shock tubes, contacts, rarefactions, blasts, smooth advection/sound waves and self-gravitating/cosmological gas, measuring conservation and resolution/rank/thread convergence. Reference: [HLLC construction](https://doi.org/10.1007/BF01414629). | Very large |

## Field-level forward modeling and inference

| # | Task | Concrete scope | Effort |
|---|---|---|---|
| 52 | Field-level forward model and Bayesian reconstruction (LEFTfield/BORG-style) | First expose a deterministic, reproducible map from whitened initial conditions and cosmological/bias parameters to matter/tracer fields. Reuse LPT initially, with a later PM/COLA option; specify cutoffs, assignment/dealiasing, growth, bias operators, stochasticity and real-/redshift-space conventions. Add survey observation operators for masks/selection, with distinct count/field likelihoods and explicit noise/normalization assumptions. Implement tangent/adjoint or equivalent differentiable operations through initial-field generation, evolution, deposition, bias and RSD; verify finite-difference directional gradients, adjoint dot products, zero/Hermitian modes and MPI consistency. Keep the forward/likelihood layer separate from inference: use MAP for debugging, then add prior-whitened field HMC, nuisance/cosmological inference, checkpointing and memory-controlled recomputation. Validate fixed-initial-condition predictions, mock-truth recovery, posterior predictive behavior and coverage. Distinguish LEFTfield-like perturbative bias modeling from BORG-like initial-condition posterior sampling; neither name implies native compatibility or a complete reimplementation. References: [LEFTfield](https://arxiv.org/abs/2409.10937), [BORG](https://arxiv.org/abs/1203.3639). | Very large |

The general framework's architecture and staged implementation are saved in [FieldLevel/FRAMEWORK_PLAN.md](FieldLevel/docs/FRAMEWORK_PLAN.md). Stages 1–3 now have an initial periodic implementation in [FieldLevel/FRAMEWORK.md](FieldLevel/docs/FRAMEWORK.md): generic power spectra and scalar priors, exact linear conditionals/MAP, linear/1LPT/2LPT with distinct cutoffs and smoothed bias/RSD adjoints, normalized Gaussian noise inference, joint field/nuisance HMC and streamed/reconstructed fields. The BAO prototype remains a compatibility application. A first MPI increment now supports distributed linear predictions/adjoints, exact conditionals, MAP and joint field/spectrum/bias/noise HMC, with local FFT slabs and 1/2/4-rank validation. A second MPI increment adds distributed 1LPT evolution, sparse particle/mesh communication and its bias/RSD adjoints, MAP and joint-HMC execution. A third MPI increment adds the full-grid EdS 2LPT source and adjoint, including second-order RSD and spectrum derivatives. A fourth MPI increment adds owning local initial/matter density, Lagrangian displacement/velocity and observed tracer reconstruction products for linear/1LPT/2LPT, matching the serial conventions. A fifth MPI increment adds same-rank checkpoint/restart with complete HMC/adaptation/RNG state, a frozen-field-metric snapshot and checksummed generation files. This supplies the initial periodic MPI engineering foundation; nonlinear scientific validation and performance/memory profiling remain open. A bounded nonlinear ZA/2LPT field-inference pass now checks information, reconstruction and dispersed-chain diagnostics across 1/2/4 ranks. Initial [Cartesian survey likelihoods](FieldLevel/docs/SURVEY.md) add masks/selection, normalized Gaussian voxel noise and a separate positive-link Poisson count model with MPI adjoints and MAP/HMC integration. These bounded checks and the first survey layer do not establish production coverage. Same-rank survey checkpoint/restart now registers full Gaussian/Poisson input identities, with bounded-memory voxel hashing and prior-family metadata; identity schema 2 rejects older schema-1 checkpoints, which require their original code/build. Wider survey calibration, likelihood nuisance/mean-density extensions, changed-rank restart, PM/COLA, high-cutoff padding and complete cosmological derivatives remain open. The existing BAO prototype's FL-1/FL-2 safety issues are resolved: pre-allocation single-rank rejection, once-only setup and checked field/filter dimensions. See [FieldLevel/KNOWN_ISSUES.md](FieldLevel/docs/KNOWN_ISSUES.md) for evidence and the remaining deferred portable-gradient, scientific-validation and workspace work. Engineering #17 resolved frozen-chain diagnostics. The periodic milestone and resolved prerequisites do not mark item 52 complete.

Item 49 now has an initial implementation in `WaveDarkMatter/`: full complex FFTW grids with nonempty/empty MPI slab support; reusable-buffer symmetric split-step Schrödinger–Poisson evolution; explicit axion-mass/code-unit conversion and comoving normalization; static or prescribed cosmological time integrals; density/phase and velocity-potential initializers; mass, static energy, spectral-tail and cosmological energy-balance diagnostics; external comoving-density and particle-acceleration hooks; and checksummed versioned checkpoints with changed-rank readback. Tests include independent direct DFTs and adaptive ODE evolution, analytic free waves/Poisson/interference, nonlinear second-order convergence, a separately solved radial soliton with mesh/volume checks, expanding energy balance, restart and corruption handling. See `WaveDarkMatter/README.md` and `tests/RESULTS.md` for units, allocation/FFT costs and limits. Self-consistent live-CDM integration, direct axion transfer-function IC preparation, production cosmological/halo calibration, self-interactions and isolated/absorbing boundaries remain extensions.

Items 50–52 remain open beyond their implemented milestones. Item 50 now has the bounded SPH module described below; item 51 now has the bounded finite-volume/Riemann implementation described below, sharing the ideal-gas primitive. Item 52 has the staged implementation and validation described above. Reuse existing gravity, LPT/PM, survey and statistical modules where their contracts apply.

Item 48 has an initial independent single-snapshot implementation in `PhaseSpaceHaloFinder/`: adaptive six-dimensional hierarchy and seed merging, isolated Newtonian unbinding, exclusive/inclusive bound masses, resolved circular-velocity estimates and a documented bound-extent host rule. Distributed parent FoF labels converge across periodic slabs, complete parents route to one owner without full particle replication, and particle assignments return to input ranks. SO seeds, explicit HOD mass/radius conversion and exclusive halo-history adapters are available. See its README for conventions and limits. A controlled external ROCKSTAR Plummer comparison passes, and on a COLA snapshot 96-99% of hosts above 100 particles match ROCKSTAR both ways with Vmax (ROCKSTAR-like by default) within 2.4% (`tests/RESULTS.md`); a real N-body (subhalo) comparison, production completeness/purity calibration, temporal stabilization and stronger halo-boundary definitions remain follow-ups; this is not a drop-in ROCKSTAR replacement.

## Survey mock catalogue generation

| # | Task | Concrete scope | Effort |
|---|---|---|---|
| 53 | Survey mock generator (MAKE_SURVEY-style) (initial implementation done: `Survey/SurveyMocks.h` snapshot projection, remapping, callback/RA-Dec/Healpix/MANGLE footprints, n(z) selection, randoms, lightcone input, ascii/HDF5 output, `Survey/make_survey/` executable, `Survey/mock_clustering_example/` end-to-end check; open: chunked reading of inputs larger than memory) | Build a reusable library API plus a configuration-driven executable taking periodic particle/halo/galaxy catalogues or existing lightcone output to observed mock catalogues. Start with a snapshot projection, explicitly distinct from an evolving lightcone. Add optional volume-preserving cuboid remapping, observer translation/rotation, Cartesian-to-RA/Dec conversion, cosmological distance-to-redshift inversion and radial peculiar-velocity RSD with explicit velocity/length/redshift conventions. Apply angular footprints/completeness (callback/HEALPix and optional supported MANGLE polygons), redshift cuts and radial selection with a specified n(z) or dN/dz convention, correct volume Jacobian and feasible downsampling checks. Generate matching unclustered random catalogues; distinguish selection probabilities, completeness corrections and FKP weights so selection is not applied twice. Reuse HOD, SimulationCode lightcones, survey coordinate utilities and Survey analysis/catalogue interfaces. Stream/chunk large catalogues, preserve source/replica IDs and use ID-keyed randomness for reproducibility across rank/thread counts; document repetitions from tiling/remapping. Validate coordinate round trips, zero-velocity and full-sky limits, footprint area, recovered radial density/completeness, random normalization, independent mock clustering and MPI consistency. Deliver an end-to-end example feeding items 25–27. Reference: [cmcbride/make_survey](https://github.com/cmcbride/make_survey). | Large |

Item 53 is future work, not an existing complete pipeline or a change to current priorities. Existing pieces are `HOD/` (galaxy phase-space population), `SimulationCode/src/Lightcone/` (evolving lightcones, periodic replicas and observed-redshift support), `Survey/GalaxiesToBox.h` (RA/Dec/redshift to Cartesian, the opposite direction to mock sky projection), and `Survey/` power/window/reconstruction routines (which consume data/random catalogues). These do not yet provide a general MAKE_SURVEY-style box-to-selected-survey executable with remapping, footprint/radial selection and matching random generation. Gravitational redshift or other relativistic observing effects should be explicit optional extensions when using compatible lightcone fields.

## Glass initial conditions

| # | Task | Concrete scope | Effort |
|---|---|---|---|
| 54 | Glass particle loads and IC integration (initial implementation) | Generate periodic, statistically isotropic glass particle loads from reproducible seeded positions using repulsive gravity and controlled damping/relaxation. Reuse the gravity/particle infrastructure with explicit force-sign, softening, timestep and stopping conventions; track residual forces/velocities and convergence. Provide stable particle IDs, versioned load output/readback, and MPI/OpenMP support without full particle replication. Apply LPT displacements and velocities at irregular glass positions through the shared IC/transfer-provider interfaces, preserving normalization, units and particle species conventions. Keep this distinct from existing external-IC inversion routines that assume an initially regular grid. Validate low-k discreteness suppression, isotropy, pair statistics, residual forces, resolution/relaxation convergence, and thread/rank consistency; compare displaced glass and lattice starts against the same target spectrum. Treat multiple-species quiet loads and overlap avoidance as explicit later extensions. | Medium–large |

Item 54 now has an initial implementation in `GlassIC/`: ID-keyed Poisson seeds, softened repulsive P3M gravity with damped and limited drifts, explicit force/velocity stopping status and residual history, portable versioned sharded load IO with changed-rank-count streaming readback, and CIC application of displacement/velocity at the same irregular glass coordinates. The shared transfer-provider bridge normalizes physical spectra to box-grid units, and the 3LPT adapter retains the transverse term and replaces relaxation velocities. Generation, short-range ghosts, migration, IO and field sampling avoid particle-payload all-gathers. See `GlassIC/README.md` for force/softening/time conventions, tests and measured limits. Larger-volume k^4/isotropy and target-spectrum calibration, timestep/softening/mesh convergence, multi-species quiet loads and automatic SimulationCode parameter integration remain follow-ups. Existing grid-only external-IC inversion is unchanged.

## Arbitrary-order Lagrangian perturbation theory

| # | Task | Concrete scope | Effort |
|---|---|---|---|
| 55 | Generic-order EdS LPT fields (implemented; small-field validation through order 6) | Compute growing-mode Lagrangian displacement fields through a user-selected perturbative order in an Einstein–de Sitter background, using recursive longitudinal and transverse source construction rather than hard-coded order-specific kernels. State factorial, sign, potential and growth conventions explicitly; retain the transverse displacement required from third order onward even for initially irrotational flow. Provide per-order fields and their summed displacement/velocity with the EdS time dependence, reusing existing FFT, derivative and Poisson infrastructure. Specify periodic zero-mode, Hermitian symmetry and nonlinear-product dealiasing conventions. Support MPI/OpenMP with reusable workspaces, bounded caching or recomputation, and reports of rapidly growing memory/FFT costs as order increases. Validate orders 1–3 against existing implementations after convention conversion, higher orders against an independent recursion/reference, plane-parallel exactness, amplitude/time scaling, Lagrangian irrotationality constraints, resolution and rank/thread consistency. Document convergence and shell-crossing limits; general cosmology, decaying modes and time-Taylor evolution are separate extensions. | Large |

Item 55 is implemented in `LPT/ArbitraryOrder.h`: finite requested-order growing-mode EdS recursion with transverse feedback, explicit support and owned-memory budgets, conservative full-bandwidth work meshes, collected/streamed outputs, EdS evaluation and cache/recompute policies. Small-field independent references now reach nonzero order 6 in float/double, with scalar/Cauchy equations, symmetry and plane-parallel checks, resolution/support sensitivity and a sampled-Jacobian truncation example. Grouped cubic minors fix a valid plane-wave compatibility failure without loosening tolerances. Double MPI/OpenMP and serial/OpenMP float checks are documented in `LPT/SCIENCE_RESULTS.md`; CI includes the new science suite and a float configuration. Higher orders remain beyond the recorded validation range. Production-field convergence/shell-crossing calibration, cheaper proven bandwidth policies, float MPI, empty slabs and SimulationCode/COLA integration remain extensions; general cosmology, decaying modes and time-Taylor evolution remain separate.

### Item 52: bounded survey calibration follow-up (13 November 2026)

Longer dispersed production chains now have an exact active-mode reference for
12 independent 2LPT initial-field mocks per Gaussian/Poisson observation model,
with masked exact-Gaussian checks, independent future-observation predictions,
mock-cluster coverage intervals and selection/mesh convergence plots. This is
a fixed-nuisance, 26-mode conditional experiment, not completion of #52.
Coarse point-sampled survey boundaries show percent-level total-integral errors;
volume-integrated observation modeling and wider nuisance/cosmological calibration
remain open. See [FieldLevel/survey_validation/RESULTS.md](FieldLevel/docs/validation/studies/survey_validation/RESULTS.md).

### Item 50: initial conservative SPH (13 November 2026)

`SPH/` now provides periodic 1D/2D/3D density-energy ideal-gas SPH with adaptive
smoothing lengths, grad-h pressure/work, viscosity and a Balsara shear limiter,
optional conservative artificial conductivity, global RK2/CFL control, slab MPI
ghosts/migration and OpenMP targets. `Hydrodynamics/EquationOfState.h` supplies
shared gas primitives also used by item 51. Explicit physical/comoving variables,
Hubble terms and an external gravity callback define the initial coupling API.
Independent variational/all-pair, conservation, dissipation, safety and serial/MPI
checks accompany sound/Sod/blast examples and plots. See
[SPH/README.md](SPH/README.md) and [SPH/tests/RESULTS.md](SPH/tests/RESULTS.md).
This does not complete #50: native SimulationCode gas/collisionless/self-gravity
coupling, production cosmological and Sedov/shear/contact/collapse calibration,
alternative-kernel/neighbor convergence, checkpoints and larger MPI performance
studies remain. The initial 3D sound-wave refinement shows weaker velocity
convergence than the 1D test; no general 3D convergence order is claimed.

### Item 51: initial finite-volume Euler and Riemann solvers (13 November 2026)

`Hydrodynamics/` now supplies reusable primitive/conserved conversions, Rusanov,
HLLE and guarded HLLC fluxes, an exact positive-state 1D reference including
generated vacuum, and uniform Cartesian 1D/2D/3D finite-volume evolution. Limited
reconstruction, SSPRK2, collective positivity fallback/retries, periodic/outflow/
reflecting boundaries, two-layer slab MPI halos, OpenMP and optional accounted
dual energy are included. Gravity/prescribed-expansion callbacks use explicit
physical/comoving conventions and share the ideal-gas EOS with SPH. Tests and
independent exact cell-average, smooth-refinement, shock/vacuum, blast and MPI
replay experiments are documented in [Hydrodynamics/README.md](Hydrodynamics/README.md)
and [Hydrodynamics/tests/RESULTS.md](Hydrodynamics/tests/RESULTS.md). This initial
scope does not complete #51: native SimulationCode gas/Poisson/species coupling,
production cosmological/high-Mach calibration, hydrostatic/collapse and Sedov
tests, general EOS, checkpoints and broader performance work remain. The measured
coarse multidimensional refinement rates are below nominal second order and are
reported explicitly; no universal convergence-order claim is made.

### Item 52: PM forward milestone (14 November 2026)

`FieldLevel/ParticleMesh.h` adds periodic fixed flat matter/Lambda PM forward
evolution from whitened real-space initial fields, with ZA growing starts, CIC
forces, KDK evolution, local matter/particle outputs and scalar conservation
diagnostics. It supports slabs/pencils and unequal pencil boxes using sparse
mesh routing without full particle replication. See
[the PM guide](FieldLevel/docs/PARTICLE_MESH.md). Reverse derivatives, memory-controlled
trajectory checkpoint/recomputation, PM posterior/MAP/HMC integration and broader
scientific calibration remain the next milestones; #52 remains open.

### Item 52: PM discrete adjoint and bounded replay (14 November 2026)

`FieldLevel/ParticleMeshAdjoint.h` adds white-field/spectrum pullbacks of the
fixed-background PM map, including both CIC position-dependence branches,
spectral forces, KDK and the ZA start. Temporary midpoint checkpoints plus
prefix recomputation bound retained particle states and report payload bytes
and replay counts. See [the adjoint guide](FieldLevel/docs/PM_ADJOINT.md).
This is not yet a PM posterior: periodic likelihood/MAP integration and then
small HMC validation are the next #52 milestones. Existing forward milestone
artifacts and the bounded scientific-validation limits remain in place.

### Item 52: periodic PM posterior and MAP (14 November 2026)

`FieldLevel/ParticleMeshPosterior.h` now connects PM dynamics and its discrete
adjoint to the shared normalized Fourier-Gaussian likelihood, white/scalar priors
and MAP optimizer. Distinct IC/force observation layouts, global parameter-gradient
reduction, deterministic noise and physical volume scaling are checked in serial
and MPI/pencil builds. A bounded fixed-parameter mock converges and a short joint
fit exercises spectrum/noise updates; broader-band line-search limits are retained.
See [the periodic PM inference guide](FieldLevel/docs/PM_POSTERIOR.md). Next is bounded
HMC scientific validation; #52 remains open beyond these milestones.

### Item 52: bounded periodic PM HMC validation

[The validation driver](FieldLevel/docs/validation/studies/pm_hmc_validation/README.md) compares fixed-step and
adapted HMC with refined two-dimensional quadrature for two active IC modes,
conditioning on the other white coordinates and scalar parameters. Both ensembles
pass the recorded moment/Monte Carlo error checks with no post-warmup divergences.
Energy/reversibility probes and serial/MPI/pencil/thread/sanitizer replays pass.
Short full-field and joint spectrum/noise traces exercise the public sampler and
canonical replay; they do not establish full-posterior mixing or calibration.
See [results and limitations](FieldLevel/docs/validation/studies/pm_hmc_validation/RESULTS.md). Larger fields, longer
joint runs, posterior predictive checks, repeated-mock coverage and PM-specific
restart/survey/tracer/cosmological derivatives remain separate; item 52 stays open.

### Item 52: bounded full-field PM posterior validation (15 November 2026)

[The full-field campaign](FieldLevel/docs/validation/studies/pm_full_validation/README.md) samples all 64 white
coordinates in four dispersed fixed-parameter and four joint spectrum/noise chains.
For this N=4/M=8 mock, six IC directions affect the likelihood and 58 have an exact
unit-Gaussian posterior. All declared feature-mixing, analytic null-moment,
normalization and posterior-predictive checks pass. Backend/thread/sanitizer replay
also passes. [Results and plots](FieldLevel/docs/validation/studies/pm_full_validation/RESULTS.md) separate latent
uncertainty from noisy predictions. This is bounded single-mock evidence;
larger fields/mode support, repeated-mock coverage and numerical convergence remain
open, along with PM checkpoint/restart, tracer/RSD/survey and cosmological derivatives.
Item 52 remains open.

### Item 52: same-layout PM inference checkpoint/restart (15 November 2026)

[ParticleMeshCheckpoint.h](FieldLevel/ParticleMeshCheckpoint.h) registers explicit PM target
identity with the existing trajectory-boundary HMC checkpoint API. It includes both
IC/observation layouts, copied grid/domain geometry, background/time nodes, versioned
force/ZA/CIC/KDK/adjoint/metric conventions and replay budget. Complete sampler,
RNG/adaptation and frozen-metric state is reused. Fixed and joint fresh-process
continuation is bitwise exact on serial, 1/2/4-rank slabs and pencils (including
unequal boxes), with sanitizer and incompatible-configuration/failure checks.
Existing non-PM fingerprints/file schemas are preserved and old-serializer linear/
LPT checkpoints resume under the new adapter. [The guide](FieldLevel/docs/PM_CHECKPOINT.md) and
[results](FieldLevel/docs/validation/tests/PM_CHECKPOINT_RESULTS.md) state the callback-versioning and same-build
contract. Changed-layout restart, larger/multi-mock calibration and PM survey/tracer/
RSD/cosmological extensions remain open; item 52 is not complete.

### Item 52: PM convergence and more active modes (15 November 2026)

[The bounded resolution campaign](FieldLevel/docs/validation/studies/pm_resolution_validation/README.md) matches one
physical 18-mode Fourier field across particle/force grids and time schedules.
Time refinement and four directional derivative sweeps pass. Four joint chains
on N=6/M=12 sample all 216 white coordinates and spectrum/noise; all 653 monitored
features pass the declared mixing checks with no post-warmup divergences.

**Spatial accuracy remains unresolved:** four force/particle criteria fail, with
last-refinement field/gradient differences of 21.9%/68.5% (force mesh) and
16.8%/50.9% (particle lattice). Independent NumPy and native tiny-kick probes reproduce
strong sparse-load/force-grid dependence. [PM-S1](FieldLevel/docs/KNOWN_ISSUES.md) prioritizes dense
particle support and controlled force/assignment investigation before accuracy or
coverage claims. The numerical force law is unchanged. [Results and plots](FieldLevel/docs/validation/studies/pm_resolution_validation/RESULTS.md)
retain the failed criteria. Item 52 remains open.

### Item 52: PM dense-particle sampling (15 November 2026)

Native N>M now supports more IC/particle cells than force cells, with initial
support below both Nyquists and compatible slab/pencil geometry. The force law,
adjoint and checkpoint codec are unchanged. Serial, 1/2/4-rank MPI, sanitizer,
normalized posterior and fresh-process restart checks pass, including old PM
checkpoint continuation. The [dense-sampling study](FieldLevel/docs/validation/studies/pm_oversampling_validation/RESULTS.md)
passes the original bounded finest-correction criteria: N=64, M=32 vs 64 changes
density/gradient by 0.735%/6.198%; particle and time refinements also pass. This
completes native oversampling support and this refinement experiment, not broad
PM calibration. The original sparse-load failures remain recorded; more modes,
nonlinear regimes, refined-data inference and repeated-mock coverage are open.
Particle/white memory grows as N^3. Item 52 remains open.

### Item 52: many-mode PM accuracy and refined-data reconstruction (15 November 2026)

[The broader PM campaign](FieldLevel/docs/validation/studies/pm_nonlinear_validation/RESULTS.md) increases initial support to
122 real directions and observations to 256 real degrees, using two seeds and
three strength/time fixtures. Particle/time finest-correction checks and
forward/adjoint/backend checks pass. Force-grid projected-gradient changes of
16.3%/75.1%/60.6% fail the unchanged 15% criterion; both stronger fixtures also
fail the noise-scaled field criterion. Finite force references still need
controlled particle sampling. Fixed-parameter MAP diagnostics now use data from
a finer discretization, with optimizer status and recovery measures explicit.
All three 12-iteration fits improve reconstruction but remain unconverged; initial
error ratios are 0.585/0.560/0.290 with correlations 0.816/0.834/0.969. The original
80-iteration attempts were interrupted for runtime; criteria are unchanged.
No force law, posterior, sampler or optimizer is changed. The original and dense
18-mode studies remain preserved. Broad PM calibration and item 52 remain open.

### Item 52: particle-sampled PM force reference (15 November 2026)

[The controlled forward study](FieldLevel/docs/validation/studies/pm_force_reference_validation/RESULTS.md) extends strong91 to
N64/96/128 on M32 and M64, with a N128/M64 timestep check. Particle corrections
N96→N128 are 0.0648%/0.0424% in observed-band density; the N128 time correction
is 0.0119%. Force M32→M64 at N128 still changes density by 3.7749% and 0.1063
noise SD per observed degree, retaining the failed 0.1-noise criterion. The density
force-grid difference is stable across particle loads. This establishes a better
sampled finite reference in the tested band, not an exact continuum solution.
A forward-only validation CLI avoids adjoint/MAP cost, preserving original full-mode
outputs bitwise; serial/MPI/sanitizer, N128 backend and launcher cleanup checks pass.
No production physics is changed. Posterior-gradient failures and unconverged MAP
remain open; next compare explicit force/assignment-window choices against this
controlled setup. Item 52 remains open.

### Item 52: explicit PM force-window convention (15 November 2026)

`PMForceWindow::CICCompensated` is an opt-in inverse squared CIC force window,
with a matching discrete adjoint and PM checkpoint law tag 2. Default `None`
retains tag-1 identity bytes and old numerical outputs/checkpoint compatibility.
[The controlled comparison](FieldLevel/docs/validation/studies/pm_window_validation/RESULTS.md) reduces the N128 M32→M64 raw density
correction from 3.775% to 1.768%; after smoothing outputs to a common observation
window, the descriptive density correction falls from 2.068% to 0.0309%.
The raw joint-gradient correction improves from 75.1% to 34.9% but still fails
15%. Dense linear-force controls, derivatives, serial/MPI/sanitizer and restart
checks pass. Sparse-load compensation is poor and remains unvalidated for use.
Refining the tiny-kick measurement resolves its amplified sparse-load truncation
error; the coarse-step failure is retained. This raw-observation study motivated the fixed response implemented below.
Broader gradient refinement and validation of actual posterior behavior remain open. MAP conditioning, broad calibration and
PM survey/tracer/RSD/cosmological work remain open; item 52 is not complete.

### Item 52: fixed PM observation convention (15 November 2026)

The [fixed-response study](FieldLevel/docs/validation/studies/pm_observation_validation/RESULTS.md) now implements the
actual signal response, transpose and checkpoint identity. With a common M32 CIC
observation window, compensated M32→M64 density and joint-gradient corrections
are 0.0309% and 0.476%, passing the original bounded criteria. The uncompensated
joint-gradient baseline still fails at 41.6%. Custom real coefficients, noise after
response, both distributed backends and legacy bitwise compatibility are tested.
This is one finite-reference realization; adequate broader gradient sampling,
posterior calibration, MAP conditioning and PM survey/tracer/RSD work remain open.

### Item 52: refined-data compensated PM reconstruction (15 November 2026)

The [new reconstruction study](FieldLevel/docs/validation/studies/pm_reconstruction_validation/RESULTS.md) generates
a compensated N128/M64/32-step mock and fits at N32/M16/16 steps with a common
M16 observation window and fixed nuisance parameters. At 24 accepted iterations,
initial coefficient error is 0.2696 of the zero-field baseline,
correlation 0.9643, and residual 0.7881 noise SD. Optimizer
status is MaximumIterations; raw and projected gradients are retained with the plots.
Serial/MPI slab/pencil/sanitizer replay and legacy bitwise comparisons pass.
A costlier N64/M32 attempt is retained; the public optimizer is unchanged.
Broader seed/amplitude/refinement checks remain explicitly deferred by request;
this is reconstruction validation, not posterior calibration. Item 52 remains open.

### Item 52: MAP optimizer and frozen-data comparison (15 November 2026)

`MAP.h` adds an options overload with limited-memory L-BFGS and value-only
gradient descent; legacy `map()` is unchanged. `BasicPosterior::value` omits the
forward-model adjoint while retaining the normalized target and priors. Distributed
field/scalar reductions, controls, curvature and history limits are tested.
[The frozen-data comparison](FieldLevel/docs/validation/studies/pm_optimizer_validation/RESULTS.md) preserves legacy24
coefficients bitwise; value-only GD cuts pullbacks from 144 to 25.
L-BFGS status is LineSearchFailed, raw gradient 0.000116038 at
83 iterations with tolerance2e-5. Its recovery/error and work/memory
measurements are reported separately. Broader refinement remains deferred, and
posterior calibration/PM survey/tracer/RSD work remains open.

### Item 52: stalled PM reconstruction diagnosis (16 November 2026)

[Read-only stationarity probes](FieldLevel/docs/validation/studies/pm_stationarity_validation/RESULTS.md) reproduce
iteration83 and capture its full state. Fresh/reused caches and replay budgets
0/1/4/8 give identical gradients/values; three stencil-preserving directional
checks pass. A particle lies3.19e-10 force cells from a final CIC face. Larger
trials cross stencils, while smaller smooth-branch decreases around2e-12 are
masked by ordinary objective evaluation/subtraction errors around1e-11. The
public value/full base targets agree; no cache/adjoint discrepancy was observed
in these tests. Solver and assignment are unchanged. Next test stable objective
change comparisons for Armijo before another full reconstruction. Broader
refinement remains deferred; convergence and general calibration remain open.

### Item 52: stable Armijo comparisons (16 November 2026)

`BasicPosterior` gains an opt-in stable change API for likelihoods implementing
local differences; Gaussian variance/normalization and field/scalar priors are
included. Defaults and ordinary target evaluations remain unchanged.
[The frozen-state/full-fit check](FieldLevel/docs/validation/studies/pm_stable_armijo_validation/RESULTS.md) accepts
the diagnosed hidden descent. Stable L-BFGS uses 594 value trials versus
1311, but ends at iteration 98 with raw gradient 0.00013902
and status LineSearchFailed: convergence remains unresolved. Stable GD
retains its 24-step coefficients bitwise. MPI/sanitizer and default-compatibility
checks pass. Broader refinement/calibration remains deferred; next inspect CIC
regularity/forward precision at the new endpoint before changing assignment.

### Item 52: stable endpoint regularity diagnostic (16 November 2026)

[Iteration-98 probes](FieldLevel/docs/validation/studies/pm_stable_stationarity_validation/RESULTS.md) reproduce the
stable fit bitwise. A final-node CIC coordinate lies 7.47e-13 grid cells from a
face. One-sided negative-gradient slopes differ across two stencil axes: the
branch-preserving side agrees with the adjoint, while the nominal descent side
increases the target. All 40 reconstructed search trials have positive stable
changes; ordinary subtraction would accept 17 tiny apparent improvements.
Fresh/reused caches and checkpoint budgets agree exactly. Two central derivative
gates remain unresolved because no eligible same-stencil pairs exist; the report
retains passed=false and consistency_passed=true. Convergence remains open.
Next consider opt-in TSC with matched deposit/gather derivatives and adjoint,
preserving CIC defaults. No assignment or stopping-rule change was made. Broader
refinement/calibration remains deferred.

### Item 52: opt-in TSC PM assignment (16 November 2026)

[Matched-assignment reconstruction](FieldLevel/docs/validation/studies/pm_tsc_validation/RESULTS.md) converges in
16 iterations with raw gradient 5.6502e-06 at the unchanged
2e-5 tolerance. TSC covers force deposition/gather, reported density, matching
adjoint and optional force compensation. CIC defaults and checkpoint identities
remain compatible. Targeted serial/MPI/pencil/sanitizer and checkpoint checks pass.
The matched TSC reference changes the discrete target; this is one finite
realization, not general convergence or posterior calibration. Broader refinement
remains deferred. Stable objective comparisons remain opt-in.

### Item 52: joint TSC MAP validation (16 November 2026)

[Initial field, amplitude and noise](FieldLevel/docs/validation/studies/pm_tsc_joint_validation/RESULTS.md) converge
jointly in 54 iterations with joint norm 9.84911e-06
at tolerance 2e-5. Scalar gradient identities and six refined directional gates
pass; serial/slab/pencil/sanitizer checks agree and fixed mode reproduces the
committed TSC driver bitwise. No library or optimizer code changed. This study
explicitly widens amplitude support to [-4,4]; Gaussian mean/scale remain fixed.
The high joint-MAP amplitude follows the whitened-coordinate density mode and
must not be interpreted as an unbiased or marginalized amplitude estimate.
Physical initial-field correlation is 0.964540. General scalar
posterior inference, coverage and broader refinement remain deferred.

### Item 52: bounded TSC posterior pilot (16 November 2026)

[Active-field/amplitude/noise HMC](FieldLevel/docs/validation/studies/pm_tsc_posterior_validation/RESULTS.md) completes
four N16 particle chains on the frozen data, M16 force mesh and 16 time steps,
as selected by the user. Exact inactive-Gaussian marginalization and physical
coordinates include the 122*A/2 Jacobian; direct target/gradient checks pass.
All Rhat values are <1.05 and there are no retained divergences, but bulk/tail ESS
criteria remain failed (minima 93.3/180.9); the report
keeps passed=false. Scalar intervals and predictions are exploratory. N32 mixing
and coverage remain unvalidated. A <=16-mesh bounded dense lookup reduces a short
case's CPU time about 21% with bitwise-identical outputs; larger meshes stay sparse.
PM/LPT, MPI/pencil, sanitizer and old-checkpoint checks pass. Broader refinement
remains deferred; more effective draws are needed before calibrated inference.

### Item 52: alternating field/scalar posterior pilot (16 November 2026)

[Optional field HMC plus conditional scalar updates](FieldLevel/docs/validation/studies/pm_tsc_posterior_validation/BLOCKED_RESULTS.md)
uses the identical N16 target and original diagnostic thresholds. Conditional
slice sweeps reuse the cached residual with zero additional PM evaluations;
rejected-field cache restoration and HMC force refresh are checked. The report
compares all 248 features by ESS per actual PM evaluation. Serial, slab/pencil
MPI and sanitizer checks pass; the original joint sampler remains the default.
This is fixture validation, with N32 mixing, coverage and a general production
blocked adapter still outside its scope. See the report for measured efficiency
and the unchanged strict mixing verdict.

### Item 52: fixed field-metric pilot (16 November 2026)

[The fixed-curvature pilot](FieldLevel/docs/validation/studies/pm_tsc_posterior_validation/METRIC_RESULTS.md)
compares a frozen 122-dimensional field mass to the alternating and original
samplers on the same N16 target. Hessian setup and all warmup/sampling evaluations
are included in efficiency. Matrix positivity, derivatives, MPI slabs/pencils,
malformed inputs, Gaussian moments and sanitized sampling are checked. The report
preserves the original mixing verdict. This dense experiment does not provide
a production full-grid metric; N32 mixing and coverage remain unvalidated.

### Item 52: shorter trajectories at matched PM cost (16 November 2026)

[The controlled trajectory experiment](FieldLevel/docs/validation/studies/pm_tsc_posterior_validation/TRAJECTORY_RESULTS.md)
compares 2..4-step trajectories with 320 retained draws against 4..8 steps with
128 draws, at the same expected total PM cost. Target, frozen metric, seeds,
warmup and diagnostic criteria stay fixed. Actual PM costs include a fresh shared
metric build; the report preserves all convergence failures. Native code and
production defaults are unchanged. N32 mixing and repeated-mock coverage remain
outside this experiment. Short trajectories clear the tail gate but leave 182
bulk failures; an intermediate maximum 6-step schedule is the next candidate.

### Item 52: intermediate trajectories at matched PM cost (16 November 2026)

[The intermediate trajectory report](FieldLevel/docs/validation/studies/pm_tsc_posterior_validation/INTERMEDIATE_RESULTS.md)
adds a maximum-6-step, 192-draw schedule at the same expected total PM cost as
the preceding maximum 4/8 schedules. Target, metric, seeds, warmup and diagnostic
thresholds remain fixed. Actual cost includes a fresh shared metric build, and
all failures stay in the report. Native code and defaults are unchanged. The
three-way comparison remains a bounded N16 experiment; N32 mixing and coverage
are still open. The intermediate schedule retains 18 bulk failures and one
tail failure (ESS 199.98); a longer fixed-schedule validation remains needed.

### Item 52: longer fixed-schedule N16 validation (16 November 2026)

[The 512-draw validation](FieldLevel/docs/validation/studies/pm_tsc_posterior_validation/LONG_VALIDATION_RESULTS.md)
keeps the intermediate maximum 6-step sampler fixed and increases the declared
budget about 2.2x. Target, matrix, seeds, warmup and convergence criteria stay
unchanged. The pilot prefix, raw features, native diagnostics and saved analysis
are replayed. All original N16 mixing gates now pass (max Rhat 1.0071,
min bulk/tail ESS 471.1/726.0, zero divergences). No production default
changes; N32 mixing and repeated-mock coverage remain open.

### Item 52: independent same-model N16 posterior validation (17 November 2026)

[New IC191/noise192 mock](FieldLevel/docs/validation/studies/pm_tsc_posterior_validation/INDEPENDENT_MOCK_RESULTS.md)
passes the unchanged fixed-protocol four-chain 512-draw mixing gates for all
248 features (max Rhat 1.0064, min bulk/tail ESS 463.9/552.9, no divergences).
Generator and inference both use N16/M16/16 steps. Data-only MAP, fresh frozen
metric and actual target checks precede sampling. Scalar truth inclusion and
initial-field recovery are reported separately from mixing; one mock does not
establish ensemble coverage. Serial/MPI slab/pencil/sanitizer engineering and
legacy/replay checks pass. N32 mixing, physical convergence, repeated-mock
coverage and a scalable full-grid metric remain deferred, not completed.

### Item 52: repeated same-model N16 posterior validation (17 November 2026)

[Results and plots](FieldLevel/docs/validation/studies/pm_tsc_posterior_validation/REPEATED_MOCK_RESULTS.md).

The original same-model N16 plan declared IC/noise seeds 291/292, 391/392,
491/492 and 591/592, excluding the previous 191/192 case. The user requested
skipping the fourth after two results were known and the third was running.
Three datasets participate; the fourth is user-skipped, not missing or failed. The frozen
four-chain 64-warmup/512-retained schedule and all 248 strict feature gates stay
unchanged. Completed: 3/3; mixing passes: 3/3;
consistency passes: 3/3; campaign passed=True.
No failed case is replaced, extended or dropped. Each dataset has a data-only
MAP and a fresh metric frozen before sampling. The report separates individual
mixing verdicts from scalar interval inclusion and field recovery. Three datasets
give broad coverage uncertainty; correlated field-mode counts are not pooled.
This is limited repeated-data recovery at fixed true hyperparameters, not SBC.
Native sources/binaries remain unchanged; synthetic reporting and raw/aggregate
replay checks are recorded. N32 mixing, physical convergence, larger calibration
ensembles and scalable full-grid metrics remain open. This validation pilot
buffers chain output until completion and has no mid-chain restart checkpoint;
finished mocks are persistent, but interrupted unfinished chains must restart.


### Item 52: periodic galaxy amplitude model (17 November 2026)

The [LEFTfield-style project plan](FieldLevel/docs/LEFTFIELD_PLAN.md) now has
a [composed Eulerian LPT model](FieldLevel/docs/EULERIAN_LPT.md): unbiased
real-space LPT matter followed by density, centered density squared, centered
tidal squared and physical-Laplacian operators, with an amplitude-only spectrum
alpha=sigma8/sigma8_fid. Cubic likelihood corner modes are explicitly enumerated.
The generic posterior now accepts fixed noise or an explicit sampled noise index,
and distinguishes a fixed-IC conditional target from the whitened-field prior.
Small Gaussian tracer-field tests cover the full field/all-scalar gradients,
normalization, direct particle Fourier sums and slab/pencil MPI. These are
forward/adjoint checks, not galaxy-catalog inference or posterior calibration.
Next: fixed-IC scalar sampling and mesh/assignment refinement. Exact catalog
IC/spectrum/cosmology inputs, challenge filter details, NUFFT integration, compact
latent/product meshes and this model's checkpoint driver remain open.


### Item 52: bounded fixed-IC amplitude pilot (17 November 2026)

Added the reusable fixed-noise linear-bias likelihood cache and a fixed-IC
amplitude/bias sampler with the coordinate Jacobian, independent quadrature
and MPI tests. The [three-mesh Gaussian tracer-field pilot](FieldLevel/docs/validation/studies/galaxy_fixed_ic/RESULTS.md)
completed all 12 declared chains, but alpha bulk ESS 270–331 falls below the
frozen 400 gate. Rhat is about 1.009. Observed mean shifts 0.048/0.067 SD are
small, but paired Monte Carlo uncertainty does not certify the 0.1 SD budget.
The pilot is preserved without extension or thinning. Next: improve the exact
joint amplitude/bias proposal and declare a new campaign. This remains a small
fixed-IC field validation, not catalog inference or completion of item 52.


### Item 52: joint amplitude/bias fixed-IC pilot (18 November 2026)

The [separate joint-proposal campaign](FieldLevel/docs/validation/studies/galaxy_fixed_ic_joint/RESULTS.md)
passes all frozen scalar mixing gates in three small configurations, on the
original data/priors and draw budget. Alpha bulk ESS 578–649 gives 1.75–2.40
times the ESS per forward call. All-parameter max Rhat is 1.00613. Paired
refinement mean shifts plus two batch MC errors stay below 0.025 SD, within
the 0.1 SD budget. The normalized Gaussian proposal densities and finite
prior support are handled in the exact joint MH kernel; proposal regularization
does not change the posterior. The original failed pilot is preserved.
This remains one fixed-IC Gaussian tracer field, not catalog or free-IC inference.
Next: catalog/NUFFT integration, compact meshes and production restart; real-data
inputs and catalog-specific model/numerical validation remain open.


### Item 52: periodic catalog observation ingestion (18 November 2026)

Added the [streaming catalog adapter](FieldLevel/docs/PERIODIC_CATALOG.md) for
weighted/unweighted periodic data, internal MPI ownership routing, explicit
density/Poisson normalization, and same-layout model mode export. CSV and optional
HDF5 converters stream bounded batches and perform one final NUFFT transform.
[Small-fixture validation](FieldLevel/docs/validation/studies/periodic_catalog/RESULTS.md)
passes serial and slab/pencil 1/2/4-rank checks and all 17 file-interface checks;
maximum mode error is 5.55e-9 at epsilon=1e-8. No challenge catalog was downloaded
and no new posterior was sampled. Next: NUFFT matter deposition/position adjoint,
compact latent/product meshes and production restart. Catalog-specific modeling,
real-data inputs and free-IC scientific validation remain open; item 52 is ongoing.


### Item 52: PSWF NUFFT matter deposition and position adjoint (18 November 2026)

Added an explicit [PSWF matter option](FieldLevel/docs/NUFFT_MATTER.md) to distributed
LPT and the composed Eulerian model. The reverse differentiates the actual fitted
spreading weights and reuses sparse node scatter/transpose, preserving original
particle ownership across rank boundaries. Smooth assignment remains the default.
[Bounded numerical validation](FieldLevel/docs/validation/studies/nufft_matter/RESULTS.md)
passes serial and slab/pencil 1/2/4-rank field/scalar/particle checks, both evolution
orders, bias/RSD and short MAP/HMC, plus smooth/checkpoint/catalog regressions.
The N12/N24 fixture's mode error is 1.72e-9 at epsilon=1e-8. Compact stencil jumps,
derivative accuracy and wider-support cost remain explicit numerical considerations;
no production benchmark or new posterior was run. Next: compact latent/product
meshes, then production restart and catalog-specific real-data/free-IC validation.
Item 52 remains ongoing.


### Item 52: compact latent and padded product meshes (18 November 2026)

Added explicit [compact/product mesh separation](FieldLevel/docs/COMPACT_MESHES.md)
to the Eulerian LPT model. `mesh` controls sampled white-field/likelihood storage,
`product_mesh` controls padded LPT/particle/operator work, and `density_mesh`
controls deposition. The isometric white-field lift and its Euclidean transpose
preserve the physical initial spectrum and compact unit-Gaussian prior; padding
guards remain on the product grid. Zero/default product_mesh keeps the old path.
[Validation](FieldLevel/docs/validation/studies/compact_meshes/RESULTS.md) passes
serial and slab/pencil 1/2/4-rank smooth/PSWF checks against independently evaluated
padded fields, plus prior/gradient/mock/metric and short HMC checks. The 8/12/24
fixture uses 512 instead of 1728 sampler field coordinates at fixed particle and
product resolution. This is a per-vector storage saving, not a total RSS or
production-speed result. Shared-mesh, fixed-IC and catalog regressions pass.
Next: production restart for the Eulerian target, followed by catalog-specific
convergence and real-data/free-IC validation. Packed Fourier storage and separate
particle/evolution/operator resolutions remain extensions. Item 52 is ongoing.


### Item 52: Eulerian HMC checkpoint/restart (18 November 2026)

Added the [Eulerian checkpoint adapter](FieldLevel/docs/EULERIAN_RESTART.md),
including shared/compact grids, both matter assignments and an explicit target
fingerprint for grids/filters/operator versions, priors/noise and observed modes.
The generic numerical-configuration codec is extensible while preserving old
default bytes. [Validation](FieldLevel/docs/validation/studies/eulerian_restart/RESULTS.md)
passes serial and slab/pencil 1/2/4-rank fresh-process joint and field-only replay,
write-failure/corruption/target rejection, and indexed output-record replay.
Existing field/survey/PM checkpoint suites pass, and the new reader resumes real
files written by a retained pre-refactor executable. Restart requires the same
layout/build/thread configuration and a versioned callback problem ID. The
specialized fixed-IC Metropolis sampler remains unchanged; no new posterior
campaign or power-loss durability test was run. Next: catalog-specific convergence
and resource checks, then a declared inference pilot. Initial cubic filtering and
real-data/free-IC scientific validation remain open. Item 52 is ongoing.


### Item 52: bounded numerical/resource screen (18 November 2026)

Added the [maintained frozen screen](FieldLevel/validation/galaxy/numerics/README.md)
with resumable per-case hashes, physical Fourier-keyed initial conditions, numerical
and local Fisher proxies, finite differences, warm timing, process RSS and a bounded
HMC smoke. [All 17 serial/MPI cases pass](FieldLevel/docs/validation/studies/numerical_screen/RESULTS.md).
The N8/N12/N24, epsilon=1e-8 baseline has maximum noise-weighted distance 0.00123
and median forward+gradient cost 0.037 s on this host. Increasing product/particle
resolution matters more than tightening tolerance here. Particle/product refinement
is coupled. Compact per-vector storage savings do not establish total-RSS savings;
measured process peaks include padded/sparse workspaces and transfer overhead.
No posterior mean-shift, coverage, real-catalog or production-scaling claim is made.
A separate [FL-7 metric issue](FieldLevel/docs/KNOWN_ISSUES.md) uses unit legacy noise
and unbiased response in the Eulerian field metric. This preserves the target but
can slow field exploration. Next: noise/response-aware frozen metric with restart
identity and a short exploration check, then a declared multi-chain pilot.
Item 52 remains ongoing.


### Item 52: FL-7 field metric correction (18 November 2026)

Resolved [FL-7](FieldLevel/docs/EULERIAN_METRIC.md): posterior-aware Gaussian
metric dispatch uses actual noise policy, covariance volume and mask with
Eulerian linear/Laplacian response; HMC keeps its saved initial snapshot. Cached
log precisions operate on sampled/compact grids and invalidate collectively
when only some ranks own changed modes. MAP routes through the same interface.
[Validation](FieldLevel/docs/validation/studies/eulerian_metric/RESULTS.md) passes
analytic/Hessian/compact/cache/MAP and serial/slab/pencil 1/2/4-rank restart checks.
The Eulerian metric identity rejects pre-fix snapshots; old backend formats and
kinetics retain replay compatibility. In a fixed four-chain field-only probe,
selected-projection bulk ESS improves from 4.5--4.9 to 171--261 with identical
data/objective/gradient bytes. Corrected max Rhat=1.023 and bulk ESS<400 mean
full mixing gates remain unmet; no extension or retuning was done. Next: a
declared longer joint field/scalar pilot, then scientific posterior/refinement
validation. Item 52 remains ongoing.
