# FML remaining-work backlog

Updated 29 October 2026, including completed Engineering items 1–7, 10, 14, 17 and 19–21, and tested initial implementations of items 25–32, 35–36 and 38. This consolidates the original review and subsequent work in this chat. The user intends to work through the full backlog over time. These initial implementations and their remaining production-validation work are recorded below; KD-tree work remains deferred. Effort estimates are relative; performance gains need measurements. Remaining new science entries are proposals from the original review and need a focused capability check before implementation.

## Completed in this sequence

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

These completions do not cover all the follow-ups below. Other ongoing COLASolver/transfer-function edits are outside this backlog update.

## Survey workflow status

Items 25–27 have initial implementations in `Survey/SurveyPower.h`, `Survey/SurveyWindow.h` and `Survey/SurveyReconstruction.h`. The standalone example and tests are in `Survey/workflow_example/` and `Survey/tests/`. Read `Survey/README.md` for conventions and limits. Independent DFT, explicit random-pair, analytic Gaussian convolution and manufactured displacement tests pass in serial, float/OpenMP, sanitizer and MPI builds. Production mock recovery, external pypower/pyrecon numerical comparisons, and application-specific mesh/padding/binning convergence remain follow-up validation work.

## HOD and spatial-statistics status

Items 28–30 have initial implementations in `HOD/HOD.h` and `CountsInCells/SpatialStatistics.h`. Tests and the HOD-to-statistics example are under those module folders. Both build without FFTW/GSL. HOD supports conditional/independent central-satellite occupations, prepared halo/member data, NFW or particle satellites and stable identity pairs. Spatial summaries use the existing KD-tree with replicated ordered tracers and distributed query samples; they do not implement the deferred KD-tree optimizations. Unit-weight counts, rectangular nonperiodic full-cell windows and periodic spherical cells are supported. Calibrated occupation fits, dedicated assembly-bias prescriptions, weighted estimators, arbitrary-mask boundary corrections and distributed tracer searches remain extensions. See each module README for the exact conventions.

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
| 25 | Survey Fourier multipoles | FKP data/random fields, observer position, local-LOS Yamamoto estimator, normalization and weighted shot noise. |
| 26 | Survey-window modeling | Window multipoles/mixing matrices and integral-constraint treatment; keep survey windows distinct from assignment-window deconvolution. |
| 27 | Complete survey BAO reconstruction | Data/random displacement APIs, RecIso/RecSym, local LOS, masks/selection and reconstructed statistics. |
| 28 | HOD galaxy mocks | Central Bernoulli and satellite Poisson occupations, NFW/particle satellites, halo-ID-keyed randomness; then velocity, concentration and assembly-bias extensions. |
| 29 | kNN-CDF statistics | Batched random query points, distributions of kth-neighbor distances and explicit self-exclusion conventions. |
| 30 | Counts-in-cells | Counts/density PDFs and factorial moments, with explicit window and shot-noise conventions. |
| 31 | Density-split and marked clustering (initial implementation) | Environment quantiles and cross-correlations; normalized marks and well-defined noise treatment. |
| 32 | Control variates and ensemble summaries (initial implementation) | Matched-phase ZA/2LPT/COLA controls, fitted coefficients and surrogate means, cross spectra, streaming covariance and reproducible metadata. |
| 33 | 3PCF and anisotropic higher-order estimators | Spherical-harmonic neighbor accumulations, then bispectrum multipoles. |
| 34 | Connected four-point statistics | Explicit connected subtraction and supported configurations; start with covariance/collapsed configurations rather than claiming a general trispectrum. |
| 35 | Faster perturbative predictions (initial implementation) | FFTLog/FAST-PT-style supported one-loop terms, preserving direct quadrature for unsupported scale-dependent kernels; later IR-resummed biased-tracer RSD templates. |
| 36 | Morphology statistics (initial implementation) | Excursion-set volume, surface area, curvature and Euler characteristic, with explicit voxel/boundary conventions. |
| 37 | Integrated lensing observables | Tomographic convergence/shear and projected galaxy–matter correlations, with source distributions and units; begin with Born projections. |
| 38 | Halo histories and external substructure interfaces (initial implementation) | Stable particle-ID overlap merger trees and external subhalo catalogue integration. |

The scientific implementations should be checked against independent analytic or established reference results, as well as serial/MPI consistency. Existing periodic spectra, reconstruction building blocks, raw shell fourth moments, halo models and lensing/lightcone utilities should remain the starting point.

## Conditional work: measure first

| # | Task | Condition for proceeding |
|---|---|---|
| 39 | SIMD/SoA pair-count specialization | Standard weighted-count hot loops show a worthwhile bottleneck; preserve the generic callback path. |
| 40 | Multigrid coarse-level rank agglomeration | Coarse levels are demonstrably dominated by communication or underused ranks. |
| 41 | Overlap multigrid interior work and halos | Profiles justify it; preserve color-sweep communication dependencies. |
| 42 | Scoped mixed precision | Memory pressure justifies float field storage with double accumulation; validate forces, displacements, higher moments, thresholds and MG residuals. |
| 43 | Pencil-decomposed or GPU FFT backend | Production runs are slab/FFT limited; plan for deposition, particles, ghosts and indexing changes too. |
| 44 | DTFE output/boundary memory improvements | Profile requested-output allocation and boundary traffic; consider batched boundary fields. |
| 45 | Broader distributed snapshot IO | Benchmark binary/HDF5/Ramses read patterns and peak memory first, then choose collective or chunked improvements. |

Items 25–30 now have tested initial implementations. Next options are production validation of these workflows, resuming the engineering queue with item 1, or new science item 33 (3PCF and anisotropic higher-order estimators). Items 2–5 are good subsequent bounded changes. The larger observational workflow is 25–27; new mock/statistics work can follow with 28–32.

Items 32 and 36 now also have tested initial implementations. `EnsembleStatistics/` provides streaming covariance, independently trained full/diagonal control variates, independent surrogate-mean uncertainty, phase provenance and a matched Fourier auto/cross-spectrum adapter. `Morphology/` provides distributed closed-voxel excursion volume, surface area, integrated mean curvature and Euler characteristic, with open/periodic boundaries and separate Crofton normalization. Both include runnable examples and analytic/independent-reference serial/MPI tests. Production follow-ups are matched ZA/2LPT/COLA versus N-body calibration and continuum Gaussian-field morphology/resolution validation; see each module's README. Items 33–34 and 37 remain pending.

Item 31 now has an initial periodic-box implementation in `Environment/`: top-hat counts/overdensities, deterministic equal-count density quantiles and query–tracer radial cross-correlations, plus one-pass normalized marked correlations with explicit sample-mean and finite-population pair conventions. It reuses the existing KD-tree/counts-in-cells and pair-counting engines and includes direct-enumeration, analytic, serial/MPI tests and a runnable example. Density noise and finite query sampling are retained explicitly; survey masks/weights, angular estimators and production noise/resolution validation remain follow-ups. See `Environment/README.md`.

Items 35 and 38 now have tested initial implementations. `PerturbationTheory/` provides EdS matter P22/P13 using FFTLog/FFT convolution with mandatory direct-reference spot checks over an explicit output range, plus independent direct quadrature for custom scale-dependent kernels. `HaloHistory/` provides sparse particle-ID overlap candidates, deterministic merger forests, optional bounded snapshot gaps, host/subhalo relationships, a versioned neutral catalogue format and explicit HOD adaptation. Both include examples, reference tests and documented limits. Expanded/IR-resummed bias/RSD templates, broader cosmological validation, native finder-specific readers, distributed halo matching and advanced core/trajectory-based tree repairs remain follow-ups.

## Short-range gravity extension

| # | Task | Concrete scope | Effort |
|---|---|---|---|
| 46 | Short-range gravity library: P3M and octree TreePM | New `ShortRangeGravity/` module with a consistent Gaussian long-/short-range force split, cosmological normalization, compact gravitational softening and tolerance-controlled cutoff. Implement a direct-sum reference and periodic cell-list P3M backend; validate combined PM/pair forces against independent periodic direct/Ewald results across separations, grid orientations, softening, split scales and cutoffs. Add OpenMP target loops and MPI read-only ghosts spanning any number of slabs, preserving ownership and avoiding periodic duplicates. Then add a dedicated gravity octree with mass moments, conservative opening and cutoff-aware node traversal, validated against P3M. Begin with Newtonian gravity and nonrelativistic massive particles; measure force errors, pair/tree work, ghost memory and rank scaling. | Large |
| 47 | COLASolver short-range gravity integration and subcycling | Depends on item 46. Assemble interpolated long-range PM and short-range particle forces before the existing kick and COLA frame correction; add the short-range force once after PM interlacing and retain the appropriate lightcone potential calculation. Validate standard N-body mode first, then COLA, initially using common timesteps. Add symmetric short-range subcycling between long-range kicks, deriving the cosmological kick/drift factors and COLA frame acceleration throughout the inner steps. Test timestep convergence, halo/orbit dynamics and serial/MPI agreement; define separate prescriptions before enabling screened gravity or particle neutrinos. | Large |

## Phase-space halo finding

| # | Task | Concrete scope | Effort |
|---|---|---|---|
| 48 | Phase-space halo/subhalo finder (ROCKSTAR-style) | Start with spatial FoF parent groups, then adaptive hierarchical refinement in six-dimensional position–velocity space to identify halo/subhalo seeds and assign particle membership. Specify phase-space scaling, peculiar-velocity/Hubble-flow conventions, periodic boundaries and mass definitions explicitly. Add gravitational unbinding, robust centers/bulk velocities, bound masses, Vmax and host/subhalo relationships, with adapters for the existing SO/HOD catalogue APIs. Design MPI ownership and boundary exchange without full particle replication. Validate controlled overlapping halos, major mergers, tidal stripping, resolution and rank/thread consistency against independent references and [ROCKSTAR](https://arxiv.org/abs/1110.4372). Deliver a single-snapshot finder first; add optional temporal consistency through the halo-history work in item 38. Reuse short-range gravity primitives from item 46 where appropriate for unbinding. | Very large |

Item 46 now has an initial implementation in `ShortRangeGravity/`: Gaussian/spline force kernels, a long-range PM filter and explicit FML normalization, direct and cell-list backends, a conservative monopole octree, and chunked multi-slab MPI ghosts. See `ShortRangeGravity/README.md` for contracts and validation. This is a force-library implementation; production mesh/interpolation convergence, clustered MPI performance and tree error-distribution validation remain follow-ups. Item 47 remains pending.

## Wave dark matter and gas dynamics

| # | Task | Concrete scope | Effort |
|---|---|---|---|
| 49 | Spectral solver for ultralight axion / fuzzy dark matter | Nonrelativistic Schrödinger–Poisson evolution with full complex wavefunction storage and complex-to-complex FFTs, symmetric split-step kinetic/potential evolution and a periodic Poisson solve. Specify axion mass, wavefunction/density normalization, units and zero-mode conventions. Start in a non-expanding periodic box; then derive cosmological variables/time factors and initial conditions, and add optional CDM/particle-gravity coupling. Reuse FFT workspaces, support MPI complex-field layouts and versioned checkpoints, and measure memory/FFT costs. Validate free-wave dispersion, interference, norm conservation, static energy, a stationary soliton reference and timestep/de Broglie-resolution convergence; derive an appropriate cosmological energy diagnostic. Treat self-interactions and isolated boundaries as later explicit extensions. Reference: [PyUltraLight](https://arxiv.org/abs/1807.04037). | Very large |
| 50 | Smoothed particle hydrodynamics (SPH) | A conservative gas-particle formulation with kernel density estimates, adaptive smoothing lengths and consistent grad-h terms, pressure forces, energy/entropy evolution and an EOS interface. Add controlled artificial viscosity/conductivity, shock/shear switches and Courant/acceleration timestep criteria. Begin with adiabatic ideal gas. Design symmetric variable-support neighbor interactions, reusable per-thread buffers and multi-slab MPI ghosts without full particle replication; reuse existing spatial backends without resuming deferred KD-tree optimizations. Couple gas/collisionless gravity with explicit cosmological source and time conventions. Validate uniform gas, sound waves, Sod shocks, Sedov blasts, shear/contact tests and gravitational collapse, including conservation and resolution/rank/thread convergence. Share gas/EOS interfaces with item 51. Reference: [conservative SPH](https://arxiv.org/abs/1007.1245). | Very large |
| 51 | Riemann solvers and conservative finite-volume hydrodynamics | Reusable Euler primitives: primitive/conserved conversions, EOS and sound-speed interfaces, Rusanov/HLLE baselines, HLLC fluxes and an exact 1D reference where applicable. Specify wave-speed estimates, entropy/vacuum treatment and positivity-preserving fallback behavior. Add limited reconstruction, second-order Godunov evolution, CFL control and periodic/outflow/reflecting boundaries, progressing from 1D verification to conservative multidimensional finite-volume evolution. Include high-Mach/dual-energy handling, MPI halos and explicit gravity/cosmological source coupling. Check common gas conventions against SPH; add thermal physics or meshless/Godunov variants only as separately validated extensions. Validate shock tubes, contacts, rarefactions, blasts, smooth advection/sound waves and self-gravitating/cosmological gas, measuring conservation and resolution/rank/thread convergence. Reference: [HLLC construction](https://doi.org/10.1007/BF01414629). | Very large |

## Field-level forward modeling and inference

| # | Task | Concrete scope | Effort |
|---|---|---|---|
| 52 | Field-level forward model and Bayesian reconstruction (LEFTfield/BORG-style) | First expose a deterministic, reproducible map from whitened initial conditions and cosmological/bias parameters to matter/tracer fields. Reuse LPT initially, with a later PM/COLA option; specify cutoffs, assignment/dealiasing, growth, bias operators, stochasticity and real-/redshift-space conventions. Add survey observation operators for masks/selection, with distinct count/field likelihoods and explicit noise/normalization assumptions. Implement tangent/adjoint or equivalent differentiable operations through initial-field generation, evolution, deposition, bias and RSD; verify finite-difference directional gradients, adjoint dot products, zero/Hermitian modes and MPI consistency. Keep the forward/likelihood layer separate from inference: use MAP for debugging, then add prior-whitened field HMC, nuisance/cosmological inference, checkpointing and memory-controlled recomputation. Validate fixed-initial-condition predictions, mock-truth recovery, posterior predictive behavior and coverage. Distinguish LEFTfield-like perturbative bias modeling from BORG-like initial-condition posterior sampling; neither name implies native compatibility or a complete reimplementation. References: [LEFTfield](https://arxiv.org/abs/2409.10937), [BORG](https://arxiv.org/abs/1203.3639). | Very large |

The existing FieldLevel BAO prototype's review follow-ups are deferred at the user's request. See [FieldLevel/KNOWN_ISSUES.md](FieldLevel/KNOWN_ISSUES.md) for MPI/input safety, portable CIC gradient verification, scientific validation and measured workspace reuse. The frozen-chain diagnostics issue was resolved by Engineering #17; the other entries remain open. These follow-ups do not mark item 52 complete.

Items 49–52 are newly requested future work, not implementations or a change to current priorities. Start each with a bounded verification stage, share gas primitives across 50–51, and reuse existing gravity, LPT/PM, survey and statistical modules where their contracts apply.

Item 48 has an initial independent single-snapshot implementation in `PhaseSpaceHaloFinder/`: adaptive six-dimensional hierarchy and seed merging, isolated Newtonian unbinding, exclusive/inclusive bound masses, resolved circular-velocity estimates and a documented bound-extent host rule. Distributed parent FoF labels converge across periodic slabs, complete parents route to one owner without full particle replication, and particle assignments return to input ranks. SO seeds, explicit HOD mass/radius conversion and exclusive halo-history adapters are available. See its README for conventions and limits. A controlled external ROCKSTAR Plummer comparison passes; cosmological catalogue comparisons, production completeness/purity calibration, temporal stabilization and stronger halo-boundary definitions remain follow-ups; this is not a drop-in ROCKSTAR replacement.

## Survey mock catalogue generation

| # | Task | Concrete scope | Effort |
|---|---|---|---|
| 53 | Survey mock generator (MAKE_SURVEY-style) (phase 1 done: `Survey/SurveyMocks.h` snapshot projection, footprints by callback/RA-Dec box, n(z) selection, randoms, tests; next: remapping, Healpix, MANGLE, lightcone input, executable, end-to-end example) | Build a reusable library API plus a configuration-driven executable taking periodic particle/halo/galaxy catalogues or existing lightcone output to observed mock catalogues. Start with a snapshot projection, explicitly distinct from an evolving lightcone. Add optional volume-preserving cuboid remapping, observer translation/rotation, Cartesian-to-RA/Dec conversion, cosmological distance-to-redshift inversion and radial peculiar-velocity RSD with explicit velocity/length/redshift conventions. Apply angular footprints/completeness (callback/HEALPix and optional supported MANGLE polygons), redshift cuts and radial selection with a specified n(z) or dN/dz convention, correct volume Jacobian and feasible downsampling checks. Generate matching unclustered random catalogues; distinguish selection probabilities, completeness corrections and FKP weights so selection is not applied twice. Reuse HOD, COLASolver lightcones, survey coordinate utilities and Survey analysis/catalogue interfaces. Stream/chunk large catalogues, preserve source/replica IDs and use ID-keyed randomness for reproducibility across rank/thread counts; document repetitions from tiling/remapping. Validate coordinate round trips, zero-velocity and full-sky limits, footprint area, recovered radial density/completeness, random normalization, independent mock clustering and MPI consistency. Deliver an end-to-end example feeding items 25–27. Reference: [cmcbride/make_survey](https://github.com/cmcbride/make_survey). | Large |

Item 53 is future work, not an existing complete pipeline or a change to current priorities. Existing pieces are `HOD/` (galaxy phase-space population), `COLASolver/src/Lightcone/` (evolving lightcones, periodic replicas and observed-redshift support), `Survey/GalaxiesToBox.h` (RA/Dec/redshift to Cartesian, the opposite direction to mock sky projection), and `Survey/` power/window/reconstruction routines (which consume data/random catalogues). These do not yet provide a general MAKE_SURVEY-style box-to-selected-survey executable with remapping, footprint/radial selection and matching random generation. Gravitational redshift or other relativistic observing effects should be explicit optional extensions when using compatible lightcone fields.

## Glass initial conditions

| # | Task | Concrete scope | Effort |
|---|---|---|---|
| 54 | Glass particle loads and IC integration | Generate periodic, statistically isotropic glass particle loads from reproducible seeded positions using repulsive gravity and controlled damping/relaxation. Reuse the gravity/particle infrastructure with explicit force-sign, softening, timestep and stopping conventions; track residual forces/velocities and convergence. Provide stable particle IDs, versioned load output/readback, and MPI/OpenMP support without full particle replication. Apply LPT displacements and velocities at irregular glass positions through the shared IC/transfer-provider interfaces, preserving normalization, units and particle species conventions. Keep this distinct from existing external-IC inversion routines that assume an initially regular grid. Validate low-k discreteness suppression, isotropy, pair statistics, residual forces, resolution/relaxation convergence, and thread/rank consistency; compare displaced glass and lattice starts against the same target spectrum. Treat multiple-species quiet loads and overlap avoidance as explicit later extensions. | Medium–large |

Item 54 is future work. Glass-load generation and applying cosmological perturbations to that load are separate stages; existing grid-only external-IC reconstruction is not evidence of glass support.
