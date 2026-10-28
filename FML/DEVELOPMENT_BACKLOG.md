# FML remaining-work backlog

Updated 28 October 2026, including the initial science workflows for items 25–30. This consolidates the original review and subsequent work in this chat. The user intends to work through the full backlog over time. Initial implementations of items 25–30 are now available and tested; KD-tree work remains deferred. Effort estimates are relative; performance gains need measurements. Remaining new science entries are proposals from the original review and need a focused capability check before implementation.

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

| # | Task | Concrete scope | Effort |
|---|---|---|---|
| 1 | Multi-slab spherical-overdensity measurements | Remove the slab-width restriction; preserve particle serialization, periodic/unwrapped distances, centering, velocities, masses and truncation flags. | Medium |
| 2 | Explicit multigrid termination status | Distinguish convergence, maximum iterations and nonfinite failure; return final residual and iteration count. | Small–medium |
| 3 | Faster pair-count rank assignment | Replace repeated rank-load scans with a deterministic min-heap, retaining tie-breaking behavior. | Small |
| 4 | Better pair-count workload estimates | Use neighboring candidate occupancy rather than only same-cell occupancy, especially for cross-catalogues. | Medium |
| 5 | Finish the MPI count audit | Widen/check particle migration counters; validate sheet-interpolation counts/displacements before narrowing; audit remaining catalogue gathers and serialization byte products. | Medium |
| 6 | Faster particle migration | Binary-search slab boundaries; collectively select a nearest-neighbor path for short drifts and retain general routing for long jumps. | Medium–large |
| 7 | Reduce Fourier-resampling communication overhead | Reuse routing, exploit contiguous mode blocks, reduce temporary packing and per-step global round reductions; retain Nyquist and layout correctness. | Medium–large |
| 8 | Distributed pair-count catalogue backend | Owned particles plus ghosts, explicit auto/cross-pair ownership, and no double counting; retain replicated APIs for small catalogues. | Large |
| 9 | Stream binary Gadget field blocks | Bound the keep mask and field-block storage; preserve endian, mixed-species, format-1/2 and MASS-block behavior. | Medium–large |
| 10 | Remove redundant Hessian/tensor copies | Allocate outputs without copying payloads that will be overwritten; share tested symmetric-tensor kernels where practical. | Medium |
| 11 | Flat generic cell storage | Offset/index-based replacement or alternate API for vector-per-cell ParticlesInBoxes; preserve compatibility through views. | Medium–large |
| 12 | Improve polyspectrum shell reuse | Deterministic, cache-aware configuration ordering to reduce shell rebuilds under a bounded cache. | Medium |
| 13 | Tile polyspectrum integration | Accumulate small compatible groups of configurations over spatial tiles; validate changed floating-point summation. | Medium–large |
| 14 | Report total workspace memory | Include input grids, shells, padding, compact/dense outputs, communication scratch and an FFT workspace allowance; distinguish cache budgets from total-memory bounds. | Medium |
| 15 | Extend explicit workspace reuse | Apply consistent reuse/release conventions to LPT, PGD, NUFFT, Hessians and polyspectra, building on FFTLog. | Incremental |
| 16 | Stream MCMC samples and moments | Batched output, online means/variances/covariances, optional sample retention, checkpoint/resume and rejected-step multiplicities. Preserve exact-quantile requirements. | Medium–large |
| 17 | Improve MCMC convergence diagnostics | Add rank-normalized split-Rhat and bulk/tail ESS separately from online moments. | Medium |
| 18 | Consolidate MPI transport contracts | Reuse checked chunking/routing across modules; standardize collective failures, instrument bytes, and reduce per-peer send staging where measurements justify it. | Incremental |
| 19 | Add reproducible performance cases | Track FFT counts, allocations/copies, exchanged bytes, peak RSS, maximum rank time, precision and thread/rank configuration. | Incremental |
| 20 | Update capability documentation | Maintain one capability matrix and consistent C++17/build/API documentation, with links to tested examples. | Small–medium |
| 21 | Stabilize a small Python analysis interface | Expose selected catalogue/statistics workflows with clear ownership and error behavior rather than wrapping every template. | Medium–large |

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
| 31 | Density-split and marked clustering | Environment quantiles and cross-correlations; normalized marks and well-defined noise treatment. |
| 32 | Control variates and ensemble summaries | Matched-phase ZA/2LPT/COLA controls, fitted coefficients and surrogate means, cross spectra, streaming covariance and reproducible metadata. |
| 33 | 3PCF and anisotropic higher-order estimators | Spherical-harmonic neighbor accumulations, then bispectrum multipoles. |
| 34 | Connected four-point statistics | Explicit connected subtraction and supported configurations; start with covariance/collapsed configurations rather than claiming a general trispectrum. |
| 35 | Faster perturbative predictions | FFTLog/FAST-PT-style supported one-loop terms, preserving direct quadrature for unsupported scale-dependent kernels; later IR-resummed biased-tracer RSD templates. |
| 36 | Morphology statistics | Excursion-set volume, surface area, curvature and Euler characteristic, with explicit voxel/boundary conventions. |
| 37 | Integrated lensing observables | Tomographic convergence/shear and projected galaxy–matter correlations, with source distributions and units; begin with Born projections. |
| 38 | Halo histories and external substructure interfaces | Stable particle-ID overlap merger trees and external subhalo catalogue integration. |

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

Items 25–30 now have tested initial implementations. Next options are production validation of these workflows, resuming the engineering queue with item 1, or new science item 31 (density-split and marked clustering). Items 2–5 are good subsequent bounded changes. The larger observational workflow is 25–27; new mock/statistics work can follow with 28–32.

## Short-range gravity extension

| # | Task | Concrete scope | Effort |
|---|---|---|---|
| 46 | Short-range gravity library: P3M and octree TreePM | New `ShortRangeGravity/` module with a consistent Gaussian long-/short-range force split, cosmological normalization, compact gravitational softening and tolerance-controlled cutoff. Implement a direct-sum reference and periodic cell-list P3M backend; validate combined PM/pair forces against independent periodic direct/Ewald results across separations, grid orientations, softening, split scales and cutoffs. Add OpenMP target loops and MPI read-only ghosts spanning any number of slabs, preserving ownership and avoiding periodic duplicates. Then add a dedicated gravity octree with mass moments, conservative opening and cutoff-aware node traversal, validated against P3M. Begin with Newtonian gravity and nonrelativistic massive particles; measure force errors, pair/tree work, ghost memory and rank scaling. | Large |
| 47 | COLASolver short-range gravity integration and subcycling | Depends on item 46. Assemble interpolated long-range PM and short-range particle forces before the existing kick and COLA frame correction; add the short-range force once after PM interlacing and retain the appropriate lightcone potential calculation. Validate standard N-body mode first, then COLA, initially using common timesteps. Add symmetric short-range subcycling between long-range kicks, deriving the cosmological kick/drift factors and COLA frame acceleration throughout the inner steps. Test timestep convergence, halo/orbit dynamics and serial/MPI agreement; define separate prescriptions before enabling screened gravity or particle neutrinos. | Large |

## Phase-space halo finding

| # | Task | Concrete scope | Effort |
|---|---|---|---|
| 48 | Phase-space halo/subhalo finder (ROCKSTAR-style) | Start with spatial FoF parent groups, then adaptive hierarchical refinement in six-dimensional position–velocity space to identify halo/subhalo seeds and assign particle membership. Specify phase-space scaling, peculiar-velocity/Hubble-flow conventions, periodic boundaries and mass definitions explicitly. Add gravitational unbinding, robust centers/bulk velocities, bound masses, Vmax and host/subhalo relationships, with adapters for the existing SO/HOD catalogue APIs. Design MPI ownership and boundary exchange without full particle replication. Validate controlled overlapping halos, major mergers, tidal stripping, resolution and rank/thread consistency against independent references and [ROCKSTAR](https://arxiv.org/abs/1110.4372). Deliver a single-snapshot finder first; add optional temporal consistency through the halo-history work in item 38. Reuse short-range gravity primitives from item 46 where appropriate for unbinding. | Very large |
