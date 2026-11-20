# Citation and documentation audit, 20 November 2026

Scope: the 120 commits in `1c32171..3d63d15`, including the intermediate citation update
`fe7e245`, and the current implementation of their public features. The paths in the user request
resolve to `Documentation/` and `Documentation/website/` in this checkout. This is a history,
method-lineage and documentation audit, not a new scientific validation campaign.
In-progress work outside that committed range (including the untracked Bacco module) is excluded.

## Corrections made

- ReACT: added direct ACTio-ReACTio repository and four framework-paper links in the reaction,
  screened-collapse and one-loop headers, and ReACT v2 to the book/website. The pinned comparison
  revision is retained. Coupled-quintessence/FDS is explicitly an FML extension, not an upstream calibration.
- e-MANTIS: added the missing book section, nonlinear-option rows, website API/range summary,
  citations to the power/HMF papers and software repository/DOI, and an index entry. Corrected the
  SimulationCode chapter's obsolete fitting-function description. Documented the different high-z
  behavior and cosmology-range policies of SimulationCode and the Boltzmann solver.
- Cosmic shear: documented the new tomographic API, NLA, source/shear systematics, optional
  scale-independent Sigma(z), Hankel transforms and COSEBIs. Added alignment/COSEBI/KiDS papers
  and CCL/CosmoSIS method/comparison credits in code, module docs, book and website.
- AxionHMcode: corrected a contradictory website claim that it was unavailable as a solver nonlinear option.
- Native Delaunay: linked Bowyer/Watson, Shewchuk paper/reference code, Devillers--Teillaud and CGAL
  in source, method notes, bibliography and website. Existing rounding/certification limitations remain authoritative.
- AdaptiveMPIGrid: linked the NBX paper/author copy in code, README and website; completed the
  existing bibliography entry with the DOI. No new large-machine scaling claim is made.
- Sampling: linked Lewis/CosmoMC for fast/slow proposals and described the new slow-coordinate-only
  update in the book and website. Added the replica-exchange method reference to the serial prototype.
- FieldLevel: replaced the outdated pending-calibration description with the completed 2/8 pilot result,
  retained subsequent mixing failures, and recorded Box-1 observation preparation and its external dependencies.
  A successful local numerical check is not described as posterior calibration.

## Feature coverage map

Entries below identify where the additions in the reviewed history are documented. Existing
citations were retained when already present; implementation optimizations and bug fixes do not
receive invented scientific-method citations.

| Recent feature family | Library source / authoritative notes | Book / website coverage |
| --- | --- | --- |
| Generic grids, domains, pencils, complex grids and distributed particle assignment | FFTWGrid, PencilDomain, PeanoHilbertDomain, MPIParticles, Interpolation | Parallelism, grids/FFTs, particles, mass assignment; pencilgrid/mpiparticles/interpol |
| Adaptive leaves, NBX exchange, ownership and AMR multigrid | AdaptiveMPIGrid and MultigridSolver/AMR notes | Multigrid chapter; amr/multigridsolver |
| Conservative SPH and finite-volume Euler | SPH and Hydrodynamics READMEs: Price, HLLC and Clawpack method references | Gas/wave-dark-matter chapter; hydro |
| AMRCode symmetron/dilaton, safeguarded cell solves | AMRCode/README.md and multigrid model headers | Modified-gravity simulations, multigrid; amr |
| Reduced Horndeski frontend, K-mouflage and screening models | Cosmology solver OPTIONS/README, SimulationCode/MODELS.md | Extended models, modified-gravity simulations; cosmology/nbodycode |
| Fading Dark Sector, coupled collapse/reaction and simulation comparisons | Cosmology, HaloModel and SimulationCode model/test notes | Background, halo model, modified-gravity simulations; cosmology/halomodel/nbodycode |
| e-MANTIS prediction port, f(R) nonlinear solver and simulation diagnostics | Emulator/EMantis, Spectra/NONLINEAR.md, SimulationCode/ANALYSIS.md | Fitting/emulators and SimulationCode chapters; cosmology/halomodel/nbodycode |
| Native tessellation, updates, DTFE, watershed, ZOBOV and ORIGAMI consumers | Triangulation/DELAUNAY.md, CosmicWeb and VoidFinders | Voids/tessellations, cosmic web; tesselation/dtfe/watershed/voidfinders/cosmicweb |
| DESI ShapeFit/full shape, Planck PR4 CamSpec/low-ell, ACT lensing, SN and KiDS data | CosmologyFitting/README.md and likelihood source provenance | Sampling/fitting chapter; inference/index |
| Tomographic cosmic shear, NLA and COSEBIs | WeakLensing/README.md, CosmicShear.h | Weak-lensing chapter; inference/index |
| Eulerian/PM forward models and adjoints, compact meshes, NUFFT deposition, likelihoods | FieldLevel/docs guides, numerical tests and preserved validation records | Field-level chapter; fieldlevel |
| MAP/L-BFGS, checkpoints, HMC metrics, tempering, prior sensitivity and Box-1 ingestion | FieldLevel/docs/VALIDATION.md and per-study frozen reports | Field-level chapter; fieldlevel (updated bounded results) |
| Survey inference and restart/target identities | FieldLevel/docs/SURVEY.md | Field-level/survey chapters; fieldlevel/survey |
| Hybrid bias in redshift space | ComputePowerSpectra/HybridBias* and SimulationCode analysis notes | Power spectra, SimulationCode; perturbationtheory/nbodycode |
| Shell overlap anti-aliasing and precision fixes | ComputePowerSpectra/ShellOverlap.h, upstream CONCEPT lineage | Power spectra; statistics/nbodycode |
| Images of grids/particles and snapshot integration | Visualization, Analysis and SnapshotAnalyzer notes | Utilities and snapshot analysis; other/snapshotanalyzer/nbodycode |
| Lightcone FoF halos | SimulationCode lightcone/output parameters | SimulationCode; nbodycode |
| Fast/slow proposal correction, build isolation and single-precision fixes | MCMC/Sampler.h; build configuration and tests | Sampling/fitting update; remaining items are engineering changes |

## Checks and limits

- The LaTeX book rebuilds with latexmk/BibTeX/makeindex: 257 pages, no undefined citations,
  unresolved references or overfull boxes. Three existing hyperref bookmark warnings about
  math in a chapter heading remain; these do not affect typeset text or citation resolution.
- Rendered PDF spot checks cover the e-MANTIS section/example, cosmic-shear equations/validation,
  and the continued field-level validation box. No clipping/overlap was observed in those pages.
- PHP 8.6.0RC4 syntax checks pass for all 85 PHP files. Static checks pass for 621 local links
  and anchors, 274 bibliography entries (unique keys), and all 264 cited keys.
- A local PHP preview loaded the e-MANTIS section and references in the browser accessibility tree.
  A subsequent browser screenshot was unavailable because the tab session expired; no full visual
  website or responsive-layout validation is claimed.
- All nine changed C++ headers contain only comment/whitespace changes relative to the reviewed HEAD.
  Git-aware clang-format on changed ranges passes with the repository `.clang-format` (available
  version 17.0.6); whole-file checking finds pre-existing formatting differences, left outside this audit.
  Numerical/MPI regressions were not rerun for this comment/documentation-only change.
- Primary paper/repository records were checked for the new citations, including the ReACT upstream
  and papers, e-MANTIS papers, alignment/COSEBI/KiDS papers, Lewis, replica exchange, Bowyer/Watson,
  Shewchuk and NBX. The e-MANTIS GitLab URL did not yield a usable page through the web reader;
  its repository link follows the port's existing provenance and the software DOI remains available.
- The generated PDF and temporary build/preview files are local build products, not source changes.
  Concurrent changes to emulator code/licensing, Bacco, CHANGELOG/CAPABILITIES and unrelated
  untracked files were not included in this task's changes.

## Reviewed commit manifest

```text
3d63d15 ReACT: one collapse per redshift for coupled dark matter (halo models 8.2 s -> 0.9 s)
f60c002 Build: never reuse objects found in other folders through VPATH
67cb744 Advance beyond2pt inference studies and prepare Box-1 catalog
ce81da3 CI: ZOBOV CGAL comparison with a wide enough buffer; allow the e-MANTIS wCDM data
552dedb MCMC: the slow update draws only the slow coordinates (faster mixing with many fast parameters)
cef2fe2 EBS: nonlinear = emantis for f(R) (e-MANTIS boost times GR)
8c9eb31 SimulationCode: e-MANTIS instead of the fitting function for the f(R) nonlinear P(k)
2fe9fef Emulator/EMantis: the e-MANTIS emulators in C++
7f953f5 Move the native Delaunay triangulation into Triangulation; USE_CGAL off everywhere
3d8b298 Native tessellation by default for DTFE, ZOBOV and ORIGAMI halos; CHANGELOG
86e0557 Native tessellation: 30% less memory for volumes and neighbours, same speed
579efb2 Native tessellation: no ghost blow-up for clustered points, 2.5x less memory
c604c51 DTFE on FML's own tessellation (no CGAL needed)
00ba667 Documentation: native Delaunay triangulation, CGAL optional, ReACT for coupled DM, FDS simulations and fits
ae28cb8 SimulationCode: FDS vs LCDM simulations against nonlinear = react
b8e9e1f ORIGAMIHalos on FML's own tessellation (no CGAL needed)
3d15916 Watershed and ZOBOV on FML's own tessellation (no CGAL needed)
ca3ea17 ReACT: halo-model reaction for coupled quintessence and the Fading Dark Sector
0c93c93 Experimental/Delaunay: compare with the library's MPIPeriodicDelaunay
0e51a58 Experimental/Delaunay: update the tessellation as points move
07220ae Experimental/Delaunay: MPI periodic Voronoi with certified ghosts, deterministic geometry
2e3d922 Experimental/Delaunay: native Delaunay triangulation and periodic Voronoi tessellation (2D/3D)
b6fe7b7 CosmologyFitting: check of EdS against exact-time two-fluid loop kernels for the FDS
a764c40 Add field-level prior sensitivity and calibration pilot controls
62d0b68 CosmologyFitting: DESI DR1 full shape, Planck PR4 CamSpec + low-ell, ACT DR6 lensing, DES-Dovekie, FDS fits
6e2c468 Tests: precision-aware tolerances; nightly single-precision job
ef24165 Add field-level tempering and independent-mock validation
ec5ca90 Merge branch 'FMLDevelop' of github.com:HAWinther/FML into FMLDevelop Merged already fixed single precision bug
a2b0f46 Fix the build with SINGLE_PRECISION_FFTW
c6bfad5 Merge pull request #11 from jbb42/fix-single-precision-shell-weights
81abefe Fix the single-precision build: shell weight in the grid's precision
406ed84 NBody: warn about the lattice self-force on finer force grids; example uses the two-point gradient
f5e4018 QuickStart: three short checked examples; on the website and in the README
fd4e46f Documentation: what FML is and the ideas behind it
fccad73 AMRCode: GSL random generator (as N-GenIC) by default
266ecf8 CosmologyFitting: BOSS/eBOSS RSD, DESI DR1 ShapeFit, KiDS-1000 COSEBIs, Brans-Dicke fits
8e2009c WeakLensing: tomographic cosmic shear (CosmicShear.h)
76e5407 Add Eulerian field-level inference and correct Gaussian field metric
c00f8cb SimulationCode: friends-of-friends halos on the lightcone (plc_halos)
e741c24 Documentation: the Fading Dark Sector in the book and on the website
cba043f Added front image to doc
24f5456 Added front image to doc
bb24789 Dilaton cell solve: safeguarded Newton (fixes non-converging AMR solves)
4c2374d Fading Dark Sector (arXiv:2611.17422) in the EBS and SimulationCode
b928b04 Organize FieldLevel and archive generated validation runs
c24a2d1 Hybrid bias spectra in redshift space; docs for it and the AMR dilaton
c7a9fe3 AMR multigrid: inject the coarse value into cells on both levels
e36bf61 Dilaton gravity on the AMR multigrid and in AMRCode
45b5c45 Updated README
5164e7a EBS tests: exact ReACT = HMcode in massless LCDM; regression tolerance 1e-6
72b4d16 Add bounded TSC posterior pilot and small-mesh assignment lookup
c7e8758 Validate joint TSC field, amplitude and noise MAP reconstruction
b3750a7 Add stable PM objective comparisons, diagnostics and opt-in TSC assignment
10d371c Add value-only MAP line searches and optional L-BFGS optimizer
7ac44a6 Documentation: AMRCode, symmetron, images, Visualization, shell overlap weights
56861b4 Validate compensated PM reconstruction against refined mock data
ed6bb2b AMRCode: symmetron gravity and less memory
293cc75 Add opt-in PM force compensation and fixed observation responses
a792eca Images of fields at the outputs (SimulationCode and SnapshotAnalyzer)
5dbff72 Polyspectra: optional shell overlap weights (shell anti-aliasing)
6cbfd3b FieldLevel: establish particle-sampled PM force reference
5e1ad26 New module: Visualization (PNG pictures of grids and particles)
a2093a9 FieldLevel: support dense PM sampling and expand accuracy validation
2770759 README: describe what the library is
e46b1b6 Website for FML 2.0; make target for the AMR symmetron example
e6b642c Updated README
53fc1f9 CI: do not run for documentation-only changes
df90883 CI badge: the status of master; do not cancel CI runs on master
9562288 Updated README
ec9fd48 README: contributors
85a92e0 Updated README
bcdb993 README: this is FML 2.0; how to get version 1.0
c0ad3a6 Merge master (GPL licence PR #9, header comments PR #6) into FMLDevelop
6717a0a CI: no tracked file over 1 MB outside an allow-list
25abaeb Add the GPL 3.0 licence (LICENCE)
c3c7941 Track the website; collect the documentation in Documentation
2c245e8 Add the FML book (LaTeX source) in Documentation
4926dad Examples make their own input: remove the TestData submodule
9f8f20c Add PM inference restart and document spatial convergence limits
bc2acd5 Faster CI; AMRCode Makefiles; fix the flaky HDF5 voids check
66153aa Fix bugs and speed up interpolation (library review part 3a)
140ad89 Validate full-field PM posterior mixing and predictive uncertainty
dc73b48 Add AMRCode (AMR N-body code), AMR symmetron solver and faster AMR multigrid
7ea7c4e Validate bounded PM HMC against quadrature and distributed replay
57eac6e Add distributed PM forward model, discrete adjoint and periodic MAP inference
e23c223 Density assignment from particles on any domain to slabs and pencils (as GADGET's PM)
cae2541 Correct NUFFT odd-grid redistribution test expectations
ce45169 Support pencil-decomposed grids throughout distributed FieldLevel
291cb1d AMR multigrid solver on AdaptiveMPIGrid, with DGP
fdb452e Fix bugs from the library review (parts 1 and 2)
208348f Pencils: documentation cleanup
87dee06 tests: build rules for the grid, domain and pencil tests
26ba25e Analysis and SnapshotAnalyzer on pencils (make USE_PENCIL_GRID=true)
e360dae SimulationCode on pencils (make USE_PENCIL_GRID=true)
727cd10 WaveDarkMatter on CFFTWGrid and CPencilGrid
502534b GADGET, HDF5 and RAMSES readers for any particle domain
ff3638f Real-space modules on any grid and domain
ce76485 Fourier-space and LPT modules on any grid
c074a18 Generic grid interface: grids on any domain, PencilGrid and complex grids
e3fd3e3 AdaptiveMPIGrid on a given domain; PencilGrid: fast transposes, planner flag, benchmark
ced33ee FML's communicator for the Peano-Hilbert domain and AdaptiveMPIGrid; particles follow the AMR grid
1c5e766 PencilGrid<N> with the interface of FFTWGrid; PencilDomain is only the pencils
71947f5 Domain (Slab/Pencil/Peano-Hilbert) and MPIParticles creation by the domain
a21bdec AdaptiveMPIGrid: sparse (NBX) irregular exchange for ghost setup and migration
a619b9b AdaptiveMPIGrid: non-blocking ghost exchange with interior/boundary split
78def61 Add AdaptiveMPIGrid: MPI-distributed adaptive grid on a Peano-Hilbert curve
223fd14 Add finite-volume Euler hydrodynamics and Riemann solvers
3b6dd0d Add conservative adaptive SPH gas dynamics and validation
26a5e9b Validate survey reconstruction and reject nonfinite proposals
3ef0f1d EBS: symmetron, dilaton and (m(a), beta(a)) models
e9e37f9 K-mouflage through the generic Horndeski route (as Hi-COLA)
aec521d Add survey checkpoint restart and harden target identities
f20267f EBS: reduced Horndeski from its Lagrangian (horndeski_table)
c5f86be Add Cartesian survey likelihoods and nonlinear MPI validation
4b9e9e7 Strengthen BAO gradient portability and inference validation
fe7e245 Document references for recent scientific methods and code lineage
80e3d53 Merge pull request #6 from hersle/skip_header_comments
57cd589 Merge pull request #9 from GarethCabournDavies/FML/license
d08c890 Propose GPLv3 licensing for FML source code
368b25b Use negative nskip to skip arbitrary number of header comment lines
```
