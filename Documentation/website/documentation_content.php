<hr/>
<header class="entry-header text-center">
  <h1 class="entry-title">Table of contents</h1>
</header>
<hr/>

<p>
<a href="index.php">Back to the main page</a>
</p>

<p>
Each page below explains the theory behind the methods and shows how to use them. Every module folder also has a README with the full conventions and the validation, an <tt>example</tt> folder and (for most) a <tt>tests</tt> folder; <tt>FML/CAPABILITIES.md</tt> lists them all and <tt>FML/BUILDING.md</tt> explains the build options. For every class and method see the <a href="doxygen/index.html" style="display: inline">doxygen documentation</a>. As always: test things before using them for something new.
</p>

<p class="page-toc">
  The building blocks of the library:
  <a href="global.php">Global (FML)</a><br>
  <a href="fftwgrid.php">Grid for Fourier transforms (FML::GRID::FFTWGrid)</a><br>
  <a href="pencilgrid.php">Pencil grids, complex grids and running FML on pencils (FML::GRID::PencilGrid)</a><br>
  <a href="mpigrid.php">General grid (FML::GRID::MPIGrid)</a><br>
  <a href="mpiparticles.php">Particle container (FML::PARTICLE::MPIParticles)</a><br>
</p>

<p class="page-toc">
  Grids, fields and solvers:
  <a href="interpol.php">Grid assignment, interpolation and NUFFT (FML::INTERPOLATION)</a><br>
  <a href="gridop.php">Grid operations: smoothing, whitening and convolutions (FML::GRID)</a><br>
  <a href="randomfield.php">Gaussian and non-Gaussian random fields (FML::RANDOM)</a><br>
  <a href="multigridsolver.php">Solving general PDEs (FML::SOLVERS::MULTIGRIDSOLVER)</a><br>
  <a href="amr.php">Adaptive mesh refinement: AdaptiveMPIGrid, AMR multigrid (Poisson, DGP, symmetron) and AMRCode</a><br>
  <a href="tidaltensor.php">Hessians and tidal tensor eigenvalues (FML::HESSIAN)</a><br>
</p>

<p class="page-toc">
  Cosmological theory:
  <a href="cosmology.php">Einstein&ndash;Boltzmann Solver: models, equations, recombination, perturbations, spectra and nonlinear power (FML::COSMOLOGY)</a><br>
  <a href="perturbationtheory.php">One-loop SPT, TNS, EFTofLSS, correlation multipoles and FFTLog (FML::PERTURBATION)</a><br>
  <a href="lpt.php">Lagrangian perturbation theory: 1&ndash;3LPT and the arbitrary-order EdS solver (FML::COSMOLOGY::LPT)</a><br>
  <a href="halomodel.php">Halo models: HMcode, Halofit+Bird, ReACT and AxionHMcode, plus HOD and baryonification</a><br>
</p>

<p class="page-toc">
  Simulations:
  <a href="nbody.php">PM N-body methods, force options and time stepping (FML::NBODY)</a><br>
  <a href="shortrange.php">Short-range gravity for P3M / TreePM, also with a Yukawa force (FML::SHORT_RANGE_GRAVITY)</a><br>
  <a href="nbodycode.php">SimulationCode: PM/COLA model catalogue, equations and analysis capabilities</a><br>
  <a href="amr.php#AMRCode">AMRCode: an adaptive-mesh-refinement N-body code</a><br>
  <a href="wavedm.php">Wave (fuzzy) dark matter: Schr&ouml;dinger&ndash;Poisson (FML::WAVEDM)</a><br>
  <a href="hydro.php">Gas dynamics: finite-volume Euler equations and SPH (FML::HYDRODYNAMICS, FML::SPH)</a><br>
</p>

<p class="page-toc">
  Analysing simulations and surveys:
  <a href="snapshotanalyzer.php">SnapshotAnalyzer: standalone Gadget post-processing and shared FML/Analysis kernels</a><br>
  <a href="corrfunc.php">Correlation functions, power spectra and polyspectra (FML::CORRELATIONFUNCTIONS)</a><br>
  <a href="survey.php">Survey multipoles, window functions and BAO reconstruction (FML::SURVEY)</a><br>
  <a href="statistics.php">kNN-CDFs, counts-in-cells, density split, Minkowski functionals and control variates</a><br>
  <a href="fof.php">Friends-of-Friends, relaxed FoF and spherical-overdensity halos (FML::FOF)</a><br>
  <a href="phasespace.php">Phase-space halo finder and merger histories (FML::PHASE_SPACE_HALOS)</a><br>
  <a href="tesselation.php">Periodic Delaunay tessellation and Voronoi diagram using CGAL (FML::TRIANGULATION)</a><br>
  <a href="dtfe.php">Delaunay Tessellation Field Estimator, DTFE (FML::TRIANGULATION)</a><br>
  <a href="watershed.php">Watershed groups from the Delaunay tessellation (FML::TRIANGULATION)</a><br>
  <a href="voidfinders.php">Void finders: ZOBOV, voxel watershed and spherical (FML::VOIDFINDER)</a><br>
  <a href="cosmicweb.php">Cosmic web classification: T-web, V-web and ORIGAMI (FML::COSMICWEB)</a><br>
</p>

<p class="page-toc">
  Inference:
  <a href="inference.php">MCMC/HMC, cosmological fits and trainable neural-network emulators</a><br>
  <a href="fieldlevel.php">Field-level inference: serial/MPI linear, LPT and PM models, HMC/MAP, reconstruction and restart (FML::FIELDLEVEL)</a><br>
</p>

<p class="page-toc">
  Other things:
  <a href="gslwrapper.php">GSL wrappers: ODESolver (FML::SOLVERS::ODESOLVER) and Spline (FML::INTERPOLATION::SPLINE)</a><br>
  <a href="fileutils.php">Reading and writing files (FML::FILEUTILS)</a><br>
  <a href="random.php">Random numbers, counter-based Philox and Latin hypercube designs</a><br>
  <a href="python.php">Python interfaces</a><br>
  <a href="other.php#Hilbert">Peano&ndash;Hilbert keys, domain decomposition and distributed grid</a><br>
  <a href="other.php">Utils: timing, units, plotting, math, KD-tree, PGD and performance cases</a><br>
</p>

<p>Scientific papers, external-code lineage and comparison targets are collected in the <a href="https://github.com/HAWinther/FML/blob/master/FML/REFERENCES.md" style="display: inline">FML reference index</a>. Each module documents its own implementation and validation scope.</p>
