<hr/>
<header class="entry-header text-center">
  <h1 class="entry-title">Table of contents</h1>
</header>
<hr/>

<p class="page-toc">
<a href="#About">About</a><br>
<a href="#Ideas">The ideas behind FML</a><br>
<a href="#Code">Getting the code: FML 2.0 and 1.0</a><br>
<a href="#Features">Features</a><br>
<a href="#Requirements">Requirements</a><br>
<a href="#Compiling">Compiling</a><br>
<a href="documentation.php">Documentation</a><br>
<a href="#Example">Code examples</a><br>
</p>

<hr/>
<header class="entry-header text-center">
  <h1 class="entry-title" id="About">About the library</h1>
</header>
<hr/>

<figure class="cinematic-figure" id="cinematic-01-cosmic-web">
  <a href="images/cinematic/01-cosmic-web.jpg" aria-label="View full-size illustration: The cosmic web">
    <img src="images/cinematic/01-cosmic-web.jpg" width="1774" height="887" loading="eager" fetchpriority="high" decoding="async" alt="A luminous cosmic matter network with filaments and clusters surrounding dark voids">
  </a>
  <figcaption><strong>The cosmic web.</strong> Conceptual false-color rendering of the cosmic matter density: clusters and filaments surround underdense voids. <span class="cinematic-credit">AI-generated concept illustration.</span></figcaption>
</figure>

<p>
FML is a C++17 library for large-scale-structure cosmology. It started as a set of general building blocks for working in parallel: grids that do Fourier transforms, particle containers, mass assignment and PDE solvers, where the library takes care of the MPI and OpenMP parallelisation so that you can write an algorithm without thinking about how the data is split over tasks. On top of these it has grown to cover almost everything one needs in large-scale structure: the expansion history and an <a href="cosmology.php" style="display: inline">Einstein&ndash;Boltzmann solver</a>, initial conditions and <a href="nbodycode.php" style="display: inline">simulations</a>, halo and void finding, clustering statistics, survey estimators and parameter inference. And not only for $\Lambda$CDM, but for any cosmological model you add. Every module has a README with its conventions and validation, examples, and tests that run in continuous integration.
</p>

<hr/>
<header class="entry-header text-center">
  <h1 class="entry-title" id="Ideas">The ideas behind FML</h1>
</header>
<hr/>

<figure class="text-center" style="margin-bottom:24px">
  <a href="images/fml-architecture.svg"><img src="images/fml-architecture.svg" width="1120" height="640" loading="lazy" style="max-width:100%;height:auto" alt="FML is built up from a domain decomposition (slabs, pencils or a Peano-Hilbert curve), grids and particles that live on a domain, the algorithms that work on them, and the programs built from the algorithms"></a>
  <figcaption>How FML is built up: a domain decomposition at the bottom, grids and particles that live on a domain and do the communication, the algorithms on top of them, and the programs built from the algorithms.</figcaption>
</figure>

<p>
The goal of the library was, and is, that one should not have to make the tools before doing the science, and that the algorithms built on them should make as few assumptions as possible about how they are going to be used. A few ideas run through all of it:
</p>
<ul>
<li><b>Parallel without having to think about it.</b> The data structures know how they are distributed. A grid knows which part of the box it owns, exchanges its ghost cells and does distributed FFTs; a particle container sends particles to the task that owns them. Algorithms written on top of these run the same way in serial, with MPI, with OpenMP or with both, and MPI and FFTW are set up automatically. The one thing to keep in mind is that collective functions are called by all tasks.</li>
<li><b>Domains, grids and particles.</b> A <i>domain</i> says which MPI task owns which part of the box: slabs (the default), <a href="pencilgrid.php" style="display: inline">pencils</a> that split $x$ and $y$ (for up to $N_{\rm mesh}^2$ tasks) or load-balanced ranges of a Peano&ndash;Hilbert curve. Grids live on a domain: <a href="fftwgrid.php" style="display: inline">$\texttt{FFTWGrid}$</a> on slabs and $\texttt{PencilGrid}$ on pencils (both with in-place FFTs and the same interface), the general <a href="mpigrid.php" style="display: inline">$\texttt{MPIGrid}$</a> and the <a href="amr.php" style="display: inline">adaptively refined</a> $\texttt{AdaptiveMPIGrid}$. Particles (<a href="mpiparticles.php" style="display: inline">$\texttt{MPIParticles}$</a>) are distributed according to the same domain as a grid or according to another one, and the operations between them, such as <a href="interpol.php#AnyDomain" style="display: inline">mass assignment</a>, work across domains. Because the algorithms are written against the common grid interface, most of the library, SimulationCode and SnapshotAnalyzer run on slabs or on pencils.</li>
<li><b>General through templates.</b> Classes and algorithms are templated on the particle type and on the number of dimensions. Your own particle class works as it is: it needs <tt>get_pos()</tt>, and the library detects at compile time which other members (velocity, mass, ID, ...) it has and uses them where it needs them. Most of the library is header-only.</li>
<li><b>Easy to extend.</b> A module is a folder with its headers, a README, an example and tests. A new analysis method uses the grids and particles and gets the parallelisation from them.</li>
<li><b>Lightweight.</b> You need a C++17 compiler and make. FFTW (Fourier transforms) and GSL (splines, ODEs) are used by most of the library; MPI, OpenMP, Lua, HDF5, CGAL and Python are optional and only needed by the modules that use them. There are no large dependencies such as Boost: the multigrid solvers, halo and void finders, KD-tree, FFTLog, samplers, the Einstein&ndash;Boltzmann solver and so on are all part of the library. This now includes an exact, MPI-parallel Delaunay triangulation, used by default by the DTFE, ZOBOV and the ORIGAMI halos, so CGAL is optional everywhere.</li>
<li><b>Any cosmological model.</b> The physics is implemented once and shared: the same background and the same quasi-static gravity ($\mu$ and $\Sigma$) drive the Einstein&ndash;Boltzmann solver, the growth factors, LPT, the halo model and the simulations. A new model (dark energy, modified gravity, interacting dark sectors, ...) is a header and an entry in a registry; some twenty are included.</li>
<li><b>Programs as examples.</b> <a href="nbodycode.php" style="display: inline">SimulationCode</a> (formerly COLASolver), <a href="snapshotanalyzer.php" style="display: inline">SnapshotAnalyzer</a>, the Einstein&ndash;Boltzmann solver, <a href="inference.php#Fitting" style="display: inline">CosmologyFitting</a> and <a href="amr.php#AMRCode" style="display: inline">AMRCode</a> were written to show what can be done with the library. They have grown into codes that are useful in their own right, and they are also the most demanding tests of the building blocks.</li>
<li><b>Tested and validated.</b> About 200 test programs compare with analytic results, manufactured solutions, conservation laws and external codes (CLASS, CAMB, CCL, FAST-PT, MG-PICOLA, ...). They run in continuous integration on every push with 1 and 4 MPI tasks, and with longer accuracy checks every night. The module documentation says what has been validated and over which range.</li>
</ul>

<p align="center">
  <a href="https://github.com/HAWinther/FML/" title="Code on GitHub"><img src="images/github.png" alt="Code on GitHub"></a>
Code on GitHub
</p>

<hr/>
<header class="entry-header text-center">
  <h1 class="entry-title" id="Code">Getting the code: FML 2.0 and 1.0</h1>
</header>
<hr/>

<p>
The current version is <b>FML 2.0</b>: the <tt>master</tt> branch on <a href="https://github.com/HAWinther/FML/" style="display: inline">GitHub</a> (tag <tt>v2.0.0</tt>). Compared with 1.0 it adds, among much else, pencil and Peano&ndash;Hilbert domain decompositions, adaptive mesh refinement, SimulationCode (the successor of COLASolver) with many more gravity and dark-sector models, a much extended Einstein&ndash;Boltzmann solver, perturbation theory, MCMC and cosmological fits, field-level inference, halo, void and subhalo finders, HOD mocks, baryonification, summary statistics, wave dark matter and gas dynamics; the full list of changes is in the <a href="https://github.com/HAWinther/FML/blob/master/FML/CHANGELOG.md" style="display: inline">changelog</a>. New development happens in the <tt>FMLDevelop</tt> branch.
</p>
<pre class="prettyprint">
git clone https://github.com/HAWinther/FML.git
</pre>
<p>
The previous version, <b>FML 1.0</b>, is the tag <tt>v1.0.0</tt>. To get it (with the <tt>TestData</tt> submodule its examples use):
</p>
<pre class="prettyprint">
git clone --branch v1.0.0 --depth 1 --recursive https://github.com/HAWinther/FML.git FML-1.0
</pre>
<p>
FML is free software: except for third-party code carrying separate license notices, its source code is licensed under the <a href="https://github.com/HAWinther/FML/blob/master/LICENCE" style="display: inline">GNU General Public License v3.0 or later</a>. Third-party code keeps its own copyright and license notices (for example <tt>FML/Global/SystemMemory.h</tt>, CC BY 3.0).
</p>
<p>
FML is developed by Hans A. Winther. Many people have helped improve it, among them Bartolomeo Fiorini, Guilherme Brando, Kazuya Koyama, Daniela Saadeh, Herman Sletmoen and Renate Mauland-Hus.
</p>

<hr/>
<header class="entry-header text-center">
  <h1 class="entry-title" id="Features">Features</h1>
</header>
<hr/>

<p>
The library provides the following building blocks and cosmology workflows. Module documentation records the supported models, conventions and validation limits:
</p>
<p><b>Parallel building blocks</b></p>
<ul>
<li>A grid class <a href="fftwgrid.php" style="display: inline">$\texttt{FFTWGrid}$</a> for holding real and complex data (float, double, long double) shared among tasks built around FFTW for doing in-place Fourier transforms with MPI, threads or both, with reusable plans for changing the resolution of Fourier grids.</li>
<li><a href="pencilgrid.php" style="display: inline">Pencil decomposition</a>: $\texttt{PencilGrid}$ splits a grid over a 2D grid of tasks (up to $N_{\rm mesh}^2$ tasks instead of $N_{\rm mesh}$), and the grid code is written against a common interface, so the library, the N-body code and the snapshot analyzer run on slabs or on pencils (<tt>make USE_PENCIL_GRID=true</tt>). Complex-to-complex grids ($\texttt{CFFTWGrid}$, $\texttt{CPencilGrid}$) for complex fields such as a wave function.</li>
<li>A general grid class <a href="mpigrid.php" style="display: inline">$\texttt{MPIGrid}$</a> for holding any data shared between tasks.</li>
<li>A general particle container <a href="mpiparticles.php" style="display: inline">$\texttt{MPIParticles}$</a> for holding particles shared between tasks and communicating them if they leave their domain (neighbour-only routing when possible).</li>
<li><a href="interpol.php" style="display: inline">Particle-grid assignment and interpolation</a> (NGP, CIC, TSC, PCS, PQS, with interlacing and deconvolution; also for particles on <a href="interpol.php#AnyDomain" style="display: inline">any domain</a>), <a href="interpol.php#NUFFT" style="display: inline">non-uniform FFTs</a> (exact Fourier transforms of particles) and sheet interpolation.</li>
<li>A general <a href="multigridsolver.php" style="display: inline">multigrid solver</a> for solving PDEs with periodic or Dirichlet boundary conditions with a simple interface for defining the equation.</li>
<li><a href="amr.php" style="display: inline">Adaptive mesh refinement</a>: an adaptively refined grid $\texttt{AdaptiveMPIGrid}$ distributed along a Peano&ndash;Hilbert curve (refinement, 2:1 balance, load balancing, ghosts), a FAS multigrid solver on it (Poisson, DGP, symmetron) and an AMR N-body code.</li>
<li><a href="gridop.php" style="display: inline">Typical grid operations</a> like smoothing, whitening, convolutions of a field in Fourier space or real space with an arbitrary kernel.</li>
<li><a href="randomfield.php" style="display: inline">Gaussian and non-Gaussian random fields</a> (local, equilateral, orthogonal or generic $f_{\rm NL}$) with the N-GenIC generator or <a href="random.php#Philox" style="display: inline">counter-based Philox</a> fields that are independent of the resolution and the number of tasks.</li>
</ul>
<p><b>Cosmological theory</b></p>
<ul>
<li>A dedicated <a href="cosmology.php" style="display: inline">Einstein&ndash;Boltzmann Solver</a> site: background, recombination, scalar/tensor/isocurvature perturbations, CMB spectra, lensing and matter power. Models include $\Lambda$CDM, evolving dark energy, neutrinos, Horndeski and modified gravity; nonlinear options include Halofit, Bird, HMcode-2016/2020, native ReACT, e-MANTIS f(R) boosts, mixed cold/axion AxionHMcode, EuclidEmulator2 and a matched-GR Brans&ndash;Dicke boost.</li>
<li><a href="perturbationtheory.php" style="display: inline">One-loop perturbation theory</a> ($P_{22}$ and $P_{13}$ with FFTLog, as FAST-PT, or by direct quadrature with custom kernels), <a href="perturbationtheory.php#RSD" style="display: inline">TNS and EFTofLSS redshift-space multipoles</a>, correlation multipoles and <a href="perturbationtheory.php#FFTLog" style="display: inline">FFTLog</a> Hankel transforms.</li>
<li><a href="lpt.php" style="display: inline">Lagrangian perturbation theory</a>: 1LPT, 2LPT and 3LPT fields (dealiasing, transverse terms and scale-dependent growth), plus a standalone arbitrary-order EdS solver with small-reference validation through sixth order, growth histories, ALPT and reconstruction.</li>
<li><a href="halomodel.php" style="display: inline">Spherical collapse and halo models</a>: the general halo model, HMcode-2016/2020, Halofit with Bird neutrino corrections, native screened collapse and ReACT reactions (also for coupled dark energy and the Fading Dark Sector), and mixed cold/axion AxionHMcode.</li>
</ul>
<p><b>Simulations</b></p>
<ul>
<li><a href="nbody.php" style="display: inline">N-body</a> methods for computing forces and moving particles (with explicit force options and Quinn, Tassev, FastPM and BullFrog time stepping), including modified gravity solved exactly or with the screening method of Winther &amp; Ferreira 2015.</li>
<li>A dedicated <a href="nbodycode.php" style="display: inline">SimulationCode</a> site for the PM/COLA N-body code: cosmology and gravity model catalogues (including interacting dark sectors, cubic Galileon and reduced Horndeski), force and time-stepping equations, massive neutrinos, separate CDM/baryons, Big-Bang starts, lightcones and on-the-fly power spectra, multipoles, bispectra, halos, hybrid bias and baryonification.</li>
<li><a href="shortrange.php" style="display: inline">Short-range gravity</a> for P3M/TreePM codes: Gaussian force split, spline softening, cell lists, an octree, a Yukawa fifth force and MPI ghost exchange.</li>
<li><a href="amr.php#AMRCode" style="display: inline">AMRCode</a>: an adaptive-mesh-refinement N-body code for $\Lambda$CDM, symmetron and dilaton gravity (the fifth force solved with screening on the refined grid) with block time steps, in the spirit of RAMSES.</li>
<li><a href="wavedm.php" style="display: inline">Wave (fuzzy) dark matter</a>: a Schr&ouml;dinger&ndash;Poisson solver in a static or expanding box, on slabs or pencils, with checkpoints.</li>
<li><a href="hydro.php" style="display: inline">Gas dynamics</a>: a finite-volume Euler solver (Rusanov, HLLE and HLLC fluxes, exact Riemann reference) and conservative SPH, with MPI, OpenMP and gravity and expansion hooks.</li>
</ul>
<p><b>Analysis of simulations and surveys</b></p>
<ul>
<li><a href="snapshotanalyzer.php" style="display: inline">SnapshotAnalyzer</a>: standalone MPI post-processing of Gadget binary/HDF5 snapshots with shared analysis kernels, spectra, halos, voids, T-web, morphology and hybrid bias (real and redshift space).</li>
<li><a href="corrfunc.php" style="display: inline">Correlation functions</a> by general pair counting, power spectra (naive binning, interlacing or direct summation) and multipoles, and <a href="corrfunc.php#PolyNew" style="display: inline">polyspectra</a> (bispectrum, trispectrum, ...) with dealiasing, selected configurations and sparse results, plus <a href="corrfunc.php#Connected" style="display: inline">connected periodic four-point statistics</a>.</li>
<li><a href="survey.php" style="display: inline">Survey clustering</a>: FKP/Yamamoto multipoles with a local line of sight, window-function multipoles and the window matrix, and BAO reconstruction (RecIso and RecSym).</li>
<li><a href="statistics.php" style="display: inline">Summary statistics</a>: kNN-CDFs and counts-in-cells, density-split and marked correlations, Minkowski functionals of excursion sets, and control variates (CARPool-type) for ensemble means.</li>
<li><a href="fof.php" style="display: inline">Friends of Friends</a> and relaxed FoF halos with spherical-overdensity masses, a <a href="phasespace.php" style="display: inline">phase-space halo and subhalo finder</a> with unbinding, and merger trees from particle-ID overlaps.</li>
<li><a href="halomodel.php#HOD" style="display: inline">HOD galaxy mocks</a> with decorated assembly bias and marginal satellite occupations and <a href="halomodel.php#Baryons" style="display: inline">baryonification</a> of gravity-only simulations.</li>
<li>FML's own <a href="tesselation.php" style="display: inline">Delaunay triangulation and periodic Voronoi tessellation</a> (2D/3D, exact predicates, MPI parallel for any particle distribution with results independent of the number of tasks, fast updates when the points move; no dependencies and 1.5-4 times faster than CGAL), the optional CGAL-based $\texttt{MPIPeriodicDelaunay}$, and the <a href="dtfe.php" style="display: inline">DTFE</a> for the density, velocity, divergence, vorticity and shear on a grid.</li>
<li><a href="watershed.php" style="display: inline">Watershed</a>, <a href="voidfinders.php" style="display: inline">void finders</a> (ZOBOV with zone merging, voxel watershed as in Revolver, spherical underdensities, also clusters) and <a href="cosmicweb.php" style="display: inline">cosmic web classification</a> (T-web, V-web, ORIGAMI).</li>
</ul>
<p><b>Inference</b></p>
<ul>
<li><a href="inference.php#NeuralNetwork" style="display: inline">Trainable neural-network emulators</a> with Jacobians and PyTorch export, and <a href="random.php#LatinHypercube" style="display: inline">Latin hypercube experiment designs</a>.</li>
<li><a href="inference.php" style="display: inline">MCMC</a>: adaptive Metropolis&ndash;Hastings with parallel chains, fast/slow parameters and checkpoints, a minimizer with profile likelihoods and Hessians, Hamiltonian Monte Carlo for millions of parameters, and rank-normalized $\hat R$ and bulk/tail ESS diagnostics.</li>
<li><a href="inference.php#CosmicShear" style="display: inline">Tomographic cosmic shear</a>: NLA intrinsic alignments, photo-z and calibration nuisances, angular correlations and logarithmic COSEBIs, with optional scale-independent Sigma(z).</li>
<li><a href="inference.php#Fitting" style="display: inline">Cosmological parameter fits</a> with Pantheon+, DES-SN5YR, DES-Dovekie, Union3, DESI DR1/DR2 BAO, BOSS/eBOSS, DESI DR1 ShapeFit and full shape, KiDS-1000 cosmic shear, BBN, Planck distance priors, Planck 2018 plik_lite, low-$\ell$ and lensing, Planck PR4 CamSpec, ACT DR6 lensing, and mock CMB likelihoods.</li>
<li><a href="fieldlevel.php" style="display: inline">Field-level inference</a> with periodic serial/MPI linear, 1LPT, 2LPT and <a href="fieldlevel.php#PM" style="display: inline">particle-mesh</a> models (on slabs or pencils), joint field/bias/noise inference, adjoint gradients, HMC/MAP, reconstruction products and same-rank checkpoint/restart; a separate BAO prototype has simulation-based calibration tests.</li>
</ul>
<p><b>Other things</b></p>
<ul>
<li><a href="gslwrapper.php" style="display: inline">Wrapper over GSL</a> to more easily solve ODEs and make 1D and 2D splines, and a <a href="fileutils.php" style="display: inline">wrapper over Lua</a> for reading parameterfiles.</li>
<li><a href="fileutils.php" style="display: inline">Reading and writing</a> common file-types: Gadget (binary and HDF5) and Ramses files, ascii files, etc.</li>
<li>A <a href="python.php" style="display: inline">Python package</a> for kNN and counts-in-cells statistics, and SWIG wrappers.</li>
<li><a href="other.php" style="display: inline">Units</a> and physical constants, a KD-tree, potential gradient descent (PGD), memory logging, timings, plots (Matplotlib) and <a href="other.php#Visualization" style="display: inline">pictures without Python</a> (projected slabs as PNG) and reproducible <a href="other.php#Performance" style="display: inline">performance cases</a>.</li>
</ul>

Most of these things are made for simulation data, but a lot of them work on survey data also (some things require some small adjustments). A few of the features above (like FFTLog <a href="https://jila.colorado.edu/~ajsh/FFTLog/" style="display: inline">[1]</a>,<a href="https://github.com/slosar/FFTLog" style="display: inline">[2]</a> and <a href="https://github.com/lava/matplotlib-cpp" style="display: inline">matplotlib-cpp</a>) are outright stolen from the web. For code that is taken from the web the header files should contain the proper acknowledgements and the appropriate licences should hopefully be included, but if you see anything missing please let me know.
</p>

<hr/>
<header class="entry-header text-center">
  <h1 class="entry-title" id="Requirements">Requirements</h1>
</header>
<hr/>
<p>
You will need a C++17 compatible compiler (e.g. gcc 7+, clang 5+) to compile the library as we use a lot of if constexpr to do compile-time computations to make things more efficient. Parts of the code can also be compiled without any external libraries, but to take full advantage of the features in the code you should at least have the FFTW3 library (with MPI support) and GSL installed. With these two libraries installed you can use most of FML. In addition to this the other libraries we use in just a few places are CGAL (tessellations, DTFE, ZOBOV, ORIGAMI halos), Lua and HDF5. You should also have an MPI compliant compiler though the code can be compiled and run in serial.
<ul>
<li><a href="https://www.fftw.org/download.html">FFTW</a> version 3+ : Required to be able to do Fourier transforms and the related algorithms that use Fourier transforms. The grid class FFTWGrid can be compiled without it.</li>
<li><a href="https://www.cgal.org/download.html">CGAL</a> version 5+ : Optional: only needed for the CGAL-based tessellation (<tt>MPIPeriodicDelaunay</tt>), which the DTFE, the watershed, ZOBOV and the ORIGAMI halos can use instead of FML's own (the default).</li>
<li><a href="https://www.gnu.org/software/gsl/">GSL</a> version 2+ : Required to solve ODEs, make splines, random numbers (though if we don't have this we use C++ &lt;random&gt; instead) and linear algebra (just a few places).</li>
<li><a href="https://www.lua.org/download.html">Lua</a> version 5+ : Required to use LuaFileParser to read parameterfiles (used by the N-body code).</li>
<li><a href="https://www.hdfgroup.org/solutions/hdf5/">HDF5</a> : Optional, to read and write Gadget files in the HDF5 format.</li>
<li>Python 3.9+ with NumPy and pybind11 : Optional, to build the <a href="python.php" style="display: inline">fml-analysis</a> Python package (SWIG for the older wrappers).</li>
</ul>
</p>

<hr/>
<header class="entry-header text-center">
  <h1 class="entry-title" id="Compiling">Compiling</h1>
</header>
<hr/>

<p>
Most of the library is in the form of header files that you just need to include in your code. A few of these need to be compiled. We could have compiled this up to a shared library, but given the options the user has (with MPI or without, with some external libraries or without, etc.) it's better to compile these files together with your code. All the Makefiles in the library include the shared configuration FML/Makefile.config: put the compiler, library paths and options for your machine in FML/Makefile.local (not tracked by git) or pass them on the command line, e.g. make USE_MPI=false USE_DEBUG=true. Having tons of define statements in the code is not great, but it's the only good way of being able to use the code without having all the features. Here is a list of the most important options:

<ul>
<li>USE_MPI : Compile with MPI support. You will need to have an MPI compiler to use this.</li>
<li>USE_OMP : Compile with OpenMP support. Can be used together with MPI for which each MPI task will use OMP_NUM_THREADS threads in certain loops.</li>
<li>USE_FFTW : If you have the FFTW3 library. You will have to provide the include and lib paths in the Makefile. If used together with USE_MPI you will need to have compiled the FFTW3 library with MPI support.</li>
<li>USE_FFTW_THREADS : If you have the FFTW3 library. Use threads to parallelize the FFTs. Can be used together with USE_MPI.</li>
<li>USE_GSL : If you have the GSL library. You will have to provide the include and lib paths in the Makefile. </li>
<li>USE_CGAL : If you have the CGAL library and want its tessellation as an option (off by default everywhere). You will have to provide the include and lib paths in the Makefile. </li>
<li>USE_LUA : If you have the Lua library. You will have to provide the include and lib paths in the Makefile.</li>
<li>USE_HDF5 : If you have the HDF5 library (for Gadget HDF5 files).</li>
<li>USE_PYTHON : If you have Python with Matplotlib installed. You will have to provide the include and lib paths in the Makefile. </li>
</ul>
Some define statements that can be added to change how the code works:
<ul>
<li>SINGLE_PRECISION_FFTW : Use float instead of double for FFTWGrid and FFTs in general. FFTW needs to be compiled with float support to use this.</li>
<li>LONG_DOUBLE_PRECISION_FFTW : Use long double instead of double for FFTWGrid and FFTs in general. FFTW needs to be compiled with long double support to use this.</li>
<li>NO_AUTO_FML_SETUP : FML, MPI and FFTW are automatically initialized and finalized in the code. If you don't want this add this define. NB: if you use this you should know how FFTW should be initialized with MPI and/or threads to avoid issues. You also need to call FML::init_fml after initializing MPI and FFTW.</li>
<li>USE_MEMORYLOG : Log allocations over a certain size (see the MemoryLog header). Useful to map out the memory footprint of the code. Currently only a few classes implement this.</li>
</ul>
In addition to these we have some Makefile options that can be useful:
<ul>
<li>USE_SANITIZER : Compile with -fsanitize=address to check for bad memory accesses. Useful for debugging.</li>
<li>USE_DEBUG : Do some extra asserts, print some more info from the algorithms while running.</li>
</ul>

Headers are included as &lt;FML/...&gt; so the include path is the folder containing FML/, and the *.cpp files of the modules you use (e.g. Global/Global.cpp, Spline/Spline.cpp, ODESolver/ODESolver.cpp) need to be compiled with the code.

The full guide to the build options is in <tt>FML/BUILDING.md</tt> and the list of modules with their tests and examples in <tt>FML/CAPABILITIES.md</tt>. From the repository root, <tt>make -C FML test</tt> builds and runs the library tests and <tt>make -C FML examples</tt> the examples; every module with a <tt>tests</tt> folder is also run in continuous integration (with 1 and 4 MPI tasks on every push, and 1, 2 and 4 tasks with the longer accuracy checks every night). The examples need no data files: those that read particles, a survey or a snapshot make a small synthetic data set (<tt>FML/ExampleData</tt>).

</p>

<hr/>
<header class="entry-header text-center">
  <h1 class="entry-title" id="Example">Code examples</h1>
</header>
<hr/>

<p>
Almost every folder in the code contains an [example] folder showing how one can use the methods and classes. Here are a few of those (the three short <a href="https://github.com/HAWinther/FML/tree/master/FML/QuickStart" style="display: inline">quick-start programs</a> are shown in full below the list):
<ul>
<li>
<a href="https://github.com/HAWinther/FML/tree/master/FML/ComputePowerSpectra/example/Main.cpp" style="display: inline">Compute power spectra (and multipoles)</a>
</li>

<li>
<a href="https://github.com/HAWinther/FML/tree/master/FML/FFTLog/example/Main.cpp" style="display: inline">Compute the 2pt correlation function from a given power-spectrum</a>
</li>

<li>
<a href="https://github.com/HAWinther/FML/tree/master/FML/FFTWGrid/example/Main.cpp" style="display: inline">How to use the FFTWGrid</a>
</li>

<li>
<a href="https://github.com/HAWinther/FML/tree/master/FML/Interpolation/example/Main.cpp" style="display: inline">Interpolating grid to positions</a>
</li>

<li>
<a href="https://github.com/HAWinther/FML/tree/master/FML/LPT/example/Main.cpp" style="display: inline">Generate particles from a given power-spectrum using 2LPT,  compute the power-spectrum of this and check that it matches</a>
</li>

<li>
<a href="https://github.com/HAWinther/FML/tree/master/FML/LuaFileParser/example/Main.cpp" style="display: inline">Read a LUA script (parameter file)</a>
</li>

<li>
<a href="https://github.com/HAWinther/FML/tree/master/FML/MPIParticles/example/Main.cpp" style="display: inline">How to create a shared set of particles</a>
</li>

<li>
<a href="https://github.com/HAWinther/FML/tree/master/FML/MakePythonWrapper" style="display: inline">Create a Python/C++ wrapper</a>
</li>

<li>
<a href="https://github.com/HAWinther/FML/tree/master/FML/Math/example/Main.cpp" style="display: inline">Use some of the math function</a>
</li>

<li>
<a href="https://github.com/HAWinther/FML/tree/master/FML/ODESolver/example/Main.cpp" style="display: inline">Solve coupled ODEs</a>
</li>

<li>
<a href="https://github.com/HAWinther/FML/tree/master/FML/PairCounting/example/Main.cpp" style="display: inline">Compute the 2pt correlation function for simulation data and for survey data with randoms</a>
</li>

<li>
<a href="https://github.com/HAWinther/FML/tree/master/FML/Plotting/example/Main.cpp" style="display: inline">Make plots using matplotlib in C++</a>
</li>

<li>
<a href="https://github.com/HAWinther/FML/tree/master/FML/RamsesUtils/example/Main.cpp" style="display: inline">Read RAMSES files</a>
</li>

<li>
<a href="https://github.com/HAWinther/FML/tree/master/FML/GadgetUtils/example/Main.cpp" style="display: inline">Read GADGET files</a>
</li>

<li>
<a href="https://github.com/HAWinther/FML/tree/master/FML/FileUtils/example/Main.cpp" style="display: inline">Read formatted ascii files</a>
</li>

<li>
<a href="https://github.com/HAWinther/FML/tree/master/FML/RandomFields/example/Main.cpp" style="display: inline">Generate non-gaussian fields, compute bispectrum and estimate fNL and compare to the input</a>
</li>

<li>
<a href="https://github.com/HAWinther/FML/tree/master/FML/Spline/example/Main.cpp" style="display: inline">Make a 1D or 2D spline of some data</a>
</li>

<li>
<a href="https://github.com/HAWinther/FML/tree/master/FML/Survey/example/Main.cpp" style="display: inline">Convert galaxies (RA,DEC,z) to Cartesian coordinates</a>
</li>

<li>
<a href="https://github.com/HAWinther/FML/tree/master/FML/Triangulation/example/Main.cpp" style="display: inline">Make a triangulation and run a watershed algorithm on density to find voids</a>
</li>

<li>
<a href="https://github.com/HAWinther/FML/tree/master/FML/Triangulation/example/DTFEExample.cpp" style="display: inline">Interpolate the density and velocity field (divergence, vorticity, shear) to a grid with the DTFE</a>
</li>

<li>
<a href="https://github.com/HAWinther/FML/tree/master/FML/VoidFinders/example/Main.cpp" style="display: inline">Find voids and clusters with ZOBOV, the voxel watershed and spherical underdensities</a>
</li>

<li>
<a href="https://github.com/HAWinther/FML/tree/master/FML/CosmicWeb/example/Main.cpp" style="display: inline">Classify the cosmic web with the T-web, V-web and ORIGAMI</a>
</li>

<li>
<a href="https://github.com/HAWinther/FML/tree/master/FML/Units/example/Main.cpp" style="display: inline">How to use the unit class</a>
</li>

<li>
<a href="https://github.com/HAWinther/FML/tree/master/FML/Cosmology/EinsteinBoltzmannSolver/Main.cpp" style="display: inline">A simple Einstein-Boltzmann solver. Compute the CMB angular- and matter power-spectra plus real space correlations function in the $\Lambda$CDM model</a>
</li>

<li>
<a href="https://github.com/HAWinther/FML/tree/master/FML/MultigridSolver/examples/FofrExample.cpp" style="display: inline">Solving the Hu-Sawicky $f(R)$ PDE for a given density field</a>
</li>
<li>
<a href="https://github.com/HAWinther/FML/tree/master/FML/MultigridSolver/examples/SymmetronExample.cpp" style="display: inline">Solving the symmetron model PDE for a given density field</a>
</li>
<li>
<a href="https://github.com/HAWinther/FML/tree/master/FML/MultigridSolver/examples/dgpExample.cpp" style="display: inline">Solving the nDGP PDE for a given density field</a>
</li>
<li>
<a href="https://github.com/HAWinther/FML/tree/master/FML/MultigridSolver/examples/Reconstruction.cpp" style="display: inline">Solving the density reconstruction equation (1LPT)</a>
</li>
<li>
<a href="https://github.com/HAWinther/FML/tree/master/FML/FriendsOfFriends/example/Main.cpp" style="display: inline">Locate Friends-of-Friends groups (halos) in a simulation</a>
</li>
<li>
<a href="https://github.com/HAWinther/FML/tree/master/FML/NBody/example/Main.cpp" style="display: inline">A simple Particle Mesh N-body solver (choice of normal or using the COLA method) with 1LPT,2PT or 3LPT IC (gaussian and non-gaussian fNL) and computation of $P(k),P_\ell(k),B(k_1,k_2,k_3)$ and a halo catalogue from the results.</a>
</li>
<li>
<a href="https://github.com/HAWinther/FML/tree/master/FML/SimulationCode" style="display: inline">A full PM/COLA N-body code</a> for many cosmologies and modified gravity models (see <a href="nbodycode.php" style="display: inline">the documentation</a>).
</li>
<li>
<a href="https://github.com/HAWinther/FML/tree/master/FML/Cosmology/EinsteinBoltzmannSolver" style="display: inline">[code]</a> The Einstein&ndash;Boltzmann solver <tt>./cmb</tt>: CMB and matter power spectra for $\Lambda$CDM and the models beyond it (see <a href="cosmology.php" style="display: inline">the documentation</a>)
</li>
<li>
<a href="https://github.com/HAWinther/FML/tree/master/FML/CosmologyFitting" style="display: inline">[code]</a> <tt>./mcmc</tt>: fit cosmological parameters to supernova, DESI BAO and Planck data (see <a href="inference.php#Fitting" style="display: inline">the documentation</a>)
</li>
<li>
<a href="https://github.com/HAWinther/FML/tree/master/FML/MCMC/example/Main.cpp" style="display: inline">[code]</a> Sample a correlated Gaussian with the adaptive Metropolis sampler, find the best fit, the Hessian and a profile likelihood
</li>
<li>
<a href="https://github.com/HAWinther/FML/tree/master/FML/PerturbationTheory/example/Main.cpp" style="display: inline">[code]</a> One-loop SPT power spectrum with the fast FFTLog algorithm, checked against direct quadrature
</li>
<li>
<a href="https://github.com/HAWinther/FML/tree/master/FML/HaloModel/example/Main.cpp" style="display: inline">[code]</a> Halo model and HMCode non-linear power spectrum
</li>
<li>
<a href="https://github.com/HAWinther/FML/tree/master/FML/SphericalCollapse/example/Main.cpp" style="display: inline">[code]</a> Spherical collapse: $\delta_c(z)$ and $\Delta_{\rm vir}(z)$
</li>
<li>
<a href="https://github.com/HAWinther/FML/tree/master/FML/Survey/workflow_example/Main.cpp" style="display: inline">[code]</a> A synthetic survey: power-spectrum multipoles, window multipoles, window matrix and BAO reconstruction
</li>
<li>
<a href="https://github.com/HAWinther/FML/tree/master/FML/HOD/example/Main.cpp" style="display: inline">[code]</a> Populate halos with HOD galaxies and measure kNN-CDFs and counts-in-cells
</li>
<li>
<a href="https://github.com/HAWinther/FML/tree/master/FML/Environment/example/Main.cpp" style="display: inline">[code]</a> Density-split cross-correlations and marked correlations
</li>
<li>
<a href="https://github.com/HAWinther/FML/tree/master/FML/Morphology/example/Main.cpp" style="display: inline">[code]</a> Minkowski functionals of the excursion sets of a field
</li>
<li>
<a href="https://github.com/HAWinther/FML/tree/master/FML/EnsembleStatistics/example/Main.cpp" style="display: inline">[code]</a> Control-variate estimate of an ensemble mean
</li>
<li>
<a href="https://github.com/HAWinther/FML/tree/master/FML/PhaseSpaceHaloFinder/example/Main.cpp" style="display: inline">[code]</a> Find halos and subhalos in phase space
</li>
<li>
<a href="https://github.com/HAWinther/FML/tree/master/FML/HaloHistory/example/Main.cpp" style="display: inline">[code]</a> Merger histories from particle IDs
</li>
<li>
<a href="https://github.com/HAWinther/FML/tree/master/FML/Baryonification/example/Main.cpp" style="display: inline">[code]</a> Baryonification of a particle distribution around a halo
</li>
<li>
<a href="https://github.com/HAWinther/FML/tree/master/FML/ShortRangeGravity/tests/test_short_range.cpp" style="display: inline">[code]</a> Short-range (P3M/TreePM) forces with cell lists and an octree, checked against direct and Ewald sums
</li>
<li>
<a href="https://github.com/HAWinther/FML/tree/master/FML/FieldLevel/example/Main.cpp" style="display: inline">[code]</a> Field-level inference of the BAO scale with HMC (and <a href="https://github.com/HAWinther/FML/tree/master/FML/FieldLevel/example/Calibration.cpp" style="display: inline">simulation-based calibration</a>)
</li>
<li><a href="https://github.com/HAWinther/FML/blob/master/FML/FieldLevel/example/FrameworkMPI.cpp" style="display: inline">[code]</a> Distributed periodic field-level inference, reconstruction and <a href="https://github.com/HAWinther/FML/blob/master/FML/FieldLevel/example/CheckpointMPI.cpp" style="display: inline">checkpoint/restart</a></li>
<li><a href="https://github.com/HAWinther/FML/blob/master/FML/Emulator/NeuralNetwork/example/Main.cpp" style="display: inline">[code]</a> Train a power-spectrum emulator on Latin hypercube points</li>
<li>
<a href="https://github.com/HAWinther/FML/tree/master/FML/PythonAnalysis/examples/catalogue_statistics.py" style="display: inline">[code]</a> kNN-CDFs and counts-in-cells from Python
</li>
</ul>
</p>

<p>
Three short programs from <a href="https://github.com/HAWinther/FML/tree/master/FML/QuickStart" style="display: inline">FML/QuickStart</a> show what working with the library looks like. Each does a real calculation and checks its result, and they are built and run in continuous integration, so they stay up to date with the library. None of them contains any MPI code, yet they run unchanged on any number of tasks (<tt>make run</tt> in <tt>FML/QuickStart/example</tt>, or e.g. <tt>mpirun -np 4 ./pm_simulation</tt>).
</p>
<p><b>The power spectrum of a catalogue</b> (<a href="https://github.com/HAWinther/FML/blob/master/FML/QuickStart/example/pofk.cpp" style="display: inline">pofk.cpp</a>). Your own particle type, distributed over the MPI tasks, and its power spectrum. The catalogue is uniformly random, so P(k) must be the shot noise.</p>
<div style="text-align: center">
  <div style="width:100%;display: inline-block;text-align: left">
    <pre class="prettyprint" style="overflow-x:auto;white-space:pre;font-size:12px">
// Quick start 1: the power spectrum of a particle catalogue.
//
// Your own particle type, distributed over the MPI tasks, and P(k) with cloud-in-cell assignment and interlacing. The
// catalogue is uniformly random, so P(k) is the shot noise 1/n: we check that.
#include &lt;FML/ComputePowerSpectra/ComputePowerSpectrum.h&gt;
#include &lt;FML/MPIParticles/MPIParticles.h&gt;
#include &lt;FML/RandomGenerator/RandomGenerator.h&gt;

#include &lt;cmath&gt;
#include &lt;iostream&gt;

// Any type with get_pos() (positions in the box [0,1)) works as a particle: no base class needed
struct Particle {
    double x[3];
    double * get_pos() { return x; }
    constexpr int get_ndim() const { return 3; }
};

int main() {
    // The same random catalogue on every task (box units)
    const size_t Npart = 200000;
    std::vector&lt;Particle&gt; catalogue(Npart);
    FML::RANDOM::RandomGenerator rng(1234);
    for (auto &amp; p : catalogue)
        for (auto &amp; x : p.x)
            x = rng.generate_uniform();

    // Distribute it: every task keeps the particles in its part of the box
    FML::PARTICLE::MPIParticles&lt;Particle&gt; part;
    part.create(catalogue, 2 * Npart / FML::NTasks + 1000, FML::PARTICLE::Input::Replicated);

    // P(k) on a 64^3 grid, CIC with interlacing, without subtracting the shot noise
    const int Nmesh = 64;
    FML::CORRELATIONFUNCTIONS::PowerSpectrumBinning&lt;3&gt; pofk(Nmesh / 2);
    pofk.subtract_shotnoise = false;
    FML::CORRELATIONFUNCTIONS::compute_power_spectrum&lt;3&gt;(
        Nmesh, part.get_particles_ptr(), part.get_npart(), part.get_npart_total(), pofk, &quot;CIC&quot;, true);

    // In a box of 1000 Mpc/h: k in h/Mpc, P in (Mpc/h)^3. The shot noise is V/N
    const double box = 1000.0, shotnoise = box * box * box / double(Npart);
    pofk.scale(box);
    double mean = 0.0;
    for (int i = 0; i &lt; pofk.n; i++)
        mean += pofk.pofk[i] / shotnoise / pofk.n;
    if (FML::ThisTask == 0) {
        std::cout &lt;&lt; &quot;# k [h/Mpc]   P(k) [(Mpc/h)^3]   P / shot noise\n&quot;;
        for (int i = 0; i &lt; pofk.n; i++)
            std::cout &lt;&lt; pofk.kbin[i] &lt;&lt; &quot; &quot; &lt;&lt; pofk.pofk[i] &lt;&lt; &quot; &quot; &lt;&lt; pofk.pofk[i] / shotnoise &lt;&lt; &quot;\n&quot;;
        std::cout &lt;&lt; &quot;Mean P / shot noise = &quot; &lt;&lt; mean &lt;&lt; &quot; (expected 1)\n&quot;;
    }
    return std::abs(mean - 1.0) &lt; 0.05 ? 0 : 1;
}
</pre>
  </div>
</div>
<p><b>A particle-mesh simulation</b> (<a href="https://github.com/HAWinther/FML/blob/master/FML/QuickStart/example/pm_simulation.cpp" style="display: inline">pm_simulation.cpp</a>). 2LPT initial conditions, a leapfrog and the power spectrum today, for a scale-free spectrum in an Einstein&ndash;de Sitter universe where the largest scales must follow linear theory.</p>
<div style="text-align: center">
  <div style="width:100%;display: inline-block;text-align: left">
    <pre class="prettyprint" style="overflow-x:auto;white-space:pre;font-size:12px">
// Quick start 2: a particle-mesh N-body simulation in about 60 lines.
//
// A scale-free simulation in an Einstein-de Sitter universe (Omega_m = 1, P(k) ~ k^-2), where linear theory is
// simple: the density grows as the scale factor a and H/H0 = a^-3/2. 2LPT initial conditions at a = 0.02, a leapfrog
// with time steps in ln a, and the power spectrum today against linear theory, P(k, a) = P(k, a_ini) (a / a_ini)^2:
// equal on the largest scales, above it on smaller scales where the growth is nonlinear. All lengths are in units of
// the box, the time variable is dtau = H0 dt / a^2 and the velocities are v = dx/dtau.
#include &lt;FML/ComputePowerSpectra/ComputePowerSpectrum.h&gt;
#include &lt;FML/MPIParticles/MPIParticles.h&gt;
#include &lt;FML/NBody/NBody.h&gt;

#include &lt;algorithm&gt;
#include &lt;cmath&gt;
#include &lt;iostream&gt;

// The particle type: what the algorithms need (position and velocity), nothing more
struct Particle {
    double pos[3]{}, vel[3]{};
    double * get_pos() { return pos; }
    double * get_vel() { return vel; }
    constexpr int get_ndim() const { return 3; }
};

int main() {
    const int Npart_1D = 128, Nmesh = 128, nsteps = 30;
    const double aini = 0.02, aend = 1.0;

    // Initial conditions: P(k)/V at aini as a function of k in units of 1/box, 2LPT, fixed amplitudes
    const double A = 1e-6;
    auto Pofk_over_volume = [&amp;](double kBox) { return kBox &gt; 0 ? A * std::pow(kBox / (2 * M_PI), -2) : 0.0; };
    auto one = [](double) { return 1.0; };
    const double vnorm = aini * aini * std::pow(aini, -1.5); // v = a^2 (H/H0) f_n Psi_n with f_1 = 1, f_2 = 2
    FML::RANDOM::RandomGenerator rng(1234);
    FML::PARTICLE::MPIParticles&lt;Particle&gt; part;
    FML::NBODY::NBodyInitialConditions&lt;3, Particle&gt;(part, Npart_1D, 1.5, Nmesh, true, &amp;rng, one, Pofk_over_volume, 2,
                                                    &quot;gaussian&quot;, 0.0, 1.0, 1.0 / aini - 1.0, {vnorm, 2 * vnorm});

    // The power spectrum of the initial conditions (CIC with interlacing; no shot noise on the initial lattice)
    auto power_spectrum = [&amp;]() {
        FML::CORRELATIONFUNCTIONS::PowerSpectrumBinning&lt;3&gt; pofk(Nmesh / 2);
        pofk.subtract_shotnoise = false;
        FML::CORRELATIONFUNCTIONS::compute_power_spectrum&lt;3&gt;(
            Nmesh, part.get_particles_ptr(), part.get_npart(), part.get_npart_total(), pofk, &quot;CIC&quot;, true);
        return pofk;
    };
    const auto pofk_ini = power_spectrum();

    // Leapfrog with steps uniform in ln a: positions at a_i, velocities half a step ahead. The force is computed at
    // a_i, with the Poisson equation nabla^2 Phi = 1.5 Omega_m a delta; the time steps are the integrals of
    // dtau = a^-3/2 da (drift) and of (a / a_i) a^-3/2 da (kick)
    auto a_of = [&amp;](double i) { return aini * std::pow(aend / aini, std::clamp(i, 0.0, double(nsteps)) / nsteps); };
    for (int i = 0; i &lt;= nsteps; i++) {
        const double a = a_of(i), avel_old = a_of(i - 0.5), avel_new = a_of(i + 0.5), anew = a_of(i + 1);
        auto nleftright = FML::INTERPOLATION::get_extra_slices(&quot;CIC&quot;);
        FML::GRID::FFTWGrid&lt;3&gt; density(Nmesh, nleftright.first, nleftright.second);
        FML::INTERPOLATION::particles_to_grid(
            part.get_particles_ptr(), part.get_npart(), part.get_npart_total(), density, &quot;CIC&quot;);
        density.fftw_r2c();
        std::array&lt;FML::GRID::FFTWGrid&lt;3&gt;, 3&gt; force;
        FML::NBODY::compute_force_from_density_fourier&lt;3&gt;(density, force, &quot;CIC&quot;, 1.5 * a);
        FML::NBODY::KickParticles&lt;3&gt;(force, part, 2.0 / a * (std::sqrt(avel_new) - std::sqrt(avel_old)), &quot;CIC&quot;);
        FML::NBODY::DriftParticles&lt;3, Particle&gt;(part, 2.0 * (1.0 / std::sqrt(a) - 1.0 / std::sqrt(anew)));
    }

    // The power spectrum today against linear theory
    const auto pofk = power_spectrum();
    double max_dev = 0.0;
    if (FML::ThisTask == 0)
        std::cout &lt;&lt; &quot;# k [2pi/box]   P(k) / P_linear(k)\n&quot;;
    for (int i = 0; i &lt; pofk.n; i++) {
        const double ratio = pofk.pofk[i] / (pofk_ini.pofk[i] * std::pow(aend / aini, 2));
        if (FML::ThisTask == 0)
            std::cout &lt;&lt; pofk.kbin[i] / (2 * M_PI) &lt;&lt; &quot; &quot; &lt;&lt; ratio &lt;&lt; &quot;\n&quot;;
        if (pofk.kbin[i] &lt; 2 * M_PI * 2.5)
            max_dev = std::max(max_dev, std::abs(ratio - 1.0));
    }
    if (FML::ThisTask == 0)
        std::cout &lt;&lt; &quot;Largest scales (k &lt;= 2 x 2pi/box): |P/P_linear - 1| &lt;= &quot; &lt;&lt; max_dev &lt;&lt; &quot;\n&quot;;
    return max_dev &lt; 0.05 ? 0 : 1;
}
</pre>
  </div>
</div>
<p><b>Slabs, pencils or a Peano&ndash;Hilbert curve</b> (<a href="https://github.com/HAWinther/FML/blob/master/FML/QuickStart/example/domains.cpp" style="display: inline">domains.cpp</a>). The same power spectrum with the particles on three different domains. The grid is made for the domain of the particles; particles on the Peano&ndash;Hilbert curve are assigned to a slab grid without being moved. The three results agree to rounding.</p>
<div style="text-align: center">
  <div style="width:100%;display: inline-block;text-align: left">
    <pre class="prettyprint" style="overflow-x:auto;white-space:pre;font-size:12px">
// Quick start 3: the same calculation on slabs, on pencils and with the particles on a Peano-Hilbert curve.
//
// A Domain says which MPI task owns which part of the box. The particles are distributed according to it, and the
// grid is made for the domain of the particles: FFTWGrid on slabs, PencilGrid on pencils. Particles on a
// Peano-Hilbert domain (load-balanced ranges of a space-filling curve) are assigned to a slab grid without moving
// them. The power spectrum is the same in all three cases.
#include &lt;FML/ComputePowerSpectra/ComputePowerSpectrum.h&gt;
#include &lt;FML/Interpolation/DistributedDeposit.h&gt;
#include &lt;FML/MPIParticles/MPIParticles.h&gt;
#include &lt;FML/PencilDomain/PencilGrid.h&gt;
#include &lt;FML/RandomGenerator/RandomGenerator.h&gt;

#include &lt;cmath&gt;
#include &lt;iostream&gt;

struct Particle {
    double x[3];
    double * get_pos() { return x; }
    constexpr int get_ndim() const { return 3; }
};
using Binning = FML::CORRELATIONFUNCTIONS::PowerSpectrumBinning&lt;3&gt;;
const int Nmesh = 64;

// P(k) of the particles with CIC on a grid of type Grid, made for the domain of the particles
template &lt;template &lt;int&gt; class Grid&gt;
Binning spectrum(FML::PARTICLE::MPIParticles&lt;Particle&gt; &amp; part) {
    Binning pofk(Nmesh / 2);
    FML::CORRELATIONFUNCTIONS::compute_power_spectrum&lt;3, Particle, Grid&gt;(
        Nmesh, part.get_particles_ptr(), part.get_npart(), part.get_npart_total(), pofk, &quot;CIC&quot;, false, part.get_domain());
    return pofk;
}

int main() {
    // A clustered catalogue (random points displaced by two plane waves, in x and in y), the same on every task
    std::vector&lt;Particle&gt; catalogue(200000);
    FML::RANDOM::RandomGenerator rng(1234);
    for (auto &amp; p : catalogue) {
        for (auto &amp; x : p.x)
            x = rng.generate_uniform();
        p.x[0] = std::fmod(p.x[0] + 0.03 * std::sin(2 * M_PI * p.x[0]) + 1.0, 1.0);
        p.x[1] = std::fmod(p.x[1] + 0.02 * std::sin(4 * M_PI * p.x[1]) + 1.0, 1.0);
    }
    const size_t nalloc = 2 * catalogue.size() / FML::NTasks + 1000;

    // 1. Slabs (the default domain)
    FML::PARTICLE::MPIParticles&lt;Particle&gt; slabs;
    slabs.create(catalogue, nalloc, FML::PARTICLE::Input::Replicated);
    auto p_slab = spectrum&lt;FML::GRID::FFTWGrid&gt;(slabs);

    // 2. Pencils: x and y split over a 2D grid of tasks
    FML::PARTICLE::MPIParticles&lt;Particle&gt; pencils(FML::pencil_domain());
    pencils.create(catalogue, nalloc, FML::PARTICLE::Input::Replicated);
    auto p_pencil = spectrum&lt;FML::GRID::PencilGrid&gt;(pencils);

    // 3. Peano-Hilbert: the particles stay on their tasks; the density is assigned to a slab grid across domains
    FML::PARTICLE::MPIParticles&lt;Particle&gt; hilbert(FML::Domain::uniform_peano_hilbert(FML::NTasks));
    hilbert.create(catalogue, nalloc, FML::PARTICLE::Input::Replicated);
    FML::GRID::FFTWGrid&lt;3&gt; density(Nmesh);
    FML::INTERPOLATION::particles_to_grid_any_domain&lt;3, 2&gt;(
        hilbert.get_particles_ptr(), hilbert.get_npart(), hilbert.get_npart_total(), density);
    density.fftw_r2c();
    FML::INTERPOLATION::deconvolve_window_function_fourier&lt;3&gt;(density, &quot;CIC&quot;);
    Binning p_hilbert(Nmesh / 2);
    FML::CORRELATIONFUNCTIONS::bin_up_power_spectrum&lt;3&gt;(density, p_hilbert);
    for (int i = 0; i &lt; p_hilbert.n; i++) // subtract the shot noise as compute_power_spectrum does
        p_hilbert.pofk[i] -= 1.0 / double(catalogue.size());
    if (FML::ThisTask == 0)
        for (int i = 0; i &lt; 4; i++)
            std::cout &lt;&lt; &quot;k = &quot; &lt;&lt; i + 1 &lt;&lt; &quot; x 2pi/box: P(k) on slabs &quot; &lt;&lt; p_slab.pofk[i] &lt;&lt; &quot;, pencils &quot; &lt;&lt; p_pencil.pofk[i]
                      &lt;&lt; &quot;, Peano-Hilbert &quot; &lt;&lt; p_hilbert.pofk[i] &lt;&lt; &quot;\n&quot;;

    double max_diff = 0.0;
    for (int i = 0; i &lt; p_slab.n; i++)
        max_diff = std::max({max_diff, std::abs(p_pencil.pofk[i] - p_slab.pofk[i]) / std::abs(p_slab.pofk[i]),
                             std::abs(p_hilbert.pofk[i] - p_slab.pofk[i]) / std::abs(p_slab.pofk[i])});
    if (FML::ThisTask == 0)
        std::cout &lt;&lt; &quot;Largest relative difference between the three: &quot; &lt;&lt; max_diff &lt;&lt; &quot;\n&quot;;
    return max_diff &lt; 1e-8 ? 0 : 1;
}
</pre>
  </div>
</div>

