<hr/>
<header class="entry-header text-center">
  <h1 class="entry-title">Table of contents</h1>
</header>
<hr/>

<p class="page-toc">
<a href="index.php">Back to the main page</a><br>
<a href="documentation.php">Back to the documentation</a><br>
<a href="#Overview">1. Overview: slabs and pencils</a><br>
<a href="#Layout">2. The pencil layout</a><br>
<a href="#Using">3. Using a PencilGrid</a><br>
<a href="#Codes">4. Running FML on pencils</a><br>
<a href="#Generic">5. Writing code for slabs and pencils</a><br>
<a href="#Complex">6. Complex grids: CFFTWGrid and CPencilGrid</a><br>
<a href="#Tools">7. Ghost layers, cell owners and moving Fourier modes</a><br>
<a href="#Domains">8. Particle domains</a><br>
<a href="#Performance">9. Performance and validation</a><br>
</p>

<header class="entry-header text-center">
  <h1 class="entry-title" id="Overview">Overview: slabs and pencils</h1>
</header>
<hr/>

<p>
A <a href="fftwgrid.php" style="display: inline">$\texttt{FFTWGrid}$</a> is split over the MPI tasks in slabs along $x$. That is simple and fast, but a slab decomposition can use at most $N_{\rm mesh}$ tasks (one $x$-slice each) and FFTW-MPI needs the number of tasks to divide $N_{\rm mesh}$. A <b>pencil</b> decomposition splits two axes over a $P_0\times P_1$ grid of tasks, so up to $N_{\rm mesh}^2$ tasks can share a grid. A $\texttt{PencilGrid}$ made without a domain spreads any $N_{\rm mesh}$ over its pencils; on the uniform pencils of <tt>FML::pencil_domain(px, py)</tt> (what the codes use) $N_{\rm mesh}$ must be divisible by $p_x$ and $p_y$.
</p>
<p>
$\texttt{PencilGrid<N>}$ ($N = 2, 3$; <tt>PencilDomain/PencilGrid.h</tt>) is a real field on pencils with in-place real-to-complex FFTs by MPI transposes. It has the same interface and conventions as $\texttt{FFTWGrid}$: positions in $[0,1)$, wave-vectors $2\pi n$, the forward transform normalized by $1/N_{\rm mesh}^N$, local coordinates and indices, range loops over indices. The grid code of FML is written against this common interface (<tt>FFTWGrid/GridInterface.h</tt>), so the library, the N-body code and the snapshot analyzer run on slabs or on pencils with the same results. Slabs are the default; nothing changes for existing code.
</p>
<p>
On few tasks slabs are a little faster (FFTW does no transposes there). Use pencils when the number of tasks approaches $N_{\rm mesh}$. Pencils need <tt>USE_MPI</tt> and <tt>USE_FFTW</tt>.
</p>

<header class="entry-header text-center">
  <h1 class="entry-title" id="Layout">The pencil layout</h1>
</header>
<hr/>

<figure class="science-figure">
<img src="images/pencil-mpi.svg" alt="A 2 by 2 process grid owns real-space x-y pencils; transposes redistribute the data to ky-kz pencils in Fourier space; ghost cells are filled from the neighbouring ranks">
<figcaption>A $2\times 2$ process grid. In position space each task has an $x$-$y$ pencil (all $z$); the transform redistributes the data so that in Fourier space each task has a $k_y$-$k_z$ pencil (all $k_x$). Ghost (extra) cells are filled from the neighbours, corners included.</figcaption>
</figure>

<ul>
<li><b>3D</b>: the task with process coordinates $(p_0, p_1)$ has the cells with $x$ in its $x$-range, $y$ in its $y$-range and all $z$. In Fourier space it has all $k_x$, its part of $k_y\in[0,N)$ (split over $P_0$) and its part of $k_z\in[0,N/2]$ (split over $P_1$).</li>
<li><b>2D</b>: $P_1 = 1$ (slabs in $x$); task $p_0$ has its $x$-range and all $y$, in Fourier space all $k_x$ and its part of $k_y\in[0,N/2]$.</li>
<li>The ranges are those of a $\texttt{PencilDomain}$ whose edges lie on cells. Without one: the process grid of <tt>FML::pencil_domain()</tt> (as square as possible) with the edges on cells as even as possible.</li>
<li>The transform: real-to-complex FFTs along $z$, a transpose within the row of tasks and FFTs along $y$, a transpose within the column and FFTs along $x$. A transpose packs a block for each task and exchanges them with <tt>MPI_Alltoallv</tt>.</li>
<li>The extra cells (<tt>n_extra_left</tt>, <tt>n_extra_right</tt>) are on both sides in $x$ and, in 3D, in $y$. <tt>communicate_boundaries()</tt> fills them; <tt>add_contribution_from_extra_cells()</tt> adds them to their owners (after assigning particles).</li>
</ul>

<header class="entry-header text-center">
  <h1 class="entry-title" id="Using">Using a PencilGrid</h1>
</header>
<hr/>

<p>
The particles live on the task of their cells, so set the global domain to pencils once at the start and make every grid on it:
</p>
<div style="text-align: center">
<div style="width:760px;display: inline-block;text-align: left">
<pre class="prettyprint">
FML::set_domain(FML::pencil_domain());     // or FML::pencil_domain(px, py)

// A grid on FML::domain (make_grid works for FFTWGrid, PencilGrid, ...)
using Grid = FML::GRID::PencilGrid&lt;3&gt;;
auto grid = FML::GRID::make_grid&lt;Grid&gt;(Nmesh, n_extra_left, n_extra_right);

grid.fill_real_grid([](std::array&lt;double, 3&gt; &amp; x) { return std::sin(2 * M_PI * x[0]); });
grid.fftw_r2c();                            // normalized by 1/Nmesh^3
for (auto &amp;&amp; index : grid.get_fourier_range()) {
    std::array&lt;double, 3&gt; kvec;
    double kmag;
    grid.get_fourier_wavevector_and_norm_by_index(index, kvec, kmag);
    grid.set_fourier_from_index(index, grid.get_fourier_from_index(index) * f(kmag));
}
grid.fftw_c2r();
grid.communicate_boundaries();

FML::PARTICLE::MPIParticles&lt;Particle&gt; part;   // on FML::domain when created
</pre>
</div>
</div>
<p>
The methods have the names and meaning of $\texttt{FFTWGrid}$. The differences: position coordinates are local in $x$ and $y$; Fourier coordinates $(k_x, i_{k_y}, i_{k_z})$ are global in $k_x$ and local in $k_y, k_z$; the Fourier rows are lines along $k_x$. <tt>dump_to_file</tt> / <tt>load_from_file</tt> write and read checkpoints, and <tt>storage_bytes(Nmesh, domain)</tt> gives the memory of a grid without allocating it.
</p>

<header class="entry-header text-center">
  <h1 class="entry-title" id="Codes">Running FML on pencils</h1>
</header>
<hr/>

<table class="table table-striped table-condensed">
<thead><tr><th>Code</th><th>How to get pencils</th></tr></thead>
<tbody>
<tr><td><a href="nbodycode.php" style="display: inline">SimulationCode</a></td><td><tt>make USE_PENCIL_GRID=true</tt>. The pencils are <tt>FML_PENCILS="px py"</tt> from the environment, or as square as possible. Every gravity model, COLA, lightcones, neutrinos, ICs and every output and analysis work on both.</td></tr>
<tr><td><a href="snapshotanalyzer.php" style="display: inline">SnapshotAnalyzer</a></td><td><tt>make USE_PENCIL_GRID=true</tt> (the same <tt>FML_PENCILS</tt>); every analysis works on both.</td></tr>
<tr><td>FML/Analysis</td><td><tt>-DFML_ANALYSIS_PENCIL_GRID</tt> (the grids are <tt>AnalysisGrid&lt;N&gt;</tt>).</td></tr>
<tr><td><a href="survey.php" style="display: inline">Survey</a></td><td><tt>-DFML_SURVEY_PENCIL_MESH</tt>: the survey meshes on the pencils of <tt>FML::domain</tt>.</td></tr>
<tr><td>Library functions on grids</td><td>Templated on the grid: pass a $\texttt{PencilGrid}$. Power spectra, polyspectra and multipoles, random fields, LPT, smoothing, Hessians, NUFFT, the multigrid solvers, voids, morphology, cosmic web, hybrid bias, sheet interpolation, glass ICs and more. Some take the grid type as a template argument, e.g. <tt>RSDReconstructionFourierMethod&lt;N, T, FML::GRID::PencilGrid&gt;</tt>.</td></tr>
<tr><td><a href="lpt.php#ArbitraryOrder" style="display: inline">EdS arbitrary-order LPT</a></td><td><tt>compute_eds_lpt(input, options)</tt> with a $\texttt{PencilGrid}$ input: computed natively on pencils.</td></tr>
<tr><td>Particle methods</td><td>On the domain of the particles: <a href="fof.php" style="display: inline">FoF</a> (<tt>FoFDomain.h</tt>), <a href="tesselation.php" style="display: inline">Delaunay</a> and <a href="dtfe.php" style="display: inline">DTFE</a> (<tt>MPIPeriodicDelaunay(domain)</tt>, <tt>DTFE(domain)</tt>), SPH, the GADGET / HDF5 / RAMSES readers.</td></tr>
<tr><td>Wave dark matter</td><td><tt>FML::WAVEDM::SolverT&lt;FML::GRID::CPencilGrid&lt;3&gt;&gt;</tt> (see <a href="#Complex" style="display: inline">below</a>).</td></tr>
<tr><td>Flat-sky weak lensing</td><td><tt>shear_from_convergence</tt> on a <tt>PencilGrid&lt;2&gt;</tt>.</td></tr>
</tbody>
</table>
<p>
Not yet on pencils: the field-level inference code (only its distributed mesh assignment).
</p>

<header class="entry-header text-center">
  <h1 class="entry-title" id="Generic">Writing code for slabs and pencils</h1>
</header>
<hr/>

<p>
Write the function once, templated on the grid, with a fast path for $\texttt{FFTWGrid}$ where it matters:
</p>
<div style="text-align: center">
<div style="width:760px;display: inline-block;text-align: left">
<pre class="prettyprint">
template &lt;int N, template &lt;int&gt; class Grid = FML::GRID::FFTWGrid&gt;
void my_filter(Grid&lt;N&gt; &amp; grid) {
    for (FML::IndexIntType irow = 0; irow &lt; grid.get_local_nrows_fourier(); irow++)
        for (auto &amp;&amp; index : grid.get_fourier_range_of_rows(irow, irow + 1)) {
            // Never decode a Fourier index yourself: ask the grid
            auto k = grid.get_fourier_integer_wavevector_from_index(index);
            ...
        }
    if constexpr (FML::GRID::is_fftwgrid_v&lt;Grid&lt;N&gt;&gt;) {
        // slab-only fast path
    }
}
</pre>
</div>
</div>
<p>The rules (the full list is at the top of <tt>FFTWGrid/GridInterface.h</tt>):</p>
<ul>
<li>Make grids with <tt>make_grid&lt;Grid&gt;(Nmesh, left, right, domain)</tt> or <tt>make_grid_like(grid, ...)</tt> on the domain of the particles, not with a constructor.</li>
<li>Ask the grid what it has: <tt>get_local_range(axis)</tt> = <tt>{first, count}</tt> in global cells (all of an axis that is not split), <tt>has_dc_mode(grid)</tt> for $k = 0$, <tt>CellOwners</tt> for the task of any cell.</li>
<li>The last component of the integer wave-vector is in $[0, N/2]$ (real-to-complex): a mode counts with weight <tt>fourier_mode_multiplicity(k[N-1], Nmesh)</tt>, or loop with <tt>for_each_nonzero_fourier_mode</tt>.</li>
<li>Never use a position-space index of one grid in a grid with different extra cells: a $\texttt{PencilGrid}$ index depends on them (the $y$-extras are inside each $x$-slice). Go through the coordinates. Fourier indices do not depend on the extra cells.</li>
<li>Particles that move must be communicated along every split axis ($y$ too).</li>
</ul>

<header class="entry-header text-center">
  <h1 class="entry-title" id="Complex">Complex grids: CFFTWGrid and CPencilGrid</h1>
</header>
<hr/>

<p>
For complex fields (e.g. a wave function) there are complex-to-complex grids with the same interface: $\texttt{CFFTWGrid<N>}$ (<tt>FFTWGrid/CFFTWGrid.h</tt>, FFTW-MPI slabs) and $\texttt{CPencilGrid<N>}$ (<tt>PencilDomain/CPencilGrid.h</tt>, pencils). The values in position space are complex, and the Fourier grid has all $N_{\rm mesh}^N$ modes: every mode has multiplicity 1 and the integer wave-vector is in $(-N/2, N/2]$ along every axis. The transforms are <tt>fftw_forward()</tt> (normalized by $1/N_{\rm mesh}^N$) and <tt>fftw_backward()</tt>. $\texttt{CFFTWGrid}$ also accepts FFTW's uneven slabs (any $N_{\rm mesh}$).
</p>
<p>
The Schr&ouml;dinger&ndash;Poisson (wave dark matter) solver in <tt>FML/WaveDarkMatter</tt> runs on either: <tt>FML::WAVEDM::Solver</tt> on slabs, <tt>SolverT&lt;FML::GRID::CPencilGrid&lt;3&gt;&gt;</tt> on pencils. Its checkpoints are independent of the decomposition: a run written on slabs can be restarted on pencils and on another number of tasks.
</p>

<header class="entry-header text-center">
  <h1 class="entry-title" id="Tools">Ghost layers, cell owners and moving Fourier modes</h1>
</header>
<hr/>

<ul>
<li><tt>GhostedCells&lt;T, N&gt;</tt> (<tt>FFTWGrid/GhostedCells.h</tt>): data of any type per cell (labels, flags, indices, ...) with ghost layers from the neighbouring tasks, on slabs and pencils alike. <tt>at(coord)</tt> reads a cell by local coordinates (ghosts included), <tt>communicate()</tt> fills the ghost layers, corners included. The voxel void finder stores its watershed labels this way.</li>
<li><tt>CellOwners&lt;N&gt;(grid).owner(global_coord)</tt>: the task that has a cell (FFTWGrid, PencilGrid, ...).</li>
<li><tt>copy_fourier_modes(src, dst, factor)</tt> (<tt>FFTWGrid/CopyFourierModes.h</tt>): copy the Fourier modes between grids of any size, decomposition and type (each mode is sent to its owner; the modes inside the Nyquist cube of the smaller mesh). <tt>redistribute_fourier_grid(src, dst)</tt> copies a field to the same mesh decomposed differently, <tt>resize_fourier_grid(grid, Nmesh)</tt> changes the resolution.</li>
</ul>

<header class="entry-header text-center">
  <h1 class="entry-title" id="Domains">Particle domains</h1>
</header>
<hr/>

<p>
How particles are split over the tasks is an <tt>FML::Domain</tt>: slabs in $x$ (<tt>Domain::uniform_slab</tt>), pencils in $x$-$y$ (<tt>Domain::uniform_pencil(px, py)</tt>, <tt>FML::pencil_domain()</tt>) or a Peano&ndash;Hilbert partition (load balanced, e.g. from the adaptive grid). <a href="mpiparticles.php" style="display: inline">$\texttt{MPIParticles}$</a> works with all of them: <tt>create</tt>, <tt>communicate_particles</tt> and <tt>set_domain</tt> (move the particles to a new domain). The particle lattice (<tt>create_particle_grid</tt>) is made on Slab and Pencil domains (for another domain: make it on slabs and <tt>set_domain</tt>). The grid code needs the particles on the domain of its grids: Slab or Pencil.
</p>

<header class="entry-header text-center">
  <h1 class="entry-title" id="Performance">Performance and validation</h1>
</header>
<hr/>

<ul>
<li>A forward plus inverse FFT on 2 tasks at $N_{\rm mesh} = 64$&ndash;$256$ takes 0.9&ndash;1.3 times the time of $\texttt{FFTWGrid}$ (<tt>PencilDomain/tests/benchmark_pencil_grid</tt>); on one task up to 1.9 times. A whole N-body run on 2 tasks ($96^3$ particles and mesh, PM and COLA with the in-run power spectra) takes 1.05&ndash;1.2 times the slab time. Pencils pay off on many tasks, which these small tests do not show.</li>
<li>On slabs the results are bit-identical to those of the slab-only code (the N-body regression suite, the snapshot analyzer, the wave solver, the EdS LPT and the reconstruction).</li>
<li>On pencils every module is compared with slabs for every process grid of 1&ndash;6 tasks (<tt>tests/test_*_pencil</tt>, <tt>tests/test_*_domain</tt>, <tt>tests/test_complex_grids</tt>, the wave-dark-matter and weak-lensing tests built for both) and agrees to round-off; the N-body regression runs on pencils too.</li>
</ul>
