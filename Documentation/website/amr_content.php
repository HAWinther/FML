<hr/>
<header class="entry-header text-center">
  <h1 class="entry-title">Table of contents</h1>
</header>
<hr/>

<p class="page-toc">
<a href="documentation.php">Back to the documentation</a><br>
<a href="#Overview">1. Overview: adaptive mesh refinement</a><br>
<a href="#Grid">2. The adaptive grid (FML::GRID::AdaptiveMPIGrid)</a><br>
<a href="#Multigrid">3. Multigrid on the adaptive grid: Poisson, DGP, the symmetron and the dilaton</a><br>
<a href="#AMRCode">4. AMRCode: an AMR N-body code</a><br>
<a href="#Limitations">5. Validation and limitations</a><br>
</p>

<hr/>
<header class="entry-header text-center">
  <h1 class="entry-title" id="Overview">Overview: adaptive mesh refinement</h1>
</header>
<hr/>

<p>
A uniform grid has the same resolution everywhere, so resolving a halo means resolving the empty voids around it just as finely. An adaptively refined grid puts small cells only where they are needed (where there are many particles, or where a field changes quickly), as the codes RAMSES and ENZO do. FML has three layers for this:
</p>
<ul>
<li><a href="#Grid" style="display: inline">$\texttt{AdaptiveMPIGrid}$</a>: an adaptively refined grid on $[0,1)^{\rm NDIM}$, distributed over the MPI tasks along a Peano&ndash;Hilbert curve, with refinement, coarsening, 2:1 balance, load balancing and ghost cells.</li>
<li><a href="#Multigrid" style="display: inline">$\texttt{AMRMultiGridSolver}$</a>: a FAS multigrid solver for (non-linear) PDEs on that grid, with ready-made solvers for the DGP, symmetron and dilaton fifth forces.</li>
<li><a href="#AMRCode" style="display: inline">AMRCode</a>: a simple AMR N-body code for $\Lambda$CDM, the symmetron and the dilaton built on the two, in the spirit of RAMSES.</li>
</ul>

<figure class="science-figure">
  <img src="images/amr_grid.png" alt="A 2D adaptive grid refined around a ring: left, the cells coloured by the MPI task that owns them (contiguous Hilbert-curve segments); right, the cells coloured by refinement level, from 4 far from the ring to 9 on it">
  <figcaption>A 2D grid refined around a moving shell on 6 tasks. Left: the owner of each leaf (contiguous pieces of the Hilbert curve). Right: the refinement level, from 4 (the base grid) to 9 on the shell; neighbouring leaves differ by at most one level (2:1 balance).</figcaption>
</figure>

<hr/>
<header class="entry-header text-center">
  <h1 class="entry-title" id="Grid">The adaptive grid (FML::GRID::AdaptiveMPIGrid)</h1>
</header>
<hr/>

<p>
The grid is a <b>linear tree</b>: only the leaves are stored, sorted along a Peano&ndash;Hilbert curve, and every task owns one contiguous piece of the curve. A cell is a key, the integer coordinates of its corner and a level; a cell at level $l$ covers a range of $2^{{\rm NDIM}(L_{\rm max}-l)}$ keys. All geometry (children, neighbours, periodic wrap) is done on the integer coordinates, and the key is only used for sorting, partitioning and finding the leaf that contains a point. The keys are FML's one Hilbert-curve implementation (<tt>PeanoHilbertDomain/HilbertCurve.h</tt>), the same as for the particle domain.
</p>

<pre class="prettyprint">
#include &lt;FML/AdaptiveMPIGrid/AdaptiveMPIGrid.h&gt;
// MyCellData must be trivially copyable
using Grid = FML::GRID::AdaptiveMPIGrid&lt;3, MyCellData&gt;;

Grid grid(levelmin, levelmax, {true, true, true});          // periodic per axis
for (IndexInt i = 0; i &lt; grid.get_n_local(); i++) {
    grid.get_data(i) = ...;
    grid.set_flag(i, Grid::REFINE);                          // or COARSEN / KEEP
}
grid.adapt(prolong, restrict);   // 2:1 balance + refine/coarsen + rebuild and fill the ghosts
grid.load_balance(weight);       // repartition by weight, migrate, rebuild the ghosts
grid.exchange_ghosts();          // after changing data

// Overlap communication with computation
auto h = grid.begin_exchange_ghosts();
for (auto i : grid.get_interior_cells()) ...;   // all neighbours are local
h.wait();
for (auto i : grid.get_boundary_cells()) ...;   // may use ghosts

auto nb = grid.get_face_neighbors(i, dim, +1);   // BOUNDARY / COARSER / SAME / FINER
grid.find_leaf(std::array&lt;double, 3&gt;{x, y, z}); // the leaf containing a point

// Particles follow the grid: put them on the task of the leaf they are in
particles.set_domain(grid.get_domain());
</pre>

<p>
<tt>adapt</tt> enforces 2:1 balance across faces, edges and corners, so neighbouring leaves differ by at most one level. Partition cuts never fall inside a family of siblings, so the leaf set does not depend on the number of tasks (the tests check this with a global hash on 1&ndash;24 tasks). There is one layer of ghosts (every remote leaf touching a local one), split so that the exchange can overlap the work on the interior. Irregular exchanges use the NBX algorithm (Hoefler et al. 2010) on many tasks, so there is no $O(N_{\rm tasks})$ collective. <tt>truncated(L)</tt> gives the grid with every leaf finer than $L$ replaced by its ancestor (and uniform grids below the base grid), which is the hierarchy the multigrid solver works on. The full design is in the <a href="https://github.com/HAWinther/FML/blob/master/FML/AdaptiveMPIGrid/README.md" style="display: inline">module README</a>.
</p>

<hr/>
<header class="entry-header text-center">
  <h1 class="entry-title" id="Multigrid">Multigrid on the adaptive grid: Poisson, DGP, the symmetron and the dilaton</h1>
</header>
<hr/>

<p>
<tt>AMRMultiGridSolver</tt> (<tt>MultigridSolver/AMRMultiGridSolver.h</tt>) is a Full Approximation Storage multigrid solver for equations $L(f) = 0$ on an <tt>AdaptiveMPIGrid</tt>, written like the <a href="multigridsolver.php" style="display: inline">uniform-grid solver</a>: the equation returns $L$ and $\partial L/\partial f$ in a cell.
</p>

<pre class="prettyprint">
#include &lt;FML/MultigridSolver/AMRMultiGridSolver.h&gt;
using namespace FML::SOLVERS::AMRMULTIGRIDSOLVER;

AMRMultiGridSolver&lt;3&gt; solver(grid);             // after adapt / load_balance
int id = solver.add_external_field(delta);       // per local leaf
AMREquation&lt;3&gt; eq = [id](const AMRMultiGridSolver&lt;3&gt; &amp; sol, int level, IndexInt i) {
    auto [lap, dlap] = sol.get_laplacian(level, i);
    return std::pair&lt;double, double&gt;{lap - sol.get_external_field(id, level, i), dlap};
};
auto result = solver.solve(eq);                  // result.converged(), .iterations, .residual
const auto &amp; phi = solver.get_solution();       // per local leaf
auto force = solver.get_solution_gradient();     // D phi per leaf
</pre>

<p>
The levels are the grid truncated at every refinement level and uniform grids below the base grid. On each level only the finest cells are relaxed (red-black Newton&ndash;Gauss&ndash;Seidel; the multi-level adaptive technique), and the FAS source carries the effect of the refinement to the coarser levels, so the cost of a V-cycle is proportional to the number of leaves. The Laplacian is flux-conservative and second order across coarse/fine faces (gradient-corrected ghost values), so its volume integral vanishes exactly on a periodic grid. <b>Partial solves</b> only update the leaves at levels $\geq L$ and keep the coarser ones fixed as Dirichlet values, which is what an N-body code with block time steps needs at its sub-steps (as RAMSES).
</p>

<p>
Three fifth-force solvers are built on it, with the parameters, equations and units of their uniform-grid versions in <a href="nbody.php" style="display: inline">FML::NBODY</a>:
</p>
<ul>
<li><tt>AMRDGPSolverCosmology&lt;3&gt;</tt> (<tt>AMRDGPSolver.h</tt>): the normal-branch DGP equation. Against the exact force of a spherical compensated overdensity it is second order (rms force error $10^{-2}$ on a uniform level-6 grid, $2.6\cdot 10^{-3}$ with levels 4&ndash;7 refined around the object), and on a uniform grid it equals <tt>DGPSolverCosmology</tt> to $7\cdot 10^{-11}$.</li>
<li><tt>AMRSymmetronSolverCosmology&lt;3&gt;</tt> (<tt>AMRSymmetronSolver.h</tt>): the generalised $(N, M)$ symmetron, $\nabla^2\chi \propto \chi^{N-1}[(a_{\rm ssb}/a)^3(1+\delta) - 1] + \chi^{M-1}$, started from the background (or the previous field).</li>
<li><tt>AMRDilatonSolverCosmology&lt;3&gt;</tt> (<tt>AMRDilatonSolver.h</tt>): the generalised dilaton (<a href="https://arxiv.org/abs/1206.3568" style="display: inline">Brax et al. 2012</a>), $\nabla^2 w = 3A_2\Omega_m a^2(H_0B)^2\, w\,[(1+\delta)a^{-3} - T(\ln w)]$ for $w = \phi/\bar\phi(a)$, with the fifth-force potential $\Phi_F = \bar\Phi_F(a) w^2$. As in the uniform solver the multigrid works with $w$ while the cell update solves for $\ln w$ (where the local equation is monotone), so $w$ stays positive across many e-folds; this uses two new solver options, a custom cell update (<tt>set_custom_update</tt>) and minmod-limited coarse/fine ghosts (<tt>set_coarse_fine_limiter</tt>) that keep a positive field positive. Against a radial reference for a screened overdensity it is second order (rms error of $w$ $5\cdot 10^{-4}$ with levels 5&ndash;7), and on a uniform grid it equals <tt>DilatonSolverCosmology</tt> to $5\cdot 10^{-14}$.</li>
</ul>
<p>
Examples and tests: <tt>MultigridSolver/examples/AMRPoissonExample.cpp</tt>, <tt>AMRDGPExample.cpp</tt>, <tt>AMRSymmetronExample.cpp</tt> and <tt>AMRDilatonExample.cpp</tt> (<tt>make amrpoisson amrdgp amrsymmetron amrdilaton</tt>). See the <a href="https://github.com/HAWinther/FML/blob/master/FML/MultigridSolver/AMRMultiGridSolver.md" style="display: inline">solver notes</a>.
</p>

<hr/>
<header class="entry-header text-center">
  <h1 class="entry-title" id="AMRCode">AMRCode: an AMR N-body code</h1>
</header>
<hr/>

<p>
<a href="https://github.com/HAWinther/FML/tree/master/FML/AMRCode" style="display: inline">FML/AMRCode</a> is a simple adaptive-mesh-refinement N-body code for flat $\Lambda$CDM, symmetron and dilaton gravity, in the spirit of RAMSES:
</p>
<pre class="prettyprint">
cd FML/AMRCode
make &amp;&amp; mpirun -np 4 ./amrnbody parameterfile.txt
</pre>
<ul>
<li><b>Initial conditions</b>: 1LPT/2LPT from a linear $P(k)$ file (or Eisenstein &amp; Hu), with the growth rates of the flat $\Lambda$CDM background.</li>
<li><b>Refinement</b>: a leaf is refined when it holds more than <tt>n_refine</tt> particles (cloud-in-cell mass, as RAMSES) and a family is merged when together it holds at most that; then the grid is load balanced and the particles follow its Peano&ndash;Hilbert domain.</li>
<li><b>Gravity</b>: cloud-in-cell assignment on the tree (mass is conserved exactly across levels), $\nabla^2\phi = \delta$ with the AMR multigrid solver, and the force from the 4th-order gradient where the neighbours are at the same level (2nd order otherwise).</li>
<li><b>Time stepping</b>: kick-drift-kick in $\ln a$ with block time steps: a particle in a leaf at level $l$ takes steps $\Delta t_{\rm coarse}/2^{l-l_{\rm min}}$, and at a sub-step Poisson is only solved on the levels that are active (a partial solve).</li>
<li><b>Symmetron</b> (<tt>gravity_model = Symmetron</tt>, with <tt>symmetron_assb</tt>, <tt>symmetron_beta</tt>, <tt>symmetron_L_mpch</tt>, <tt>symmetron_N</tt>, <tt>symmetron_M</tt>): the scalar field of the quasi-static symmetron equation, with screening, is solved on the same leaves with the same multigrid, with full and partial solves just like the potential (so at every sub-step), and its fifth force is added to gravity. Before symmetry breaking the run is $\Lambda$CDM.</li>
<li><b>Dilaton</b> (<tt>gravity_model = Dilaton</tt>, with <tt>dilaton_A2</tt>, <tt>dilaton_beta0</tt>, <tt>dilaton_r</tt>, <tt>dilaton_xi</tt>, <tt>dilaton_start_redshift</tt>, <tt>dilaton_epsilon</tt>): the field $w = \phi/\bar\phi$ is solved on the leaves exactly like the symmetron (full and partial solves). Before <tt>dilaton_start_redshift</tt> (default 4) the run is $\Lambda$CDM, since the coupling is small and the field contrasts on the particle lattice are extreme.</li>
<li><b>Outputs</b>: $P(k)$ at the output redshifts, optionally Gadget snapshots and a PNG of the particles in a slab (<tt>output_images</tt>). A parameter file can <tt>include = file</tt> another, and keys that are not parameters give a warning.</li>
</ul>
<pre class="prettyprint">
gravity_model    = Symmetron
symmetron_assb   = 0.5    # symmetry breaking at z = 1
symmetron_beta   = 1.0    # coupling
symmetron_L_mpch = 1.0    # range in vacuum (Mpc/h)
</pre>
<p>
With these parameters ($64^3$ particles in 128 Mpc/$h$) $P_{\rm Symmetron}/P_{\Lambda\rm CDM}$ at $z = 0$ is 1.023 and 1.031 at $k = 0.3$ and $0.5\,h$/Mpc, against 1.019 and 1.027 from SimulationCode with its exact symmetron solver on a uniform grid. On smaller scales it stays near 1.03 while the uniform-grid run reaches 1.09 at $k = 2\,h$/Mpc: the AMR resolves the halos, inside which the fifth force is screened.
</p>
<p>
The dilaton ($64^3$ in 128 Mpc/$h$, $\beta_0 = 0.5$, $r = 1$, $\xi = 10^{-3}$) agrees with SimulationCode's exact solver to 0.5% in $P/P_{\Lambda\rm CDM}$ up to $k = 1\,h$/Mpc (for $A_2 = 10^5$: 1.046 against 1.042 at $k = 0.3$ and 1.078 against 1.079 at $k = 1\,h$/Mpc) and is again lower on smaller scales, where the AMR resolves the screened halos. It costs about 70% more time than $\Lambda$CDM.
</p>
<p>
A $64^3$ $\Lambda$CDM run in a 100 Mpc/$h$ box with levels 6&ndash;10 runs to $z = 0$ in about a minute on 2 cores. Large scales follow linear theory (the growth factor agrees with CAMB to $5\cdot 10^{-4}$), $P(k)$ on 1 and 2 tasks agrees to 0.4%, and the refinement adds the small-scale power a uniform grid misses (a factor 1.35 at $k = 1$ and 3.3 at $k = 3\,h$/Mpc). See the <a href="https://github.com/HAWinther/FML/blob/master/FML/AMRCode/README.md" style="display: inline">AMRCode README</a> for the details and timings.
</p>

<hr/>
<header class="entry-header text-center">
  <h1 class="entry-title" id="Limitations">Validation and limitations</h1>
</header>
<hr/>

<ul>
<li><b>Grid</b>: identical leaf sets on 1&ndash;24 tasks, invariants (coverage, ownership, ghosts, 2:1 balance) checked after every <tt>adapt</tt> and <tt>load_balance</tt>, and clean under AddressSanitizer. <tt>adapt</tt> changes a leaf by at most one level per call, and there is no OpenMP yet.</li>
<li><b>AMRCode tests</b>: the deposit conserves the mass to $10^{-15}$; the force of a Gaussian blob of particles goes from 50% rms error on the uniform base grid to 1.5% with levels 5&ndash;9 (the particle noise); a Zel'dovich plane wave just before shell crossing is followed to 1% (0.15% between subcycled and non-subcycled steps); partial solves recover the full solution to $10^{-10}$.</li>
<li><b>AMRCode limits</b>: flat $\Lambda$CDM, the symmetron and the dilaton (no massive neutrinos; the solver also has DGP), a fixed coarse step, no OpenMP, and partial solves next to large fixed regions can stagnate (they are then redone as full solves). Memory is about 2.7 kB per particle for $\Lambda$CDM and 4.8 kB with the symmetron, most of it in the multigrid levels, so a $256^3$ run needs some 50&ndash;90 GB.</li>
</ul>
