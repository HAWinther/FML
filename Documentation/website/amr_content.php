<hr/>
<header class="entry-header text-center">
  <h1 class="entry-title">Table of contents</h1>
</header>
<hr/>

<p class="page-toc">
<a href="documentation.php">Back to the documentation</a><br>
<a href="#Overview">1. Overview: adaptive mesh refinement</a><br>
<a href="#Grid">2. The adaptive grid (FML::GRID::AdaptiveMPIGrid)</a><br>
<a href="#Multigrid">3. Multigrid on the adaptive grid: Poisson, DGP and the symmetron</a><br>
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
<li><a href="#Multigrid" style="display: inline">$\texttt{AMRMultiGridSolver}$</a>: a FAS multigrid solver for (non-linear) PDEs on that grid, with ready-made solvers for the DGP and symmetron fifth forces.</li>
<li><a href="#AMRCode" style="display: inline">AMRCode</a>: a simple AMR N-body code for $\Lambda$CDM built on the two, in the spirit of RAMSES.</li>
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
  <h1 class="entry-title" id="Multigrid">Multigrid on the adaptive grid: Poisson, DGP and the symmetron</h1>
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
Two fifth-force solvers are built on it, with the parameters, equations and units of their uniform-grid versions in <a href="nbody.php" style="display: inline">FML::NBODY</a>:
</p>
<ul>
<li><tt>AMRDGPSolverCosmology&lt;3&gt;</tt> (<tt>AMRDGPSolver.h</tt>): the normal-branch DGP equation. Against the exact force of a spherical compensated overdensity it is second order (rms force error $10^{-2}$ on a uniform level-6 grid, $2.6\cdot 10^{-3}$ with levels 4&ndash;7 refined around the object), and on a uniform grid it equals <tt>DGPSolverCosmology</tt> to $7\cdot 10^{-11}$.</li>
<li><tt>AMRSymmetronSolverCosmology&lt;3&gt;</tt> (<tt>AMRSymmetronSolver.h</tt>): the generalised $(N, M)$ symmetron, $\nabla^2\chi \propto \chi^{N-1}[(a_{\rm ssb}/a)^3(1+\delta) - 1] + \chi^{M-1}$, started from the background (or the previous field).</li>
</ul>
<p>
Examples and tests: <tt>MultigridSolver/examples/AMRPoissonExample.cpp</tt>, <tt>AMRDGPExample.cpp</tt> and <tt>AMRSymmetronExample.cpp</tt> (<tt>make amrpoisson amrdgp amrsymmetron</tt>). See the <a href="https://github.com/HAWinther/FML/blob/master/FML/MultigridSolver/AMRMultiGridSolver.md" style="display: inline">solver notes</a>.
</p>

<hr/>
<header class="entry-header text-center">
  <h1 class="entry-title" id="AMRCode">AMRCode: an AMR N-body code</h1>
</header>
<hr/>

<p>
<a href="https://github.com/HAWinther/FML/tree/master/FML/AMRCode" style="display: inline">FML/AMRCode</a> is a simple adaptive-mesh-refinement N-body code for flat $\Lambda$CDM, in the spirit of RAMSES:
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
<li><b>Outputs</b>: $P(k)$ at the output redshifts and optionally Gadget snapshots.</li>
</ul>
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
<li><b>AMRCode limits</b>: flat $\Lambda$CDM only (no massive neutrinos or modified gravity in the code, although the solver has DGP and the symmetron), a fixed coarse step, no OpenMP, and partial solves next to large fixed regions can stagnate (they are then redone as full solves).</li>
</ul>
