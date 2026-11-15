<hr/>
<header class="entry-header text-center">
  <h1 class="entry-title">Table of contents</h1>
</header>
<hr/>

<p class="page-toc">
<a href="documentation.php">Back to the documentation</a><br>
<a href="#Overview">1. Overview: gas dynamics</a><br>
<a href="#FiniteVolume">2. Finite-volume Euler equations (FML::HYDRODYNAMICS)</a><br>
<a href="#SPH">3. Smoothed particle hydrodynamics (FML::SPH)</a><br>
<a href="#Cosmology">4. Gravity and expanding backgrounds</a><br>
<a href="#Validation">5. Validation and limitations</a><br>
</p>

<hr/>
<header class="entry-header text-center">
  <h1 class="entry-title" id="Overview">Overview: gas dynamics</h1>
</header>
<hr/>

<p>
FML has two independent, header-only solvers for an ideal gas, a grid code and a particle code, which share the equation of state (<tt>Hydrodynamics/EquationOfState.h</tt>):
</p>
<ul>
<li><a href="#FiniteVolume" style="display: inline">$\texttt{FML::HYDRODYNAMICS}$</a>: Riemann solvers (Rusanov, HLLE, HLLC), an exact Riemann solver as a reference, and a conservative finite-volume solver on a uniform Cartesian grid in 1D, 2D and 3D.</li>
<li><a href="#SPH" style="display: inline">$\texttt{FML::SPH}$</a>: conservative density&ndash;energy smoothed particle hydrodynamics with adaptive smoothing lengths in 1D, 2D and 3D.</li>
</ul>
<p>
Both run in serial, with MPI (slabs in $x$, empty tasks allowed, no global gathers) and OpenMP, and need neither FFTW nor GSL. Both take gravity (or any external force) through a callback and can evolve the gas in a prescribed expanding background, but neither is coupled to SimulationCode yet. Units are the user's: any consistent length, mass and time.
</p>

<hr/>
<header class="entry-header text-center">
  <h1 class="entry-title" id="FiniteVolume">Finite-volume Euler equations (FML::HYDRODYNAMICS)</h1>
</header>
<hr/>

<p>
The solver evolves the conserved density, momentum and total energy, $\partial_t U + \nabla\cdot F(U) = S$, with the fluxes at the cell faces from a Riemann solver. Second order in space uses limited linear reconstruction of the primitive variables (generalised minmod/MC slopes), and the time stepping is SSPRK2 with a global CFL time step. A Sod shock tube:
</p>

<pre class="prettyprint">
#include &lt;FML/Hydrodynamics/FiniteVolume.h&gt;
namespace H = FML::HYDRODYNAMICS;

H::Options&lt;1&gt; options;
options.shape = {256};
options.lengths = {1};
options.gamma = 1.4;
// Outflow, Periodic or Reflecting, per axis and side
options.boundaries = {{{H::Boundary::Outflow, H::Boundary::Outflow}}};
H::FiniteVolume&lt;1&gt; gas(options, [](const auto &amp; x) {
    // {rho, v, P}
    return x[0] &lt; 0.5 ? H::Primitive&lt;1&gt;{1, {0}, 1} : H::Primitive&lt;1&gt;{0.125, {0}, 0.1};
});
while (gas.time() &lt; 0.2)
    gas.advance(0.2 - gas.time());
auto cells = gas.primitive_cells();   // local cells
auto totals = gas.diagnostics();      // mass, momentum, energies (collective)
</pre>

<p>
The Riemann solvers can also be used on their own (<tt>H::riemann(left, right, eos, axis, H::FluxMethod::HLLC)</tt>, and <tt>H::ExactRiemann</tt> for the exact solution with shocks, contacts, rarefactions and vacuum). HLLC keeps the contact wave and falls back to HLLE for degenerate states. Positivity is not forced with floors: a step that would give a nonphysical state is recomputed with first-order Rusanov fluxes, and then with a smaller time step. An optional dual-energy formulation evolves the internal energy as well, for high-Mach flows where the pressure from $E - K$ is inaccurate; its corrections to the total energy are recorded. The boundaries (periodic, outflow or reflecting) are set per axis. See the <a href="https://github.com/HAWinther/FML/blob/master/FML/Hydrodynamics/README.md" style="display: inline">module README</a> for the conventions.
</p>

<figure class="science-figure">
  <img src="images/hydro_fv_sod.png" alt="Density, velocity and pressure of the Sod shock tube from the finite-volume solver compared with the exact solution">
  <figcaption>The Sod shock tube at $t = 0.2$ from the finite-volume solver (HLLC, second order) with 128, 256 and 512 cells against the exact Riemann solution.</figcaption>
</figure>

<hr/>
<header class="entry-header text-center">
  <h1 class="entry-title" id="SPH">Smoothed particle hydrodynamics (FML::SPH)</h1>
</header>
<hr/>

<p>
The SPH solver evolves gas particles in a periodic box with the variational density&ndash;energy formulation (see <a href="https://arxiv.org/abs/1012.1885" style="display: inline">Price 2012</a>). The density and the smoothing length of each particle are solved together, $\rho_i = \sum_j m_j W(r_{ij}, h_i)$ with $h_i^{\rm D}\rho_i = \eta^{\rm D} m_i$, with the grad-$h$ correction $\Omega_i$; the kernel is the cubic B-spline with support $2h$. The pair forces are antisymmetric and the energy equation matches them, so momentum and energy are conserved (up to the time integration). Shocks are captured with artificial viscosity (with an optional Balsara switch), and an optional artificial conductivity smooths contact discontinuities. The time stepping is Heun RK2 with a global step.
</p>

<pre class="prettyprint">
#include &lt;FML/SPH/SPH.h&gt;
using namespace FML::SPH;

// position, velocity, mass, specific internal energy, h guess, unique id
std::vector&lt;Particle&lt;3&gt;&gt; local;
Options options;
options.boxsize = 1;
options.gamma = 5.0 / 3;
options.h_min = 0.005;
options.h_max = 0.1;
Solver&lt;3&gt; gas(std::move(local), options);
while (gas.time() &lt; 0.1)
    gas.advance(0.1 - gas.time());
auto fields = gas.evaluate();       // density, pressure, div/curl v, accelerations, ...
auto totals = gas.diagnostics();
</pre>

<p>
The particles can start on any task; they are moved to the slab they are in. Neighbours across the slab boundaries are exchanged as ghosts (also from non-neighbouring slabs when the smoothing lengths are large). See the <a href="https://github.com/HAWinther/FML/blob/master/FML/SPH/README.md" style="display: inline">module README</a>.
</p>

<figure class="science-figure">
  <img src="images/sph_sod.png" alt="Density, velocity and pressure of the Sod shock tube from SPH particles compared with the exact solution">
  <figcaption>The Sod shock tube with SPH at $t = 0.1$ (a periodic double-Sod setup, the interior before the waves from the other interface arrive) at three resolutions ($N$ = 225, 450 and 900), against the exact solution.</figcaption>
</figure>

<hr/>
<header class="entry-header text-center">
  <h1 class="entry-title" id="Cosmology">Gravity and expanding backgrounds</h1>
</header>
<hr/>

<p>
Both solvers take a collective callback that returns an acceleration (for the grid: per cell, added to the momentum and energy equations; for SPH: per particle), so a gravity solver (for example FML's PM forces) or any external force can be connected. In 3D both can also evolve the gas in a prescribed expanding background: a callback gives $a(t)$ and $H(t)$, the coordinates are comoving and the velocities peculiar, and the expansion terms are added (for SPH: $\dot{\bf v} = {\bf a}_{\rm hydro}/a - H{\bf v} + {\bf g}$ and $\dot u = \dot u_{\rm hydro}/a - 3(\gamma-1)Hu$). The cosmological drag and adiabatic cooling of a homogeneous gas are tested against their exact solutions.
</p>

<hr/>
<header class="entry-header text-center">
  <h1 class="entry-title" id="Validation">Validation and limitations</h1>
</header>
<hr/>

<ul>
<li><b>Finite volume</b>: 64&ndash;65 checks on 1, 2 and 4 tasks (all fluxes and axes, exact Sod and star states, vacuum, mixed 3D boundary corners, one-cell slabs and empty tasks, conservation, the fallbacks). Smooth advection and sound waves converge at second order in 1D (lower in 2D and 3D at the tested resolutions), and the Sod errors decrease with resolution in every field. The measured results are in <tt>Hydrodynamics/tests/RESULTS.md</tt>.</li>
<li><b>SPH</b>: 51&ndash;53 checks (kernel normalization and derivatives, density and grad-$h$ roots and full force sums against all-pair references, conservation, Sod, MPI with empty tasks and forced message chunking); see <tt>SPH/tests/RESULTS.md</tt>.</li>
<li><b>Limits</b>: a constant-$\gamma$ ideal gas only (no cooling, heating or general equation of state), a global time step (no individual time steps), no checkpoints for SPH, and no coupled cosmological self-gravity calibration yet: the solvers provide the gas dynamics and the hooks, not a full cosmological hydro code.</li>
</ul>
