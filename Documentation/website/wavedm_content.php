<hr/>
<header class="entry-header text-center">
  <h1 class="entry-title">Table of contents</h1>
</header>
<hr/>

<p class="page-toc">
<a href="documentation.php">Back to the documentation</a><br>
<a href="#Overview">1. Overview: wave (fuzzy) dark matter</a><br>
<a href="#Equations">2. The Schr&ouml;dinger&ndash;Poisson equations and units</a><br>
<a href="#Use">3. Using the solver (FML::WAVEDM)</a><br>
<a href="#Stepping">4. Time stepping and expansion</a><br>
<a href="#Hooks">5. Initial conditions, external matter and diagnostics</a><br>
<a href="#Validation">6. Validation and limitations</a><br>
</p>

<hr/>
<header class="entry-header text-center">
  <h1 class="entry-title" id="Overview">Overview: wave (fuzzy) dark matter</h1>
</header>
<hr/>

<figure class="cinematic-figure" id="cinematic-09-ultralight-dark-matter-wavedm">
  <a href="images/cinematic/09-ultralight-dark-matter.jpg" aria-label="View full-size illustration: Ultralight dark matter">
    <img src="images/cinematic/09-ultralight-dark-matter.jpg" width="1774" height="887" loading="lazy" decoding="async" alt="Illustrative wave-dark-matter density fringes surrounding a bright extended central core">
  </a>
  <figcaption><strong>Ultralight dark matter.</strong> Conceptual rendering of interference fringes around a solitonic core. <span class="cinematic-credit">AI-generated concept illustration.</span></figcaption>
</figure>

<p>
If dark matter is an ultralight scalar field (mass $\sim 10^{-22}$ eV), its de Broglie wavelength is of the size of a galaxy core, and on those scales it behaves as a wave: it forms solitonic cores in halos and interference patterns around them. <tt>FML/WaveDarkMatter</tt> evolves such a field with the Schr&ouml;dinger&ndash;Poisson equations in a periodic 3D box with a pseudo-spectral split-step method, in a static or prescribed expanding background. It runs in serial, with MPI (on slabs or <a href="pencilgrid.php" style="display: inline">pencils</a>) and OpenMP, and has checkpoints. It is a library: it is not (yet) an evolution mode of SimulationCode. For the nonlinear power spectrum of mixed cold and axion dark matter see <a href="halomodel.php" style="display: inline">AxionHMcode</a>, and for the linear evolution the axion model of the <a href="cosmology.php" style="display: inline">Einstein&ndash;Boltzmann solver</a>.
</p>

<hr/>
<header class="entry-header text-center">
  <h1 class="entry-title" id="Equations">The Schr&ouml;dinger&ndash;Poisson equations and units</h1>
</header>
<hr/>

<p>
With a rescaled comoving wavefunction $\psi$ (so that $|\psi|^2$ is the comoving mass density), $\eta = \hbar/m$ and $U = a\Phi$, the equations are
$$i\frac{\partial\psi}{\partial t} = -\frac{\eta}{2a^2}\nabla^2\psi + \frac{U}{\eta a}\psi,\qquad \nabla^2 U = 4\pi G\left(|\psi|^2 + \rho_{\rm external} - \bar\rho_{\rm total}\right),$$
with $t$ the cosmic time. At $a = 1$ they are the usual non-expanding Schr&ouml;dinger&ndash;Poisson system. The units are chosen by the user (a length, a time and a mass unit), and $\eta$ and $G$ are given in them; <tt>Physics::from_si</tt> converts from the boson mass in eV and SI units. Coordinates are in $[0, L)$ in the code length unit (not divided by the box size, unlike the particle coordinates elsewhere in FML).
</p>

<hr/>
<header class="entry-header text-center">
  <h1 class="entry-title" id="Use">Using the solver (FML::WAVEDM)</h1>
</header>
<hr/>

<pre class="prettyprint">
#include &lt;FML/WaveDarkMatter/SchrodingerPoisson.h&gt;
using namespace FML::WAVEDM;

const auto physics = Physics::from_si(1e-22,                         // boson mass in eV/c^2
                                      FML::UTILS::SI::kpc,           // code length in metres
                                      FML::UTILS::SI::Gyr,           // code time in seconds
                                      1e7 * FML::UTILS::SI::Msun);   // code mass in kg
Solver solver(64, 20.0, physics);   // 64^3 points, comoving box of 20 code lengths
solver.initialize_density_phase(
    [](auto x) { return 1.0 + 0.01 * std::cos(2 * M_PI * x[0] / 20.0); },
    [](auto) { return 0.0; });
solver.step(0.001);                  // fixed a; expanding: step_scale_factor(a_next, H)
auto d = solver.diagnostics();       // mass, energies, high-k tail

// On pencils instead of slabs
FML::WAVEDM::SolverT&lt;FML::GRID::CPencilGrid&lt;3&gt;&gt; solver_on_pencils(64, 20.0, physics);
</pre>

<p>
The wavefunction is a full complex field, so it lives on FML's complex-to-complex grids (<tt>CFFTWGrid</tt> on FFTW-MPI slabs or <tt>CPencilGrid</tt> on pencils, see the <a href="pencilgrid.php#Complex" style="display: inline">pencil page</a>), with all $N^3$ independent Fourier modes. Any mesh size works, also with uneven or empty slabs, and slabs and pencils agree to round-off. A self-gravitating step costs four complex FFTs.
</p>

<hr/>
<header class="entry-header text-center">
  <h1 class="entry-title" id="Stepping">Time stepping and expansion</h1>
</header>
<hr/>

<p>
A step is the symmetric kick-drift-kick split
$$\psi \to e^{-iU_{\rm initial}B/2\eta}\psi,\qquad \psi_k \to e^{-i\eta k^2 A/2}\psi_k,\qquad U_{\rm final}\ \text{from the new density},\qquad \psi\to e^{-iU_{\rm final}B/2\eta}\psi,$$
with $A = \int dt/a^2$ and $B = \int dt/a$ over the step. The potential is recomputed after the kinetic step (the density does not change in the potential steps), so the method is second order also with a prescribed expansion. <tt>step_scale_factor(a_next, H)</tt> computes $dt$, $A$ and $B$ from $H(a)$ by Gauss&ndash;Legendre quadrature, and <tt>advance(TimeFactors)</tt> takes them from an existing cosmology. <tt>suggested_timestep()</tt> bounds the kinetic and potential phases per step. There is no filtering or renormalization that would hide accumulated errors: the de Broglie wavelength must be resolved.
</p>

<hr/>
<header class="entry-header text-center">
  <h1 class="entry-title" id="Hooks">Initial conditions, external matter and diagnostics</h1>
</header>
<hr/>

<ul>
<li><b>Initial conditions</b>: a complex field directly, a density and a phase, or a density and a velocity potential (a single-stream, irrotational flow). The module does not make cosmological axion initial conditions from a transfer function itself; a consistent field must be supplied.</li>
<li><b>External matter</b>: <tt>set_external_density</tt> adds a comoving density (for example from CDM particles) to the Poisson source, and <tt>gravity_acceleration()</tt> gives the comoving acceleration $-\nabla U/a^3$ for particles. These are hooks: a coupled wave-plus-particle simulation is not provided.</li>
<li><b>Diagnostics</b>: the mass, kinetic and potential energies, and the mass in the modes near the Nyquist frequency (to watch the resolution). With expansion the energy is not conserved but obeys $dE/dt = -H(2K + W)$; <tt>EnergyBalance</tt> tracks the residual of this identity.</li>
<li><b>Checkpoints</b>: the wave, the external source and the epoch are saved and can be read on a different number of tasks, or on pencils after writing on slabs and back.</li>
</ul>

<hr/>
<header class="entry-header text-center">
  <h1 class="entry-title" id="Validation">Validation and limitations</h1>
</header>
<hr/>

<p>
The tests (on 1&ndash;4 tasks, with uneven and empty slabs, on slabs and pencils, and under sanitizers) compare the complex transforms with direct DFTs, free-wave dispersion and two-wave interference with their exact solutions, the nonlinear evolution with an independent high-accuracy ODE integration (SciPy, DOP853), and the isolated soliton with a radial boundary-value solution; they check the time-step convergence, backward evolution, the energy balance with expansion and the checkpoints. See the <a href="https://github.com/HAWinther/FML/blob/master/FML/WaveDarkMatter/README.md" style="display: inline">module README</a>. Not included: self-interactions, isolated boundaries, relativistic evolution, a live coupling to CDM particles and calibrated cosmological runs. The split-step method follows <a href="https://arxiv.org/abs/1807.04037" style="display: inline">Edwards et al. (2018), PyUltraLight</a>; the implementation is independent.
</p>
