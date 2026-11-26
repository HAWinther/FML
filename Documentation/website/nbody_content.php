<hr/>
<header class="entry-header text-center">
  <h1 class="entry-title">Table of contents</h1>
</header>
<hr/>

<p class="page-toc">
<a href="index.php">Back to the main page</a><br>
<a href="documentation.php">Back to the documentation</a><br>
<a href="#Overview">1. Overview: N-body</a><br>
<a href="#Methods">2. Methods</a><br>
<a href="#COLA">3. COLA simulations</a><br>
<a href="#MG">4. Modified gravity simulations</a><br>
<a href="#ForceOptions">5. Force options and time-stepping coefficients</a><br>
<a href="#ForceKernels">6. Force kernels</a><br>
<a href="#ShortRange">7. Short-range forces (P3M / TreePM)</a><br>
</p>

<header class="entry-header text-center">
  <h1 class="entry-title" id="Overview">Overview: N-body</h1>
</header>
<hr/>  

<p>
We provide some methods for performing some very simple fixed grid N-body simulations (Particle Mesh). The domain decomposition we use in this library is not suitable for huge high-resolution simulations so we only provide these methods and there are some really good and efficient N-body codes like RAMSES and GADGET that one should use for proper N-body anyway. Everything is in the namespace FML::NBODY (NBody/NBody.h). For a full (PM and COLA) simulation code built on these methods see the <a href="nbodycode.php" style="display: inline">N-body code</a> page.
</p>

<header class="entry-header text-center">
  <h1 class="entry-title" id="Methods">Methods</h1>
</header>
<hr/>  

<p>
In a N-body simulation we solve the equations
$$\frac{d^2 x}{d\tau^2} = - \nabla\Phi$$
$$\nabla^2\Phi = C \delta$$
where, for cosmological simulations, $d\tau = H_0 dt / a^2$ and $C = \frac{3}{2}\Omega_M a$ and for Newtonian ones $C = 4\pi G \rho_{\rm mean} / B_{\rm Boxsize}^2$. This can be solved with the Leapfrog method
$$x_{n+1} = x_n + v_{n+1/2}\Delta\tau$$
$$v_{n+1/2} = v_{n-1/2} - (\nabla\Phi)_n\Delta\tau$$
The first step is moving the particles (Drift) and the last step is updating the velocities (Kick). The force is computed by performing ${\rm NDIM}+1$ FFTs:
$$\nabla\Phi = C F^{-1}[ -\frac{i\vec{k}}{k^2}F[\delta]]$$
The methods below can perform these steps:
<div style="text-align: center">
  <div style="width:600px;display: inline-block;text-align: left">
    <pre class="prettyprint">
// This is solving the Poisson equation D^2phi = norm_poisson_equation * delta and returns DPhi
// density_assignment_method_used is used if we deconvolve the window function (see below)
template &lt;int N&gt;
void compute_force_from_density_real(
                   const FFTWGrid&lt;N&gt; &amp; density_grid_real,
                   std::array&lt;FFTWGrid&lt;N&gt;, N&gt; &amp; force_real,
                   std::string density_assignment_method_used,
                   double norm_poisson_equation);

// Same as above, but where the density field is already in fourier space
template &lt;int N&gt;
void compute_force_from_density_fourier(
                   const FFTWGrid&lt;N&gt; &amp; density_grid_fourier,
                   std::array&lt;FFTWGrid&lt;N&gt;, N&gt; &amp; force_real,
                   std::string density_assignment_method_used,
                   double norm_poisson_equation);

// Both take a final optional argument const ForceOptions &amp; options (see "Force options" below):
// the Fourier kernels used for 1/D^2 and D (continuous, discrete_2pt or discrete_4pt; the 2pt Green's
// function is the one of Hockney &amp; Eastwood 1988 used by e.g. FastPM) and whether to deconvolve the
// density assignment and interpolation window functions

// Advance positions from velocity. Takes care of communication in case particles move past domain boundary
template &lt;int N, class T&gt;
void DriftParticles(
                   FML::PARTICLE::MPIParticles&lt;T&gt; &amp; part,
                   double delta_time,
                   bool periodic_box = true);

// Does *not* take care of communication in case particles move past domain boundary
template &lt;int N, class T&gt;
void DriftParticles(
                   T * p,
                   size_t NumPart,
                   double delta_time,
                   bool periodic_box = true);

// Advance velocity from force
template &lt;int N, class T&gt;
void KickParticles(std::array&lt;FFTWGrid&lt;N&gt;, N&gt; &amp; force_grid,
                   FML::PARTICLE::MPIParticles&lt;T&gt; &amp; part,
                   double delta_time,
                   std::string interpolation_method);

template &lt;int N, class T&gt;
void KickParticles(std::array&lt;FFTWGrid&lt;N&gt;, N&gt; &amp; force_grid,
                   T * p,
                   size_t NumPart,
                   double delta_time,
                   std::string interpolation_method);

// Do one kick-drift-kick 2nd order leapfrog N-body step
template &lt;int N, class T&gt;
void KickDriftKickNBodyStep(
                   int Nmesh,
                   FML::PARTICLE::MPIParticles&lt;T&gt; &amp; part,
                   double delta_time,
                   std::string density_assignment_method,
                   double norm_poisson_equation);

// Do one 4th order leapfrog N-body step
template &lt;int N, class T&gt;
void YoshidaNBodyStep(
                   int Nmesh,
                   FML::PARTICLE::MPIParticles&lt;T&gt; &amp; part,
                   double delta_time,
                   std::string density_assignment_method,
                   double norm_poisson_equation);

// From a given density field delta(k,zini) generate N-body IC using 1, 2 or 3 LPT.
// The LPT potentials are returned in phi_nLPT_potentials.
// velocity_norms are the factors we multiply the nLPT displacement fields by to get the
// velocities (1LPT, 2LPT, 3LPTa, 3LPTb), e.g. f_i(zini) H(zini)/H0 * aini^2 for the fiducial
// N-body velocity units. The optional arguments allow for a scale-dependent 1LPT velocity
// and for using another density field for the 2LPT source and the stored 1LPT field (COLA
// with several species)
template &lt;int N, class T&gt;
void NBodyInitialConditions(FML::PARTICLE::MPIParticles&lt;T&gt; &amp; part,
                   int Npart_1D,
                   double buffer_factor,

                   const FFTWGrid&lt;N&gt; &amp; delta_fourier,
                   std::vector&lt;FFTWGrid&lt;N&gt;&gt; &amp; phi_nLPT_potentials,
                   int LPT_order,

                   double box,
                   double zini,
                   std::vector&lt;double&gt; velocity_norms,
                   std::function&lt;double(double)&gt; velocity_norm_1LPT_of_kBox = nullptr,
                   const FFTWGrid&lt;N&gt; * delta_fourier_2LPT_source = nullptr,
                   const FFTWGrid&lt;N&gt; * delta_fourier_1LPT_stored = nullptr);

// Generate a random field (gaussian or non-local non-gaussian) of a given type
// and then call the method above
template &lt;int N, class T&gt;
void NBodyInitialConditions(
                   FML::PARTICLE::MPIParticles&lt;T&gt; &amp; part,
                   int Npart_1D,
                   double buffer_factor,

                   int Nmesh,
                   bool fix_amplitude,
                   FML::RANDOM::RandomGenerator * rng,
                   std::function&lt;double(double)&gt; Pofk_of_kBox_over_Pofk_primordal,
                   std::function&lt;double(double)&gt; Pofk_of_kBox_over_volume_primordial,
                   int LPT_order,
                   std::string type_of_random_field,
                   double fNL,

                   double box,
                   double zini,
                   std::vector&lt;double&gt; velocity_norms);
</pre>
</div>
</div>
</p>

<header class="entry-header text-center">
  <h1 class="entry-title" id="COLA">COLA simulations</h1>
</header>
<hr/>  

<p>
In COLA simulations we solve for displacements on top of that predicted by LPT. With this we can do simulations much faster than usual N-body (by sacrificing accuracy on small scales). Instead of propagating $x$ and $v$ via
$$\frac{dx}{d\tau} = v$$
$$\frac{d v}{d\tau} = - \nabla\Phi$$
we instead define $x_{\rm COLA} = x - x_{\rm LPT}$ so that the equation becomes
$$\frac{dx}{d\tau} = v_{\rm COLA} + \frac{dx_{\rm LPT}}{d\tau}$$
$$\frac{d v_{\rm COLA}}{d\tau} = - \nabla\Phi - \frac{d^2x_{\rm LPT}}{d\tau^2}$$
and we propagate $x$ and $v_{\rm COLA}$ instead. Note that in this frame the initial velocity is $v_{\rm COLA} = 0$ and this helps us keep in line with linear theory on the largest scales even with large time-steps. Also note that $x_{\rm LPT} \sim D_1(\tau)\Psi_1(q,\tau_{\rm ini}) + D_2(\tau)\Psi_2(q,\tau_{\rm ini}) + \ldots$ so the derivatives of $x_{\rm LPT}$ above just become some growth-factor times the initial displacement field at particle positions when we start the simulation. The library is set up such that when we create initial conditions for N-body then we store displacement fields at the particle positions if they have the related get_D_1LPT (and 2LPT, 3LPTa, 3LPTb if you want to use higher order) methods. With this it's easy to turn a normal N-body into a COLA simulation - all we need is to compute some ODEs to get the growth-factors and add in the additional terms in the equations of motion (see the N-body example and the <a href="nbodycode.php" style="display: inline">COLA code</a> for this).
</p>


<header class="entry-header text-center">
  <h1 class="entry-title" id="MG">Modified gravity simulations</h1>
</header>
<hr/>  

<p>
The library contains some methods for computing the additional fifth-force in modified gravity models. We can compute the fifth-force in any linear model exactly, we can estimate the fifth-force in models with a screening mechanism using an approximate method and if you want you can always use the multigrid solver to solve the exact equation.
</p>

<p>
The methods below compute the fifth-force in terms of a fifth-force "density contrast" that is such that the total force (in fourier space) is just
$$F[\nabla \Phi_{\rm tot}] = F[\nabla \Phi + \nabla \Phi_{\rm fifth-force}] \propto \frac{i\vec{k}}{k^2}(\delta(k) + \delta_{\rm fifth-force})$$
i.e. after computing this we only need to add this to the density contrast we would use to compute the usual gravitational force in order to get the total-force.
</p>

<p>
<div style="text-align: center">
  <div style="width:600px;display: inline-block;text-align: left">
    <pre class="prettyprint">
// For models where the fifth-force is given as F = Dphi where 
// F[D^2 phi] = m^2 F[phi] + f(k)F[delta]
// Here the coupling_factor is f(k)k^2/(k^2 + m^2)
// e.g. for linearized f(R) the coupling factor is 1/3 * k^2/(k^2 + m^2)
// and for linearized DGP the coupling factor is 1/(3betaDGP)
template &lt;int N&gt;
  void compute_delta_fifth_force(
                   const FFTWGrid&lt;N&gt; &amp; density_fourier,
                   FFTWGrid&lt;N&gt; &amp; density_mg_fourier,
                   std::function&lt;double(double)&gt; coupling_factor_of_kBox);

// Approximate method of Winther and Ferreira 2015
// For MG models that have a potential dependent screening (f(R) and the alike)
// D^2 phi = m^2 phi + f(Phi)delta
// e.g. for f(R) the screening factor is f(Phi) = 1.5 |f_R / Phi|
// and the coupling factor is 1/3 * k^2 / (k^2 + m^2)
// poisson_norm is used to convert delta to Phi and is defined via D^2 Phi = poisson_norm * delta
// (the metric potential, not the code potential, so C = 1.5 OmegaM a (H0 B)^2 / a^2)
template &lt;int N&gt;
  void compute_delta_fifth_force_potential_screening(
                   const FFTWGrid&lt;N&gt; &amp; density_fourier,
                   FFTWGrid&lt;N&gt; &amp; density_mg_fourier,
                   std::function&lt;double(double)&gt; coupling_factor_of_kBox,
                   std::function&lt;double(double)&gt; screening_factor_of_newtonian_potential,
                   double poisson_norm);

// Approximate method of Winther and Ferreira 2015
// For MG models that have a density dependent screening (DGP and the alike)
// D^2 phi = m^2 phi + f(rho)delta
// e.g. for DGP m=0 and the screening factor f(rho) = 2(sqrt(1 + C) - 1)/C where C = const * rho 
// and coupling_factor = 1/(3betaDGP)
template &lt;int N&gt;
  void compute_delta_fifth_force_density_screening(
                   const FFTWGrid&lt;N&gt; &amp; density_fourier,
                   FFTWGrid&lt;N&gt; &amp; density_mg_fourier,
                   std::function&lt;double(double)&gt; coupling_factor_of_kBox,
                   std::function&lt;double(double)&gt; screening_factor_of_density,
                   double smoothing_scale,
                   std::string smoothing_method);
</pre>
</div>
</div>
</p>

<header class="entry-header text-center">
  <h1 class="entry-title" id="ForceOptions">Force options and time-stepping coefficients</h1>
</header>
<hr/>

<p>
The force routines take an explicit <tt>ForceOptions</tt> argument (no global state): the Laplacian kernel (continuous $-k^2$, or the discrete two- or four-point Green's functions; see <a href="#ForceKernels" style="display: inline">Force kernels</a> for the equations), the gradient kernel (continuous $i\vec k$ or discrete two- or four-point stencils; zero at the Nyquist frequency of an even mesh, where the derivative is ambiguous) and whether to deconvolve the assignment window. The two-point discrete Green function is described in Hockney &amp; Eastwood (1988) and is also used by <a href="https://arxiv.org/abs/1603.00476" style="display: inline">FastPM (Feng et al. 2016)</a>; see also the force-discretization discussion in <a href="https://arxiv.org/abs/astro-ph/0505010" style="display: inline">GADGET-2 (Springel 2005)</a>. <tt>options.lattice_greens</tt> points to a <tt>LatticeGreensFunction</tt> (<tt>NBody/LatticeGreensFunction.h</tt>), which multiplies the Green's function by the closed-form correction that makes the linear force exact for particles close to a cubic lattice (any assignment order, an integer ratio of grid to particle lattice, lattice offset, interlacing shifts and deconvolution); see <tt>NBody/ForceOptions.md</tt>.
</p>
<div style="text-align: center">
  <div style="width:600px;display: inline-block;text-align: left">
    <pre class="prettyprint">
FML::NBODY::ForceOptions options;
options.laplace_kernel  = FML::NBODY::DISCRETE_GREENS_FUNCTION_FROM_TWO_POINT;
options.gradient_kernel = FML::NBODY::DISCRETE_FOUR_POINT_GRADIENT;
options.deconvolve = true;
FML::NBODY::compute_force_from_density_fourier&lt;3&gt;(density, force, "CIC", norm, options);
</pre>
</div>
</div>
<p>
<tt>NBody/TimeStepping.h</tt> gives the kick and drift coefficients of the Quinn (<a href="https://arxiv.org/abs/astro-ph/9710043" style="display: inline">1997</a>), Tassev (<a href="https://arxiv.org/abs/1301.0322" style="display: inline">2013</a>), FastPM (<a href="https://arxiv.org/abs/1603.00476" style="display: inline">Feng et al. 2016</a>) and BullFrog (<a href="https://arxiv.org/abs/2409.19049" style="display: inline">Rampf, List &amp; Hahn 2024</a>) schemes for any expansion history and growth factors $D_1$, $D_2$ (independently of the COLA code), with steps linear, logarithmic, power-law or uniform in the growth factor. FastPM and BullFrog follow the linear (and for BullFrog the 2LPT) growing mode exactly whatever the step size; see the <a href="nbodycode.php#TimeStepping" style="display: inline">COLA code</a> for their accuracy.
</p>
<div style="text-align: center">
  <div style="width:600px;display: inline-block;text-align: left">
    <pre class="prettyprint">
#include &lt;FML/NBody/TimeStepping.h&gt;
TimestepOptions options;
options.method  = TimestepMethod::BullFrog;
options.spacing = ScaleFactorSpacing::GrowthFactor;
TimeStepping stepping(options, E_of_a, force_amplitude_of_a, growth);
for (auto step : stepping.steps(a_start, a_end, nsteps)) {
  // v = step.velocity_scale * v + step.delta_time_kick * F(step.apos);  x += step.delta_time_drift * v
}
</pre>
</div>
</div>

<header class="entry-header text-center">
  <h1 class="entry-title" id="ForceKernels">Force kernels</h1>
</header>
<hr/>

<p>
A PM force is built from three kernels: the Green's function $G(\vec k)$ that inverts the Laplacian, the gradient $\vec D(\vec k)$ that differentiates the potential ($\nabla \to i\vec D$) and the assignment window $W(\vec k)$. With $\Delta x = 1/N_{\rm mesh}$, $\delta(\vec k)$ the Fourier transform of the assigned density and $C$ the Poisson prefactor ($\nabla^2\Phi = C\delta$), the force on the grid is
$$\Phi(\vec k) = G(\vec k)\, C\, \delta(\vec k),\qquad \vec F(\vec k) = -i\vec D(\vec k)\,\frac{\Phi(\vec k)}{W^n(\vec k)},\qquad W(\vec k) = \prod_j \left[{\rm sinc}\frac{k_j\Delta x}{2}\right]^p,$$
with $n = 0$ (<tt>deconvolve = false</tt>, the default) or $n$ = <tt>deconvolve_window_power</tt> (2: assignment and interpolation with the window; 1 if one of them is exact, as with the NUFFT) and $p$ the assignment order (1 NGP, 2 CIC, 3 TSC, 4 PCS, 5 PQS). The exact force of the band-limited density has $G = -1/k^2$ and $\vec D = \vec k$. The discrete kernels are the Fourier transforms of finite-difference stencils: $\phi(x_i + l\Delta x) \to e^{ilk\Delta x}\phi(k)$, so a symmetric difference gives $i\sin$ and a symmetric second difference a $\cos$. They act direction by direction, $\vec D = (D(k_1), D(k_2), D(k_3))$ and $G = -1/\sum_j L(k_j)$ with $L(k)$ minus the one-dimensional Laplacian. Below $x = k\Delta x$, so $x = \pi$ at the grid Nyquist frequency $k_{\rm Ny} = \pi/\Delta x$. The options are set with <tt>ForceOptions</tt> (<tt>laplace_kernel</tt>, <tt>gradient_kernel</tt>) in the library and <tt>force_greens_function_kernel</tt>, <tt>force_gradient_kernel</tt> in the <a href="nbodycode.php" style="display: inline">N-body code</a>.
</p>

<p><b>Green's functions</b> (<tt>greens_function_laplace_operator_fourier</tt>):</p>
<div class="table-responsive">
<table class="table table-striped table-condensed">
<thead><tr><th>Name</th><th>Real-space stencil for $\partial^2\phi$</th><th>$L(k)\,\Delta x^2$</th><th>Small $x$</th></tr></thead>
<tbody>
<tr><td><tt>continuous</tt></td><td>(spectral)</td><td>$x^2$</td><td>exact</td></tr>
<tr><td><tt>discrete_2pt</tt></td><td>$\dfrac{\phi_{i+1} - 2\phi_i + \phi_{i-1}}{\Delta x^2}$</td><td>$4\sin^2\dfrac{x}{2} = 2(1-\cos x)$</td><td>$x^2\left(1 - \dfrac{x^2}{12}\right)$</td></tr>
<tr><td><tt>discrete_4pt</tt></td><td>$\dfrac{-\phi_{i+2} + 16\phi_{i+1} - 30\phi_i + 16\phi_{i-1} - \phi_{i-2}}{12\Delta x^2}$</td><td>$\dfrac{15 - 16\cos x + \cos 2x}{6}$</td><td>$x^2\left(1 - \dfrac{x^4}{90}\right)$</td></tr>
</tbody>
</table>
</div>
<p>
<tt>discrete_2pt</tt> is the 7-point Laplacian in 3D, the discrete Green's function of Hockney &amp; Eastwood (1988), also used by <a href="https://arxiv.org/abs/1603.00476" style="display: inline">FastPM</a>; it is second order and makes the potential too large at high $k$ ($L/k^2 = 0.81$ at $k_{\rm Ny}/2$ and $4/\pi^2 = 0.41$ at $k_{\rm Ny}$ along an axis). <tt>discrete_4pt</tt> is the fourth-order (5-point per direction) stencil ($L/k^2 = 0.95$ at $k_{\rm Ny}/2$ and $0.54$ at $k_{\rm Ny}$). Both vanish only at $\vec k = 0$, where $G$ is set to zero. Until November 2026 <tt>discrete_4pt</tt> was $L = D_4^2$, the square of the four-point gradient below, which vanishes at $k_{\rm Ny}$: the modes near it got a potential up to $\sim 10^4$ times too large and runs with it blew up. $G$ and $\vec D$ need not come from the same stencil: composing the two-point gradient twice gives the wide stencil $(\phi_{i+2} - 2\phi_i + \phi_{i-2})/(4\Delta x^2)$, not <tt>discrete_2pt</tt>.
</p>

<p><b>Gradients</b> (<tt>gradient_kernel_fourier</tt>):</p>
<div class="table-responsive">
<table class="table table-striped table-condensed">
<thead><tr><th>Name</th><th>Real-space stencil for $\partial\phi$</th><th>$D(k)\,\Delta x$</th><th>Small $x$</th></tr></thead>
<tbody>
<tr><td><tt>continuous</tt></td><td>(spectral)</td><td>$x$ ($0$ at $x = \pi$)</td><td>exact</td></tr>
<tr><td><tt>discrete_2pt</tt></td><td>$\dfrac{\phi_{i+1} - \phi_{i-1}}{2\Delta x}$</td><td>$\sin x$</td><td>$x\left(1 - \dfrac{x^2}{6}\right)$</td></tr>
<tr><td><tt>discrete_4pt</tt></td><td>$\dfrac{-\phi_{i+2} + 8\phi_{i+1} - 8\phi_{i-1} + \phi_{i-2}}{12\Delta x}$</td><td>$\dfrac{8\sin x - \sin 2x}{6}$</td><td>$x\left(1 - \dfrac{x^4}{30}\right)$</td></tr>
<tr><td>6th order (finite difference only)</td><td>$\dfrac{45\delta_1\phi - 9\delta_2\phi + \delta_3\phi}{60\Delta x}$, $\ \delta_l\phi = \phi_{i+l} - \phi_{i-l}$</td><td>$\dfrac{45\sin x - 9\sin 2x + \sin 3x}{30}$</td><td>$x\left(1 - \dfrac{x^6}{140}\right)$</td></tr>
</tbody>
</table>
</div>
<p>
$D/k$ is 0.64, 0.85 and 0.93 at $k_{\rm Ny}/2$ for the 2-, 4- and 6-point stencils. Every gradient is zero for the Nyquist component $k_j = k_{\rm Ny}$ of an even mesh: the discrete ones because $\sin\pi = 0$, the continuous one by construction, since that mode is its own mirror image and $ik_j$ would depend on the sign chosen for it; with zero there the force of a reflected density is the reflected force. The finite-difference force (<tt>compute_force_from_potential_real</tt>, <tt>force_use_finite_difference_force</tt> with <tt>force_finite_difference_stencil_order</tt> = 2, 4, 6) applies the stencils in real space to the potential computed with the chosen Green's function; orders 2 and 4 give the same force as the Fourier gradients <tt>discrete_2pt</tt> and <tt>discrete_4pt</tt> (to $10^{-15}$), order 6 exists only there.
</p>

<p>
<b>The force of a combination.</b> Without the window the force of a single mode relative to the exact force $\vec k/k^2$, along the wave vector, is
$$\frac{F_{\rm PM}}{F_{\rm exact}} = \frac{|\vec k|\,\hat k\cdot\vec D(\vec k)}{\sum_j L(k_j)} \;\to\; \frac{k\,D(k)}{L(k)}\quad(\vec k \text{ along an axis}).$$
Along an axis, at small $x$, this is 1 (continuous/continuous), $1 - x^2/6$ (two-point $\vec D$, continuous $G$), $1 - x^4/30$ (four-point $\vec D$, continuous or four-point $G$), $1 - x^2/12$ (two-point $\vec D$ and $G$) and $1 + x^2/12$ (four-point $\vec D$, two-point $G$). The discrete kernels are anisotropic: along the diagonal each component of the same $|\vec k|$ is $\sqrt 3$ times smaller and the force is closer to exact.
</p>
<div class="table-responsive">
<table class="table table-striped table-condensed">
<thead>
<tr><th>$\vec D$</th><th>$G$</th><th>axis, $k_{\rm Ny}/4$</th><th>axis, $k_{\rm Ny}/2$</th><th>diagonal, $k_{\rm Ny}/4$</th><th>diagonal, $k_{\rm Ny}/2$</th></tr>
</thead>
<tbody>
<tr><td>continuous</td><td>continuous</td><td>1</td><td>1</td><td>1</td><td>1</td></tr>
<tr><td><tt>discrete_2pt</tt></td><td>continuous</td><td>0.900</td><td>0.637</td><td>0.966</td><td>0.869</td></tr>
<tr><td><tt>discrete_4pt</tt></td><td>continuous</td><td>0.988</td><td>0.849</td><td>0.999</td><td>0.980</td></tr>
<tr><td><tt>discrete_2pt</tt></td><td><tt>discrete_2pt</tt></td><td>0.948</td><td>0.785</td><td>0.983</td><td>0.931</td></tr>
<tr><td><tt>discrete_4pt</tt></td><td><tt>discrete_2pt</tt></td><td>1.041</td><td>1.047</td><td>1.016</td><td>1.050</td></tr>
<tr><td><tt>discrete_4pt</tt></td><td><tt>discrete_4pt</tt></td><td>0.992</td><td>0.898</td><td>0.999</td><td>0.986</td></tr>
<tr><td>continuous</td><td><tt>discrete_2pt</tt></td><td>1.053</td><td>1.234</td><td>1.017</td><td>1.072</td></tr>
</tbody>
</table>
</div>

<figure class="science-figure">
  <img src="images/force_kernels.png" alt="Three panels against k over the Nyquist frequency: the gradient kernels D(k)/k, which fall from 1 to 0 at the Nyquist frequency, faster for the lower-order stencils; the Laplacians L(k)/k^2, with the old discrete_4pt kernel going to zero at the Nyquist frequency; and the PM force relative to the exact force for six combinations of gradient and Green's function">
  <figcaption>The force kernels along a grid axis: the gradients $D(k)/k$ (left; the sixth order is the finite-difference force), the Laplacians $L(k)/k^2 = -1/(k^2 G)$ (middle; dotted: the pre-November 2026 <tt>discrete_4pt</tt>, $D_4^2$) and the force of the combinations relative to the exact force, without the window (right). Made with <tt>Documentation/tools/force_kernels_figure.py</tt>.</figcaption>
</figure>

<p>
Pairing a discrete Green's function with the continuous gradient boosts the force at high $k$ (the potential is too large and nothing compensates). These are the kernels alone: the window then multiplies the force by $W^{2-n}$ (CIC: $W^2 = 0.66$ along an axis at $k_{\rm Ny}/2$), and the aliased images of the assignment add to it, which interlacing (<tt>force_interlacing_order</tt>) reduces: the force is averaged over $M$ copies of the particles shifted by fractions of a cell, and for the half-cell diagonal ($M = 2$) the images at $\vec k + 2k_{\rm Ny}\vec n$ with odd $\sum_j n_j$ get the phase $-1$ and cancel.
</p>
<p>
<b>Choosing.</b> The library default (no <tt>ForceOptions</tt> given) is continuous $G$ and $\vec D$ without deconvolution. The N-body code's <tt>fiducial</tt> is the continuous Green's function and, for the gradient, continuous for $N_{\rm mesh} = N_p^{1/3}$, <tt>discrete_4pt</tt> for $N_p^{1/3} \lt N_{\rm mesh} \le 3N_p^{1/3}$ and <tt>discrete_2pt</tt> for finer grids, with the particle lattice offset by half a force cell: on a force grid an integer factor finer than the particle grid the particles start on grid nodes, and the continuous gradient (large near the Nyquist frequency) turns the CIC response to the lattice into spurious forces (<tt>NBody/ForceOptions.md</tt>). In the PM settings study ($64^3$ particles in 250 Mpc/$h$) the <tt>discrete_4pt</tt> Green's function gives $P(k)$ within 0.1% of the continuous one to $k = 0.3\,h$/Mpc and <tt>discrete_2pt</tt> +8.5%; the grid size and the window matter more (<tt>SimulationCode/tests/check_pm_settings.py</tt>). The derivations, the validation and the lattice-calibrated Green's function are in Chapter 23 of the <a href="main.pdf" style="display: inline">FML book</a>.
</p>

<header class="entry-header text-center">
  <h1 class="entry-title" id="ShortRange">Short-range forces (P3M / TreePM)</h1>
</header>
<hr/>

<p>
The building blocks for adding a short-range particle&ndash;particle force to the PM force (Gaussian force split, spline softening, cell lists, octree, Yukawa fifth force and MPI ghosts) are on the <a href="shortrange.php" style="display: inline">short-range gravity page</a>.
</p>
