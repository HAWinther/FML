<hr/>
<header class="entry-header text-center">
  <h1 class="entry-title">Table of contents</h1>
</header>
<hr/>

<p class="page-toc">
<a href="index.php">Back to the main page</a><br>
<a href="documentation.php">Back to the documentation</a><br>
<a href="#Overview">1. Classical 1LPT, 2LPT and 3LPT</a><br>
<a href="#ArbitraryOrder">2. Arbitrary-order EdS LPT</a><br>
<a href="#Methods">3. Methods we provide</a><br>
<a href="#Growth">4. Growth factors</a><br>
<a href="#Recon">5. Reconstruction</a><br>
<a href="#Dealias">6. Dealiasing the 2LPT and 3LPT sources</a><br>
<a href="#SixthOrder">Sixth-order checks and plane-wave cancellation</a><br>
</p>

<header class="entry-header text-center">
  <h1 class="entry-title" id="Overview">Classical Lagrangian perturbation theory: 1LPT, 2LPT and 3LPT</h1>
</header>
<hr/>

<figure class="cinematic-figure" id="cinematic-02-ripples-to-structure">
  <a href="images/cinematic/02-ripples-to-structure.jpg" aria-label="View full-size illustration: From ripples to structure">
    <img src="images/cinematic/02-ripples-to-structure.jpg" width="1774" height="887" loading="lazy" decoding="async" alt="A panoramic progression from smooth density ripples to sheets, filaments and collapsed clusters">
  </a>
  <figcaption><strong>From ripples to structure.</strong> Conceptual progression from small primordial density perturbations to sheets, filaments and collapsed nodes; stages are illustrative, not a calibrated simulation. <span class="cinematic-credit">AI-generated concept illustration.</span></figcaption>
</figure>  

<p>
The established scale-independent and scale-dependent routines live in <tt>FML/LPT/DisplacementFields.h</tt>. They compute displacement and velocity fields through third order for initial conditions, COLA, reconstruction and analysis. They include the third-order transverse contribution; optional Fourier padding removes aliasing from the quadratic and cubic sources. This established interface remains the one used by SimulationCode and COLA. The separate arbitrary-order EdS engine is documented below.
</p>

<p>
The position of particles $x$ are related to the Lagrangian positions $q$ via
$$x = q + \Psi(q,t)$$
where $\Psi = \Psi^{\rm 1LPT} + \Psi^{\rm 2LPT} + \ldots$ is the displacement field. We provide functions for computing displacement fields up to third order. Below all positions (and gradients) are in code units, i.e. in $[0,1)$.
</p>
<p>
Given a density field $\delta$ we compute the 1LPT potential $\phi^{\rm 1LPT}$ defined via $\Psi^{\rm 1LPT} = D_1 \nabla\phi^{\rm 1LPT}$ by solving the Poisson equation
$$\nabla^2\phi^{\rm 1LPT} = -\delta$$
Note that we use the sign convention $\Psi^{\rm iLPT} = +\nabla\phi^{\rm iLPT}$ here and not the one often found in the literature where $\Psi^{\rm 1LPT} = -\nabla\phi^{\rm 1LPT}$ and $\Psi^{\rm 2LPT} = +\nabla\phi^{\rm 2LPT}$!
</p>
<p>
Likewise the 2LPT potential is computed by solving
$$\nabla^2\phi^{\rm 2LPT} = -\frac{3}{7}\sum_{i&gt;j} [\nabla_{ii}\phi^{\rm 1LPT}\nabla_{jj}\phi^{\rm 1LPT} - (\nabla_{ij}\phi^{\rm 1LPT})^2]$$
where all terms are evaluated via Fourier transforms (where the factor $-3/7 = D_2/D_1^2$).
</p>
<p>
The full displacement field can then be computed as
$$\vec{\Psi} = \frac{D_1}{D_1^{\rm ini}}\nabla\phi^{\rm 1LPT} + \frac{D_2}{D_2^{\rm ini}}\nabla\phi^{\rm 2LPT}$$
and the velocity can be found as
$$\frac{d\vec{x}}{d\tau} = BH\left[\frac{d\log D_1}{d\tau}\frac{D_1}{D_1^{\rm ini}}\nabla\phi^{\rm 1LPT} + \frac{d\log D_2}{d\tau}\frac{D_2}{D_2^{\rm ini}}\nabla\phi^{\rm 2LPT}\right]$$
where $B$ is the physical size of the box ($BH$ has units of velocity). Note that $D_1=D_2=1$ at the initial time by definition, however in Einstein-de Sitter (matter dominated $\Lambda$CDM) the physically relevant solution has $D_2 = -\frac{3}{7} D_1^2$ so we need to add an additional factor of $-\frac{3}{7}$ in front of the 2LPT terms when assigning positions and velocities. We include this factor by default.
</p>
<p>
It's only in simple models that the growth factor is scale-independent. In general the displacement field does not factor in time-space, but rather in time-Fourier-mode and the growth factor will depend on $k$. The methods here can handle this case. For scale-independent growth we only need to compute the displacement field once and then automatically have it for all times. For the case where we have scale-dependent growth we need to provide the function(s) $D_i(k,t)/D_i(k,t_{\rm ini})$ and then evaluate it from
$$\phi^{\rm iLPT}(k,t) = \frac{D_i(k,t)}{D_i(k,t_{\rm ini})} \phi^{\rm iLPT}(k,t_{\rm ini})$$
The velocity in this case can similarly be computed from
$$\vec{v} = HB \cdot F^{-1}\left[ \frac{d\log D_1(k,t)}{d\tau} \frac{D_1(k,t)}{D_1(k,t_{\rm ini})}F[\nabla\phi^{\rm 1LPT}] + \frac{d\log D_2(k,t)}{d\tau} \frac{D_2(k,t)}{D_2(k,t_{\rm ini})}F[\nabla\phi^{\rm 2LPT}]\right]$$
</p>
<p>
The equation for 3LPT is given by
$$\Psi^{\rm 3LPT} = \left(\frac{D_1}{D_1^{\rm ini}}\right)^3\left(\nabla \phi^{3a} + \nabla \phi^{3b} + \nabla\times A\right)$$
where
$$\nabla^2 \phi^{3a} = -\frac{1}{3}\det[\phi^{(1)}_{ij}]$$
$$\nabla^2 \phi^{3b} = -\frac{5}{9}[\nabla^2\phi^{(1)}\nabla^2\phi^{(2)} - \sum_{i,j} \phi^{(1)}_{ij} \phi^{(2)}_{ij} ]$$
$$\nabla^2 A = \frac{1}{3}\nabla\phi^{(2)}_i\times \nabla\phi^{(1)}_i$$
The potentials we provide are normalized such that this expression gives the 3LPT displacement field. The main difference wrt 1LPT and 2LPT is that the displacement field is not a pure potential flow (it has non-zero curl - well it's still curl free in Eulerian coordinates, but the transformation to Lagrangian coordinates introduces a curl. This curl part does not contribute to the initial power-spectrum of $\Psi$) so the only method we provide for 3LPT is to compute the full displacement vector $\Psi$ with all 3 orders included. The products in the 2LPT and 3LPT sources alias on the FFT grid (see <a href="https://arxiv.org/pdf/2008.09588.pdf" style="display: inline">Michaux et al. 2020</a>); the optional <a href="#Dealias" style="display: inline">dealiasing</a> below removes this.
</p>

<figure class="science-figure">
  <img src="images/lpt-orders.svg" alt="Schematic progression from a Lagrangian particle lattice through first-order displacements to second- and third-order deformations, plus Fourier padding for dealiasing">
  <figcaption>Illustrative displacement patterns only. Higher orders add nonlinear couplings, and 3LPT includes a transverse contribution; Fourier padding reduces aliasing in the source products.</figcaption>
</figure>

<header class="entry-header text-center">
  <h1 class="entry-title" id="ArbitraryOrder">Arbitrary-order EdS LPT</h1>
</header>
<hr/>

<p>
<tt>FML/LPT/ArbitraryOrder.h</tt> adds a standalone, native C++ solver for the growing-mode, single-fluid, pressureless Einstein&ndash;de Sitter recursion at a caller-selected finite order $P$. It is independent of the traditional 1&ndash;3LPT APIs above. In particular, it is not yet wired into SimulationCode/COLA, and its EdS time dependence is not a general modified-gravity or scale-dependent-growth solution.
</p>
<p>The displacement is expanded without factorials, using $g=a/a_{\rm ref}$:</p>
$$\Psi(q,a)=\sum_{n=1}^{P}g^n\Psi_n(q),\qquad
\frac{d\Psi}{d\ln a}=\sum_{n=1}^{P}n g^n\Psi_n(q).$$
<p>
At each order the divergence and curl sources are built from the full displacement gradients $A^{(r)}_{ij}=\partial_j\Psi_{r,i}$, including transverse components. With $Q(A,B)=\mathrm{tr}(A)\mathrm{tr}(B)-\mathrm{tr}(AB)$, $R(A,B,C)=\epsilon_{ijk}\epsilon_{pqr}A_{ip}B_{jq}C_{kr}$, and $K_i(A,B)=\epsilon_{ijk}\sum_l A_{lj}B_{lk}$, the compact EdS recursion is
</p>
$$\begin{aligned}
d_n\equiv\nabla\!\cdot\Psi_n={}&-\frac12\sum_{r+s=n}M_{rs}\left(1-\frac{4rs}{(2n+3)(n-1)}\right)Q(A^{(r)},A^{(s)})\\
&-\frac16\sum_{r+s+t=n}M_{rst}\left(1-\frac{4(rs+st+tr)}{(2n+3)(n-1)}\right)R(A^{(r)},A^{(s)},A^{(t)}),\\
c_n\equiv\nabla\!\times\Psi_n={}&\sum_{r+s=n,\,r<s}\frac{s-r}{n}K(A^{(r)},A^{(s)}).
\end{aligned}$$
<p>
Pair/triple sums use the corresponding symmetry multiplicities. A Fourier Helmholtz inversion recovers each vector coefficient from its divergence and curl. The independent sparse reference and low-order checks validate coefficient agreement through order six in double precision, including transverse feedback and nonlinear intermediate modes that later return to the requested output band.
</p>
<p><b>Input and scope.</b> Supply normalized Fourier coefficients of a real, periodic reference density and explicit strictly sub-Nyquist support $|m_i|\leq K_i$. The conservative work mesh retains the full bandwidth required through order $P$; outputs are projected back to the original mesh. The planner requires a per-rank memory budget before it allocates work grids. Collection and callback-based per-order streaming are available. Distributed execution uses nonempty FFTW slabs and MPI remapping; OpenMP accelerates the local row contractions.</p>
<div class="table-responsive">
<table class="table table-striped table-condensed">
<thead><tr><th>Capability</th><th>Current contract</th></tr></thead>
<tbody>
<tr><td>Physical model</td><td>Periodic, pressureless, single-fluid EdS growing mode; unit-box coordinates; no factorial in the series</td></tr>
<tr><td>Requested order</td><td>Finite caller-selected $P$; double-precision reference validation through $P=6$</td></tr>
<tr><td>Parallel layout</td><td>Serial or MPI slabs with mesh width divisible by rank count; OpenMP contractions. Empty slabs are not supported.</td></tr>
<tr><td>Returned fields</td><td>Fourier displacement at every order, or one order at a time; evaluator returns $\Psi$ and $d\Psi/d\ln a$</td></tr>
<tr><td>Not included</td><td>General cosmology, decaying modes, scale-dependent growth, direct SimulationCode/COLA integration</td></tr>
</tbody>
</table>
</div>
<div style="text-align: center"><div style="width:700px;display: inline-block;text-align: left"><pre class="prettyprint">
#include &lt;FML/LPT/ArbitraryOrder.h&gt;
using namespace FML::COSMOLOGY::LPT;
EdSLPTOptions options;
options.max_order = 6;
options.input_mode_limit = {1, 1, 1}; // required band limit for the input
options.max_owned_bytes_per_rank = 512ULL * 1024 * 1024;
auto orders = compute_eds_lpt(delta_fourier, options);
auto fields_at_epoch = orders.evaluate(a / a_reference,
                                      options.max_owned_bytes_per_rank);</pre></div></div>
<p>For derivations, API conventions, reference equations, measured memory and MPI results, see the <a href="https://github.com/HAWinther/FML/tree/master/FML/LPT">LPT source and documentation</a>.</p>

<p>
<b>On pencils.</b> <tt>compute_eds_lpt(input, options)</tt> and <tt>for_each_eds_lpt_order</tt> take the grid type of the input: with a <a href="pencilgrid.php" style="display: inline">$\texttt{PencilGrid}$</a> the orders are computed natively on pencils (uniform pencil work grids, the input embedded and the orders cropped by sending each Fourier mode to its owner, the Hermitian $k_z = 0$ and Nyquist planes exchanged between the owners of mirrored modes) and returned on grids decomposed as the input. The results agree with the slabs to round-off; on slabs nothing changed.
</p>

<header class="entry-header text-center">
  <h1 class="entry-title" id="SixthOrder">Sixth-order checks and plane-wave cancellation</h1>
</header>
<hr/>

<p>The independent sparse reference now covers small band-limited seeds through order six in float and double. The sixth-order checks include transverse feedback; a single plane wave remains exactly one-dimensional, with nonlinear coefficients suppressed to roundoff. Fourier resampling retains nonlinear intermediate modes that can return to the output band.</p>
<p>Sixth order is a validated small-reference range, not a universal production-resolution claim. Conservative work meshes grow quickly with order and seed support. Use the memory planner; cached and recomputed workspace choices trade retained memory for additional FFT work. The arbitrary-order solver remains separate from SimulationCode/COLA and assumes EdS growing-mode time dependence. See <a href="https://github.com/HAWinther/FML/blob/master/FML/LPT/SCIENCE_RESULTS.md" style="display: inline">scientific results</a>.</p>

<header class="entry-header text-center">
  <h1 class="entry-title" id="Methods">Methods we provide</h1>
</header>
<hr/>  

<p>
See LPT/example for an example on how to compute displacement fields from a given power-spectrum and then make particles from it (IC generation).
<div style="text-align: center">
  <div style="width:600px;display: inline-block;text-align: left">
    <pre class="prettyprint">
// From a given phi^iLPT(k, tini) compute the vector Dphi^iLPT(x,t)
// The vector is allocated in the method
// The factor DoverDini = D_nLPT(t) / D_nLPT(tini) where tini is the redshift we have phi_fourier at
template&lt;int N&gt;
  void from_LPT_potential_to_displacement_vector(
      const FFTWGrid&lt;N&gt; &amp; phi_fourier,
      std::array&lt;FFTWGrid&lt;N&gt;, N&gt; &amp; psi_real,
      double DoverDini = 1.0);

// Compute phi^1LPT(k,tini) in Fourier space
// Where tini is the time we have delta at
// The phi grid is allocated in the method
template&lt;int N&gt;
  void compute_1LPT_potential_fourier(
      const FFTWGrid&lt;N&gt; &amp; delta_fourier,
      FFTWGrid&lt;N&gt; &amp; phi_1LPT_fourier);

// Compute phi^2LPT(k,tini) in Fourier space
// Where tini is the time we have delta at
// The phi grid is allocated in the method
template&lt;int N&gt;
  void compute_2LPT_potential_fourier(
      const FFTWGrid&lt;N&gt; &amp; delta_fourier,
      FFTWGrid&lt;N&gt; &amp; phi_2LPT_fourier);

// From phi^iLPT(k,t_ini) to Psi(x,t) using growth-factors: Di(k,t)/Di(k, t_ini)
// The vector is allocated in the method
template&lt;int N&gt;
  void from_LPT_potential_to_displacement_vector_scaledependent(
      const FFTWGrid&lt;N&gt; &amp; phi_fourier,
      std::array&lt;FFTWGrid&lt;N&gt;, N&gt; &amp; psi_real,
      std::function&lt;double(double)&gt; growth_function_ratio);

// Compute 1LPT, 2LPT and 3LPT potentials at (k,tini) at the same time
// Where tini is the time we have delta at
// With ignore_curl_term we don't compute the vector potential A
template&lt;int N&gt;
  void compute_3LPT_potential_fourier(
      const FFTWGrid&lt;N&gt; &amp; delta_fourier,
      FFTWGrid&lt;N&gt; &amp; phi_1LPT_fourier,
      FFTWGrid&lt;N&gt; &amp; phi_2LPT_fourier,
      FFTWGrid&lt;N&gt; &amp; phi_3LPT_a_fourier,
      FFTWGrid&lt;N&gt; &amp; phi_3LPT_b_fourier,
      std::array&lt;FFTWGrid&lt;N&gt;, N&gt; &amp; phi_3LPT_Avec_fourier,
      bool ignore_curl_term);

// This method is not really needed as it follows from the above, but anyway
// Compute the displacement field Psi(k,tini) and its derivative up to 3LPT
// Where tini is the time we have delta at
// dlogDdt is the derivative of the (log) growth factor at the initial time
// With ignore_curl_term we don't compute the vector potential A
template&lt;int N&gt;
  void compute_3LPT_displacement_field(
      const FFTWGrid&lt;N&gt; &amp; delta_fourier,
      std::array&lt;FFTWGrid&lt;N&gt;, N&gt; &amp; Psi,
      std::array&lt;FFTWGrid&lt;N&gt;, N&gt; &amp; dPsidt,
      double dlogDdt,
      bool ignore_curl_term);

// Compute augmented LPT (ALPT) Psi(x,t) potential (Kitaura & Hess 2013)
// DoverDini_1LPT = D(t)/D(tini) where tini is the time we have delta at
template&lt;int N&gt;
  void compute_ALPT_potential_fourier(
      const FFTWGrid&lt;N&gt; &amp; delta_fourier,
      const FFTWGrid&lt;N&gt; &amp; phi_2LPT_fourier,
      FFTWGrid&lt;N&gt; &amp; phi_ALPT_fourier,
      double smoothing_scale,
      std::string smoothing_method,
      double DoverDini_1LPT = 1.0,
      double DoverDini_2LPT = 1.0);

// Spherical collapse approximation potential (Bernardeau 1994)
template&lt;int N&gt;
  void compute_spherical_collapse_potential(
      const FFTWGrid&lt;N&gt; &amp; delta_fourier,
      FFTWGrid&lt;N&gt; &amp; phi_sc_fourier,
      double DoverDini = 1.0);

// Assign the displacement field to particles when the growth is scale-dependent
// LPT_potential_order is 1LPT, 2LPT, 3LPTa or 3LPTb and tells us where in the
// particle to store it (the particle must have get_D_1LPT etc.)
template&lt;int N, class T&gt;
  void assign_displacement_fields_scaledependent(
      MPIParticles&lt;T&gt; &amp; part,
      FFTWGrid&lt;N&gt; &amp; LPT_potential_fourier,
      std::function&lt;double(double)&gt; DoverDini_of_k,
      std::string LPT_potential_order);
</pre>
</div>
</div>
</p>
<p>
These methods are templated on the dimension $N$ because why not (and N=2 is useful for testing).
</p>
<p>The optional ALPT potential combines 2LPT on large scales with a spherical-collapse approximation on small scales, following <a href="https://arxiv.org/abs/1212.3514" style="display: inline">Kitaura &amp; Hess (2013)</a> and the spherical-collapse approximation of <a href="https://arxiv.org/abs/astro-ph/9311066" style="display: inline">Bernardeau (1994)</a>. The FML routines implement these ingredients in Fourier space.</p>

<header class="entry-header text-center">
  <h1 class="entry-title" id="Growth">Growth factors</h1>
</header>
<hr/>  

<p>
Growth factors are easily evaluated using the ODESolver and Spline and for $\Lambda$CDM one can call the method below to generate splines of the growth-factors up to 3LPT (not including the vector potential at 3LPT yet). The growth-factors are normalized to unity at redshift $z_{\rm ini}$. The IC are set to match the EdS solution so $D_1=1$, $D_2 = -3/7$ and so on. For models with a time-dependent effective Newton's constant there is also compute_LPT_growth_factors_GeffGLCDM which takes the extra function GeffOverG_of_a. These need GSL.
<div style="text-align: center">
  <div style="width:600px;display: inline-block;text-align: left">
    <pre class="prettyprint">
void compute_LPT_growth_factors_LCDM(
      double OmegaM,
      double zini,
      std::function&lt;double(double)&gt; HoverH0_of_a,
      FML::INTERPOLATION::SPLINE::Spline &amp; D_1LPT_of_loga,
      FML::INTERPOLATION::SPLINE::Spline &amp; D_2LPT_of_loga,
      FML::INTERPOLATION::SPLINE::Spline &amp; D_3LPTa_of_loga,
      FML::INTERPOLATION::SPLINE::Spline &amp; D_3LPTb_of_loga);
</pre>
</div>
</div>
</p>

<header class="entry-header text-center">
  <h1 class="entry-title" id="Recon">Other LPT related stuff: Reconstruction</h1>
</header>
<hr/>  

<p>
Defining $\Theta$ via $q = \nabla_x \Theta(x)$ then
$$\det[\Theta_{,ij}] = 1 + \delta(x)$$
this equation reduces to a cubic equation in the Laplacian
$$(\nabla^2\Theta)^3 + p \nabla^2\Theta = q$$
where $p,q$ depend on cross-derivatives and $\delta$. Thus without expanding in a perturbative series we can solve this equation exactly using the multigrid solver, see examples for how to do this. This can again be used to find the displacement field $\Psi(q) = q-x$ by the following method: Compute $\Theta(x)$ in a $x$-box and create 'particles' that have both $x$ and $q$ coordinates. Swap $q$ and $x$ so that we now have a $q$-box and communicate particles so that they end up at their $q$ positions. The particles are now in a $q$-box and we can bin $q-x = \Psi(q)$ to a grid to get the displacement field. This has many applications like for example <a href="https://arxiv.org/pdf/1709.06350.pdf" style="display: inline">initial density reconstruction</a>. Though it does not give us time-derivatives.
</p>

<p>
We have methods for solving the usual 1LPT reconstruction equation (i.e. given observed galaxies try to move them back to the initial density field or to remove RSD). From observed $x$ compute (and smooth) $\delta_{\rm obs}$ and solve the LPT equation
$$\nabla\cdot \Psi + \frac{f}{b}\nabla[(\Psi\cdot r)r] = -\frac{\delta_{\rm obs}}{b}$$
where $r$ is the unit norm line of sight vector, $f$ is the growth rate and $b$ the galaxy bias. Given $\Psi$ subtract $\Psi$ from $x$ to get the $q$'s, or subtract $-\frac{f}{b}(\Psi\cdot r)r$ to remove the RSD contribution. This can be solved with the multigrid solver (see examples) or with Fourier methods (see LPT/Reconstruction.h; though this is just approximate as the term $(\Psi\cdot r)r$ is not curl-free, but this can be alleviated by doing it iteratively):
</p>

<p>
<div style="text-align: center">
  <div style="width:600px;display: inline-block;text-align: left">
    <pre class="prettyprint">
template &lt;int N, class T, template &lt;int&gt; class Grid = FFTWGrid&gt;
  void RSDReconstructionFourierMethod(
      MPIParticles&lt;T&gt; &amp; part,
      std::string density_assignment_method,
      std::vector&lt;double&gt; los_direction,
      int Nmesh,
      int niterations,
      double beta,
      std::pair&lt;std::string, double&gt; smoothing_options,
      bool survey_data);
</pre>
</div>
</div>
</p>

<p>
The grids are <tt>FFTWGrid</tt> by default; with <tt>RSDReconstructionFourierMethod&lt;N, T, FML::GRID::PencilGrid&gt;</tt> they are <a href="pencilgrid.php" style="display: inline">pencils</a> made on the domain of the particles (the same result for every particle to round-off). The grid always has the extra cells of the CIC interpolation of the displacement, also with NGP assignment.
</p>

<header class="entry-header text-center">
  <h1 class="entry-title" id="Dealias">Dealiasing of the 2LPT and 3LPT sources</h1>
</header>
<hr/>

<p>
The 2LPT and 3LPT sources are products of fields computed in real space, so modes above the Nyquist frequency fold back into the resolved modes. With the optional final argument <tt>Dealiasing::Padding</tt> the Fourier coefficients are embedded in a grid $3/2$ times larger (2LPT, the quadratic rule) or twice as large (3LPT, cubic terms), the potentials are computed there and cropped back to the original mesh. Only modes with $2|n_i| &lt; N$ in every direction are kept, so the ambiguous Nyquist planes of even meshes are zeroed; without padding the derivatives that need a sign at the Nyquist frequency ($ik_i$, $k_ik_j$ for $i\ne j$) are set to zero, which keeps the results covariant under reflections. The padded grids are $\approx 3.4$ (2LPT) and 8 (3LPT) times larger; the MPI remapping exchanges only the retained coefficients.
</p>
<div style="text-align: center">
  <div style="width:600px;display: inline-block;text-align: left">
    <pre class="prettyprint">
using namespace FML::COSMOLOGY::LPT;
compute_2LPT_potential_fourier(delta, phi2, Dealiasing::Padding);
compute_3LPT_displacement_field&lt;3&gt;(delta, psi, velocity, dlogDdt, false, Dealiasing::Padding);
</pre>
</div>
</div>
<p>
<tt>LPT/GrowthHistory.h</tt> exposes $D_1$, $D_2$ and their growth rates for the shared <a href="nbody.php#ForceOptions" style="display: inline">time-stepping coefficients</a>.
</p>
