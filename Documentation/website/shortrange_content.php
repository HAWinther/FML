<hr/>
<header class="entry-header text-center">
  <h1 class="entry-title">Table of contents</h1>
</header>
<hr/>

<p class="page-toc">
<a href="index.php">Back to the main page</a><br>
<a href="documentation.php">Back to the documentation</a><br>
<a href="#Overview">1. Overview: P3M and TreePM</a><br>
<a href="#Split">2. The force split and softening</a><br>
<a href="#Yukawa">3. Newtonian plus Yukawa gravity</a><br>
<a href="#Methods">4. Cell lists, the octree and the API</a><br>
<a href="#PM">5. Coupling to the PM force</a><br>
<a href="#MPI">6. MPI ghosts and validation</a><br>
</p>

<header class="entry-header text-center">
  <h1 class="entry-title" id="Overview">Overview: P3M and TreePM</h1>
</header>
<hr/>

<p>
A particle-mesh (PM) force is accurate only on scales larger than a few mesh cells. P3M and TreePM codes add a short-range correction computed from particle pairs: the potential is split into a long-range part, smooth on the mesh scale and solved with FFTs, and a short-range part that falls off exponentially beyond a few split scales and is summed directly (cell lists, P3M) or with a tree (TreePM). <tt>FML/ShortRangeGravity</tt> (namespace FML::SHORT_RANGE_GRAVITY, header-only) provides these building blocks for Newtonian and Newtonian-plus-Yukawa gravity in a periodic box: the split kernels, a cell list, a monopole octree, the long-range Fourier filter and the MPI ghost exchange. Integration into the <a href="nbodycode.php" style="display: inline">COLA code</a> is not done yet.
</p>

<header class="entry-header text-center">
  <h1 class="entry-title" id="Split">The force split and softening</h1>
</header>
<hr/>

<p>
With the Gaussian potential split used by GADGET-2 (<a href="https://arxiv.org/abs/astro-ph/0505010" style="display: inline">Springel 2005</a>), the long-range potential is
$$\Phi_{\rm long}(\vec k) = \Phi(\vec k)\,e^{-k^2r_s^2},$$
and the complementary short-range pair acceleration of an unsoftened point mass is
$$\vec a_{\rm short} = Gm\frac{\vec r_j - \vec r}{r^3}\left[{\rm erfc}(q) + \frac{2q}{\sqrt\pi}e^{-q^2}\right],\qquad q = \frac{r}{2r_s}.$$
The sum is cut at the radius where the short-range window falls below a tolerance (e.g. $10^{-6}$). Particles are softened with the compact cubic-spline density of support $h$,
$$W(u) = \frac{8}{\pi h^3}\begin{cases}1 - 6u^2 + 6u^3 & u &lt; 1/2\\ 2(1-u)^3 & 1/2\le u &lt; 1\end{cases},\qquad u = r/h,$$
so the force is exactly Newtonian beyond $h$ ($h\approx 2.8\epsilon_{\rm Plummer}$). Inside $h$ the short-range force is the spline-softened Newtonian force minus the Gaussian long-range force (not the softened force times the unsoftened window, which gives a different total force).
</p>

<header class="entry-header text-center">
  <h1 class="entry-title" id="Yukawa">Newtonian plus Yukawa gravity</h1>
</header>
<hr/>

<p>
A fifth force with range $\lambda$ and strength $\alpha$ gives the point-mass potential. The complementary Yukawa Ewald expression below follows equations 30–32 of <a href="https://doi.org/10.1103/PhysRevE.103.043204" style="display: inline">Phys. Rev. E 103, 043204</a>; the stable evaluation, softening prescription and force plumbing are FML's implementation.
$$\Phi = -\frac{Gm}{r}\left[1 + \alpha e^{-r/\lambda}\right],\qquad F\propto 1 + \alpha\left(1 + \frac{r}{\lambda}\right)e^{-r/\lambda}.$$
The long-range Fourier kernel becomes $\frac{e^{-k^2r_s^2}}{k^2} + \alpha\frac{e^{-(k^2+\mu^2)r_s^2}}{k^2+\mu^2}$ with $\mu = 1/\lambda$, and the short-range Yukawa Green function is the Ewald decomposition
$$G_{\rm short}(r) = \frac{e^{-\mu r}{\rm erfc}(q - z) + e^{\mu r}{\rm erfc}(q + z)}{2r},\qquad q = \frac{r}{2r_s},\ z = \mu r_s,$$
evaluated with scaled error functions, series and heat-kernel integrals in the extreme limits (validated against 100-digit reference values). Negative $\alpha$ (repulsive) is supported.
</p>

<header class="entry-header text-center">
  <h1 class="entry-title" id="Methods">Cell lists, the octree and the API</h1>
</header>
<hr/>

<p>
The <b>cell list</b> bins the sources in cells at least one cutoff wide and sums all pairs in the 27 neighbouring cells. The <b>octree</b> stores mass, centre of mass and bounds per node and accepts a node as a monopole if it lies entirely within the cutoff and outside the softening region, within a single periodic image, and is small compared both to the distance ($\ell/d &lt; \theta$) and to the split scale. An opening angle of 0 gives exact sums. Results are accelerations; both are OpenMP parallel over targets.
</p>
<div style="text-align: center">
  <div style="width:700px;display: inline-block;text-align: left">
    <pre class="prettyprint">
#include &lt;FML/ShortRangeGravity/ShortRangeGravity.h&gt;
#include &lt;FML/ShortRangeGravity/Octree.h&gt;
using namespace FML::SHORT_RANGE_GRAVITY;
Options options;
options.boxsize = 250;
options.split_scale = 2;
options.softening_support = .1;
options.cutoff_tolerance = 1e-6;
options.coupling = 1;                              // G, or fml_acceleration_coupling(C, M, L)
Kernel kernel(options);                            // or Kernel(options, NewtonYukawa{alpha, lambda})
std::vector&lt;Particle&gt; particles;                   // position, mass, unique ID
auto exact  = direct_sum(particles, particles, kernel);
auto p3m    = CellList(particles, kernel).compute(particles);
auto treepm = Octree(particles, kernel, TreeOptions{.3, 16, 40}).compute(particles);</pre>
  </div>
</div>

<header class="entry-header text-center">
  <h1 class="entry-title" id="PM">Coupling to the PM force</h1>
</header>
<hr/>

<p>
<tt>apply_long_range_filter(fourier, kernel)</tt> multiplies a Fourier density (or potential) by the long-range filter before the ordinary PM force solve. For FML's box coordinates with Poisson equation $\nabla_x^2\Phi = C\delta$ (in SimulationCode $C = \frac{3}{2}\Omega_Ma$) and total mass $M$, a pair interacts as
$$-\nabla_x\Phi_{\rm pair} = \frac{CL^2}{4\pi M}\,m\frac{\vec r_j - \vec r}{|\vec r_j - \vec r|^3},$$
so <tt>fml_acceleration_coupling(C, M, L)</tt> gives the coupling that matches the mesh force.
</p>

<header class="entry-header text-center">
  <h1 class="entry-title" id="MPI">MPI ghosts and validation</h1>
</header>
<hr/>

<p>
With slab domain decomposition, <tt>exchange_slab_ghosts</tt> sends each particle once to every slab within the cutoff (including periodic and non-adjacent slabs), in bounded chunks and without gathering the particles; the local cell list or tree then computes the forces on the owned particles. The tests check the spline against independent quadrature, the direct/cell/tree agreement for uniform, clustered, coincident and periodic particles, periodic Ewald sums at several split and softening scales, the FML force sign and normalization, ghost completeness and the Yukawa kernels against 100-digit references. The octree is currently slower than the cell list; higher multipoles and better acceptance criteria are possible extensions.
</p>
