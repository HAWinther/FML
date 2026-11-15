<hr/>
<header class="entry-header text-center">
  <h1 class="entry-title">Table of contents</h1>
</header>
<hr/>

<p class="page-toc">
<a href="index.php">Back to the main page</a><br>
<a href="documentation.php">Back to the documentation</a><br>
<a href="#Overview">1. Overview: Grid assignment and interpolation</a><br>
<a href="#Assign">2. Grid assignment</a><br>
<a href="#Interpol">3. Interpolation</a><br>
<a href="#Conv">4. Convolutions</a><br>
<a href="#Interlace">5. Grid assignment using interlacing</a><br>
<a href="#Window">6. Deconvolution</a><br>
<a href="#Sheet">7. Sheet interpolation</a><br>
<a href="#NUFFT">8. Non-uniform FFTs (NUFFT)</a><br>
<a href="#AnyDomain">9. Assignment for particles on any domain (Peano&ndash;Hilbert)</a><br>
</p>

<header class="entry-header text-center">
  <h1 class="entry-title" id="Overview">Overview: Grid assignment and interpolation</h1>
</header>
<hr/>  
<p>
When working with particles we often want to assign particles to a grid to create a density field in either real or Fourier space (it is useful to have different methods for this as we can save Fourier transforms). We also want to be able to interpolate a grid to the particle positions. Here we provide methods for this. Everything lives in the namespace FML::INTERPOLATION and is found in Interpolation/ParticleGridInterpolation.h.
</p>

<p>
<b>Any grid.</b> Assignment and interpolation (with interlacing, offsets and deconvolution), the NUFFT and sheet interpolation work on any grid with FML's grid interface, e.g. a <a href="pencilgrid.php" style="display: inline">$\texttt{PencilGrid}$</a> with the particles on its pencils: the same function names. On grids other than <tt>FFTWGrid</tt> the kernel is applied as a separable stencil (the weights and index offsets computed once per axis for each particle).
</p>

<header class="entry-header text-center">
  <h1 class="entry-title" id="Assign">Grid assignment</h1>
</header>
<hr/>  

<p>
The code provides methods for assigning particles to a grid, i.e. computing the convolution $\delta * W$ with $W$ being an arbitrary order B-spline kernel (or using any user provided kernel). The kernel is defined via
$$H^{\rm ORDER} = H * H^{\rm ORDER-1}$$
where $*$ is a convolution and $H$ is the top-hat function $H = 1$ if a particle falls within $1/2$ of a cellsize from the center. In many dimensions the kernel is simply $H = \prod_{i=1}^{\rm NDIM} H_i$. For ORDER$=1$ we get NGP (Nearest Grid Point), ORDER$=2$ is CIC (Cloud in Cell), ORDER$=3$ is TSC (Triangular Shaped Cloud), ORDER$=4$ is PCS (Piecewise Cubic Spline) and ORDER$=5$ is PQS (Piecewise Quartic Spline). A nice property of this kernel is that its Fourier transform is simply
$$\left(\prod_{i=1}^{\rm NDIM}\frac{\sin(\frac{\pi}{2}\frac{k}{k_{\rm ny}})}{ (\frac{\pi}{2}\frac{k}{k_{\rm ny}}) }\right)^{\rm ORDER}$$
where $k_{\rm ny} = \pi N_{\rm grid}$ is the Nyquist frequency of the grid (see Deconvolution below for where this is used). Note that we need to visit ${\rm ORDER}^{\rm NDIM}$ cells per particle to assign the density so the cost of the method goes up quite steeply with the order. For most applications CIC is what you want to use as it gives a good compromise between speed and accuracy.
</p>

<p>
<div style="text-align: center">
  <div style="width:600px;display: inline-block;text-align: left">
    <pre class="prettyprint">
// Assign particle to grid using NGP, CIC, TSC, ... etc.
// The result is the overdensity field rho/rho_mean - 1
template&lt;int N, class T&gt;
  void particles_to_grid(
      const T *part,
      size_t NumPart,
      size_t NumPartTot,
      FFTWGrid&lt;N&gt; &amp; overdensity,
      std::string density_assignment_method);

// The same with the order as a template parameter (ORDER = 1 is NGP, 2 is CIC, ...)
template&lt;int N, int ORDER, class T&gt;
  void particles_to_grid(
      const T *part,
      size_t NumPart,
      size_t NumPartTot,
      FFTWGrid&lt;N&gt; &amp; overdensity);
</pre>
</div>
</div>
</p>
<p>
The particle class <a href="global.php#Part" style="display: inline">must have</a> the method get_pos() which gives a pointer to the position of the particle and get_ndim() that tells us how many components the positions have. If your particles have different mass you should also have a get_mass() function. This mass will then be used when we assign particles to a grid. If we don't have this method then we assume that all particles have the same mass. For galaxy surveys the get_mass function should just return the weight of the galaxy. The masses can be arbitrarily normalized (though having them in units of the mean mass is a good convention to use) as we use the mass divided by the mean mass as the weight when assigning to the grid.
</p>

<header class="entry-header text-center">
  <h1 class="entry-title" id="Interpol">Interpolation</h1>
</header>
<hr/>  

<p>
We also provide methods for interpolating a grid to particle positions using the same kernels as above (so interpolation_method takes the same values NGP, CIC, TSC etc. as above). There is also a version that interpolates $N$ grids at once (e.g. the components of a force vector) which is faster than doing them one by one:
<div style="text-align: center">
  <div style="width:600px;display: inline-block;text-align: left">
    <pre class="prettyprint">
template&lt;int N, class T&gt;
  void interpolate_grid_to_particle_positions(
      const FFTWGrid&lt;N&gt; &amp; grid,
      const T *part,
      size_t NumPart,
      std::vector&lt;FloatType&gt; &amp; interpolated_values,
      std::string interpolation_method);

template&lt;int N, class T&gt;
  void interpolate_grid_vector_to_particle_positions(
      const std::array&lt;FFTWGrid&lt;N&gt;, N&gt; &amp; grid_vec,
      const T *part,
      size_t NumPart,
      std::array&lt;std::vector&lt;FloatType&gt;, N&gt; &amp; interpolated_values_vec,
      std::string interpolation_method);
</pre>
</div>
</div>
</p>

<header class="entry-header text-center">
  <h1 class="entry-title" id="Conv">Convolution</h1>
</header>
<hr/>  

<p>
And if you want to convolve the grid with an arbitrary kernel you can use (ORDER is the width of the kernel in units of the cell size):
<div style="text-align: center">
  <div style="width:600px;display: inline-block;text-align: left">
    <pre class="prettyprint">
template&lt;int N, int ORDER&gt;
  void convolve_grid_with_kernel(
      const FFTWGrid&lt;N&gt; &amp; grid_in,
      FFTWGrid&lt;N&gt; &amp; grid_out,
      const std::function&lt;FloatType(std::array&lt;double, N&gt; &amp; dx)&gt; &amp; convolution_kernel);
</pre>
</div>
</div>
</p>
<p>
The argument to the kernel is how far away the point is from the cell in units of the cell-size so the NGP kernel would be $1$ if all $|dx_i| &lt; 0.5$ and zero otherwise.
</p>

<header class="entry-header text-center">
  <h1 class="entry-title" id="Interlace">Grid assignment using interlacing</h1>
</header>
<hr/>  

<p>
Assign particles to the grid using the original particles and particles displaced by $1/2$ grid-cell and combine this to get the Fourier transform of the density field. This helps reduce aliasing and is useful for computing correlation functions in Fourier space from particles. NB: this method does not deconvolve the window function used for the density assignment so you will have to call deconvolve_window_function_fourier (see next section) if you want to do this:
</p>

<p>
<div style="text-align: center">
  <div style="width:600px;display: inline-block;text-align: left">
    <pre class="prettyprint">
// Returns density_grid_fourier with delta(k). This grid has to be allocated beforehand
template&lt;int N, class T&gt;
void particles_to_fourier_grid(T * part,
                               size_t NumPart,
                               size_t NumPartTot,
                               FFTWGrid&lt;N&gt; &amp; density_grid_fourier,
                               std::string density_assignment_method,
                               bool interlacing);
</pre>
</div>
</div>
</p>

<header class="entry-header text-center">
  <h1 class="entry-title" id="Window">Deconvolution</h1>
</header>
<hr/>  
<p>
When assigning to the grid we don't compute $\delta$, but rather $\delta * W$ where $W$ is the density assignment kernel. In Fourier space this convolution becomes a simple product and we can get back the "true" $\delta$ if we want by dividing by the Fourier transform of the kernel we use. The method below does this and is used by for example algorithms to compute Fourier space correlation functions:
<div style="text-align: center">
  <div style="width:600px;display: inline-block;text-align: left">
    <pre class="prettyprint">
template &lt;int N&gt;
void deconvolve_window_function_fourier(
    FFTWGrid&lt;N&gt; &amp; density_grid_fourier,
    std::string density_assignment_method);
</pre>
</div>
</div>
</p>

<header class="entry-header text-center">
  <h1 class="entry-title" id="Sheet">Sheet interpolation</h1>
</header>
<hr/>
<p>
If the particles started out on a regular lattice (as in an N-body simulation) then Interpolation/SheetInterpolation.h provides spectral sheet interpolation, drawing on the phase-space sheet work of <a href="https://arxiv.org/abs/1210.6652" style="display: inline">Hahn, Abel &amp; Kaehler (2013)</a>, <a href="https://arxiv.org/abs/1909.00008" style="display: inline">Stuecker et al. (2020)</a> and <a href="https://arxiv.org/abs/2409.19049" style="display: inline">List, Hahn &amp; Rampf (2024)</a>. FML's tracer-grid algorithm and choices are its own implementation. The displacement field $\Psi(q) = x - q$ is Fourier interpolated to $R^{\rm NDIM}$ tracer points per particle which are then assigned to the grid. This strongly suppresses shot noise and lattice effects on large scales, but rings (Gibbs) after shell crossing so it is most useful at early times or on large scales. The particles need a get_q() method (their Lagrangian coordinate):
<div style="text-align: center">
  <div style="width:600px;display: inline-block;text-align: left">
    <pre class="prettyprint">
template &lt;int N, class T&gt;
void particles_to_fourier_grid_sheet(
    const T * part,
    size_t NumPart,
    int Npart_1D,
    int refinement,
    FFTWGrid&lt;N&gt; &amp; density_fourier,
    std::string density_assignment_method,
    double nufft_epsilon = 1e-6);
</pre>
</div>
</div>
</p>

<header class="entry-header text-center">
  <h1 class="entry-title" id="NUFFT">Non-uniform FFTs (NUFFT)</h1>
</header>
<hr/>

<p>
Mass assignment with NGP/CIC/TSC/... gives the Fourier transform of the particles multiplied by a window function and contaminated by aliasing. The NUFFT (NUFFT/NUFFT.h) computes it to a chosen precision $\epsilon$ instead,
$$F(\vec k) = {\rm norm}\sum_pc_pe^{-i\vec k\cdot\vec x_p}\quad\text{(type 1)},\qquad f(\vec x_p) = \sum_{\vec k}F(\vec k)e^{i\vec k\cdot\vec x_p}\quad\text{(type 2)},$$
with the spreading, FFT, and deconvolution method described by <a href="https://doi.org/10.1137/18M120885X" style="display: inline">Barnett, Magland &amp; af Klinteberg (2019)</a>. The fiducial spreading kernel is now the order-zero prolate spheroidal wave function (PSWF); the exponential-of-semicircle (ES) kernel remains available as an alternative. The PSWF is defined by the <a href="https://dlmf.nist.gov/30.2" style="display: inline">spheroidal differential equation</a>; its numerical properties are treated in <a href="https://doi.org/10.1007/978-1-4614-8259-8" style="display: inline">Osipov, Rokhlin &amp; Xiao (2013)</a>. FML follows the PSWF width/shape strategy introduced in <a href="https://finufft.readthedocs.io/en/v2.5.0/changelog.html" style="display: inline">FINUFFT 2.5</a>, but computes the kernel independently. Type 1 spreads each particle onto a fine grid, FFTs, deconvolves and retains the requested coarse-grid modes; type 2 reverses this operation. The implementation works with MPI and OpenMP. Its accuracy was compared against <a href="https://github.com/flatironinstitute/finufft/releases/tag/v2.5.1" style="display: inline">FINUFFT 2.5.1</a>. <tt>particles_to_fourier_grid_nufft</tt> is a drop-in replacement for <tt>particles_to_grid</tt> + <tt>fftw_r2c</tt> without window and aliasing, and <tt>compute_force_nufft</tt> gives the accurate PM force at particles (used in the COLA code for the early-time force).
</p>

<header class="entry-header text-center">
  <h1 class="entry-title" id="AnyDomain">Assignment for particles on any domain (Peano&ndash;Hilbert)</h1>
</header>
<hr/>
<p>
<tt>particles_to_grid</tt> needs every particle to be on the task that owns its cell of the grid (it stops with a clear message if a particle is outside the local domain, e.g. if <tt>communicate_particles</tt> was not called after moving them). When the particles are distributed differently from the grid, for example along a Peano&ndash;Hilbert curve (<tt>FML::Domain::peano_hilbert</tt>, or following an <a href="amr.php" style="display: inline">adaptive grid</a>) while the grid is in slabs or pencils, use <tt>Interpolation/DistributedDeposit.h</tt>. It does the assignment as GADGET's PM step (Springel 2005), without moving the particles: every task deposits its particles into a dense patch over the bounding box of their cells and sends each part of the patch to the task that owns those cells.
</p>
<pre class="prettyprint">
#include &lt;FML/Interpolation/DistributedDeposit.h&gt;

// As particles_to_grid (NGP-PQS, masses, interlacing offsets), particles on any task
FML::INTERPOLATION::particles_to_grid_any_domain(particles, grid, "CIC");
</pre>
<p>
The result equals that of <tt>particles_to_grid</tt> up to the order of the floating-point sums (tested to $10^{-15}$ relative on slabs and every pencil layout, 1&ndash;4 tasks), is deterministic with OpenMP, and the memory of the patch is bounded (a non-compact domain is done in several passes). <tt>deposit_to_partitioned_mesh</tt> does the same for any box decomposition of a mesh, with a callback per cell.
</p>
