<hr/>
<header class="entry-header text-center">
  <h1 class="entry-title">Table of contents</h1>
</header>
<hr/>

<p class="page-toc">
<a href="index.php">Back to the main page</a><br>
<a href="documentation.php">Back to the documentation</a><br>
<a href="#Overview">1. Overview: Delaunay Tessellation Field Estimator (DTFE)</a><br>
<a href="#Create">2. Creating the estimator</a><br>
<a href="#Density">3. The density field</a><br>
<a href="#Velocity">4. The velocity field and its derivatives</a><br>
<a href="#Fields">5. General fields</a><br>
</p>

<header class="entry-header text-center">
  <h1 class="entry-title" id="Overview">Overview: Delaunay Tessellation Field Estimator (DTFE)</h1>
</header>
<hr/>

<p>
The <a href="https://www.astro.rug.nl/~voronoi/DTFE/dtfe.html" style="display: inline">DTFE</a> uses the <a href="tesselation.php" style="display: inline">Delaunay triangulation</a> of a set of particles to interpolate quantities known at the particles to a regular grid. The density estimate and volume-covering interpolation follow <a href="https://arxiv.org/abs/astro-ph/0011007" style="display: inline">Schaap &amp; van de Weygaert (2000)</a>; see also the public-code paper by <a href="https://arxiv.org/abs/1105.0370" style="display: inline">Cautun &amp; van de Weygaert (2011)</a>. FML uses CGAL for tessellation geometry and has its own MPI slab/buffer and grid-sampling implementation. Resolution adapts to local point density. The density at a particle is
$$\rho_i = (D+1)\frac{m_i}{W_i}$$
where $W_i$ is the volume of all the tetrahedra that have the particle as a vertex (the contiguous Voronoi cell). Any quantity known at the particles (the velocity, the density, ...) is then interpolated linearly inside each Delaunay tetrahedron, so its gradient is constant in each tetrahedron. This gives us the velocity divergence, vorticity and shear directly from the particles without having to assign the velocity to a grid first (which suffers from empty cells in underdense regions). The interpolated density integrates exactly to the total mass.
</p>

<p>
The grid values are either the field at the grid point or the average over the grid cell (estimated with $n_{\rm sub}^3$ regularly spaced samples in the cell), and the averages are either volume weighted or mass weighted (weighted by the DTFE density). Grid point (ix,iy,iz) is at (ix,iy,iz)/Nmesh as in the rest of FML. Positions are in $[0,1)$, gradients are with respect to the positions in units of the box and densities are in units of the mean density (i.e. $1+\delta$).
</p>

<p>
With MPI each task computes the part of the grid it owns (the slab in $[x_{\rm min}, x_{\rm max})$) so the grid slabs need to match the particle domains: Nmesh must be divisible by the number of tasks. The buffer must be large enough for the tetrahedra the samples land in to be correct; we count the samples that land in tetrahedra that might not be correct and give a warning (increase buffer_fraction if this happens). This is only implemented in 3D and requires CGAL. Everything is in the namespace FML::TRIANGULATION (Triangulation/DTFE.h).
</p>

<p>
<b>Any box domain and grid.</b> <tt>DTFE(domain)</tt> works with particles on a Slab or Pencil domain, and the fields can be put on any grid type (<tt>density_to_grid&lt;G&gt;</tt>, <tt>velocity_to_grid&lt;G&gt;</tt>, e.g. a <a href="pencilgrid.php" style="display: inline">$\texttt{PencilGrid}$</a>).
</p>

<header class="entry-header text-center">
  <h1 class="entry-title" id="Create">Creating the estimator</h1>
</header>
<hr/>

<p>
<div style="text-align: center">
  <div style="width:600px;display: inline-block;text-align: left">
    <pre class="prettyprint">
// Make the tessellation and compute the DTFE density at the particles
// The particles must be inside the local domain (e.g. MPIParticles)
// and must not move while we use this
FML::TRIANGULATION::DTFE&lt;Particle&gt; dtfe;
dtfe.create(part.get_particles_ptr(), part.get_npart());

// The full signature
void create(T * p, size_t NumPart, double buffer_fraction = 0.5, double random_fraction = 0.3);

// The DTFE density at the particles (the first NumPart elements are the local particles)
const std::vector&lt;double&gt; &amp; density_at_particles = dtfe.get_density();

// Free up the memory when done
dtfe.free();
</pre>
</div>
</div>
</p>

<p>
All the grid methods below take the options
<div style="text-align: center">
  <div style="width:600px;display: inline-block;text-align: left">
    <pre class="prettyprint">
// How we get the value in a grid cell
enum class DTFESampling {
  GridPoint,  // The interpolated field at the grid point
  CellAverage // The average over the grid cell (nsub^3 samples in the cell)
};

// How we weight the samples in a cell when averaging
enum class DTFEWeighting {
  Volume, // Volume weighted average
  Mass    // Mass weighted average (weighted by the DTFE density)
};
</pre>
</div>
</div>
</p>

<header class="entry-header text-center">
  <h1 class="entry-title" id="Density">The density field</h1>
</header>
<hr/>

<p>
<div style="text-align: center">
  <div style="width:600px;display: inline-block;text-align: left">
    <pre class="prettyprint">
// The DTFE density field (in units of the mean density) on a grid (allocated here)
void density_to_grid(
    int Nmesh,
    FFTWGrid&lt;3&gt; &amp; grid,
    DTFESampling sampling = DTFESampling::CellAverage,
    int nsub = 2) const;

// Example
FML::GRID::FFTWGrid&lt;3&gt; density;
dtfe.density_to_grid(Nmesh, density);
</pre>
</div>
</div>
</p>

<header class="entry-header text-center">
  <h1 class="entry-title" id="Velocity">The velocity field and its derivatives</h1>
</header>
<hr/>

<p>
For this the particles must have a get_vel method. Only the grids that are asked for are allocated:
<div style="text-align: center">
  <div style="width:600px;display: inline-block;text-align: left">
    <pre class="prettyprint">
struct DTFEVelocityGrids {
  // The velocity v_x, v_y, v_z
  std::array&lt;FFTWGrid&lt;3&gt;, 3&gt; velocity;
  // The divergence theta = div v
  FFTWGrid&lt;3&gt; divergence;
  // The vorticity omega = curl v
  std::array&lt;FFTWGrid&lt;3&gt;, 3&gt; vorticity;
  // The shear sigma_ij = (d_i v_j + d_j v_i)/2 - delta_ij theta/3
  // in the order xx, xy, xz, yy, yz, zz
  std::array&lt;FFTWGrid&lt;3&gt;, 6&gt; shear;
};

void velocity_to_grid(
    T * p,
    int Nmesh,
    DTFEVelocityGrids &amp; result,
    DTFESampling sampling = DTFESampling::CellAverage,
    DTFEWeighting weighting = DTFEWeighting::Volume,
    int nsub = 2,
    bool compute_velocity = true,
    bool compute_divergence = true,
    bool compute_vorticity = true,
    bool compute_shear = true) const;

// Example
FML::TRIANGULATION::DTFEVelocityGrids vel;
dtfe.velocity_to_grid(part.get_particles_ptr(), Nmesh, vel);
// vel.divergence, vel.vorticity[i], vel.shear[i], vel.velocity[i] are now FFTWGrid&lt;3&gt;'s
</pre>
</div>
</div>
</p>
<p>
The shear and divergence can be used directly for a <a href="cosmicweb.php" style="display: inline">V-web classification</a>.
</p>

<header class="entry-header text-center">
  <h1 class="entry-title" id="Fields">General fields</h1>
</header>
<hr/>

<p>
Any set of fields known at the particles can be interpolated to a grid, optionally with their gradients. For complete freedom there is also the general method that calls a function for every grid cell we own with the density, the interpolated fields and their gradients:
<div style="text-align: center">
  <div style="width:600px;display: inline-block;text-align: left">
    <pre class="prettyprint">
// values[nfields * i + f] is field f of particle i
// gradient_grids: grid 3*f+j is the derivative of field f wrt x_j (nullptr to skip)
void fields_to_grid(
    int Nmesh,
    const std::vector&lt;double&gt; &amp; values,
    int nfields,
    std::vector&lt;FFTWGrid&lt;3&gt;&gt; &amp; field_grids,
    std::vector&lt;FFTWGrid&lt;3&gt;&gt; * gradient_grids = nullptr,
    DTFESampling sampling = DTFESampling::CellAverage,
    DTFEWeighting weighting = DTFEWeighting::Volume,
    int nsub = 2) const;

// Calls store(ix, iy, iz, rho, values, gradients) for every grid cell we own
// ix is the local index of the slab (0 &lt;= ix &lt; local_nx), rho the density,
// values the nfields interpolated fields and gradients[3*f + j] the derivative
// of field f with respect to x_j
template &lt;class F&gt;
void interpolate_to_grid(
    int Nmesh,
    ptrdiff_t local_x_start,
    ptrdiff_t local_nx,
    const std::vector&lt;double&gt; &amp; values,
    int nfields,
    DTFESampling sampling,
    DTFEWeighting weighting,
    int nsub,
    F &amp;&amp; store) const;
</pre>
</div>
</div>
</p>
<p>
See Triangulation/example/DTFEExample.cpp for a complete example.
</p>
