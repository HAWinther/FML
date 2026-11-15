<hr/>
<header class="entry-header text-center">
  <h1 class="entry-title">Table of contents</h1>
</header>
<hr/>

<p class="page-toc">
<a href="index.php">Back to the main page</a><br>
<a href="documentation.php">Back to the documentation</a><br>
<a href="#Overview">1. Overview: Cosmic web classification</a><br>
<a href="#TWeb">2. T-web</a><br>
<a href="#VWeb">3. V-web</a><br>
<a href="#ORIGAMI">4. ORIGAMI</a><br>
<a href="#Halos">5. ORIGAMI halos</a><br>
<a href="#Storage">6. Compact output</a><br>
</p>

<header class="entry-header text-center">
  <h1 class="entry-title" id="Overview">Overview: Cosmic web classification</h1>
</header>
<hr/>

<p>
Methods for classifying the cosmic web into voids, sheets (walls), filaments and knots (halos). In Eulerian space we classify the cells of a grid from the eigenvalues $\lambda_1 \geq \lambda_2 \geq \lambda_3$ of a tensor: a cell is a void, sheet, filament or knot if $0$, $1$, $2$ or $3$ of the eigenvalues are above the threshold $\lambda_{\rm th}$. The T-web follows the tidal-tensor classification of <a href="https://arxiv.org/abs/astro-ph/0610280" style="display: inline">Hahn et al. (2007)</a> and <a href="https://arxiv.org/abs/0809.4135" style="display: inline">Forero-Romero et al. (2009)</a>; the V-web follows <a href="https://arxiv.org/abs/1201.3367" style="display: inline">Hoffman et al. (2012)</a>. In Lagrangian space <a href="https://arxiv.org/abs/1201.2353" style="display: inline">ORIGAMI (Falck, Neyrinck &amp; Szalay 2012)</a> classifies particles by shell crossing. FML's threshold, axis sampling and halo-grouping details are implementation choices; the citation describes the method, not exact catalogue equivalence. Everything is in the namespace FML::COSMICWEB (CosmicWeb/CosmicWeb.h, CosmicWeb/ORIGAMI.h and CosmicWeb/ORIGAMIHalos.h). The eigenvalues are computed with Jacobi rotations (accurate also for (nearly) degenerate eigenvalues) so unlike TidalTensor/Hessian.h this does not need GSL. Grid cell (ix,iy,iz) is at (ix,iy,iz)/Nmesh and gradients are with respect to positions in units of the box.
</p>

<p>
The result of the grid classifications:
<div style="text-align: center">
  <div style="width:600px;display: inline-block;text-align: left">
    <pre class="prettyprint">
enum WebType : int { VOID = 0, SHEET = 1, FILAMENT = 2, KNOT = 3 };

struct CosmicWeb {
  // The class of every cell (0 = void, 1 = sheet, 2 = filament, 3 = knot)
  FFTWGrid&lt;3&gt; classification;
  // The eigenvalues lambda_1 &gt;= lambda_2 &gt;= lambda_3 (only if asked for)
  std::array&lt;FFTWGrid&lt;3&gt;, 3&gt; eigenvalues;
  // The fraction of the volume and of the mass in voids, sheets, filaments and knots
  std::array&lt;double, 4&gt; volume_fraction;
  std::array&lt;double, 4&gt; mass_fraction;
};

// The general method: classify a symmetric tensor given by its components
// xx, xy, xz, yy, yz, zz. With density (in units of the mean) we also get the mass fractions
void classify_tensor(
    const std::array&lt;FFTWGrid&lt;3&gt;, 6&gt; &amp; tensor,
    double lambda_threshold,
    CosmicWeb &amp; result,
    bool store_eigenvalues = false,
    const FFTWGrid&lt;3&gt; * density = nullptr);
</pre>
</div>
</div>
</p>

<p>
<b>Pencils.</b> <tt>CosmicWebT&lt;G&gt;</tt> (<tt>CosmicWeb</tt> = slabs) classifies on any grid type, e.g. a <a href="pencilgrid.php" style="display: inline">$\texttt{PencilGrid}$</a>; the compact byte classification keeps the local $x$ and $y$ ranges.
</p>

<header class="entry-header text-center">
  <h1 class="entry-title" id="TWeb">T-web</h1>
</header>
<hr/>

<p>
The T-web uses the tidal tensor $T_{ij} = \partial_i\partial_j\phi$ with $\nabla^2\phi = \delta$ (the density contrast smoothed with a Gaussian). In Fourier space $T_{ij}(k) = \frac{k_ik_j}{k^2}\delta(k)$ and the trace is $\delta$. This requires FFTW. See <a href="https://arxiv.org/abs/astro-ph/0610280" style="display: inline">Hahn et al. (2007)</a> and <a href="https://arxiv.org/abs/0809.4135" style="display: inline">Forero-Romero et al. (2009)</a>.
<div style="text-align: center">
  <div style="width:600px;display: inline-block;text-align: left">
    <pre class="prettyprint">
struct TWebOptions {
  double smoothing_scale{0.0};       // Gaussian smoothing scale in units of the box (0: none)
  double lambda_threshold{0.0};      // Threshold for the eigenvalues
  bool store_eigenvalues{false};     // Also return the eigenvalues
  std::string density_assignment_method{"CIC"}; // For the particle method
};

// From a density contrast grid in real space
void TWeb(const FFTWGrid&lt;3&gt; &amp; delta, const TWebOptions &amp; options, CosmicWeb &amp; result);

// From particles (assigned to a grid of size Nmesh)
template &lt;class T&gt;
void TWeb(T * p, size_t NumPart, int Nmesh, const TWebOptions &amp; options, CosmicWeb &amp; result);

// Just the tidal tensor (components xx, xy, xz, yy, yz, zz in real space)
void tidal_tensor(const FFTWGrid&lt;3&gt; &amp; delta, double smoothing_scale, std::array&lt;FFTWGrid&lt;3&gt;, 6&gt; &amp; tensor);

// Example
FML::COSMICWEB::TWebOptions options;
options.smoothing_scale = 2.0 / boxsize; // 2 Mpc/h
options.lambda_threshold = 0.2;
FML::COSMICWEB::CosmicWeb web;
FML::COSMICWEB::TWeb(part.get_particles_ptr(), part.get_npart(), Nmesh, options, web);
</pre>
</div>
</div>
</p>

<header class="entry-header text-center">
  <h1 class="entry-title" id="VWeb">V-web</h1>
</header>
<hr/>

<p>
The V-web uses the velocity shear $\Sigma_{ij} = -\frac{1}{2H}(\partial_iv_j + \partial_jv_i)$, following <a href="https://arxiv.org/abs/1201.3367" style="display: inline">Hoffman et al. (2012)</a>. The velocity gradient can for example be computed with the <a href="dtfe.php" style="display: inline">DTFE</a>: from the traceless shear $\sigma_{ij}$ and the divergence $\theta$ we have $\partial_iv_j + \partial_jv_i = 2(\sigma_{ij} + \delta_{ij}\theta/3)$.
<div style="text-align: center">
  <div style="width:600px;display: inline-block;text-align: left">
    <pre class="prettyprint">
// The tensor we classify is normalization * (d_i v_j + d_j v_i) / 2
// with normalization = -1/H (in the units of the velocity and positions)
void VWeb(
    const std::array&lt;FFTWGrid&lt;3&gt;, 6&gt; &amp; shear,
    const FFTWGrid&lt;3&gt; &amp; divergence,
    double normalization,
    double lambda_threshold,
    CosmicWeb &amp; result,
    bool store_eigenvalues = false);

// Example
FML::TRIANGULATION::DTFEVelocityGrids vel;
dtfe.velocity_to_grid(part.get_particles_ptr(), Nmesh, vel);
FML::COSMICWEB::CosmicWeb web;
FML::COSMICWEB::VWeb(vel.shear, vel.divergence, -1.0 / H, 0.1, web);
</pre>
</div>
</div>
</p>

<header class="entry-header text-center">
  <h1 class="entry-title" id="ORIGAMI">ORIGAMI</h1>
</header>
<hr/>

<p>
ORIGAMI (<a href="https://arxiv.org/abs/1201.2353" style="display: inline">Falck, Neyrinck &amp; Szalay 2012</a>) finds the morphology of the particles from stream crossings in Lagrangian space. The particles must have started on a regular lattice with $N_{\rm lattice}^3$ points. Two particles on the same row of the lattice have crossed along that row if their order along the row direction is now reversed. A particle is crossed along an axis if it has crossed with a particle within search_fraction of the row (in either direction). We check 9 axes (x, y, z and the two diagonals in each coordinate plane) and the morphology is the largest number of crossed axes in one of the 4 sets of orthogonal axes: 0 = void, 1 = wall, 2 = filament, 3 = halo. Positions must be in $[0,1)$ and the displacement of a particle from its lattice point must be less than half the box. With MPI the particles can be anywhere (we move the rows around).
<div style="text-align: center">
  <div style="width:600px;display: inline-block;text-align: left">
    <pre class="prettyprint">
enum OrigamiMorphology : int { ORIGAMI_VOID = 0, ORIGAMI_WALL = 1, ORIGAMI_FILAMENT = 2, ORIGAMI_HALO = 3 };

struct OrigamiOptions {
  // Use the 3 extra sets of axes with the 45 degree diagonals (as in Falck et al. 2012)
  // If false only the Cartesian axes are used
  bool use_diagonal_axes{true};
  // How far along a row of the lattice we look for crossings (Falck et al.: 1/4)
  double search_fraction{0.25};
};

// lattice_coordinate: optional, the lattice point (ix,iy,iz) of a particle. The default uses
// the particle ID with id = (ix * Nlattice + iy) * Nlattice + iz which is how
// MPIParticles::create_particle_grid and the NBody initial conditions number the particles
// crossed_axes: optional, the bitmask of the axes each particle is crossed along
template &lt;class T&gt;
void ORIGAMI(
    T * p,
    size_t NumPart,
    int Nlattice,
    std::vector&lt;int&gt; &amp; morphology,
    const OrigamiOptions &amp; options = OrigamiOptions(),
    std::function&lt;std::array&lt;int, 3&gt;(T &amp;)&gt; lattice_coordinate = nullptr,
    std::vector&lt;unsigned int&gt; * crossed_axes = nullptr);
</pre>
</div>
</div>
</p>

<header class="entry-header text-center">
  <h1 class="entry-title" id="Halos">ORIGAMI halos</h1>
</header>
<hr/>

<p>
The halo grouping step of ORIGAMI. The halo particles (morphology 3) are linked on the <a href="tesselation.php" style="display: inline">Delaunay tessellation</a> of all the particles (this requires CGAL):
<ol>
<li>The density of every particle is mass / Voronoi volume in units of the mean (VTFE).</li>
<li>Halo particles above core_density_threshold that are Delaunay neighbors form the cores.</li>
<li>The other halo particles are added to the cores through their Delaunay neighbors in rounds: in each round every halo particle that is not in a halo yet and has a neighbor in a halo joins the halo of its densest such neighbor (from the previous round, so the result does not depend on the order or the number of tasks). Every halo then has exactly one core.</li>
<li>The halo particles that are not connected to a core form groups by themselves (connected pieces).</li>
<li>Groups with fewer than min_particles particles are not kept. No unbinding is done.</li>
</ol>
<div style="text-align: center">
  <div style="width:600px;display: inline-block;text-align: left">
    <pre class="prettyprint">
struct OrigamiHalo {
  long long int id;                 // Sorted by the number of particles, largest first
  long long int npart;
  long long int ncore;              // Number of core particles (0 for groups without a core)
  double mass;
  std::array&lt;double, 3&gt; pos;        // Center of mass
  std::array&lt;double, 3&gt; pos_densest; // The densest particle
  double density_max;
};

struct OrigamiHaloOptions {
  double core_density_threshold{200.0};
  long long int min_particles{20};
  bool keep_groups_without_core{true};
  double buffer_fraction{0.5};      // Tessellation buffer (MPI)
  double random_fraction{0.3};      // Guard points (MPI)
};

// halo_id: for each local particle the id of its halo in the catalogue (-1 if none)
// halos: the catalogue (on all tasks)
template &lt;class T&gt;
void ORIGAMIHalos(
    T * p,
    size_t NumPart,
    int Nlattice,
    const std::vector&lt;int&gt; &amp; morphology,
    std::vector&lt;long long int&gt; &amp; halo_id,
    std::vector&lt;OrigamiHalo&gt; &amp; halos,
    const OrigamiHaloOptions &amp; options = OrigamiHaloOptions(),
    std::function&lt;std::array&lt;int, 3&gt;(T &amp;)&gt; lattice_coordinate = nullptr);

// Example
std::vector&lt;int&gt; morphology;
FML::COSMICWEB::ORIGAMI(part.get_particles_ptr(), part.get_npart(), Npart_1D, morphology);
std::vector&lt;long long int&gt; halo_id;
std::vector&lt;FML::COSMICWEB::OrigamiHalo&gt; halos;
FML::COSMICWEB::ORIGAMIHalos(part.get_particles_ptr(), part.get_npart(), Npart_1D, morphology, halo_id, halos);
</pre>
</div>
</div>
</p>

<header class="entry-header text-center">
  <h1 class="entry-title" id="Storage">Compact output</h1>
</header>
<hr/>

<p>
The classification is returned in an FFTWGrid by default. With <tt>options.classification_storage = ClassificationStorage::Bytes</tt> it is stored as one byte per cell instead (8 times less memory in double precision); <tt>get_classification({ix, iy, iz})</tt> works with both (local slab coordinates). The <a href="statistics.php#Morphology" style="display: inline">Minkowski functionals</a> of excursion sets complement the cosmic-web classification.
</p>
