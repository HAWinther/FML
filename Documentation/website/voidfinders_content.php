<hr/>
<header class="entry-header text-center">
  <h1 class="entry-title">Table of contents</h1>
</header>
<hr/>

<p class="page-toc">
<a href="index.php">Back to the main page</a><br>
<a href="documentation.php">Back to the documentation</a><br>
<a href="#Overview">1. Overview: Void finders</a><br>
<a href="#Catalogue">2. The void catalogue</a><br>
<a href="#Zobov">3. ZOBOV</a><br>
<a href="#Voxel">4. Voxel watershed</a><br>
<a href="#Merging">5. Zone merging</a><br>
<a href="#Spherical">6. Spherical underdensities</a><br>
</p>

<header class="entry-header text-center">
  <h1 class="entry-title" id="Overview">Overview: Void finders</h1>
</header>
<hr/>

<figure class="cinematic-figure" id="cinematic-04-inside-cosmic-void">
  <a href="images/cinematic/04-inside-cosmic-void.jpg" aria-label="View full-size illustration: Inside a cosmic void">
    <img src="images/cinematic/04-inside-cosmic-void.jpg" width="1774" height="887" loading="lazy" decoding="async" alt="A nearly empty cosmic void framed by a distant irregular filamentary boundary">
  </a>
  <figcaption><strong>Inside a cosmic void.</strong> Conceptual view from an underdense cosmic region toward its filamentary boundary; the void has no physical shell or sharp wall. <span class="cinematic-credit">AI-generated concept illustration.</span></figcaption>
</figure>

<p>
We provide three void finders for periodic boxes, the first two following <a href="https://github.com/seshnadathur/Revolver" style="display: inline">Revolver</a> (Nadathur et al. 2019):
<ul>
<li>ZobovVoidFinder: watershed on the Voronoi densities of the particles (<a href="https://arxiv.org/abs/0712.3049" style="display: inline">ZOBOV</a>, Neyrinck 2008). Requires CGAL.</li>
<li>VoxelVoidFinder: watershed on a smoothed density field on a grid. The watershed step, VoxelWatershed, can be used directly on any density field (e.g. from the <a href="dtfe.php" style="display: inline">DTFE</a>).</li>
<li>SphericalVoidFinder: spheres with a given integrated density contrast, following <a href="https://doi.org/10.1111/j.1365-2966.2005.09500.x" style="display: inline">Padilla, Ceccarelli &amp; Lambas (2005)</a>; FML's recentering, overlap and distributed search policies are implementation details.</li>
</ul>
As in Revolver the watershed voids are by default the watershed basins (zones) without any merging, but the zones can also be merged into a hierarchy of voids. The watershed finders can also find overdense structures ("clusters") by running the same algorithm on 1/density. Positions, volumes and radii are in units of the box (positions in $[0,1)$) and densities are in units of the mean density. The particles must be inside the local domain (e.g. MPIParticles) and the result is the same on all tasks. Everything is in the namespace FML::VOIDFINDER (the headers in the folder VoidFinders) and VoidFinders/example shows how to use all of them.
</p>

<p>
<b>Pencils.</b> The voxel watershed and the spherical void finder take the grid type as a template argument (<tt>VoxelVoidFinder&lt;T, Grid&gt;</tt>) and work with the particles on a <a href="pencilgrid.php" style="display: inline">pencil</a> domain (the watershed labels are kept in <tt>GhostedCells</tt>, with ghost layers along $x$ and $y$).
</p>

<header class="entry-header text-center">
  <h1 class="entry-title" id="Catalogue">The void catalogue</h1>
</header>
<hr/>

<p>
All the void finders return a std::vector&lt;Void&gt;:
<div style="text-align: center">
  <div style="width:600px;display: inline-block;text-align: left">
    <pre class="prettyprint">
struct Void {
  // Number in the catalogue (sorted by the minimum density, maximum for clusters)
  long long int id;
  // The center: the circumcenter of the minimum and its 3 lowest density mutually
  // adjacent neighbors (ZOBOV), the position of the minimum (voxel) or the center (spherical)
  std::array&lt;double, 3&gt; pos;
  // The position of the minimum density particle/voxel (maximum for clusters)
  std::array&lt;double, 3&gt; pos_min;
  // The barycenter: weighted by the Voronoi volumes (ZOBOV) or by 1/density (voxel)
  std::array&lt;double, 3&gt; pos_barycenter;
  double volume;
  // The effective radius (3 volume / 4 pi)^(1/3)
  double radius;
  // The density of the minimum (maximum for clusters) in units of the mean
  double density_min;
  // The average density in units of the mean
  double density_avg;
  // The lowest density at which the void links to a neighbor divided by density_min
  // (ZOBOV's DensRatio, the significance of the void)
  double density_ratio;
  // Number of particles (ZOBOV) or voxels (voxel) in the void
  long long int nmembers;
  // Is this a cluster (overdensity) and not a void
  bool cluster;
  // With zone merging: the id of the void this void is part of (-1 for a top-level void)
  // and the number of zones in it
  long long int parent;
  long long int nzones;

  double delta_min() const;
  double delta_avg() const;
  // The lambda parameter delta_avg R^1.2 (R^1.6 for clusters) with R in units of the boxsize given
  double lambda(double boxsize) const;
};

// Write the catalogue to file (only task 0 writes), distances are multiplied by boxsize
// Columns as in Revolver followed by the extra info
void write_catalogue(const std::string &amp; filename, const std::vector&lt;Void&gt; &amp; voids, double boxsize);
</pre>
</div>
</div>
</p>

<header class="entry-header text-center">
  <h1 class="entry-title" id="Zobov">ZOBOV</h1>
</header>
<hr/>

<p>
We make a periodic Delaunay tessellation of the particles, compute the Voronoi densities (mass / Voronoi volume) and every particle is linked to its lowest density Delaunay neighbor. A zone (<a href="watershed.php" style="display: inline">watershed basin</a>) is all the particles that end up at the same density minimum. Every zone that passes the cuts (minimum density below min_dens_cut and at least min_num_particles particles) is a void. The volume is the sum of the Voronoi volumes, the average density is mass/volume and the density ratio is the lowest link density to a neighboring zone (a link between two neighboring particles has the larger of the two densities) divided by the minimum density. The particles only need a get_pos method (get_mass is used if it exists).
<div style="text-align: center">
  <div style="width:600px;display: inline-block;text-align: left">
    <pre class="prettyprint">
// Options for ZobovVoidFinder (defaults as in Revolver)
struct ZobovOptions {
  double buffer_fraction{0.5};        // Tessellation buffer (MPI)
  double random_fraction{0.3};        // Guard points (MPI)
  bool find_clusters{false};          // Find overdensities instead of voids
  double min_dens_cut{1.0};           // Voids: minimum density below this
  double max_dens_cut{1.0};           // Clusters: maximum density above this
  long long int min_num_particles{5}; // Minimum number of particles in a void
  ZoneMerging merging{ZoneMerging::None};
  double link_density_threshold{0.2}; // For ZoneMerging::Threshold
};

template &lt;class T&gt;
std::vector&lt;Void&gt; ZobovVoidFinder(T * p, size_t NumPart, const ZobovOptions &amp; options = ZobovOptions());

// Example
FML::VOIDFINDER::ZobovOptions zobov_options;
auto voids = FML::VOIDFINDER::ZobovVoidFinder(part.get_particles_ptr(), part.get_npart(), zobov_options);
FML::VOIDFINDER::write_catalogue("zobov_voids.txt", voids, boxsize);
</pre>
</div>
</div>
</p>

<header class="entry-header text-center">
  <h1 class="entry-title" id="Voxel">Voxel watershed</h1>
</header>
<hr/>

<p>
The voxel void finder in Revolver: assign the particles to a grid (CIC), smooth with a Gaussian and find the watershed zones of the density field. Every voxel is linked to the lowest of its 6 neighbors if that is lower than itself, and a zone is all the voxels that end up at the same minimum. The defaults for the grid size and the smoothing scale are as in Revolver: the cell size is half the radius of the sphere with one particle on average and the smoothing scale is the mean particle separation. Smoothing requires FFTW.
<div style="text-align: center">
  <div style="width:600px;display: inline-block;text-align: left">
    <pre class="prettyprint">
struct VoxelOptions {
  int Nmesh{0};                               // 0: as in Revolver
  double smoothing_scale{-1.0};               // Units of the box. Negative: as in Revolver. 0: none
  std::string smoothing_method{"revolver"};   // "revolver" (Gaussian sampled on the grid, truncated
                                              // at 4 sigma as scipy does) or "gaussian" (in Fourier space)
  std::string density_assignment_method{"CIC"};
  bool find_clusters{false};
  double min_dens_cut{1.0};
  double max_dens_cut{1.0};
  ZoneMerging merging{ZoneMerging::None};
  double link_density_threshold{0.2};
};

// density_grid: optional, returns the smoothed density field we used
template &lt;class T&gt;
std::vector&lt;Void&gt; VoxelVoidFinder(
    T * p,
    size_t NumPart,
    const VoxelOptions &amp; options = VoxelOptions(),
    FFTWGrid&lt;3&gt; * density_grid = nullptr);

// The watershed on any density field (in units of the mean density, 1 + delta)
std::vector&lt;Void&gt; VoxelWatershed(
    const FFTWGrid&lt;3&gt; &amp; density,
    bool find_clusters = false,
    double density_cut = 1.0,
    ZoneMerging merging = ZoneMerging::None,
    double link_density_threshold = 0.2);

// Example: voids in the DTFE density field
FML::GRID::FFTWGrid&lt;3&gt; density;
dtfe.density_to_grid(Nmesh, density);
auto voids = FML::VOIDFINDER::VoxelWatershed(density);
</pre>
</div>
</div>
</p>

<header class="entry-header text-center">
  <h1 class="entry-title" id="Merging">Zone merging</h1>
</header>
<hr/>

<p>
For both watershed finders the zones can be merged into voids by setting merging in the options:
<div style="text-align: center">
  <div style="width:600px;display: inline-block;text-align: left">
    <pre class="prettyprint">
enum class ZoneMerging {
  None,      // Every zone is a void (as in Revolver)
  Threshold, // Zones are merged through links with link density below a threshold (as in VIDE)
  Full       // The full ZOBOV hierarchy (no threshold)
};
</pre>
</div>
</div>
</p>
<p>
We go through the links between the zones from the lowest density up (union-find). When two voids join at a link the shallower one (the one whose deepest zone has the larger density) is finished: it is the set of zones it had just before, its leak density is the link and its parent is the deeper one, which grows to include it. This is the same as growing a void from each zone by raising the "water level" until it reaches a zone deeper than its own (jozov in ZOBOV). With merging the catalogue has all the voids in the hierarchy and parent is the id of the void a void is part of (-1 for the top-level voids). Ties are broken by the position of the minimum so the result does not depend on how the zones are numbered.
</p>

<header class="entry-header text-center">
  <h1 class="entry-title" id="Spherical">Spherical underdensities</h1>
</header>
<hr/>

<p>
Candidate centers are the cells of a smoothed density grid with density below candidate_density_cut. Around each candidate we grow a sphere until the density inside reaches $(1 + \Delta)$ times the mean: the radius is the distance to the first particle that would make the density inside larger than this (computed with the exact particle counts, using the masses if the particles have get_mass). Optionally we move the center with a random walk to get a larger sphere. Then we go through the spheres from the largest to the smallest and keep the ones that do not overlap with the ones we already kept (and that are larger than the minimum radius). The catalogue is sorted by radius (largest first) and density_ratio is not defined (set to $0$).
<div style="text-align: center">
  <div style="width:600px;display: inline-block;text-align: left">
    <pre class="prettyprint">
struct SphericalOptions {
  double delta_threshold{-0.8};               // The integrated density contrast Delta(&lt;R) inside a void
  double candidate_density_cut{0.5};          // Candidate centers: grid cells with density below this
  VoxelOptions grid{};                        // The grid for the candidates (as for the voxel finder)
  double max_radius_in_mean_separation{8.0};  // Largest radius we look for
  double min_radius_in_mean_separation{2.0};  // Smallest radius we keep
  int recentering_steps{0};                   // Random walk steps for moving the center
  double max_overlap{0.0};                    // Two voids overlap if their distance is
                                              // less than (1 - max_overlap) (R1 + R2)
};

template &lt;class T&gt;
std::vector&lt;Void&gt; SphericalVoidFinder(T * p, size_t NumPart, const SphericalOptions &amp; options = SphericalOptions());
</pre>
</div>
</div>
</p>
