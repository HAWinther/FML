<hr/>
<header class="entry-header text-center">
  <h1 class="entry-title">Table of contents</h1>
</header>
<hr/>

<p class="page-toc">
<a href="index.php">Back to the main page</a><br>
<a href="documentation.php">Back to the documentation</a><br>
<a href="#Overview">1. Overview: Watershed</a><br>
<a href="#Methods">2. Available methods</a><br>
</p>

<header class="entry-header text-center">
  <h1 class="entry-title" id="Overview">Overview: Watershed</h1>
</header>
<hr/>  

<p>
Create a <a href="tesselation.php" style="display: inline">Delaunay triangulation</a> and assign a quantity (for example the particle density) to each vertex. From this locate the local minima, i.e. vertices where all of the neighbors have a larger value of the quantity. Every particle is then linked to the neighbor with the smallest quantity (steepest descent) and a watershed basin is the collection of particles that end up at the same local minimum (ties are broken using the positions so the result does not depend on the number of tasks). We can also go further and continue filling "water" into the basins until they overflow into a neighboring basin and thereby get larger and larger basins: for this we can return the links between the basins (the lowest value of the quantity at which two basins connect) and the zone merging is done in the <a href="voidfinders.php" style="display: inline">ZOBOV void finder</a>. This can be used as a void and halo finder (which also gives us sub-voids and sub-halos). Everything is in the namespace FML::TRIANGULATION (Triangulation/PeriodicDelaunay.h and Triangulation/WatershedBinning.h).
</p>
<figure class="science-figure">
<img src="images/watershed-basins.svg" alt="Schematic density landscape with watershed minima and ridges separating catchment basins">
<figcaption>Schematic watershed segmentation: particles follow neighboring links toward minima, while ridges separate the resulting catchment basins.</figcaption>
</figure>

<header class="entry-header text-center">
  <h1 class="entry-title" id="Methods">Available methods</h1>
</header>
<hr/>  

<p>
<div style="text-align: center">
  <div style="width:600px;display: inline-block;text-align: left">
    <pre class="prettyprint">
// A general watershed algorithm
// The result (watershed_groups and links) is only on task 0
template&lt;class T, class U&gt;
  void WatershedGeneral(
      MPIPeriodicDelaunay&lt;T, VertexDataWatershed&gt; &amp; PdT,
      T *p,
      size_t NumPart,
      std::vector&lt;double&gt; &amp; quantity,
      std::vector&lt;U&gt; &amp; watershed_groups,
      std::vector&lt;WatershedLink&gt; * links = nullptr);

// Same as above but where quantity is the density (mass of particle over Voronoi volume)
// and where we create the triangulation within the method. The particle must have
// set_volume / get_volume. With do_density_maximum we find maxima instead (halos)
template&lt;class T, class U&gt;
  void WatershedDensity(
      T *p,
      size_t NumPart,
      std::vector&lt;U&gt; &amp; watershed_groups,
      double buffer_fraction = 0.30,
      double random_fraction = 0.5,
      bool do_density_maximum = false,
      std::vector&lt;WatershedLink&gt; * links = nullptr);

// A link between two basins (index into watershed_groups) and the
// lowest value of the quantity at which they connect
struct WatershedLink {
  long long int a;
  long long int b;
  double quantity;
};
</pre>
</div>
</div>
</p>
<p>
The class U determines what we compute for each basin. The fiducial class is WatershedBasin&lt;T, NDIM&gt; which contains:
<div style="text-align: center">
  <div style="width:600px;display: inline-block;text-align: left">
    <pre class="prettyprint">
template &lt;class T, int NDIM&gt;
struct WatershedBasin {
  double pos_min[NDIM];        // Position of the minimum
  double pos_circ[NDIM];       // Circumcenter of the minimum and its NDIM lowest-quantity
                               // mutually adjacent neighbors (as in Revolver)
  double quantity_leak;        // The lowest quantity at which the basin links to a neighbor
  double pos_barycenter[NDIM]; // Volume weighted barycenter
  double volume;               // Total volume
  double volume_min;           // Volume of the minimum particle
  double density_min;          // Density of the minimum particle
  double mass;                 // Total mass
  double mean_quantity;        // Mean of the quantity
  int ningroup;                // Number of particles in the basin
};
</pre>
</div>
</div>
</p>
<p>
If your own class has the methods set_pos_circ and set_leak_quantity then these are filled in as well. For ready made void finders built on this (ZOBOV with zone merging, a voxel based watershed and spherical underdensities) see the <a href="voidfinders.php" style="display: inline">void finders</a> page.
</p>
