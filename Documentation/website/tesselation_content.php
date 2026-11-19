<hr/>
<header class="entry-header text-center">
  <h1 class="entry-title">Table of contents</h1>
</header>
<hr/>

<p class="page-toc">
<a href="index.php">Back to the main page</a><br>
<a href="documentation.php">Back to the documentation</a><br>
<a href="#Overview">1. Overview: Tessellation</a><br>
<a href="#Native">2. FML's own triangulation (no CGAL)</a><br>
<a href="#NativeMPI">3. The periodic Voronoi tessellation with MPI</a><br>
<a href="#Moving">4. Moving points</a><br>
<a href="#NativeValidation">5. Validation and speed</a><br>
<a href="#Create">6. CGAL: creating a triangulation</a><br>
<a href="#Volume">7. CGAL: Voronoi volume of the particles</a><br>
<a href="#Vertex">8. CGAL: adding a vertex base</a><br>
<a href="#Nbor">9. CGAL: neighbors and checking the result</a><br>
<a href="#More">10. What we use it for</a><br>
</p>

<header class="entry-header text-center">
  <h1 class="entry-title" id="Overview">Overview: Delaunay triangulation and Voronoi tessellation</h1>
</header>
<hr/>  

<p>
Delaunay triangulation is loosely speaking a way of linking together particles to form triangles (tetrahedra) that are 'as equilateral as possible'. The dual graph of a Delaunay triangulation is called the Voronoi diagram and provides a great way of assigning volumes to particles: a Voronoi cell contains the region in space that is closer to its center point than any other point. This has many applications like for example computing volume (instead of mass) averaged fields (<a href="https://www.astro.rug.nl/~voronoi/DTFE/dtfe.html" style="display: inline">DTFE</a> / VTFE), locating halos and voids (<a href="https://arxiv.org/abs/0712.3049" style="display: inline">ZOBOV</a>) and many others.
</p>

<p>
FML has two implementations. Its <b>own</b> Delaunay triangulation and periodic Voronoi tessellation (2D and 3D, header-only, no dependencies; <tt>FML/Experimental/Delaunay</tt> while it is developed) is exact, MPI parallel for any distribution of the particles, gives results that do not depend on the number of tasks, can follow moving points (the building block of a future moving-mesh hydro solver) and is 1.5-4 times faster than CGAL. ZOBOV and the ORIGAMI halos run on it without CGAL; the DTFE is being ported now, after which CGAL will be optional everywhere. The <b>CGAL</b>-based tessellation (sections 6-9) is the older one.
</p>

<p>
For the CGAL tessellation we use the <a href="https://www.cgal.org/" style="display: inline">CGAL</a> library to produce a periodic triangulation in parallel (compile with USE_CGAL and choose the dimension, 2 or 3, with the define CGAL_NDIM; the fiducial is 3). CGAL (for good reasons) does not have an MPI implementation so what we do is to produce a periodic triangulation on each task with the local particles plus a buffer of particles from the neighboring tasks. The rest of the box is filled with a sparse set of guard points (without these the triangulation of a slab is never 1-sheeted which makes CGAL much slower). We then check which cells (tetrahedra) can be trusted: a cell is guaranteed to be exactly the same as in a triangulation of the full box if none of its vertices are guard points and its circumsphere lies inside the region covered by the domain plus the buffer. If some cells around the local particles can not be trusted we print a warning telling you to increase the buffer. Everything lives in the namespace FML::TRIANGULATION (Triangulation/PeriodicDelaunay.h). A good advice is to use it on as few tasks as you possibly can.
</p>

<figure class="science-figure">
<img src="images/delaunay-voronoi.svg" alt="A Delaunay triangulation on the left and its dual Voronoi cells on the right">
<figcaption>Conceptual 2D example: Delaunay edges connect neighboring sites; the dual Voronoi cells partition space by nearest site.</figcaption>
</figure>

<p>
<b>Any box domain.</b> <tt>MPIPeriodicDelaunay(domain)</tt> tessellates particles on a Slab or Pencil domain (e.g. a run on <a href="pencilgrid.php" style="display: inline">pencils</a>): each task gets the particles within the buffer of its box from up to 8 neighbours.
</p>

<header class="entry-header text-center">
  <h1 class="entry-title" id="Native">FML's own triangulation (no CGAL)</h1>
</header>
<hr/>

<p>
Everything rests on two exact predicates: <tt>orient</tt> (the sign of the determinant of the simplex edges) and <tt>insphere</tt> (is a point inside the circumsphere of a simplex). They are evaluated in floating point with <a href="https://doi.org/10.1007/PL00009321" style="display: inline">Shewchuk's</a> error bound, and when the sign is uncertain they are recomputed exactly with floating-point expansions, so the result is exact for any input. <tt>Delaunay&lt;D&gt;</tt> inserts the points in Peano-Hilbert order (Bowyer-Watson: walk to the point, find the cavity of simplices whose circumspheres contain it, replace it). Cospherical points, such as a lattice, are resolved with the symbolic perturbation of <a href="https://doi.org/10.1016/j.comgeo.2010.09.010" style="display: inline">Devillers &amp; Teillaud</a>, as in CGAL, so every input has one well-defined Delaunay triangulation.
</p>

<header class="entry-header text-center">
  <h1 class="entry-title" id="NativeMPI">The periodic Voronoi tessellation with MPI</h1>
</header>
<hr/>

<p>
<tt>PeriodicVoronoi&lt;D&gt;</tt> is collective: each task gives its points (slabs, pencils, Peano-Hilbert ranges, random, even empty tasks) and gets the Voronoi cells of its points (volume, centroid) and every face touching them (area, centroid, normal, the neighbour's global id, task and index, the periodic shift). Each task triangulates its points together with <i>ghosts</i> (other tasks' points and periodic images): first a buffer around the bounding box of its points, then rounds of <i>certification</i>, where the circumsphere of every simplex with a local point is searched on every task whose region it meets. Points found inside are inserted, and a simplex that was searched and survives is final. No buffer size to tune and no assumption on the domain. Coordinates are quantized so periodic images are exact copies, and everything is computed in an order fixed by global keys, so the cells and faces are <b>bitwise identical for any number of tasks</b>, and a face between two tasks is identical on both.
<div style="text-align: center">
  <div style="width:700px;display: inline-block;text-align: left">
    <pre class="prettyprint">
#include &lt;FML/Experimental/Delaunay/PeriodicVoronoi.h&gt;
using namespace FML::DELAUNAY;

std::vector&lt;std::array&lt;double, 3&gt;&gt; points = ...; // this task's points in [0, boxsize)^3
PeriodicVoronoi&lt;3&gt; voronoi(points, boxsize);       // collective
for (size_t i = 0; i &lt; voronoi.ncells(); i++) {
    const auto &amp; cell = voronoi.cells()[i];        // cell.id, cell.volume, cell.centroid
    auto [begin, end] = voronoi.cell_faces(i);     // the faces of cell i
    for (auto it = begin; it != end; ++it) {
        const auto &amp; face = voronoi.faces()[*it &gt;= 0 ? *it : -*it - 1];
        // face.area, face.centroid, face.normal, face.shift, face.task_i, face.task_j, ...
    }
}
</pre>
</div>
</div>
</p>

<p>
For the rest of the library <tt>PeriodicVoronoiTessellation&lt;NDIM&gt;</tt> (Triangulation/VoronoiTessellation.h) gives the same interface as the CGAL class: <tt>create(part, NumPart)</tt>, the Voronoi volumes, <tt>for_each_neighbor(i, f)</tt> over the Delaunay neighbours, and the values of neighbours on other tasks fetched from their owners. The watershed (<tt>WatershedOnGraph</tt>, Triangulation/Watershed.h) runs on the neighbour graph of either tessellation.
</p>

<header class="entry-header text-center">
  <h1 class="entry-title" id="Moving">Moving points</h1>
</header>
<hr/>

<p>
<tt>voronoi.update(new_points)</tt> gives exactly (bitwise) what a new tessellation of the moved points would give, but reuses the ghosts and the triangulation: vertices are moved in place where their simplices stay valid, otherwise removed (the hole is filled with the Delaunay triangulation of its link) and inserted again, or the whole set is retriangulated, whichever is estimated to be cheaper. Points may cross the box edges and other tasks' regions. With 1% of 200000 points moving the triangulation takes 0.39 s instead of 1.71 s for a new build; when all points move by more than about 1% of the interparticle spacing an update costs about as much as a build.
</p>

<header class="entry-header text-center">
  <h1 class="entry-title" id="NativeValidation">Validation and speed</h1>
</header>
<hr/>

<p>
The predicates equal exact 128-bit integer determinants on 400000 small-integer cases (thousands exactly degenerate). Triangulations of random, clustered, lattice and duplicated points are valid and Delaunay (checked by brute force); the periodic Voronoi volumes sum to the box to $10^{-15}$, cells close to $10^{-13}$, lattices give cubes. With MPI, slab, pencil, Peano-Hilbert and random distributions on 1-4 tasks give cells and faces bitwise identical to everything on one task, also after updates with points crossing task and box boundaries. The Delaunay edges are identical to those of CGAL 5.6's periodic triangulations for $10^5$-$10^6$ points in 2D and 3D, and ZOBOV voids and ORIGAMI halos are identical to the CGAL ones. Timings on one core:
</p>
<table class="table table-striped table-condensed" style="max-width:700px; margin:auto">
<tr><th>Case</th><th>FML triangulation</th><th>FML geometry</th><th>CGAL periodic triangulation</th></tr>
<tr><td>2D, $10^5$ random</td><td>0.14 s</td><td>0.17 s</td><td>0.63 s</td></tr>
<tr><td>3D, $10^5$ random</td><td>0.76 s</td><td>0.60 s</td><td>1.70 s</td></tr>
<tr><td>3D, $10^6$ random</td><td>7.3 s</td><td>7.5 s</td><td>18.0 s</td></tr>
</table>
<p>
With MPI (3D, $4\times10^5$ points in slabs) a build takes 5.7 s on one task and 3.2 s on two. See <tt>Experimental/Delaunay/README.md</tt> for the details.
</p>

<header class="entry-header text-center">
  <h1 class="entry-title" id="Create">CGAL: creating a triangulation</h1>
</header>
<hr/> 

<p>
The class is templated on the particle class and on the vertex base (what data do you want to store at the vertices (particles positions) ) and the create method takes in a function that assigns this data to the vertices:
<div style="text-align: center">
  <div style="width:600px;display: inline-block;text-align: left">
    <pre class="prettyprint">
template&lt;class T, class VD = VD_FIDUCIAL&gt;
  class MPIPeriodicDelaunay;

// This class has the create function:
void create(
  T *part,
  size_t NumPart,
  double buffer_fraction = 0.25,
  double random_fraction = 0.3,
  std::function&lt;void(VD *, T *p)&gt; assignment_function = fiducial_assignment_function&lt;VD, T&gt;);

// Create a simple triangulation with the fiducial settings
FML::TRIANGULATION::MPIPeriodicDelaunay&lt;Particle&gt; PdT;
PdT.create(part.get_particles_ptr(), part.get_npart());
</pre>
</div>
</div>
</p>
<p>
Above buffer_fraction is the fraction of the neighboring domain(s) that we communicate. The random_fraction is how many guard points we add outside the domain plus buffer compared to the number of particles (yes, doing more work is actually faster!). Something like $10-30\%$ seems to be optimal in the tests I have done. With one task we don't need any of this and both are set to zero. The points are inserted in a random order (with a fixed seed so the result is reproducible) as this is much faster than inserting them in the order they come in. If you don't want any data assigned to the vertices then don't provide the vertex base or the assignment function and we will use the fiducial one.
</p>

<p>
When creating the triangulation we store a handle to the vertices which you can get by calling the class methods:
<div style="text-align: center">
  <div style="width:600px;display: inline-block;text-align: left">
    <pre class="prettyprint">
const std::vector&lt;Vertex_handle&gt; &amp; get_vertex_handles_regular() const;
const std::vector&lt;Vertex_handle&gt; &amp; get_vertex_handles_boundary() const;

// The boundary particles we got from the neighbor tasks
std::vector&lt;T&gt; &amp; get_boundary_particles();

// The underlying CGAL triangulation
PeriodicDelaunay &amp; get_delaunay_triangulation();

// Example
auto &amp; vs = PdT.get_vertex_handles_regular();

// Get the vertex of particle i
int i = 0;
auto v = vs[i];

// Get the point of particle i
auto point = v-&gt;point();

// If we created it with a vertex base, say struct VertexBase{ double quantity; } 
// fetch this for particle i
auto quantity = v-&gt;info().quantity;
</pre>
</div>
</div>
</p>

<header class="entry-header text-center">
  <h1 class="entry-title" id="Volume">CGAL: Voronoi volume of the particles</h1>
</header>
<hr/> 

<p>
Once we have created the triangulation from a set of particles we can extract the volume as follows:
<div style="text-align: center">
  <div style="width:600px;display: inline-block;text-align: left">
    <pre class="prettyprint">
std::vector&lt;double&gt; volumes;
PdT.VoronoiVolume(volumes);
</pre>                        
</div>                        
</div>                        
</p>                          
<p>
This method fills the vector volumes with the volumes (areas in 2D) of the particles used to create the triangulation (stored in the same order as the particles). There is also the method voronoi_volume(v) that gives the volume of the Voronoi cell of a single vertex. The volumes are in units of the boxsize, so they sum to $1$ over all particles on all tasks.
</p>

<header class="entry-header text-center">
  <h1 class="entry-title" id="Vertex">CGAL: adding a vertex base</h1>
</header>
<hr/> 

<p>
The CGAL triangulation is just a collection of points and links between them and has no info about the particles that they came from. We can deal with this by providing a vertex base and a vertex base assignment function to add data to the vertices. This can be anything you want and is really useful for computing stuff on the triangulation (though this does cost a lot of memory so be aware of this!). Here is an example of adding a ptr to the particle at the vertices:
<div style="text-align: center">
  <div style="width:600px;display: inline-block;text-align: left">
    <pre class="prettyprint">
// The data we want at the vertices (we store a pointer to the particle)
struct VertexData{ Particle *part_ptr{nullptr}; };


// Set up the class with this vertex base
FML::TRIANGULATION::MPIPeriodicDelaunay&lt;Particle, VertexData&gt; PdT;

// Set up the vertex assignment function (NB: the guard points don't correspond to a real particle and get a nullptr)
std::function&lt;void(VertexData *, Particle *)&gt; vertex_assignment_function = [](VertexData *v, Particle *p){
   v-&gt;part_ptr = p;
};

// Create the triangulation
PdT.create(part, NumPart, buffer_fraction, random_fraction, vertex_assignment_function);
</pre>
</div>
</div>
</p>

<header class="entry-header text-center">
  <h1 class="entry-title" id="Nbor">CGAL: neighbors and checking the result</h1>
</header>
<hr/>

<p>
To loop over the neighbors of a vertex (this also takes care of the virtual copies CGAL uses for triangulations that are not 1-sheeted) and to check how many cells around the local particles can not be trusted:
<div style="text-align: center">
  <div style="width:600px;display: inline-block;text-align: left">
    <pre class="prettyprint">
// Call f(w) for all vertices w adjacent to v. work is just a work array
// (pass the same one for every call to avoid allocations)
std::vector&lt;Vertex_handle&gt; work;
PdT.for_each_adjacent_vertex(v, work, [&amp;](Vertex_handle w){
  // ... do something with the neighbor
});

// The number of cells that have one of the local particles as a vertex
// and that might not be the same as in a triangulation of the full box
// If this is zero then all Voronoi cells of the local particles are exact
long long int nbad = PdT.num_bad_cells();

// Free up all the memory
PdT.free();
</pre>
</div>
</div>
</p>

<header class="entry-header text-center">
  <h1 class="entry-title" id="More">What we use it for</h1>
</header>
<hr/>

<p>
The triangulation is the basis of several other methods in the library: the <a href="watershed.php">watershed</a> algorithm (the ZOBOV void finder), <a href="dtfe.php">DTFE</a> for interpolating fields from particles to a grid, the <a href="voidfinders.php">void finders</a> and <a href="cosmicweb.php">ORIGAMI</a> halo finding. ZOBOV and the ORIGAMI halos can use either tessellation (FML's own is the default in builds without CGAL); the DTFE still uses CGAL and is being ported. FML's own tessellation is also the first step towards an AREPO-style moving-mesh hydro solver.
</p>                          
