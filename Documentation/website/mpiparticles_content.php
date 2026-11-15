<hr/>
<header class="entry-header text-center">
  <h1 class="entry-title">Table of contents</h1>
</header>
<hr/>

<p class="page-toc">
<a href="index.php">Back to the main page</a><br>
<a href="documentation.php">Back to the documentation</a><br>
<a href="#Overview">1. Overview: MPIParticles</a><br>
<a href="#Req">2. Requirements on particle class</a><br>
<a href="#Create">3. Creating MPIParticles</a><br>
<a href="#Save">4. Saving and loading of particles</a><br>
<a href="#Access">5. Accessing the particles</a><br>
<a href="#Migration">6. Particle migration</a><br>
</p>

<header class="entry-header text-center">
  <h1 class="entry-title" id="Overview">Overview: MPIParticles</h1>
</header>
<hr/>  

<p>
MPIParticles is a general class for keeping a set of particles that are shared among tasks and to communicate these particles if they move outside of the domain given to each task. The particle class it is templated on only needs a position (and if it has dynamically allocated memory, methods for appending and assigning the particle data to/from the communication buffers we use, see below).
</p>
<p align="center">
Figure 1: Here we show the full domain. The red region is the domain the current task is responsible for. Particles moving outside the boundaries will be communicated to the respective task once we call communicate_particles().
<img src="images/partbox.jpg">
</p>

<p>
<b>Domains.</b> How the particles are split over the tasks is an <tt>FML::Domain</tt>: slabs in $x$ (the default), pencils in $x$-$y$ or a Peano&ndash;Hilbert partition. Give one to the constructor (<tt>MPIParticles&lt;T&gt; part(domain)</tt>) or use the global <tt>FML::domain</tt>. <tt>create</tt>, <tt>communicate_particles</tt> and <tt>set_domain(new_domain)</tt> (move the particles to their new owners) work with every domain. <tt>create_particle_grid</tt> makes the lattice on Slab and Pencil domains and throws <tt>std::invalid_argument</tt> for others (make it on slabs and call <tt>set_domain</tt>). The grids of FML need the particles on their domain: slabs for <tt>FFTWGrid</tt>, pencils for <a href="pencilgrid.php" style="display: inline">$\texttt{PencilGrid}$</a>.
</p>

<header class="entry-header text-center">
  <h1 class="entry-title" id="Req">Requirements on particle class</h1>
</header>
<hr/>  

<p>
MPIParticles can be used with any particle class (for more info about particle classes within the library see <a href="global.php" style="display: inline">Global</a>). The minimal requirement is that it has the two methods
<div style="text-align: center">
  <div style="width:600px;display: inline-block;text-align: left">
    <pre class="prettyprint">
// Here FloatingPointType can be any floating point type (float, double, long double)
FloatingPointType *get_pos();

// The dimension of the position (must be constexpr)
constexpr int get_ndim() const;
</pre>
</div>
</div>
</p>

<p>
If you have dynamically allocated memory inside the class then you must also provide the methods
<div style="text-align: center">
  <div style="width:600px;display: inline-block;text-align: left">
    <pre class="prettyprint">
// How many bytes to communicate
int get_particle_byte_size();

// Append the particle data to a buffer
void append_to_buffer(char *data);

// Assign the particle data from a buffer
void assign_from_buffer(char *data);
</pre>
</div>
</div>
</p>

<p>
You can of course also provide these methods if you want even if it's not required (which might be useful in case your layout is such that we have a lot of padding between elements). Here is an example of how to make these methods yourself (note that in this example we don't need to provide these methods as we have no dynamically allocated memory inside the class so the fiducial methods we use if these methods are not provided will work):
<div style="text-align: center">
  <div style="width:600px;display: inline-block;text-align: left">
    <pre class="prettyprint">
template&lt;int NDIM&gt;
struct Particle{
  double x[NDIM];
  double v[NDIM];

  // We must have a way to get the position on the particle
  double* get_pos(){ 
    return x; 
  }
  
  // The dimension we are in
  constexpr int get_ndim() const {
    return NDIM;   
  }
  
  // The velocity
  double* get_vel(){ 
    return v; 
  }
  
  // We need the size of the particle 
  int get_particle_byte_size(){ 
    return sizeof(x) + sizeof(v);
  }

  // Append the particle to a buffer
  void append_to_buffer(char *data){ 
    int bytes = sizeof(x);
    std::memcpy(data, x, bytes); 
    data += bytes;
  
    bytes = sizeof(v);
    std::memcpy(data, v, bytes); 
   }

  // Assign the particle from a buffer
  void assign_from_buffer(char *data){ 
    int bytes = sizeof(x);
    std::memcpy(x, data, bytes); 
    data += bytes;

    bytes = sizeof(v);
    std::memcpy(v, data, bytes); 
  }
};
</pre>
</div>
</div>
</p>

<header class="entry-header text-center">
  <h1 class="entry-title" id="Create">Creating MPIParticles</h1>
</header>
<hr/> 

<p>
We offer methods to create MPIParticles from an existing set of particles or to make a regular grid of particles. For the former we can either copy over the particles, provide a selection function that tells us what particles to keep (create(std::vector&lt;T&gt; &amp; part, size_t nallocate, std::function&lt;bool(T &amp;)&gt; selection_function)) or just simply move the storage of existing particles straight into the class (move_from). See the doxygen documentation, the examples or the source for more info.
</p> 

<p>
Here we set up the class and make a regular particle distribution out of it:
<div style="text-align: center">
  <div style="width:600px;display: inline-block;text-align: left">
    <pre class="prettyprint">
FML::PARTICLE::MPIParticles&lt; Particle&lt;3&gt; &gt; part;

// Let's fill it with some particles. Make a regular distribution
// (the particles get IDs (ix * Npart_1D + iy) * Npart_1D + iz if they have a set_id method)
const int Npart_1D = 6;           // Make Npart_1D^NDIM particles
const double buffer_factor = 2.0; // How many more particles we allocate for
part.create_particle_grid(
   Npart_1D, 
   buffer_factor, 
   FML::xmin_domain, 
   FML::xmax_domain);

// Move some particles out of bounds
for(int i = 0; i &lt; 5; i++){
  part[i].get_pos()[0] = (std::rand() % 1000)/1000.0;
}

// Communicate the particles so they end up in the right task
part.communicate_particles();
</pre>
</div>
</div>
</p>

<p>
Creating from an existing set of particles:
<div style="text-align: center">
  <div style="width:600px;display: inline-block;text-align: left">
    <pre class="prettyprint">
// If we have a set of particles
Particle *p;
int NumPart;

// ... make the particles

//... then we can make MPIParticles out of them by running create
FML::PARTICLE::MPIParticles&lt; Particle &gt; part;
const bool all_tasks_have_the_same_particles = false; // If all tasks reads the same file for example
auto nallocate_local = 10000;                         // Maximum number of particles we think we need on each task (exits if we ever go over)
part.create(
    p,
    NumPart, 
    nallocate_local, 
    FML::xmin_domain, 
    FML::xmax_domain, 
    all_tasks_have_the_same_particles);
</pre>
</div>
</div>
</p>

<header class="entry-header text-center">
  <h1 class="entry-title" id="Save">Saving and loading of particles</h1>
</header>
<hr/>  
<p>
<div style="text-align: center">
  <div style="width:600px;display: inline-block;text-align: left">
    <pre class="prettyprint">
// Save the particles on each task to output.X with X = ThisTask
part.dump_to_file("output");

// Read particles from file output.X with X = ThisTask (with the same number of tasks)
part.load_from_file("output");
</pre>
</div>
</div>
</p>

<header class="entry-header text-center">
  <h1 class="entry-title" id="Access">Accessing the particles</h1>
</header>
<hr/>  
<p>
Different ways of accessing the particles:
<div style="text-align: center">
  <div style="width:600px;display: inline-block;text-align: left">
    <pre class="prettyprint">

// Iterate over all the active particles (not buffer)
for(auto pit = part.begin(); pit != part.end(); ++pit){
  // Reference to the current particle
  auto &amp; p = *pit;
  auto pos = p.get_pos();
  ...
}

// Iterate over all the active particles (not buffer)
for(auto &amp; p : part){
  auto pos = p.get_pos();
  ...
}

// Explicit for-loop over the active particles
for(size_t i = 0; i &lt; part.get_npart(); i++){
  auto &amp; p = part[i]; 
  auto pos = p.get_pos();
  ...
}

// Get a reference to the raw storage (FML::Vector&lt;T&gt; &amp;)
// Note that this contains all particles. 
// Active particles at the front and buffer particles at the back
auto &amp; part_vector = part.get_particles();

// Number of particles on this task and in total
size_t npart_local = part.get_npart();
size_t npart_total = part.get_npart_total();

// Get a pointer to the first element in the raw storage
// The i'th particle is accessed through p[i]
auto part_ptr = part.get_particles_ptr();

</pre>
</div>
</div>
</p>

<header class="entry-header text-center">
  <h1 class="entry-title" id="Migration">Particle migration</h1>
</header>
<hr/>

<p>
<tt>communicate_particles()</tt> moves every particle to the task owning its $x$-coordinate (any contiguous partition of $[0,1)$ into slabs, also non-uniform or empty). Owners are found by binary search over the slab boundaries, and all tasks agree on a route: no traffic at all, neighbours only (count and payload exchanged with at most two tasks, the usual case after a time step) or a general all-to-all exchange. Messages are split to respect a maximum message size, and particles with dynamically sized data are serialized through the reflection interface. <tt>get_last_migration_route()</tt> reports the route taken.
</p>
