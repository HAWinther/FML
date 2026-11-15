<hr/>
<header class="entry-header text-center">
  <h1 class="entry-title">Table of contents</h1>
</header>
<hr/>

<p class="page-toc">
<a href="index.php">Back to the main page</a><br>
<a href="documentation.php">Back to the documentation</a><br>
<a href="#Overview">1. Overview: FML</a><br>
<a href="#Part">2. Particle classes</a><br>
</p>

<header class="entry-header text-center">
  <h1 class="entry-title" id="Overview">Overview: FML</h1>
</header>
<hr/>  

<p>
The domain we are working with in the library is a cubic box $[0,1)^{\rm NDIM}$ of any dimension(s). It is generally assumed periodic for which the point $1$ wraps around to $0$, but several classes and algorithms lets you specify if this is the case or not. There are no units assumed in the library. Grids have the same number of gridcells in each dimension. The default parallelization over MPI is a slab-based domain decomposition along the $x$-axis; the domain is an object, <tt>FML::domain</tt> (<tt>Global/Domain.h</tt>), that can also be <a href="pencilgrid.php" style="display: inline">pencils</a> in $x$&ndash;$y$ or contiguous pieces of a Peano&ndash;Hilbert curve.
</p>

<p>
The library is divided into several namespaces. Global variables and functions are placed in the top namespace FML and are defined in Global.
</p>

<p>
We have just a few global variables (that we could just as well have computed where they are needed). The most important ones are FML::ThisTask and FML::NTasks. This is the ID of a given task and the total number of tasks. Next we have FML::xmin_domain and FML::xmax_domain. This is the $x$-domain of the grid the given task is responsible for. We also have FML::NThreads which is the number of threads we have available (this is just OMP_NUM_THREADS). All the MPI communication in the library goes through FML::communicator (MPI_COMM_WORLD by default). We also have some general type definitions and a few useful functions.
</p>

<p>
MPI, FFTW and the variables above are initialized (and finalized) automatically. If you don't want this, compile with the define NO_AUTO_FML_SETUP: then you have to initialize MPI yourself and call FML::init_fml() (and FML::GRID::init_fftw() if you use FFTW) afterwards. This is also how you run the library on a sub-communicator: initialize MPI, set FML::communicator to your communicator and then call FML::init_fml(), see Global.h.
</p>

<p>
Error handling. This is hard to do properly so we keep it simple: if we detect an error we typically call assert_mpi(expression, message) which gives some info about where the error happened and what went wrong and then aborts all tasks (MPI_Abort, to avoid deadlocks). Several of the individual classes can throw errors instead which can be useful in some cases, but see the individual classes to change this behavior.
</p>

<p>
Some simple methods to find max, min and sum of a single value (or an array) over all the tasks. These operations are done in-place, i.e. they overwrite the value we pass in a pointer to. The last method gathers a value from all tasks (and returns them on all tasks).
<div style="text-align: center">
  <div style="width:600px;display: inline-block;text-align: left">
    <pre class="prettyprint">
  template&lt;class T&gt;
    void MaxOverTasks(T * value);
  template&lt;class T&gt;
    void MinOverTasks(T * value);
  template&lt;class T&gt;
    void SumOverTasks(T * value);
  template&lt;class T&gt;
    void SumArrayOverTasks(T * values, int n);
  template&lt;class T&gt;
    std::vector&lt;T&gt; GatherFromTasks(T * value);
</pre>
</div>
</div>
</p>

<p>
In Global we define a container FML::Vector&lt;T&gt; which is just normal std::vector but with a custom allocator to keep track of the memory use (when compiled with USE_MEMORYLOG = true, i.e. the define MEMORY_LOGGING). We currently only use Vector for FFTWGrid, MPIGrid and MPIParticles so this is all it tracks. The define MIN_BYTES_TO_LOG=XXX sets how large (in bytes) an allocation has to be to be logged and MAX_ALLOCATIONS_IN_MEMORY=YYY sets the max number of elements we keep track of before giving up and just stop logging.
</p>

<p>
To add a label to an already allocated object you can call:
<div style="text-align: center">
  <div style="width:600px;display: inline-block;text-align: left">
    <pre class="prettyprint">
FML::Vector&lt; double &gt; data(1000);
FML::MemoryLog::get()-&gt;add_label(data.data(), data.size(), "My vector data");
</pre>
</div>
</div>
</p>
<p>
And the classes that uses this has special methods for this:
<div style="text-align: center">
  <div style="width:600px;display: inline-block;text-align: left">
    <pre class="prettyprint">
// FFTWGrid
FML::GRID::FFTWGrid&lt;NDIM&gt; grid(Nmesh);
grid.add_memory_label("label");

// MPIParticles
FML::PARTICLE::MPIParticles&lt;Particle&gt; part;
// ...create the particles
part.add_memory_label("label");
</pre>
</div>
</div>
</p>

<p>
To see what we have in memory at the current time:
<div style="text-align: center">
  <div style="width:600px;display: inline-block;text-align: left">
    <pre class="prettyprint">
FML::MemoryLog::get()-&gt;print();  
</pre>
</div>
</div>
</p>
<p>
Example output:
<div style="text-align: center">
  <div style="width:600px;display: inline-block;text-align: left">
    <pre class="prettyprint">
#=====================================================

We are only tracking allocation of standard container
and only allocations larger than 0 bytes

Memory in use:               8.8 MB
Peak memory use:             8.8 MB

We have the following things allocated on task 0: 
Address: 0x106453000 Size: 0.8 (MB) Label: [This is GridA]
Address: 0x106517000 Size: 8   (MB) Label: [This is GridB]

Total memory as function of time:
 Time (sec):             0 Memory:             0 (MB)
 Time (sec):       2.3e-05 Memory:           0.8 (MB)
 Time (sec):      0.001174 Memory:           8.8 (MB)

#=====================================================
</pre>
</div>
</div>
</p>

<header class="entry-header text-center">
  <h1 class="entry-title" id="Part">Particle Classes</h1>
</header>
<hr/>  

<p>
A lot of the algorithms inside the library are templated on particle classes. These will accept any kind of particle as long as it has the two methods
<div style="text-align: center">
  <div style="width:600px;display: inline-block;text-align: left">
    <pre class="prettyprint">
class MyParticle{
  ...

public:
  // Here PositionType can be any floating-point type (float, double, long double)
  PositionType *get_pos();

  // The dimension of the position (must be constexpr)
  constexpr int get_ndim() const;
};
</pre>
</div>
</div>
</p>

<p>
The get_pos method returns a pointer to the position (assumes NDIM consecutive values in memory) and get_ndim gives the dimension of the position vector. If you have dynamically allocated memory inside the class (i.e. sizeof(MyParticle) does not give the actual size) then you must provide methods for how to append the particle to a buffer and assign the particle from a buffer that we use to do the communication between tasks:
<div style="text-align: center">
  <div style="width:600px;display: inline-block;text-align: left">
    <pre class="prettyprint">
class MyParticle{
  ...

public: 
  ...

  // How many bytes to communicate
  int get_particle_byte_size();
  
  // Append the particle data to a buffer
  void append_to_buffer(char *data);
  
  // Assign the particle data from a buffer
  void assign_from_buffer(char *data);
};
</pre>
</div>
</div>
</p>

<p>
For example if we have a std::vector&lt;double&gt; stuff; inside the class then the method would look something like this:
<div style="text-align: center">
  <div style="width:600px;display: inline-block;text-align: left">
    <pre class="prettyprint">
std::vector&lt;double&gt; stuff; 

// We need to communicate both the data and its size (plus any other things in the class)
int get_particle_byte_size(){ 
  return stuff.size() * sizeof(double) + sizeof(int) + (...other things we have...); 
}
  
// Append the particle data to a buffer
void append_to_buffer(char *data){
  // Write the size of stuff
  int nelements = stuff.size();
  std::memcpy(data, &amp;nelements, sizeof(int));
  data += sizeof(int);

  // Write the data in stuff
  std::memcpy(data, stuff.data(), stuff.size() * sizeof(double));
  data += stuff.size() * sizeof(double);

  (...other things we have...)
}
  
// Assign the particle data from a buffer
void assign_from_buffer(char *data){
  // Read size of stuff
  int nelements;
  std::memcpy(&amp;nelements, data, sizeof(int));
  data += sizeof(int);

  // Allocate stuff
  stuff.resize(nelements);
  
  // Read stuff
  std::memcpy(stuff.data(), data, stuff.size() * sizeof(double));
  data += stuff.size() * sizeof(double);

  (...other things we have...)
}
</pre>
</div>
</div>
</p>

<p>
Some of the algorithms can do more stuff if certain quantities are present in the particles and some algorithms *require* certain additional methods. If you don't have this then you will either get a compile error or a runtime error (if this cannot be deduced at compile time). For example when we do grid assignment then the algorithm checks to see if we have a get_mass method. If we do then it will use this, if not then it will assume that all particles have the same mass. If we want to transform particles from realspace to redshift-space then we *must* have a get_vel method that gives us the velocity of the particle. Here is a list of the methods that some algorithms have as either an optional or a required part:
<div style="text-align: center">
  <div style="width:600px;display: inline-block;text-align: left">
    <pre class="prettyprint">
// (Below the types we used are not fixed and can be any type you want)

// Optional: VELOCITY
double *get_vel();  // Returns the velocity of the particle

// Optional: MASS (Fiducial value 1.0 if not present)
double get_mass();           // Returns the mass of the particle
void set_mass(double _mass); // Set the mass of the particle

// Optional: delta-f weight (e.g. neutrino particles). With this the density assignment gives
// delta = sum of the weights, i.e. the particles only carry the perturbation
double get_deltaf_weight();

// Optional: VOLUME (Fiducial value 1.0 if not present)
double get_volume();             // Returns the volume of the particle
void set_volume(double _volume); // Set the volume of the particle

// Optional: ID (create_particle_grid numbers the particles as id = (ix * N + iy) * N + iz)
long long int get_id();         // Returns the ID of the particle
void set_id(long long int _id); // Set the ID of the particle

//================================================================================
// RAMSES related methods
//================================================================================
// (for example when reading a RAMSES file it checks if these are present and stores the related data if they are)

// Optional: FAMILY (only used for RAMSES files)
char get_family();             // Returns the type of the particle
void set_family(char _family); // Set the type of the particle

// Optional: TAG (only used for RAMSES files)
char get_tag();          // Returns the tag of the particle
void set_tag(char _tag); // Set the tag of the particle

// Optional: LEVEL (only used for RAMSES files)
int get_level();            // Returns the level of the particle
void set_level(int _level); // Set the level of the particle

//================================================================================
// Lagrangian perturbation theory related stuff
//================================================================================

// Optional: Lagrangian Position 
double *get_q();  // Returns the Lagrangian coordinate of the particle

// Optional: Displacement fields (used by the COLA method)
double *get_D_1LPT();   // Returns the 1LPT displacement field of the particle
double *get_D_2LPT();   // Returns the 2LPT displacement field of the particle
double *get_D_3LPTa();  // Returns the 3LPTa displacement field of the particle
double *get_D_3LPTb();  // Returns the 3LPTb displacement field of the particle
double *get_dDdloga_1LPT();  // dD/dloga of the 1LPT displacement (for scale-dependent growth)
double *get_dDdloga_2LPT();  // dD/dloga of the 2LPT displacement (for scale-dependent growth)

//================================================================================
// Galaxy related methods (required for some galaxy related methods)
//================================================================================

// Optional: galaxy position on the sky and weight
double get_RA();         // Returns the right ascension of the galaxy
void set_RA(double _RA); // Set the right ascension of the galaxy

double get_DEC();          // Returns the declination of the galaxy
void set_DEC(double _DEC); // Set the declination of the galaxy

double get_z();        // Returns the redshift of the galaxy
void set_z(double _z); // Set the redshift of the galaxy

double get_weight();             // Returns the weight of the galaxy
void set_weight(double _weight); // Set the weight of the galaxy

//================================================================================
// Halo related method
//================================================================================

// Optional: Friends of Friend ID 
size_t get_fofid();            // Returns the FoFID of the particle (SIZE_MAX if it does not belong to a FoF group)
void set_fofid(size_t _FoFID); // Sets the FoFID of the particle (set by the FoF finder if this method exists,
                               // the same ID on all tasks)

</pre>
</div>
</div>
</p>

<p>
To get some info about what methods we have detected in your particle class you can call
<div style="text-align: center">
  <div style="width:600px;display: inline-block;text-align: left">
    <pre class="prettyprint">
FML::PARTICLE::info&lt; MyParticle &gt;();
</pre>
</div>
</div>
</p>

<!--
<p>
Here is a full list of the methods (dn't
<div style="text-align: center">
  <div style="width:600px;display: inline-block;text-align: left">
    <pre class="prettyprint">
template&lt; int NDIM &gt;
class Particle&lt; NDIM &gt;
  private:
    pos[NDIM];
    vel[NDIM];
    q[NDIM];
    D_1LPT[NDIM];
    D_2LPT[NDIM];
    D_3LPT[NDIM];
    long long int id;
    char FamilyAndTag[2];
    int level;
    double mass;
    double volume;
    double RA;
    double DEC;
    double redshift;
    double weight;

  public:

    constexpr int get_ndim() { return NDIM; }

    double *get_pos() { return pos; };
    double *get_vel() { return vel; };

    double get_mass() { return mass};
    void set_mass(double _mass) { mass = _mass};
    double get_volume() { return volume;};
    void set_volume(double _volume) { volume = _volume; };

    int get_id() { return id; };
    void set_id(int _id) { id = _id; }

    char get_family() { return FamilyAndTag[0]; };
    void set_family(char _family) { FamilyAndTag[0] = _family; };

    char get_tag() { return FamilyAndTag[1]; };
    void set_tag(char _tag) { FamilyAndTag[1] = _tag; };

    int get_level() { return level; };
    void set_level(int _level) { level = _level; };

    double *get_q() { return q; };
    double *get_D_1LPT() { return D_1LPT; };
    double *get_D_2LPT() { return D_2LPT; };
    double *get_D_3LPT() { return D_3LPT; };

    double get_RA() { return RA; };
    void set_RA(double _RA) { RA = _RA; };
    
    double get_DEC() { return DEC; };
    void set_DEC(doubel _DEC) { DEC = _DEC; };
    
    double get_z() { return redshift; };
    void set_z(double _z) { redshift = _z; };
    
    double get_weight() { return weight; };
    void set_weight(double _weight) { weight = _weight; };
}
</pre>
</div>
</div>
</p>
-->

