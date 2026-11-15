<hr/>
<header class="entry-header text-center">
  <h1 class="entry-title">Table of contents</h1>
</header>
<hr/>

<p class="page-toc">
<a href="index.php">Back to the main page</a><br>
<a href="documentation.php">Back to the documentation</a><br>
<a href="#Overview">1. Overview: FileUtils</a><br>
<a href="#Lua">2. Lua</a><br>
<a href="#Ascii">3. Ascii</a><br>
<a href="#Gadget">4. Gadget</a><br>
<a href="#Ramses">5. Ramses</a><br>
<a href="#Dict">6. Dictionary</a><br>
</p>

<header class="entry-header text-center">
  <h1 class="entry-title" id="Overview">Overview: FileUtils</h1>
</header>
<hr/> 

<p>
Read and write some common filetypes. Everything is in the namespace FML::FILEUTILS (and the sub-namespaces GADGET and RAMSES), apart from the ParameterMap which is in FML::UTILS.
</p> 

<header class="entry-header text-center">
  <h1 class="entry-title" id="Lua">Lua</h1>
</header>
<hr/> 

<p>
Lua provides a good way of reading parameter files. We provide a simple LuaFileParser (LuaFileParser/LuaFileParser.h, requires compiling with USE_LUA). Example parameterfile:
<div style="text-align: center">
  <div style="width:600px;display: inline-block;text-align: left">
    <pre class="prettyprint">
---------------------------
-- Example parameter file
---------------------------

-- An array of strings
AStringArray = {"Fratton", "752", "Jaduja"}

-- An array of doubles
ADoubleArray = {1.0, 2.0, 3.0}

-- A simple string
AString = "Hello"

-- A simple bool
ABool = true

-- A simple double
ADouble = 1.2

-- We can use simple formulas to set a variable
ANewDouble = ADouble * 2^10 * ADoubleArray[2]
</pre>
</div>
</div>
</p>


<p>
...which can be read as follows:
<div style="text-align: center">
  <div style="width:600px;display: inline-block;text-align: left">
    <pre class="prettyprint">
  std::string filename = "input.lua";
  FML::FILEUTILS::LuaFileParser lfp(filename);

  //===============================================================
  // Read the parameters in the file. If optional and not found the value is set to the fiducial value
  // To throw an error if we don't find it use lfp.required (true) instead of optional (false)
  //===============================================================
  auto ABool        = lfp.read_bool                ("ABool",                   false, lfp.optional);
  auto ADouble      = lfp.read_double              ("ADouble",                   0.0, lfp.optional);
  auto ANewDouble   = lfp.read_double              ("ANewDouble",                0.0, lfp.optional);
  auto AString      = lfp.read_string              ("AString",      "Fiducial value", lfp.optional);
  auto AStringArray = lfp.read_string_array        ("AStringArray",               {}, lfp.optional);
  auto ADoubleArray = lfp.read_number_array&lt;double&gt;("ADoubleArray",               {}, lfp.optional);
</pre>
</div>
</div>
</p>

<header class="entry-header text-center">
  <h1 class="entry-title" id="Ascii">Ascii</h1>
</header>
<hr/> 

<p>
Read simple ascii files:
<div style="text-align: center">
  <div style="width:600px;display: inline-block;text-align: left">
    <pre class="prettyprint">
// Read a regular ascii file with nskip header lines and containing ncols columns
// nestimated_lines is the amount we allocate for originally. Reallocated if file is larger
// Not perfect for really large files due to all the allocations we have to do
DVector2D read_regular_ascii(
    std::string filename,
    int ncols,
    std::vector&lt;int&gt; cols_to_keep,
    int nskip,
    size_t nestimated_lines = 10000);

// As above, but include every line read with probability fraction_to_read
DVector2D read_regular_ascii_subsampled(
    std::string filename,
    int ncols,
    std::vector&lt;int&gt; cols_to_keep,
    int nskip,
    size_t nestimated_lines = 10000,
    double fraction_to_read = 1.0,
    unsigned int randomSeed = 1234);
</pre>
</div>
</div>
</p>

<header class="entry-header text-center">
  <h1 class="entry-title" id="Gadget">Gadget</h1>
</header>
<hr/> 

<p>
Read and write Gadget files (files with position, velocity and ID) in the binary format (GadgetUtils/GadgetUtils.h) and, if compiled with USE_HDF5, the HDF5 format (GadgetUtils/GadgetUtilsHDF5.h; the classes GadgetHDF5Reader and GadgetHDF5Writer have the same interface as the binary ones and is_hdf5_snapshot(fileprefix) tells you which one you have). When working with MPI it has the option to just keep particles belonging to the local task (for cases where all tasks read all the same files). Should also include a method where N tasks read and distribute them to other tasks at some point (useful when reading very large files for which having all tasks read all files will be super slow). If you are just looking for a Gadget reader there is also a C++ library by S. Bird on github: <a href="https://github.com/sbird/GadgetReader" style="display: inline">GadgetReader</a>. Anyway here is how to use it:
<div style="text-align: center">
  <div style="width:600px;display: inline-block;text-align: left">
    <pre class="prettyprint">
// Container to store it in
std::vector&lt;Particle&gt; part;

// Set up reader
FML::FILEUTILS::GADGET::GadgetReader g;
const std::string fileprefix = "snapshots/gadget"; // reads gadget.0, gadget.1, ...
const bool verbose = true;
const double buffer_factor = 1.0;
const bool only_keep_part_in_domain = true;

// Read all gadget files and fill the data in part
// If only_keep_part_in_domain then with MPI we only keep particles belonging to local task
// If buffer_factor &gt; 1 we allocate storage for more particles than needed
g.read_gadget(fileprefix, part, buffer_factor, only_keep_part_in_domain, verbose);
</pre>
</div>
</div>
</p>
<p>
<div style="text-align: center">
  <div style="width:600px;display: inline-block;text-align: left">
    <pre class="prettyprint">
// More options:
// If we have elements stored in a different order or some fields missing
// then we can set it as follows before reading
std::vector&lt;std::string&gt; fields_in_file = {"POS", "VEL", "ID"};
g.set_fields_in_file(fields_in_file);

// If we just want to read a single Gadget file then we can do it as follows
g.read_gadget_single(filename, part, only_keep_part_in_domain, verbose);
</pre>
</div>
</div>
</p>
<p>
There are also methods for writing a Gadget file to disk:
</p>
<p>
<div style="text-align: center">
  <div style="width:600px;display: inline-block;text-align: left">
    <pre class="prettyprint">
std::vector&lt;Particle&gt; part;
... create some particles ...

// Write a single GADGET file (per task)
FML::FILEUTILS::GADGET::GadgetWriter gw;
const double scale_factor = 1.0;
const double boxsize = 1000;
const double OmegaM = 0.3;
const double OmegaLambda = 0.7;
const double h = 0.7;
const int nfiles = FML::NTasks;
// Factor to convert user-units to positions in [0,box)
// (here we assume we have positions in [0,1) in part)
const double pos_norm = boxsize;    
// Factor to convert user-units to sqrt(a)dxdt in km/s
// (here we assume we have peculiar velocities in km/s in part)
const double vel_norm = 1.0 / std::sqrt(scale_factor); 
gw.write_gadget_single("test." + std::to_string(FML::ThisTask),
                       part.data(),
                       part.size(),
                       NumPartTotal,
                       nfiles,
                       scale_factor,
                       boxsize,
                       OmegaM,
                       OmegaLambda,
                       h,
                       pos_norm,
                       vel_norm);

// To write several sets of particles (e.g. CDM, baryons and neutrinos in different containers)
// as different Gadget particle types in the same file see write_gadget_species
</pre>
</div>
</div>
</p>

<header class="entry-header text-center">
  <h1 class="entry-title" id="Ramses">Ramses</h1>
</header>
<hr/> 
<p>
Ramses reader (reads only info and particle files; RamsesUtils/RamsesUtils.h). With MPI we have the option of only storing the particles that belong to the local task. What we store depends on what you have in your particle class: if you only have a position then we only store positions, but if you have any of tag, mass, family, level, velocity etc. then we store that also.
<div style="text-align: center">
  <div style="width:600px;display: inline-block;text-align: left">
    <pre class="prettyprint">
// Container to store it in
// We check if the class has the methods set_id, set_family, set_tag, set_level
// set_mass, get_pos*, get_vel* and if they are present the data will be stored in the particle. 
// If not we ignore it
std::vector&lt;Particle&gt; part;

// Set up the Ramses reader
// snapdir is the folder with the snapshot (output_0000X)
// keep_only_particles_in_domain means each task only keeps particles in its own domain
// so after we are done the particles are distributed among tasks
// If buffer_factor &gt; 1 we allocate storage for more particles than needed
std::string snapdir = "snapshots/output_00008";
const double buffer_factor = 1.0;
const bool keep_only_particles_in_domain = true;
const bool verbose = true;
FML::FILEUTILS::RAMSES::RamsesReader reader(snapdir, buffer_factor, keep_only_particles_in_domain, verbose);

// Read all particles and store it in part
reader.read_ramses(part);
</pre>
</div>
</div>
</p>

<p>
<div style="text-align: center">
  <div style="width:600px;display: inline-block;text-align: left">
    <pre class="prettyprint">
// More options:
// The fiducial file format is POS,VEL,MASS,ID,LEVEL,FAMILY,TAG, but
// if the format is different one can set it here and we can also set if we want
// to store the resulting data or not
std::vector&lt;std::string&gt; fileformat{"POS", "VEL", "MASS", "ID"};
reader.set_file_format(fileformat);

// To just read the ith file (part_0000X.out0000[i+1]) and store it in part
reader.read_ramses_single(i, part);
</pre>
</div>
</div>
</p>
<p>
After reading the positions will be in [0,1) and the velocities will be the dimensionless $\frac{a^2 \frac{dx}{dt}}{(H_0 B)}$ (standard RAMSES units).
</p>

<header class="entry-header text-center">
  <h1 class="entry-title" id="Dict">Dictionary</h1>
</header>
<hr/> 

<p>
A simple class to hold parameters of different types:
<div style="text-align: center">
  <div style="width:600px;display: inline-block;text-align: left">
    <pre class="prettyprint">
// Add some values
FML::UTILS::ParameterMap p;
p["AString"] = std::string("Hello");
p["ADouble"] = 1.0;
p["AnInt"]   = 5;
p["AVector"] = std::vector&lt;double&gt;{0.0, 1.0, 2.0};

// Fetch values from the map (if not found will throw an error)
auto AString = p.get&lt;std::string&gt;("AString");
auto ADouble = p.get&lt;double&gt;("ADouble");
auto AnInt   = p.get&lt;int&gt;("AnInt");
auto AVector = p.get&lt; std::vector&lt;double&gt; &gt;("AVector");

// Fetch values from the map (if not found it will use the fiducial value)
auto AnotherInt = p.get&lt;int&gt;("AnotherInt", 10);
</pre>
</div>
</div>
</p>


