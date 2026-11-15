<hr/>
<header class="entry-header text-center">
  <h1 class="entry-title">Table of contents</h1>
</header>
<hr/>

<p class="page-toc">
<a href="index.php">Back to the main page</a><br>
<a href="documentation.php">Back to the documentation</a><br>
<a href="#Overview">1. Overview: FFTWGrid</a><br>
<a href="#GridStructure">2. How the grid is structured</a><br>
<a href="#Types">3. Internal Types</a><br>
<a href="#Raw">4. Accessing the raw data</a><br>
<a href="#FFTs">5. Fourier transforms</a><br>
<a href="#Loading">6. Loading and saving a grid</a><br>
<a href="#Status">7. Status of the grid</a><br>
<a href="#Wisdom">8. Planning and FFTW wisdom</a><br>
<a href="#Memory">9. Memory Logging</a><br>
<a href="#Comm">10. Communicate boundaries</a><br>
<a href="#Example">11. Example: use the grid to solve the Poisson equation</a><br>
<a href="#Resampling">12. Changing the resolution of a Fourier grid</a><br>
</p>

<header class="entry-header text-center">
  <h1 class="entry-title" id="Overview">Overview: FFTWGrid</h1>
</header>
<hr/>  

<p>
The FFTWGrid class is a class for holding either a real or a Fourier grid in the same grid (to save memory) and to perform in-place Fourier transforms of the grid with FFTW. With MPI the grid cells are divided among the tasks according to their $x$ coordinate (see Figure 1 below). We also have the possibility to allocate extra slices on the left and right of the local grid to keep boundary values from neighboring tasks that can be filled whenever we need it. This feature is very useful for several algorithms for example when we assign particles to the grid to create a density field and the kernel we use are broader than 1 grid cell (i.e. anything beyond Nearest-Grid-Point assignment) and when we want to compute derivatives of the grid.
</p>
<p>
See the header file FFTWGrid.h for all available methods and for how they are implemented. Below are some examples of how to use the grid and some more info about the available methods. 
</p>

<p>
<b>Pencils and complex grids.</b> The same interface exists on a pencil decomposition, <a href="pencilgrid.php" style="display: inline">$\texttt{PencilGrid}$</a>, which splits $x$ and $y$ over a 2D grid of tasks. There are also complex-to-complex grids, $\texttt{CFFTWGrid}$ (slabs) and $\texttt{CPencilGrid}$ (pencils), with <tt>fftw_forward()</tt> / <tt>fftw_backward()</tt> and all $N^3$ modes. Code written against the common grid interface (<tt>FFTWGrid/GridInterface.h</tt>) works with all of them: see <a href="pencilgrid.php#Generic" style="display: inline">writing code for slabs and pencils</a>.
</p>

<header class="entry-header text-center">
<h1 class="entry-title" id="GridStructure">How the grid is structured</h1>
</header>
<hr/>  

<p>
The grid is stored as a one dimensional array and the index of a real cell in the main grid is $i = iz + 2(N/2+1)(iy + N\cdot (ix + \ldots))$, i.e. the last coordinate runs first and we have some extra <a href="https://www.fftw.org/fftw3_doc/Multi_002dDimensional-DFTs-of-Real-Data.html" style="display: inline">padding</a> in the last dimension due to how FFTW works. The index of a fourier (a complex number) cell is $i = ikz + (N/2+1)(iky + N\cdot (ikx+\ldots))$. The $N/2+1$ is because a real-to-complex FFT necessarily satisfies $f(k) = f^*(-k)$ so only the positive $k_z$ values are stored. This is only important if you want to do the loops over the grid yourself and not use the built in methods which takes care of this for you.
</p>
<p>
On a given task a cell will have local coordinates $(ix,iy,iz)$ where $ix\in[0,n_{x-\rm local})$ and $iy,iz\in[0,N)$. The relation between local and global coordinates are given by $ix_{\rm global} = n_{x-\rm start-local} + ix_{\rm local}$ (the other coords being the same). All methods asking for coord assume you give it local coordinates.
</p>
<p>
On top of the regular cells we have extra cells on the left and the right of the grid. These are for holding the value of cells on neighboring tasks and can be filled by running grid.communicate_boundaries(). The cells on the left can be accessed with local coordinates $ix=-1,-2,\ldots$ and so on and the cells on the right can be accessed with $ix = n_{x-\rm local}, \ldots$. The extra cells are laid out in memory as if they were normal cells.
</p>
<figure class="science-figure">
<img src="images/fftwgrid-storage.svg" alt="FFTWGrid local slab with optional neighboring slices and the padded real-to-complex transform buffer">
<figcaption>FFTWGrid stores a local slab with optional neighboring slices. FFT padding accommodates the real-to-complex transform, which reuses the aligned buffer for Fourier values.</figcaption>
</figure>

<header class="entry-header text-center">
<h1 class="entry-title" id="Types">Internal Types</h1>
</header>
<hr/>  
<p>
The grid uses the internal type FloatType for real quantities and ComplexType for fourier quantities. This is just double and std::complex&lt;double&gt; unless we use single or long double precision (see FFTWGlobal.h for type definitions). Gridcell indices (local ones) have the type IndexIntType which is typically set to be long long int (it has to be signed if you want to change it). Coordinates are just normal int's.
</p>

<header class="entry-header text-center">
<h1 class="entry-title" id="Raw">Accessing the raw data</h1>
</header>
<hr/> 
<p>
We provide iterators for looping over all the local gridcells in either real-space or fourier-space (and methods for getting the position/fourier wavenumber of the cell in the global grid):
<div style="text-align: center">
<div style="width:600px;display: inline-block;text-align: left">
<pre class="prettyprint">
// Loop over all real cells
for(auto &amp;&amp; real_index : grid.get_real_range()){
  // Get the value in the cell
  auto value = grid.get_real_from_index(real_index);
  
  // Assign a value to the cell
  grid.set_real_from_index(real_index, value);
}

// Loop over all fourier cells
for(auto &amp;&amp; fourier_index : grid.get_fourier_range()){
  std::array&lt;double, NDIM&gt; kvec;
  double kmag2;
  
  // Get the Fourier wavevector k_vec and the norm |k_vec|^2 of the current cell
  grid.get_fourier_wavevector_and_norm2_by_index(fourier_index, kvec, kmag2);
  
  // Fetch the value in the cell
  auto value = grid.get_fourier_from_index(fourier_index);

  // Set the value in the cell
  grid.set_fourier_from_index(fourier_index, value);
}
</pre>
</div>
</div>
</p>
<p>
If you want to parallelize the range based for loop over real cells this is easily done by splitting the loop into a loop over slices plus a loop over cells in the slices and adding the OpenMP pragma to the former loop. For fourier cells there is no padding to worry about so one can just have one raw loop over all the cells and parallelize that or do it the same way as for the real case:
<div style="text-align: center">
<div style="width:600px;display: inline-block;text-align: left">
<pre class="prettyprint">
// Loop over all real cells in parallel
// Each thread works on a different x-slice
#pragma omp parallel for
for(int islice = 0; islice &lt; grid.get_local_nx(); islice++){
  // Loop over all real cells in a given slice (fixed x-value)
  for(auto &amp;&amp; real_index : grid.get_real_range(islice, islice+1)){
    ...
  }
}

// Loop over all Fourier cells in parallel
// Each thread works on a different kx-slice
#pragma omp parallel for
for (int islice = 0; islice &lt; grid.get_local_nx(); islice++) {
    // Loop over all fourier cells in a given slice (fixed kx value)
    for (auto &amp;&amp; fourier_index : grid.get_fourier_range(islice, islice + 1)) {
      ...
    }
}
</pre>
</div>
</div>
</p>

</p>
<p>
If you want access to the raw data there are methods for getting that:
</p>
<p>
<div style="text-align: center">
<div style="width:600px;display: inline-block;text-align: left">
<pre class="prettyprint">FloatType *get_real_grid_left();               // The left most slice (start at slice ix = -nleft_extra)
FloatType *get_real_grid();                    // The main grid       (start at slice ix = 0)
FloatType *get_real_grid_right();              // The right grid      (start at slice ix = NLocal_x)
FloatType *get_real_grid_by_slice(int slice);  // Get the ix'th slice (i.e. -nleft_extra &lt;= ix &lt; NLocal_x+nright_extra)
ComplexType *get_fourier_grid();               // The main Fourier grid 
</pre>
</div>
</div>
</p>

<p>
Or access the cells by their (local) coordinates. Some of the other methods we have:
</p>
<p>
<div style="text-align: center">
<div style="width:600px;display: inline-block;text-align: left">
<pre class="prettyprint">// Value of a cell given its local coordinate (ix,iy,...), ix can also be in the extra slices
FloatType get_real(const std::array&lt;int, N&gt; &amp; coord) const;
void set_real(const std::array&lt;int, N&gt; &amp; coord, const FloatType value);
void add_real(const std::array&lt;int, N&gt; &amp; coord, const FloatType value);
ComplexType get_fourier(const std::array&lt;int, N&gt; &amp; coord) const;
void set_fourier(const std::array&lt;int, N&gt; &amp; coord, const ComplexType value);

// Converting between index and coordinate
std::array&lt;int, N&gt; get_coord_from_index(const IndexIntType index_real) const;
std::array&lt;int, N&gt; get_fourier_coord_from_index(const IndexIntType index_fourier) const;
IndexIntType get_index_real(const std::array&lt;int, N&gt; &amp; coord) const;
IndexIntType get_index_fourier(const std::array&lt;int, N&gt; &amp; coord) const;

// Position of a real cell in [0,1)^N and the wavevector of a Fourier cell (k = 2pi n)
std::array&lt;double, N&gt; get_real_position(const std::array&lt;int, N&gt; &amp; coord) const;
std::array&lt;double, N&gt; get_fourier_wavevector(const std::array&lt;int, N&gt; &amp; coord) const;
std::array&lt;int, N&gt; get_fourier_integer_wavevector_from_index(const IndexIntType index) const;

// Info about the grid
int get_nmesh() const;
ptrdiff_t get_local_nx() const;       // Number of x-slices on this task
ptrdiff_t get_local_x_start() const;  // The global x-index of the first local slice
int get_n_extra_slices_left() const;
int get_n_extra_slices_right() const;
bool nan_in_grids() const;            // Check for NaNs (useful for debugging)
void info();                          // Print info about the grid
explicit operator bool() const;       // Is the grid allocated?

// Free the memory or allocate a new grid
void free();
void reallocate(int Nmesh, int nleft, int nright);
</pre>
</div>
</div>
</p>
<p>
So for example if you insist on writing explicit loops for going through the grid and not use the built in methods then you can do it like this (here for a 3D grid):
<div style="text-align: center">
<div style="width:600px;display: inline-block;text-align: left">
<pre class="prettyprint">
FFTWGrid&lt;3&gt; grid(Nmesh);
auto local_nx = grid.get_local_nx();
auto local_x_start = grid.get_local_x_start();

// Loop over all the local real grid cells
auto *r = grid.get_real_grid();
for(int i = 0; i &lt; local_nx; i++){
  int iglobal = i + local_x_start;
  for(int j = 0; j &lt; Nmesh; j++){
    for(int k = 0; k &lt; Nmesh; k++){
      // The current cell has global coordinates (iglobal,j,k) 
      
      // Index of the cell in the real grid
      long long int real_index = (i*Nmesh + j)*2*(Nmesh/2+1) + k;

      // Set the value in the cell
      r[real_index] = 0.0;
    }
  }
}

// Loop over the local fourier grid cells
auto *f = grid.get_fourier_grid();
for(int i = 0; i &lt; local_nx; i++){
  int iglobal = i + local_x_start;
  for(int j = 0; j &lt; Nmesh; j++){
    for(int k = 0; k &lt; Nmesh/2+1; k++){
      // Index of the cell in the fourier grid
      long long int fourier_index = (i*Nmesh + j)*(Nmesh/2+1) + k;

      // The Fourier wavevector corresponding to the current cell
      double kx = iglobal &lt;= Nmesh/2 ? iglobal : iglobal - Nmesh;
      double ky = j       &lt;= Nmesh/2 ? j       : j       - Nmesh;
      double kz = k       &lt;= Nmesh/2 ? k       : k       - Nmesh;

      // Set the value in the cell
      f[fourier_index] = 0.0;
    }
  }
}
</pre>
</div>
</div>

</p>
<p>
We provide some simple functions for filling the grid with data:
</p>
<p>
<div style="text-align: center">
<div style="width:600px;display: inline-block;text-align: left">
<pre class="prettyprint">
// Fill the whole grid with a constant value
void fill_real_grid(const FloatType val);
void fill_fourier_grid(const ComplexType val);

// Fill the main grid from a function specifying the value at a given position (or wavevector)
void fill_real_grid(const std::function&lt;FloatType(std::array&lt;double,NDIM&gt;&amp;)&gt; &amp; func);
void fill_fourier_grid(const std::function&lt;ComplexType(std::array&lt;double,NDIM&gt;&amp;)&gt; &amp; func);

// For example fill the grid with sin(2pi(x+y))
std::function&lt;FloatType(std::array&lt;double,NDIM&gt;&amp;)&gt; func = [](std::array&lt;double,NDIM&gt;&amp; pos) -&gt; FloatType {
  return std::sin(2*M_PI*(pos[0] + pos[1]));
};
grid.fill_real_grid(func);
</pre>
</div>
</div>
</p>

<header class="entry-header text-center">
<h1 class="entry-title" id="FFTs">Fourier transforms</h1>
</header>
<hr/> 

<p>
The code uses the same <a href="https://www.fftw.org/fftw3_doc/The-1d-Discrete-Fourier-Transform-_0028DFT_0029.html#The-1d-Discrete-Fourier-Transform-_0028DFT_0029" style="display: inline">convention</a> as FFTW, here in 1D and non-discrete for simplicity:
$$f(k) = \int f(x)e^{-ikx}dx$$
$$f(x) = \frac{1}{2\pi}\int f(k)e^{ikx}dk$$
So for example with derivatives $\nabla f \to +ik f$. The normalization factor we get from taking a transform and then the inverse transform, $\frac{1}{N^{\rm NDIM}}$, is automatically taken into account when we do a complex-to-real transform. You can transform the fields as follows:
<div style="text-align: center">
  <div style="width:600px;display: inline-block;text-align: left">
    <pre class="prettyprint">
// Transform to Fourier space
grid.fftw_r2c();

// Transform back to real space (grid is now the same as we started with up to numerical errors - normalization is taken care of)
grid.fftw_c2r();
</pre>
  </div>
</div>
</p>

<header class="entry-header text-center">
<h1 class="entry-title" id="Loading">Loading and saving a grid</h1>
</header>
<hr/>  
<p>
To save a grid to file and later load it you can use the class methods:
<div style="text-align: center">
<div style="width:600px;display: inline-block;text-align: left">
<pre class="prettyprint">void load_from_file(std::string fileprefix);
void dump_to_file(std::string fileprefix);
</pre>
</div>
</div>
</p>
<p>
This will create files fileprefix.X with X being the task number (FML::ThisTask). There is no support for reading in a grid run on a different number of tasks.
</p>

<header class="entry-header text-center">
<h1 class="entry-title" id="Status">Status of the grid</h1>
</header>
<hr/> 

<p>
If you want to keep control over if the grid is in real or fourier space you can specify this:
<div style="text-align: center">
<div style="width:600px;display: inline-block;text-align: left">
<pre class="prettyprint">// Create a grid and set the status of grid to be in real space
FFTWGrid&lt;3&gt; grid(Nmesh);
grid.set_grid_status_real(true);

// Perform a Fourier transform
grid.fftw_r2c();

// The status of the grid is now in Fourier space
assert( grid.get_grid_status_real() == false );
</pre>
</div>
</div>
</p> 
<p>
Whenever we do a transform the status of the grid is updated. If you have DEBUG_FFTWGRID defined then if you try to do a Fourier transform to real space and the status is already real then you will get a warning.
</p> 

<header class="entry-header text-center">
  <h1 class="entry-title" id="Wisdom">Planning and FFTW wisdom</h1>
</header>
<hr/> 

<p>
The default FFTW planning flag is FFTW_ESTIMATE as this does not need to touch the data in the grid (otherwise we would have to make copies). If you want to make better plans you can use the class methods below for creating FFTW wisdom:
<div style="text-align: center">
  <div style="width:600px;display: inline-block;text-align: left">
    <pre class="prettyprint">
void create_wisdom(int planner_flag, int numthreads = FML::NThreads);
void load_wisdom(std::string filename) const;
void save_wisdom(std::string filename) const;

// Make a plan
FFTWGrid&lt;3&gt; grid(Nmesh);
grid.create_wisdom(FFTW_MEASURE, FML::NThreads);
</pre>
  </div>
</div>
</p> 
<p>
Here planner_flag is FFTW_MEASURE, FFTW_PATIENT or FFTW_EXHAUSTIVE. These methods take care of communicating wisdom to all tasks. Note that you have to create wisdom before filling the grid with data otherwise it will get destroyed. numthreads is how many threads we want to use in the FFT and only relevant if USE_FFTW_THREADS is set.
</p>


<header class="entry-header text-center">
  <h1 class="entry-title" id="Memory">Memory Logging</h1>
</header>
<hr/> 

<p>
If you want to keep track of how much memory is allocated by the grids at any given time one can compile with USE_MEMORYLOG = true. This makes a map of all big allocations that can be printed out. For the memory logger to output to you exactly what grids are allocated you must provide a label for this:
<div style="text-align: center">
  <div style="width:600px;display: inline-block;text-align: left">
    <pre class="prettyprint">
// Make a grid
FFTWGrid&lt;3&gt; grid(Nmesh);

// Add a label
grid.add_memory_label("My grid");
</pre>
  </div>
</div>
</p>

<header class="entry-header text-center">
  <h1 class="entry-title" id="Comm">Communicate boundaries</h1>
</header>
<hr/>  
<p>
To get the boundary values from neighboring tasks in the extra slices we simply call:
<div style="text-align: center">
  <div style="width:600px;display: inline-block;text-align: left">
    <pre class="prettyprint">
grid.communicate_boundaries();
</pre>
</div>
</div>
</p>

<header class="entry-header text-center">
<h1 class="entry-title" id="Example">Example: Use the class to solve the Poisson equation</h1>
</header>
<hr/>  

<p>
Create a grid with Nmesh gridcells per dimension having n_left and n_right extra x-slices on the left and right for holding the boundary from the neighboring tasks (needed by some algorithms, in this example it's not needed): 
<div style="text-align: center">
<div style="width:600px;display: inline-block;text-align: left">
<pre class="prettyprint">const int NDIM = 3;
const int Nmesh = 16;
const int n_left = 1;
const int n_right = 1;
FML::GRID::FFTWGrid&lt;NDIM&gt; grid(Nmesh, n_left, n_right); 

// Number of x-slices on current task
auto local_nx = grid.get_local_nx();

// The cell with local x-coord=0 has x-coord of this in the global grid
auto local_x_start = grid.get_local_x_start();</pre>
</div>
</div>
Loop through all the real grid cells and set the value of the grid cells to be $\phi(x,y,z) = \sin(2\pi x)$:
<div style="text-align: center">
<div style="width:600px;display: inline-block;text-align: left">
<pre class="prettyprint">for(auto &amp;&amp; real_index : grid.get_real_range()){
  // A vector with the local coordinates (ix,iy,iz) of the cell
  auto coord = grid.get_coord_from_index(real_index);
  
  // x position of the cell in the global grid
  double x_global = (coord[0] + local_x_start) / double(Nmesh); 
  
  // Get the value in the cell
  auto value = grid.get_real_from_index(real_index);
  
  // Assign a value to the cell: sin(2pi x)
  value = std::sin(2.0 * M_PI * x_global );
  grid.set_real_from_index(real_index, value);
}</pre>
</div>
</div>
</p>

<p>
Perform a Fourier transform of the grid to Fourier space:
<div style="text-align: center">
<div style="width:600px;display: inline-block;text-align: left">
<pre class="prettyprint">grid.fftw_r2c();</pre>
</div>
</div>
</p>
<p>
Loop over all the Fourier grid cells and divide by $-k^2$:
<div style="text-align: center">
<div style="width:600px;display: inline-block;text-align: left">
<pre class="prettyprint">
std::array&lt;double,NDIM&gt; kvec;
double kmag2;
for(auto &amp;&amp; fourier_index : grid.get_fourier_range()){
  // Get the Fourier wavevector k_vec and the norm |k_vec|^2 of the current cell
  grid.get_fourier_wavevector_and_norm2_by_index(fourier_index, kvec, kmag2);
  
  // Fetch the value in the cell
  auto value = grid.get_fourier_from_index(fourier_index);
  
  // Divide the cell by -k^2 and assign it and set DC mode to 0
  if(kmag2 &gt; 0.0)
    grid.set_fourier_from_index(fourier_index, -value / kmag2);
  else 
    grid.set_fourier_from_index(fourier_index, 0.0);
}</pre>
</div>
</div>
</p>
<p>
Perform a Fourier transform of the grid back to real space:
<div style="text-align: center">
<div style="width:600px;display: inline-block;text-align: left">
<pre class="prettyprint">grid.fftw_c2r();</pre>
</div>
</div>
</p>
<p>
We have now solved the Poisson equation $\nabla^2 \phi = \sin(2\pi x)$ and have the solution $\phi(x,y,z) = -\frac{1}{4\pi^2}\sin(2\pi x)$ in real space in the grid. Note that the code above works just as well in serial as in parallel as the parallelization is taken into account behind the scenes.
</p>

<header class="entry-header text-center">
  <h1 class="entry-title" id="Resampling">Changing the resolution of a Fourier grid</h1>
</header>
<hr/>

<p>
<tt>resize_fourier_grid(input, nmesh)</tt> copies the Fourier coefficients to a grid of another size (no interpolation): a mode survives only if every component satisfies $2|n_i| &lt; \min(N_{\rm in}, N_{\rm out})$, which drops the ambiguous Nyquist planes of even meshes. For repeated resampling (e.g. padding and cropping for dealiased LPT) a <tt>FourierResamplingPlan</tt> computes the MPI routing once and then moves only the coefficients, in bounded messages:
</p>
<div style="text-align: center">
  <div style="width:600px;display: inline-block;text-align: left">
    <pre class="prettyprint">
FML::GRID::FFTWGrid&lt;3&gt; output(newmesh, 1, 1);
FML::GRID::FourierResamplingPlan&lt;3&gt; plan(input, output);
plan.apply(input, output);
plan.apply(next_field, next_output);   // same meshes and slab layout
</pre>
</div>
</div>
<p>
<tt>allocated_storage_bytes()</tt> reports the retained payload capacity of a grid.
</p>
