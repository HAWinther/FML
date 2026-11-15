<hr/>
<header class="entry-header text-center">
  <h1 class="entry-title">Table of contents</h1>
</header>
<hr/>

<p class="page-toc">
<a href="index.php">Back to the main page</a><br>
<a href="documentation.php">Back to the documentation</a><br>
<a href="#Overview">1. Overview: MPIGrid</a><br>
<a href="#Own">2. Make a grid of your own type</a><br>
<a href="#Function">3. Assign grid from a function</a><br>
<a href="#Access">4. Accessing the data</a><br>
<a href="#Comm">5. Communicate boundaries</a><br>
<a href="#Arithmetic">6. Arithmetic and converting to FFTWGrid</a><br>
</p>

<header class="entry-header text-center">
  <h1 class="entry-title" id="Overview">Overview: MPIGrid</h1>
</header>
<hr/>  

<p>
MPIGrid is a general class for storing a grid of any type across many tasks. Algorithms using this class generally assume the type is a field or ring type, i.e. it implements the arithmetic operations +,-,* and maybe / together with multiplication by a scalar, but that is no requirement. Just as with FFTWGrid, with MPI the grid cells are divided among the tasks according to their $x$ coordinate. Unfortunately I have not bothered to make the syntax for this class the same as for FFTWGrid which should be done. This class is useful, but it can be improved a lot (especially on how we return data from it, right now it's not very safe as all get methods give references and not just values which is bad). Its current use is mainly for our multigrid solver.
</p>
<p>
<b>Pencils.</b> An <tt>MPIGrid</tt> can also be made on a domain, <tt>MPIGrid&lt;NDIM, T&gt;(N, periodic, domain, n_left, n_right)</tt>: on a Pencil domain it is split along $x$ and $y$ (the extra cells too). <tt>ConvertToMPIGrid(grid, mpigrid)</tt> copies any grid (<tt>FFTWGrid</tt>, <tt>PencilGrid</tt>) into an <tt>MPIGrid</tt> with the same local cells (made on <tt>multigrid_domain_of(grid)</tt>). See <a href="pencilgrid.php" style="display: inline">pencil grids</a>.
</p>

<header class="entry-header text-center">
  <h1 class="entry-title" id="Own">Make a grid of your own type</h1>
</header>
<hr/>  

<p>
Make a grid of your own type (a vector in each cell):
<div style="text-align: center">
  <div style="width:600px;display: inline-block;text-align: left">
    <pre class="prettyprint">
// A simple gridcell
typedef struct {
double vec[3];
double mass;
} GridCell;

// Make a grid of it
const int NDIM   = 3;
const int Nmesh  = 16;
const int nleft  = 1;
const int nright = 1;
const bool periodic = true;
FML::GRID::MPIGrid&lt;NDIM, GridCell&gt; grid(Nmesh, periodic, nleft, nright);  
</pre>
</div>
</div>

</p>


<header class="entry-header text-center">
  <h1 class="entry-title" id="Function">Assign grid from a function</h1>
</header>
<hr/>  

<p>
Make a grid containing doubles:
<div style="text-align: center">
  <div style="width:600px;display: inline-block;text-align: left">
    <pre class="prettyprint">
// Make a grid of it
const int NDIM   = 3;
const int Nmesh  = 16;
const int nleft  = 1;
const int nright = 1;
const bool periodic = true;
FML::GRID::MPIGrid&lt;NDIM, double&gt; grid(Nmesh, periodic, nleft, nright);  
</pre>
</div>
</div>
Assign the grid from a function:
<div style="text-align: center">
  <div style="width:600px;display: inline-block;text-align: left">
    <pre class="prettyprint">
// We want to set each gridcell equal to x+y+z
std::function&lt;double(std::array&lt;double,NDIM&gt; &amp;)&gt; func = [](std::array&lt;double,NDIM&gt; &amp; pos){
  double value = 0.0;
  for(int idim = 0; idim &lt; NDIM; idim++)
    value += pos[idim];
  return value;
};

// Assign grid
grid.set_y(func);
</pre>
</div>
</div>

</p>

<header class="entry-header text-center">
  <h1 class="entry-title" id="Access">Accessing the data</h1>
</header>
<hr/> 
<p>
There are several ways to access the data, see the header file. Here is an example:
<div style="text-align: center">
  <div style="width:600px;display: inline-block;text-align: left">
    <pre class="prettyprint">
// Get a pointer to the main grid
auto *y = grid.get_y();

// Loop through the grid
auto NtotLocal = grid.get_NtotLocal();
for(auto index = 0; index &lt; NtotLocal; index++){
  // Get coordinate of cell
  auto coord = grid.coord_from_index(index);

  // Get a list of the index of cells to the right/left top/bottom etc. 
  auto index_list = grid.get_neighbor_gridindex(index);

  // Get the global position of the cell
  auto pos = grid.get_pos(coord); 
  //...or grid.get_pos(index) if you don't have computed coord

  // Set the value to 0
  grid.set_y(index, 0.0);
} 
</pre>
</div>
</div>
</p> 


<header class="entry-header text-center">
  <h1 class="entry-title" id="Comm">Communicate boundaries</h1>
</header>
<hr/>  
<p>
To get the boundary values from neighboring tasks in the extra slices we must call:
<div style="text-align: center">
  <div style="width:600px;display: inline-block;text-align: left">
    <pre class="prettyprint">
grid.communicate_boundaries();
</pre>
</div>
</div>
</p>

<header class="entry-header text-center">
  <h1 class="entry-title" id="Arithmetic">Arithmetic and converting to FFTWGrid</h1>
</header>
<hr/>  

<p>
The grids can be added, subtracted, multiplied and divided cell by cell (and by a scalar), and we can compute a (finite difference) gradient and the norm:
<div style="text-align: center">
  <div style="width:600px;display: inline-block;text-align: left">
    <pre class="prettyprint">
grid += other_grid;   // also -=, *=, /=
grid *= 2.0;          // also +=, -=, /= with a scalar
auto gradient = grid.get_gradient(index); // std::array of the gradient in the cell
double norm = grid.norm();
</pre>
</div>
</div>
To go between an MPIGrid and an FFTWGrid (e.g. to Fourier transform it) see ConvertMPIGridFFTWGrid.h.
</p>
