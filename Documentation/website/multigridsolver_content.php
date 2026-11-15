<hr/>
<header class="entry-header text-center">
  <h1 class="entry-title">Table of contents</h1>
</header>
<hr/>

<p class="page-toc">
<a href="index.php">Back to the main page</a><br>
<a href="documentation.php">Back to the documentation</a><br>
<a href="#Overview">1. Overview: Multigrid solver</a><br>
<a href="#Example">2. Example: Poisson equation</a><br>
<a href="#Methods">3. Methods to help defining the equation</a><br>
<a href="#Examples">4. More examples</a><br>
<a href="#Status">5. Convergence status</a><br>
</p>

<header class="entry-header text-center">
  <h1 class="entry-title" id="Overview">Overview: Multigrid solver</h1>
</header>
<hr/>  

<p>
This is a multigrid solver for very general PDEs: it can handle any type (combination of types), scalar, vectors whatever. The only restriction is that the solver type class is a ring type (i.e. it implements all the arithmetic operations +,-,* and multiplication/division with a scalar). The solver is in the namespace FML::SOLVERS::MULTIGRIDSOLVER (MultigridSolver/MultiGridSolver.h). What the solver does is to first make a guess for the solution and then iterate over the grid and update the solution by some root finding method (we use Newton's method). To improve the convergence we apply the multigrid technique: the solution is restricted down to a coarser grid and we repeat the steps above. We do this recursively down to the coarsest level (typically $2$ cells per dimension). Then we restrict the solution up and use this to correct the solution at the finer level and do this recursively up to the finest grid. One such step is called a V-cycle. We repeat this until convergence. Between every sweep of the grid where boundary cells are modified a communication is performed. For more info about how the multigrid solver works see for example Section 3.2 of <a href="https://arxiv.org/abs/1506.06384" style="display: inline">Winther et al. 2015</a>.
</p>
<p>
It's hairy doing this in parallel if we allow any number of gridcells. The solution is that we place a few restrictions on the user: the number of tasks $N_{\rm Tasks}$ must divide the gridsize $N_{\rm grid}$ and $\frac{N_{\rm grid}}{N_{\rm Tasks}}$ must be divisible by $2^M$ where $M$ is the number of levels we need. These restrictions (which are not very restrictive) are enough to ensure that there are no coarse cells split between tasks and we don't have to do any communication when restricting down and prolonging up between levels. The coarsest level we can have is one where there is $1$ cell per MPI task. If we have a large number of tasks then the convergence will not be as fast as it would be if we could go all the way down. If this is an issue then use fewer tasks if possible (and compensate with more threads, i.e. use a hybrid OpenMP + MPI).
</p>

<p align="center">
Figure 1: Sketch of how the multigrid solver works.
<p align="center">
<img src="images/multigrid.gif">
</p>
</p>

<p>
<b>Slabs and pencils.</b> The solver runs on the domain of its grid: slabs, or pencils for a field from a <a href="pencilgrid.php" style="display: inline">$\texttt{PencilGrid}$</a> (all the modified-gravity solvers of the N-body code, also with <tt>make USE_PENCIL_GRID=true</tt>). The 27-cell stencil that equations with mixed derivatives use (e.g. DGP) is built from per-axis offsets, so meshes that are not a power of two are as fast as the power-of-two fast path.
</p>

<header class="entry-header text-center">
  <h1 class="entry-title" id="Example">Example: Poisson equation</h1>
</header>
<hr/> 

<p>
Below is a simple example for how to solve the PDE
$$\nabla^2 \phi(x) = S(x)$$
We write this as $\mathcal{L} = \nabla^2\phi - S$ thus we want to find $\phi$ such that $\mathcal{L} = 0$. The equation can be discretized (here in 1D for simplicity) as
$$\mathcal{L}_i = \frac{\phi_{i+1} + \phi_{i-1} - 2\phi_{i}}{h^2} - S_i$$
We also need the derivative of this function
$$\frac{d\mathcal{L}}{d\phi_i} = -\frac{2}{h^2}$$
In general dimensions this would be a sum over the $xx$, $yy$, etc. derivatives and the derivative of $\mathcal{L}$ would be $\frac{-2{\rm NDIM}}{h^2}$. These two functions are all we need to implement in the solver.
</p> 

<p>
Set up the solver:
<div style="text-align: center">
  <div style="width:600px;display: inline-block;text-align: left">
    <pre class="prettyprint">
using namespace FML::SOLVERS::MULTIGRIDSOLVER;

// Solver parameters
using SolverType     = double;  // The type of phi 
const int N          = 128;     // Number of cells
const int Nlevels    = -1;      // Number of multigrid levels (-1 lets the solver pick this)
const int NDIM       = 2;       // Dimension we are working in
const int nleft      = 1;       // Extra slices (we need at least 1 for derivatives at the boundary)
const int nright     = 1;       //  --- || ---
const bool periodic  = true;    // Periodic box?
const bool verbose   = true;    

MultiGridSolver&lt;NDIM,SolverType&gt; mg(N, Nlevels, verbose, periodic, nleft, nright);

// Optional: the convergence criterion and the number of sweeps
mg.set_epsilon(1e-7);
mg.set_maxsteps(20);
</pre>
</div>
</div>
</p>

<p>
Define the equation and the convergence criterion you want:
<div style="text-align: center">
  <div style="width:600px;display: inline-block;text-align: left">
    <pre class="prettyprint">
// The right hand side S: this is here just D^2 of phi = Prod_i sin(2 pi x_i)
auto source = [=](std::array&lt;double,NDIM&gt; &amp; x) -&gt; SolverType {
    SolverType sol = (-4.0 * M_PI * M_PI);
    for(int idim = 0; idim &lt; NDIM; idim++)
      sol *= sin(2.0 * M_PI * x[idim]);
    return sol;
};

// Define the equation: we here compute and return L and dLdy
// This function gets called for every cell and index is the index of the given cell
MultiGridFunction&lt;NDIM,SolverType&gt; Equation = [&amp;](MultiGridSolver&lt;NDIM,SolverType&gt; *sol, int level, IndexInt index){
  auto index_list = sol-&gt;get_neighbor_gridindex(level, index);
  auto coordinate = sol-&gt;get_Coordinate(level, index);
  auto L  = sol-&gt;get_Laplacian(level, index_list) - source(coordinate);
  auto dL = sol-&gt;get_derivLaplacian(level, index_list);
  return std::pair&lt;SolverType,SolverType&gt;{L, dL};
};

// Set the convergence criterion: here we just use the fiducial choice 
// (ConvergenceCriterionResidual is rms-residual &lt; epsilon)
MultiGridConvCrit ConvergenceCriterion = [&amp;](double rms_residual, double rms_residual_ini, int step_number){
  return mg.ConvergenceCriterionResidual(rms_residual, rms_residual_ini, step_number);
};
</pre>
</div>
</div>
</p>

<p>
Set the initial guess, run the solver and fetch the solution:
<div style="text-align: center">
  <div style="width:600px;display: inline-block;text-align: left">
    <pre class="prettyprint">
// Set the initial guess
mg.set_initial_guess( SolverType(0.0) );

// Run the solver
mg.solve(Equation, ConvergenceCriterion);

// Get the solution (an MPIGrid)
auto sol = mg.get_grid(0);
</pre>
</div>
</div>
</p>

<p>
In this example it's very simple as we could use built in functions to define the equation. With the given source the solution should just be $\phi(x_1,\ldots) = \prod_{i=1}^{\rm NDIM} \sin(2\pi x_i)$.
</p>

<header class="entry-header text-center">
  <h1 class="entry-title" id="Methods">Methods to help defining the equation</h1>
</header>
<hr/> 

<p>
We provide some functions to make it easier to define the equation. If this is not enough then you will have to do it yourself. There is much more information in the header file defining the solver.
<div style="text-align: center">
  <div style="width:600px;display: inline-block;text-align: left">
    <pre class="prettyprint">
// The solution in a given cell
T get_Field(int level, IndexInt index);

// Position of the cell in the global grid
std::array&lt;double,NDIM&gt;   get_Coordinate(int level, IndexInt index);

// The closest 2NDIM cells (plus the cell itself). This is what the methods below require
std::array&lt;IndexInt,2*NDIM+1&gt; get_neighbor_gridindex(int level, IndexInt index);

// The gradient Df and its derivative
std::array&lt;T,NDIM&gt;  get_Gradient     (int level, const std::array&lt;IndexInt,2*NDIM+1&gt; &amp; nbor_index_list);
std::array&lt;T,NDIM&gt;  get_derivGradient(int level, const std::array&lt;IndexInt,2*NDIM+1&gt; &amp; nbor_index_list);

// The Laplacian operator D^2f and its derivative
T get_Laplacian     (int level, const std::array&lt;IndexInt,2*NDIM+1&gt; &amp; nbor_index_list);
T get_derivLaplacian(int level, const std::array&lt;IndexInt,2*NDIM+1&gt; &amp; nbor_index_list);

// Higher order gradients df/dx_i and d^2f/dx_i^2 with accuracy O(h^(2 order))
std::array&lt;T,NDIM&gt; get_Gradient (int level, IndexInt index, int order);
std::array&lt;T,NDIM&gt; get_Gradient2(int level, IndexInt index, int order);

// If you need even more cells: get all 3^NDIM cells around a given cell
std::array&lt;IndexInt, FML::power(3,NDIM) &gt; get_cube_gridindex(int level, IndexInt index);
</pre>
</div>
</div>
</p>

<header class="entry-header text-center">
  <h1 class="entry-title" id="Examples">More examples</h1>
</header>
<hr/> 

<p>
In the <a href="https://github.com/HAWinther/FML/tree/master/FML/MultigridSolver/examples" style="display: inline">source code on GitHub</a> we provide a lot more examples (and ready made solvers for $f(R)$, the symmetron and DGP in FofrSolver.h, SymmetronSolver.h and DGPSolver.h) like:

Poisson equation:
$$\nabla^2\phi = \delta$$

Density reconstruction equation for the displacement field $\Psi$:
$$\nabla\cdot \Psi + \frac{f}{b}\nabla\cdot[(\Psi\cdot r)r] = -\frac{\delta}{b}$$

The symmetron model equation:
$$\nabla^2\phi = \frac{1}{2}\frac{B^2}{L^2}\left[ (1+\delta)\left(\frac{a_{\rm ssb}}{a}\right)^3\phi - \phi + \phi^3\right]$$

Hu-Sawicky $f(R)$ gravity equation:
$$\nabla^2 f_R = A\delta + B \frac{1}{f_R^{1/(n+1)}}$$

Quasi-linear continuity equation for the velocity field:
$$fH\delta + \nabla\cdot((1+\delta)\vec{v}) = 0$$

DGP equation:
$$\nabla^2 \phi + A[(\nabla^2\phi)^2 - (\nabla_i\nabla_j\phi)^2] = B\delta$$

And the cubic variant:
$$(\nabla^2 \phi)^3 + p (\nabla^2 \phi) + q = 0$$

And examples of how to use your own types with the solver.
</p>

<header class="entry-header text-center">
  <h1 class="entry-title" id="Status">Convergence status</h1>
</header>
<hr/>

<p>
<tt>solve(equation, criterion)</tt> returns a <tt>SolveResult</tt> with the status (<tt>Converged</tt>, <tt>MaxIterations</tt> or <tt>NonFinite</tt>), the final and initial RMS residuals and the number of V-cycles. A non-finite residual on any level stops the solve at the next check instead of running to the iteration limit, and convergence is decided collectively over the MPI tasks.
</p>
<div style="text-align: center">
  <div style="width:600px;display: inline-block;text-align: left">
    <pre class="prettyprint">
auto result = solver.solve(equation, criterion);
if (result.status != FML::SOLVERS::MULTIGRIDSOLVER::SolveStatus::Converged) {
  // handle MaxIterations / NonFinite before using solver.get_grid()
}
</pre>
</div>
</div>
