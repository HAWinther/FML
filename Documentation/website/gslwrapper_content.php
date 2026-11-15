<hr/>
<header class="entry-header text-center">
  <h1 class="entry-title">Table of contents</h1>
</header>
<hr/>

<p class="page-toc">
<a href="index.php">Back to the main page</a><br>
<a href="documentation.php">Back to the documentation</a><br>
<a href="#Overview">1. Overview: GSL wrappers</a><br>
<a href="#ODESolver">2. ODESolver</a><br>
<a href="#Spline">3. Spline</a><br>
</p>

<header class="entry-header text-center">
  <h1 class="entry-title" id="Overview">Overview: GSL wrappers</h1>
</header>
<hr/>  

<p>
We provide some simple wrappers on the <a href="https://www.gnu.org/software/gsl/" style="display: inline">GSL</a> library for solving ODEs (namespace FML::SOLVERS::ODESOLVER in ODESolver/ODESolver.h) and making splines (namespace FML::INTERPOLATION::SPLINE in Spline/Spline.h). The classes take care of all the memory handling and provide a simple interface for performing the operations. These require compiling with USE_GSL.
</p>

<header class="entry-header text-center">
  <h1 class="entry-title" id="ODESolver">ODESolver</h1>
</header>
<hr/>  

<p>
Solve coupled ODEs by providing the right hand side as a lambda (std::function):
<div style="text-align: center">
  <div style="width:600px;display: inline-block;text-align: left">
    <pre class="prettyprint">
using namespace FML::SOLVERS::ODESOLVER;

// The RHS of the ODE y'' + y = 0 written as y0' = y1 ; y1' = -y0
ODEFunction dydx = [&amp;](double x, const double* y, double* dydx) {
  dydx[0] =  y[1];
  dydx[1] = -y[0];
  return GSL_SUCCESS;
};

// x-values we store the points at
DVector x_array = FML::MATH::linspace(xmin, xmax, npts);

// Initial value
DVector y_ini{1.0, 0.0};

// Solve the ODE over the points in x_array
ODESolver ode;
ode.solve(dydx, x_array, y_ini);

// You can also choose the method if you want
// ode.solve(dydx, x_array, y_ini, gsl_odeiv2_step_rk2);

// Get the result (y and y' at all the points in x_array)
auto y_array  = ode.get_data_by_component(0);
auto dy_array = ode.get_derivative_data_by_component(0);

// The accuracy can be changed with (first step size, absolute and relative error)
// ode.set_accuracy(1e-3, 1e-10, 1e-10);
</pre>
</div>
</div>
</p>
<p>
Some methods require the Jacobian of the system to be provided, that is done in the same way (it's the last argument to solve). See the header file for more info.
</p>

<header class="entry-header text-center">
  <h1 class="entry-title" id="Spline">Spline</h1>
</header>
<hr/> 
 
<p>
Make a simple cubic (customizable) spline from arrays. Lookups are thread-safe as long as it's not created inside a thread.
<div style="text-align: center">
  <div style="width:600px;display: inline-block;text-align: left">
    <pre class="prettyprint">
// Make a spline
DVector x_array = FML::MATH::linspace(xmin, xmax, npts);
DVector y_array = FML::MATH::linspace(xmin, xmax, npts);
Spline spline(x_array, y_array, "My Spline");

// Evaluate f(0) and its derivatives
double value = spline(0.0);
double deriv = spline.deriv_x(0.0);
double deriv2 = spline.deriv_xx(0.0);
</pre>
</div>
</div>
</p>
<p>
You can get warnings if you evaluate it out of bounds (call set_out_of_bounds_warning(true)).
</p>

<p>
Make a 2D spline from arrays. Lookups are thread-safe as long as it's not created inside a thread.
<div style="text-align: center">
  <div style="width:600px;display: inline-block;text-align: left">
    <pre class="prettyprint">
// Make a spline
DVector x_array   = FML::MATH::linspace(xmin, xmax, npts);
DVector y_array   = FML::MATH::linspace(xmin, xmax, npts);
DVector2D z_array(npts, DVector(npts));

// ... fill it: z_array[ix][iy] = f(x_array[ix], y_array[iy]) ...

Spline2D spline(x_array, y_array, z_array, "My Spline");

// Evaluate f(0,0)
double value = spline(0.0, 0.0);
</pre>
</div>
</div>
</p>
<p>
You can get warnings if you evaluate it out of bounds. You can also choose the interpolation method, see the header file and the GSL documentation for more info.
</p>
