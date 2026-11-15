<hr/>
<header class="entry-header text-center">
  <h1 class="entry-title">Table of contents</h1>
</header>
<hr/>

<p class="page-toc">
<a href="index.php">Back to the main page</a><br>
<a href="documentation.php">Back to the documentation</a><br>
<a href="#Overview">1. Overview: Utils</a><br>
<a href="#Timing">2. Timing</a><br>
<a href="#Units">3. Units</a><br>
<a href="#Plot">4. Plotting</a><br>
<a href="#Visualization">4b. Pictures without Python (Visualization)</a><br>
<a href="#Math">5. Math</a><br>
<a href="#Wrap">6. Making Python wrapper</a><br>
<a href="#KDTree">7. KD-tree</a><br>
<a href="#PGD">8. Potential gradient descent (PGD)</a><br>
<a href="#Performance">9. Performance cases and counters</a><br>
<a href="#Hilbert">Standalone Peano–Hilbert domain decomposition</a><br>
</p>

<header class="entry-header text-center">
  <h1 class="entry-title" id="Overview">Overview: Utils</h1>
</header>
<hr/>  

<p>
Here we provide some random stuff that is useful to have.
</p>

<header class="entry-header text-center">
  <h1 class="entry-title" id="Timing">Timing</h1>
</header>
<hr/>  

<p>
We provide a simple interface for timing the code (<code>FML/Timing/Timings.h</code>):
<div style="text-align: center">
  <div style="width:600px;display: inline-block;text-align: left">
    <pre class="prettyprint">
// Set up timer
FML::UTILS::Timings t;

// Make a timing
t.StartTiming("Label");
// ...
t.EndTiming("Label");

// New timing using same label = time adds up
t.StartTiming("Label");
// ...
t.EndTiming("Label");

// Make a timing and print the result straight away (EndTiming also returns the time in seconds)
t.StartTiming("New label");
// ...
double seconds = t.EndTiming("New label", true);

// ...

// Print all the timings we have made
t.PrintAllTimings();
</pre>
</div>
</div>
</p>

<header class="entry-header text-center">
  <h1 class="entry-title" id="Units">Units</h1>
</header>
<hr/>  

<p>
A simple class (<code>FML/Units/Units.h</code>) for dealing with arbitrary physical units. You can either choose between common units like SI, Planck, ParticlePhysics, Cosmology or define your own units by specifying the length, mass, time, temperature and charge of your unit in SI. It also keeps common physical constants and provides these in the units you have chosen.
</p>

<p>
The class contains 5 numbers defining the units:
<div style="text-align: center">
  <div style="width:600px;display: inline-block;text-align: left">
    <pre class="prettyprint">
double m   = 1.0;  // Size of the unit length in meters
double s   = 1.0;  // Size of the unit time in seconds
double kg  = 1.0;  // Size of the unit mass in kg
double K   = 1.0;  // Size of the unit temperature in Kelvin
double Co  = 1.0;  // Size of the unit charge in Coulomb
</pre>
</div>
</div>
</p>

<p>
And contains the following physical constants and units:
<div style="text-align: center">
  <div style="width:600px;display: inline-block;text-align: left">
    <pre class="prettyprint">
// Physical constants
double k_b;             // Boltzmann constant
double k_e;             // Coulomb constant
double G;               // Gravitational constant
double hbar;            // Plancks constant  
double c;               // Speed of light
double m_e;             // Mass of electron
double m_H;             // Mass of hydrogen
double sigma_T;         // Thomson cross section
double lambda_2s1s;     // Decay time 2s to 1s
double epsilon_0;       // Binding energy of hydrogen groundstate
double H0_over_h;       // Hubble constant without h: 100km/s/Mpc
double xhi0;            // Helium ground state
double xhi1;            // Helium first excited state

// Derived units
double yr;              // 1 year
double Gyr;             // 1 gigayear
double mm;              // 1 millimeter
double cm;              // 1 centimeter
double km;              // 1 kilometer
double N;               // 1 Newton
double J;               // 1 Joule
double W;               // 1 Watt
double eV;              // 1 electronvolt
double MeV;             // 1 megaelectronvolt
double Mpc;             // 1 Megaparsec
double kpc;             // 1 kiloparsec
double Gpc;             // 1 Gigaparsec
double Msun;            // 1 Solar mass
double velocity;        // Unit length over unit time
double density;         // Unit mass over unit volume
</pre>
</div>
</div>
</p>

<p>
Example use:
<div style="text-align: center">
  <div style="width:600px;display: inline-block;text-align: left">
    <pre class="prettyprint">
// Standard SI units
auto SI = FML::UTILS::ConstantsAndUnits("SI");
SI.info();

// Electron mass energy me c^2
double emass_in_eV = SI.m_e * SI.c * SI.c / SI.eV;

// Planck units
auto Planck = FML::UTILS::ConstantsAndUnits("Planck");
Planck.info();

// How many Planck lengths in a Mpc
auto planck_length_in_mpc = Planck.Mpc / Planck.m;
</pre>
</div>
</div>
</p>

<header class="entry-header text-center">
  <h1 class="entry-title" id="Plot">Plotting</h1>
</header>
<hr/>  

<p>
It's possible to produce plots directly from C++ using <a href="https://github.com/lava/matplotlib-cpp" style="display: inline">matplotlib-cpp</a> which is included in the library (<code>FML/Plotting/Matplotlibcpp.h</code>, compile with USE_PYTHON):
<div style="text-align: center">
  <div style="width:600px;display: inline-block;text-align: left">
    <pre class="prettyprint">
  namespace plt = FML::UTILS::Matplotlib;

  // Call matplotlib
  plt::xlim(0.0, 1.0);
  plt::plot(x_array, y_array);
  plt::title("y = 2 + sin(2 pi x)");

  // Save plot to file
  plt::save("plot.png");
</pre>
</div>
</div>
</p>

<header class="entry-header text-center">
  <h1 class="entry-title" id="Visualization">Pictures without Python (Visualization)</h1>
</header>
<hr/>

<p>
<tt>FML/Visualization</tt> makes pictures of fields and particles straight from C++ with no dependencies, for a quick look at a run or the frames of a movie. <tt>project_grid</tt> projects a slab of an FFTWGrid or PencilGrid along any axis (the mean or the sum over the slab, which may wrap around the box) and <tt>project_particles</tt> the particles themselves (NGP or CIC, mass weighted); every task gets the same image. Colour maps: magma, inferno, viridis, plasma, cividis, coolwarm, grey and cubehelix (add <tt>_r</tt> to reverse), on a linear, log or symmetric scale with fixed limits or percentiles. The PNG writer has its own deflate, and <tt>write_npy</tt> writes the values for numpy. SimulationCode and SnapshotAnalyzer use it for their <a href="nbodycode.php#Analysis" style="display: inline">images</a>.
</p>
<pre class="prettyprint">
#include &lt;FML/Visualization/Visualization.h&gt;
using namespace FML::VISUALIZATION;

Image image = project_grid(density, 2, 0.45, 0.1);   // a slab of 10% of the box, seen along z
if (FML::ThisTask == 0)
    write_png("density.png", image, "magma", ColorScale::automatic_log());
</pre>

<header class="entry-header text-center">
  <h1 class="entry-title" id="Math">Math</h1>
</header>
<hr/>  

<p>
Some useful math functions (<code>FML/Math/Math.h</code>, namespace <code>FML::MATH</code>):
<div style="text-align: center">
  <div style="width:600px;display: inline-block;text-align: left">
    <pre class="prettyprint">
// Python linspace. Generate a linearly spaced array
DVector linspace(double xmin, double xmax, int num);

// Find roots f(x) = 0 
double find_root_bisection(
    const std::function&lt;double(double)&gt;&amp; function,
    std::pair&lt;double, double&gt; xrange,
    double epsilon = 1e-7);

// Airy function (first time called it generates a spline that is used later on)
double Airy_Ai(double z);

// Hyperspherical bessel functions using recursion formula
DVector Hyperspherical_j_ell_array(int lmax, const double nu, const double chi, double K);

// Spherical bessel functions using recursion formula
DVector j_ell_array(int lmax, const double x);

//  Spherical bessel functions from CXX or GSL with fix for very small or large arguments
double j_ell(const int ell, const double arg);

// General method for evaluating continued fraction. Gives back the result and if it converged
std::pair&lt;double,bool&gt; GeneralizedLentzMethod(
    const std::function&lt;double(int)&gt; &amp;a,
    const std::function&lt;double(int)&gt; &amp;b,
    double epsilon,
    int maxsteps);

// WKB approximation for the hyperspherical bessel functions (for a curved Universe)
// For a flat Universe call with nu = 1.0, chi = k*(eta0-eta) and K = 0.0
double HyperSphericalBesselWKB(int ell, double nu, double chi, double K);
</pre>
</div>
</div>

</p>

<header class="entry-header text-center">
  <h1 class="entry-title" id="Wrap">Making Python wrapper</h1>
</header>
<hr/>  

<p>
If you for some reason want to call the library (or any other C++ code) from Python there is an example of how to do this in the <a href="https://github.com/HAWinther/FML/tree/master/FML/MakePythonWrapper" style="display: inline">MakePythonWrapper</a> folder (and HaloModel/PythonWrapper). This is really easy with <a href="https://www.swig.org/" style="display: inline">SWIG</a>: one just has to define a simple interface, compile the C++ code as a shared library, run SWIG and that's it, you can now run C++ code and pass the results back to Python (or the other way around). This works well with threads, but I have never used it with MPI so have no idea how that would work.
</p>

<header class="entry-header text-center">
  <h1 class="entry-title" id="KDTree">KD-tree</h1>
</header>
<hr/>

<p>
<tt>KDTree/KDTree.h</tt> is a kd-tree for nearest-neighbour, $k$-nearest-neighbour and range searches in any dimension, optionally in a periodic box (minimum-image distances, each point returned once for any radius). The queries are const and can run in parallel; it is used by the <a href="statistics.php" style="display: inline">kNN and counts-in-cells</a> statistics, the density-split clustering and baryonification.
</p>

<header class="entry-header text-center">
  <h1 class="entry-title" id="PGD">Potential gradient descent (PGD)</h1>
</header>
<hr/>

<p>
PGD (Dai, Feng &amp; Seljak 2018) sharpens the halos of a cheap PM/COLA simulation by moving the particles towards the minima of a filtered potential,
$$\vec S = -\alpha\,{\rm norm}\,\nabla\hat O_l\hat O_s\nabla^{-2}\delta,\qquad \hat O_l = e^{-k_l^2/k^2},\quad \hat O_s = e^{-k^4/k_s^4},$$
with the fitted $\alpha(a)$ (<tt>pgd_fitting_formula_alpha</tt>), see <tt>PGD/PotentialGradientDecent.h</tt>.
</p>

<header class="entry-header text-center">
  <h1 class="entry-title" id="Performance">Performance cases and counters</h1>
</header>
<hr/>

<p>
<tt>FML/Performance</tt> is a suite of reproducible benchmark cases of representative operations (FFT round trips, grid copies, Fourier resampling, particle migration and polyspectra) with explicit provenance: <tt>make -C FML benchmark</tt> and <tt>python3 FML/Performance/run.py --exe ... --ranks 1 2 4 --threads 1 2</tt> run every case in a fresh process for each rank/thread combination and record JSON reports, raw logs, the build and host information. Opt-in counters (<tt>FML_ENABLE_PERFORMANCE_COUNTERS</tt>) record FFTs, grid allocations and traced communication payloads alongside wall time and RSS; ordinary builds are unaffected. The reports embed the compiler, flags, library versions, repository revision and a hash of the sources. Python users: see the <a href="python.php" style="display: inline">Python page</a>.
</p>

<header class="entry-header text-center">
  <h1 class="entry-title" id="Hilbert">Peano–Hilbert domain decomposition</h1>
</header>
<hr/>

<p><tt>FML/PeanoHilbertDomain</tt> maps periodic positions to Hilbert keys (<tt>HilbertCurve&lt;NDIM&gt;</tt>, up to 21 bits per coordinate in 3D), partitions a distributed prefix histogram into contiguous key intervals (<tt>PeanoHilbertDomain</tt>), and stores a distributed grid in local key order. <tt>DistributedGrid3D</tt> supports collective remote reads and face/full periodic halo exchange, including empty rank intervals. The same keys are used by the <a href="amr.php" style="display: inline">adaptive grid</a> and by the particle domain <tt>FML::Domain::peano_hilbert(...)</tt>, so <a href="mpiparticles.php" style="display: inline">MPIParticles</a> can be distributed along the curve; the density of such particles is assigned to slab or pencil grids with <a href="interpol.php#AnyDomain" style="display: inline">particles_to_grid_any_domain</a>.</p>
<p>It is not an FFT layout, and SimulationCode does not use it. Strongly clustered identical keys cannot be split among ranks. See <a href="https://github.com/HAWinther/FML/blob/master/FML/PeanoHilbertDomain/README.md" style="display: inline">the standalone API and tests</a>.</p>

