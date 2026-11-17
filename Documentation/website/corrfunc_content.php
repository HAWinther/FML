<hr/>
<header class="entry-header text-center">
  <h1 class="entry-title">Table of contents</h1>
</header>
<hr/>

<p class="page-toc">
<a href="index.php">Back to the main page</a><br>
<a href="documentation.php">Back to the documentation</a><br>
<a href="#Overview">1. Overview: Correlation functions</a><br>
<a href="#Pair">2. Paircounting</a><br>
<a href="#Pofk">3. Power-spectra</a><br>
<a href="#Multipoles">4. Power-spectrum multipoles</a><br>
<a href="#FFTLog">5. Hankel transforms (FFTLog)</a><br>
<a href="#Bi">6. Bispectrum</a><br>
<a href="#Poly">7. General polyspectra</a><br>
<a href="#PolyNew">8. Polyspectra: dealiasing, selected configurations, sparse results and memory</a><br>
<a href="#Related">9. Related pages: survey multipoles, matched spectra, hybrid bias</a><br>
<a href="#References">Method references and scope</a><br>
<a href="#Connected">Connected four-point statistics</a><br>
</p>

<header class="entry-header text-center">
  <h1 class="entry-title" id="Overview">Overview: Correlation functions</h1>
</header>
<hr/>  

<p>
We provide methods for computing correlation functions in real and Fourier space: the two point correlation function in real space (for simulation boxes and for surveys) and $N$-point correlation functions in Fourier space (power-spectrum, bispectrum, trispectrum, ...). What is missing here and would be great to have is the survey versions of the Fourier space methods (which is straightforward, but just needs to be done). The Fourier space methods live in the namespace FML::CORRELATIONFUNCTIONS and are found in ComputePowerSpectra/ComputePowerSpectrum.h.
</p>

<header class="entry-header text-center">
  <h1 class="entry-title" id="Pair">Paircounting</h1>
</header>
<hr/>  

<p>
Paircounting is not always the easiest thing to parallelize effectively unless you have shared memory. This stems from the fact that we in principle need to correlate each pair of particles which in general reside on different tasks which would lead to tons of communication. Our paircounting algorithm is parallelized with MPI and OpenMP, but with the limitation that using MPI all tasks need to have all the same particles to prevent having to do this communication! The way we do it is to make a grid, bin the particles to the grid and then correlate particles inside these boxes with neighboring boxes. This means we only consider the pairs we need to consider and not pairs that are separated by distances larger than the maximum radius we want. With MPI each task is responsible for correlating a part of the grid. The pair counting is done by a general method where the user supplies the binning function - the function that gets called for each pair and adds up the result - so the algorithm itself does not have to be modified if one wants to bin up a quantity other than just how many pairs we have. See PairCounting/example for examples of how to use it.
</p>

<p>
All the pair counting lives in the namespace FML::CORRELATIONFUNCTIONS::PAIRCOUNTS (PairCounting/PairCount.h). We have methods for the radial correlation function $\xi(r)$ and for the angular one $\xi(r,\mu)$ in both a simulation box and for survey data (with randoms). The particles must have get_pos(), get_ndim() and get_weight() (just return 1.0 if you don't have weights). Note that with MPI all tasks need to have <i>all</i> the particles. For a simulation box (auto pair counts; there is also a version taking two sets of particles for cross pair counts):
<div style="text-align: center">
  <div style="width:600px;display: inline-block;text-align: left">
    <pre class="prettyprint">
using DVector = std::vector&lt;double&gt;;

// xi(r). Normalized paircounts means DD / total number of pairs
template &lt;typename T1&gt;
  void RadialCorrelationFunctionBox(
      const T1 * part1,
      size_t npart1,
      double rmin,
      double rmax,
      int nrbins,
      DVector &amp; r_array,
      DVector &amp; paircounts_array,
      bool normalize_paircounts,
      DVector &amp; corr_func_array,
      bool periodic_box,
      bool verbose);

// xi(r,mu). The angle mu is defined wrt the line of sight from the observer
// to the midpoint of the pair. Put the observer at (0,0,-effective_infinity)
// to get a fixed line of sight direction along the z-axis
template &lt;typename T1&gt;
  void AngularCorrelationFunctionBox(
      const T1 * part1,
      size_t npart1,
      double rmin,
      double rmax,
      int nrbins,
      double mumin,
      double mumax,
      int nmubins,
      DVector &amp; r_array,
      DVector &amp; mu_array,
      DVector2D &amp; paircounts_array,
      bool normalize_paircounts,
      DVector2D &amp; corr_func_array,
      DVector observer_position,
      bool periodic_box,
      bool verbose);
</pre>
</div>
</div>
</p>

<p>
For survey data we also need randoms and we compute all of DD, DR, RD and RR (the first and second set of particles can be the same for auto correlations):
<div style="text-align: center">
  <div style="width:600px;display: inline-block;text-align: left">
    <pre class="prettyprint">
template &lt;typename T1, typename T2 = T1, typename R1 = T1, typename R2 = T2&gt;
  void RadialCorrelationFunctionSurvey(
      const T1 * part1, size_t npart1,
      const R1 * rand1, size_t nrand1,
      const T2 * part2, size_t npart2,
      const R2 * rand2, size_t nrand2,
      double rmin,
      double rmax,
      int nrbins,
      DVector &amp; r_array,
      DVector &amp; paircounts_D1D2_array,
      DVector &amp; paircounts_D1R2_array,
      DVector &amp; paircounts_R1D2_array,
      DVector &amp; paircounts_R1R2_array,
      DVector &amp; corr_func_array,
      DVector observer_position,
      bool periodic_box,
      bool verbose);

// ... and AngularCorrelationFunctionSurvey for xi(r,mu)

// From xi(r,mu) to the multipoles xi_ell(r) (needs GSL)
void FromAngularCorrelationToMultipoles(
    const DVector &amp; r_array,
    const Spline2D &amp; xi_mu_r_spline,
    std::vector&lt;DVector&gt; &amp; multipoles,
    double mumin = -1.0,
    double mumax = 1.0);
</pre>
</div>
</div>
</p>

<p>
All of the methods above take the optional arguments nextratobin, extra_quantities_array and extra_quantities_to_bin. With these you can bin up any other quantity over the pairs (e.g. the pairwise velocity $v_{12}$) by providing a function that is called for every pair:
<div style="text-align: center">
  <div style="width:600px;display: inline-block;text-align: left">
    <pre class="prettyprint">
// dr is the separation vector, r its length and mu the angle wrt the line of sight
// Add the nextratobin quantities you want to bin up to storage[0], storage[1], ...
template &lt;typename T1, typename T2 = T1&gt;
using ExtraQuantitiesToBinFunction = std::function&lt;void(
    const double * dr, double r, double mu,
    const T1 &amp; part1, const T2 &amp; part2,
    double * storage)&gt;;

// For complete freedom: the general pair counter that calls binning_function
// for every pair with separation less than rmax
template &lt;class T, class U&gt;
  void GeneralPairCounter(
      FML::PARTICLE::ParticlesInBoxes&lt;T&gt; &amp; grid,
      FML::PARTICLE::ParticlesInBoxes&lt;U&gt; &amp; grid2,
      const std::function&lt;void(int, double *, T &amp;, U &amp;)&gt; &amp; binning_function,
      bool auto_pair_binning,
      double rmax,
      bool periodic_box,
      bool verbose = false);
</pre>
</div>
</div>
</p>

<p>
For survey data we provide the common estimators for computing the correlation function from (normalized) paircounts: LZ (<a href="https://doi.org/10.1086/172900" style="display: inline">Landy &amp; Szalay 1993</a>), HAM (<a href="https://doi.org/10.1086/173288" style="display: inline">Hamilton 1993</a>), HEW (Hewett 1982), DP (<a href="https://doi.org/10.1086/160884" style="display: inline">Davis &amp; Peebles 1983</a>) and PH (Peebles &amp; Hauser 1974). See the estimator comparison by <a href="https://arxiv.org/abs/astro-ph/9912088" style="display: inline">Kerscher, Szapudi &amp; Szalay (2000)</a> and <a href="https://doi.org/10.1051/0004-6361/201220790" style="display: inline">Landy &amp; Szalay (2013)</a>. For cross correlations where we only have randoms for one of the sets there are also the variants HAM_no_R1, HAM_no_R2, HEW_no_R1, HEW_no_R2, DP_no_R1 and DP_no_R2. See <a href="https://doi.org/10.1111/j.1365-2966.2006.10158.x" style="display: inline">Blake et al. 2006</a> for more info. The survey methods above use LZ.
<div style="text-align: center">
  <div style="width:600px;display: inline-block;text-align: left">
    <pre class="prettyprint">
double CorrelationFunctionEstimator(
    double D1D2,
    double D1R2,
    double R1D2,
    double R1R2,
    const std::string &amp; estimator);
</pre>
</div>
</div>
</p>

<header class="entry-header text-center">
  <h1 class="entry-title" id="Pofk">Power-spectrum</h1>
</header>
<hr/>

<figure class="cinematic-figure" id="cinematic-10-fourier-universe">
  <a href="images/cinematic/10-fourier-universe.jpg" aria-label="View full-size illustration: The Universe in Fourier space">
    <img src="images/cinematic/10-fourier-universe.jpg" width="1774" height="887" loading="lazy" decoding="async" alt="A real-space cosmic density field transitions to a Fourier-mode lattice with highlighted wavenumber shells">
  </a>
  <figcaption><strong>The Universe in Fourier space.</strong> Conceptual transition from real-space density to Fourier modes; shell colors denote wavenumber bands rather than measured power. <span class="cinematic-credit">AI-generated concept illustration.</span></figcaption>
</figure>  

<p>
The power-spectrum is just binning up the norm of the Fourier modes so this is easy to do. For parallelization we just need to merge the binnings on each task in the end. We also provide a method using interlaced grids to reduce the effect of aliasing. The usual brute-force way of dealing with this is just using a very large grid, but with interlacing we can get away with a much smaller grid and thereby saving computational time and memory. For computing power-spectra we provide the following methods:
<div style="text-align: center">
  <div style="width:600px;display: inline-block;text-align: left">
    <pre class="prettyprint">
// Bin up the power-spectrum of a given fourier grid
template&lt;int N&gt;
  void bin_up_power_spectrum(
      const FFTWGrid&lt;N&gt; &amp; fourier_grid,
      PowerSpectrumBinning&lt;N&gt; &amp; pofk);

// The same, but divide by the square of the window function of the density assignment
template&lt;int N&gt;
  void bin_up_deconvolved_power_spectrum(
      const FFTWGrid&lt;N&gt; &amp; fourier_grid,
      PowerSpectrumBinning&lt;N&gt; &amp; pofk,
      std::string density_assignment_method);

// Cross power-spectrum of two grids
template&lt;int N&gt;
  void bin_up_cross_power_spectrum(
      const FFTWGrid&lt;N&gt; &amp; fourier_grid_1,
      const FFTWGrid&lt;N&gt; &amp; fourier_grid_2,
      PowerSpectrumBinning&lt;N&gt; &amp; pofk);

// Assign particles to grid using density_assignment_method = NGP,CIC,TSC,PCS,...
// Fourier transform, deconvolve the window function for the assignment above,
// With interlacing we assign particles to an interlaced grid (displaced by dx/2 in all directions)
// Fourier transform both and add them together to cancel the leading aliasing contributions
// bin up power-spectrum and subtract shot-noise 1/NumPartTotal
// Set pofk.subtract_shotnoise = false if this is not wanted
template&lt;int N, class T&gt;
  void compute_power_spectrum(
      int Ngrid,
      T *part,
      size_t NumPart,
      size_t NumPartTotal,
      PowerSpectrumBinning&lt;N&gt; &amp; pofk,
      std::string density_assignment_method,
      bool interlacing);

// Brute force. Add particles to the grid using direct summation
// This gives alias free P(k), but scales as O(Npart)*O(Nmesh^N)
template&lt;int N, class T&gt;
  void compute_power_spectrum_direct_summation(
      int Ngrid,
      const T *part,
      size_t NumPart,
      PowerSpectrumBinning&lt;N&gt; &amp; pofk);
</pre>
</div>
</div>
</p>
<p>
See PowerSpectrumBinning.h for more info about how PowerSpectrumBinning is defined, but to get the power-spectrum data:
<div style="text-align: center">
  <div style="width:600px;display: inline-block;text-align: left">
    <pre class="prettyprint">
// nbins linearly spaced bins between kmin and kmax (in units of 1/Boxsize, i.e. the fundamental mode is 2pi)
PowerSpectrumBinning&lt;N&gt; pofk(kmin, kmax, nbins, PowerSpectrumBinning&lt;N&gt;::LINEAR_SPACING);

// ... compute it

// Put in physical units (scale k and P(k))
pofk.scale(boxsize);

// Fetch the arrays with k and P(k)
auto k  = pofk.k;    
auto pk = pofk.pofk;
</pre>
</div>
</div>
</p>
<p>
Direct summation above means computing the overdensity field as $\delta(k) = -1 + \frac{1}{N_{\rm part}}\sum_{i=1}^{N_{\rm part}} e^{-ik\cdot x_i}$. This gives us an alias-free estimate for the power-spectrum, however it's insanely slow as it scales as $N_{\rm grid}^{\rm NDIM} \cdot N_{\rm part}$. This is only really useful for testing.
</p>

<header class="entry-header text-center">
  <h1 class="entry-title" id="Multipoles">Power-spectrum multipoles</h1>
</header>
<hr/>  

<p>
Compute power-spectrum multipoles for when we have a fixed line of sight direction:
<div style="text-align: center">
  <div style="width:600px;display: inline-block;text-align: left">
    <pre class="prettyprint">
// Compute power-spectrum multipoles P0,P1,...,Pn-1 for the case
// where we have a fixed line_of_sight_direction (typical coordinate axes like (0,0,1))
// Pell contains P0,P1,P2,...Pell_max where ell_max = n-1 is the size of Pell
template&lt;int N&gt;
  void compute_power_spectrum_multipoles_fourier(
      const FFTWGrid&lt;N&gt; &amp; fourier_grid,
      std::vector&lt;PowerSpectrumBinning&lt;N&gt;&gt; &amp; Pell,
      std::vector&lt;double&gt; line_of_sight_direction);

// Simple method to estimate multipoles from simulation data
// Take particles in realspace and use their velocity to put them into
// redshift space. Fourier transform and compute multipoles from this like in the method above.
// We do this for all coordinate axes and return the mean P0,P1,... we get from this
// velocity_to_displacement is factor to convert your velocity to a coordinate shift in [0,1)
// e.g. c/(a H Box) with H ~ 100 h km/s/Mpc and Box boxsize in Mpc/h if velocities are peculiar
// (at z = 0 this is 1/(100 Box) for velocities in km/s)
// Subtracts shot-noise for the monopole. Set Pell[0].subtract_shotnoise = false if this is not wanted
template&lt;int N, class T&gt;
  void compute_power_spectrum_multipoles(
      int Ngrid,
      MPIParticles&lt;T&gt; &amp; part,
      double velocity_to_displacement,
      std::vector&lt;PowerSpectrumBinning&lt;N&gt;&gt; &amp; Pell,
      std::string density_assignment_method,
      bool interlacing);

// Cross multipoles of two fields (same binning as above)
template&lt;int N&gt;
  void compute_cross_power_spectrum_multipoles_fourier(
      const FFTWGrid&lt;N&gt; &amp; fourier_grid1,
      const FFTWGrid&lt;N&gt; &amp; fourier_grid2,
      std::vector&lt;PowerSpectrumBinning&lt;N&gt;&gt; &amp; Pell,
      std::vector&lt;double&gt; line_of_sight_direction);
</pre>
</div>
</div>
</p>


<header class="entry-header text-center">
  <h1 class="entry-title" id="FFTLog">Hankel transforms (FFTLog)</h1>
</header>
<hr/>  

<p>
Hankel transforms and particularly the <a href="https://jila.colorado.edu/~ajsh/FFTLog/" style="display: inline">FFTLog algorithm</a> have many great uses, in particular to convert between a 3D radial correlation function in real and Fourier space. We provide a slightly modified version of the <a href="https://github.com/slosar/FFTLog" style="display: inline">C++ library</a> by A. Slosar to do just that (namespace FML::SOLVERS::FFTLOG in FFTLog/FFTLog.h):
<div style="text-align: center">
  <div style="width:600px;display: inline-block;text-align: left">
    <pre class="prettyprint">
//==========================================================================
// Compute the correlation function xi(r) from a power spectrum P(k), sampled
// at logarithmically spaced points k[j]
//==========================================================================
std::pair&lt;DVector,DVector&gt; ComputeCorrelationFunction(const DVector &amp; k, const DVector &amp; pk);

//==========================================================================
// Compute the power spectrum P(k) from a correlation function xi(r), sampled
// at logarithmically spaced points r[i]
//==========================================================================
std::pair&lt;DVector,DVector&gt; ComputePowerSpectrum(const DVector &amp; r, const DVector &amp; xi);

//==========================================================================
// Compute xi_l^m(r) = Int dk/(2pi^2) k^m j_l(kr) P(k)
// (the usual xi(r) is xi_0^2). The k-values must be logarithmically spaced
//==========================================================================
std::pair&lt;DVector,DVector&gt; ComputeXiLM(int ell, int m, const DVector &amp; k, const DVector &amp; pk);

//==========================================================================
// Compute the discrete Hankel transform of the function a(r). See the FFTLog
// documentation for a description of exactly what this function computes.
//==========================================================================
std::pair&lt;DVector, CVector&gt; DiscreteHankelTransform(
    const DVector &amp; r,
    const CVector &amp; a,
    double mu,
    double q = 0,
    double kcrc = 1,
    bool noring = true,
    CVector* u =  nullptr);
</pre>
</div>
</div>
</p>
<p>
The first method computes
$$\xi(r) = \int_0^\infty \frac{k^3 P(k)}{2\pi^2} \frac{\sin(kr)}{kr} \frac{dk}{k}$$
and returns $(r,\xi)$. The second one computes
$$P(k) = \int_0^\infty \xi(r) \frac{\sin(kr)}{kr} 4\pi r^2 dr$$
and returns $(k,P(k))$. The third one computes
$$a(k) = \int a(r) (kr)^qJ_\mu(kr) kdr$$
and returns $(k,a)$ where $J_\mu$ is a Bessel function. Note that this is only implemented in 3D.

Read <a href="https://jila.colorado.edu/~ajsh/FFTLog/" style="display: inline">this</a> for how to properly use this algorithm. You need to be careful to extend the input data in a sensible way to prevent ringing. You should always check that you have a good setup by applying it to test data where you know the result.
</p>

<header class="entry-header text-center">
  <h1 class="entry-title" id="Bi">Bispectrum</h1>
</header>
<hr/>  

<p>
Computing the bispectrum $\left&lt;\delta(k_1)\delta(k_2)\delta(k_3)\right&gt; \equiv \delta(k_1+k_2+k_3)B(k_1,k_2,k_3)$ requires us to bin up over closed triangles in Fourier space. Just as with paircounts in real space doing this the straightforward way is not a parallelization friendly operation as Fourier modes reside on different tasks. However there is a way to do the binning with just Fourier transforms. With $N_b$ $k$-bins we can do the whole binning with $N_b$ Fourier transforms and $N_b^3/8$ real-space integrals (and we are forced to store $N_b$ grids so it can get memory expensive for large $N_b$). What we compute is
$$F_i = \int d^Dq_i \delta_{\rm shell~about~k_i}e^{iqx}$$
$$N_i = \int d^Dq_i 1_{\rm shell~about~k_i}e^{iqx}$$
where $X_{\rm shell~about~k_i}$ equals $X$ for $-\Delta k/2 &lt; q - k_i&lt; \Delta k/2$ and is zero otherwise (i.e. it's $X$ in a shell about $k_i$). We can then estimate the bispectrum as
$$B(k_1,k_2,k_3) = \frac{\int d^3 x F_{123}}{\int d^3 x N_{123}}$$
where $F_{123} = F_1F_2F_3$ and $N_{123} = N_1N_2N_3$. What this in effect is doing is binning up $\delta\delta\delta$ over closed triangles where $N_{123}$ counts the number of triangles in each bin (which we need to normalize the sum). One can estimate $N_{123}$ analytically and use this, but the fiducial option is to compute it numerically (which means another $N_b$ Fourier transforms). If one were to compute many bispectra with the same settings then one only needs to compute $N_{123}$ once: compute it, store it and pass it to the next binning with bofk.set_bincount(N123) which cuts down the runtime by a factor of $2$. The same method easily generalizes to higher order correlators (e.g. for the trispectrum we get $F_{1234}$ and $N_{1234}$ in the integrals above). The dimension $D$ has no big role in this algorithm (other than the fact that the complexity goes as $\propto D^3$) so in principle it works in any dimension $D\geq 2$.
</p>
<p>
When computing it from particles we subtract the shot-noise $\frac{P(k_1)+P(k_2)+P(k_3)}{\bar{n}} + \frac{1}{\bar{n}^2}$ (set bofk.subtract_shotnoise = false if this is not wanted). For the trispectrum the shot-noise is not subtracted as this requires the bispectrum. The results are fetched with bofk.get_spectrum(i,j,k), the reduced bispectrum $Q$ with bofk.get_reduced_spectrum(i,j,k) and the number of triangles in a bin with bofk.get_bincount(i,j,k). It is straightforward to generalize this to compute bispectrum multipoles and do this for a survey setup (though I have not bothered doing this yet). See <a href="https://arxiv.org/pdf/1506.02729.pdf" style="display: inline">Scoccimarro 2015</a> for more info.
</p>

<p>
<div style="text-align: center">
  <div style="width:600px;display: inline-block;text-align: left">
    <pre class="prettyprint">
// The binning: nbins linearly spaced bins between kmin and kmax (in units of 1/Boxsize, i.e. the fundamental mode is 2pi)
// or nbins bins from 0 to the Nyquist frequency of a grid of size Nmesh
BispectrumBinning&lt;N&gt; bofk(kmin, kmax, nbins);
BispectrumBinning&lt;N&gt; bofk(nbins, Nmesh);

// Computes the bispectrum B(k1,k2,k3) from particles
// This deconvolves the window function and (optionally) uses interlacing to reduce the alias contribution
template&lt;int N, class T&gt;
  void compute_bispectrum(
      int Ngrid,
      T *part,
      size_t NumPart,
      size_t NumPartTotal,
      BispectrumBinning&lt;N&gt; &amp;bofk,
      std::string density_assignment_method,
      bool interlacing);

// Computes the bispectrum B(k1,k2,k3) from a fourier grid
template&lt;int N&gt;
  void compute_bispectrum(
      const FFTWGrid&lt;N&gt; &amp; density_k,
      BispectrumBinning&lt;N&gt; &amp; bofk);

</pre>
</div>
</div>
</p>

<header class="entry-header text-center">
  <h1 class="entry-title" id="Poly">General polyspectra</h1>
</header>
<hr/>  

<p>
The general version of the method described above, it returns the matrix $\left&lt;\delta(k_1)\delta(k_2)\cdots\right&gt; = P(k_1,k_2,k_3,\ldots)$. Below ORDER is how many fields to correlate. 2 is power-spectrum, 3 is bispectrum, 4 is trispectrum etc. The output has a memory footprint of $N_{\rm bins}^{\rm ORDER}$ elements. We have to do $N_b^{\rm ORDER}$ integrals so needless to say with a large number of bins this is quite expensive for large orders.
<div style="text-align: center">
  <div style="width:600px;display: inline-block;text-align: left">
    <pre class="prettyprint">
// Computes the polyspectrum P(k1,k2,k3,...) from particles
// This deconvolves the window function and does interlacing to reduce the alias contribution if interlacing = true
// For the monospectrum and bispectrum we subtract shot-noise. Set polyofk.subtract_shotnoise = false if this is not wanted
template&lt;int N, class T, int ORDER&gt;
  void compute_polyspectrum(
      int Ngrid,
      T *part,
      size_t NumPart,
      size_t NumPartTotal,
      PolyspectrumBinning&lt;N, ORDER&gt; &amp; polyofk,
      std::string density_assignment_method,
      bool interlacing);

// Computes the polyspectrum P(k1,k2,k3,...) from a fourier grid
template&lt;int N, int ORDER&gt;
  void compute_polyspectrum(
      const FFTWGrid&lt;N&gt; &amp; density_k,
      PolyspectrumBinning&lt;N, ORDER&gt; &amp; polyofk);

// We also have the wrapper methods (with the same signature as the ones above,
// but without the ORDER template parameter) for computing
// P(k1,k2) (overkill, but useful for testing), B(k1,k2,k3) and T(k1,k2,k3,k4)
  void compute_monospectrum( ... )  // P(k1,k2) (diagonal is just the power-spectrum P(k))
  void compute_bispectrum( ... )    // B(k1,k2,k3)
  void compute_trispectrum( ... )   // T(k1,k2,k3,k4)
</pre>
</div>
</div>
</p>

<header class="entry-header text-center">
  <h1 class="entry-title" id="PolyNew">Polyspectra: dealiasing, selected configurations, sparse results and memory</h1>
</header>
<hr/>

<p>
The polyspectra use the FFT method of Scoccimarro: for each $k$-shell $i$ the field restricted to the shell, $\delta_i(\vec x) = \int_{\vec k\in i}\delta(\vec k)e^{i\vec k\cdot\vec x}$, is computed with one FFT, and the bispectrum of a shell triplet is
$$\hat B(k_1,k_2,k_3) = \frac{\sum_{\vec x}\delta_1(\vec x)\delta_2(\vec x)\delta_3(\vec x)}{\sum_{\vec x}I_1(\vec x)I_2(\vec x)I_3(\vec x)},$$
where $I_i$ is the same with $\delta = 1$ (it counts the closed triangles). The products of shell fields alias on the FFT mesh: a product of three fields can wrap around and "close" a configuration modulo the mesh. By default (<tt>bins.dealias = true</tt>) only genuinely closed integer-wavevector configurations are counted: when the shell range could wrap the shell fields are evaluated on a padded mesh (and the even-mesh Nyquist planes are excluded for orders &gt; 2). Further options:
</p>
<ul>
<li><b>Selected configurations</b>: <tt>bins.set_configurations({{1, 2, 3}, {4, 4, 4}})</tt> computes only the requested (canonical) triangles, and only their shells are built.</li>
<li><b>Sparse results</b>: <tt>SparseBispectrumBinning</tt> / <tt>SparseTrispectrumBinning</tt> store one value per canonical configuration instead of $n_{\rm bins}^{\rm ORDER}$ (100 trispectrum bins would otherwise need 1.5 GB per task); lookups are permutation-aware.</li>
<li><b>Shell overlap weights</b> (<tt>bins.shell_overlap_weights = true</tt>, off by default; <tt>polyspectrum_shell_overlap_weights_default</tt> sets it for every new binning; <tt>bispectrum_shell_overlap_weights</tt> in SimulationCode and SnapshotAnalyzer): each Fourier cell counts with the exact fraction of its volume inside the shell instead of 0 or 1 from where its centre is ("shell anti-aliasing" as in <a href="https://github.com/jmd-dk/concept" style="display: inline">CONCEPT</a>, whose closed-form overlap of a cell and a ball is ported in <tt>ShellOverlap.h</tt>). The shells are then continuous spheres: a bin's mean $k$ moves smoothly with its edges, and for equilateral bins one cell wide the triangle count follows the continuum $8\pi^2 k^3\Delta k^3$ to better than 1% from $k = 3k_f$, where 0/1 shells scatter by 10&ndash;30%. This mostly matters at low $k$ and for thin bins, and makes comparisons with bin-averaged theory simpler. It does not reduce the statistical errors, neighbouring bins share their boundary cells, and it costs about 10% more time.</li>
<li><b>Memory</b>: the shell grids are kept in a least-recently-used cache with a maximum number of shells and/or a byte budget per task; the geometry counts $\sum I_1I_2I_3$ are cached and reused, and <tt>bins.statistics.memory</tt> reports the evaluation storage component by component.</li>
</ul>
<div style="text-align: center">
  <div style="width:600px;display: inline-block;text-align: left">
    <pre class="prettyprint">
SparseBispectrumBinning&lt;3&gt; bins(100, nmesh);
bins.set_configurations({{1, 2, 3}, {4, 4, 4}});
bins.options.max_cached_shells = 3;
bins.options.shell_memory_budget_bytes = 256ULL * 1024 * 1024;
compute_bispectrum(delta_fourier, bins);
double B = bins.get_spectrum(3, 1, 2);       // any permutation
</pre>
</div>
</div>

<header class="entry-header text-center">
  <h1 class="entry-title" id="Related">Related pages</h1>
</header>
<hr/>

<p>
Power-spectrum multipoles of a survey with a local line of sight, the window function and reconstruction are on the <a href="survey.php" style="display: inline">survey page</a>. Matched auto- and cross-spectra of simulations with the same phases (and control variates) are on the <a href="statistics.php#Ensemble" style="display: inline">summary statistics page</a>, kNN-CDFs, counts-in-cells, density-split and marked correlations as well. The hybrid Lagrangian bias spectra in real and redshift space (with their Zel'dovich and one-loop predictions, also in modified gravity) are computed in the <a href="nbodycode.php#Analysis" style="display: inline">COLA code</a>, and the one-loop SPT power spectrum on the <a href="perturbationtheory.php" style="display: inline">perturbation theory page</a>.
</p>

<header class="entry-header text-center">
  <h1 class="entry-title" id="Connected">Connected four-point statistics</h1>
</header>
<hr/>

<p>
<tt>FML/ConnectedFourPoint</tt> separates raw fourth moments, the three Wick contractions and their connected remainder for known zero-mean fields. Its Fourier API averages over ordered closing quartets in chosen shells, using padding to exclude closure modulo the mesh. It integrates over unresolved diagonals and angles, rather than returning a fully shape-resolved trispectrum. Physical shell-normalized outputs have units of length$^9$.
</p><div style="text-align: center"><div style="max-width:800px;display:inline-block;text-align:left;width:100%"><pre class="prettyprint">#include &lt;FML/ConnectedFourPoint/Fourier.h&gt;
using namespace FML::FOURPOINT;
ShellOptions options{boxsize, k_edges};
auto results = connected_shell_four_point(delta_fourier, options,
                                         {{0,0,0,0}, {0,0,1,1}});</pre></div></div>
<p>
A second adapter takes the sample covariance of powers from independent periodic boxes, subtracts an explicitly supplied Gaussian covariance, and returns the collapsed trispectrum $T_{ij}=V(C_{ij}-C^{G}_{ij})$. This is a single-box covariance, without dividing by the number of realizations. Same-sample Wick products are biased finite-sample plug-in estimates; noisy external references also require uncertainty treatment. Survey windows, super-sample covariance and automatic connected particle shot-noise subtraction are outside this API. The older <tt>compute_trispectrum</tt> still reports a raw fourth moment.
</p>

<header class="entry-header text-center">
  <h1 class="entry-title" id="References">Method references and scope</h1>
</header>
<hr/>

<p>
The separate <tt>ConnectedFourPoint</tt> module uses the periodic covariance/trispectrum convention of <a href="https://arxiv.org/abs/astro-ph/9901099" style="display: inline">Scoccimarro, Zaldarriaga &amp; Hui (1999)</a>. It reports raw, Wick-disconnected and connected shell-averaged fourth moments, and a collapsed trispectrum from independent-box power covariance. A same-sample Wick subtraction is a finite-sample plug-in statistic; an exact ensemble covariance has a different bias interpretation. This module does not add survey-window terms or automatically remove connected particle shot noise. See the <a href="https://github.com/HAWinther/FML/blob/master/FML/ConnectedFourPoint/README.md" style="display: inline">estimator definitions and validation scope</a>. The older <tt>compute_trispectrum</tt> API remains a raw fourth moment.
</p>
