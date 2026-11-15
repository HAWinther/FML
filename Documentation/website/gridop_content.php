<hr/>
<header class="entry-header text-center">
  <h1 class="entry-title">Table of contents</h1>
</header>
<hr/>

<p class="page-toc">
<a href="index.php">Back to the main page</a><br>
<a href="documentation.php">Back to the documentation</a><br>
<a href="#Overview">1. Overview: Grid operations</a><br>
<a href="#Smooth">2. Smoothing</a><br>
<a href="#White">3. Whitening</a><br>
<a href="#Conv">4. Convolutions</a><br>
<a href="#Modes">5. Looping over Fourier modes and shifting grids</a><br>
<a href="#PDF">6. One-point PDF</a><br>
</p>

<header class="entry-header text-center">
  <h1 class="entry-title" id="Overview">Overview: Grid operations</h1>
</header>
<hr/>  

<p>
Some common operations on grids. These live in the namespace FML::GRID and are found in Smoothing/SmoothingFourier.h (except the real space convolution with a kernel which is in Interpolation/ParticleGridInterpolation.h).
</p>

<header class="entry-header text-center">
  <h1 class="entry-title" id="Smooth">Smoothing</h1>
</header>
<hr/>  

<p>
Smooth the grid by Fourier transforming and applying a filter. The filter can be sharpk ($H(1-kR)$, i.e. $0$ for $kR &gt; 1$ and $1$ otherwise), tophat (the Fourier transform of $H(R-x)$ in 2D or 3D) or gaussian ($e^{-\frac{1}{2}(kR)^2}$). The smoothing scale $R$ is in units of the boxsize. You can also provide your own filter as a function of $(kB)^2$ where $B$ is the boxsize:
<div style="text-align: center">
  <div style="width:600px;display: inline-block;text-align: left">
    <pre class="prettyprint">
template&lt;int N&gt;
  void smoothing_filter_fourier_space(
      FFTWGrid&lt;N&gt; &amp; fourier_grid,
      double smoothing_scale,
      std::string smoothing_method);

template&lt;int N&gt;
  void custom_smoothing_filter_fourier_space(
      FFTWGrid&lt;N&gt; &amp; fourier_grid,
      const std::function&lt;double(double)&gt; &amp; custom_filter_of_kBox_squared);
</pre>
</div>
</div>
</p>

<header class="entry-header text-center">
  <h1 class="entry-title" id="White">Whitening</h1>
</header>
<hr/>  

<p>
Transform a grid to Fourier space, normalize all modes such that $|f(k)| = 1$ and transform back (the method below does just the middle part).
<div style="text-align: center">
  <div style="width:600px;display: inline-block;text-align: left">
    <pre class="prettyprint">
template&lt;int N&gt;
  void whitening_fourier_space(FFTWGrid&lt;N&gt; &amp; fourier_grid); 

// When starting in real-space
grid.fftw_r2c();
whitening_fourier_space(grid);
grid.fftw_c2r();
</pre>
</div>
</div>
</p>


<header class="entry-header text-center">
  <h1 class="entry-title" id="Conv">Convolutions</h1>
</header>
<hr/>  

<p>
Convolve the grid (in real space) with an arbitrary (compact) kernel:
<div style="text-align: center">
  <div style="width:600px;display: inline-block;text-align: left">
    <pre class="prettyprint">
// In the namespace FML::INTERPOLATION
template&lt;int N, int ORDER&gt;
  void convolve_grid_with_kernel(
      const FFTWGrid&lt;N&gt; &amp; grid_in,
      FFTWGrid&lt;N&gt; &amp; grid_out,
      const std::function&lt;FloatType(std::array&lt;double, N&gt; &amp; dx)&gt; &amp; convolution_kernel);
</pre>
</div>
</div>
</p>
<p>
The argument to the kernel is how far away the point is from the cell in units of the cell-size so the NGP kernel would be $1$ if all $|dx_i| &lt; 0.5$ and zero otherwise. ORDER is how wide the kernel is (how many grid-cells; this is needed to limit the number of cells we visit).
</p>

<p>
We can also perform convolutions of two grids in real and Fourier space (using Fourier transforms):
<div style="text-align: center">
  <div style="width:600px;display: inline-block;text-align: left">
    <pre class="prettyprint">
template &lt;int N&gt;
  void convolution_fourier_space(const FFTWGrid&lt;N&gt; &amp; fourier_grid_f,
                                 const FFTWGrid&lt;N&gt; &amp; fourier_grid_g,
                                 FFTWGrid&lt;N&gt; &amp; fourier_grid_result);

template &lt;int N&gt;
  void convolution_real_space(const FFTWGrid&lt;N&gt; &amp; real_grid_f,
                              const FFTWGrid&lt;N&gt; &amp; real_grid_g,
                              FFTWGrid&lt;N&gt; &amp; real_grid_result);
</pre>
</div>
</div>
</p>
<p>
This is computing the grids $F(k) = \int d^N q f(q)g(k-q)$ and $F(x) = \int d^N y f(y)g(x-y)$ respectively. This requires three Fourier transforms and one temporary grid (we don't touch the input grids).
</p>

<header class="entry-header text-center">
  <h1 class="entry-title" id="Modes">Looping over Fourier modes and shifting grids</h1>
</header>
<hr/>

<p>
Some helpers for working with grids in Fourier space (OpenMP parallelized; kmag is in units of 1/boxsize):
<div style="text-align: center">
  <div style="width:600px;display: inline-block;text-align: left">
    <pre class="prettyprint">
// Call f(fourier_index, kmag) for every local mode
template &lt;int N, class F&gt;
  void for_each_fourier_mode(const FFTWGrid&lt;N&gt; &amp; fourier_grid, F &amp;&amp; f);

// Replace every mode by func(kmag, value)
template &lt;int N, class F&gt;
  void transform_fourier_modes(FFTWGrid&lt;N&gt; &amp; fourier_grid, F &amp;&amp; func);

// Multiply every mode by g(kmag)
template &lt;int N, class F&gt;
  void multiply_fourier_by_function_of_k(FFTWGrid&lt;N&gt; &amp; fourier_grid, F &amp;&amp; g);

// a -&gt; wa * a + wb * b
template &lt;int N&gt;
  void linear_combination_fourier(FFTWGrid&lt;N&gt; &amp; a, const FFTWGrid&lt;N&gt; &amp; b, double wa, double wb);

// Translate a grid f(x) -&gt; f(x + sign * shift) using Fourier interpolation
// (shift is in units of the boxsize)
template &lt;int N&gt;
  void shift_grid_fourier(FFTWGrid&lt;N&gt; &amp; fourier_grid, const std::array&lt;double, N&gt; &amp; shift, double sign = +1.0);
template &lt;int N&gt;
  void shift_grid_real(FFTWGrid&lt;N&gt; &amp; real_grid, const std::array&lt;double, N&gt; &amp; shift, double sign = +1.0);

// Example: multiply by a Gaussian filter
multiply_fourier_by_function_of_k(grid, [&amp;](double k){ return std::exp(-0.5 * k * k * R * R); });
</pre>
</div>
</div>
</p>
<p>
Note that for an even grid size the Nyquist modes are their own mirror images so a shift by $s$ followed by $-s$ is only the identity for these modes if $s$ is a multiple of the cell size.
</p>

<header class="entry-header text-center">
  <h1 class="entry-title" id="PDF">One-point PDF</h1>
</header>
<hr/>

<p>
Compute the PDF of whatever quantity is in a real grid (e.g. the density) with linear bins over the range of values found in the grid. The PDF is normalized such that $\int p(x)dx = 1$:
<div style="text-align: center">
  <div style="width:600px;display: inline-block;text-align: left">
    <pre class="prettyprint">
template &lt;int N&gt;
  void compute_grid_PDF(
      const FFTWGrid&lt;N&gt; &amp; real_grid,
      int nbins,
      std::vector&lt;double&gt; &amp; x,
      std::vector&lt;double&gt; &amp; pdf);
</pre>
</div>
</div>
</p>

