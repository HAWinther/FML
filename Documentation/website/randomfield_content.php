<hr/>
<header class="entry-header text-center">
  <h1 class="entry-title">Table of contents</h1>
</header>
<hr/>

<p class="page-toc">
<a href="index.php">Back to the main page</a><br>
<a href="documentation.php">Back to the documentation</a><br>
<a href="#Overview">1. Overview: Random fields</a><br>
<a href="#Gauss">2. Gaussian random field</a><br>
<a href="#fnl">3. Local and non-local non-gaussianity</a><br>
<a href="#noise">4. Gaussian white noise</a><br>
<a href="#Philox">5. Philox mode-keyed fields (independent of resolution and decomposition)</a><br>
</p>

<header class="entry-header text-center">
  <h1 class="entry-title" id="Overview">Overview: Random fields</h1>
</header>
<hr/>  

<p>
We generate a Gaussian random field in Fourier space by drawing the norm from a Rayleigh distribution and the phase from a uniform distribution.  In practice this is done by drawing $A,\theta\in U(0,1)$ and setting
$$\delta(k) = \sqrt{-\log(A)\frac{P(k)}{V}}e^{2\pi i \theta}$$
where the dimensionless ratio $\frac{P(k)}{V}$ is provided by the user. 
If the option <a href="https://arxiv.org/abs/1806.01871" style="display: inline">amplitude_fixed</a> is set then $A$ is fixed to $e^{-1}$ so $|\delta|^2 \equiv \frac{P(k)}{V}$, i.e. only the phases are random. This process is straightforward, however we must be careful to ensure that $\delta(k) = \delta(-k)^*$ (since $\delta(x)$ is real) and the Fourier conjugate to $k$, $-k$, might reside on a different task (this is solved by seeding the RNG the same way on each task based on $k$). The methods for Gaussian fields live in the namespace FML::RANDOM::GAUSSIAN (RandomFields/GaussianRandomField.h) and the non-Gaussian ones in FML::RANDOM::NONGAUSSIAN (RandomFields/NonLocalGaussianRandomField.h).
</p>

<header class="entry-header text-center">
  <h1 class="entry-title" id="Gauss">Gaussian random field</h1>
</header>
<hr/>  

<p>
<div style="text-align: center">
  <div style="width:600px;display: inline-block;text-align: left">
    <pre class="prettyprint">
// Generate a gaussian random field in Fourier space
template&lt;int N&gt;
  void generate_gaussian_random_field_fourier(
      FFTWGrid&lt;N&gt; &amp; grid,
      RandomGenerator *rng,
      std::function&lt;double(double)&gt; Pofk_over_V_of_kB,
      bool fix_amplitude);

// This is just the method above + a FFT to realspace
template&lt;int N&gt;
  void generate_gaussian_random_field_real(
      FFTWGrid&lt;N&gt; &amp; grid,
      RandomGenerator *rng,
      std::function&lt;double(double)&gt; Pofk_over_V_of_kB,
      bool fix_amplitude);

</pre>
</div>
</div>
</p>
<p>
The function Pofk_over_V_of_kB takes the dimensionless argument $k\cdot B$ where $B$ is the boxsize and returns the dimensionless $\frac{P(k)}{V}$. For example if $P(k) = k^3$ in 3D then
<div style="text-align: center">
  <div style="width:600px;display: inline-block;text-align: left">
    <pre class="prettyprint">
double Boxsize = 100.0; // Boxsize in whatever units you want to use
std::function&lt;double(double)&gt; Pofk_over_V_of_kB = [&amp;](double kBox){
  double k = kBox / Boxsize;
  double pofk = k * k * k;
  double volume = Boxsize * Boxsize * Boxsize;
  return pofk / volume;
};

// Generate it (here rng is a FML::RANDOM::RandomGenerator, see random.php)
FFTWGrid&lt;3&gt; grid(Nmesh);
generate_gaussian_random_field_real(grid, &amp;rng, Pofk_over_V_of_kB, false);
</pre>
</div>
</div>

</p>

<header class="entry-header text-center">
  <h1 class="entry-title" id="fnl">Local and non-local non-gaussian fields</h1>
</header>
<hr/>  

<p>
Local non-gaussianity in its simplest form is given by
$$\Phi = \phi + f_{\rm NL}(\phi^2 - \left&lt;\phi^2\right&gt;)$$
where $\phi$ is a Gaussian random field. More generally we can define (in Fourier space)
$$\Phi = \phi + f_{\rm NL}(K(\phi,\phi) - \left&lt;K(\phi,\phi)\right&gt;)$$
for some non-local kernel $K$ which (in Fourier space) can be written
$$F[K(\phi,\phi)] \equiv \int K(k_1,k_2)\phi(k_1)\phi(k_2)\delta(k - k_1-k_2)d^3k_1d^3k_2$$
for some Fourier space kernel $K(k_1,k_2)$. We provide some methods for computing such fields. Defining
$$[\mathcal{P}_mA](x) = \int e^{-ikx}\left(\frac{P(k)}{V}\right)^m A(k)d^3k$$
where $P(k)$ is the power-spectrum and $V$ is the volume of the box (just to have correct units, but this factors out below) then we can evaluate
$$K[\phi,\phi] = A\phi^2 + B \mathcal{P}_{2/3}( [\mathcal{P}_{-1/3}\phi]^2 ) + C \mathcal{P}_{1/3}( \phi[\mathcal{P}_{-1/3}\phi] ) + D  \mathcal{P}_{2/3}( \phi[\mathcal{P}_{-2/3}\phi] ) +  E \mathcal{P}_{3/3}( \phi[\mathcal{P}_{-3/3}\phi] ) + F \mathcal{P}_{3/3}( [\mathcal{P}_{-2/3}\phi][\mathcal{P}_{-1/3}\phi] )$$
for a given set of coefficients $A,B,C,D,E,F$. Specific choices for these coefficients give us local, equilateral and orthogonal configurations (the three options of type_of_fnl below). If you want to specify your own values then use type_of_fnl=generic and provide them in kernel_values (in the order $A,C,D,E$ and as currently written we assume $B=-D$ and $F=-E$).
</p>
<p>
<div style="text-align: center">
  <div style="width:600px;display: inline-block;text-align: left">
    <pre class="prettyprint">
template&lt;int N&gt;
  void generate_nonlocal_gaussian_random_field_fourier(
      FFTWGrid&lt;N&gt; &amp; phi_fourier,
      RandomGenerator *rng,
      std::function&lt;double(double)&gt; Pofk_over_V_of_kB,
      bool fix_amplitude,
      double fNL,
      std::string type_of_fnl,
      double u = 0.0,
      std::vector&lt;double&gt; kernel_values = {});
</pre>
</div>
</div>
</p>
<p>
For more info about this see <a href="https://arxiv.org/pdf/1108.5512.pdf" style="display: inline">Scoccimarro et al. 2012</a> and <a href="https://arxiv.org/pdf/1008.0638.pdf" style="display: inline">Schmidt and Kamionkowski 2010</a>.
</p>

<header class="entry-header text-center">
  <h1 class="entry-title" id="noise">Gaussian white noise</h1>
</header>
<hr/>  

<p>
We can generate a white noise field in real space. It is normalized such that the power-spectrum of this field is unity.
<div style="text-align: center">
  <div style="width:600px;display: inline-block;text-align: left">
    <pre class="prettyprint">
template&lt;int N&gt;
  void generate_white_noise_field_real(
      FFTWGrid&lt;N&gt; &amp; grid,
      RandomGenerator * rng);
</pre>
</div>
</div>
</p>

<header class="entry-header text-center">
  <h1 class="entry-title" id="Philox">Philox mode-keyed fields</h1>
</header>
<hr/>

<p>
The default backend reproduces the 2LPTic/N-GenIC realization (seeding by lines of the grid) to allow comparisons with other codes. With <tt>RandomFieldBackend::Philox</tt> every Fourier mode instead gets its random numbers from a counter-based Philox generator whose counter is the (signed) wavevector itself and the stream: the first nonzero coordinate is made positive to pick one representative of $\{\vec k,-\vec k\}$, the partner is its conjugate, and the zero and Nyquist modes are set to zero. Thus the field is bit-identical for any number of MPI tasks and threads, and a mode has the same amplitude and phase on grids of different resolution (useful for convergence tests and for matched low/high-resolution simulations). Changing the stream gives another independent realization. The same option works for the real-space and non-Gaussian generators and in the COLA code (<tt>ic_random_field_backend = "philox"</tt>).
</p>
<div style="text-align: center">
  <div style="width:600px;display: inline-block;text-align: left">
    <pre class="prettyprint">
using namespace FML::RANDOM::GAUSSIAN;
RandomFieldOptions options;
options.backend = RandomFieldBackend::Philox;
options.seed = 1234;    // uint64
options.stream = 0;     // independent realizations
generate_gaussian_random_field_fourier&lt;3&gt;(grid, nullptr, Pofk, false, options);
</pre>
</div>
</div>
