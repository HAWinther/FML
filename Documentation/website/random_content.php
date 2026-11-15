<hr/>
<header class="entry-header text-center">
  <h1 class="entry-title">Table of contents</h1>
</header>
<hr/>

<p class="page-toc">
<a href="index.php">Back to the main page</a><br>
<a href="documentation.php">Back to the documentation</a><br>
<a href="#Overview">1. Overview: Random numbers</a><br>
<a href="#Random">2. Creating random numbers</a><br>
<a href="#Philox">3. Counter-based random numbers (Philox)</a><br>
<a href="#References">Method references and scope</a><br>
<a href="#LatinHypercube">Latin hypercube experiment designs</a><br>
</p>

<header class="entry-header text-center">
  <h1 class="entry-title" id="Overview">Overview: Random numbers</h1>
</header>
<hr/>  

<p>
To be able to have a unified interface for algorithms that require random numbers we provide a base class RandomGenerator (namespace FML::RANDOM in RandomGenerator/RandomGenerator.h) with methods to produce these from whatever library you want. This is by default std::mt19937, but we also provide a wrapper on the GSL library. If you want something else just make a class that inherits from the base class and implement the required methods.
</p>

<header class="entry-header text-center">
  <h1 class="entry-title" id="Random">Creating random numbers</h1>
</header>
<hr/>  

<p>
How to produce uniform/normal random numbers:
<div style="text-align: center">
  <div style="width:600px;display: inline-block;text-align: left">
    <pre class="prettyprint">
using namespace FML::RANDOM;
RandomGenerator r;
r.set_seed(1234);

// Uniform in [0,1)
std::cout &lt;&lt; r.generate_uniform() &lt;&lt; "\n";

// Normal distribution with mean 0 and standard deviation 1
std::cout &lt;&lt; r.generate_normal()  &lt;&lt; "\n";

// For other ranges just transform: uniform in [a,b) and normal(mu, sigma)
double x = a + (b - a) * r.generate_uniform();
double y = mu + sigma * r.generate_normal();
</pre>
</div>
</div>
</p>

<p>
If you want to use GSL (fiducial is gsl_rng_ranlxd1, but can be changed in the header file; requires USE_GSL) then:
<div style="text-align: center">
  <div style="width:600px;display: inline-block;text-align: left">
    <pre class="prettyprint">
std::unique_ptr&lt;RandomGenerator&gt; r = std::make_unique&lt;GSLRandomGenerator&gt;(1234);
std::cout &lt;&lt; r-&gt;generate_uniform() &lt;&lt; "\n";
</pre>
</div>
</div>
</p>

<p>
We also provide methods for cloning the state of a RNG (r.clone() returns a std::unique_ptr&lt;RandomGenerator&gt; with a copy of the state) which is very useful for parallel algorithms.
</p>

<header class="entry-header text-center">
  <h1 class="entry-title" id="Philox">Counter-based random numbers (Philox)</h1>
</header>
<hr/>

<p>
A stateful generator gives results that depend on the order in which numbers are drawn, and therefore on the number of MPI tasks and threads. A counter-based generator (Salmon et al. 2011) is instead a keyed bijection: the random number for a given (key, counter) is computed directly, $r = {\rm Philox}_k(c)$, with no state to advance. <tt>RandomGenerator/CounterBasedRandom.h</tt> (namespace FML::RANDOM::COUNTER_BASED) is an independent implementation of Philox4x32-10, checked against the known-answer vectors of Random123. A <tt>Stream</tt> is keyed by a seed, an object ID (e.g. a halo ID, a query ID or a wavevector) and a purpose, so that e.g. the HOD, the kNN query points and the Philox random fields are reproducible independently of the decomposition.
</p>
<div style="text-align: center">
  <div style="width:600px;display: inline-block;text-align: left">
    <pre class="prettyprint">
#include &lt;FML/RandomGenerator/CounterBasedRandom.h&gt;
using namespace FML::RANDOM::COUNTER_BASED;
Stream s(seed, halo_id, /*purpose=*/1);   // independent of order, task and thread
double u = s.uniform();                   // in (0,1)
double g = s.normal();
</pre>
</div>
</div>

<header class="entry-header text-center">
  <h1 class="entry-title" id="LatinHypercube">Latin hypercube experiment designs</h1>
</header>
<hr/>

<p>
<tt>FML::SAMPLING::LatinHyperCube</tt> chooses parameter-space designs for emulator training and scans. Strength 1 puts exactly one point in each marginal stratum. Strength 2 also stratifies every two-dimensional projection and requires $n=p^2$, prime $p$, and dimension $d\le p+1$. Scrambling jitters within each cell; disabling it uses cell centers. Seeded draws are reproducible using FML's conversions, but do not reproduce NumPy's point sequence.
</p><div style="text-align: center"><div style="max-width:800px;display:inline-block;text-align:left;width:100%"><pre class="prettyprint">#include &lt;FML/LatinHyperCube/LatinHyperCube.h&gt;
using namespace FML::SAMPLING;
LatinHyperCube lhs(3, true, 1, LHSOptimization::random_cd, 1234);
auto unit_points = lhs.random(100);
auto cosmologies = scale(unit_points, {0.1, 0.6, 0.55}, {0.5, 1.0, 0.85});
double discrepancy = centered_discrepancy(unit_points);</pre></div></div>
<p><tt>random_cd</tt> lowers centered discrepancy through coordinate swaps, preserving marginal stratification but not strength-2 orthogonality. <tt>lloyd</tt> spreads points through damped centroidal steps and can lose the Latin property. Integer designs, reset and fast-forward are also available; <a href="https://github.com/HAWinther/FML/blob/master/FML/LatinHyperCube/README.md" style="display: inline">the README</a> gives the complete interface.</p>

<header class="entry-header text-center">
  <h1 class="entry-title" id="References">Method references and scope</h1>
</header>
<hr/>

<p>
<tt>LatinHyperCube</tt> implements <a href="https://doi.org/10.1080/00401706.1979.10489755" style="display: inline">McKay, Beckman &amp; Conover (1979) Latin hypercube sampling</a> and <a href="https://doi.org/10.1080/01621459.1993.10476423" style="display: inline">Tang (1993) orthogonal-array Latin hypercubes</a>, following SciPy method/API choices with its own random-number stream. Random coordinate swaps preserve marginal stratification; Lloyd optimization need not. Discrepancy and optimization sources and differences from SciPy are in the <a href="https://github.com/HAWinther/FML/blob/master/FML/LatinHyperCube/README.md" style="display: inline">module README</a>.
</p>
