<hr/>
<header class="entry-header text-center">
  <h1 class="entry-title">Table of contents</h1>
</header>
<hr/>

<p class="page-toc">
<a href="index.php">Back to the main page</a><br>
<a href="documentation.php">Back to the documentation</a><br>
<a href="#Overview">1. Overview: beyond the two-point function</a><br>
<a href="#kNN">2. kNN-CDFs and counts-in-cells (FML::COUNTS_IN_CELLS)</a><br>
<a href="#Environment">3. Density-split and marked correlations (FML::ENVIRONMENT)</a><br>
<a href="#Morphology">4. Minkowski functionals of excursion sets (FML::MORPHOLOGY)</a><br>
<a href="#Ensemble">5. Control variates and ensemble summaries (FML::ENSEMBLE)</a><br>
<a href="#References">Method references and scope</a><br>
<a href="#Matched">Phase-tracked ensemble means</a><br>
</p>

<header class="entry-header text-center">
  <h1 class="entry-title" id="Overview">Overview: beyond the two-point function</h1>
</header>
<hr/>

<p>
The late-time density field is far from Gaussian, so the power spectrum does not contain all the information. These modules compute summary statistics that are sensitive to the higher-order (and environment-dependent) structure of the field, plus a variance-reduction method for estimating ensemble means of any of them. The polyspectra (bispectrum, trispectrum) are on the <a href="corrfunc.php" style="display: inline">correlation function page</a>, and the cosmic-web statistics on the <a href="cosmicweb.php" style="display: inline">cosmic web page</a>.
</p>

<header class="entry-header text-center">
  <h1 class="entry-title" id="kNN">kNN-CDFs and counts-in-cells</h1>
</header>
<hr/>

<p>
For a set of query points distributed uniformly in the volume, the $k$-nearest-neighbour cumulative distribution function (Banerjee &amp; Abel 2021) is the fraction of queries whose $k$-th nearest tracer is within $r$,
$${\rm CDF}_k(r) = P(r_k \le r) = P(N(&lt;r)\ge k),$$
so it is a cumulative count-in-cells probability: with $P_N(r)$ the probability of finding exactly $N$ tracers in a sphere of radius $r$,
$$P_0(r) = 1 - {\rm CDF}_1(r),\qquad P_j(r) = {\rm CDF}_j(r) - {\rm CDF}_{j+1}(r).$$
The CDFs are sensitive to all connected $n$-point functions; for a Poisson process with mean count $\bar N = \bar nV$ they reduce to the binomial/Poisson law ${\rm CDF}_k = 1 - \sum_{j&lt;k}e^{-\bar N}\bar N^j/j!$. The counts-in-cells statistics are the occupancy PDF $P_N$, the density PDF of $\delta = N/(\bar nV) - 1$, and the falling factorial moments
$$F_q = \langle N(N-1)\cdots(N-q+1)\rangle = \bar N^q\,\langle(1+\delta_V)^q\rangle_{\rm continuous}$$
which remove Poisson self-counting (so $\langle N^2\rangle - \langle N\rangle - \langle N\rangle^2 = \bar N^2\sigma_V^2$).
</p>
<p>
The implementation uses the library's KD-tree with periodic wrapping (or an explicit non-periodic observation window), streams the results into exact integer hit counts and histograms, and uses Philox-generated query points keyed by a global query ID (so the queries are independent of the MPI decomposition). The tracer catalogue is replicated on each task while the queries are distributed; results are reduced over tasks and threads exactly.
</p>
<div style="text-align: center">
  <div style="width:700px;display: inline-block;text-align: left">
    <pre class="prettyprint">
#include &lt;FML/CountsInCells/SpatialStatistics.h&gt;
using namespace FML::COUNTS_IN_CELLS;
FML::KDTREE::pointVec tracers;                 // complete tracer catalogue on each task
SpatialStatistics statistics(tracers, boxsize);
Bounds volume{{0.,0.,0.}, {boxsize,boxsize,boxsize}};
auto queries = uniform_queries(local_query_count, volume, seed, global_first_query_id);
std::vector&lt;double&gt; radii{5., 10., 15., 20.};
auto cdf    = statistics.knn_cdf(queries, {1, 2, 4, 8}, radii);   // cdf.cdf[order index][radius index]
auto counts = statistics.counts_in_cells(queries, radii);       // PDFs, factorial moments, ...</pre>
  </div>
</div>
<p>
Validation: exact agreement with brute force for distances and counts (periodic and non-periodic, coincident and self-excluded tracers), and the mean kNN-CDF of an ensemble of uniform catalogues equals the exact binomial law to the statistical precision.
</p>

<header class="entry-header text-center">
  <h1 class="entry-title" id="Environment">Density-split and marked correlations</h1>
</header>
<hr/>

<p>
<b>Density split</b> (Paillas et al., Pinon et al. 2025). The density in a top-hat sphere of radius $R$ is measured around query points, $\delta_R = N(&lt;R)/\bar N_R - 1$, the queries are split into $Q$ quantiles of $\delta_R$, and the cross-correlation of each quantile with the tracers is measured,
$$\xi_q(r) = \frac{D_qT(r)}{N_qN_T f_{\rm shell}(r)} - 1,\qquad \sum_q\frac{N_q}{N}\,\xi_q(r) = \xi_{\rm parent}(r).$$
The quantiles carry the environment dependence of clustering (voids to clusters) and are much more informative than $\xi(r)$ alone. All quantiles are accumulated in one pair traversal.
</p>
<p>
<b>Marked correlations</b> (Skibba et al. 2006, White 2016). Each tracer gets a mark, e.g. the inverse-density mark $m = \left[\frac{1+\delta_s}{1+\delta_s+\delta_R}\right]^p$ that up-weights low-density regions (sensitive to screening in modified gravity), and
$$M(r) = \frac{1 + W(r)}{1 + \xi(r)} = \frac{WW(r)}{DD(r)},\qquad WW = \sum_{ij}\frac{m_im_j}{\bar m^2}.$$
The finite-population baseline of the mark normalization is reported separately.
</p>
<div style="text-align: center">
  <div style="width:700px;display: inline-block;text-align: left">
    <pre class="prettyprint">
#include &lt;FML/Environment/EnvironmentClustering.h&gt;
using namespace FML::ENVIRONMENT;
auto env = top_hat_environment(tracers, centers, L, smoothing_radius);       // env.delta
auto ds  = density_split_correlation(centers, env.delta, tracers, 5, L, rmin, rmax, nbins);
auto own = top_hat_environment(tracers, tracers, L, smoothing_radius, true); // exclude self
auto marks  = inverse_density_marks(own.delta, 1.0, 1.0);
auto marked = marked_correlation(tracers, marks, L, rmin, rmax, nbins);</pre>
  </div>
</div>

<header class="entry-header text-center">
  <h1 class="entry-title" id="Morphology">Minkowski functionals of excursion sets</h1>
</header>
<hr/>

<p>
The morphology of the excursion set $\{\vec x: \delta(\vec x)\ge\nu\}$ is described by the four Minkowski functionals: the volume $V_0$, the surface area $V_1$, the integrated mean curvature $V_2$ and the Euler characteristic $V_3$ (number of components minus tunnels plus cavities). For a Gaussian random field they are known analytically (Tomita 1986, Schmalzing &amp; Buchert 1997), e.g.
$$v_0 = \frac{1}{2}{\rm erfc}\left(\frac{\nu}{\sqrt{2}}\right),\quad v_1 = \frac{2}{3}\frac{\lambda}{\sqrt{2\pi}}e^{-\nu^2/2},\quad v_2 = \frac{2}{3}\frac{\lambda^2}{\sqrt{2\pi}}\nu e^{-\nu^2/2},\quad v_3 = \frac{\lambda^3}{\sqrt{2\pi}}(\nu^2-1)e^{-\nu^2/2},$$
with $\lambda^2 = \sigma_1^2/(6\pi\sigma_0^2)$, so deviations measure non-Gaussianity. The excursion set is represented as a union of closed voxels and the functionals are computed exactly from the counts of occupied cubes $C$, faces $F$, edges $E$ and vertices $V$ of the cell complex:
$$V = Ch^3,\qquad A = (2F - 6C)h^2,\qquad \int H\,dA = \pi(3C - 2F + E)h,\qquad \chi = V - E + F - C,$$
and alternatively with the isotropic Crofton estimators for a sampled smooth field. Only one ghost slice is exchanged between MPI tasks.
</p>
<div style="text-align: center">
  <div style="width:700px;display: inline-block;text-align: left">
    <pre class="prettyprint">
#include &lt;FML/Morphology/VoxelMorphology.h&gt;
using namespace FML::MORPHOLOGY;
auto results = excursions(smoothed_density, {-0.5, 0.0, 1.0}, boxsize,
                          Boundary::Periodic, ThresholdRule::AtLeast);
// results[i].measures.{volume, surface_area, integrated_mean_curvature, euler_characteristic}
// results[i].measures.crofton_density</pre>
  </div>
</div>
<p>
Validation: an independent explicit set-union enumeration, analytic cases (cavities, rings, periodic seams), and for Gaussian random fields ($128^3$, smoothed over 4 cells) the Crofton densities agree with the analytic predictions to ~1% ($V_0$, $V_1$), 1&ndash;4% ($V_2$) and 2&ndash;7% ($V_3$).
</p>

<header class="entry-header text-center">
  <h1 class="entry-title" id="Ensemble">Control variates and ensemble summaries</h1>
</header>
<hr/>

<p>
Estimating the mean of a summary statistic $\vec y$ from expensive simulations can be sped up with cheap surrogates $\vec c$ run with the same initial phases (e.g. ZA, 2LPT or COLA against full N-body; CARPool, Chartier et al. 2021). With a coefficient matrix $B$ fitted on an independent training set and the surrogate mean $\vec\mu$ from a large independent cheap ensemble,
$$\vec z_i = \vec y_i - B(\vec c_i - \vec\mu),\qquad \hat{\vec y} = \langle\vec z\rangle,\qquad {\rm Cov}(\hat{\vec y}) = \frac{{\rm Cov}(\vec z)}{N_{\rm eval}} + B\,{\rm Cov}(\hat{\vec\mu})B^T,$$
with $B = C_{yc}C_{cc}^{-1}$ (full, with a ridge regularization) or bin by bin. When $\vec y$ and $\vec c$ are highly correlated the variance of $\vec z$ is far below that of $\vec y$. The module provides streaming (Welford) means and covariances with MPI reductions, the fit, held-out evaluation, provenance tracking of the phases (training, mean estimation and evaluation must use disjoint phase blocks, and antithetic pairs must be averaged into one row) and matched auto/cross power spectra with their coherence.
</p>
<div style="text-align: center">
  <div style="width:700px;display: inline-block;text-align: left">
    <pre class="prettyprint">
#include &lt;FML/EnsembleStatistics/ControlVariates.h&gt;
using namespace FML::ENSEMBLE;
Metadata tags{"cosmology digest", "Pk: k centers, units, estimator", "Nbody", "2LPT"};
PairedTraining training(n_target, n_control, tags);
training.add(y, c, PhaseKey{1, seed1}, PhaseKey{1, seed1});        // matched phases
auto mu    = MeanEstimate::independent(cheap_ensemble.global(), cheap_keys, tags);
auto model = training.global().fit(mu);
ControlEvaluation held_out(model);
held_out.add(y_new, c_new, PhaseKey{3, seed3}, PhaseKey{3, seed3});
auto estimate = held_out.global().estimate();   // mean, covariance_of_mean</pre>
  </div>
</div>
<p>
Validation: Monte Carlo of the full train / independent-mean / held-out pipeline (2000 experiments): the controlled mean is unbiased, its empirical variance equals the predicted covariance of the mean to 1&ndash;4% and the 95% intervals cover 94&ndash;96%.
</p>

<header class="entry-header text-center">
  <h1 class="entry-title" id="Matched">Phase-tracked ensemble means</h1>
</header>
<hr/>

<p>Control-variate training, cheap-control mean estimation and held-out target evaluation now carry phase identities. Matched target/control rows use the same initial phases; the three stages must use disjoint phase blocks. Average antithetic pairs into one statistical row before estimating moments. The uncertainty of the independently estimated control mean contributes $B\,\mathrm{Cov}(\hat\mu)B^T$ to the controlled mean covariance.</p>
<p>A matched PM/ZA/LPT/COLA workflow exercises the pipeline against independent NumPy calculations and MPI decompositions. This tests bookkeeping, covariance propagation and selected dynamics, rather than establishing a mesh-converged N-body target. See <a href="https://github.com/HAWinther/FML/blob/master/FML/EnsembleStatistics/matched_validation/README.md" style="display: inline">the runnable matched workflow</a>.</p>

<header class="entry-header text-center">
  <h1 class="entry-title" id="References">Method references and scope</h1>
</header>
<hr/>

<p>
Voxel morphology uses the integral-geometric/lattice approach of <a href="https://arxiv.org/abs/astro-ph/9702130" style="display: inline">Schmalzing &amp; Buchert (1997)</a> and <a href="https://arxiv.org/abs/astro-ph/9904384" style="display: inline">Schmalzing et al. (1999), lattice-counting appendix</a>. FML implements its own cubical-complex counts with explicit threshold, boundary and normalization rules. Control-variate mean estimation follows <a href="https://arxiv.org/abs/2009.08970" style="display: inline">Chartier et al. (2021), CARPool</a>; the implementation uses frozen coefficients and independent phase blocks for training, control-mean estimation and evaluation. It does not implement the separate CARPool covariance-estimation extension.
</p>
