<hr/>
<header class="entry-header text-center">
  <h1 class="entry-title">Table of contents</h1>
</header>
<hr/>

<p class="page-toc">
<a href="index.php">Back to the main page</a><br>
<a href="documentation.php">Back to the documentation</a><br>
<a href="#Overview">1. Overview: clustering of a galaxy survey</a><br>
<a href="#Geometry">2. Coordinates, weights and normalization</a><br>
<a href="#Power">3. Fourier-space multipoles (FKP / Yamamoto)</a><br>
<a href="#Window">4. The survey window and the window matrix</a><br>
<a href="#Recon">5. BAO reconstruction</a><br>
<a href="#Validation">6. Example and validation</a><br>
<a href="#References">Method references and scope</a><br>
<a href="#Convergence">External comparisons and window convergence</a><br>
</p>

<header class="entry-header text-center">
  <h1 class="entry-title" id="Overview">Overview: clustering of a galaxy survey</h1>
</header>
<hr/>

<figure class="cinematic-figure" id="cinematic-07-survey-slice">
  <a href="images/cinematic/07-survey-slice.jpg" aria-label="View full-size illustration: A survey’s slice of the cosmos">
    <img src="images/cinematic/07-survey-slice.jpg" width="1774" height="887" loading="lazy" decoding="async" alt="A wedge-shaped galaxy point cloud tracing filaments and voids within a survey footprint">
  </a>
  <figcaption><strong>A survey’s slice of the cosmos.</strong> Conceptual galaxy survey footprint extending from an observer; the wedge boundaries describe selection, not an edge of the Universe. <span class="cinematic-credit">AI-generated concept illustration.</span></figcaption>
</figure>

<p>
A galaxy survey is not a periodic box: the number density varies across the sky and with redshift, the line of sight changes from galaxy to galaxy and the measured power spectrum is the true one convolved with the survey window. <tt>FML/Survey</tt> (namespace FML::SURVEY, requires FFTW, works with MPI and OpenMP) provides the standard pipeline for this:
</p>
<ul>
<li><tt>GalaxiesToBox.h</tt>: RA, Dec and redshift to Cartesian coordinates for a given cosmology.</li>
<li><tt>SurveyMesh.h</tt>: weighted catalogue points in physical coordinates, a padded (non-periodic) geometry, CIC painting and MPI routing from any initial distribution of the catalogue over tasks.</li>
<li><tt>SurveyPower.h</tt>: the FKP-weighted monopole, quadrupole and hexadecapole with a local (Yamamoto) or fixed line of sight.</li>
<li><tt>SurveyWindow.h</tt>: the window multipoles $Q_\ell(s)$ up to $\ell = 8$ from the randoms and the matrix that convolves a theory $P_\ell(k)$ with the window, with an optional integral constraint.</li>
<li><tt>SurveyReconstruction.h</tt>: BAO reconstruction in a survey (RecIso and RecSym) with a conjugate-gradient solver.</li>
</ul>
<figure class="science-figure">
  <img src="images/survey-window-reconstruction.svg" alt="Survey clustering workflow from masked galaxy catalogue and local line of sight through Yamamoto multipoles, window convolution, and BAO reconstruction">
  <figcaption>The measured multipoles are shaped by the survey geometry; reconstruction estimates and removes large-scale bulk flows before remeasurement.</figcaption>
</figure>

<p>
<b>Pencils.</b> Compiled with <tt>-DFML_SURVEY_PENCIL_MESH</tt> the survey meshes are <a href="pencilgrid.php" style="display: inline">$\texttt{PencilGrid}$s</a> on the pencils of <tt>FML::domain</tt> (<tt>FML::set_domain(FML::Domain::uniform_pencil(px, py))</tt>); the multipoles, windows and reconstruction work the same way.
</p>

<header class="entry-header text-center">
  <h1 class="entry-title" id="Geometry">Coordinates, weights and normalization</h1>
</header>
<hr/>

<p>
All lengths (positions, box size, smoothing radius, separations) are physical and in the same units; $k$ is in the inverse unit and $P$ in length cubed. The box is a padded FFT embedding of the survey, not a periodic box, and the observer is a physical point:
</p>
<div style="text-align: center">
  <div style="width:700px;display: inline-block;text-align: left">
    <pre class="prettyprint">
#include &lt;FML/Survey/SurveyPower.h&gt;
using namespace FML::SURVEY;

Geometry geometry;
geometry.nmesh   = 128;
geometry.boxsize = 1600.;                  // Mpc/h
geometry.minimum = {-400., -800., -200.};  // corner of the box
geometry.observer = {0., 0., 0.};
// geometry.local_los = false; geometry.fixed_los = {0., 0., 1.};   // plane-parallel instead

std::vector&lt;CatalogPoint&gt; data, randoms;   // position, effective weight, selection density nbar
// e.g. weight = systematic_weight * fkp_weight(nbar, P0) with w_FKP = 1 / (1 + nbar P0)</pre>
  </div>
</div>
<p>
With data weights $w_g$, random weights $w_r$ and the selection density $\bar n$ the FKP field and its normalization are
$$F(\vec x) = \sum_{g} w_g\,\delta_D(\vec x - \vec x_g) - \alpha\sum_r w_r\,\delta_D(\vec x - \vec x_r),\qquad \alpha = \frac{\sum_g w_g}{\sum_r w_r},\qquad I_2 = \alpha\sum_r\bar n_r w_r^2 \approx \int d^3x\,\bar n^2w^2.$$
The normalization is explicit: $I_2$ can be overridden when the selection normalization is estimated elsewhere, and no number density is ever inferred from a bounding-box volume.
</p>

<header class="entry-header text-center">
  <h1 class="entry-title" id="Power">Fourier-space multipoles (FKP / Yamamoto)</h1>
</header>
<hr/>

<p>
The multipoles with a local line of sight $\hat\eta$ (the direction to the first galaxy of the pair, Yamamoto et al. 2006) are estimated as
$$\hat P_\ell(k) = \frac{2\ell+1}{I_2}\int\frac{d\Omega_k}{4\pi}\,{\rm Re}\left[A_0(\vec k)A_\ell^*(\vec k)\right] - N_\ell,\qquad A_\ell(\vec k) = \int d^3x\,F(\vec x)\,\mathcal{L}_\ell(\hat k\cdot\hat\eta(\vec x))\,e^{-i\vec k\cdot\vec x}.$$
The Legendre polynomial is expanded into products of $\hat k_i$ and $\hat\eta_i(\vec x)$ so that $A_\ell$ is a sum of FFTs of the field weighted by line-of-sight tensors (<a href="https://arxiv.org/abs/1505.05341" style="display: inline">Bianchi et al. 2015</a>, <a href="https://arxiv.org/abs/1506.02729" style="display: inline">Scoccimarro 2015</a>). The shot noise $N_\ell$ is computed from the actual catalogue (weights squared for the data, $\alpha^2$ times weights squared for the randoms, with the line-of-sight tensor moments), including the aliased images of CIC-painted Poisson noise (<a href="https://doi.org/10.1086/427087" style="display: inline">Jing 2005</a>):
$$N(\vec k) = N_{\rm cont}\prod_i\left[1 - \tfrac{2}{3}\sin^2\left(\tfrac{k_ih}{2}\right)\right]\Big/W^2_{\rm CIC}(\vec k).$$
</p>
<div style="text-align: center">
  <div style="width:700px;display: inline-block;text-align: left">
    <pre class="prettyprint">
PowerOptions options;
options.edges = {0., .01, .02, .03, .04, .05};   // k bins
options.ells  = {0, 2, 4};
auto power = estimate_power(data, randoms, geometry, options);
// power.k, power.nmodes, power.power[i] (multipole options.ells[i]), power.shotnoise[i], power.alpha, power.i2</pre>
  </div>
</div>

<header class="entry-header text-center">
  <h1 class="entry-title" id="Window">The survey window and the window matrix</h1>
</header>
<hr/>

<p>
The measured multipoles are the true ones convolved with the window. In configuration space the convolution is a product (Wilson et al. 2017):
$$\hat\xi_\ell(s) = \sum_{L,\ell'}\begin{pmatrix}\ell & L & \ell'\\0&0&0\end{pmatrix}^2(2\ell+1)\,Q_L(s)\,\xi_{\ell'}(s),\qquad Q_L(s) = \frac{2L+1}{I_2}\left\langle\int d^3x\,W(\vec x)W(\vec x + \vec s)\,\mathcal{L}_L(\hat s\cdot\hat\eta)\right\rangle_{\hat s},$$
with $W = \alpha\,n_r$ the selection field of the randoms. The window multipoles are computed with FFTs of the randoms' line-of-sight tensors (one FFT per tensor), with the self-pairs of each random removed exactly. On the mesh the separation vectors in a shell are not isotropic; the shell average uses ${\rm mean}(GF) - {\rm mean}(G)[{\rm mean}(F) - \langle F\rangle_{\rm sphere}]$, which is exact for a pair function that is isotropic on the shell (a plain average leaked the large isotropic part into $Q_4$&ndash;$Q_8$).
</p>
<p>
The window matrix maps theory bandpowers to the convolved multipoles: the input $P_\ell$ (piecewise constant in fine $k$ bins) is Hankel-transformed to $\xi_\ell$ at 16 Gauss&ndash;Legendre nodes per separation bin, multiplied by the window as above, and transformed back with spherical Bessel kernels. The optional global integral constraint subtracts $C\,Q_\ell(s)$ with $C$ chosen such that the convolved monopole vanishes at $k = 0$.
</p>
<div style="text-align: center">
  <div style="width:700px;display: inline-block;text-align: left">
    <pre class="prettyprint">
#include &lt;FML/Survey/SurveyWindow.h&gt;
std::vector&lt;double&gt; separation_edges;            // increasing, last &lt;= boxsize / 2
auto window = estimate_window(randoms, geometry, separation_edges, power.alpha, power.i2, 8);
std::vector&lt;double&gt; input_edges;                 // finer than the measured bins and extending beyond them
for (int i = 0; i &lt;= 200; ++i) input_edges.push_back(0.005 * i);
auto matrix = make_binned_window_matrix(window, input_edges, options.edges, {0,2,4}, {0,2,4}, true);
auto predicted = matrix.apply(theory_bandpowers);   // P0 bins, then P2, then P4</pre>
  </div>
</div>

<header class="entry-header text-center">
  <h1 class="entry-title" id="Recon">BAO reconstruction</h1>
</header>
<hr/>

<figure class="cinematic-figure" id="cinematic-11-sound-horizon">
  <a href="images/cinematic/11-sound-horizon.jpg" aria-label="View full-size illustration: The sound horizon">
    <img src="images/cinematic/11-sound-horizon.jpg" width="1774" height="887" loading="lazy" decoding="async" alt="An early acoustic enhancement and a later galaxy distribution with a faint guide to the preferred pair separation">
  </a>
  <figcaption><strong>The sound horizon.</strong> Conceptual acoustic propagation and the later preferred galaxy separation. The shell highlights a statistical scale, not a sharp shell in every late-time realization. <span class="cinematic-credit">AI-generated concept illustration.</span></figcaption>
</figure>

<p>
Reconstruction (Eisenstein et al. 2007) undoes the large-scale bulk flows that smear the BAO peak. The tracer density $\delta_t = D_s/(\alpha R_s) - 1$ is formed from the separately Gaussian-smoothed data and random meshes, and the Zel'dovich displacement $\vec\Psi = \nabla\phi$ including the linear redshift-space distortion is found by solving
$$-\nabla\cdot\left[\left(\mathbb{I} + \frac{f}{b}\,\hat n\hat n\right)\nabla\phi\right] = \frac{\delta_t}{b}$$
with the local radial direction $\hat n$ (or a fixed line of sight). The operator is discretized with a positive-definite finite-volume stencil (forward gradients paired with their adjoint divergence) and solved with conjugate gradients preconditioned by the Fourier Laplacian; the report records convergence, iterations, the final residual and the removed source mean. Then
</p>
<ul>
<li>the data move by $-\vec\Psi - f(\vec\Psi\cdot\hat n)\hat n$;</li>
<li>in <b>RecSym</b> the randoms move by the same displacement, in <b>RecIso</b> by $-\vec\Psi$ only.</li>
</ul>
<p>
The reconstructed power spectrum is $P(\vec k)$ of $F = D_{\rm shifted} - \alpha R_{\rm shifted}$, normalized with the original randoms.
</p>
<div style="text-align: center">
  <div style="width:700px;display: inline-block;text-align: left">
    <pre class="prettyprint">
#include &lt;FML/Survey/SurveyReconstruction.h&gt;
ReconstructionOptions rec;
rec.bias = 2.;
rec.growth_rate = .8;
rec.smoothing_radius = 15.;
rec.convention = ReconstructionConvention::RecSym;
auto shifted = reconstruct(data, randoms, geometry, rec);
auto reconstructed = estimate_power(shifted.data, randoms, geometry, options, &amp;shifted.randoms);</pre>
  </div>
</div>

<header class="entry-header text-center">
  <h1 class="entry-title" id="Validation">Example and validation</h1>
</header>
<hr/>

<p>
<tt>Survey/workflow_example</tt> generates a synthetic Cartesian survey and writes the pre- and post-reconstruction multipoles, $Q_0$&ndash;$Q_8$ and a window matrix (<tt>make -C FML/Survey/workflow_example && mpirun -np 4 ./survey_workflow</tt>). The tests compare the FFT multipoles with a direct DFT, the window multipoles with explicit weighted random pairs (and a uniform cube with its analytic $Q_4$), the window convolution with an analytic Gaussian result, the Poisson shot noise of random catalogues, and the reconstruction solver against manufactured solutions and an independent spectral solution (slope $0.995$, correlation $0.99995$); they also check MPI ownership and order, masks, normalization and RecIso/RecSym.
</p>

<header class="entry-header text-center">
  <h1 class="entry-title" id="Convergence">External comparisons and window convergence</h1>
</header>
<hr/>

<p>The raw survey multipoles have now been compared with pypower under matched painting, line-of-sight and normalization choices. Window tests cover the full $\ell=0,2,4$ mixing blocks, finite bands, global integral constraint, manufactured angular products and mesh/padding refinement. Radial integration now splits at interpolation knots of the window table, fixing a quadrature error across changes of slope.</p>
<p>For a theory prediction, refine the input-band width and k support, the random-catalogue mesh, the radial window table and physical padding separately. No fixed number of cells guarantees accurate high multipoles. These fixtures do not certify arbitrary masks or production survey recovery. See <a href="https://github.com/HAWinther/FML/blob/master/FML/Survey/power_validation/RESULTS.md" style="display: inline">power comparisons</a> and <a href="https://github.com/HAWinther/FML/blob/master/FML/Survey/window_validation/RESULTS.md" style="display: inline">window convergence results</a>.</p>

<header class="entry-header text-center">
  <h1 class="entry-title" id="References">Method references and scope</h1>
</header>
<hr/>

<p>
The FFT multipole estimator follows <a href="https://arxiv.org/abs/1505.05341" style="display: inline">Bianchi et al. (2015)</a>, and correlation-space window mixing follows <a href="https://arxiv.org/abs/1511.07799" style="display: inline">Wilson et al. (2017)</a>. FML implements its own finite-band window matrix and compares selected estimator/window outputs with <a href="https://github.com/cosmodesi/pypower" style="display: inline">pypower</a>. These comparisons do not remove finite-mesh, radial-quadrature or input-band errors; see the <a href="https://github.com/HAWinther/FML/blob/master/FML/Survey/window_validation/README.md" style="display: inline">window convergence procedure</a>.
</p>
