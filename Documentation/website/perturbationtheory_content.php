<hr/>
<header class="entry-header text-center">
  <h1 class="entry-title">Table of contents</h1>
</header>
<hr/>

<p class="page-toc">
<a href="index.php">Back to the main page</a><br>
<a href="documentation.php">Back to the documentation</a><br>
<a href="#Overview">1. Overview</a><br>
<a href="#FFTLog">2. FFTLog: Hankel transforms</a><br>
<a href="#SPT">3. One-loop standard perturbation theory</a><br>
<a href="#Fast">4. The fast algorithm (FFTLog for $P_{22}$, FFT convolution for $P_{13}$)</a><br>
<a href="#Direct">5. Direct quadrature and custom kernels</a><br>
<a href="#Validation">6. Validation</a><br>
<a href="#References">Method references and scope</a><br>
<a href="#RSD">TNS and EFTofLSS redshift-space spectra</a><br>
<a href="#Correlation">Correlation multipoles and survey predictions</a><br>
</p>

<header class="entry-header text-center">
  <h1 class="entry-title" id="Overview">Overview</h1>
</header>
<hr/>

<p>
This page covers two modules: <tt>FML/FFTLog</tt> (namespace FML::SOLVERS::FFTLog), fast spherical-Bessel (Hankel) transforms of logarithmically sampled functions, e.g. $P(k)\to\xi(r)$; and <tt>FML/PerturbationTheory</tt> (namespace FML::PERTURBATION), the one-loop real-space matter power spectrum of standard perturbation theory, either with a fast $\mathcal{O}(N\log N)$ algorithm built on FFTLog or by direct quadrature, which also accepts user-supplied (e.g. scale-dependent or modified-gravity) kernels.
</p>

<header class="entry-header text-center">
  <h1 class="entry-title" id="FFTLog">FFTLog: Hankel transforms</h1>
</header>
<hr/>

<p>
FFTLog computes integrals of the form described by <a href="https://doi.org/10.1016/0021-9991(78)90107-9" style="display: inline">Talman (1978)</a> and <a href="https://arxiv.org/abs/astro-ph/9905191" style="display: inline">Hamilton (2000)</a>. FML's original FFTLog implementation is adapted from <a href="https://github.com/slosar/FFTLog" style="display: inline">Slosar's FFTLog</a>, based on <a href="https://github.com/jwgcarlson/Copter" style="display: inline">Copter</a>; the spherical-transform path documents its mcfit-style conventions in the code.
$$\tilde a(k) = \int_0^\infty a(r)\,(kr)^{q}J_\mu(kr)\,k\,dr$$
for $a$ sampled on a logarithmic grid. Writing $a(r)r^{-q}$ as a discrete Fourier series in $\ln r$, each Fourier mode $r^{i\omega}$ transforms analytically,
$$\int_0^\infty r^{i\omega}(kr)^qJ_\mu(kr)\,k\,dr = k^{-i\omega}\,2^{q+i\omega}\frac{\Gamma\left(\frac{\mu+1+q+i\omega}{2}\right)}{\Gamma\left(\frac{\mu+1-q-i\omega}{2}\right)},$$
so the transform is an FFT, a multiplication by these coefficients and an inverse FFT. The bias $q$ improves convergence for power-law tails and the choice of $k_cr_c$ removes ringing. With $\mu = \ell + 1/2$ this gives the multipole correlation functions
$$\xi_\ell^m(r) = \int\frac{k^2dk}{2\pi^2}\,k^m P(k)\,j_\ell(kr).$$
</p>
<div style="text-align: center">
  <div style="width:700px;display: inline-block;text-align: left">
    <pre class="prettyprint">
#include &lt;FML/FFTLog/FFTLog.h&gt;
using namespace FML::SOLVERS::FFTLog;
Workspace workspace;                                          // keeps FFTW plans and coefficients
auto xi    = ComputeCorrelationFunction(k, pk, workspace);    // {r, xi(r)}
auto xi24  = ComputeXiLM(2, 4, k, pk, workspace, -1.0);       // ell = 2, k^4 weighting
auto back  = ComputePowerSpectrum(xi.first, xi.second, workspace);</pre>
  </div>
</div>
<p>
A workspace owns two aligned FFT buffers and two plans; repeated transforms of the same length reuse plans and the coefficient table. Use one workspace per thread.
</p>

<header class="entry-header text-center">
  <h1 class="entry-title" id="SPT">One-loop standard perturbation theory</h1>
</header>
<hr/>

<p>
In standard perturbation theory the density contrast is expanded in powers of the linear field,
$$\delta(\vec k) = \sum_n\int\frac{d^3q_1\cdots d^3q_n}{(2\pi)^{3(n-1)}}\delta_D(\vec k - \vec q_{1\cdots n})F_n(\vec q_1,\ldots,\vec q_n)\,\delta_L(\vec q_1)\cdots\delta_L(\vec q_n),$$
with the Einstein&ndash;de Sitter kernels, e.g.
$$F_2(\vec q,\vec p) = \frac{5}{7} + \frac{\mu}{2}\left(\frac{q}{p} + \frac{p}{q}\right) + \frac{2}{7}\mu^2,\qquad \mu = \hat q\cdot\hat p,$$
and $F_3$ from the standard recursion. The one-loop power spectrum is $P = P_L + P_{22} + P_{13}$ with
$$P_{22}(k) = 2\int\frac{d^3q}{(2\pi)^3}\,F_2^2(\vec q,\vec k - \vec q)\,P_L(q)P_L(|\vec k - \vec q|),\qquad P_{13}(k) = 6P_L(k)\int\frac{d^3q}{(2\pi)^3}\,F_3(\vec k,\vec q,-\vec q)\,P_L(q).$$
Redshift dependence enters only through the supplied $P_L(k,z)$ (the loops scale as its amplitude squared). These are plain SPT predictions without counterterms, bias, redshift-space terms or IR resummation.
</p>

<header class="entry-header text-center">
  <h1 class="entry-title" id="Fast">The fast algorithm</h1>
</header>
<hr/>

<p>
<b>$P_{22}$</b> (Schmittfull, Vlah &amp; McDonald 2016; McEwen et al. 2016, FAST-PT). Expanding $F_2^2$ in Legendre polynomials turns the convolution into a sum of products of correlation-function-like integrals $X_\ell^n(r) = \int dq\,q^{n+2}P(q)j_\ell(qr)/(2\pi^2)$ in configuration space,
$$\xi_{22}(r) = \tfrac{1219}{735}(X_0^0)^2 + \tfrac{1}{3}X_0^{-2}X_0^2 - \tfrac{124}{35}X_1^{-1}X_1^1 + \tfrac{1342}{1029}(X_2^0)^2 + \tfrac{2}{3}X_2^{-2}X_2^2 - \tfrac{16}{35}X_3^{-1}X_3^1 + \tfrac{64}{1715}(X_4^0)^2,$$
followed by one more transform $P_{22}(k) = 4\pi\int dr\,r^2j_0(kr)\xi_{22}(r)$. All transforms use FFTLog on a common grid. The $X$'s of an abruptly truncated spectrum ring and $P_{22}$ is a cancellation among their products, so the spectrum is tapered smoothly to zero over the outer 10% of the logarithmic range with the FAST-PT window $x - \sin(2\pi x)/(2\pi)$ (without it $P_{22}$ was off by a factor ~5 at $k = 0.1\,h$/Mpc for an Eisenstein&ndash;Hu spectrum).
</p>
<p>
<b>$P_{13}$</b>. The angular integral of $F_3$ is analytic, leaving a one-dimensional kernel that depends only on $q/k$:
$$P_{13}(k) = \frac{k^2P_L(k)}{1008\pi^2}\int d\ln q\;q\,P_L(q)\,K(q/k),$$
with $K(1) = -88$, $K(0) = -168$ and $K(\infty) = -488/5$. On a logarithmic grid this is a convolution, evaluated with a zero-padded (linear, not circular) FFT. Every fast call also checks at least three output modes against direct quadrature and throws if they disagree.
</p>
<div style="text-align: center">
  <div style="width:700px;display: inline-block;text-align: left">
    <pre class="prettyprint">
#include &lt;FML/PerturbationTheory/FastOneLoop.h&gt;
using namespace FML::PERTURBATION;
FastEdS workspace;
FastOptions options;                 // taper, padding, reference checks
// k: strictly increasing, log-spaced, at least 64 samples
auto result = workspace.evaluate(k, P_linear, output_kmin, output_kmax, options);
// result.k, result.linear, result.p22, result.p13, result.loop, result.total
// result.max_reference_relative_error</pre>
  </div>
</div>

<header class="entry-header text-center">
  <h1 class="entry-title" id="Direct">Direct quadrature and custom kernels</h1>
</header>
<hr/>

<p>
<tt>OneLoop.h</tt> (no FFTW needed) integrates $P_{22}$ directly over the triangle $(q,|\vec k - \vec q|)$ with composite logarithmic Gauss&ndash;Legendre quadrature and $P_{13}$ with an independent one-dimensional quadrature. It accepts any isotropic scalar kernels, e.g. for scale-dependent growth or modified gravity:
</p>
<div style="text-align: center">
  <div style="width:700px;display: inline-block;text-align: left">
    <pre class="prettyprint">
#include &lt;FML/PerturbationTheory/OneLoop.h&gt;
DirectOptions integration;
integration.order = 12;          // Gauss-Legendre order
integration.log_step = 0.15;     // maximum log segment width
Spectrum P = [](double k) { return /* linear power */; };

Kernels model;
model.name = "my-model/z=1";
model.f2 = [](double q, double p, double mu_qp) { return /* symmetrized F2 */; };
model.f3_collapsed = [](double k, double q, double mu_kq) { return /* symmetrized F3(k, q, -q) */; };
auto prediction = direct_one_loop(output_k, P, qmin, qmax, integration, model);</pre>
  </div>
</div>

<header class="entry-header text-center">
  <h1 class="entry-title" id="Validation">Validation</h1>
</header>
<hr/>

<p>
The direct quadrature agrees with FAST-PT (<tt>one_loop_dd</tt>, version 4.1.1) to $10^{-3}$ for an Eisenstein&ndash;Hu $\Lambda$CDM spectrum at $k = 0.02$&ndash;$1\,h$/Mpc ($4\cdot 10^{-4}$ for a CLASS spectrum), and the fast path agrees with the direct one to $2\cdot 10^{-4}$ for every mode in $0.02$&ndash;$1\,h$/Mpc. The custom-kernel path with the EdS $F_3$ from the standard recursion reproduces the analytic $P_{13}$ kernel; the tests also check the analytic $P_{22}\propto k^4$ and $P_{13}$ low-$k$ limits, amplitude scaling, plan reuse and thread safety.
</p>

<header class="entry-header text-center">
  <h1 class="entry-title" id="RSD">TNS and EFTofLSS redshift-space spectra</h1>
</header>
<hr/>

<p>
<tt>TNS.h</tt> computes one-loop density/velocity spectra, nonlinear bias terms and the TNS $A+B$ corrections. Growth, bias and Gaussian/Lorentzian fingers-of-God parameters can vary without rebuilding the loop tables. <tt>EFTofLSS.h</tt> supplies CLASS-PT-style biased-tracer multipoles, IR resummation, counterterms and stochastic terms, with separate bias-monomial tables. Wavenumbers and the counterterm conventions must match the chosen model.
</p><div style="text-align: center"><div style="max-width:800px;display:inline-block;text-align:left;width:100%"><pre class="prettyprint">#include &lt;FML/PerturbationTheory/TNS.h&gt;
using namespace FML::PERTURBATION;
TNS tns(k_lin, P_lin, k_out);                 // all loop tables (bias and f independent) on k_out
TNS::Parameters p;                            // f, b1, b2, bs2, b3nl, sigma_v [Mpc/h], N, fog, AB
p.f = 0.76; p.b1 = 2.0; p.b2 = 0.5; p.bs2 = -4.0 / 7.0 * (p.b1 - 1); p.b3nl = 32.0 / 315.0 * (p.b1 - 1);
p.sigma_v = 4.0; p.fog = TNS::FoG::Lorentzian;  // or Gaussian, None
auto pkmu = tns.pkmu(0.5, p);                 // P(k, mu)
auto pell = tns.multipoles(p);                // P_0, P_2, P_4 (Gauss-Legendre in mu)
auto &amp; T = tns.tables();                      // P22/P13 of dd, dt, tt; bias terms; A_n, B_n by powers of mu</pre></div></div>
<p>These models return $P_0,P_2,P_4$. FAST-PT and CLASS-PT fixtures test selected loop terms and multipoles; they do not establish an observational fitting range. <a href="https://github.com/HAWinther/FML/blob/master/FML/PerturbationTheory/README.md#eftoflss-class-pt" style="display: inline">The EFTofLSS example</a> specifies the bias basis, no-wiggle split, resummation and units.</p>

<header class="entry-header text-center">
  <h1 class="entry-title" id="Correlation">Correlation multipoles and survey predictions</h1>
</header>
<hr/>

<p>
<tt>CorrelationMultipoles.h</tt> transforms signed $P_0,P_2,P_4$ spectra with reusable FFTLog workspaces:
$$\xi_\ell(s)=\frac{i^\ell}{2\pi^2}\int_{k_\min}^{k_\max}dk\,k^2P_\ell(k)W(k)j_\ell(ks).$$
It includes the quadrupole sign, a configurable edge taper and optional Gaussian damping. It neither extrapolates the supplied spectrum nor removes finite-band ringing automatically. Refine the k range, sampling and taper for the separation interval of interest.
</p><p>For a survey power prediction, evaluate the model on the input bands of the <a href="survey.php#Window" style="display: inline">window matrix</a> and apply that matrix directly. It already includes the band-to-correlation and correlation-to-power operations. See <a href="https://github.com/HAWinther/FML/blob/master/FML/PerturbationTheory/CORRELATION_MULTIPOLES.md" style="display: inline">the transform guide</a> and <a href="https://github.com/HAWinther/FML/blob/master/FML/Survey/window_validation/README.md#applying-this-to-perturbative-theory" style="display: inline">the theory-to-survey recipe</a>.</p>

<header class="entry-header text-center">
  <h1 class="entry-title" id="References">Method references and scope</h1>
</header>
<hr/>

<p>
The redshift-space TNS model follows <a href="https://arxiv.org/abs/1006.0699" style="display: inline">Taruya, Nishimichi &amp; Saito (2010)</a>. The one-loop EFTofLSS multipoles use the conventions of <a href="https://arxiv.org/abs/2004.10607" style="display: inline">Chudaykin et al. (2020), CLASS-PT</a>. FML implements these independently and compares selected terms with FAST-PT and CLASS-PT fixtures; the <a href="https://github.com/HAWinther/FML/blob/master/FML/PerturbationTheory/README.md" style="display: inline">module documentation</a> specifies bias, IR-resummation, noise and unit conventions. <tt>CorrelationMultipoles.h</tt> transforms these spectra using the FFTLog method of <a href="https://arxiv.org/abs/astro-ph/9905191" style="display: inline">Hamilton (2000)</a>; finite-band support and tapering remain numerical choices requiring convergence checks.
</p>
