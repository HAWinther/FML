<hr/>
<header class="entry-header text-center">
  <h1 class="entry-title">Table of contents</h1>
</header>
<hr/>

<p class="page-toc">
<a href="index.php">Back to the main page</a><br>
<a href="documentation.php">Back to the documentation</a><br>
<a href="#Collapse">1. Spherical collapse (FML::COSMOLOGY::SPHERICALCOLLAPSE)</a><br>
<a href="#Halomodel">2. Halo model and HMcode library</a><br>
<a href="#Nonlinear">3. Nonlinear power in the Einstein&ndash;Boltzmann Solver</a><br>
<a href="#Axion">4. Standalone AxionHMcode</a><br>
<a href="#HOD">5. HOD mock galaxies (FML::HOD)</a><br>
<a href="#Baryons">6. Baryonification (FML::BARYONIFICATION)</a><br>
<a href="#References">Method references and scope</a><br>
<a href="#AssemblyBias">Decorated HOD and marginal satellite occupations</a><br>
<a href="#Reaction">Native screened collapse and halo-model reactions</a><br>
</p>

<header class="entry-header text-center">
  <h1 class="entry-title" id="Collapse">Spherical collapse</h1>
</header>
<hr/>

<p>
A spherical top-hat overdensity $\delta$ of radius $R$ evolves (in terms of $y = r/R - 1$ or directly $\delta$) according to
$$\ddot\delta + 2H\dot\delta - \frac{4}{3}\frac{\dot\delta^2}{1+\delta} = 4\pi G_{\rm eff}\bar\rho_m\,\delta(1+\delta),$$
where $G_{\rm eff}$ can be modified in models beyond GR (the model is given by a <tt>SphericalCollapseModel</tt> providing $H(a)$, $\Omega_m(a)$ and $G_{\rm eff}/G$). The code integrates this together with the linear equation $\ddot\delta_L + 2H\dot\delta_L = 4\pi G_{\rm eff}\bar\rho_m\delta_L$, finds by root finding the initial amplitude for which the non-linear $\delta$ diverges exactly at the requested collapse redshift, and extracts the critical density $\delta_c(z) = \delta_L(z_{\rm coll})$ (1.686 in EdS), the virial overdensity $\Delta_{\rm vir}(z)$ and the times of non-linear onset, turnaround and virialization. It also provides the growth factor $D$ and growth rate $f = d\ln D/d\ln a$ of the model.
</p>

<header class="entry-header text-center">
  <h1 class="entry-title" id="Halomodel">The halo model and HMCode</h1>
</header>
<hr/>

<p>
In the halo model all mass is in halos, so the nonlinear power spectrum is the sum of a two-halo and a one-halo term,
$$P(k) = P_{2h}(k) + P_{1h}(k),\qquad P_{1h}(k) = \int dM\,n(M)\left(\frac{M}{\bar\rho}\right)^2|u(k|M)|^2,\qquad P_{2h}(k) = P_L(k)\left[\int dM\,n(M)\frac{M}{\bar\rho}b(M)u(k|M)\right]^2,$$
where $u(k|M)$ is the normalized Fourier transform of the halo profile (NFW by default, with the analytic transform), $n(M)$ the mass function and $b(M)$ the halo bias. The mass function is written in terms of the peak height $\nu = \delta_c/\sigma(M)$,
$$n(M)\,dM = \frac{\bar\rho}{M}f(\nu)\,d\nu,\qquad f_{\rm ST}(\nu) = A\sqrt{\frac{2q}{\pi}}\left[1 + (q\nu^2)^{-p}\right]e^{-q\nu^2/2}$$
(Sheth&ndash;Tormen by default) with $\sigma^2(M) = \int\frac{k^2dk}{2\pi^2}P_L(k)W^2(kR)$ and $\delta_c$, $\Delta_{\rm vir}$ from spherical collapse. The concentration&ndash;mass relation, multiplicity, bias and profile can be replaced by user functions. The general <tt>HaloModel</tt> supports a standard prescription and HMcode-2016; <tt>HMcode2020</tt> is a separate gravity-only prescription sharing the same halo-model building blocks. The single-redshift APIs return the linear, one-halo, two-halo and total components, while table APIs expose $P_{\rm NL}/P_L$ over time and scale.
</p>

<figure class="science-figure">
  <img src="images/halo-model-power.svg" alt="Schematic halo-model power decomposition showing two-halo power on large scales and one-halo power on small scales, with their total">
  <figcaption>Conceptual component shapes, not a calibrated prediction. HMcode modifies halo structure and the transition to improve agreement with simulations.</figcaption>
</figure>
<p>
The solver consumes a nonlinear model as a boost to the input linear spectrum,
$$P_{\rm NL}(k,z)=B(k,z)P_L(k,z),\qquad B(k,z)=\frac{P_{\rm NL}(k,z)}{P_L(k,z)}.$$
The same ratio is used in the solver's existing Limber lensing calculation. Method choice matters: the fitting functions have different neutrino handling and model domains, summarized below.
</p>

<header class="entry-header text-center">
  <h1 class="entry-title" id="Nonlinear">Nonlinear power models in the Einstein&ndash;Boltzmann Solver</h1>
</header>
<hr/>

<div class="table-responsive">
<table class="table table-striped table-condensed">
<thead><tr><th>Setting</th><th>Method</th><th>Model scope and notes</th></tr></thead>
<tbody>
<tr><td><tt>none</tt></td><td>Linear theory</td><td>Default; no nonlinear boost.</td></tr>
<tr><td><tt>halofit</tt></td><td><a href="https://arxiv.org/abs/1208.2701" style="display: inline">Takahashi et al. (2012) Halofit</a></td><td>Baseline fitting formula. Massive-neutrino correction is not applied; FML warns when neutrinos are massive.</td></tr>
<tr><td><tt>halofit_bird</tt></td><td>Halofit + <a href="https://arxiv.org/abs/1109.4416" style="display: inline">Bird et al. (2012)</a> neutrino terms</td><td>Takahashi-compatible Bird correction, in the CLASS-style form; linear neutrino effects remain in the supplied spectrum.</td></tr>
<tr><td><tt>hmcode2016</tt></td><td><a href="https://arxiv.org/abs/1505.07833" style="display: inline">HMcode-2015</a> with the neutrino update of <a href="https://arxiv.org/abs/1602.02154" style="display: inline">Mead et al. (2016)</a></td><td>As CAMB <tt>mead2016</tt>; includes massive-neutrino treatment. The library prescription is built on the general halo model.</td></tr>
<tr><td><tt>hmcode2020</tt></td><td><a href="https://arxiv.org/abs/2009.01858" style="display: inline">HMcode-2020 (Mead et al. 2021)</a></td><td>Gravity-only prescription with cold-matter variances, neutrino corrections, BAO damping, halo bloating and the smoothed one/two-halo transition. Solver accepts LCDM and smooth w0wa backgrounds, including curvature; other models are rejected.</td></tr>
<tr><td><tt>react</tt></td><td>Matched pseudo spectrum times <a href="https://arxiv.org/abs/1812.05594" style="display: inline">ReACT reaction</a></td><td>Native halo-model reaction. Solver support: flat LCDM, Hu&ndash;Sawicki $f(R)$ with $n=1$, CPL $w_0w_a$, and normal-branch DGP on a LCDM background; massive neutrinos are supported. Reactions stop at $z=2.5$ by default.</td></tr>
<tr><td><tt>axionhmcode</tt></td><td>Native mixed cold/axion halo model</td><td><tt>model = axion</tt>; axion/cold spectra enter separately. The Dome calibration is limited in mass, fraction and redshift; extrapolation must be assessed explicitly.</td></tr>
<tr><td><tt>ee2</tt></td><td>EuclidEmulator2 nonlinear boost</td><td>Flat LCDM or w0wa within the emulator domain. Tabulated over $0\le z\le10$; linear above $z=10$.</td></tr>
<tr><td><tt>jbd_boost</tt></td><td>Brans–Dicke boost from a matched GR nonlinear spectrum</td><td><tt>model = jbd</tt>; GR baseline selected by <tt>jbd_boost_gr</tt>, EuclidEmulator2 by default.</td></tr>
</tbody>
</table>
</div>
<p>
ReACT uses $P_{\rm NL,target}(k,z)=R(k,z)P_{\rm NL,pseudo}(k,z)$, where the pseudo spectrum is HMcode-2020 in a matched GR cosmology. It uses FML's screened spherical collapse, halo model and modified-gravity one-loop kernels; above its redshift limit, FML explicitly uses the matched baseline with $R=1$. Nonlinear options add the fourth matter-power output column; the linear CDM+baryon column is unchanged. For build requirements, validation ranges and scientific limits, see <a href="https://github.com/HAWinther/FML/blob/master/FML/Cosmology/Spectra/NONLINEAR.md">the nonlinear-method notes</a> and <a href="https://github.com/HAWinther/FML/blob/master/FML/HaloModel/README.md">ReACT documentation</a>.
</p>

<header class="entry-header text-center">
  <h1 class="entry-title" id="Reaction">Native screened collapse and halo-model reactions</h1>
</header>
<hr/>

<p>
<tt>ScreenedSphericalCollapse.h</tt> solves collapse with a model-dependent fifth force; <tt>HaloModelReaction.h</tt> combines target and matched-pseudo halo terms. <tt>PerturbationTheory/MGOneLoop.h</tt> supplies time-dependent modified-gravity kernels for the one-loop matching scale. These form the native <tt>nonlinear = react</tt> solver path; no external ReACT runtime backend is required.
</p>
<p>
The reaction multiplies a matched HMcode-2020 pseudo spectrum. Comparisons with a pinned ACTio-ReACTio build cover selected GR, CPL, $f(R)$ and nDGP cases, with documented environment and third-order-kernel conventions. In particular, the upstream and corrected $f(R)$ third-order kernels are distinct choices; reference-code agreement alone is not a new accuracy calibration. Above the configured reaction redshift range, the solver uses the matched baseline with $R=1$.
</p><p>See <a href="https://github.com/HAWinther/FML/blob/master/FML/HaloModel/README.md" style="display: inline">the collapse/reaction API and reference comparisons</a> and <a href="https://github.com/HAWinther/FML/blob/master/FML/Cosmology/Spectra/NONLINEAR.md" style="display: inline">nonlinear method scope</a>.</p>

<header class="entry-header text-center">
  <h1 class="entry-title" id="Axion">Standalone AxionHMcode for mixed cold and ultralight-axion matter</h1>
</header>
<hr/>

<p>
<a href="https://github.com/HAWinther/FML/tree/master/FML/AxionHMcode">FML/AxionHMcode</a> is a separate native C++17 implementation of the mixed cold/ultralight-axion halo model of <a href="https://arxiv.org/abs/2209.13445" style="display: inline">Vogt et al. (2022)</a>, with the <a href="https://arxiv.org/abs/2409.11469" style="display: inline">Dome et al. (2024)</a> mass-relation and smoothing calibration. It takes separate linear CDM+baryon and axion auto-spectra at the requested redshift and returns nonlinear cold, cold&ndash;axion cross, axion and total power. The total is
</p>
$$P_{\rm total}=(1-f)^2P_{\rm cold}+2f(1-f)P_{\rm cold,axion}+f^2P_{\rm axion},\qquad f=\frac{\Omega_{\rm axion}}{\Omega_b+\Omega_{\rm cdm}+\Omega_{\rm axion}}.$$
<p>
Its ingredients include the axion Jeans cutoff, a soliton-plus-NFW axion profile, clustered and unclustered axion fractions, and cold/axion cross power. The <tt>Basic</tt> and calibrated <tt>Dome2024</tt> variants are available. This is a standalone library with its own example and tests; it is not currently an Einstein&ndash;Boltzmann Solver nonlinear option. It requires supplied linear spectra (for example from axionCAMB), GSL, a flat late-time LCDM expansion, and has no massive-neutrino or baryonic-feedback model. The FML axion-like EDE background is different physics and does not supply the required axion transfer spectra. The Dome calibration bounds and extrapolation policy are documented in the <a href="https://github.com/HAWinther/FML/blob/master/FML/AxionHMcode/README.md">module README</a>.
</p>
<figure class="science-figure">
  <img src="images/axionhmcode-components.svg" alt="AxionHMcode flow from supplied cold and axion linear spectra to nonlinear cold, cross, axion, and weighted total power, with a schematic soliton-to-NFW profile">
  <figcaption>Component flow and profile are conceptual. AxionHMcode currently consumes separate supplied linear spectra; this diagram is not numerical output.</figcaption>
</figure>

<header class="entry-header text-center">
  <h1 class="entry-title" id="HOD">HOD mock galaxies</h1>
</header>
<hr/>

<p>
A halo occupation distribution (HOD) populates a halo catalogue with galaxies. The built-in Zheng et al. (2007) model has the mean central and satellite occupations
$$\langle N_c\rangle(M) = \frac{1}{2}\left[1 + {\rm erf}\left(\frac{\log_{10}M - \log_{10}M_{\rm min}}{\sigma_{\log M}}\right)\right],\qquad \langle N_s|N_c = 1\rangle(M) = \left(\frac{M - M_0}{M_1}\right)^\alpha\ (M&gt;M_0).$$
Centrals are Bernoulli draws placed at the halo centre (with an optional velocity bias); satellites are Poisson draws (only if there is a central, unless <tt>require_central = false</tt>) placed either on an NFW profile truncated at the halo radius (by inverting the enclosed-mass CDF) or on randomly chosen member particles, with velocities from the halo dispersion times a satellite velocity bias. Any callable returning the two means from a halo can replace Zheng07.
</p>
<p>
Every random number comes from a counter-based Philox generator keyed by the halo ID and a purpose (central occupation, satellite count, positions, velocities, ...), so a mock is reproducible independently of the halo order, the MPI decomposition and the number of OpenMP threads. Each galaxy has the identity (halo ID, ordinal).
</p>
<div style="text-align: center">
  <div style="width:700px;display: inline-block;text-align: left">
    <pre class="prettyprint">
#include &lt;FML/HOD/HOD.h&gt;
using namespace FML::HOD;
std::vector&lt;Halo&gt; halos;           // ID, mass, radius, concentration, position, velocity, dispersion
PreparedCatalog prepared(std::move(halos));
Zheng07 occupation;
occupation.log10_mmin = 13.;
occupation.sigma_log_mass = .4;
occupation.m0 = 1e12;
occupation.m1 = 1e14;
occupation.alpha = 1.;
Options options;
options.seed = 1234;
options.boxsize = 1000.;
auto mock = populate(prepared, occupation, options);</pre>
  </div>
</div>
<p>
The example populates a synthetic distributed halo catalogue and measures kNN-CDFs and counts-in-cells of the galaxies (see <a href="statistics.php" style="display: inline">summary statistics</a>).
</p>

<header class="entry-header text-center">
  <h1 class="entry-title" id="AssemblyBias">Decorated HOD and marginal satellite occupations</h1>
</header>
<hr/>

<p>
<tt>HOD/AssemblyBias.h</tt> decorates the central probability and marginal satellite mean using a secondary-property percentile at fixed halo mass. The strengths lie in $[-1,1]$; positive values favor the high-percentile population. The split is the low-population fraction. Population-weighted first moments are conserved, rather than realized counts in a finite mock. The caller must supply a common conditional percentile calibration across MPI ranks.
</p><div style="text-align: center"><div style="max-width:800px;display:inline-block;text-align:left;width:100%"><pre class="prettyprint">#include &lt;FML/HOD/AssemblyBias.h&gt;
using namespace FML::HOD;
Zheng07 baseline;
HeavisideAssemblyBias assembly{.5, .4, .6};
auto occupation = [&amp;](const Halo &amp; h) {
    return assembly(baseline(h), calibrated_percentile(h));
};
auto mock = populate(prepared, occupation, options);</pre></div></div>
<p>Satellite decoration acts on the marginal mean $\mu=p_\mathrm{cen}\lambda_\mathrm{sat|cen}$, avoiding an unintended change in mean counts when the central probability changes. Central-required and independent sampling have distinct covariance conventions. See <a href="https://github.com/HAWinther/FML/blob/master/FML/HOD/README.md" style="display: inline">occupation contracts and percentile requirements</a>.</p>

<header class="entry-header text-center">
  <h1 class="entry-title" id="Baryons">Baryonification</h1>
</header>
<hr/>

<figure class="cinematic-figure" id="cinematic-08-baryonic-feedback">
  <a href="images/cinematic/08-baryonic-feedback.jpg" aria-label="View full-size illustration: Dark matter and baryonic feedback">
    <img src="images/cinematic/08-baryonic-feedback.jpg" width="1774" height="887" loading="lazy" decoding="async" alt="A small central galaxy and extended gas outflows embedded in a larger translucent matter halo">
  </a>
  <figcaption><strong>Dark matter and baryonic feedback.</strong> Conceptual halo components and gas redistribution by feedback; the illustration does not represent a hydrodynamic calculation by FML. <span class="cinematic-credit">AI-generated concept illustration.</span></figcaption>
</figure>

<p>
Baryonification (Schneider &amp; Teyssier 2015, Schneider et al. 2019) mimics the effect of baryonic feedback on a gravity-only simulation by displacing particles around halos. For each halo a reference (dark-matter-only) profile and a baryon-corrected profile with the same total mass are built; with $x = r/R_{200}$:
</p>
<ul>
<li>Reference: truncated NFW, $\rho\propto\left[(cx)(1+cx)^2(1 + (x/\epsilon)^2)^2\right]^{-1}$.</li>
<li>Gas: $\rho\propto(1 + x/\theta_{\rm co})^{-\beta}(1 + (x/\theta_{\rm ej})^2)^{-(7-\beta)/2}$ with $\beta = 3 - (M_c/M_{200})^\mu$.</li>
<li>Central galaxy: $\rho\propto r^{-2}e^{-(r/2R_h)^2}$, with stellar fractions $f_\star = A_\star(M_1/M_{200})^{\eta}$.</li>
<li>Collisionless matter (dark matter and satellite stars), optionally adiabatically relaxed: $r_f/r_i - 1 = A\left[(M_i/M_f)^n - 1\right]$.</li>
</ul>
<figure class="science-figure">
  <img src="images/baryonification.svg" alt="Schematic halo profile remapping from a dark-matter-only reference to gas, central galaxy, and collisionless components while matching enclosed mass">
  <figcaption>Illustrative radial mapping. Particle radii change so source and target enclosed masses match; particle IDs, masses, and velocities stay fixed.</figcaption>
</figure>
<p>
A particle at radius $r_i$ is moved to $r_f$ such that the enclosed masses match, $M_{\rm target}(r_f) = M_{\rm source}(r_i)$; the displacements of overlapping halos are added, evaluated at the original positions (periodic KD-tree searches). Velocities, masses and IDs are untouched. The profiles are normalized on a finite support (30 $R_{200}$ by default) and the implementation is header-only and OpenMP parallel; it is a finite-support implementation of the profile family, not a calibrated reproduction of the published power-spectrum suppression.
</p>
<div style="text-align: center">
  <div style="width:700px;display: inline-block;text-align: left">
    <pre class="prettyprint">
#include &lt;FML/Baryonification/Baryonification.h&gt;
using namespace FML::BARYONIFICATION;
HaloProperties halo{1e14, 0.7, 5.0};          // M200, R200, concentration
ModelParameters model;
model.gas_mass_scale = 1e14;
model.gas_ejection_radius = 4.0;
auto profiles = make_halo_profiles(halo, model, mean_density);
std::vector&lt;HaloDisplacement&gt; halos;
halos.push_back({{50.0, 50.0, 50.0}, profiles.mapping()});
Baryonifier correction(std::move(halos), boxsize);
correction.apply(particles.data(), particles.size());</pre>
  </div>
</div>
<p>
The N-body code (<a href="nbodycode.php" style="display: inline">SimulationCode</a>) can apply baryonification to its outputs.
</p>

<header class="entry-header text-center">
  <h1 class="entry-title" id="References">Method references and scope</h1>
</header>
<hr/>

<p>
Decorated HOD assembly bias follows <a href="https://arxiv.org/abs/1512.03050" style="display: inline">Hearin et al. (2016)</a> and the Halotools Heaviside prescription. <tt>HOD/AssemblyBias.h</tt> decorates the central probability and the <em>marginal</em> satellite mean at fixed mass; callers supply calibrated conditional percentiles. Population-weighted first moments are preserved, rather than each realized galaxy count. See the <a href="https://github.com/HAWinther/FML/blob/master/FML/HOD/README.md" style="display: inline">HOD conventions and comparison scope</a>. Native halo-model reactions follow <a href="https://arxiv.org/abs/1812.05594" style="display: inline">Cataneo et al. (2019)</a> and <a href="https://arxiv.org/abs/2005.12184" style="display: inline">Bose et al. (2020)</a>; <a href="https://github.com/nebblu/ACTio-ReACTio" style="display: inline">ACTio-ReACTio</a> supplies comparison fixtures.
</p>
