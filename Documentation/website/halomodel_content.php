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
<a href="#Reaction">4. Native screened collapse and halo-model reactions</a><br>
<a href="#Axion">5. Standalone AxionHMcode</a><br>
<a href="#HOD">6. HOD mock galaxies (FML::HOD)</a><br>
<a href="#AssemblyBias">7. Decorated HOD and marginal satellite occupations</a><br>
<a href="#Baryons">8. Baryonification (FML::BARYONIFICATION)</a><br>
<a href="#EMantis">9. e-MANTIS emulators</a><br>
<a href="#Bacco">10. baccoemu emulators: matter, baryons and galaxy bias</a><br>
<a href="#References">11. Method references and scope</a><br>
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
<tr><td><tt>react</tt></td><td>Matched pseudo spectrum times <a href="https://arxiv.org/abs/1812.05594" style="display: inline">ReACT reaction</a></td><td>Native halo-model reaction. Solver support: flat LCDM, Hu&ndash;Sawicki $f(R)$ with $n=1$, CPL $w_0w_a$, normal-branch DGP on a LCDM background, coupled quintessence and the Fading Dark Sector; massive neutrinos are supported. Reactions stop at $z=2.5$ by default.</td></tr>
<tr><td><tt>emantis</tt></td><td><a href="https://arxiv.org/abs/2303.08899" style="display: inline">e-MANTIS f(R) boost</a> times matched GR power</td><td>Flat <tt>model = fr</tt>, <tt>fR_n = 1</tt>; <tt>emantis_gr</tt> selects EE2 (default), HMcode or Halofit. Nonzero |fR0|: 1e-7 to 1e-4. Trained redshifts through z=3; higher redshifts continue with the linear f(R)/GR ratio. See <a href="halomodel.php#EMantis" style="display: inline">ranges and conventions</a>.</td></tr>
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
</p>
<p>
<b>Coupled dark matter.</b> For coupled quintessence and the <a href="https://arxiv.org/abs/2611.17422" style="display: inline">Fading Dark Sector</a> the CDM particles have a field-dependent mass $m\propto e^{-\beta\phi}$ and feel an unscreened fifth force. With the CDM and baryons as one fluid (as in SimulationCode) the real collapses and the one-loop kernels use $G_{\rm eff}/G=1+2\beta^2f_c^2$ and an extra friction $f_c(-\beta\,d\phi/d\ln a)$, $f_c=\rho_c/\rho_{cb}$, in the target background. The real halos take $\sigma(M)$ and the growth from the target itself (the growth is scale independent), and their concentrations are lower when the late growth is larger. Checks: $c=c'=0$ equals LCDM HMcode-2020 to $10^{-4}$; the collapse model's linear growth equals the Boltzmann solver's to $2\times10^{-3}$; at the FDS best fit the reaction differs from 1 by less than 0.5% for $k\leq1\,h/$Mpc, and against LCDM with the same $A_s$ and matter-radiation equality the extra nonlinear boost at $k=1\,h$/Mpc is 1-2.5% for $z=0$-2. A pair of $64^3$ FDS and LCDM simulations already follows it to about 0.3% (<a href="nbodycode.php" style="display: inline">SimulationCode</a>, <tt>scripts/fds_vs_lcdm_simulations.py</tt>).
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
Its ingredients include the axion Jeans cutoff, a soliton-plus-NFW axion profile, clustered and unclustered axion fractions, and cold/axion cross power. The <tt>Basic</tt> and calibrated <tt>Dome2024</tt> variants are available. The standalone library has its own example and tests and is also exposed by the Einstein&ndash;Boltzmann solver as <tt>nonlinear = axionhmcode</tt> for <tt>model = axion</tt>. It requires supplied linear spectra (for example from axionCAMB), GSL, a flat late-time LCDM expansion, and has no massive-neutrino or baryonic-feedback model. The FML axion-like EDE background is different physics and does not supply the required axion transfer spectra. The Dome calibration bounds and extrapolation policy are documented in the <a href="https://github.com/HAWinther/FML/blob/master/FML/AxionHMcode/README.md">module README</a>.
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

<header class="entry-header text-center"><h1 class="entry-title" id="EMantis">e-MANTIS emulators</h1></header>
<p><tt>Emulator/EMantis/EMantis.h</tt> ports the prediction code and trained Gaussian processes of
<a href="https://gitlab.obspm.fr/e-mantis/e-mantis">e-MANTIS 1.4.0</a>.
Cite <a href="https://arxiv.org/abs/2303.08899">Saez-Casares, Rasera &amp; Li</a> for the f(R) power boost,
<a href="https://arxiv.org/abs/2410.05226">the halo-mass-function paper</a> for f(R)/wCDM HMFs,
and <a href="https://doi.org/10.5281/zenodo.7738362">the software</a> (also for the wCDM power boost, which has no dedicated paper in this release).
The API also exposes the sigma8 emulator and generic exported GP predictions. Training is not ported.</p>
<p>The solver computes P_NL,fR = P_NL,GR B_fR using a second GR solve with the same initial conditions.
The f(R) boost covers scale factors 0.25 to 1 and k up to 9.7 h/Mpc (held above; extrapolated toward unity below 0.029 h/Mpc).
Out-of-range cosmologies throw unless <tt>emantis_extrapolate_cosmo</tt> is enabled to clamp them.
The API's <tt>logfR0</tt> is <em>minus</em> log10|fR0|. SimulationCode uses an EE2 GR baseline for its nonlinear comparison column,
clamps cosmological inputs and returns linear theory above z=3; the solver instead continues the boost with linear evolution.
<a href="https://github.com/HAWinther/FML/blob/master/FML/Emulator/EMantis/README.md">The module notes</a>
record all parameter ranges and Python comparison tests. Agreement with Python at roughly 1e-10 measures the port, not physical emulator accuracy.</p>
<p>Cite <a href="https://doi.org/10.1093/mnras/stad3343">Saez-Casares, Rasera &amp; Li, MNRAS 527, 7242 (2024)</a> and
<a href="https://doi.org/10.1051/0004-6361/202450193">Saez-Casares, Rasera, Richardson &amp; Corasaniti, A&amp;A 691, A323 (2024)</a>.
e-MANTIS is distributed under the GPL (version 3 or later), and so is the port, like FML itself: <tt>Emulator/EMantis</tt> keeps the
licence text (<tt>COPYING</tt>) and the notices of the original
(<a href="https://github.com/HAWinther/FML/blob/master/FML/Emulator/EMantis/README.md#licence">licence notes</a>).</p>

<header class="entry-header text-center"><h1 class="entry-title" id="Bacco">baccoemu emulators: matter, baryons and galaxy bias</h1></header>
<p><tt>Emulator/Bacco/Bacco.h</tt> ports the prediction part of <a href="https://bitbucket.org/rangulo/baccoemu">baccoemu 2.3.0</a>
(Angulo, Zennaro, Aric&ograve;, Pellejero Ib&aacute;&ntilde;ez, Burger; MIT licence): neural-network emulators trained on the BACCO
simulations, with massive neutrinos and $w_0$&ndash;$w_a$ dark energy.</p>
<ul>
<li><tt>MatterPowerSpectrum</tt>: $\sigma_8$, $\sigma_{12}$ and $A_s$, the linear $P(k)$ of the cold or total matter
(<a href="https://arxiv.org/abs/2104.14568">Aric&ograve; et al. 2021</a>), the no-wiggle and BAO-smeared spectra, the nonlinear boost and
$P_{\rm NL}$ (<a href="https://arxiv.org/abs/2004.06245">Angulo et al. 2021</a>, or the wider ranges of
<a href="https://arxiv.org/abs/2303.05537">Aric&ograve; et al. 2023</a>), and the baryonic suppression of $P(k)$
(<a href="https://arxiv.org/abs/2011.15018">Aric&ograve; et al. 2021</a> or <a href="https://arxiv.org/abs/2506.18974">Burger et al. 2025</a>).</li>
<li><tt>MatterBispectrum</tt>: the baryonic suppression of the matter bispectrum (<a href="https://arxiv.org/abs/2506.18974">Burger et al. 2025</a>).</li>
<li><tt>LbiasExpansion</tt>: the 15 spectra of the second order Lagrangian bias expansion and the galaxy auto and cross spectra in real space,
$k\le0.75\,h$/Mpc (<a href="https://arxiv.org/abs/2101.12187">Zennaro et al.</a>).</li>
<li><tt>LbiasExpansionRSD</tt>: the same in redshift space (multipoles 0, 2, 4), the galaxy multipoles and $P(k,\mu)$ with fingers of god and
noise, $k\le0.64\,h$/Mpc (<a href="https://arxiv.org/abs/2207.06437">Pellejero Ib&aacute;&ntilde;ez et al. 2023</a>; LPT from an emulator of
<a href="https://arxiv.org/abs/2012.04636">velocileptors</a>), and the bias coevolution relations (<a href="https://arxiv.org/abs/2110.05408">Zennaro et al. 2022</a>).</li>
</ul>
<div style="text-align: center"><div style="max-width:800px;display:inline-block;text-align:left;width:100%"><pre class="prettyprint">#include &lt;FML/Emulator/Bacco/Bacco.h&gt;
using namespace FML::EMULATOR::BACCO;
Parameters p;
p.Omega_cold = 0.31; p.Omega_b = 0.049; p.sigma8_cold = 0.81;   // or p.Omega_m, p.A_s
p.h = 0.67; p.n_s = 0.965; p.M_nu = 0.06;
MatterPowerSpectrum emu;                                    // options: nonlinear_model, baryonic_model, ...
auto pnl = emu.nonlinear_pk(p, a, k, /*cold=*/false);       // total matter, (Mpc/h)^3
LbiasExpansionRSD lbz;
LbiasExpansionRSD::RSDParameters rsd; rsd.f_sat = 0.1; rsd.lambda_FoG = 2.5;
auto multipoles = lbz.galaxy_pk({b1, b2, bs2, blaplacian}, rsd, p, a, k);</pre></div></div>
<p>The trained networks are not distributed with FML: run <tt>Emulator/Bacco/get_data.sh</tt> once (it downloads them from bacco.dipc.org,
checks pinned SHA-256 sums and converts them with numpy and h5py; <tt>--matter</tt> fetches only the ~10 MB of matter emulators). Without
them the constructors throw an error that says how to get them. The predictions agree with the Python package to ~1e-15 (with the
networks evaluated in double precision in both) in 251 test cases;
<a href="https://github.com/HAWinther/FML/blob/master/FML/Emulator/Bacco/README.md">the module notes</a> list the parameter ranges.</p>

<header class="entry-header text-center">
  <h1 class="entry-title" id="References">Method references and scope</h1>
</header>
<hr/>

<p>
Decorated HOD assembly bias follows <a href="https://arxiv.org/abs/1512.03050" style="display: inline">Hearin et al. (2016)</a> and the Halotools Heaviside prescription. <tt>HOD/AssemblyBias.h</tt> decorates the central probability and the <em>marginal</em> satellite mean at fixed mass; callers supply calibrated conditional percentiles. Population-weighted first moments are preserved, rather than each realized galaxy count. See the <a href="https://github.com/HAWinther/FML/blob/master/FML/HOD/README.md" style="display: inline">HOD conventions and comparison scope</a>. Native halo-model reactions also cite <a href="https://arxiv.org/abs/2105.12114">the massive-neutrino extension</a> and <a href="https://arxiv.org/abs/2210.01094">ReACT v2</a>. Native halo-model reactions follow <a href="https://arxiv.org/abs/1812.05594" style="display: inline">Cataneo et al. (2019)</a> and <a href="https://arxiv.org/abs/2005.12184" style="display: inline">Bose et al. (2020)</a>; <a href="https://github.com/nebblu/ACTio-ReACTio" style="display: inline">ACTio-ReACTio</a> supplies comparison fixtures.
</p>
