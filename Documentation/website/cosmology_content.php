<hr/>
<header class="entry-header text-center">
  <h1 class="entry-title">Table of contents</h1>
</header>
<hr/>

<p class="page-toc">
<a href="index.php">Back to the main page</a><br>
<a href="documentation.php">Back to the documentation</a><br>
<a href="#Overview">1. Solver overview</a><br>
<a href="#Background">2. Background cosmology</a><br>
<a href="#Recombination">3. Recombination and reionization</a><br>
<a href="#Perturbations">4. Linear scalar perturbations</a><br>
<a href="#Spectra">5. CMB, matter spectra and key equations</a><br>
<a href="#Lensing">6. Lensing, nonlinear power and two-point functions</a><br>
<a href="#Tensors">7. Tensor and isocurvature modes</a><br>
<a href="#Models">8. Complete model catalogue</a><br>
<a href="#Usage">9. Running the solver</a><br>
<a href="#Validation">10. Validation against CLASS and CAMB</a><br>
<a href="#References">Method references and scope</a><br>
<a href="#Axion">Mixed ultralight-axion dark matter</a><br>
</p>

<header class="entry-header text-center">
  <h1 class="entry-title" id="Overview">Overview: an Einstein&ndash;Boltzmann solver</h1>
</header>
<hr/>

<p>
<tt>FML/Cosmology/EinsteinBoltzmannSolver</tt> is a C++17 program (<tt>cmb</tt>) and reusable library for the background expansion, recombination, linear scalar/tensor/isocurvature perturbations, CMB spectra, lensing and matter power. Its model registry covers standard species, dark-energy fields and fluids, modified gravity and interacting sectors. Shared-physics predictions are compared with CLASS, hi_class and CAMB. The original teaching solver is kept separately in <tt>Cosmology/Legacy</tt>.
</p>

<p>The pipeline is</p>
<div style="text-align: center">
  <div style="display: inline-block;text-align: left">
    <pre class="prettyprint">
CosmologyParameters -> BackgroundCosmology -> RecombinationHistory -> Perturbations -> PowerSpectrum -> Lensing, TwoPoint
                              ^                                             ^
                    Model = species + gravity  (lcdm, w0wa, musigma, dgp, quintessence, jbd, fr, coupled_quintessence,
                                                fading_dark_sector, dark_scattering, covariant_dark_scattering, axion,
                                                cubic_galileon, horndeski, horndeski_table, ede, ide,
                                                symmetron, dilaton, mbeta)</pre>
  </div>
</div>
<figure class="science-figure">
  <img src="images/boltzmann-pipeline.svg" alt="Einstein-Boltzmann calculation from species and gravity through background, recombination, perturbations, source functions, CMB spectra, matter power, and lensing">
  <figcaption>Species and gravity models feed the background and perturbation equations; projected sources produce the observables.</figcaption>
</figure>
<p>
Internally everything is in SI units with the time variable $x = \ln a$, in the conformal Newtonian gauge
$$ds^2 = a^2(\eta)\left[-(1+2\Psi)d\eta^2 + (1+2\Phi)\delta_{ij}dx^i dx^j\right]$$
(the Dodelson/Callin conventions). The parameter fits of <a href="inference.php" style="display: inline">CosmologyFitting</a> call the solver directly.
</p>

<header class="entry-header text-center">
  <h1 class="entry-title" id="Background">Background cosmology</h1>
</header>
<hr/>

<p>
The expansion rate follows from the Friedmann equation summed over the species of the model,
$$H^2(a) = H_0^2\left[\Omega_{b}a^{-3} + \Omega_{\rm CDM}a^{-3} + \Omega_\gamma a^{-4} + \Omega_\nu a^{-4} + \frac{\rho_{\nu,\rm massive}(a)}{\rho_{c,0}} + \Omega_K a^{-2} + \Omega_{\rm DE}(a)\right],$$
where the massive neutrinos are integrated over the Fermi&ndash;Dirac distribution on a Gauss&ndash;Laguerre momentum grid and the dark-energy term is whatever the model provides (a cosmological constant, $w_0w_a$, a scalar field solved with its Klein&ndash;Gordon equation, ...). In a modified-gravity model the gravity object returns $E^2(x) = H^2/H_0^2$ from the species densities, e.g. the DGP Friedmann equation $H^2 - \epsilon H/r_c = 8\pi G\rho/3$. From $H$ the code tabulates the conformal Hubble rate $\mathcal{H} = aH$ and its derivatives, conformal time $\eta(x) = \int c\,dx/\mathcal{H}$, cosmic time, and the comoving, angular-diameter and luminosity distances with the curvature functions
$$r(\chi) = \begin{cases}\sin(\sqrt{|K|}\chi)/\sqrt{|K|} & K&gt;0\\ \chi & K=0\\ \sinh(\sqrt{|K|}\chi)/\sqrt{|K|} & K&lt;0\end{cases}$$
The primordial helium fraction $Y_p$ is either a parameter or taken from standard BBN (the PArthENoPE table, as CLASS) for the given $\omega_b$ and $N_{\rm eff}$.
</p>

<header class="entry-header text-center">
  <h1 class="entry-title" id="Recombination">Recombination and reionization</h1>
</header>
<hr/>

<p>
The free electron fraction $X_e = n_e/n_H$ is computed with RECFAST 1.5 by default (effective three-level hydrogen and helium atoms with the fudge factors and the Gaussian correction to $K_H$, triplet helium, and the baryon temperature evolved with Compton heating). The simpler model of the course is also available (<tt>recombination = peebles</tt>): the Saha equation
$$\frac{X_e^2}{1-X_e} = \frac{1}{n_b}\left(\frac{m_e T_b}{2\pi}\right)^{3/2}e^{-\epsilon_0/T_b}$$
while $X_e \approx 1$, then the Peebles equation
$$\frac{dX_e}{dx} = \frac{C_r(T_b)}{H}\left[\beta(T_b)(1-X_e) - n_H\alpha^{(2)}(T_b)X_e^2\right],$$
where $C_r$ is the probability that an electron in the $n=2$ state reaches the ground state before being ionized. Reionization is added with the CAMB/CLASS $\tanh$ model. From $X_e$ follow the optical depth $\tau(x) = \int_x^0 n_e\sigma_T a\,c\,dx'/\mathcal{H}$, the visibility function $\tilde g = -\tau' e^{-\tau}$, the redshift of last scattering $z_*$, the drag epoch and the sound horizon
$$r_s(\eta) = \int_0^\eta c_s\,d\eta',\qquad c_s = \frac{c}{\sqrt{3(1+R)}},\quad R = \frac{3\rho_b}{4\rho_\gamma}.$$
</p>

<header class="entry-header text-center">
  <h1 class="entry-title" id="Perturbations">Linear perturbations</h1>
</header>
<hr/>

<p>
Each Fourier mode $k$ is evolved from deep in the radiation era (adiabatic initial conditions set by the curvature perturbation $\mathcal{R}$) to today. A species provides its variables and equations and talks to the metric only through its moments $\delta\rho$, $\delta p$, $(\rho+p)\theta$ and $(\rho+p)\sigma$; the gravity object turns the summed moments into $\Phi$, $\Psi$ and $\Phi'$. In general relativity (with $'$ = $d/dx$)
$$\Phi' = \Psi - \frac{c^2k^2}{3\mathcal{H}^2}\Phi + \frac{H_0^2}{2\mathcal{H}^2}\left[\frac{\Omega_{\rm CDM}\delta_{\rm CDM} + \Omega_b\delta_b}{a} + \frac{4(\Omega_\gamma\Theta_0 + \Omega_\nu\mathcal{N}_0)}{a^2}\right],\qquad \Psi = -\Phi - \frac{12H_0^2}{c^2k^2a^2}\left[\Omega_\gamma\Theta_2 + \Omega_\nu\mathcal{N}_2\right].$$
Cold dark matter and baryons are fluids,
$$\delta_{\rm CDM}' = \frac{ck}{\mathcal{H}}v_{\rm CDM} - 3\Phi',\quad v_{\rm CDM}' = -v_{\rm CDM} - \frac{ck}{\mathcal{H}}\Psi,$$
$$\delta_b' = \frac{ck}{\mathcal{H}}v_b - 3\Phi',\quad v_b' = -v_b - \frac{ck}{\mathcal{H}}\Psi + \tau' R(3\Theta_1 + v_b),$$
and photons (temperature $\Theta_\ell$ and polarization $\Theta_{P\ell}$) and massless neutrinos ($\mathcal{N}_\ell$) are Boltzmann hierarchies truncated at $\ell_{\rm max}$,
$$\Theta_\ell' = \frac{\ell ck}{(2\ell+1)\mathcal{H}}\Theta_{\ell-1} - \frac{(\ell+1)ck}{(2\ell+1)\mathcal{H}}\Theta_{\ell+1} + \tau'\left[\Theta_\ell - \frac{1}{10}\Pi\,\delta_{\ell,2}\right],\qquad \Pi = \Theta_2 + \Theta_{P0} + \Theta_{P2}.$$
At early times, when $\tau' \gg ck/\mathcal{H}$, the photon&ndash;baryon fluid is tightly coupled and evolved with the tight-coupling approximation (the stiff equations are switched to the full hierarchy when the coupling weakens). Massive neutrinos use the momentum-resolved Ma&ndash;Bertschinger hierarchy and switch to the CLASS fluid approximation deep inside the horizon. The modes are solved in parallel over $k$ (OpenMP), and the source functions are stored on 2D splines in $(x,k)$.
</p>

<header class="entry-header text-center">
  <h1 class="entry-title" id="Spectra">Line-of-sight integration, $C_\ell$ and $P(k)$</h1>
</header>
<hr/>

<p>
Instead of evolving the photon hierarchy to $\ell \sim 2500$, the multipoles today are obtained with the line-of-sight integral (Seljak &amp; Zaldarriaga 1996)
$$\Theta_\ell(k,x=0) = \int_{-\infty}^0 \tilde S(k,x)\,j_\ell\big[k(\eta_0 - \eta(x))\big]\,dx,\qquad \tilde S = \tilde g\left[\Theta_0 + \Psi + \frac{1}{4}\Pi\right] + e^{-\tau}\left[\Psi' - \Phi'\right] - \frac{1}{ck}\frac{d}{dx}(\mathcal{H}\tilde g v_b) + \frac{3}{4c^2k^2}\frac{d}{dx}\left[\mathcal{H}\frac{d}{dx}(\mathcal{H}\tilde g\Pi)\right],$$
and similarly for the $E$-mode polarization. In curved universes the spherical Bessel functions are replaced by hyperspherical Bessel functions $\Phi^\nu_\ell(\chi)$ (with CLASS's flat approximation where it is accurate). The angular power spectra are
$$C_\ell^{XY} = 4\pi\int \frac{dk}{k}\,\mathcal{P}_{\mathcal R}(k)\,\Theta^X_\ell(k)\Theta^Y_\ell(k),\qquad \mathcal{P}_{\mathcal R}(k) = A_s\left(\frac{k}{k_{\rm pivot}}\right)^{n_s - 1 + \frac{1}{2}\alpha_s\ln(k/k_{\rm pivot})},$$
for $XY = TT, EE, TE$, and the matter power spectrum (total matter and CDM+baryons) is
$$P(k,x) = \frac{2\pi^2}{k^3}\mathcal{P}_{\mathcal R}(k)\,|\Delta_M(k,x)|^2,\qquad \Delta_M = \frac{c^2k^2\Phi}{\frac{3}{2}\Omega_{M}a^{-1}H_0^2}.$$
</p>
<p>
These expressions sit on the background and Einstein/Boltzmann evolution. For a flat $\Lambda$CDM example, the expansion is $E^2(a)=\Omega_r a^{-4}+\Omega_m a^{-3}+\Omega_\Lambda$; curvature adds $\Omega_Ka^{-2}$ and massive neutrinos use the time-dependent Fermi&ndash;Dirac density integral. For a smooth CPL component, $w(a)=w_0+w_a(1-a)$ and $\rho_{\rm DE}(a)=\rho_{{\rm DE},0}a^{-3(1+w_0+w_a)}e^{-3w_a(1-a)}$. Species exchange density, pressure, velocity and anisotropic-stress moments with the perturbed metric; modified-gravity field equations supply $\Phi$, $\Psi$ and their time derivatives. The detailed model equations and supported domains are summarized in the catalogue below.
</p>

<header class="entry-header text-center">
  <h1 class="entry-title" id="Lensing">Lensing, Halofit and two-point functions</h1>
</header>
<hr/>

<p>
The lensing potential spectrum $C_\ell^{\phi\phi}$ is computed from the Weyl potential by line of sight (with Limber at high $\ell$), and lensed $TT$, $EE$, $TE$ and $BB$ spectra with the full-sky correlation-function method of Challinor &amp; Lewis (2005). A selected nonlinear boost multiplies the Limber Weyl-potential power by $P_{\rm NL}/P_{\rm L}$. The <tt>nonlinear</tt> parameter selects:
</p>
<div class="table-responsive">
<table class="table table-striped table-condensed">
<thead><tr><th>Value</th><th>Model</th><th>Domain / behavior</th></tr></thead>
<tbody>
<tr><td><tt>none</tt></td><td>Linear theory</td><td>Default.</td></tr>
<tr><td><tt>halofit</tt></td><td><a href="https://arxiv.org/abs/1208.2701" style="display: inline">Takahashi et al. (2012) Halofit</a></td><td>No massive-neutrino correction; a warning is issued for massive neutrinos.</td></tr>
<tr><td><tt>halofit_bird</tt></td><td>Halofit with the updated <a href="https://arxiv.org/abs/1109.4416" style="display: inline">Bird et al. (2012)</a> neutrino correction</td><td>Takahashi-compatible correction in the CLASS-style form; reduces to Halofit when $f_\nu=0$.</td></tr>
<tr><td><tt>hmcode2016</tt></td><td><a href="https://arxiv.org/abs/1505.07833" style="display: inline">HMcode-2015</a> / CAMB <tt>mead2016</tt> update (<a href="https://arxiv.org/abs/1602.02154" style="display: inline">Mead et al. 2016</a>)</td><td>Includes massive neutrinos; shares the general halo-model building blocks.</td></tr>
<tr><td><tt>hmcode2020</tt></td><td><a href="https://arxiv.org/abs/2009.01858" style="display: inline">Gravity-only HMcode-2020 (Mead et al. 2021)</a></td><td>Cold variances, neutrino correction, BAO damping and smoothed halo transition. Solver supports LCDM and smooth CPL backgrounds, with curvature.</td></tr>
<tr><td><tt>react</tt></td><td>Native <a href="https://arxiv.org/abs/1812.05594" style="display: inline">ReACT</a> reaction on an HMcode-2020 matched pseudo spectrum</td><td>Native implementation; flat LCDM, $f(R)$ with $n=1$, smooth CPL, and normal-branch DGP with LCDM background. Reactions are restricted to $z\leq2.5$.</td></tr>
<tr><td><tt>axionhmcode</tt></td><td>Native mixed cold/axion halo model</td><td><tt>model = axion</tt>; axion/cold spectra enter separately. The Dome calibration is limited in mass, fraction and redshift; extrapolation must be assessed explicitly.</td></tr>
<tr><td><tt>ee2</tt></td><td>EuclidEmulator2 nonlinear boost</td><td>Flat LCDM or w0wa within the emulator domain. Tabulated over $0\le z\le10$; linear above $z=10$.</td></tr>
<tr><td><tt>jbd_boost</tt></td><td>Brans–Dicke boost from a matched GR nonlinear spectrum</td><td><tt>model = jbd</tt>; GR baseline selected by <tt>jbd_boost_gr</tt>, EuclidEmulator2 by default.</td></tr>
</tbody>
</table>
</div>
<p>The reaction method is $P_{\rm NL,target}=R\,P_{\rm NL,pseudo}$. Above the configured ReACT redshift limit, the solver uses its matched HMcode baseline with $R=1$ as an explicit approximation. All nonlinear choices write total-matter $P_{\rm NL}$ as the fourth power-spectrum column; the linear CDM+baryon column remains linear. HMcode-2020 here is gravity-only and does not add baryonic feedback.</p>
<p>
Two-point functions: the CMB correlation function $C(\theta) = \sum_\ell\frac{2\ell+1}{4\pi}C_\ell P_\ell(\cos\theta)$ (and the polarization $\xi_\pm$ with beams), the matter correlation function
$$\xi(r) = \int\frac{k^2dk}{2\pi^2}P(k)\frac{\sin kr}{kr}$$
(with a Filon transform), $\sigma_R$ and $\sigma_8$, and the angular $C_\ell$ and $w(\theta)$ of a projected tracer with a redshift window (exact and Limber).
</p>

<header class="entry-header text-center">
  <h1 class="entry-title" id="Tensors">Tensor and isocurvature modes</h1>
</header>
<hr/>

<p>
<b>Tensor modes.</b> With $r&gt;0$ the gravitational-wave equation
$$h'' + \left(2 + \alpha_M + \frac{\mathcal{H}'}{\mathcal{H}}\right)h' + \frac{c^2k^2}{\mathcal{H}^2}h = \frac{16\pi G a^2}{M_*^2\mathcal{H}^2}\,p\,\pi_T$$
is solved together with the tensor photon and neutrino hierarchies (the anisotropic stress $\pi_T$ damps the modes that enter in the radiation era: the amplitude ratio $0.8026$ of Weinberg 2004 is reproduced). The friction $\alpha_M$ and the effective Planck mass $M_*^2$ come from the theory of gravity (Horndeski, Brans&ndash;Dicke, $f(R)$). The tensor $TT$, $EE$, $TE$ and $BB$ are added to the scalar spectra and lensed with them; $n_t$ follows the consistency relation unless set.
</p>
<p>
<b>Isocurvature modes.</b> CDM, baryon, neutrino density and neutrino velocity isocurvature with power $P_{II} = f^2A_s(k/k_{\rm pivot})^{n_I-1}$, each correlated with the adiabatic mode with $P_{\mathcal{R}I} = c\sqrt{P_{\mathcal{RR}}P_{II}}$ (CLASS parameters). The correlation matrix is Cholesky-factored into independent components, each a perturbation solve with mixed initial conditions, and the spectra are summed.
</p>

<header class="entry-header text-center">
  <h1 class="entry-title" id="Models">Models beyond $\Lambda$CDM</h1>
</header>
<hr/>

<p>
A model combines a background/species prescription with linear perturbation equations and a theory of gravity, selected by name (<tt>model = ...</tt>). Background and perturbation models have separate registries under <tt>Cosmology/Background</tt> and <tt>Cosmology/Perturbations</tt>. The full parameter list for a selected model is generated by <tt>./cmb --write-defaults</tt>; parameters belonging to other models are rejected.
</p>
<div class="table-responsive"><table class="table table-striped table-condensed">
<thead><tr><th>Run name</th><th>Representative parameters</th><th>Background and perturbation physics</th></tr></thead>
<tbody>
<tr><td><tt>lcdm</tt></td><td>curvature, neutrino masses</td><td>GR with photons, baryons, CDM, massless/massive neutrinos, curvature and $\Lambda$.</td></tr>
<tr><td><tt>w0wa</tt></td><td><tt>w0, wa</tt></td><td>Smooth CPL expansion, $w(a)=w_0+w_a(1-a)$; background-only fluid (no DE perturbations).</td></tr>
<tr><td><tt>musigma</tt></td><td><tt>mu0, Sigma0</tt></td><td>Phenomenological growth and lensing: $\mu(a)=1+\mu_0\Omega_{\rm DE}(a)/\Omega_{{\rm DE},0}$ and an independent $\Sigma(a)$.</td></tr>
<tr><td><tt>dgp</tt></td><td><tt>Omega_rc, dgp_branch</tt></td><td>DGP background and quasi-static perturbations, $\mu=1+1/(3\beta)$, $\Sigma=1$; normal or self-accelerating branch.</td></tr>
<tr><td><tt>quintessence</tt>, <tt>quintessence_inverse_power_law</tt>, <tt>quintessence_pngb</tt>, <tt>quintessence_hilltop</tt></td><td><tt>quint_*</tt></td><td>Canonical scalar with exponential, inverse-power-law, pseudo-Nambu&ndash;Goldstone or hilltop potential; background and scalar perturbations.</td></tr>
<tr><td><tt>jbd</tt></td><td><tt>bd_omega</tt>, <tt>bd_GeffG_today</tt></td><td>Jordan&ndash;Brans&ndash;Dicke scalar-tensor gravity with $M_*^2\propto\phi$ and evolved scalar perturbations.</td></tr>
<tr><td><tt>fr</tt></td><td><tt>fR0, fR_n</tt></td><td>Hu&ndash;Sawicki $f(R)$: $\mu=\frac{1+4Q}{(1+3Q)(1+f_R)}$, $\Sigma=1/(1+f_R)$, $Q=(k/a)^2f_{RR}/(1+f_R)$.</td></tr>
<tr><td><tt>coupled_quintessence</tt></td><td><tt>cq_lambda, cq_beta</tt></td><td>Exponential quintessence coupled to CDM, $m_c\propto e^{-\beta\phi}$; energy exchange, drag and fifth-force factor $1+2\beta^2$.</td></tr>
<tr><td><tt>fading_dark_sector</tt></td><td><tt>fds_c, fds_cprime</tt></td><td>The Fading Dark Sector of <a href="https://arxiv.org/abs/2611.17422" style="display: inline">Obied, Vafa &amp; Wu (2026)</a>, $V_\mathrm{DE}\propto e^{-c\phi}$, $m_\mathrm{DM}\propto e^{-c'\phi}$: <tt>coupled_quintessence</tt> with $\lambda=c$, $\beta=c'$. Defaults are the paper's best fit, $c=-0.85$, $c'=0.05$ (<tt>params/fading_dark_sector.ini</tt>).</td></tr>
<tr><td><tt>dark_scattering</tt></td><td><tt>ds_w, ds_xi, ds_cs2</tt></td><td>Perturbed dark-energy fluid with sound speed and elastic DM&ndash;DE momentum exchange; $\xi=0$ gives wCDM perturbations.</td></tr>
<tr><td><tt>covariant_dark_scattering</tt></td><td><tt>ds_w, cds_alpha, ds_cs2</tt></td><td>Covariant CDM–DE elastic momentum exchange; wCDM background with drag proportional to $\alpha a^3/(\Omega_{c0}E)$ per $\ln a$.</td></tr>
<tr><td><tt>axion</tt></td><td><tt>ax_log10_m, ax_fraction, ax_switch_mH</tt></td><td>Mixed cold/ultralight-axion dark matter: frozen scalar, oscillatory evolution and averaged fluid; separate cold and axion power, with axion isocurvature options.</td></tr>
<tr><td><tt>cubic_galileon</tt></td><td>&ndash;</td><td>Tracker background and full braided scalar perturbations; tracker obeys $E^4-\rho E^2=\Omega_\phi$.</td></tr>
<tr><td><tt>horndeski</tt></td><td><tt>hd_parametrization, hd_cK, hd_cB, hd_cM, ...</tt></td><td>Horndeski $\alpha$-basis with $\alpha_T=0$, scalar perturbations, ghost/gradient/stiffness checks and $M_*^2$ evolution.</td></tr>
<tr><td><tt>ede</tt>, <tt>ede_zc</tt></td><td><tt>ede_fEDE, ede_log10zc, ede_theta_i, ede_n</tt></td><td>Axion-like early dark energy, $V=m^2f^2[1-\cos(\phi/f)]^n+\Lambda$; field background and perturbations through oscillations, then cycle-averaged fluid.</td></tr>
<tr><td><tt>ide</tt></td><td><tt>ide_xi, ide_w, ide_cs2, ide_stability_check</tt></td><td>Interacting dark energy with $Q=\xi H\rho_{\rm DE}$ along the CDM velocity and stability checks.</td></tr>
<tr><td><tt>horndeski_table</tt></td><td><tt>model_file, hd_stability_check</tt></td><td>Reduced Horndeski ($K, G_3, G_4$) from its Lagrangian: the background and the $\alpha$-basis perturbations from the table made by SimulationCode's <tt>horndeski_frontend.py</tt> (e.g. the cubic Galileon, Brans&ndash;Dicke or K-mouflage), with stability checks.</td></tr>
<tr><td><tt>symmetron</tt>, <tt>dilaton</tt>, <tt>mbeta</tt></td><td><tt>sym_assb, sym_beta, sym_L_mpch, sym_N, sym_M</tt>; <tt>dil_A2, dil_beta0, dil_r, dil_xi</tt>; <tt>mb_beta0, mb_n_beta, mb_m0, mb_n_m</tt></td><td>$\Lambda$CDM background with the quasi-static $\mu(k,a)$ of the symmetron, the dilaton and the general $(m(a), \beta(a))$ parametrization: the linear limit of SimulationCode's gravity models of the same names.</td></tr>
</tbody>
</table></div>
<p>
For the $\mu$&ndash;$\Sigma$ family the scalar Poisson and lensing equations are modified as
$$-k^2\Psi = 4\pi G a^2\mu(k,a)\bar\rho\Delta,\qquad -k^2(\Psi - \Phi) = 8\pi G a^2\Sigma(k,a)\bar\rho\Delta.$$
Massive neutrinos use momentum-resolved hierarchies (with a fluid approximation at high momentum); photons/baryons include tight coupling and polarization. Models are checked in their GR/$\Lambda$CDM limits and against unused Einstein constraints, independent integrations and, where available, CLASS, hi_class or CLASS_EDE. The model README contains the full validation map.
</p>

<header class="entry-header text-center">
  <h1 class="entry-title" id="Axion">Mixed ultralight-axion dark matter</h1>
</header>
<hr/>

<figure class="cinematic-figure" id="cinematic-09-ultralight-dark-matter">
  <a href="images/cinematic/09-ultralight-dark-matter.jpg" aria-label="View full-size illustration: Ultralight dark matter">
    <img src="images/cinematic/09-ultralight-dark-matter.jpg" width="1774" height="887" loading="lazy" decoding="async" alt="Illustrative wave-dark-matter density fringes surrounding a bright extended central core">
  </a>
  <figcaption><strong>Ultralight dark matter.</strong> Conceptual wave-dark-matter interference and a central coherent core. FML SimulationCode does not evolve these nonlinear wave dynamics. This is not the nonlinear evolution computed by the Einstein&ndash;Boltzmann solver. <span class="cinematic-credit">AI-generated concept illustration.</span></figcaption>
</figure>

<p>For <tt>model = axion</tt>, <tt>ax_fraction</tt> is the axion share of the dark-matter density parameter and <tt>ax_log10_m</tt> sets the mass in eV. The quadratic scalar starts frozen and its initial amplitude is solved to obtain the requested density today. Once oscillations are sufficiently rapid, a cycle-averaged fluid replaces the scalar, with scale-dependent sound speed $c_s^2=k^2/(k^2+4m^2a^2)$. <tt>ax_switch_mH</tt> controls the switch, including an axionCAMB comparison choice. The solver writes cold and axion auto-spectra; axion isocurvature may be specified directly or from the inflationary Hubble scale.</p><div style="text-align: center"><div style="max-width:800px;display:inline-block;text-align:left;width:100%"><pre class="prettyprint"># Run from FML/Cosmology/EinsteinBoltzmannSolver
./cmb params/axion.ini
# Optional nonlinear mixed model:
./cmb params/axion.ini nonlinear=axionhmcode</pre></div></div>
<p>AxionHMcode uses the separate linear inputs; it is a nonlinear prescription with a restricted calibration, rather than a wave simulation. See <a href="https://github.com/HAWinther/FML/blob/master/FML/Cosmology/EinsteinBoltzmannSolver/params/axion.ini" style="display: inline">the parameter example</a> and <a href="https://github.com/HAWinther/FML/blob/master/FML/AxionHMcode/README.md" style="display: inline">calibration and comparison limits</a>.</p>

<header class="entry-header text-center">
  <h1 class="entry-title" id="Usage">Running the solver</h1>
</header>
<hr/>

<div style="text-align: center">
  <div style="width:700px;display: inline-block;text-align: left">
    <pre class="prettyprint">
cd FML/Cosmology/EinsteinBoltzmannSolver
make                                    # builds ./cmb (needs GSL)
./cmb                                   # fiducial model, outputs in data/
./cmb params/planck2018.ini             # parameters from a file
./cmb m_nu=0.06 lensing=true output_dir=data/test   # command-line overrides
./cmb model=fr fR0=1e-5 pert_k_max=1    # a model beyond LCDM
./cmb --write-defaults my.ini           # every parameter with its default and description
make test                               # ~650 unit checks and regression tests</pre>
  </div>
</div>
<p>
Outputs: <tt>background.txt</tt>, <tt>recombination.txt</tt>, <tt>perturbations_k*.txt</tt>, <tt>cells.txt</tt> ($D_\ell^{TT,EE,TE}$ in $\mu$K$^2$), <tt>cells_lensed.txt</tt>, <tt>matter_power.txt</tt> ($k$ in $h$/Mpc, $P_m$, $P_{cb}$), <tt>transfer.txt</tt> and the two-point functions. The fiducial run takes ~45 s on 2 cores; <tt>precision = fast</tt> (tuned for CMB likelihoods: $\Delta\chi^2\approx 0.2$ for Planck-like noise) computes lensed spectra to $\ell = 2500$ in ~15 s. From C++:
</p>
<div style="text-align: center">
  <div style="width:700px;display: inline-block;text-align: left">
    <pre class="prettyprint">
#include &lt;FML/Cosmology/Background/BackgroundCosmology.h&gt;
#include &lt;FML/Cosmology/Recombination/RecombinationHistory.h&gt;
#include &lt;FML/Cosmology/Perturbations/Perturbations.h&gt;
#include &lt;FML/Cosmology/Spectra/PowerSpectrum.h&gt;
using namespace FML::COSMOLOGY;

CosmologyParameters p = CosmologyParameters::Planck2018();
BackgroundCosmology cosmo(p);            // or (p, model) for a model beyond LCDM
cosmo.solve();
RecombinationHistory rec(cosmo, rec_options);
rec.solve();
Perturbations pert(cosmo, rec, pert_options);
pert.solve();
PowerSpectrum power(cosmo, rec, pert, spectrum_options);
power.solve();
double H0  = cosmo.H_of_x(0.0);                          // [1/s]
double DTT = power.get_Dell_TT_lensed(200);              // [muK^2]
double Pk  = power.get_matter_power_spectrum(0.0, k);    // x = ln a, k in 1/m</pre>
  </div>
</div>

<header class="entry-header text-center">
  <h1 class="entry-title" id="Validation">Validation against CLASS</h1>
</header>
<hr/>

<table class="table table-striped table-condensed">
<thead><tr><th>Quantity</th><th>Agreement with CLASS (same physics, RECFAST)</th></tr></thead>
<tbody>
<tr><td>Background $H(z)$, distances</td><td>$10^{-7}$</td></tr>
<tr><td>$X_e(z)$</td><td>$3\cdot 10^{-4}$</td></tr>
<tr><td>$z_*$, $r_s$</td><td>$10^{-6}$</td></tr>
<tr><td>$C_\ell^{TT}$</td><td>$5\cdot 10^{-4}$ rms for $\ell = 30$&ndash;$1000$, $3\cdot 10^{-3}$ for $1000$&ndash;$2000$</td></tr>
<tr><td>$C_\ell^{EE}$</td><td>$3\cdot 10^{-3}$ rms</td></tr>
<tr><td>$P(k)$, $\sigma_8$</td><td>$0.1\%$, $4\cdot 10^{-5}$</td></tr>
<tr><td>Lensed/unlensed $TT$, effect of $m_\nu$</td><td>$10^{-3}$, $5\cdot 10^{-4}$</td></tr>
<tr><td>Halofit $P_{\rm NL}$ ($k\le 1\,h$/Mpc)</td><td>$&lt;10^{-3}$</td></tr>
<tr><td>Tensor $BB$ (lensed total)</td><td>$0.1\%$ for $\ell = 50$&ndash;$300$</td></tr>
</tbody>
</table>
<p>
The physics, numerics and validation of every module are described in <tt>Cosmology/EinsteinBoltzmannSolver/report/paper.pdf</tt>.
</p>

<header class="entry-header text-center">
  <h1 class="entry-title" id="References">Method references and scope</h1>
</header>
<hr/>

<p>
The Brans&ndash;Dicke nonlinear option <tt>jbd_boost</tt> follows <a href="https://arxiv.org/abs/2403.03786" style="display: inline">Sletmoen &amp; Winther (2024)</a>, mapping to a matched GR nonlinear spectrum. Mixed ultralight-axion evolution uses <a href="https://github.com/dgrin1/axionCAMB" style="display: inline">axionCAMB</a> as a model/comparison reference, with AxionHMcode as its separate nonlinear prescription. The <a href="https://github.com/HAWinther/FML/blob/master/FML/Cosmology/EinsteinBoltzmannSolver/README.md" style="display: inline">solver overview</a> records option-specific domains and validation. Dark-sector model references are also listed on the <a href="nbodycode.php#References" style="display: inline">SimulationCode page</a>.
</p>
