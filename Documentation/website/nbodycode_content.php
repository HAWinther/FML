<hr/>
<header class="entry-header text-center">
  <h1 class="entry-title">Table of contents</h1>
</header>
<hr/>

<p class="page-toc">
<a href="index.php">Back to the main page</a><br>
<a href="documentation.php">Back to the documentation</a><br>
<a href="#Overview">1. Overview: N-body solver</a><br>
<a href="#Running">2. Running the code and available models</a><br>
<a href="#ModelCatalog">3. Complete cosmology and gravity model catalogue</a><br>
<a href="#TimeStepping">4. Time-stepping</a><br>
<a href="#LPTgrowth">5. LPT growth factors</a><br>
<a href="#LPTdisp">6. LPT displacement fields</a><br>
<a href="#GravityModelGR">7. GravityModel : GR</a><br>
<a href="#GravityModelSymmetron">8. GravityModel : Symmetron</a><br>
<a href="#GravityModelJBD">9. GravityModel : Jordan-Brans-Dicke (JBD)</a><br>
<a href="#GravityModelDGP">10. GravityModel : DGP</a><br>
<a href="#GravityModelFOFR">11. GravityModel : $f(R)$</a><br>
<a href="#MassiveNu">12. Massive neutrinos</a><br>
<a href="#COLA">13. COLA</a><br>
<a href="#Cosmology">14. CosmologyModel</a><br>
<a href="#CosmologyLCDM">15. CosmologyModel : $\Lambda$CDM</a><br>
<a href="#CosmologyJBD">16. CosmologyModel : JBD</a><br>
<a href="#MuGravity">17. GravityModel : $\mu(a,k)$ and $G_{\rm eff}$</a><br>
<a href="#Species">18. Separate CDM and baryons, neutrino particles</a><br>
<a href="#EarlyForce">19. Forces, the lattice offset and starting from the Big Bang</a><br>
<a href="#NbodyGauge">20. The N-body gauge</a><br>
<a href="#ICs">21. Initial conditions: Philox fields and internal transfer functions</a><br>
<a href="#Analysis">22. On-the-fly analysis: SO masses, hybrid bias spectra, baryonification, lightcone, images</a><br>
<a href="#References">Method references and scope</a><br>
<a href="#Interacting">Interacting dark sectors and species evolution</a><br>
<a href="#Horndeski">Reduced Horndeski from a Lagrangian</a><br>
<a href="#AxionIC">Ultralight axions in initial conditions</a><br>
<a href="#SnapshotAnalysis">Post-process snapshots with shared analysis</a><br>
</p>

<header class="entry-header text-center">
  <h1 class="entry-title" id="Overview">Overview: N-body solver</h1>
</header>
<hr/>

<figure class="cinematic-figure" id="cinematic-03-gravity-assembling-halo">
  <a href="images/cinematic/03-gravity-assembling-halo.jpg" aria-label="View full-size illustration: Gravity assembling a halo">
    <img src="images/cinematic/03-gravity-assembling-halo.jpg" width="1774" height="887" loading="lazy" decoding="async" alt="Matter streams feed an extended halo with selected curved infall and orbital trajectories">
  </a>
  <figcaption><strong>Gravity assembling a halo.</strong> Conceptual dark-matter density and selected particle trajectories illustrating anisotropic infall and orbital mixing in a halo. <span class="cinematic-credit">AI-generated concept illustration.</span></figcaption>
</figure>  

<p>
<a href="https://github.com/HAWinther/FML/tree/master/FML/SimulationCode" style="display: inline">SimulationCode</a> is FML's standalone particle-mesh (PM) and COLA cosmological simulation. It advances particles with MPI (slab or pencil decomposition) and OpenMP, computes forces on FFT grids, and separates the homogeneous expansion (<tt>Cosmology</tt>) from the perturbation growth and force law (<tt>GravityModel</tt>). That split lets a background be paired with GR or a supported modified-gravity model. The slab layout favors fast approximate and medium-scale runs, parameter studies and analysis over very high-resolution production simulations.
</p>
<figure class="science-figure">
  <img src="images/simulation-workflow.svg" alt="SimulationCode loop from initial conditions and LPT through mesh density, gravity, force interpolation, particle integration, and analysis outputs">
  <figcaption>One simulation step: cosmology and gravity set the force evolution; the selected integrator advances particles, with COLA evolving motion relative to its LPT reference.</figcaption>
</figure>

<header class="entry-header text-center">
  <h1 class="entry-title" id="Running">Running the code and available models</h1>
</header>
<hr/>

<p>
<div style="text-align: center">
  <div style="width:600px;display: inline-block;text-align: left">
    <pre class="prettyprint">
make                                   # settings in ../Makefile.config (or ../Makefile.local)
mpirun -np 4 ./nbody parameterfile.lua
./nbody --parameters PARAMETERS.md     # write the reference of all parameters
</pre>
</div>
</div>
</p>
<p>
All parameters, defaults, allowed values and model-specific fields are listed in the generated <a href="https://github.com/HAWinther/FML/blob/master/FML/SimulationCode/PARAMETERS.md" style="display: inline">parameter reference</a>; <tt>parameterfile.lua</tt> is a commented example. Unknown names produce a warning with a suggestion. See the <a href="https://github.com/HAWinther/FML/blob/master/FML/SimulationCode/README.md" style="display: inline">SimulationCode README</a> for build, input and validation details.
</p>
<p>
SimulationCode supports generated or supplied initial conditions; scale-independent or scale-dependent 1&ndash;3LPT COLA frames; Quinn, Tassev, FastPM and BullFrog stepping; GR and screened/unscreened modified-gravity forces; linear-grid or particle $\delta f$ neutrinos; separate CDM and baryon particles; snapshot and lightcone output; and on-the-fly power spectra, multipoles, bispectra, halo catalogues and hybrid bias spectra. It can also read transfer functions from the internal Einstein&ndash;Boltzmann solver. The independent arbitrary-order EdS LPT module is a separate library component and is not yet the COLA frame used here.
</p>

<p>
<b>Pencils.</b> Compile with <tt>make USE_PENCIL_GRID=true</tt> to run on a <a href="pencilgrid.php" style="display: inline">pencil decomposition</a> instead of slabs (useful when the number of tasks approaches the mesh size, or the mesh is not divisible by it). The pencils are <tt>FML_PENCILS="px py"</tt> from the environment, or as square as possible; the meshes must be divisible by <tt>px</tt> and <tt>py</tt>. Every model, COLA, lightcones, neutrinos, the ICs and every output and analysis work on both; the objects are rebuilt when the flag changes.
</p>

<header class="entry-header text-center">
  <h1 class="entry-title" id="ModelCatalog">Complete cosmology and gravity model catalogue</h1>
</header>
<hr/>

<p>The model names below are the values of <tt>cosmology_model</tt> and <tt>gravity_model</tt>. The background model describes $H(a)$ and species; the gravity model supplies LPT growth and the force response. The same cosmology can therefore be paired with different gravity models where physically supported.</p>
<div class="table-responsive">
<table class="table table-striped table-condensed">
<thead><tr><th><tt>cosmology_model</tt></th><th>Background equation / capability</th><th>Important scope</th></tr></thead>
<tbody>
<tr><td><tt>LCDM</tt></td><td>$E^2(a)=\Omega_r a^{-4}+\rho_\nu(a)/\rho_{\rm crit,0}+\Omega_m a^{-3}+\Omega_Ka^{-2}+\Omega_\Lambda$.</td><td>Curvature and massive neutrinos are available.</td></tr>
<tr><td><tt>w0waCDM</tt></td><td>CPL dark energy: $w(a)=w_0+w_a(1-a)$.</td><td>Smooth background component.</td></tr>
<tr><td><tt>DGP</tt></td><td>Self-accelerating DGP expansion selected by $\Omega_{rc}$.</td><td>Can be paired with the DGP force model; the separate gravity model can also use a LCDM background.</td></tr>
<tr><td><tt>JBD</tt></td><td>Jordan&ndash;Brans&ndash;Dicke background and evolving scalar $\phi(a)$.</td><td>Use the matching JBD force model.</td></tr>
<tr><td><tt>Quintessence</tt> variants</td><td>$V(\phi)=V_0u(\phi)$: exponential, inverse-power, pseudo-Nambu&ndash;Goldstone (axion-like thawing), or hilltop.</td><td>Names: <tt>Quintessence</tt>, <tt>quintessence_inverse_power_law</tt>, <tt>quintessence_pngb</tt>, <tt>quintessence_hilltop</tt>. Smooth in SimulationCode; no dark-energy perturbations.</td></tr>
<tr><td><tt>EDE</tt></td><td>$V(\phi)=m^2f^2[1-\cos(\phi/f)]^n+\Lambda$; shooting sets the peak fraction and redshift.</td><td>Background only: start after the EDE episode using compatible Boltzmann initial conditions.</td></tr>
<tr><td><tt>DarkScattering</tt>, <tt>CovariantDarkScattering</tt></td><td>wCDM expansion with CDM–DE momentum exchange in the velocity equations.</td><td><tt>cosmology_ds_w</tt>, <tt>cosmology_ds_xi</tt> or <tt>cosmology_cds_alpha</tt>; smooth DE on N-body scales.</td></tr>
<tr><td><tt>CoupledQuintessence</tt></td><td>Exponential scalar potential and varying CDM mass $m_c\propto e^{-\beta\phi}$.</td><td>Requires matching <tt>gravity_model = CoupledQuintessence</tt>.</td></tr>
<tr><td><tt>FadingDarkSector</tt></td><td>The Fading Dark Sector (<a href="https://arxiv.org/abs/2611.17422" style="display: inline">Obied, Vafa &amp; Wu 2026</a>): <tt>CoupledQuintessence</tt> with $\lambda=c$, $\beta=c'$.</td><td><tt>cosmology_fds_c</tt>, <tt>cosmology_fds_cprime</tt> (best fit $-0.85$, $0.05$); use with <tt>gravity_model = FadingDarkSector</tt>.</td></tr>
<tr><td><tt>IDE</tt></td><td>$Q=\xi H\rho_\mathrm{DE}$ with local-expansion-dependent particle mass updates.</td><td>The dark energy is smooth on N-body scales; void mass limits and resolution matter.</td></tr>
<tr><td><tt>CubicGalileon</tt></td><td>Tracker scalar-field expansion.</td><td>Use the matching CubicGalileon force model.</td></tr>
<tr><td><tt>Tabulated</tt></td><td>Read $H(z)$, a CLASS background table, or $w(a)$ from a file.</td><td>Matter and radiation parameters must match the supplied background.</td></tr>
</tbody>
</table>
</div>
<div class="table-responsive">
<table class="table table-striped table-condensed">
<thead><tr><th><tt>gravity_model</tt></th><th>Linear force law / model parameters</th><th>Nonlinear treatment</th></tr></thead>
<tbody>
<tr><td><tt>GR</tt></td><td>$\tilde\nabla^2\tilde\Phi=\frac32\Omega_m a\delta_m$.</td><td>Standard PM force.</td></tr>
<tr><td><tt>f(R)</tt></td><td>Hu&ndash;Sawicki scalaron; $G_{\rm eff}/G=1+\frac13\frac{k^2}{k^2+a^2m^2}$.</td><td>Linear, approximate screening, large-scale-linear matching, or exact quasi-static multigrid field solve.</td></tr>
<tr><td><tt>DGP</tt></td><td>$G_{\rm eff}/G=1+1/(3\beta)$.</td><td>Linear, approximate Vainshtein screening, large-scale matching, or exact nonlinear field solve.</td></tr>
<tr><td><tt>JBD</tt></td><td>$G_{\rm eff}(a)/G=1/\phi(a)$.</td><td>Linear scalar-tensor force with time-varying coupling.</td></tr>
<tr><td><tt>Symmetron</tt></td><td>Time-dependent symmetry breaking, coupling $\beta(a)$ and scalar mass $m(a)$.</td><td>Linear, approximate screening or exact quasi-static multigrid solve.</td></tr>
<tr><td><tt>Dilaton</tt></td><td>Chameleon-like $m(a)=m_0a^{-r}$ and evolving coupling $\beta(a)$.</td><td>Linear, approximate screening or exact quasi-static multigrid solve.</td></tr>
<tr><td><tt>mbeta</tt></td><td>General parameterization $\beta(a)=\beta_0a^n$, $m(a)=m_0H_0a^m$.</td><td>Linear or approximate chameleon screening.</td></tr>
<tr><td><tt>CoupledQuintessence</tt></td><td>CDM fifth force $G(1+2\beta^2)$ and velocity term $+\beta\phi'v$.</td><td>Unscreened; baryons feel gravity only. Supports one effective fluid or separate CDM/baryon particles.</td></tr>
<tr><td><tt>FadingDarkSector</tt></td><td>The same as <tt>CoupledQuintessence</tt> with $\beta=c'$: fifth force $G(1+2c'^2)$, velocity term $+c'\phi'v$.</td><td>Needs <tt>cosmology_model = FadingDarkSector</tt>; outputs identical to <tt>CoupledQuintessence</tt> with the same parameters.</td></tr>
<tr><td><tt>CubicGalileon</tt></td><td>$G_\mathrm{eff}/G=1+1/(3\beta(a))$ with DGP-form scalar equation.</td><td>Linear, density-smoothed screening approximation or quasi-static multigrid; deep-void discriminant limits apply.</td></tr>
<tr><td><tt>Horndeski</tt></td><td>Reduced $K(\phi,X),G_3(\phi,X),G_4(\phi)$ theory from background/force tables.</td><td>Linear, approximate screening or DGP-form multigrid where supported; use the matching tabulated background.</td></tr>
<tr><td><tt>mu</tt></td><td>Unscreened $G_{\rm eff}/G=\mu(a,k)$: DE, Bertschinger&ndash;Zukin, or a table.</td><td>Scale-dependent growth is solved; use scale-dependent COLA when appropriate.</td></tr>
</tbody>
</table>
</div>
<p>
For $f(R)$, DGP, Symmetron, Dilaton and <tt>mbeta</tt>, screening can be linear-only, approximate, approximate plus an enforced large-scale linear limit, or an exact nonlinear multigrid solve where that model provides one. Exact field solves are expensive and mainly useful for tests and comparisons. A complete generated parameter reference is more reliable than copying parameter defaults here because fields depend on the selected model.
</p>

<header class="entry-header text-center">
  <h1 class="entry-title" id="Interacting">Interacting dark sectors and species evolution</h1>
</header>
<hr/>

<p>
Dark scattering changes CDM velocities while leaving the wCDM background unchanged. With <tt>particle_baryons = true</tt>, CDM receives the full drag and baryons none, with species-specific kick, drift and growth factors. With one CDM+baryon particle fluid, the drag is weighted by the CDM fraction. Coupled quintessence also changes CDM mass and adds a CDM-sourced fifth force; the matching cosmology and gravity models are required. The species paths support Quinn/FastPM and their COLA frames; do not assume BullFrog or massive-neutrino-particle compatibility.
</p>
<p>
IDE differs from these momentum-exchange models: particle masses respond to the local expansion as well as the background. Its volume estimate is smoothed on the particle grid, so the mass response and growth depend on resolution; limiting densities in deep voids are part of the implementation. Internal-solver initial conditions include the corresponding linear perturbations. Smooth dark energy at rest is an N-body-scale approximation and can differ from the full solver on the largest scales.
</p><p><a href="https://github.com/HAWinther/FML/blob/master/FML/SimulationCode/MODELS.md" style="display: inline">Model equations and restrictions</a> and <a href="https://github.com/HAWinther/FML/blob/master/FML/SimulationCode/SPECIES.md" style="display: inline">species validation</a> give the supported combinations and measured limits.</p>

<header class="entry-header text-center">
  <h1 class="entry-title" id="Horndeski">Reduced Horndeski from a Lagrangian</h1>
</header>
<hr/>

<p><tt>scripts/horndeski_frontend.py</tt> takes SymPy expressions for $K,G_3,G_4$ in a JSON model, derives and solves the background, and writes an expansion table and a quasi-static force table. Select <tt>cosmology_model = Tabulated</tt> with that expansion and <tt>gravity_model = Horndeski</tt> with <tt>gravity_model_horndeski_filename</tt>. The code checks that the tabulated expansion agrees with the simulation background.</p>
<p>The force combines a Newtonian rescaling $1/(2G_4)$ with a DGP-form fifth force. Screening may be linear, approximated using a smoothed density, or solved with the DGP multigrid equation when its coefficients permit. The cubic Galileon supplies a dedicated tracker model and density-smoothing controls. In sufficiently deep Galileon voids the quasi-static equation loses a real branch; the implementation uses a zero-discriminant limit and the multigrid solve can stall. K-mouflage ($K = -1 + X + K_0X^2$ with a conformal coupling, in the Jordan frame as in Hi-COLA, <a href="https://arxiv.org/abs/2407.00855" style="display: inline">Sen Gupta, Fiorini &amp; Baker</a>) is one such model: its Newtonian force is rescaled by $A^2$ and its fifth force is linear (the kinetic screening is not modelled, as in Hi-COLA). Examples and frontend checks against cubic Galileon, Brans–Dicke and K-mouflage are in <a href="https://github.com/HAWinther/FML/tree/master/FML/SimulationCode/input/horndeski" style="display: inline">the model examples</a> and <a href="https://github.com/HAWinther/FML/blob/master/FML/SimulationCode/MODELS.md" style="display: inline">the model guide</a>.</p>

<header class="entry-header text-center">
  <h1 class="entry-title" id="TimeStepping">Time-stepping</h1>
</header>
<hr/>  

<p>
We use a second order leapfrog method
$$\tilde{v}_{n+1/2} = \tilde{v}_{n-1/2} + \Delta\tilde{\tau}_{n-1/2}^{n+1/2} F_{n}$$
$$\tilde{x}_{n+1} = \tilde{x}_{n} + \Delta\tilde{\tau}_{n}^{n+1} \tilde{v}_{n+1/2}$$
where $\Delta\tau = \frac{da}{a^3 E(a)}$. The force is formally given by
$$F_n = \tilde{\nabla} \tilde{\Phi} = \frac{3}{2}\Omega_M a \tilde{\nabla} \tilde{\nabla}^{-2}\delta_M $$
In general models the prefactor is $P(a) = \frac{3}{2}\Omega_M a \frac{G_{\rm eff}(a)}{G}$.

We use the method of Quinn et al. and set
$$\Delta\tilde{\tau}_{n}^{n+1} = \int_{a_n}^{a_{n+1}} \frac{da}{a^3 E(a)}$$
and
$$\Delta\tilde{\tau}_{n-1/2}^{n+1/2} = \int_{a_{n-1/2}}^{a_{n+1/2}} \frac{P(a)}{P(a_n)} \frac{da}{a^3 E(a)}$$
since we have an extra explicit factor of $a$ coming from the prefactor to the Poisson equation.

We also have the option of using the method of Tassev et al. where the integrand for $\Delta\tilde{\tau}_{n}^{n+1}$ is multiplied by $(a / a_{n+1/2})^{\rm nLPT}$ and where $\Delta\tilde{\tau}_{n-1/2}^{n+1/2} = \frac{a_{n+1/2}^{\rm nLPT} - a_{n-1/2}^{\rm nLPT}}{{\rm nLPT}\cdot a_{n}^{\rm nLPT-1}} \frac{1}{a_n^3 E(a_n)}$ where ${\rm nLPT}$ is a free parameter.
</p>
<p>
Two integrators that follow the linear growing mode exactly whatever the step size are also available (<tt>timestep_method</tt>):
</p>
<ul>
<li><b>FastPM</b> (Feng et al. 2016): the kick and drift factors are computed from the linear growth factor, $\Delta v \propto \Delta(a^2E\,dD/da)/(\ldots)$, so that a linear field is advanced exactly.</li>
<li><b>BullFrog</b> (Rampf, List &amp; Hahn 2024): drift&ndash;kick&ndash;drift in growth-factor time with the kick chosen so that the second-order (2LPT) growing mode is followed after every step. It already does what COLA does, so it is not combined with COLA.</li>
</ul>
<p>
With 10 steps from $z = 19$, FastPM and BullFrog get $P(k,z=0)$ to about 0.5% for $k &lt; 0.1\,h$/Mpc and 1&ndash;2% up to $k = 0.5\,h$/Mpc, where Quinn steps are off by 7&ndash;12%. Steps can be placed linearly, logarithmically, as a power law in $a$ or uniformly in the growth factor $D(a)$ (which reduces the BullFrog error by 15&ndash;40%).
</p>

<header class="entry-header text-center">
  <h1 class="entry-title" id="LPTgrowth">LPT growth factors</h1>
</header>
<hr/>  

<p>
The general implementation of the LPT equations is as follows. Below $D$ means the CDM+baryon growth-factor and $D_{\nu}$ is the massive neutrino growth factor.
$$
\begin{align}
\frac{d^2D_1}{d\log a^2} + \left(2 + \frac{d\log E}{d\log a}\right)\frac{dD_1}{d\log a} &= \frac{3\Omega_M}{2a^3 E^2} \frac{G_{\rm eff}(a,k)}{G} g_{1\nu}h_{\rm 1LPT} D_1\\
\frac{d^2D_2}{d\log a^2} + \left(2 + \frac{d\log E}{d\log a}\right)\frac{dD_2}{d\log a} &= \frac{3\Omega_M}{2a^3 E^2} \frac{G_{\rm eff}(a,k)}{G} g_{2\nu}h_{\rm 2LPT} (D_2 - D_1^2)\\
\frac{d^2D_{3a}}{d\log a^2} + \left(2 + \frac{d\log E}{d\log a}\right)\frac{dD_{3a}}{d\log a} &= \frac{3\Omega_M}{2a^3 E^2} \frac{G_{\rm eff}(a,k)}{G} g_{3\nu a}h_{\rm 3LPTa} (D_{3a} - 2D_1^3)\\
\frac{d^2D_{3b}}{d\log a^2} + \left(2 + \frac{d\log E}{d\log a}\right)\frac{dD_{3b}}{d\log a} &= \frac{3\Omega_M}{2a^3 E^2} \frac{G_{\rm eff}(a,k)}{G} g_{3\nu b}h_{\rm 3LPTb} (D_{3b} + D_1^3 - D_1D_2)
\end{align}
$$
where the effects of neutrinos are incorporated as
$$g_{1 \nu } = \frac{\Omega_{cb}}{\Omega_M} + \frac{\Omega_{\nu}}{\Omega_M}\frac{D_{1\nu}}{D_1} = 1 - f_\nu + f_\nu\frac{D_{1\nu}}{D_1}$$
$$g_{2 \nu } = g_{3 \nu a} = g_{3 \nu b} = \frac{\Omega_{cb}}{\Omega_M} = 1 - f_\nu$$
with $\Omega_{cb} = \Omega_{\rm CDM} + \Omega_b$ and $f_\nu = \Omega_\nu/\Omega_M$.
i.e. their effect is fully incorporated into 1LPT. We can either compute $\frac{D_{1\nu}}{D_1}$ via transfer functions or solve for the neutrino growth factor by solving
$$\frac{d^2D_{1\nu}}{d\log a^2} + \left(2 + \frac{d\log E}{d\log a}\right)\frac{dD_{1\nu}}{d\log a} = \frac{3\Omega_M}{2a^3 E^2} \frac{G_{\rm eff}(a,k)}{G} g_{1\nu}h_{\rm 1LPT} D_{1} - \frac{3\Omega_M}{2a^3 E^2}\frac{G_{\rm eff}(a,k)}{G} \frac{k^2}{k_{\rm fs}^2} D_{1\nu}$$
where $k_{\rm fs}^2 = \frac{3\Omega_M H_0^2}{2ac_{s,\nu}^2}$ is the free-streaming scale and $c^2_{s,\nu} = \frac{25\zeta(5)}{3\zeta(3)}\frac{(k_bT_\nu(a))^2}{m_\nu^2}$ is the sound speed (in the non-relativistic limit).

The other free functions are model dependent. For $\Lambda$CDM we have $\frac{G_{\rm eff}(a,k)}{G} = h_{\rm 1LPT} = h_{\rm 2LPT} = h_{\rm 3LPTa} = h_{\rm 3LPTb} = 1$.

The initial conditions are taken to be (from analytical solutions of a GR Einstein-de Sitter Universe)
$$D_1^{\rm ini} = 1, \,\,D_2^{\rm ini} = -\frac{3}{7}, \,\,D_{3a}^{\rm ini} = -\frac{1}{3}, \,\,D_{3b}^{\rm ini} = \frac{5}{21}$$
and the growth-rates $f \equiv \frac{d\log D}{d\log a}$ are given by
$$f_1^{\rm ini} = 1, \,\,f_2^{\rm ini} = 2, \,\,f_{3a}^{\rm ini} = 3, \,\,f_{3b}^{\rm ini} = 3$$
For neutrinos we use the input from an Einstein-Boltzmann solver to compute $D_\nu^{\rm ini} = \frac{T_\nu^{\rm ini}}{T_{cb}^{\rm ini}}$ and the corresponding growth-rate (here $T$ are the transfer functions).
</p>

<header class="entry-header text-center">
  <h1 class="entry-title" id="LPTdisp">LPT displacement fields</h1>
</header>
<hr/>  
<p>

For computing the initial displacement fields we set $\Psi^{iLPT} = \nabla\phi^{iLPT}$. The LPT displacement fields are computed from
$$\nabla^2 \phi^{\rm 1LPT} = -\delta_M(q,a_{\rm ini})$$
$$\nabla^2\phi^{\rm 2LPT} = \frac{D_2^{\rm ini}}{(D_1^{\rm ini})^2}\sum_{i&gt;j} [\nabla_{ii}\phi^{\rm 1LPT}\nabla_{jj}\phi^{\rm 1LPT} - (\nabla_{ij}\phi^{\rm 1LPT})^2]$$
and for 3LPT 
$$\Psi^{\rm 3LPT} = \left(\nabla \phi^{\rm 3LPTa} + \nabla \phi^{\rm 3LPTb} + \nabla\times A\right)$$
where
$$\nabla^2 \phi^{\rm 3LPTa} = \frac{D_{3a}^{\rm ini}}{(D_1^{\rm ini})^3}\det[\phi^{\rm 1LPT}_{ij}]$$
$$\nabla^2 \phi^{\rm 3LPTb} = \frac{D_{3b}^{\rm ini}}{(D_1^{\rm ini}D_2^{\rm ini})}[\nabla^2\phi^{\rm 1LPT}\nabla^2\phi^{\rm 2LPT} - \sum_{i,j} \phi^{\rm 1LPT}_{ij} \phi^{\rm 2LPT}_{ij} ]$$
$$\nabla^2 A = \frac{D_A^{\rm ini}}{D_1^{\rm ini}D_2^{\rm ini}}\nabla\phi^{\rm 2LPT}\times \nabla\phi^{\rm 1LPT}$$
and the prefactor is just $\frac{1}{3}$. The main difference wrt 1LPT and 2LPT is that the displacement field is not a pure potential flow (it has non-zero curl - well it's still curl free in Eulerian coordinates, but the transformation to Lagrangian coordinates introduces a curl. This curl part does not contribute to the initial power-spectrum of $\Psi$). We don't use this curl term for COLA (and for making IC with 3LPT you have to set a flag to use this).
</p>

<header class="entry-header text-center">
  <h1 class="entry-title" id="GravityModelGR">GravityModel : GR</h1>
</header>
<hr/>  

<p>
Poisson's equation is given by
$$\tilde{\nabla}^2\tilde{\Phi} = \frac{3}{2}\Omega_M a \delta_M$$
The force is computed from this as $\tilde{\nabla}\tilde{\Phi}$, i.e.
$$\frac{d^2\tilde{x}}{d\tilde{\tau}^2} = -\tilde{\nabla}\tilde{\Phi}$$
where $d\tilde{\tau} = \frac{H_0dt}{a^2}$ and $\tilde{x} = \frac{x}{B}$.
</p>

<header class="entry-header text-center">
  <h1 class="entry-title" id="GravityModelSymmetron">GravityModel : Symmetron</h1>
</header>
<hr/>

<figure class="cinematic-figure" id="cinematic-05-screened-gravity">
  <a href="images/cinematic/05-screened-gravity.jpg" aria-label="View full-size illustration: Screened gravity">
    <img src="images/cinematic/05-screened-gravity.jpg" width="1774" height="887" loading="lazy" decoding="async" alt="A dense matter halo with illustrative teal scalar-force contours subdued in its interior">
  </a>
  <figcaption><strong>Screened gravity.</strong> Conceptual scalar-field visualization: an additional force is suppressed in the dense halo and more apparent in its diffuse environment; colors and contours are illustrative. <span class="cinematic-credit">AI-generated concept illustration.</span></figcaption>
</figure> 

<p>
For the symmetron we have (when assuming the field sits at the minimum)
$$\chi(a) = \sqrt{1 - \frac{a_{\rm ssb}^3}{a^3}}$$
$$\beta(a) = \beta_0\sqrt{1 - \frac{a_{\rm ssb}^3}{a^3}}$$
$$\frac{m^2}{H_0^2} = \frac{1}{(L_0H_0)}\sqrt{1 - \frac{a_{\rm ssb}^3}{a^3}}$$
for $a\geq a_{\rm ssb}$ and $\phi,\beta$ is zero otherwise so the fifth-force is not active where $a_{\rm ssb}$ is the symmetry breaking scale-factor, $L_0$ is the range of the fifth-force in the background today and $\beta$ is the coupling strength relative to gravity (i.e. maximum unscreened fifth-force is $2\beta^2$ times gravity).

We have
$$\frac{G_{\rm eff}(a,k)}{G} = 1 + 2\beta^2(a) \frac{\left(\frac{k}{H_0}\right)^2}{\left(\frac{k}{H_0}\right)^2 + \frac{a^2m^2(a)}{H_0^2} }$$ 

The exact (quasi-static) field equation the code can solve is 
$$\tilde{\nabla}^2 \chi = \frac{a^2B_0^2}{2L_0^2}(\frac{a_{\rm ssb}^3}{a^3}\chi (1+\delta) - \chi + \chi^3) $$
where $B_0$ is the boxsize and the fifth-force potential is
$$\tilde{\Phi}_F = \frac{3\Omega_M a^2 \beta_0^2 L_0^2}{B_0^2 a_{\rm assb}^3}\chi^2 $$
and the total force $\tilde{\nabla}\tilde{\Phi} + \tilde{\nabla}\tilde{\Phi}_F$.
</p>

<p>
We also have the option to evolve the full time-dependent equation. Then we evolve
$$\frac{d}{d\log a}[a^3 E \frac{d\chi}{d\log a}] = \frac{a^3}{E}\left(\frac{\tilde{\nabla}^2\chi}{a^2(H_0B_0)^2} - \frac{1}{2(H_0L_0)^2}(\frac{a_{\rm ssb}^3}{a^3}\chi (1+\delta) - \chi + \chi^3)\right)$$
by writing this as the system 
$$\frac{d\chi}{d\log a} = \frac{Q}{Ea^3}$$
$$\frac{dQ}{d\log a} = \frac{a^3}{E}\left(\frac{\tilde{\nabla}^2}{a^2(H_0B_0)^2} - \frac{1}{2(H_0L_0)^2}(\frac{a_{\rm ssb}^3}{a^3}\chi (1+\delta) - \chi + \chi^3)\right)$$
where $Q \equiv a^3 E \frac{d\chi}{d\log a}$. Here $\chi$ acts as the "position" and $Q$ as the "velocity" and we evolve these using a usual Kick-Drift-Kick scheme. For evolving this to be stable we need very small time-steps. The way we do it is that we have a separate time-stepping for $\chi$ where we use many more steps between the usual Kick and Drift steps for the particles. The time-step is set by looking at the expected period (in the scale-factor) of oscillation for the largest density we have in the box
$$\delta a_{\rm one~period} \sim \frac{2\pi (L_0H(a)))}{\sqrt{(a_{\rm ssb}/a)^3 (1+ \delta_{\rm max})}}$$
and then set
$$\delta a = \frac{\delta a_{\rm one~period}}{N}$$
where $N$ ("nsteps per oscillation") is a free parameter that should be $\gtrsim 10$ to avoid the system breaking down.

There are also some (tiny) additional friction terms in the geodesic equation that we ignore (though these are trivial to include).

</p>

<header class="entry-header text-center">
  <h1 class="entry-title" id="GravityModelJBD">GravityModel : Jordan-Brans-Dicke (JBD)</h1>
</header>
<hr/> 

For JBD we have 
$$\frac{G_{\rm eff}(a)}{G} = \frac{1}{\phi(a)}$$
where $\phi(a=1)^{-1}$ is a free parameter (GeffG_today). This model is basically linear so there is no sizable non-linear corrections to solve for.

<header class="entry-header text-center">
  <h1 class="entry-title" id="GravityModelDGP">GravityModel : DGP</h1>
</header>
<hr/>  

For DGP we have
$$\frac{G_{\rm eff}(a)}{G} = 1 + \frac{1}{3\beta(a)}$$
where 
$$\beta(a) = 1 + 2r_cH_0 E(a)\left(1 + \frac{1}{3}\frac{d\log E}{d\log a}\right)$$
The exact field equation which the code can solve is given by
$$\tilde{\nabla}^2 \tilde{\Phi}_F + \frac{2(r_cH_0)^2}{3\beta a^4}[(\tilde{\nabla}^2 \tilde{\Phi}_F)^2 - (\tilde{\nabla}_i\tilde{\nabla}_j \tilde{\Phi}_F)^2] = \frac{1}{3\beta} \cdot \frac{3}{2}\Omega_M a \delta_M$$
and the total force is $\tilde{\nabla}\tilde{\Phi} + \tilde{\nabla}\tilde{\Phi}_F$.
The linear field equation is simply
$$\tilde{\nabla}^2 \tilde{\Phi}_F = \frac{1}{3\beta} \cdot \frac{3}{2}\Omega_M a \delta_M$$
so $\tilde{\Phi}_F = \frac{1}{3\beta} \tilde{\Phi}$ and the total force is just $\frac{G_{\rm eff}(a)}{G}$ times the newtonian force.
The approximate way of including screening is to solve
$$\tilde{\nabla}^2 \tilde{\Phi}_F = \frac{1}{3\beta} \cdot \frac{3}{2}\Omega_M a \delta_M^{\rm eff}$$
where $\delta_M^{\rm eff} = \delta_M F(\delta_M^R)$ where we allow for a smoothing of the density field that goes into the function. The screening function is
$$F(\delta) = 2\frac{\sqrt{1 + C\delta} - 1}{C},\,\,\, C = \frac{8\Omega_M (r_cH_0)^2}{9\beta^2}$$
The last option in the code is to combine the two previous solutions (linear and screened) as to ensure we recover the correct linear theory on large scales:
$$\tilde{\Phi}_F \to \tilde{\Phi}_F^{\rm linear} f(k) + \tilde{\Phi}_F^{\rm screened}(1 - f(k))$$
where $f(k) = e^{-\frac{1}{2}(k/k_{\rm linear-scale})^2}$ and $k_{\rm linear-scale}$ is a free parameter.

The modified 2LPT factor is given by:
$$h_{\rm 2LPT} = 1 - \frac{2(r_cH_0)^2\Omega_M}{9a^3\beta^3\frac{G_{\rm eff}(a)}{G}}$$
We have not included the equivalent for higher order LPT.

<header class="entry-header text-center">
  <h1 class="entry-title" id="GravityModelFOFR">GravityModel : $f(R)$</h1>
</header>
<hr/>  

<p>
For $f(R)$ gravity we have
$$\frac{G_{\rm eff}(a,k)}{G} = 1 + \frac{1}{3} \frac{\left(\frac{k}{H_0}\right)^2}{\left(\frac{k}{H_0}\right)^2 + \frac{a^2m^2(a)}{H_0^2} }$$ 
where 

$$\frac{m^2}{H_0^2} = \frac{1}{(n+1)|f_{R0}|}\left(a^{-3}\Omega_M + 4\Omega_\Lambda\right) \left(\frac{a^{-3}\Omega_M + 4\Omega_\Lambda}{\Omega_M + 4\Omega_\Lambda}\right)^{n+1}$$

and the field value in the cosmological background is given by

$$|f_{R}(a)| = |f_{R0}| \left(\frac{\Omega_M + 4\Omega_\Lambda}{a^{-3}\Omega_M + 4\Omega_\Lambda}\right)^{n+1}$$

The exact field equation which the code can solve is given by

$$\tilde{\nabla}(e^u\tilde{\nabla} u) = \frac{1}{3}\cdot \frac{3}{2}\Omega_M a \delta_M - \frac{1}{2}a^4(a^{-3}\Omega_M + 4\Omega_\Lambda)\left[ e^{\frac{u(a) - u}{n+1}} - 1\right]$$
where $\frac{a^2|f_R|}{2(H_0B_0)^2} = \tilde{\Phi}_F$ is the fifth-force potential and $e^{u(a)} = \frac{a^2|f_{R}(a)|}{2(H_0B_0)^2}$ ($B_0$ being the boxsize). The total force is given by $\tilde{\nabla}\tilde{\Phi}_F + \tilde{\nabla}\tilde{\Phi}$.

The linear field equation that the code can solve can be written
$$\tilde{\nabla}^2 \tilde{\Phi}_F = \frac{a^2 m^2(a)}{H_0^2} \tilde{\Phi}_F + \frac{1}{3}\cdot \frac{3}{2}\Omega_M a \delta_M$$

The approximate way of including the effects of screening is to solve
$$\tilde{\nabla}^2 \tilde{\Phi}_F = \frac{a^2 m^2(a)}{H_0^2} \tilde{\Phi}_F + \frac{1}{3}\cdot \frac{3}{2}\Omega_M a \delta_M^{\rm eff}$$
where the effective mass density is computed from the gravitational potential $\delta_M^{\rm eff} = \delta_M F(\Phi) $ where
$$F(\Phi) = \left|\frac{3f_R(a)}{2\Phi}\right|$$
and if this exceeds $1$ it's taken to be unity.

The last option in the code is to combine the two previous solutions (linear and screened) as to ensure we recover the correct linear theory on large scales:

$$\tilde{\Phi}_F \to \tilde{\Phi}_F^{\rm linear} f(k) + \tilde{\Phi}_F^{\rm screened}(1 - f(k))$$
where $f(k) = e^{-\frac{1}{2}(k/k_{\rm linear-scale})^2}$ and $k_{\rm linear-scale}$ is a free parameter.

For the LPT specific functions see Winther et al. (COLA with scale-dependent growth).
</p>

<header class="entry-header text-center">
  <h1 class="entry-title" id="MassiveNu">Massive neutrinos</h1>
</header>
<hr/>  

<p>
The massive neutrinos as computed by storing the initial density field $\delta_{\rm cb}(k,a_{\rm ini})$ and the neutrino density at other times is then found as
$$\delta_{\nu}(k,a) = \delta_{\rm cb}(k,a_{\rm ini}) \frac{T_{\nu}(k,a)}{T_{\rm cb}(k,a_{\rm ini})}$$
where $T_{\nu}$ is the massive neutrino transfer function and $T_{\rm cb}$ is the CDM+baryon transfer function (obtained from an Einstein-Boltzmann solver).

The total density field in fourier space is then computed as
$$\delta_M(k,a) =  \frac{\Omega_M - \Omega_{M\nu}}{\Omega_M}\delta_{\rm cb}(k,a) + \frac{\Omega_{M\nu}}{\Omega_M}\delta_{\nu}(k,a)$$
wher $\delta_{\rm cb}(k,a)$ comes from binning particles to the grid.
</p>

<header class="entry-header text-center">
  <h1 class="entry-title" id="COLA">COLA</h1>
</header>
<hr/>  

<p>
In COLA simulations we solve for displacements on top of that predicted by LPT. With this we can do simulations much faster than usual N-body (by sacrificing accuracy on small scales). Instead of propagating $\tilde{x}$ and $\tilde{v}$ via
$$\frac{d \tilde{x}}{d\tilde{\tau}} = \tilde{v}$$
$$\frac{d \tilde{v}}{d\tilde{\tau}} = - \tilde{\nabla}\tilde{\Phi}$$
we instead define $\tilde{x}_{\rm COLA} = \tilde{x} - \tilde{x}_{\rm LPT}$ so that the equation becomes
$$\frac{d\tilde{x}}{d\tilde{\tau}} = \tilde{v}_{\rm COLA} + \frac{d\tilde{x}_{\rm LPT}}{d\tilde{\tau}}$$
$$\frac{d \tilde{v}_{\rm COLA}}{d\tilde{\tau}} = - \tilde{\nabla}\tilde{\Phi} - \frac{d^2\tilde{x}_{\rm LPT}}{d\tilde{\tau}^2}$$
and we propagate $\tilde{x}$ and $\tilde{v}_{\rm COLA}$ instead. Note that in this frame the initial velocity is $\tilde{v}_{\rm COLA} = 0$ and this helps us keep in line with linear theory on the largest scales even with large time-steps. 
</p>

<p>
</p>

<p>
If the growth-factors $D_i$ do not depend on $k$ (or if we ignore this dependence) then the path predicted by LPT is given by 
$$\tilde{x}_{\rm LPT} \equiv \tilde{\Psi}^{\rm 1LPT} + \tilde{\Psi}^{\rm 2LPT} + \ldots = \frac{D_1(a)}{D_1(a_{\rm ini})}\tilde{\Psi}^{\rm 1LPT}(q,a_{\rm ini}) + \frac{D_1(a)}{D_1(a_{\rm ini})}\tilde{\Psi}^{\rm 2LPT}(q,a_{\rm ini}) + \ldots$$
and the displacement fields (in configuration space; Lagrangian coordinates) are given by
$$\tilde{\Psi}^{iLPT}(q,a) = \tilde{\Psi}^{iLPT}(q,a_{\rm ini}) \frac{D_i(a)}{D_i(a_{\rm ini})}$$
$$\frac{d}{d\log a}\tilde{\Psi}^{iLPT}(q,a) = \tilde{\Psi}^{iLPT}(q,a_{\rm ini}) f_i(a)\frac{D_i(a)}{D_i(a_{\rm ini})}$$
where $f_i = \frac{d\log D_i}{d\log a}$ is the growth rate.

We store the displacement field at the particles when we create them (from an initial Lagrangian grid) and then we can get the displacement field at any time simply by multiplying with growth-factors.
</p>

<p>
For scale-dependent COLA it's much more involved. We have the same equations above just that they only hold in Fourier space so the previous strategy of storing the results in the particles is out the window! To get the displacement field at any time we must compute
$$\tilde{\Psi}^{iLPT}(k,a) = \tilde{\Psi}^{iLPT}(k,a_{\rm ini}) \frac{D_i(a,k)}{D_i(a_{\rm ini},k)}$$
$$\frac{d}{d\log a}\tilde{\Psi}^{iLPT}(k,a) = \tilde{\Psi}^{iLPT}(k,a_{\rm ini}) f_i(a,k)\frac{D_i(a,k)}{D_i(a_{\rm ini},k)}$$
at every step by performing Fourier transforms. Then the particles must fetch the displacement field from their original Lagrangian positions $q$ (not the Eulerian positions $x$ they are physically at). The way this is done is to store both Eulerian and Lagrangian positions in the particle, these are then swapped and we press communicate for which the particles end up on the task they started off from and we can assign the displacement fields swap back and communicate them back. The communication is typically not a big issue for performance, but the Fourier transforms can be. Thus if you want the code to run really fast you probably shouldn't use scale-dependent COLA unless you really have to (there is a flag for this and you can do tests to see the effect of it). However if the force grid is larger than the grid we used to create the IC (as it typically is) then this extra cost is not that bad.
</p>

<header class="entry-header text-center">
  <h1 class="entry-title" id="Cosmology">CosmologyModel</h1>
</header>
<hr/>  

<p>
The neutrinos energy density is computed as
$$\frac{\rho_\nu(a)}{\rho_{\rm crit 0}} = \frac{\Omega_{\nu}}{a^4} \frac{B(y)}{B(y=0)}$$
where the Boltzmann integral $B = \int_0^\infty \frac{x^2\sqrt{x^2 + y^2}}{e^x + 1}\,{\rm d}x$ and $y = \frac{m_\nu}{N_\nu k_bT_\nu(a)}$.
</p>

<header class="entry-header text-center">
  <h1 class="entry-title" id="CosmologyLCDM">CosmologyModel: $\Lambda$CDM</h1>
</header>
<hr/>  

<p>
The Hubble equation is given by
$$E^2(a) = \left[ \frac{\Omega_R}{a^4} + \frac{\rho_\nu(a)}{\rho_{\rm crit 0}}  + \frac{\Omega_b+\Omega_{\rm CDM}}{a^3} + \frac{\Omega_K}{a^2} + \Omega_\Lambda \right]$$
</p>

<header class="entry-header text-center">
  <h1 class="entry-title" id="CosmologyJBD">CosmologyModel : JBD</h1>
</header>
<hr/>  

<p>
The Hubble equation is given by
$$E^2(a) = \frac{1}{\phi(a)}\left[ \frac{\Omega_R}{a^4} + \frac{\rho_\nu(a)}{\rho_{\rm crit 0}}  + \frac{\Omega_b+\Omega_{\rm CDM}}{a^3} + \frac{\Omega_K}{a^2} + \Omega_\Lambda \right] \frac{1}{1 + \frac{d\log \phi}{d\log a} - \frac{\omega_{\rm BD}}{6}\left(\frac{d\log \phi}{d\log a}\right)^2 }$$
And the equation for the scalar field is given by ... (to add this)

These two equation are solve using shoot-match to figure out the initial condition we need for $\phi$ to give the desired value of $\phi(a=1)$ (which determines the Newtonian gravitational constant today). For this we require physical density parameters and $h$ is a derived quantity.
</p>

<!--
<header class="entry-header text-center">
  <h1 class="entry-title" id="Nongaussianity">Non-gaussianity</h1>
</header>
<hr/>  

<p>
For the non-gaussian potential we generate a gaussian random field $\phi$ using the primordial power-spectrum $P(k) = \frac{2\pi^2}{k^3}A_s\left(\frac{k}{k_{\rm pivot}}\right)^{n_s-1}$ and convert this to a power-spectrum of the Bardeen potential by multiplying by $\frac{3}{2}\Omega_M a_{\rm fnl}^{-1}\frac{D(a_{\rm fnl})}{D(a = 1)}$. We use this to create a non-gaussian random field $\phi + f_{\rm NL}K(\phi,\phi)$ and finally we normalize this field as to have the desired initial power-spectrum.
</p>
-->

<header class="entry-header text-center">
  <h1 class="entry-title" id="MuGravity">GravityModel : $\mu(a,k)$ and $G_{\rm eff}$</h1>
</header>
<hr/>

<p>
Phenomenological modified gravity without screening: the Poisson equation is solved with $G_{\rm eff}/G = \mu(a,k)$,
$$\nabla^2\Phi = 4\pi G a^2\mu(a,k)\bar\rho\delta,$$
with $\mu = 1 + \mu_0\Omega_{\rm DE}(a)/\Omega_{\rm DE,0}$ (the late-time parametrization of Planck and DES), the Bertschinger&ndash;Zukin form $\mu = \frac{1 + \beta_1\lambda_1^2k^2a^s}{1 + \lambda_1^2k^2a^s}$, or a table $\mu(a)$ / $\mu(a,k)$ (e.g. from hi_class). The LPT growth factors are solved with $\mu(a,k)$, so a $k$-dependent $\mu$ gives scale-dependent growth (use scale-dependent COLA). The ratio $P_\mu/P_{\rm GR}$ agrees with $(D_\mu/D_{\rm GR})^2$ from an independent growth solution to 0.05% for $k\le 0.1\,h$/Mpc. <tt>Geff</tt> reads $G_{\rm eff}/G(a)$ from a file.
</p>

<header class="entry-header text-center">
  <h1 class="entry-title" id="Species">Separate CDM and baryons, neutrino particles</h1>
</header>
<hr/>

<p>
<b>CDM and baryons</b> (<tt>particle_baryons = true</tt>) are two sets of particles with masses in the ratio $\Omega_b/\Omega_{\rm CDM}$ that both feel the total CDM+baryon density. Their initial conditions use the species transfer functions from one random field, $\delta_i = \delta_{cb}T_i/T_{cb}$, with velocities from $f_i = f_{cb} + d\ln(T_i/T_{cb})/d\ln a$, so the relative baryon&ndash;CDM mode left over from before decoupling is followed (it agrees with linear theory, CAMB $1 - T_b/T_{\rm CDM}$, on all scales when both species start on the same lattice; interleaved lattices give a large spurious difference in a PM code). With COLA both species share the frame of the total field.
</p>
<p>
<b>Massive neutrinos as particles</b> (<tt>particle_neutrinos = true</tt>) use the $\delta f$ method of Elbers et al. (2021): each particle gets a thermal momentum drawn from the Fermi&ndash;Dirac distribution, moves with relativistic equations of motion, and carries the weight $w = 1 - f_0(q)/f_i$ so that only the perturbation of the distribution function contributes to the density (much less shot noise). The force is sourced by $\delta_m = (1-f_\nu)\delta_{cb} + f_\nu\delta_\nu$. The neutrino power spectrum follows linear theory to 2% (0.4 eV) and the effect on $P_{cb}$ agrees with the linear-neutrino option to $&lt;0.2\%$.
</p>

<header class="entry-header text-center">
  <h1 class="entry-title" id="EarlyForce">Forces, the lattice offset and starting from the Big Bang</h1>
</header>
<hr/>

<p>
With a force grid finer than the particle lattice by an integer factor the particles start exactly on grid nodes, where the CIC kernel has a kink and the density responds to the size of the displacement; this gives spurious forces at the lattice frequency. The code offsets the lattice by half a force cell and offers other gradient kernels and an accurate force on a grid equal to the particle grid (interlacing, TSC and deconvolution, or the exact NUFFT force). A BullFrog step is exact for 2LPT, so the simulation can start at (almost) $a = 0$ from the particle lattice with the Zel'dovich velocity: the first step produces the 2LPT displacement itself (List, Hahn &amp; Rampf 2024). This needs an accurate force while the particles are close to the lattice, so <tt>force_early_method = "nufft"</tt> uses the NUFFT force at $z &gt; 10$; then 10 steps from $z = 10^7$ give $P(k)$ within 0.2% of the reference up to $k = 0.4\,h$/Mpc. The growth-factor tables start in the radiation era with the M&eacute;sz&aacute;ros growing modes.
</p>

<header class="entry-header text-center">
  <h1 class="entry-title" id="NbodyGauge">The N-body gauge</h1>
</header>
<hr/>

<p>
On very large scales a Newtonian simulation misses the effect of radiation and massive neutrinos on the growth. In the N-body gauge (Fidler et al. 2015, 2016) the particles follow the Newtonian equations exactly to linear order in GR if the force is sourced by $\rho_{cb}(\delta_{cb} + \delta_{\rm extra})$, where $\delta_{\rm extra}(k,z)$ is a linear field that contains the photons, neutrinos and a metric term. With <tt>force_nbody_gauge = true</tt> this source is added at every force computation and the initial conditions are made from N-body-gauge transfer functions (made with CLASS by <tt>scripts/make_nbody_gauge_transfer.py</tt>). For a start at $z = 49$ the effect on $P(k,z=0)$ is about 2% at $k = 10^{-3}\,h$/Mpc.
</p>

<header class="entry-header text-center">
  <h1 class="entry-title" id="ICs">Initial conditions: Philox fields and internal transfer functions</h1>
</header>
<hr/>

<p>
The Gaussian field can be generated with the 2LPTic/N-GenIC generator (the default, to match other codes) or with a counter-based Philox generator keyed by the wavevector (<tt>ic_random_field_backend = "philox"</tt>), which gives the same modes for any resolution, MPI decomposition and thread count (see <a href="randomfield.php" style="display: inline">random fields</a>). Non-Gaussian initial conditions (local, equilateral, orthogonal $f_{\rm NL}$) can be imposed at any redshift (LSS or CMB convention), together with fixed amplitudes and paired phases. Besides $P(k)$ and CAMB transfer tables, the initial conditions can come directly from the library's <a href="cosmology.php" style="display: inline">Einstein&ndash;Boltzmann solver</a> (<tt>ic_type_of_input = "internal"</tt> or precomputed <tt>"tables"</tt>), with CDM and baryon transfer functions converted to the CDM-synchronous gauge.
</p>

<header class="entry-header text-center">
  <h1 class="entry-title" id="Analysis">On-the-fly analysis: spectra, halos, hybrid bias, baryonification and images</h1>
</header>
<hr/>

<ul>
<li><b>Power spectra</b> every step (CDM+baryon and total matter), and at the outputs $P(k)$ with interlacing, redshift-space multipoles and the bispectrum.</li>
<li><b>Halos:</b> FoF catalogues and mass functions (with the Tinker and Watson fits), plus spherical-overdensity masses (200m, 200c, vir, ...) with shrinking-sphere centres.</li>
<li><b>Hybrid bias spectra</b> (<a href="https://arxiv.org/abs/1910.07097" style="display: inline">Modi, Chen &amp; White 2020</a>; <a href="https://arxiv.org/abs/2101.11014" style="display: inline">Kokron et al. 2021</a>; <a href="https://arxiv.org/abs/2110.05408" style="display: inline">Zennaro et al. 2021</a>): the 15 cross spectra of the Lagrangian bias operators $1, \delta_L, \delta_L^2, s^2, \nabla^2\delta_L$ advected with the particles, with the Zel'dovich and one-loop LPT predictions (EdS or the full scale- and time-dependent kernels of $f(R)$ and nDGP with screening). With <tt>hybridbias_rsd = true</tt> also their redshift-space multipoles $P^\ell_{ab}(k)$ (the operators at the redshift-space positions of the particles, $\ell \le$ <tt>hybridbias_rsd_ellmax</tt>), with the exact Zel'dovich prediction, the Zel'dovich control variate and the one-loop LPT prediction (<a href="https://arxiv.org/abs/2012.04636" style="display: inline">Chen et al. 2021</a>, extended to $\nabla^2\delta_L$).</li>
<li><b>Snapshot baryonification:</b> the <a href="halomodel.php#Baryons" style="display: inline">baryonification</a> of each output (on a copy of the particles) with the corrected/original $P(k)$ ratio.</li>
<li><b>Lightcone</b> particles (ascii, GADGET or HDF5, in batches), HEALPix onion-shell density maps and their $C_\ell$; optionally the potential, radial velocity and observed redshift $1 + z_{\rm obs} = (1+z)(1 + v_r/c - \Phi/c^2)$ of each particle. With <tt>plc_halos = true</tt> also friends-of-friends <b>halos on the lightcone</b>: found every step among the particles near the shell of the step and recorded where their center of mass crosses the lightcone (the same catalogue as linking all the particles; halo counts per shell within 2% of FoF on the particle lightcone, at about 10-15% extra run time).</li>
<li>Particles in GADGET binary or HDF5 format.</li>
<li><b>Images</b> (<tt>images = true</tt>): a PNG (and optionally the values as <tt>.npy</tt>) of a slab of each chosen field projected along an axis, <tt>image_&lt;field&gt;_z&lt;z&gt;.png</tt> in the snapshot folder, made with <a href="other.php#Visualization" style="display: inline">FML/Visualization</a>. The fields (<tt>images_fields</tt>): <tt>density</tt>; <tt>velocity_divergence</tt>, $-\nabla\cdot v/(aH)$, which equals $f\delta$ in linear theory; <tt>baryons_minus_cdm</tt>; <tt>neutrinos</tt>; and <tt>fifth_force</tt>, $|F_5|/|F_N|$, computed with the gravity model's own screening method or exact solver, so screened regions are dark. The images do not change the run.</li>
</ul>
<figure class="science-figure">
  <img src="images/simulation_images.png" alt="Six projected slabs of a small test simulation at z = 0: the density, the velocity divergence, baryons minus CDM, the linear neutrino density, and the density and the fifth-force ratio of an f(R) run">
  <figcaption>The images of the test runs at $z = 0$ ($32^3$ particles in 256 Mpc/$h$): density, velocity divergence, $\delta_b - \delta_c$, linear neutrinos, and the density and $|F_5|/|F_N|$ of an $f(R)$ run with screening.</figcaption>
</figure>
<p>
All parameters, with types, defaults and allowed values, are listed in <tt>SimulationCode/PARAMETERS.md</tt> (<tt>./nbody --parameters PARAMETERS.md</tt>); misspelled parameter names give a warning with a suggestion.
</p>

<header class="entry-header text-center">
  <h1 class="entry-title" id="AxionIC">Ultralight axions in initial conditions</h1>
</header>
<hr/>

<p>The internal Boltzmann provider can use a mixed cold/ultralight-axion linear model to generate particle initial conditions. Set <tt>ic_axion_fraction</tt> and <tt>ic_axion_log10_m</tt> with <tt>ic_type_of_input = "internal"</tt> (or matching precomputed tables). The particle transfer represents CDM plus axions and the particles carry all dark matter, subsequently evolved as collisionless matter in an LCDM simulation. Axion wave dynamics and a separate nonlinear axion component are not evolved. See <a href="https://github.com/HAWinther/FML/blob/master/FML/InitialConditions/README.md" style="display: inline">initial-condition model scope</a> before interpreting this as a mixed-dark-matter simulation.</p>

<header class="entry-header text-center">
  <h1 class="entry-title" id="SnapshotAnalysis">Post-process snapshots with shared analysis</h1>
</header>
<hr/>

<p><a href="snapshotanalyzer.php" style="display: inline">SnapshotAnalyzer</a> runs independently on Gadget binary or HDF5 outputs. It shares the library's power, multipole, bispectrum, FoF/SO and baryonification kernels with SimulationCode and adds configurable T-web, Minkowski, hybrid-bias, void and DTFE workflows. Use it to reanalyse existing snapshots without rerunning their dynamics. <a href="https://github.com/HAWinther/FML/blob/master/FML/Analysis/README.md" style="display: inline">FML/Analysis</a> also exposes the context and settings for use from another application.</p>

<header class="entry-header text-center">
  <h1 class="entry-title" id="References">Method references and scope</h1>
</header>
<hr/>

<p>
Dark scattering follows <a href="https://arxiv.org/abs/1007.1034" style="display: inline">Simpson (2010)</a> and <a href="https://arxiv.org/abs/1412.1080" style="display: inline">Baldi &amp; Simpson (2015)</a>: momentum exchange changes the CDM velocity evolution while baryons remain uncoupled. Coupled quintessence follows <a href="https://arxiv.org/abs/astro-ph/9908023" style="display: inline">Amendola (2000)</a>, with varying CDM mass, a fifth force and a velocity term. IDE uses the fluid interaction discussed by <a href="https://arxiv.org/abs/0804.0232" style="display: inline">Valiviita, Majerotto &amp; Maartens (2008)</a>; FML's particle masses also respond to local expansion. These are distinct prescriptions; their species and large-scale limitations, including <a href="https://arxiv.org/abs/2510.12551" style="display: inline">covariant dark scattering</a>, are in <a href="https://github.com/HAWinther/FML/blob/master/FML/SimulationCode/MODELS.md" style="display: inline">MODELS.md</a>.
</p>
<p>
The cubic Galileon uses the quasi-static screening treatment discussed by <a href="https://arxiv.org/abs/1306.3219" style="display: inline">Barreira et al. (2013)</a>. Reduced Horndeski models use the Lagrangian-to-background/force workflow of <a href="https://arxiv.org/abs/2209.01666" style="display: inline">Wright et al. (2023), Hi-COLA</a> and the quasi-static force derivation of <a href="https://arxiv.org/abs/1111.6749" style="display: inline">Kimura, Kobayashi &amp; Yamamoto (2012)</a>. FML has its own symbolic frontend and force tables. Approximate screening follows <a href="https://arxiv.org/abs/1403.6492" style="display: inline">Winther &amp; Ferreira (2014)</a> and, for Vainshtein COLA, <a href="https://arxiv.org/abs/2303.09549" style="display: inline">Brando, Koyama &amp; Winther (2023)</a>; the model documentation specifies the supported force solvers and approximations.
</p>
