<hr/>
<header class="entry-header text-center">
  <h1 class="entry-title">Table of contents</h1>
</header>
<hr/>

<p class="page-toc">
<a href="index.php">Back to the main page</a><br>
<a href="documentation.php">Back to the documentation</a><br>
<a href="#Overview">1. Overview: field-level inference</a><br>
<a href="#Model">2. The forward models</a><br>
<a href="#Likelihood">3. Posterior and parametrization</a><br>
<a href="#Gradient">4. Adjoint gradients</a><br>
<a href="#Sampler">5. Sampling with HMC</a><br>
<a href="#Validation">6. Validation and simulation-based calibration</a><br>
<a href="#References">Method references and scope</a><br>
<a href="#Framework">Periodic field-level inference: serial and MPI</a><br>
<a href="#Outputs">Reconstruction products and restart</a><br>
<a href="#PM">Particle-mesh forward model, adjoint and posterior</a><br>
</p>

<header class="entry-header text-center">
  <h1 class="entry-title" id="Overview">Overview: field-level inference</h1>
</header>
<hr/>

<figure class="cinematic-figure" id="cinematic-06-reconstructing-universe">
  <a href="images/cinematic/06-reconstructing-universe.jpg" aria-label="View full-size illustration: Reconstructing the hidden Universe">
    <img src="images/cinematic/06-reconstructing-universe.jpg" width="1774" height="887" loading="lazy" decoding="async" alt="Layered observed galaxy tracers, an inferred continuous matter field and primordial fluctuations">
  </a>
  <figcaption><strong>Reconstructing the hidden Universe.</strong> Conceptual reconstruction from discrete observed tracers to a continuous density field and initial fluctuations; it is not a unique recovered realization. <span class="cinematic-credit">AI-generated concept illustration.</span></figcaption>
</figure>

<p>
Standard analyses compress the galaxy field into summary statistics (power spectrum, bispectrum, ...) and lose information whenever the field is non-Gaussian. Field-level inference instead compares a forward model of the full field with the data mode by mode and samples the joint posterior of the initial conditions and the cosmological parameters (e.g. BORG, LEFTfield; for the BAO scale see Babi&#263; et al. 2024, <a href="https://arxiv.org/abs/2407.01524" style="display: inline">arXiv:2407.01524</a>). <tt>FML/FieldLevel</tt> now provides a periodic inference framework and a separate compact BAO prototype. The framework supports serial and MPI linear/1LPT/2LPT models, joint field and nuisance inference, reconstruction outputs and checkpoint/restart. The BAO prototype described in the following sections targets the scale $\beta = r_{\rm BAO}/r_{\rm fid}$: the parameters are the $N^3$ values of the initial white-noise field and $\beta$, the forward model is linear, 1LPT or 2LPT with Lagrangian bias and redshift-space distortions, and the posterior is sampled with <a href="inference.php#HMC" style="display: inline">Hamiltonian Monte Carlo</a> using hand-written adjoint gradients. This BAO application runs on one node (FFTWGrid, OpenMP), with bias, growth-rate and noise parameters held fixed. For joint nuisance inference and MPI, use the <a href="#Framework" style="display: inline">general framework below</a>.
</p>

<header class="entry-header text-center">
  <h1 class="entry-title" id="Model">BAO prototype: the forward models</h1>
</header>
<hr/>

<p>
<b>Initial field.</b> The white noise $s$ (one unit Gaussian per cell) is coloured with the BAO spectrum and cut sharply at $k_{\rm max}$:
$$\delta_L(\vec k) = T(k,\beta)\,s(\vec k),\qquad T = \sqrt{P(k,\beta)N^3/V},\qquad P(k,\beta) = P_{\rm smooth}(k)\left[1 + A\sin(k\beta r_{\rm fid} + \varphi)e^{-k/k_D}\right],$$
with an Eisenstein&ndash;Hu no-wiggle $P_{\rm smooth}$ by default (any function can be supplied).
</p>
<p>
<b>Linear:</b> $\delta_{\rm pred} = b\,\delta_L$. <b>LPT models:</b> one particle per cell at the Lagrangian position $\vec q$ is moved by
$$\vec x = \vec q + \vec\Psi_1 + \vec\Psi_2 + f\left(\Psi_{1,z} + 2\Psi_{2,z}\right)\hat z,\qquad \vec\Psi_1(\vec k) = \frac{i\vec k}{k^2}\delta_L(\vec k),\qquad \vec\Psi_2 = -\frac{3}{7}\nabla\phi_2,$$
$$\nabla^2\phi_2 = \sum_{i&gt;j}\left[\phi_{,ii}\phi_{,jj} - \phi_{,ij}^2\right] = \frac{1}{2}\left[\delta_L^2 - t_{ij}t_{ij}\right],\qquad t_{ij} = \partial_i\partial_j\nabla^{-2}\delta_L$$
(the last term is the redshift-space displacement along a fixed line of sight with growth rate $f$). Each particle carries a Lagrangian bias weight
$$w = 1 + b_1^L\delta_L + b_2\left(\delta_L^2 - \langle\delta_L^2\rangle\right) + b_{G_2}\mathcal{G}_2,\qquad \mathcal{G}_2 = t_{ij}t_{ij} - \delta_L^2,\qquad b_1^L = b - 1,$$
and the prediction is the Fourier transform of the weighted particles, $\delta_{\rm pred}(\vec k) = \frac{1}{N_p}\sum_pw_pe^{-i\vec k\cdot\vec x_p}$. On large scales it tends to the Kaiser limit $(b + f\mu^2)\delta_L$ (checked to 0.3%). The particles are assigned to a grid of $2N$ cells with a compact smooth kernel (6 cells wide) whose Fourier transform is divided out, which reproduces the direct sum over particles to $3\cdot 10^{-5}$ and keeps the gradient continuous. (Cloud-in-cell assignment is also available. With particles started exactly on the grid nodes, CIC responds to a displacement through $|\Psi|$, which made a plane wave of amplitude $10^{-4}$ produce a second harmonic at 38% of the fundamental, so CIC particles start mid-cell.)
</p>

<header class="entry-header text-center">
  <h1 class="entry-title" id="Likelihood">Posterior and parametrization</h1>
</header>
<hr/>

<p>
The data are the modes $d(\vec k)$ with $0 &lt; |\vec k| &lt; k_{\rm max}$, with Gaussian noise of power $P_\epsilon$:
$$-\ln P(s,\beta|d) = \frac{1}{2}\sum_{\rm cells}s^2 + \frac{1}{2}\sum_{0&lt;|\vec k|&lt;k_{\rm max}}\frac{|d(\vec k) - \delta_{\rm pred}(\vec k)|^2}{P_\epsilon/V},$$
the sum running over the full Hermitian set of modes (the exact likelihood of a real field). When the data determine a mode well, changing $\beta$ at fixed $s$ changes the predicted field, so $\beta$ and $s$ are strongly correlated and the sampler crawls. The sampled field is therefore partially centred, mode by mode,
$$s(\vec k) = \left(\frac{T(k,\beta_{\rm ref})}{T(k,\beta)}\right)^{c_k}x(\vec k),\qquad c_k = \frac{b^2P}{b^2P + P_\epsilon}$$
($c = 0$ samples the white noise, $c = 1$ the density), with the prior and the Jacobian included exactly. For the linear model this lowered the autocorrelation time of $\beta$ from about 13 to about 3 iterations.
</p>

<header class="entry-header text-center">
  <h1 class="entry-title" id="Gradient">Adjoint gradients</h1>
</header>
<hr/>

<p>
HMC needs $\nabla U$ with respect to all $N^3 + 1$ parameters. It is computed with the adjoint (reverse-mode) chain rule at about twice the cost of the forward model: starting from $\bar G(\vec k) = -(d - \delta_{\rm pred})/\sigma^2$ on the modes, the adjoint of the Fourier transform gives $\bar\delta$ on the density grid; the adjoint of the assignment gives, per particle, the position adjoint $\bar{\vec x}_p = w_p\sum_y\bar\delta_y\nabla W(\vec x_p - \vec y)$ and the weight adjoint $\bar w_p = \sum_y\bar\delta_yW(\vec x_p - \vec y)$; these propagate back through the redshift-space mapping, the 2LPT operator (whose adjoint is $\bar S_2 = \frac{3}{7}\sum_cD_c^T\bar\Psi_{2,c}$ with $D_c^T = -D_c$), the bias weights and $\mathcal{G}_2$ to $\bar\delta_L$, $\bar t_{ij}$ and $\bar{\vec\Psi}_1$, and finally
$$Q(\vec k) = \widetilde{\bar\delta_L}(\vec k) + \sum_{ij}\frac{k_ik_j}{k^2}\widetilde{\bar t_{ij}}(\vec k) - \sum_c\frac{ik_c}{k^2}\widetilde{\bar\Psi_{1,c}}(\vec k),\qquad \nabla_xU = \mathcal{F}^{-1}[T_{\rm eff}Q] + \nabla_x(\text{prior}),\qquad \frac{\partial U}{\partial\beta} = N^3\sum_{\vec k}{\rm Re}\left[Q^*\frac{\partial T_{\rm eff}}{\partial\beta}x\right] + \ldots$$
All gradients agree with finite differences to about $10^{-8}$, for every model and parametrization.
</p>

<header class="entry-header text-center">
  <h1 class="entry-title" id="Sampler">Sampling with HMC</h1>
</header>
<hr/>

<p>
The field block gets the Fourier-diagonal mass $M_k = 1 + (b + f\mu^2)^2P(k)/P_\epsilon$ (the exact Hessian of the linear model), applied with FFTs, and $\beta$ a diagonal mass adapted during warmup; $\beta$ reflects at the edges of its uniform prior. A target acceptance of about 0.65 mixes better than the default 0.8.
</p>
<div style="text-align: center">
  <div style="width:700px;display: inline-block;text-align: left">
    <pre class="prettyprint">
#include &lt;FML/FieldLevel/FieldLevelBAO.h&gt;
using namespace FML::FIELDLEVEL;
Settings st;
st.nmesh = 32;  st.boxsize = 1000;  st.kmax = 0.095;
st.model = Model::LPT;  st.lpt_order = 2;
st.bias = 1.5;  st.b2 = 0.3;  st.bG2 = -0.2;  st.growth_rate = 0.7;
st.noise_power = 1000;
FieldLevelBAO fl(st, BAOSpectrum::eisenstein_hu());

std::mt19937_64 rng(42);
fl.set_data(fl.make_mock(fl.white_noise(rng), 1.02, rng));   // or your own data modes

FML::MCMC::HMCOptions opt;
opt.warmup = 150;  opt.samples = 400;  opt.n_leapfrog = 12;  opt.target_acceptance = 0.65;
auto chain = sample(fl, fl.white_noise(rng), 1.0, opt);
auto beta  = FML::MCMC::recorded_column(chain, 0);</pre>
  </div>
</div>

<header class="entry-header text-center">
  <h1 class="entry-title" id="Validation">Validation and simulation-based calibration</h1>
</header>
<hr/>

<p>
For the linear model the marginal posterior of $\beta$ is known exactly, $\ln\mathcal{L}(\beta) = -\frac{1}{2}\sum_{\vec k}\left[|d|^2/C_k + \ln C_k\right]$ with $C_k = (b^2P + P_\epsilon)/V$, and the HMC chains agree with it (32$^3$, $k_{\rm max} = 0.095\,h$/Mpc: exact $1.0232\pm 0.0333$, HMC $1.0221\pm 0.0330$). The 2LPT displacement of two crossed plane waves equals the analytic one to $10^{-16}$, also in redshift space.
</p>
<p style="text-align: center">
<img src="images/fieldlevel_chains.png" alt="Field-level chains" style="max-width: 900px; width: 100%;"><br/>
<i>Chains and posteriors of $\beta$: the linear model against its exact marginal posterior (top) and 1LPT (bottom).</i>
</p>
<p>
The whole chain (forward model, gradients and sampler) is tested statistically with simulation-based calibration (Talts et al. 2018): for each mock, $\beta$ is drawn from the prior, the initial field from its prior and the data from the forward model plus noise, and the posterior is sampled. If everything is correct, $z = (\bar\beta - \beta_{\rm true})/\sigma_\beta$ is distributed as $\mathcal{N}(0,1)$ and the posterior quantile of $\beta_{\rm true}$ is uniform.
</p>
<table class="table table-striped table-condensed">
<thead><tr><th>Model (32$^3$, $k_{\rm max} = 0.095\,h$/Mpc)</th><th>Mocks</th><th>mean $z$</th><th>std $z$</th><th>KS $p$ (quantiles uniform)</th><th>$\sigma_\beta$</th></tr></thead>
<tbody>
<tr><td>Linear (HMC)</td><td>30</td><td>$-0.11\pm 0.19$</td><td>1.04</td><td>0.76</td><td>0.033</td></tr>
<tr><td>Linear (exact posterior)</td><td>30</td><td>$-0.08$</td><td>1.03</td><td>0.88</td><td>0.033</td></tr>
<tr><td>1LPT (HMC, first CIC model)</td><td>24</td><td>$-0.07\pm 0.20$</td><td>1.00</td><td>0.85</td><td>0.037</td></tr>
<tr><td>2LPT + $b_2$ + $b_{G_2}$ + RSD (HMC)</td><td>20</td><td>$-0.20\pm 0.24$</td><td>1.05</td><td>0.57</td><td>0.035</td></tr>
</tbody>
</table>
<p style="text-align: center">
<img src="images/fieldlevel_calibration_2lpt.png" alt="Calibration of the 2LPT model" style="max-width: 900px; width: 100%;"><br/>
<i>Simulation-based calibration of the 2LPT + Lagrangian bias + redshift-space model.</i>
</p>
<p>
These tests check that the sampler recovers the posterior of the model that generated the data; they do not test the forward model against simulations. The separate framework now supports joint field/bias/noise inference and MPI linear/1LPT/2LPT models (see below). Dealiasing and tests against independent N-body mocks remain separate requirements for scientific use.
</p>

<header class="entry-header text-center">
  <h1 class="entry-title" id="Framework">Periodic field-level inference: serial and MPI</h1>
</header>
<hr/>

<p>
The general framework accepts a fixed or differentiable parameterized initial power spectrum, a linear, Zel'dovich or EdS-factorized 2LPT forward model, and a Gaussian tracer-field likelihood. It can infer spectrum parameters, bias and positive noise power jointly with the white initial field. For LPT the scalar bias block is $b_1,b_2,b_{s^2},b_{\nabla^2}$; plane-parallel RSD uses a fixed growth rate. Initial, bias and likelihood cutoffs are separate. A fixed Fourier response can model a transfer function; a survey mask and a count likelihood are not yet included.
</p>
<p>
<tt>Model</tt> is the single-rank implementation; <tt>DistributedLinearModel</tt> and <tt>DistributedLPTModel</tt> keep fields and modes local to FFTW slabs or, with <tt>USE_PENCIL_GRID</tt>, <a href="pencilgrid.php" style="display: inline">PencilGrid</a> pencils. MAP and HMC use the same interfaces, with exact Gaussian conditionals available for linear evolution. All ranks must participate in each collective call with matching scalar parameters and controls. Even meshes must be divisible by the rank count and have nonempty slabs; double FFTW is the validated configuration. Distributed LPT requires an initial cutoff strictly below half the initial-grid Nyquist and an even density mesh at least as large as the initial mesh.
</p><div style="text-align: center"><div style="max-width:800px;display:inline-block;text-align:left;width:100%"><pre class="prettyprint">#include &lt;FML/FieldLevel/DistributedLPT.h&gt;
namespace FF = FML::FIELDLEVEL::FRAMEWORK;
// After FML initialization, on every rank:
FF::Configuration c;
c.evolution = FF::Evolution::LPT2;
c.mesh = 8; c.density_mesh = 16; c.boxsize = 100;
c.initial_cutoff = .115; c.bias_cutoff = .08; c.likelihood_cutoff = .10;
FF::DistributedLPTModel model(c, FF::Spectrum::fixed([](double) { return 200.; }));
auto work = model.workspace();
auto p = model.initial_parameters();
auto data = model.mock(model.white_noise(10), p, 11, work);
FF::DistributedLPTPosterior posterior(model, data);
auto fit = FF::map(posterior, model.white_noise(12), p, false);
// A small periodic mock with field MAP at fixed scalar parameters.</pre></div></div>
<p>Adjoint, exact-linear and 1/2/4-rank checks establish numerical consistency. Nonlinear posterior calibration, high-cutoff padded adjoints, survey likelihoods and COLA adjoints remain open (the PM model below has its adjoint). See <a href="https://github.com/HAWinther/FML/blob/master/FML/FieldLevel/docs/FRAMEWORK.md" style="display: inline">the framework API and validation limits</a>.</p>

<header class="entry-header text-center">
  <h1 class="entry-title" id="Outputs">Reconstruction products and restart</h1>
</header>
<hr/>

<p>
A collective <tt>model.reconstruct(white, parameters, workspace, velocity_factor)</tt> returns owning local arrays for initial density, unbiased real-space matter density, three displacements, three velocities and observed tracer modes. Matter density is an Eulerian mesh field; displacement and velocity are samples at Lagrangian positions. Displacements are in box units. Supply $aHfL$ as the velocity factor to obtain compatible peculiar-velocity units; 2LPT uses $\Psi_1+2\Psi_2$ for velocity. Physical outputs exclude tracer bias, RSD, noise and the observation response. No full-field gather is performed.
</p><div style="text-align: center"><div style="max-width:800px;display:inline-block;text-align:left;width:100%"><pre class="prettyprint">make -C FML/FieldLevel/example USE_MPI=true framework_mpi
mpirun -np 4 FML/FieldLevel/example/framework_mpi 2lpt

make -C FML/FieldLevel/example USE_MPI=true checkpoint_mpi
mpirun -np 4 FML/FieldLevel/example/checkpoint_mpi /shared/toy-chain start 60
mpirun -np 4 FML/FieldLevel/example/checkpoint_mpi /shared/toy-chain resume 90</pre></div></div>
<p><tt>Checkpoint.h</tt> adds <tt>sample_checkpointed</tt>, preserving the state, RNG caches, gradients and adaptation at complete trajectory boundaries. Resume with the same rank/slab layout, build, thread configuration, data and full warmup/sample schedule; a versioned <tt>problem_id</tt> is required. Each call returns only new sample rows. Use one writer job per shared checkpoint directory; earlier sample output is separate from the checkpoint. Rank-changing restore and schedule extension are not supported. See <a href="https://github.com/HAWinther/FML/blob/master/FML/FieldLevel/docs/FRAMEWORK.md#checkpointrestart-stage-4e-13-november-2026" style="display: inline">restart contracts and output semantics</a>.</p>

<header class="entry-header text-center">
  <h1 class="entry-title" id="PM">Particle-mesh forward model, adjoint and posterior</h1>
</header>
<hr/>

<p>
Beyond LPT, the framework has a periodic <b>particle-mesh (PM) forward model</b>: <tt>ParticleMeshForwardT&lt;Grid&gt;</tt> (<tt>FieldLevel/ParticleMesh.h</tt>) evolves the whitened initial field with a PM N-body integration in a fixed flat matter + $\Lambda$ background over a fixed schedule of scale factors, on slabs or pencils, and returns the final matter density and particle fields. <tt>ParticleMeshAdjointT</tt> (<tt>ParticleMeshAdjoint.h</tt>) is its exact discrete adjoint: the gradient of any objective of the outputs with respect to the white field and the spectrum parameters, with bounded memory through checkpointing and recomputation of the time steps.
</p>
<p>
<tt>PeriodicPMPosteriorT</tt> (<tt>ParticleMeshPosterior.h</tt>) connects them to the same Gaussian Fourier likelihood, priors, MAP optimizer and HMC sampler as the LPT models: the data are the Fourier modes of the final matter density on the force mesh below a cutoff, with a homogeneous Gaussian noise (whose amplitude can be fitted jointly). <tt>ParticleMeshCheckpoint.h</tt> adds checkpoint/restart of the HMC chains.
</p>
<div style="text-align: center"><div style="max-width:800px;display:inline-block;text-align:left;width:100%"><pre class="prettyprint">#include &lt;FML/FieldLevel/ParticleMeshPosterior.h&gt;
namespace FF = FML::FIELDLEVEL::FRAMEWORK;
// c: the IC mesh and the force/observation mesh (density_mesh) and the cutoffs
FF::PeriodicPMModel model(c, spectrum, pm_settings, noise_parameter, {4});
auto work = model.workspace();
auto p = model.initial_parameters();
auto data = model.mock(model.white_noise(91), p, 92, work);
FF::PeriodicPMPosterior posterior(model, data);
auto fit = FF::map(posterior, initial_white, p, false, 200, 2e-5);
model.predict(fit.white, fit.parameters, work);
const auto &amp; fields = model.reconstruction(work);</pre></div></div>
<p>
The adjoint is checked with directional finite differences, and HMC against quadrature of a two-mode posterior and with full-field and joint-parameter chains on small mocks. The PM model is a prototype: the predicted density is that of the PM integration itself, and a resolution study shows that at the tested (small) meshes it is not yet converged in the force mesh and the number of particles, so it should not be read as the continuum density without such a study. There is no tracer bias, redshift-space distortions or survey geometry yet. See the <a href="https://github.com/HAWinther/FML/blob/master/FML/FieldLevel/docs/PARTICLE_MESH.md" style="display: inline">PM model</a>, <a href="https://github.com/HAWinther/FML/blob/master/FML/FieldLevel/docs/PM_POSTERIOR.md" style="display: inline">posterior</a> and <a href="https://github.com/HAWinther/FML/blob/master/FML/FieldLevel/docs/KNOWN_ISSUES.md" style="display: inline">known-issues</a> notes.
</p>

<header class="entry-header text-center">
  <h1 class="entry-title" id="References">Method references and scope</h1>
</header>
<hr/>

<p>
Scientific reference directions are <a href="https://arxiv.org/abs/1203.3639" style="display: inline">Jasche &amp; Wandelt (2013), BORG</a>, <a href="https://arxiv.org/abs/2409.10937" style="display: inline">Stadler, Schmidt &amp; Reinecke, LEFTfield</a>, and the <a href="https://arxiv.org/abs/2407.01524" style="display: inline">field-level BAO application</a>. FML is an independent periodic-box implementation, not a port of these codes. In addition to the BAO prototype described above, the <a href="https://github.com/HAWinther/FML/blob/master/FML/FieldLevel/docs/FRAMEWORK.md" style="display: inline">framework</a> supports distributed linear, 1LPT and 2LPT models, joint field/bias/noise inference, HMC and MAP, reconstruction outputs, and same-rank checkpoint/restart. Exact linear, adjoint and MPI comparisons test implementation consistency; nonlinear scientific validation and realistic survey extensions remain open.
</p>
