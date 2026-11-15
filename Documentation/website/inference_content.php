<hr/>
<header class="entry-header text-center">
  <h1 class="entry-title">Table of contents</h1>
</header>
<hr/>

<p class="page-toc">
<a href="index.php">Back to the main page</a><br>
<a href="documentation.php">Back to the documentation</a><br>
<a href="#Overview">1. Overview: Bayesian inference</a><br>
<a href="#Metropolis">2. Adaptive Metropolis&ndash;Hastings (FML::MCMC::Sampler)</a><br>
<a href="#Minimizer">3. Best fits, profile likelihoods and Hessians (FML::MCMC::Minimizer)</a><br>
<a href="#HMC">4. Hamiltonian Monte Carlo (FML::MCMC::run_hmc)</a><br>
<a href="#Diagnostics">5. Convergence diagnostics</a><br>
<a href="#Fitting">6. Cosmological parameter fits (CosmologyFitting)</a><br>
<a href="#Emulator">7. The EuclidEmulator2</a><br>
<a href="#References">Method references and scope</a><br>
<a href="#NeuralNetwork">Trainable neural-network emulators</a><br>
</p>

<header class="entry-header text-center">
  <h1 class="entry-title" id="Overview">Overview: Bayesian inference</h1>
</header>
<hr/>

<p>
Given data $d$ and a model with parameters $\theta$, the posterior is $P(\theta|d)\propto\mathcal{L}(d|\theta)\pi(\theta)$. <tt>FML/MCMC</tt> (namespace FML::MCMC, nothing in it knows about cosmology) samples it with adaptive Metropolis&ndash;Hastings for a handful of expensive parameters, or with Hamiltonian Monte Carlo when the gradient is available and the number of parameters is large (up to millions, as in <a href="fieldlevel.php" style="display: inline">field-level inference</a>). It also finds best fits, profile likelihoods and Hessian covariances, and computes modern convergence diagnostics. <tt>FML/CosmologyFitting</tt> uses it with the <a href="cosmology.php" style="display: inline">Einstein&ndash;Boltzmann solver</a> as the theory to fit cosmological parameters to supernova, BAO and CMB data.
</p>

<header class="entry-header text-center">
  <h1 class="entry-title" id="Metropolis">Adaptive Metropolis&ndash;Hastings</h1>
</header>
<hr/>

<p>
From the current point $\theta$ a proposal $\theta' = \theta + sL\vec z$ ($\vec z$ standard normal, $LL^T$ the proposal covariance, $s$ a scale) is accepted with probability $\min[1, P(\theta'|d)/P(\theta|d)]$. During the burn-in the covariance is adapted to the sample covariance of the chain (Haario et al. 2001) and the scale towards 25% acceptance; at the end of the burn-in the covariance pooled from all chains is frozen, so the sampling phase is an exact Markov chain. Several chains run in parallel (OpenMP threads and MPI processes) and the run stops when every chain has enough samples and the Gelman&ndash;Rubin statistic satisfies $R - 1 &lt; \epsilon$,
$$\hat R = \sqrt{\frac{\frac{n-1}{n}W + \frac{1}{n}B}{W}},$$
with $W$ the mean within-chain and $B/n$ the between-chain variance. Fast/slow parameters (Lewis 2013): parameters that are cheap to vary (e.g. likelihood nuisance parameters, with the theory cached) are updated <tt>oversample_fast</tt> times per step along their conditional directions. The full state of each chain is checkpointed, and a resumed run continues exactly as an uninterrupted one. Chains are written in GetDist format.
</p>
<div style="text-align: center">
  <div style="width:700px;display: inline-block;text-align: left">
    <pre class="prettyprint">
#include &lt;FML/MCMC/Sampler.h&gt;
using namespace FML::MCMC;
LikelihoodFunction like = [](const std::vector&lt;double&gt; &amp; x) {
  LikelihoodResult r;
  r.loglike = -0.5 * (x[0] * x[0] + (x[1] - 1.0) * (x[1] - 1.0) / 4.0);
  r.derived = {x[0] + x[1]};
  return r;
};
SampledParameter a{"a", "a"}, b{"b", "b"};
a.start = 0.0; a.min = -10; a.max = 10; a.proposal = 1.0;
b.start = 0.0; b.min = -10; b.max = 10; b.proposal = 2.0;
Posterior post({a, b}, {"a_plus_b"}, {"a+b"}, like);
SamplerOptions opt;
opt.output_root = "chains/toy";      // GetDist chains, .paramnames, .ranges, .covmat, .summary.txt
Sampler mc(post, opt);
mc.run();                            // mc.mean(), mc.stddev(), mc.covariance(), mc.best_fit(), mc.Rminus1()</pre>
  </div>
</div>

<header class="entry-header text-center">
  <h1 class="entry-title" id="Minimizer">Best fits, profile likelihoods and Hessians</h1>
</header>
<hr/>

<p>
The minimizer maximizes the posterior with an adaptive Nelder&ndash;Mead simplex (Gao &amp; Han 2012) in coordinates whitened by a covariance, so the first simplex has the size of the posterior in every direction; it needs no derivatives and tolerates the small numerical noise of a Boltzmann code. It restarts around the best point until a restart gains less than the tolerance, runs several starts in parallel (their spread shows whether the minimum is global), and supports fast/slow parameters. A profile likelihood fixes one parameter on a grid and minimizes over the rest,
$$-2\ln\mathcal{L}_{\rm prof}(\theta_i) = \min_{\theta_{j\ne i}}\left[-2\ln\mathcal{L}(\theta)\right],$$
which, unlike the marginalized posterior, is free of prior-volume effects. The Hessian of $-\ln P$ by central differences (with steps adapted to 1.5 conditional $\sigma$) gives a covariance, e.g. as the proposal of a following MCMC.
</p>

<header class="entry-header text-center">
  <h1 class="entry-title" id="HMC">Hamiltonian Monte Carlo</h1>
</header>
<hr/>

<p>
HMC (Duane et al. 1987, Neal 2011) introduces momenta $\vec p$ with a mass matrix $M$ and the Hamiltonian
$$H(\vec x,\vec p) = U(\vec x) + \frac{1}{2}\vec p^TM^{-1}\vec p,\qquad U = -\ln P(\vec x|d),$$
draws $\vec p\sim\mathcal{N}(0,M)$, integrates Hamilton's equations with the leapfrog scheme
$$\vec p_{1/2} = \vec p - \frac{\epsilon}{2}\nabla U(\vec x),\qquad \vec x' = \vec x + \epsilon M^{-1}\vec p_{1/2},\qquad \vec p' = \vec p_{1/2} - \frac{\epsilon}{2}\nabla U(\vec x'),$$
and accepts the end point with probability $\min[1,e^{H_0 - H_1}]$. Because the trajectories follow the posterior, the acceptance stays high in very many dimensions. <tt>HMC.h</tt> (header-only) splits the state into a large block whose mass is given as operators (e.g. a Fourier-diagonal filter for a field, nothing of it stored per step) and a set of parameters with a diagonal mass, adapted during warmup in Stan's windows. The step size is tuned by dual averaging (Hoffman &amp; Gelman 2014), the number of leapfrog steps is jittered, bounded parameters reflect at the bounds, and divergent trajectories are counted.
</p>
<div style="text-align: center">
  <div style="width:700px;display: inline-block;text-align: left">
    <pre class="prettyprint">
#include &lt;FML/MCMC/HMC.h&gt;
using namespace FML::MCMC;
HMCPotential U = [](const std::vector&lt;double&gt; &amp; x, std::vector&lt;double&gt; &amp; grad) {
  grad = {x[0], 4 * x[1]};
  return 0.5 * x[0] * x[0] + 2 * x[1] * x[1];
};
HMCOptions opt;                  // warmup, samples, n_leapfrog, target_acceptance, seed
opt.lower = {-10, 0};            // optional bounds (reflection)
opt.upper = {10, 5};
HMCResult r = run_hmc(U, {0.1, 0.1}, HMCMass{}, opt);
auto x0 = recorded_column(r, 0); // post-warmup samples of parameter 0</pre>
  </div>
</div>

<p>The generic HMC engine now exposes <tt>HMCState</tt> and <tt>HMCRunControl</tt> for bounded runs and exact continuation at trajectory boundaries, with a versioned state codec in <tt>HMCCheckpoint.h</tt>. Applications supply their target identity and storage policy. The <a href="fieldlevel.php#Outputs" style="display: inline">field-level wrapper</a> adds collective per-rank checkpoint files and restart validation; see the <a href="https://github.com/HAWinther/FML/blob/master/FML/MCMC/README.md#hmc-checkpoint-state" style="display: inline">HMC state contract</a>.</p>

<header class="entry-header text-center">
  <h1 class="entry-title" id="Diagnostics">Convergence diagnostics</h1>
</header>
<hr/>

<p>
<tt>Diagnostics.h</tt> implements the rank-normalized split-$\hat R$ (and its folded version) and the bulk and tail effective sample sizes of Vehtari et al. (2021), as in Stan and ArviZ: the draws are replaced by normal scores of their ranks, each chain is split in two, and the ESS uses multi-chain autocovariances (by FFT) truncated with Geyer's initial positive and monotone sequence,
$$S_{\rm eff} = \frac{NM}{\hat\tau},\qquad \hat\tau = -1 + 2\sum_{t'=0}^{K}\hat P_{t'},\qquad \hat P_{t'} = \hat\rho_{2t'} + \hat\rho_{2t'+1}.$$
The tail ESS uses the indicators of the 5% and 95% quantiles. Degenerate input (too few chains or draws, a frozen chain, non-finite values) is reported with an explicit status instead of a misleading number. The numbers are checked against fixtures generated with ArviZ.
</p>
<div style="text-align: center">
  <div style="width:700px;display: inline-block;text-align: left">
    <pre class="prettyprint">
#include &lt;FML/MCMC/Diagnostics.h&gt;
const auto d = FML::MCMC::diagnose_chains(chains);          // chains[chain][draw]
if (d.valid()) std::cout &lt;&lt; d.rhat &lt;&lt; " " &lt;&lt; d.ess_bulk &lt;&lt; " " &lt;&lt; d.ess_tail &lt;&lt; "\n";
const auto hmc = FML::MCMC::diagnose_hmc_chains(runs, 0);    // several HMCResult runs, column 0</pre>
  </div>
</div>

<header class="entry-header text-center">
  <h1 class="entry-title" id="Fitting">Cosmological parameter fits</h1>
</header>
<hr/>

<p>
<tt>./mcmc</tt> in <tt>FML/CosmologyFitting</tt> samples (or minimizes, profiles, or computes the Hessian of) the posterior of any parameter of the Einstein&ndash;Boltzmann solver, including the parameters of the models beyond $\Lambda$CDM, from a parameter file in the same format as <tt>./cmb</tt>. The theory is computed only as far as the likelihoods need (supernovae need only the background, BAO also the recombination history). Available likelihoods:
</p>
<ul>
<li>Supernovae: Pantheon+, DES-SN5YR and Union3 (absolute magnitude marginalized analytically), and binned JLA.</li>
<li>BAO: DESI DR1 and DR2 with their full covariance; BBN prior on $\omega_b$.</li>
<li>CMB: Planck 2018 distance priors, Planck 2018 plik_lite TT/TE/EE (+ low-$\ell$ TT), Planck 2018 lensing (also CMB-marginalized), and an exact Gaussian mock CMB likelihood for forecasts.</li>
</ul>
<div style="text-align: center">
  <div style="width:700px;display: inline-block;text-align: left">
    <pre class="prettyprint">
cd FML/CosmologyFitting
make
python3 python/fetch_sn_data.py               # Pantheon+ and DES-SN5YR data (once)
./mcmc params/mcmc_lcdm.ini                     # flat LCDM: DESI DR2 + Planck distance priors + Pantheon+
./mcmc params/mcmc_w0wa.ini                     # w0-wa dark energy
./mcmc params/mcmc_w0wa.ini sampler=minimize minimize_profile=w0,-1.05,-0.65,9 minimize_hessian=true
./mcmc params/mcmc_planck_lite_lcdm.ini         # Planck 2018 plik_lite (use precision = fast)
python3 python/plot_chains.py chains/lcdm_desi_planckdp_pantheonplus

# params/mcmc_w0wa.ini: any ./cmb parameter fixes a value; sample_* = start, min, max, step
likelihoods = desi_dr2, planck_dp, pantheonplus
model = w0wa
m_nu = 0.06
sample_H0        = 68.0,  50.0, 90.0, 0.5
sample_omega_b   = 0.0224, 0.018, 0.027, 0.0002
sample_omega_cdm = 0.119, 0.09, 0.15, 0.001
sample_w0        = -0.9, -2.0, 0.0, 0.05
sample_wa        = -0.3, -3.0, 2.0, 0.2
mcmc_chains = 4</pre>
  </div>
</div>
<p>
Results: DESI DR2 + BBN gives $\Omega_m = 0.2975\pm 0.0085$ and $H_0 = 68.53\pm 0.60$ as in the DESI DR2 analysis; DESI DR2 + Planck distance priors + Pantheon+ gives $w_0 = -0.848\pm 0.054$, $w_a = -0.55\pm 0.22$ with $\Delta\chi^2 = -7.6$ relative to $\Lambda$CDM (close to the DESI DR2 result with the full CMB likelihoods). The plik_lite likelihood reproduces planck-lite-py to $10^{-13}$ in $\ln\mathcal{L}$ and the lensing likelihood Cobaya to $10^{-14}$; the Planck plik_lite + lensing best fit agrees with Planck 2018 ($\omega_b = 0.022397$, $\omega_c = 0.11982$, $n_s = 0.9670$, $H_0 = 67.49$, $\sigma_8 = 0.8129$).
</p>
<p style="text-align: center">
<img src="images/fitting_w0wa.png" alt="w0-wa constraints" style="max-width: 450px; width: 100%;"><br/>
<i>$w_0$&ndash;$w_a$ dark energy from DESI DR2 BAO, Planck distance priors and Pantheon+ supernovae.</i>
</p>

<header class="entry-header text-center">
  <h1 class="entry-title" id="Emulator">The EuclidEmulator2</h1>
</header>
<hr/>

<p>
<tt>FML/Emulator</tt> contains the minimal code to evaluate the EuclidEmulator2 (Knabenhans et al. 2021, GPL), which predicts the non-linear boost $B(k,z) = P_{\rm NL}/P_{\rm L}$ for $w_0w_a$CDM cosmologies with massive neutrinos to about 1% for $k\le 10\,h$/Mpc from a principal-component and polynomial-chaos expansion trained on N-body simulations.
</p>

<header class="entry-header text-center">
  <h1 class="entry-title" id="NeuralNetwork">Trainable neural-network emulators</h1>
</header>
<hr/>

<p>
<tt>Emulator/NeuralNetwork</tt> provides a native header-only multilayer perceptron for training and evaluation, with optional OpenMP batch evaluation. It supports several activations, MSE/L1/Huber losses, standardized inputs and outputs, Adam with L2 weight decay, gradient clipping, validation-loss scheduling, early stopping and stochastic weight averaging. The model exposes input Jacobians and saves to a portable text format. Accuracy depends on the training domain and held-out data.
</p><div style="text-align: center"><div style="max-width:800px;display:inline-block;text-align:left;width:100%"><pre class="prettyprint">#include &lt;FML/Emulator/NeuralNetwork/NeuralNetwork.h&gt;
using namespace FML::EMULATOR::NEURALNETWORK;
TrainingOptions opt;
opt.hidden_layers = {64, 64};
opt.activation = Activation::gelu;
TrainingHistory history;
auto net = train(X_train, Y_train, X_val, Y_val, opt, &amp;history);
auto prediction = net(x);
auto jacobian = net.jacobian(x);
net.save(&quot;emulator.txt&quot;);
auto restored = NeuralNetwork::load(&quot;emulator.txt&quot;);</pre></div></div>
<p>Networks trained with Sesame's PyTorch Lightning emulator can be exported with <tt>export_lightning_emulator.py</tt> and loaded in C++; export needs PyTorch, evaluation does not. See <a href="https://github.com/HAWinther/FML/blob/master/FML/Emulator/NeuralNetwork/README.md" style="display: inline">the training and export guide</a>. The <a href="random.php#LatinHypercube" style="display: inline">Latin hypercube sampler</a> supplies training designs. This trainable module is separate from the fixed EuclidEmulator2 cosmology emulator.</p>

<header class="entry-header text-center">
  <h1 class="entry-title" id="References">Method references and scope</h1>
</header>
<hr/>

<p>
The native neural-network emulator (<tt>Emulator/NeuralNetwork</tt>) follows the <a href="https://github.com/renmau/Sesame_pipeline/blob/main/Pipeline/neural_network_emulator/Emulator.py" style="display: inline">Sesame pipeline training recipe</a>, using <a href="https://arxiv.org/abs/1412.6980" style="display: inline">Kingma &amp; Ba, Adam</a>, <a href="https://arxiv.org/abs/1803.05407" style="display: inline">Izmailov et al., stochastic weight averaging</a>, and <a href="https://proceedings.mlr.press/v9/glorot10a.html" style="display: inline">Glorot &amp; Bengio (2010) initialization</a>. The C++ implementation is independent; PyTorch supplies comparison fixtures and exportable checkpoints. Input/output standardization and initialization differ from Sesame defaults, as documented in the <a href="https://github.com/HAWinther/FML/blob/master/FML/Emulator/NeuralNetwork/README.md" style="display: inline">module README</a>.
</p>
