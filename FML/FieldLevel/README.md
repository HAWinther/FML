# Field-level BAO inference (prototype)

`FieldLevelBAO.h` is a small, self-contained prototype of field-level inference of the BAO scale in the spirit of
[arXiv:2407.01524](https://arxiv.org/abs/2407.01524): instead of compressing the data into a power spectrum, the
posterior of the initial density field and the BAO scale is sampled jointly, mode by mode, with Hamiltonian Monte
Carlo (HMC). It is a prototype: serial (OpenMP in the particle loops), a toy wiggle template, linear bias, and
Gaussian noise of constant power. It is meant to show what the full problem needs (forward model, adjoint gradient,
sampler) and to be checked exactly where an exact answer exists.

## Model

- Parameters: the white-noise initial field s (one unit Gaussian per cell of an `nmesh`^3 grid) and
  beta = r_BAO / r_fid with a uniform prior on [`beta_min`, `beta_max`].
- Linear field: delta_k = T(k, beta) s_k with T = sqrt(P(k, beta) N^3 / V) for 0 < |k| < `kmax` and 0 above.
  P(k, beta) = P_smooth(k) [1 + A sin(k beta r_fid + phase) exp(-k / k_damp)] (`BAOSpectrum`); the default
  P_smooth is the Eisenstein & Hu no-wiggle spectrum normalised at k = 0.1 h/Mpc.
- Forward models (`Settings::model`):
  - `Linear`: pred_k = b delta_k.
  - `LPT1`: Zel'dovich displacement Psi_k = i k / k^2 delta_k, one particle per cell moved to x = q + Psi, CIC on an
    `nmesh_density`^3 grid (default 2 `nmesh`), pred_k = b delta_CIC,k.
- Likelihood: Gaussian per mode below `kmax` with noise power P_eps:
  -ln L = 1/2 sum_{all k, 0<|k|<kmax} |d_k - pred_k|^2 / (P_eps / V), the sum over the full Hermitian set of modes
  (the exact likelihood of a real field; the half-complex layout weights modes by 1 or 2).
- FFT conventions are those of `FFTWGrid` (r2c divides by N^3), so <|delta_k|^2> = P / V.

## Gradients

The gradient of U = -ln posterior is computed with hand-written adjoints at about twice the cost of a forward model:

1. Gbar_k = -(d_k - pred_k) / sigma^2 on the modes.
2. Linear: Q = b Gbar / N^3. 1LPT: delta-bar = c2r(b Gbar) / Nd^3 on the CIC grid; per particle
   x-bar = (Nd^3 / Np) sum_y delta-bar_y grad W(x - y) (the gradient of the CIC kernel); Psi-bar = x-bar;
   Q = sum_c (-i k_c / k^2) r2c(Psi-bar_c).
3. grad_s = c2r(T Q) + s and dU/dbeta = N^3 sum_full Re(conj(Q) dT/dbeta s_k).

Every gradient is checked against central finite differences (tests, agreement to ~1e-9 relative).

## Sampler

`sample(model, x0, beta0, options)` samples (x, beta) jointly with the generic HMC of
[`MCMC/HMC.h`](../MCMC/README.md#hmc) (dual-averaging step size, jittered leapfrog, windowed mass adaptation):

- Mass: Fourier-diagonal for the field, M_k = 1 + b^2 P / P_eps (the exact Hessian of the linear model), applied
  with FFTs as the operator block of `HMCMass`, and a diagonal mass for beta, adapted during warmup. Beta reflects at
  the prior edges. A target acceptance of ~0.65 mixes better here than the default 0.8.
- Parametrisation: with s sampled directly, beta and s are strongly correlated where the data dominate (changing beta
  at fixed s changes the predicted field). The sampled field is therefore partially centred,
  s_k = (T(beta_ref) / T(beta))^c x_k with c = S / (S + N) = b^2 P / (b^2 P + P_eps) per mode (`Settings::centering`;
  0 = white, 1 = fully centred). The prior and the Jacobian are included exactly. In the linear test this lowers the
  autocorrelation time of beta from ~13 to ~3 iterations.

## Validation

`tests/test_field_level.cpp` (`make -C FieldLevel/tests run`, about a minute):

- Adjoint gradients in x and beta against finite differences for both models and the three parametrisations.
- Mock data: sum |d|^2 / C over 20 mocks equals the number of modes (C = (b^2 P + P_eps) / V).
- Linear model: two independent HMC runs check rank-normalized split-Rhat and bulk/tail ESS through `MCMC/Diagnostics.h`. These are regression checks on beta, not production convergence certification. For the linear model the marginal posterior of beta is known exactly,
  ln L(beta) = -1/2 sum_full (|d_k|^2 / C_k + ln C_k). HMC agrees with it in the mean and the width
  (24^3, kmax = 0.07: exact 1.0469 +- 0.0669, HMC 1.0521 +- 0.0664, ESS ~190).
- 1LPT: beta is recovered from mock data made with the same model.

The example (`example/`, `./fieldlevel [linear|1lpt] nmesh kmax n_samples beta_true out`) at larger kmax
(the 1LPT run takes about 25 minutes on 2 cores):

| Model | Grid | kmax [h/Mpc] | beta_true | Exact marginal | HMC |
| --- | --- | --- | --- | --- | --- |
| Linear | 32^3, L = 1000 Mpc/h | 0.095 | 1.02 | 1.0232 +- 0.0333 | 1.0221 +- 0.0330 (1000 samples, ESS ~430) |
| 1LPT | 48^3 (CIC 96^3) | 0.14 | 1.02 | - | 1.049 +- 0.018 (300 samples, ESS ~40) |

## What a production version needs

- A better forward model (2LPT or a short PM run, EFT bias expansion with b_1, b_2, b_G2, ..., redshift-space
  distortions) and its adjoint; the structure here (forward model, Gbar, adjoint chain to Q) carries over directly.
- MPI slabs: the FFTs are already FFTWGrid; the CIC and its adjoint need the usual particle exchange.
- Sampling the bias and noise parameters as well (slice or HMC blocks), and a realistic noise model.
- Longer runs over several dispersed chains, using the shared rank-normalized split-Rhat and bulk/tail ESS diagnostics on beta and relevant field summaries, plus divergence checks and repeated-mock coverage.
