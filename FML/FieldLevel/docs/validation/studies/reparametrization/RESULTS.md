# Reparametrization and analytic bias marginalization on the joint-pilot target

Recorded 22 November 2026. **All twelve dataset/method combinations FAIL the pilot's
mixing gates.** [Machine-readable summary](summary.json); [driver](../../../../validation/galaxy/reparametrization/README.md).

## Question

The free-IC joint pilot failed every gate. Two remedies aimed at the b1-sigma8
degeneracy were tested: sampling (b_delta * sigma8 ratio, sigma8 ratio) instead of
(b_delta, sigma8 ratio), and integrating the four bias coefficients out analytically
(`MarginalizedBias.h`, exact for unbounded Gaussian bias priors).

## Target and budget

The joint pilot's target: N4 white field (64 coordinates), N8 product and N16 PSWF
density meshes, L=100, P_ref=20000 exp(-5k), noise power 1000, initial/matter/
likelihood cutoffs 1.7 (sphere) / 1.8 / 1.6 kf (cube), 26 observed modes. The
sigma8-ratio prior stays uniform on [0.5,1.5]. Because exact marginalization needs
Gaussian bias priors, **all three methods** use Gaussian priors with the means and
standard deviations of the pilot's uniform priors: b_delta N(2, 1.155), b_delta2 and
b_K2 N(0, 2.309), b_laplacian N(0, 57.7). All three therefore sample the same
posterior of (field, sigma8 ratio); the marginal runs add one exact conditional bias
draw per retained sample. This is not the uniform-prior target of the original pilot.

Datasets: 0 is the pilot's mock (truth and seeds); 1-3 have truths drawn from these
priors. Each dataset/method has four chains with the pilot's scalar starts and field
seeds, 256 warmup and 1024 retained draws, up to 12 jittered leapfrog steps,
target acceptance 0.9, step-size and scalar-mass adaptation, and the Eulerian field
metric frozen at the start (at the conditional bias mean for the marginal runs).
Gates: Rhat < 1.01 and bulk/tail ESS >= 400 for sigma8 ratio, b1*sigma8, the four
biases and the pilot's six field projections; zero divergences.

| Dataset | Generating sigma8 ratio, b_delta, b_delta2, b_K2, b_laplacian |
|---|---|
| 0 | 1, 1.6, 0.3, -0.2, 7 |
| 1 | 0.626, 1.34, 6.04, -0.483, 27.9 |
| 2 | 0.509, 2.18, -0.101, -0.0669, -96 |
| 3 | 1.04, 1.78, 2.92, 2.36, 88.5 |

## Results

| Dataset | Method | max Rhat | min bulk ESS | sigma8 ratio Rhat / ESS | b1*sigma8 Rhat / ESS | Divergences | Gate |
|---|---|---:|---:|---:|---:|---:|---|
| 0 | joint | 3.04 | 4.5 | 1.38 / 9 | 2.01 / 5 | 0 | FAIL |
| 0 | reparam | 3.08 | 4.5 | 1.53 / 7 | 2.32 / 5 | 1 | FAIL |
| 0 | marginal | 2.61 | 4.8 | 1.05 / 114 | 1.83 / 6 | 0 | FAIL |
| 1 | joint | 2.96 | 4.6 | 1.78 / 6 | 2.63 / 5 | 0 | FAIL |
| 1 | reparam | 2.88 | 4.7 | 1.56 / 7 | 2.88 / 5 | 0 | FAIL |
| 1 | marginal | 2.47 | 4.9 | 1.88 / 6 | 2.44 / 5 | 0 | FAIL |
| 2 | joint | 2.41 | 4.9 | 1.29 / 11 | 1.81 / 6 | 0 | FAIL |
| 2 | reparam | 2.43 | 4.9 | 1.27 / 12 | 1.71 / 6 | 0 | FAIL |
| 2 | marginal | 1.53 | 7.2 | 1.11 / 26 | 1.53 / 7 | 0 | FAIL |
| 3 | joint | 2.88 | 4.6 | 1.38 / 9 | 1.85 / 6 | 0 | FAIL |
| 3 | reparam | 2.82 | 4.6 | 1.82 / 6 | 2.50 / 5 | 0 | FAIL |
| 3 | marginal | 2.90 | 4.6 | 1.42 / 8 | 1.46 / 8 | 0 | FAIL |

Every combination costs about 11,500 gradient evaluations per chain (18-20 s serial).
Only one divergence occurred (dataset 0, reparametrized).

### Why: separated modes, not a ridge

Each chain stays in one mode for its whole run ([traces](b1sigma8_traces.png)).
Modes differ in the sign of b_delta * sigma8 together with the field (the linear
prediction is invariant under s -> -s, b_delta -> -b_delta; only the quadratic terms
and the b_delta prior distinguish them), and also among same-sign field
configurations (dataset 3, marginal: all chains have b_delta*sigma8 > 0 but different
field projections). Mean potentials differ between chains by 9-15 units
(dataset 0: -38.4, -29.1, -29.0, -35.5; dataset 3: -15.5, -18.9, -13.6, -28.2), more
than the ~6 expected spread within one typical set of ~70 dimensions, so several
chains occupy lower-probability modes. Neither proposal can move chains between such
modes. The diagnosis behind both suggestions (a curved b1-sigma8 ridge) was wrong as
the main obstacle for this target.

### Within-mode mixing

Median single-chain ESS of 1024 draws (Geyer initial positive sequence):

| Dataset | Method | sigma8 ratio | b1*sigma8 | b_K2 | cos100 |
|---|---|---:|---:|---:|---:|
| 0 | joint | 21 | 18 | 24 | 11 |
| 0 | reparam | 13 | 11 | 9 | 12 |
| 0 | marginal | 45 | 191 | 157 | 58 |
| 1 | joint | 10 | 9 | 7 | 11 |
| 1 | reparam | 9 | 11 | 15 | 11 |
| 1 | marginal | 7 | 90 | 27 | 34 |
| 2 | joint | 32 | 31 | 26 | 31 |
| 2 | reparam | 37 | 41 | 34 | 34 |
| 2 | marginal | 324 | 505 | 679 | 523 |
| 3 | joint | 22 | 7 | 26 | 15 |
| 3 | reparam | 39 | 15 | 17 | 29 |
| 3 | marginal | 17 | 25 | 22 | 25 |

Marginalization improves within-mode mixing substantially on datasets 0 and 2
(b1*sigma8 ~10x, sigma8 ratio 2-10x at equal gradient cost) and little on 1 and 3.
The reparametrization shows no consistent gain.

## Conclusions and limits

- `BiasMarginalizedPosterior` is exact (tests: finite-difference gradients, Gaussian
  identities, slab/pencil MPI) and reduces the sampled scalars to the spectrum
  parameters. It is worth keeping for any free-IC sampler, but it does not by
  itself fix this target.
- The obstacle is multimodality of the field posterior with 26 data modes, 18 active
  real field directions and five scalars. Next steps should target transport between
  modes (tempering with the marginal target, which removes four scalars from every
  replica, or mode-jumping moves such as the s -> -s, b -> -b reflection), or a
  better-constrained target (more modes, a resolution closer to the real analysis)
  before more sampler work.
- Synthetic same-model data only; four datasets; fixed budget, no retuning or
  extension. Gaussian bias priors differ from the original pilot's uniform priors.
