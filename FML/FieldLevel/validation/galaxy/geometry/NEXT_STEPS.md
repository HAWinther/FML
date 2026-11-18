# Proposed next test: likelihood-tempered replica exchange

Not implemented or sampled in this audit. The nine selected paths retain large
profiled barriers at full likelihood; this motivates a separate connectivity test.
It does not prove all routes have a barrier or predict replica exchange rates.

Use the family

```
pi_beta(white, parameters) proportional to exp[-U_prior - beta * L]
```

where L is the original fixed-noise negative log likelihood (or its quadratic
part, since noise is fixed and shared constants cancel). Keep all physical priors
unchanged. beta=1 is exactly the desired posterior. At beta=0 this fixture has a
proper product of standard Gaussian white coordinates and bounded uniform scalars;
refresh that replica directly from the prior, then evaluate its original L for swaps.
Do not divide the white prior by beta or pass infinite noise into a finite-noise API.

Existing field/scalar kernels can target each positive beta with likelihood terms
scaled consistently. For the present fixed-noise Gaussian likelihood, variance
P_noise/beta has the right state-dependent quadratic; its normalization differs
from beta*L only by a replica-specific state-independent constant. Gaussian noise
parameters or other likelihood families would need a separate derivation. Scalar
cache and field metric must use the same beta; caches must rebuild after swaps.

For adjacent replicas, exchange complete field/scalar states using

```
log_accept_swap = (beta_i - beta_j) * (L_i - L_j)
```

Shared priors cancel. Refresh HMC momenta per transition; do not accidentally swap
persistent adaptation or metric snapshots with the physical states. Keep each
replica's metric fixed at its declared reference. Alternate disjoint even/odd
neighbor swaps and save the whole ensemble plus schedule/counters. All component
seeds, temperature values and the ensemble state belong in the checkpoint identity.
Only beta=1 draws enter posterior diagnostics. Hot states and replay draws cannot
be pooled with them.

Suggested stages:

1. Test swap-ratio sign and joint invariant distribution against a small analytic
   Gaussian reference; verify beta=0 prior refreshes and positive-beta kernel/cache
   consistency. Test full-ensemble exact fresh-process replay and target rejection.
2. Freeze a short connectivity pilot with an initial ladder such as
   beta=[1, 1/2, 1/4, 1/8, 1/16, 0]. Measure adjacent exchange acceptance, movement
   of labeled states through the ladder and round trips; this is not a posterior
   convergence/coverage claim. Path-barrier reduction alone cannot certify overlap,
   especially next to beta=0. Add ladder resolution only in a separately recorded
   tuning stage, never silently in retained production samples.
3. Once connectivity is measured, freeze a separate four-cold-chain campaign and
   compare all existing physical scalar/field monitors at unchanged gates, with
   the full replica cost and exact checkpoint/restart. Keep all failed campaigns.

This is a computationally larger algorithm and a proposed next experiment, not
an achieved scientific validation milestone. A future production implementation
needs collective replica identity/state validation and a clear MPI process layout.
