# Conditional HMC stability diagnostic

This serial validation driver probes the completed eight-dataset prior-predictive
pilot without changing its records, priors, data or checkpoints. It is exploratory
sampler diagnosis after examining failed outcomes, not a new calibration campaign.

The frozen plan uses three predetermined cold draws (0, 1024, 2047) from all four
chains of every dataset: 96 states. Five variants each receive eight paired
standard-normal momentum draws per state, giving 3840 primary trajectories.

| Case | Step | Leapfrog steps | Field mass parameters |
|---|---:|---:|---|
| baseline | 0.15 | 8 | Original ensemble starting scalars |
| half_step | 0.075 | 16 | Original ensemble starting scalars |
| quarter_step | 0.0375 | 32 | Original ensemble starting scalars |
| conditional_metric | 0.15 | 8 | Current stored scalars |
| conditional_half | 0.075 | 16 | Current stored scalars |

All variants have integration time epsilon*L=1.2. Changing the mass changes the
physical trajectory even at fixed integration time; it is not a matched path-
length experiment. Costs differ and are reported as actual target-gradient calls.
Seeds use `0x56 | mock/ensemble/draw/trial` domains separate from original sampling,
generation and rank analysis. Standardized random momenta and accept/reject RNG
seeds are paired across variants; transformed momenta differ when the mass differs.

The current-scalar metric remains constant during each trajectory. The scalar
parameters are held fixed throughout the conditional field update, and the mass
does not depend on the field being proposed. This would be a valid conditional
HMC kernel; it does not require an unimplemented field-dependent/Riemannian mass
correction. Its numerical usefulness still needs testing. The tested linearized
metric omits nonlinear density/tidal response, so current parameters alone need
not produce a better preconditioner.

## Reproduce outside the library

Build `stability_probe` and `stability_test` using the normal FML external build
configuration. Supply the completed calibration pilot and a fresh output folder.

```sh
make stability_probe stability_test
./stability_test
python3 run.py prepare /absolute/output --source /absolute/completed-pilot --exe /absolute/stability_probe
python3 run.py run /absolute/output --exe /absolute/stability_probe
python3 analyze.py /absolute/output --exe /absolute/stability_probe
```

Preparation copies compact text inputs and hashes the plan, source pilot records,
checkpoints, implementation and executable before evaluating comparison outcomes.
Run uses four processes, a fixed per-dataset timeout and exactly the declared
budget. Partial failed outputs remain for inspection and are not silently retried
or replaced. Complete per-dataset receipts allow skipping already completed work
with matching hashes. No original pilot file is written.

## Controls and interpretation

Probe.h observes the actual `run_hmc` potential/mass callbacks and reconstructs
initial/final Hamiltonian energies. It verifies energy error against the native
acceptance probability and divergence counter. Trial zero for every state/case is
also executed through the ordinary field sampler; all 480 comparisons must match
exactly. These duplicate validation executions are not pooled into probe averages.
A separate stiff-Gaussian control checks instability, smaller-step stabilization
and a correctly matched metric, with 96 exact native/probe matches.

Every saved state's predictions and quadratic must reproduce before probing.
Three field directions and all five scalar derivatives are checked by support-
aware finite differences. Frames near scalar bounds use restricted step sizes;
these are local derivative checks, not a claim about all posterior derivatives.

Reports retain acceptance probabilities, finite/nonfinite energy errors,
divergences, accepted white-field RMS jumps and gradient counts for every case.
The full-white jump includes prior-only directions and is not an active-field
mixing diagnostic or a substitute for ESS per second. Input states are correlated
and include unconverged chains; no population MCSE, posterior samples, coverage
claim or chain-convergence improvement is inferred from these probes.

See [results](../../../docs/validation/studies/stability_probe/RESULTS.md). The
completed pilot is preserved separately; production sampler defaults are unchanged.
