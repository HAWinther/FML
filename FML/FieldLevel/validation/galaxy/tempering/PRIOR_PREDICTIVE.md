# Prior-predictive truth controls

The opt-in mode generates the five physical scalars from the model's exact
bounded-uniform priors, all 64 white coordinates from the standard Gaussian
prior, and an independent Gaussian noise realization. Model, physical cutoffs,
noise power, scalar priors and sampler kernels stay fixed. This is a serial
validation driver; no production MPI replica orchestration is added.

## Configuration

For `run.py`, start from a compatible layout-3 plan and set:

```json
{
  "seed_layout": 3,
  "truth_mode": "prior_predictive_v1",
  "truth_parameters": null,
  "mock_id": 101,
  "truth_seed": 7101,
  "noise_seed": 7102,
  "initial_seed_base": 7103,
  "scalar_truth_seed": 7107
}
```

This fragment is a control-smoke example, not a frozen calibration campaign. Use
new IDs/seeds for any independently declared campaign. IDs 1 and 2 are reserved
for the existing fixed-truth mocks. Seeds are positive 32-bit integers; all raw
truth/noise/scalar seeds and initial_seed_base through base+ensembles-1 must be
distinct. The runner rejects fixed truths in prior-predictive mode and changed
plan/binary identity on restart. Plans using the old fixed-truth convention and
`FML_MOCK_1` keep the same data, target/summary wire bytes and trajectories.

The native schedule file for the smoke is:

```text
0 64 4 3
FML_MOCK_2 101 7101 7102 7103 7107
```

The first line is warmup, samples, ensembles, sampler seed layout. The second
line is marker, mock ID, white-truth seed, noise seed, initial-field base seed,
and scalar-truth seed. The native CLI still takes DIRECTORY, start/resume,
ENSEMBLE, sweep budget, ladder file, exchange passes and settings file.

## RNG and prior identity

Generation keys are
`0x5400000000000000 | role<<48 | mock_id<<32 | raw_seed`, where roles 0,1,2,3
mean white truth, scalar truth, noise and initial field respectively. Initial
fields use raw_seed=initial_seed_base+ensemble. Sampler/exchange keys keep the
existing, separate `0x53` namespace. There is no truth-dependent initialization,
metric, seed replacement or adaptation.

`PriorPredictive.h` samples each bounded-uniform scalar with a distinct Philox
coordinate counter and purpose `0x53424353`. It deliberately does not reuse the
sampler's beta=0 draw implementation. Gaussian white and noise use the existing
model's white generator with their separate full-width keys. Unsupported scalar
priors fail explicitly rather than silently using a uniform surrogate.

The checkpoint identity binds mode/version, scalar seed, generated truth vector
and each exact scalar prior name/bounds, in addition to the existing dataset,
mock metadata, sampler schedule and replica layout. Changing any generation
seed or mode rejects resume. Fixed-truth identity has no new fields.

New-mode target.json contains truth_mode, scalar_truth_seed, scalar_priors and
full generation_keys (decimal strings to preserve 64-bit precision), along with
truth/noise white vectors, prediction at truth, integer observation modes and the
full weighted likelihood quadratic at truth. This audit payload appears only in
the new mode. Cold records already retain predictions and all replica quadratics,
so likelihood-discrepancy monitoring needs no change to their format.

## Independent checks

Build in an external directory with the usual FML configuration:

```sh
make tempering_pilot tempering_test tempering_mock_test tempering_prior_test
./tempering_test
./tempering_mock_test
./tempering_prior_test /absolute/output/prior-draws.csv
python3 check_prior.py /absolute/output/prior-draws.csv --targets /absolute/native-run/target.json
python3 check_sbc.py
```

The draw reference checks every scalar, white and noise value with an independent
Python Philox implementation, then checks uniform CDFs, cross-coordinate moments,
white/noise covariance and Fourier-noise power on 1024 complete draws. Its Philox
reference also checks the official zero-key/counter known-answer vector. Actual
native target checks reconstruct the observed Fourier noise using an independent
orthonormal FFT and verify the weighted truth likelihood quadratic.

The analytic normal-normal SBC control uses four independent exact posterior
endpoints per dataset and 4096 datasets. Parameter ranks alone also pass for a
deliberately data-ignoring prior sampler; a data-dependent likelihood discrepancy
correctly detects it. Ties randomize the truth's position among equal values, and
invalid shapes/nonfinite quantities reject. `sbc_ranks.py` requires independent
posterior endpoints from its caller; it does not decorrelate MCMC records.

[Control results](../../../docs/validation/studies/prior_predictive_controls/RESULTS.md)
preserve the actual native target, legacy comparison, source-checkpoint and
four-chain pause/restart checks. The nonlinear smoke has only 64 retained draws
per chain and cannot establish the proposed full-budget mixing gates or SBC.
Next: add the bounded campaign/recovery analysis with Q monitoring, freeze its
controls and budgets, then generate independent nonlinear prior-predictive mocks.
