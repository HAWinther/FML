# Bounded prior sensitivity

This offline driver reweights all completed cold records from the original target
and two independent fixed-truth mocks. No new forward calls or MCMC are needed.
The four cases and reliability screens are in plan.json; every outcome is retained.

Run independent checks:

```sh
python3 check.py
python3 ../refinement/check_weights.py
```

Keep outputs outside the library. Supply source directories in the declared order
(original, mock 1, mock 2); each contains joint.chain{0..3}.csv,
posterior-summary.json, completed.json, plan.json and ensemble*/CURRENT/target.json.
CURRENT and target.json are separate files inside each ensemble directory.

```sh
python3 analyze.py prepare /absolute/output --sources /original /mock1 /mock2
python3 analyze.py analyze /absolute/output
```

Preparation copies and hashes the plan, cold records, target, source diagnostics,
completion receipts and checkpoints. Analysis rejects changed inputs, source
files, implementation or shared weight math. Reanalysis requires the same frozen
files; an altered study needs a new output directory. An exclusive lock prevents
concurrent writers. No source file is written.

summary.json reports all eleven coordinates, mean/SD/median/central intervals,
paired mean shifts and MCSE, overlap fractions and every reliability screen.
prior-sensitivity.png shows amplitude densities and scalar shifts. Interval
precision is not certified by a mean-shift MCSE. Population SD and midpoint
weighted quantiles are used consistently for baseline and alternatives.

See [results](../../../docs/validation/studies/prior_sensitivity/RESULTS.md) and
[calibration follow-up](CALIBRATION_PLAN.md). These are importance-sensitivity
estimates, not alternative-prior MCMC convergence or coverage validation.
