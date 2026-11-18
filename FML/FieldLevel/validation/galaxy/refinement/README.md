# Posterior numerical-refinement validation

This serial fixture tests numerical sensitivity on the passing four-chain N4
Eulerian 2LPT posterior. It keeps all 64 compact white coordinates, physical
parameters/priors, spectrum, box, initial/matter/likelihood bands, fixed noise and
observed data unchanged. Product/displacement and particle-lattice meshes refine
together; deposition resolution and PSWF tolerance also vary. It is not a new
refined-target MCMC run, independent mock test or general continuum certificate.

The frozen plan evaluates all 8192 cold states at six settings and compares every
refinement to baseline, plus the two finest settings to each other. No thinning,
hot-state/replay pooling, prior changes or extension-until-pass is permitted.
Source plan/result hashes, checkpoint hashes and sealed input hashes bind the
source. Baseline predictions and quadratics must reproduce the sampler output.

Because the integration variables and priors are identical, the exact density
ratio is `exp(Q_baseline - Q_case)`, with Q the original fixed-noise likelihood
quadratic. No prior factor or field-dimension Jacobian is introduced. Predictions
remain in the same ordered compact Fourier-mode basis with the same weights.

`weights.py` computes self-normalized ratio estimates. Paired mean-shift errors
use the difference between the two ratio-estimator influence functions. Batch
lengths 64 and 128 are evaluated within each chain, together with the standard
error across the four independent chain means; the largest estimate is used.
The inverse-squared-weight overlap index is **not** an autocorrelation-corrected
MCMC ESS. Marginal quantile and SD changes are point estimates; quantile uncertainty
is not gated. Numerical shift precision does not imply equally small uncertainty
in the underlying posterior means. All thresholds are explicit in plan.json.

```sh
make refinement_predict USE_MPI=false USE_OMP=false
python3 check_weights.py
python3 run.py /absolute/output --source /absolute/tempering-long/campaign \
  --exe /absolute/build/refinement_predict --plan plan.json --workers 4
python3 analyze.py /absolute/output
```

Completed prediction batches have SHA-verified receipts and resume without repeat
pooling. The runner locks its output tree, locks the source while sealing inputs,
and passes the output lock descriptor into native children. SIGINT/SIGTERM or
`output/PAUSE` stop after active batches; rerun with `--resume`. `--max-chunks 1`
limits additional batches for each case/chain task. An interrupted batch without
a committed receipt repeats its indexed prediction file. Source chains are read
only; no new posterior draws are generated. Inputs and receipts must match on
resume. Malformed controls/modes/states and existing native outputs reject.

Raw inputs, predictions, native executables and plots belong outside the library.
The durable result lives in `docs/validation/studies/posterior_refinement/`.
