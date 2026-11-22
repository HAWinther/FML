# Reparametrization / analytic bias marginalization comparison

Compares three samplers on the joint-pilot target with Gaussian bias priors (the
means/widths of the pilot's uniform priors): joint HMC over (field, sigma8 ratio,
four biases), the same with b_delta replaced by b_delta * sigma8 ratio, and HMC over
(field, sigma8 ratio) with the biases integrated out exactly (`MarginalizedBias.h`),
adding one conditional bias draw per sample. Four datasets (0: the pilot's mock;
1-3: truths drawn from the priors), four chains each, the pilot's budget and starts.

```sh
make -C FML/FieldLevel/validation/galaxy/reparametrization reparam_study USE_MPI=false
for d in 0 1 2 3; do for m in joint reparam marginal; do for c in 0 1 2 3; do
  reparam_study run /absolute/output $d $m $c; done; done; done   # independent; run in parallel
reparam_study analyze /absolute/output                             # writes diagnostics.csv
```

Each run writes `d<D>-<method>-c<C>.csv` (draws of the twelve monitored quantities)
and `.json` (acceptance, divergences, gradient evaluations, time, potential).
Keep outputs outside the library. Results: [RESULTS.md](../../../docs/validation/studies/reparametrization/RESULTS.md).
