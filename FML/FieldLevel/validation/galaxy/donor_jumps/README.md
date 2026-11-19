# Immutable donor joint field/scalar jumps

This exploratory sampler candidate targets physical-field/bias trapping that
remained after exact-support amplitude interweaving. It adds one full-target MH
translation per positive-temperature replica per sweep, after the unchanged local
field/scalar/support kernels.

Freeze twelve donor states from the completed dataset-203 exact-support study:
four ensembles × draws 0/255/511. Choose an ordered distinct pair (i,j) uniformly
and gamma=1 or 0.35 with equal probability, independently of the current state:

    w_new = w + gamma P(w_i-w_j)
    p_new = p + gamma (p_i-p_j)

P is the existing exact initial-band projector; all five scalar coordinates are
physical. Prior-null white coordinates remain fixed. Bounds reject, without
clipping/reflection/resampling. Pair (j,i) has the same probability as (i,j), so
this is a symmetric translation with Jacobian one. Full posterior MH uses U-U_new.
The pool is immutable and never receives candidate samples. Independent chains
remain independent conditional on the shared fixed proposal pool and data.

## Controls and local screen

Test.cpp independently checks pair symmetry, inverses, MH pair balance, null
preservation and known unequal-weight correlated Gaussian-mixture marginals.
A local-only sampler is a negative control for separated-mode trapping. These
are serial donor-kernel tests; the exact projector retains its separate slab and
pencil 1/2/4-rank verification. A production distributed donor loader is not added.

screen_plan.json fixes the local feasibility gates before execution. Twelve saved
states are both donors and starts, so this is in-sample exploratory evidence.
Probe.cpp reproduces their actual predictions/Q, checks 132 endpoint transfers
and the preserved-null prior-energy correction, then evaluates exactly 768 seeded
random proposals. Cross-ensemble donor labels are not proven mode identities.
Expected acceptance and physical displacement are not chain ESS or calibration.

```sh
make donor_test donor_probe donor_pilot
./donor_test > /absolute/unit.log
python3 screen.py prepare /absolute/empty-screen --source /absolute/support-study/candidate/mock203 --exe /absolute/donor_probe --unit-exe /absolute/donor_test --unit-log /absolute/unit.log
python3 screen.py probe /absolute/screen
python3 screen.py analyze /absolute/screen
```

## Frozen actual-chain test

Keep the exact-support rescaling, original frozen field mass, thirteen replicas,
four exchange passes, half-step 0.075 ×16, priors, initial fields and seed domains.
Only the extra donor kernel changes. A fresh four-chain 256/512 candidate compares
to the cached exact-support cohort, which is hashed and never rewritten.
All twelve original coordinate/transport/divergence gates remain. Six additional
alpha-weighted initial-mode gates are predeclared for this experiment and reported
separately. A robust result must pass both sets. No extension, dropping, pooling,
new donor states or retuning until pass.

The native checkpoint schema adds donor attempts, acceptances, support rejections,
large-scale counts and actual target-value calls (total/sampling). Identity binds
the pool bytes and algorithm version. Disabled mode reproduces the old support
kernel's draws/summary/target; enabled mode reproduces complete checkpoints and
counters across restart. Changed/disabled/missing/truncated pools reject resume.
The generated driver reuses the unchanged historical/support sources, with an
explicit native-hooks.patch outside the library.

```sh
python3 preflight.py /absolute/empty-controls --exe /absolute/donor_pilot --old-exe /absolute/support_pilot --pool /absolute/screen/donor-pool.txt
python3 study.py prepare /absolute/empty-study --source /absolute/support-study/candidate/mock203 --screen /absolute/screen --controls /absolute/controls/controls.json --exe /absolute/donor_pilot --diagnostics /absolute/calibration_diagnostics
python3 study.py run /absolute/study --exe /absolute/donor_pilot --diagnostics /absolute/calibration_diagnostics --max-rounds 1
python3 study.py run /absolute/study --exe /absolute/donor_pilot --diagnostics /absolute/calibration_diagnostics --resume
python3 study.py analyze /absolute/study --exe /absolute/donor_pilot --diagnostics /absolute/calibration_diagnostics
```

Actual HMC gradients and EXTRA donor target-value calls are reported separately;
a field-gradient ratio alone omits the donor overhead. A conservative mixed-call
proxy counts each extra value call like a gradient call, without claiming equal
runtime per call. Both baseline and candidate use four workers at different times
on shared hardware; raw timing is not a dedicated speed benchmark. Primary counters
exclude auxiliary replay, included in wall time.

Study/cohort locks, bounded checkpoint chunks and complete-ensemble replay remain.
SIGINT/SIGTERM/PAUSE drains active chunks; resume clears receipts. A cumulative
two-hour cap uses 1200-second reservations; crashed batches retain their charge,
and missing budget state beside native work rejects resume. This selected-data
study is not independent calibration, a real-catalog result or production MPI
replica orchestration. Raw pools, chains, builds and inventories stay outside the
library. See [results](../../../docs/validation/studies/donor_jumps/RESULTS.md).
