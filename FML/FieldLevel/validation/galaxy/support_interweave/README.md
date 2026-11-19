# Exact initial-support amplitude/field interweaving

This experiment replaces the existing all-white rescaling with

    w_new = w + (alpha/alpha_new - 1) P w
    log_J = r * (log(alpha) - log(alpha_new))

P is the orthogonal projector of the Eulerian LPT model's exact spherical initial
filter, including zero-mode exclusion. r is its GLOBAL real Fourier rank, counted
using Hermitian multiplicities on the sampled grid. Neither observation-mode
selection, per-rank stored-cell counts, padded product meshes nor a fitted Jacobian
rank define it. The actual Gaussian prior remains on every stored white coordinate.

The proposal uses the same symmetric reflected physical-alpha increment and full
posterior MH correction. For this amplitude-only model it preserves the initial
physical field and complete 1LPT/2LPT prediction. It leaves prior-null directions
fixed. The old all-white rescaling with its full-dimension Jacobian is correct;
this tests whether avoiding uninformed motion improves mixing. On the mesh-4
fixture the transformation rank is eighteen while the latent dimension stays 64.

## Kernel and backend controls

Support.h uses the model's FFTWGrid or PencilGrid and domain, with a reusable local
FFT workspace and bounded local memory; it never gathers the full field. Scalars
and rank are replicated; input/allocation failures propagate collectively. The
model's immutable initial-band contract is explicit in the type restriction.
The native scientific driver remains serial even though these kernel tests run
on distributed slabs and pencils.

Test.cpp checks independent active/excluded/zero Fourier tones, projector
idempotence/self-adjointness/orthogonality, global rank eighteen with different
shared/compact/likelihood layouts, prediction invariance, exact prior-energy gap,
null preservation, inverse and signed-Jacobian/MH reciprocity on 1LPT/2LPT, and
rank-local invalid-input/allocation failures. Independent Gaussian conditional
sampling plus amplitude quadrature tests uniform/Gaussian priors with/without
observations and 1/64 dimensions. A deliberate full-dimension Jacobian on the
active-only move must fail that reference.

Run support_test serially and support_test_mpi on 1/2/4 ranks. Match the launcher
to the MPI implementation used by the compiler; inspect the reported communicator
rather than counting launched processes. See saved controls for the actual flags.

## Frozen actual-chain comparison

A fresh four-chain cohort uses 256 discarded warmup sweeps +512 retained cold
draws, the existing thirteen-replica refined ladder, four exchange passes,
step 0.075 ×16 and the ORIGINAL frozen field metric. The nonlinear Gauss-Newton
candidate was not adopted. Only the rescaling support/Jacobian changes. Data,
starts, seed domains, priors, scalar kernels, field step, model and all twelve
coordinate/transport/divergence gates stay fixed. No extension, dropping, pooling
or retuning until pass. The completed original-metric refined-ladder cohort is
reused read-only as the baseline; all its records/checkpoints/diagnostics are hashed.

Report rescaling acceptance separately from convergence. Compare actual primary
HMC field-gradient counts including warmup. Extra support FFT work is outside
that counter and included in raw wall time. The baseline shared eight workers
across two cohorts; the fresh candidate uses four. Wall times do not certify a
speedup. Auxiliary restart replay is excluded from gradient counts but included
in timing. Failed-chain ESS and posterior summaries remain provisional.

make_driver.py adds explicit unique hooks to the unchanged historical native
source, generating SupportMain.cpp outside the library. The projector is created
once per worker and reused. Native checkpoint identity includes policy kind,
algorithm version and global rank, in addition to existing model/filter/target/
settings identity. Changed/missing/malformed policy rejects resume. Legacy mode
must reproduce original draws/summary/target; active mode must reproduce complete
checkpoint/summary/draws across restart. Runner SHA256 identity also pins policy.

```sh
make support_test support_pilot
./support_test
python3 preflight.py /absolute/empty-controls --exe /absolute/support_pilot --old-exe /absolute/old_tempering_pilot
python3 study.py prepare /absolute/empty-comparison --source /absolute/ladder-study/refined/mock203 --tests /absolute/tests.json --controls /absolute/controls/controls.json --exe /absolute/support_pilot --diagnostics /absolute/calibration_diagnostics
python3 study.py run /absolute/comparison --exe /absolute/support_pilot --diagnostics /absolute/calibration_diagnostics --max-rounds 1
python3 study.py run /absolute/comparison --exe /absolute/support_pilot --diagnostics /absolute/calibration_diagnostics --resume
python3 study.py analyze /absolute/comparison --exe /absolute/support_pilot --diagnostics /absolute/calibration_diagnostics
```

Study/cohort locks prevent concurrent writes. SIGTERM/SIGINT or PAUSE drains native
checkpoint chunks; --resume clears pause receipts. A cumulative two-hour cap uses
1200-second batch reservations; crashes retain their charge and missing budget
state beside native work rejects resume. Every chain performs complete-ensemble
replay. This selected-dataset comparison is not independent calibration, a
real-catalog result or production MPI replica orchestration. Large builds, raw
chains, logs and full inventories stay outside the source tree.
See [results](../../../docs/validation/studies/support_interweave/RESULTS.md).

physical_diagnostics.py adds a read-only exploratory supplement after the frozen
result. It reports both white and alpha-weighted in-band initial-mode diagnostics
with the original diagnostic binary. These are explicitly post-hoc, do not change
the twelve original gates, and cannot turn a failed study into a pass.
