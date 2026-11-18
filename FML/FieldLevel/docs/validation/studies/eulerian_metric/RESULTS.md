# FL-7 posterior-aware field metric validation

Recorded 18 November 2026. **FL-7 is fixed. Full scientific mixing remains open.**
[Machine-readable results](summary.json); [API, formula and restart change](../../../EULERIAN_METRIC.md);
[frozen field-only probe plan](../../../../validation/galaxy/metric/plan.json).

The corrected Gaussian metric uses the posterior noise policy and covariance volume,
Eulerian linear/Laplacian response, initial/matter/likelihood bands and actual mode
mask. It operates on the sampled mesh, caches frozen log precisions and collectively
invalidates rank-local masks, including owners with no active modes. The target
potential/gradient and data generation are unchanged.

## Algebra, integration and restart

Analytic mode/power errors are at most 1.13687e-12 on
noise powers 1, 1000 and 1e6, metric exponents -1,-0.5,0,0.5,1 and shared/compact
slab/pencil grids. An independently finite-differenced zero-field posterior Hessian
matches to 3.22258e-10 scaled error. The suite checks
noise-index policy, covariance-volume normalization and actual likelihood masks,
response cancellation, null modes, symmetry/positivity/inverse powers, cache reuse,
collective callback failure/retry, asymmetric-mask rejection, forward-tape preservation and MAP descent.

The final suite passes 78 serial and 156/160/160 MPI checks at 1/2/4 ranks.
Compact regressions pass 38 serial and 78/166/208 MPI checks. Corrected Eulerian
fresh-process checkpoint prepare/resume passes 15/43 serial and 30/86 per MPI rank
count, with exact state/metrics/statistics and retained suffix/sink indices.
Executable examples replay all 16/32/64 per-rank JSON draw files byte-for-byte
in serial/two-rank slabs/four-rank pencils, including orphan-file replacement.

The updated Eulerian metric identity deliberately rejects pre-FL-7 checkpoints.
A copied real pre-fix Eulerian checkpoint fails before continuation and retains
CURRENT unchanged. Existing field/survey/PM checkpoint suites pass on 1/2/4 ranks;
real pre-refactor legacy files also resume under the current reader. The generic
posterior dispatch preserves those default metric paths and identity bytes.

## Fixed-budget exploration comparison

Genuinely pre-fix and corrected serial executables run the identical probe:
compact N4, product N8, deposition N16, PSWF epsilon=1e-8, P=20000 exp(-5k),
noise power 1000, fixed physical scalars, identical initial/data/trajectory seeds,
four chains, 16 warmup and 128 retained draws, eight leapfrog steps at size 0.15,
no jitter or adaptation. There are 18 active initial real modes and 26 real
likelihood modes. Six fixed cosine/sine projections are retained for diagnostics.
The data, initial objective, all field gradients and scalar gradients compare as
**identical serialized bytes**, not merely approximate target agreement.

| Projection | Legacy Rhat | Corrected Rhat | Legacy bulk ESS | Corrected bulk ESS | Legacy tail ESS | Corrected tail ESS |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 4.3790 | 1.0049 | 4.5 | 261.0 | 11.6 | 400.1 |
| 1 | 2.9155 | 1.0066 | 4.9 | 207.4 | 11.5 | 357.6 |
| 2 | 4.0992 | 1.0235 | 4.5 | 171.0 | 11.6 | 323.1 |
| 3 | 3.3336 | 1.0079 | 4.7 | 179.5 | 11.6 | 385.2 |
| 4 | 4.1726 | 1.0160 | 4.5 | 198.3 | 11.7 | 333.8 |
| 5 | 4.1433 | 1.0047 | 4.5 | 231.8 | 11.5 | 223.9 |

Projections 0/1 are cosine/sine of n=(1,0,0), 2/3 of (1,1,0), and 4/5 of (0,0,1).
FML's existing rank-normalized split/folded diagnostics are used; status `ok`
means computable, not converged. The old chains are visibly unmixed, so their
ESS is a diagnostic flag, not a calibrated asymptotic efficiency estimate.

Both versions have zero sampling divergences and mean acceptance near 0.99/0.98.
Acceptance alone does not establish exploration. Total measured chain times are
48.72 s legacy and 41.31 s corrected
on this host; the key change is field movement at the same integration budget,
not a production runtime-speedup claim. No draws or settings were extended/retuned.
A replay with the final cache/mask guards reproduces every corrected trace,
diagnostic and target byte exactly. The replay is not pooled as additional draws.

Corrected maximum Rhat is 1.0235; it exceeds 1.01 for
two projections, and bulk ESS remains below 400. This short field-only probe
therefore does **not** pass all production mixing gates. It establishes improved
selected-projection exploration following the metric fix; it does not establish
marginalized sigma8 mixing, all-coordinate validation, coverage or a real-catalog
result. A declared longer joint field/scalar pilot is the next scientific step.

All builds use Clang 17, double precision and bounds checks; MPI runs use Open MPI
and two threads per rank. Source/binary hashes, pre-fix headers, exact commands,
raw traces/diagnostics, logs and plots are retained outside the library under
workspace `outputs/fieldlevel-fl7/`. The comparison figure was visually inspected.
