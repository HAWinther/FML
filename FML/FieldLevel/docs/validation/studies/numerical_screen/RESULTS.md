# Bounded Eulerian numerical/resource screen

Recorded 18 November 2026. All **17 predeclared numerical/smoke cases pass**.
[Machine-readable results](summary.json), [frozen plan](../../../../validation/galaxy/numerics/plan.json),
[driver/measurement contract](../../../../validation/galaxy/numerics/README.md),
[readiness finding](../../../NUMERICAL_SCREEN.md).

One synthetic Gaussian tracer field uses a common Fourier-keyed initial realization,
L=100, P=20000 exp(-5k), alpha_true=1, noise power 1000 and fixed physical initial,
matter and likelihood cutoffs. There are 80 active initial real modes and 124 real
likelihood modes; observations are not a discrete galaxy catalog. Assessments at
alpha=0.9,1,1.1 hold all other priors, bias and data fixed. The reference has compact
N8, product/particle N24, deposition N48 and epsilon=1e-12. Its finiteness and FD
checks pass, but a finite reference is not an exact continuum solution.

| Serial case | Compact/product/deposition | Epsilon | Max noise-weighted distance | Median forward+gradient, s | Process peak RSS, MiB |
| --- | --- | --- | --- | --- | --- |
| reference | 8/24/48 | 1e-12 | 0 | 0.8232 | 30.1 |
| baseline | 8/12/24 | 1e-08 | 0.00123 | 0.0373 | 9.8 |
| loose_tolerance | 8/12/24 | 1e-06 | 0.00123 | 0.0243 | 8.9 |
| tight_tolerance | 8/12/24 | 1e-10 | 0.00123 | 0.0619 | 9.2 |
| finest_tolerance | 8/12/24 | 1e-12 | 0.00123 | 0.0944 | 8.0 |
| deposition_32 | 8/12/32 | 1e-10 | 0.00123 | 0.0631 | 14.1 |
| deposition_48 | 8/12/48 | 1e-10 | 0.00123 | 0.1312 | 27.4 |
| product_16 | 8/16/32 | 1e-10 | 8.29e-06 | 0.1546 | 17.7 |
| product_24 | 8/24/48 | 1e-10 | 1.55e-10 | 0.5442 | 28.8 |
| latent_12 | 12/16/32 | 1e-10 | 8.29e-06 | 0.1626 | 19.1 |
| shared_16 | 16/16/32 | 1e-10 | 8.29e-06 | 0.1538 | 16.1 |

The baseline N8/N12/N24, epsilon=1e-8 has distance 0.00122887
against the 0.1 screen budget; active-field score error is
0.000172385 against 1e-3. Its local Fisher score
distance is 0.000673303; the local alpha-shift proxy is
0.000417541 Fisher sigma. The scaled Fisher
has full rank 5 and local alpha sigma 0.0924797.
These are **local proxies**, not a measured posterior-mean shift or the planned
0.1 posterior-SD scientific gate. Finite differences use h=1e-4 and error scaled
by max(1,|analytic|,|numeric|); all are below the frozen 1e-5 threshold.

Tightening tolerance and refining deposition alone barely change the N12 residual
floor. Joint product/particle refinement to N16 reduces the distance to
8.28998e-06. Particle and product resolutions are currently
coupled; this screen does not independently establish either continuum limit.
The compact/shared comparisons preserve the same physical Fourier realization
and operator/particle grids, rather than reusing incompatible real-cell RNG draws.

## Resource interpretation

Baseline median forward+gradient is about 0.037 s, with 9.8 MiB process peak and
about 0.185 s for a two-draw/two-leapfrog tiny-step HMC block. The fine reference
cost is about 0.823 s per gradient. Different numerical meshes and kernel widths
explain these costs; this is not an algorithmic library speedup measurement.

At fixed N16 products/N32 deposition, compact N8, compact N12 and shared N16 use
512/1728/4096 field coordinates but peak at 17.7/19.1/16.1 MiB in their individual
process runs. **No total-RSS saving is established.** Setup, assessment and HMC
all contribute to the peak, and sparse/padded workspaces plus compact transfer
scratch remain. One RSS measurement per case does not resolve small differences
statistically. Per-field-vector dimension savings remain valid.

MPI maximum prediction difference from the same serial N16 case is 1.55158e-16.
Slab and pencil gradients on four ranks are 1.73 and 1.64
times faster than their respective one-rank MPI builds on this small fixture.
Serial has no OpenMP; MPI uses two threads per rank. Runtime has three warm-cache
repeats, reported as median with raw repeats retained; setup/cold calls and the
full two-draw HMC cost are also stored. RSS is getrusage per-process peak; sums
of rank peaks are not simultaneous job RSS or unique physical memory.
All native serial work including data generation took 35.8 s and
all six MPI cases took 12.9 s, below the frozen caps.
No full-size timing, production-memory or large-rank extrapolation is made.

## Larger sampler readiness

The two tiny-step HMC draws have no divergences but establish no mixing.
Independent code inspection found FL-7: the Eulerian metric reuses unbiased
matter parameters with log noise zero, ignoring the posterior noise power and
Eulerian linear/Laplacian response. Its active-mode precision is
393.6--462.0 times the noise-aware **linear surrogate** on this
fixture. This is not a nonlinear Hessian or observed mixing ratio. It does not
change the Metropolis-corrected target, but can impair field exploration.

The numerical candidate for the next small synthetic sampler test is the default
N8/N12/N24, epsilon=1e-8. First address FL-7 and assess exploration; then freeze a
larger multi-chain budget and posterior/refinement diagnostics. Do not start a
long pilot merely because acceptance at step=1e-4 is high. No real data, posterior
calibration, mean-shift measurement or coverage result is claimed.

The maintained runner pins plan, executable and data hashes and every completed
output inventory. Resume verifies them, launches no completed native cases and
reproduces summary bytes exactly. A real child-process Ctrl-C test verifies group
cleanup. Logs, raw modes, per-case timings and plots are outside the source tree
in workspace `outputs/fieldlevel-numerical-screen/`. All cases, including loose
tolerance and fine reference, are retained; no gate or draw budget was retuned.
