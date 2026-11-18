# Exploratory geometry audit of the separated chain

Recorded 18 November 2026. No new Monte Carlo sampling.
[Maintained driver](../../../../validation/galaxy/geometry/README.md);
[diagnostic scope](../../../../validation/galaxy/geometry/plan.json);
[machine-readable results](summary.json);
[proposed likelihood-tempering test](../../../../validation/galaxy/geometry/NEXT_STEPS.md).

## Question and selected states

The amplitude/field interweaving campaign still fails all eleven coordinate gates.
One chain remains in a different high-amplitude/bias configuration. This audit asks
whether adjusting the four bias coefficients can remove barriers on representative
paths between those configurations, and checks the adjoint at the actual endpoints.
The observed separation motivated this exploratory diagnostic; it is not a blind
validation or evidence calculation. No chain is discarded or reweighted.

Use primary retained draw 512 from zero-based chain 1 (plot label Chain 2) and the
same draw from chains 0, 2 and 3. Endpoints are not chosen by maximizing posterior
values. Three paths per pair interpolate whitened fields, physical alpha*white,
or white-field direction along a spherical arc with interpolated norm. Alpha and
physical biases interpolate linearly. Each path has 101 points. A 101-point pure
rescaling control preserves the physical density: 1010 total evaluations.

## Exact bounded-bias profiles and numerical checks

At each fixed field/amplitude, the Gaussian likelihood is quadratic in four biases.
All priors are the original bounded uniforms. Enumerating all 81 lower/free/upper
active sets and solving their free equations finds the lowest feasible objective;
a scaled convex KKT residual certifies the minimum. No proposal ridge or additional
Gaussian bias penalty is added. Independent clipped, coupled and flat reference
cases pass. Every profiled value is checked against the full existing posterior.

Maximum scaled KKT residual: 5.14768e-15. Maximum cache/profile/full-target
error: 6.7446e-14. Rescaling-control likelihood spans are
1.7053e-13 before and 1.7053e-13
after profiling, consistent with fixed physical density.

At the four selected primary states, three independent field directions and all
five physical scalar finite differences give 32 derivative comparisons. Maximum
scaled discrepancy from the adjoint is 5.77318e-09.
The actual scalar perturbations remain inside support. Nonfinite derivatives are
explicit failures. Adding these checks and replaying the audit leaves every
profile/output CSV byte identical. These checks find no endpoint adjoint defect;
they do not prove correctness at every possible state or independently establish
posterior calibration.

## Observed barriers

A displayed barrier is the largest **sampled point potential minus the higher
endpoint potential**. The profile minimizes biases at each point, while leaving
that path's field and amplitude fixed. Shared likelihood constants are omitted.
Profiles are not marginalized densities and contain no configuration-volume factor.

| Path | Sampled direct barrier | Sampled profiled barrier | Max benefit from bias profiling |
|---|---:|---:|---:|
| white-to0 | 237.97 | 133.42 | 113.03 |
| physical-to0 | 338.08 | 140.71 | 225.92 |
| arc-to0 | 376.44 | 134.67 | 243.15 |
| white-to2 | 157.36 | 124.41 | 39.64 |
| physical-to2 | 157.06 | 124.43 | 39.02 |
| arc-to2 | 299.09 | 132.17 | 169.85 |
| white-to3 | 145.11 | 127.79 | 27.42 |
| physical-to3 | 151.43 | 128.24 | 37.42 |
| arc-to3 | 326.18 | 133.81 | 195.89 |

All nine tested routes retain sampled barriers of 124.4–140.7 negative-log-posterior
units even after bias minimization. Bias profiling helps substantially on some
paths but does not remove the observed field-configuration barrier. These paths
do not exclude a lower curved route elsewhere. A point-potential barrier also
cannot by itself quantify an HMC transition rate or configuration probability mass.

## Exploratory likelihood tempering

Reusing the same evaluated paths, set `U_beta = white_prior + beta * data_quadratic`.
With uniform bias priors, multiplication by positive beta preserves the fixed-field
bias minimizer; at beta=0 bias values are irrelevant to the prior energy. Thus this
scan requires no new forward runs or Monte Carlo samples. It was added after the
initial geometry result and is explicitly exploratory.

| Likelihood beta | Min barrier across paths | Max barrier across paths |
|---:|---:|---:|
| 1 | 124.41 | 140.71 |
| 0.5 | 50.80 | 65.01 |
| 0.25 | 12.33 | 30.76 |
| 0.125 | 0.00 | 13.92 |
| 0.0625 | 0.00 | 5.58 |
| 0 | 0.00 | 0.00 |

This supports testing likelihood-only replica exchange, with the unchanged prior
and the desired posterior at beta=1. The proposed initial ladder includes
1, 1/2, 1/4, 1/8, 1/16 and 0, subject to a separately declared short connectivity
pilot. Lower path barriers do not demonstrate adjacent energy overlap, exchange
acceptance, round trips or cold-chain convergence. None of these replicas has
been sampled in this audit. Full prior/target and swap-ratio tests must precede it.

## Scope and artifacts

This audit diagnoses selected paths in the same N4/N8/N16 synthetic target. It is
not multimodality proof, all-route optimization, posterior reweighting, coverage,
physical refinement or a real-catalog result. Straight interpolation may pass
through unrepresentative field norms; the arc paths provide a complementary check,
not a guarantee of typical-set connectivity. No inference intervals are certified.

Raw endpoint/point hashes, generated paths, native source/binary hashes, build/run
commands and output CSVs remain outside the library under workspace
`outputs/fieldlevel-geometry-audit/`. Plots under `audit/`: `posterior-paths.png`,
`profiled-barriers.png`, `tempered-barriers.png`. All prior sampling campaigns remain
unchanged. The geometry probe is serial; no production MPI profiling/tempering
API is added. Item 52 remains ongoing.
