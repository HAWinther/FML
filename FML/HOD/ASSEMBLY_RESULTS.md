# Decorated HOD assembly-bias validation

Validated 12 November 2026. This is a bounded extension of backlog item 28:
explicit marginal satellite means and a Heaviside decorated-HOD prescription.
It does not supply empirical percentile calibration, a fit to observed galaxy
clustering, or a cosmological assembly-bias calibration.

## Mathematical and sampling contract

The decorator takes a conditional secondary-property percentile q at fixed mass,
a low-population fraction f in (0,1), and separate central/satellite strengths in
[-1,1]. q>f denotes the high population, of fraction h=1-f. Central probabilities
are convex combinations of the baseline p and an extremal bounded allocation;
marginal satellite means interpolate toward allocating satellites to the preferred
population. Therefore `f*p_low+h*p_high=p` and
`f*mu_low+h*mu_high=mu` when the declared population fractions apply at fixed mass.

The existing `OccupationMeans{p,lambda}` contract remains conditional-rate based.
The new `MarginalOccupationMeans{p,mu}` and `sample_marginal_occupation` make the
unconditional satellite target explicit. `populate` accepts either return type.
For central-required draws, use Poisson(mu/p) when the central is present; for
independent satellites, use Poisson(mu) irrespective of central presence. This
preserves the intended marginal mean even when assembly bias changes p.

The central-required law cannot realize p=0 with mu>0. It rejects such a state,
and checks its conditional rate before drawing the central. Independent mode
supports it. Nonfinite/out-of-range rates and resource failures reject explicitly;
realized draws above the cap are never clipped. Population first-moment conservation
does not fix realized counts, higher moments or clustering. The central-required
law retains the population covariance mu*(1-p) because Ncentral*Nsatellite equals
Nsatellite; the independent-at-fixed-q law can acquire mass-conditioned covariance
when the decorated populations are mixed. See [README.md](README.md) for the exact conventions and compatibility
constraints at maximal strengths.

## Independent reference calculations

The C++ exporter supplies 16,000 pointwise cases spanning central probabilities
0, 1 and near-zero/intermediate values; zero/nonzero satellite means; unequal
population splits; positive, negative, zero and maximal strengths; and percentiles
below, equal to, immediately above and beyond the split.

An independent 80-digit standard-library Decimal implementation evaluates the
usual signed-perturbation bounds and opposite-population compensation. The C++
implementation instead uses convex extremal populations and exponent-separated
satellite scaling. Maximum discrepancy, scaled by max(1, absolute reference
component), is **1.026e-16**.

An optional second comparison executes the original Halotools
`HeavisideAssembias._galprop_perturbation` method extracted from its source AST,
with supplied baseline arrays, strengths, bounds and splits. It does not replace
the method's arithmetic with the local reference. Maximum scaled discrepancy is
**8.882e-16**. The pinned source is from `astropy/halotools` commit
`256b9020d47145939f03c8595a4ca43134cacf68`; its source hash, URL and license accompany the review artifacts.
This is an isolated original pointwise-kernel comparison, not a full Halotools
installation, percentile-estimation or stochastic mock-population comparison.

The comparison explicitly supplies marginal satellite baselines. Halotools'
unmodulated Zheng07 satellite mean is lambda(M), while FML's existing baseline is
p(M)*lambda(M). No agreement between unmatched default full models is implied.
The physical prescription follows
[Hearin et al.](https://arxiv.org/abs/1512.03050); the checked
[Halotools source](https://halotools.readthedocs.io/en/latest/_modules/halotools/empirical_models/assembias_models/heaviside_assembias.html)
records the reference perturbation algorithm.

## Stochastic, stability and replay checks

Native tests check:

- Weighted population-mean conservation, bounded central probabilities,
  nonnegative satellite means and the strength sign across 4,500 parameter cases.
- Exact baseline marginal means at zero strengths, split-equality convention,
  and conversion from existing conditional-rate means before decoration.
- Exponent-separated rare-population scaling, avoiding intermediate underflow;
  unused overflowing extrema at zero strength; explicit failure for overflowing
  output means.
- Four 120,000-row count experiments with independent halos within each (low/high
  populations, each with
  conditional/independent satellite laws), with analytic first, second and central
  cross moments. Each comparison uses an eight-standard-error bound computed from
  analytic second/fourth moments, rather than a fitted tolerance from the sample.
- Independent satellite draws at p=0, compatibility failures for central-required
  models, invalid percentiles/parameters, pre-draw rate validation and resource caps.
- Stable galaxy identities, positions and velocities after halo reorderings,
  MPI ownership changes and OpenMP thread changes with fixed calibrated percentiles.

A separate standalone C++ before/after comparison uses the original HOD header
with both satellite laws and both NFW/member profiles. All emitted galaxy identities
and hexadecimal phase-space values are byte-identical to the new header. Existing
braced `sample_occupation` calls compile unchanged. A new-header smoke program
also builds/runs with no Global.cpp, MPI, FFTW, GSL or Python dependency.

## Example and build matrix

The new example constructs four fixed-mass synthetic populations, each containing
1000 halos with uniform concentration on [4,12] and analytic percentile
q=(c-4)/8. It uses split=.5 and strengths (.4,.6), so each population has exactly
500 high/500 low halos. Mean/galaxy outputs are rank-local CSV shards; only small
summary arrays are reduced. The catalogue and galaxies are never gathered.

Every per-halo example mean matches the independent Decimal equations, and the
expected central/satellite sums match their baseline sums in every mass bin.
The default seed yields 3825 galaxies. That realized count is stochastic;
only the expected population first moments are constrained. Shards from 1/2/4
MPI ranks and 1/4 OpenMP threads replay exactly after sorting by the full integer
identity pair. IDs above 2^53 remain integers in checks and output.

These percentiles are analytically known for a synthetic distribution, not
estimated from a real halo catalogue. Positions are synthetic; no cosmological
clustering/lensing response or fitted HOD parameters are claimed.

| Build/runtime | Existing HOD checks | New assembly-bias checks | Reference/example |
|---|---:|---:|---|
| Apple Clang serial | 37 | 22 | Pass |
| Clang17 OpenMP, 1/4 threads | 38 | 23 | Pass |
| Open MPI/Clang17 + OpenMP, 1/2/4 ranks | 38 | 23 | Pass |
| Apple Clang ASan+UBSan | 37 | 22 | Pass |

Every build compiles the old tests, new tests and new example from source.
Captured builds have no warnings. The actual module Makefile `run` target and
both example targets pass in clean copied directories. Module tests now include
the Decimal reference, needing only the Python standard library; the optional
Halotools kernel comparison additionally needs NumPy and an explicitly supplied
source file. CI discovers HOD tests through the existing module runner, so no
network-dependent test or new service is introduced. The complete Linux GitHub
workflow was not dispatched from this local work.

## Cost and remaining limits

Decoration is constant work per halo, stores no catalogue or global rank table,
and does not add fields to `Halo`. `PreparedCatalog`, stream namespaces,
phase-space generation, ownership and legacy input conventions are retained.
Marginal-rate validation is performed before the fill's parallel region. No speed
or peak-memory improvement is claimed.

The caller must calibrate the conditional CDF and its population fractions using
consistent halo definitions and selection. Computing percentiles independently
on each MPI subset can change the model. Finite mass-bin trends, small samples
and atoms/ties at the split need explicit calibration/tie handling; the library
cannot infer or certify them. Empirical rank preparation, occupation fitting,
cosmological clustering/lensing calibration and radius-dependent velocity models
remain future work. Exact realized-count preservation is a distinct algorithm.
