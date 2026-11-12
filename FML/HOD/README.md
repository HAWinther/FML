# HOD mock galaxies

`HOD.h` provides a local, header-only halo occupation and galaxy phase-space API.
`AssemblyBias.h` adds a mean-preserving Heaviside decorated-HOD prescription.
It needs neither FFTW nor GSL. It uses the shared Philox primitive in
`RandomGenerator/CounterBasedRandom.h`; existing random-field realizations retain
the same Philox mapping.

```cpp
using namespace FML::HOD;
std::vector<Halo> halos;
// Fill stable halo ID, mass, radius, concentration, position, bulk velocity,
// and 1D velocity dispersion. Optionally supply member-particle phase space.
PreparedCatalog prepared(std::move(halos));
Zheng07 occupation;
occupation.log10_mmin = 13.;
occupation.sigma_log_mass = .4;
occupation.m0 = 1e12;
occupation.m1 = 1e14;
occupation.alpha = 1.;
Options options;
options.seed = 1234;
options.boxsize = 1000.;
auto mock = populate(prepared, occupation, options);
```

`PreparedCatalog` owns and validates the halo data once and sorts member-particle
indices by stable member ID. Reuse it for seed, occupation and velocity-bias scans;
move a large input catalogue to avoid a copy. It does not infer halo mass/radius
conventions, cosmology, a concentration–mass relation, or velocity dispersion.
Adapt FoF/SO or external catalogues explicitly so these definitions remain clear.

## Occupations

The built-in model returns

```
p_c(M) = 0.5 [1 + erf((log10(M)-log10(Mmin))/sigma_log_mass)]
lambda(M) = ((M-M0)/M1)^alpha, for M>M0; otherwise zero.
```

By default, draw `Ncentral ~ Bernoulli(p_c)`; satellites are zero if the central
is absent and `Poisson(lambda)` otherwise. Thus the unconditional satellite mean
is `p_c*lambda`, with the central/satellite covariance implied by this conditional
model. With `require_central=false`, the satellite count is independently drawn
from `Poisson(p_c*lambda)`. The unconditional mean is preserved but its variance
and central correlation change. These conventions are explicit, not inferred
from the provided mean's name. Note that Halotools' `Zheng07Sats` defaults to `modulate_with_cenocc=False`, i.e. an
unconditional satellite mean lambda(M) that is not multiplied by p_c: to match that, return
`satellites_if_central = lambda/p_c` from a custom callable with `require_central=false`, or compare with
`modulate_with_cenocc=True` (this API's default). The two differ where p_c < 1 (low masses).

A custom callable can replace `Zheng07` and return
`OccupationMeans{central_probability, satellites_if_central}` from a `const Halo&`.
It is evaluated serially before the parallel phase. Use `sample_occupation(id,
means, options)` to obtain counts without allocating any galaxy phase space.

Poisson sampling uses a product method below mean 10 and Hörmann transformed
rejection above it. Supported means are finite and at most 1e8.
`max_satellites_per_halo` is an explicit resource guard (default one million): an
oversized mean or draw fails, rather than clipping a Poisson occupation. Allocation
sizes are checked before the output vector is built.

## Explicit marginal means and assembly bias

`OccupationMeans{p,lambda}` retains its existing conditional-rate convention.
`MarginalOccupationMeans{p,mu}` instead declares the unconditional first moments
`E[Ncentral]=p`, `E[Nsatellite]=mu`. A population callback can return either type.
For direct counts, call `sample_marginal_occupation(id, means, options)` for the
new type; the original `sample_occupation` and its braced calls are unchanged.

With marginal means, central-required satellites use `Poisson(mu/p)` when the
central is present and zero otherwise. Thus `E[Nsatellite]=mu` and
`Cov(Ncentral,Nsatellite)=mu*(1-p)`. With `require_central=false`, satellites are
independent `Poisson(mu)` even at p=0. The conditional model cannot represent
p=0, mu>0 and rejects it. Its conditional rate, or the independent marginal rate,
must be finite, at most 1e8 and within `max_satellites_per_halo`; marginal inputs
are checked before drawing the central. A realized draw above the resource cap
still fails rather than being clipped. Higher moments differ between these laws.

```cpp
#include <FML/HOD/AssemblyBias.h>
using namespace FML::HOD;
Zheng07 baseline;
HeavisideAssemblyBias assembly{.5, .4, .6}; // split, central strength, satellite strength
// calibrated_percentile(h) must return q = CDF(secondary property | halo mass).
auto occupation = [&](const Halo& h) {
    return assembly(baseline(h), calibrated_percentile(h));
};
auto mock = populate(prepared, occupation, options);
```

Both strengths lie in [-1,1]. Positive strength favors the higher-percentile
population; negative strength favors the lower. `split` must lie strictly between
0 and 1 and is the low-population fraction f; high fraction h=1-f. `q>split` is
high and equality belongs to low. The secondary percentile must be finite in
[0,1]. Parameters may depend on mass by constructing the struct in the caller's
occupation callback; the built-in object has constant parameters.

At fixed mass, the central prescription interpolates between the baseline p and
an extremal bounded distribution. For positive strength, centrals fill the high
population first: if p<=h, `(p_low,p_high)=(0,p/h)`; otherwise
`((p-h)/f,1)`. Negative strength exchanges the preferred populations. Satellites
interpolate between baseline mu and `(mu_low,mu_high)=(0,mu/h)` for positive
strength or `(mu/f,0)` for negative. Consequently

```
f*p_low + h*p_high = p
f*mu_low + h*mu_high = mu
```

`assembly(baseline(h),q)` first converts the existing conditional rate into
`mu=p*lambda`, then decorates **marginal** satellites. This avoids changing their
intended mean inadvertently when central probability changes. To start from an
already unconditional satellite mean, use
`assembly.decorate_marginal(MarginalOccupationMeans{p,mu},q)`. For example, supplying
`mu=lambda` matches the baseline satellite-mean convention of Halotools'
`Zheng07Sats(modulate_with_cenocc=False)`; its default differs from FML's p*lambda
baseline. The decoration arithmetic is checked against the original Halotools
pointwise perturbation kernel, not its complete catalogue-population workflow.

Some combinations of maximal central and satellite strengths produce p'=0 with
mu'>0. They are valid independent-satellite models and incompatible with
central-required sampling. The latter rejects rather than altering the declared
marginal means. Use compatible strengths or explicitly choose the independent law.

**Conservation requires calibrated population fractions at fixed mass.** The API
accepts a supplied percentile; it does not compute empirical ranks, gather halo
catalogues or infer a concentration/environment distribution. Do not recompute
percentiles independently on each MPI subset. Calibrate a common conditional CDF
or supply a stable rank-independent map, with explicit mass binning and tie
handling. Ties/atoms at the split and finite-bin mass trends can make the actual
high fraction differ from 1-split and spoil conservation for that sample. The
library cannot validate the scientific calibration of caller-supplied percentiles.

The equations conserve population **first moments**, not exact counts for a finite
mock, higher moments or galaxy clustering. For central-required sampling,
`Ncentral*Nsatellite=Nsatellite`, so the mass-conditioned population covariance
remains `mu*(1-p)` when the first moments are conserved. For independent sampling,
counts are independent at fixed q, but mixing populations with different p and mu
can change their mass-conditioned covariance. The
existing ID-keyed streams are retained, but different strengths can change counts
and hence available galaxy ordinals. Passing zero strengths preserves the baseline
marginal means exactly; conversions between marginal and conditional rates still
have normal floating-point rounding. No fixed-realization count-conservation
algorithm is implied.

The prescription follows [Hearin et al.](https://arxiv.org/abs/1512.03050) and the
[Halotools Heaviside implementation](https://halotools.readthedocs.io/en/latest/_modules/halotools/empirical_models/assembias_models/heaviside_assembias.html).
It is independently implemented here using convex extremal populations. Satellite
scaling uses exponent-separated products to avoid unnecessary intermediate
underflow/overflow. Overflowing or nonfinite decorated means fail explicitly;
final representational underflow follows double-precision rounding.

## Positions, velocities and identities

- Centrals are at the supplied halo center. Their velocity is bulk velocity plus
  a Gaussian with 1D dispersion `central_velocity_bias * halo.velocity_dispersion`
  (zero by default).
- NFW satellites use isotropic directions and an inverse mass CDF for the NFW
  profile truncated at `halo.radius`. Log-radius inversion and a scaled small-c
  form keep the inverse CDF well behaved across a wide concentration range. The concentration is multiplied by
  `satellite_concentration_factor`. The 1D velocity dispersion is
  `satellite_velocity_bias * halo.velocity_dispersion` (factor one by default).
  This is an explicit Gaussian phase-space prescription, not a radius-dependent
  Jeans or equilibrium distribution-function solver.
- `SatelliteProfile::Particles` samples supplied member particles uniformly without
  replacement (no two satellites on the same particle), with replacement only when a halo has more satellites
  than members. It uses their positions and
  `halo.velocity + satellite_velocity_bias*(member.velocity-halo.velocity)`.
  Member-ID sorting preserves the realization under member reordering. Members
  must be provided on the owning halo's rank; membership discovery/exchange is
  outside this API.

Mass parameters use the same units as halo masses. Positions/radii/box size use
one consistent length unit; velocities/dispersion use one consistent velocity
unit. `boxsize>0` wraps positions into `[0,boxsize)`, while `boxsize=0` retains
unwrapped Cartesian positions. Periodic NFW radii must be smaller than half the
box. No RSD shift is automatically applied. FML mesh/particle APIs that expect
positions in `[0,1)` still need an explicit coordinate conversion.

Each galaxy has the identity pair `(halo_id, ordinal)`: central ordinal zero,
satellites numbered from one, even when a central is absent. No lossy hash is
advertised as a unique galaxy ID. Halo IDs must be globally unique; local
preparation checks local duplicates. Galaxy objects provide the standard FML
`get_pos`, `get_vel` and `get_ndim` methods.

## Replay and MPI/OpenMP

There are no hidden MPI collectives or catalogue gathers. Populate locally owned
halos and retain their galaxy ownership, even if positions wrap across a slab;
redistribute galaxies explicitly if a later mesh API requires slab ownership.
Per-halo streams separate central occupation, satellite counts, NFW positions,
member choices, and central/satellite velocities. Fixed inputs, seed and stream
replay independently of halo order, rank ownership, and OpenMP thread count.
Integer Philox mapping is portable; transcendental rounding can vary across
compilers/libm implementations. Binary identity is promised within the same
floating-point build, not across every architecture or fast-math setting.

The output vector follows local input halo order, central first then satellites;
concatenate/sort by identity when comparing different MPI decompositions.
Prepared inputs can be shared read-only. The phase-space fill is parallelized
across halos; validation, model callbacks and output allocation occur beforehand.

## Example and tests

`example/` populates a synthetic distributed halo catalogue, **explicitly**
replicates the small galaxy sample, sorts it by identity, and runs kNN-CDF and
counts-in-cells summaries with distributed random volume queries. It writes
`hod_galaxies.txt`, `knn_cdf.txt`, `counts_summary.txt`, and `counts_pdf.txt`.
The last count-PDF row per radius is an overflow probability, not a clipped
occupation. The example is an interface demonstration, not a calibrated galaxy
population or a cosmological HOD fit.

```
make -C FML/HOD/example
mpirun -np 4 FML/HOD/example/mock_statistics
make -C FML/HOD/tests run
make -C FML/CountsInCells/tests run
```

Tests include Philox known answers, Bernoulli/Poisson moments and covariance,
NFW inverse-CDF accuracy and sampled radial/isotropy checks, velocity moments,
particle phase-space selection, order/member/rank/thread replay and invalid data.
The dedicated assembly-bias tests additionally check marginal count moments and
central/satellite cross moments, conservation and endpoint/sign bounds,
conditional compatibility, resource failures and replay. A standard-library
80-digit Decimal comparison runs with the module tests; optional comparison with
the pinned original Halotools kernel needs NumPy and a separately supplied source
file. See [ASSEMBLY_RESULTS.md](ASSEMBLY_RESULTS.md) for the exact validation scope.

`example/AssemblyBias.cpp` produces rank-local galaxy/mean CSV shards and global
fixed-mass summaries without gathering the catalogue. Its concentration
percentiles come from an explicitly synthetic uniform conditional distribution;
it is an API example, not calibrated galaxy clustering. Run:

```sh
make -C FML/HOD/example USE_MPI=false
FML/HOD/example/assembly_bias /tmp/hod-assembly
# Optional parameters: prefix central_strength satellite_strength split seed
FML/HOD/example/assembly_bias /tmp/hod-assembly .4 .6 .5 9183
```

Occupation fitting, empirical percentile calibration and cosmological assembly-bias
clustering/lensing tests remain extensions, together with radius-dependent satellite
velocity models. The header APIs remain local and require neither Global/MPI,
FFTW, GSL nor Python. The optional validation scripts use Python, and the example
uses Global/MPI only for explicit ownership and summaries.

References:
[Zheng et al. (2007)](https://arxiv.org/abs/0703457),
[Halotools occupation conventions](https://halotools.readthedocs.io/en/stable/api/halotools.empirical_models.Zheng07Sats.html),
and the [NumPy distribution implementation](https://github.com/numpy/numpy/blob/main/numpy/random/src/distributions/distributions.c)
for the transformed-rejection algorithm convention.
