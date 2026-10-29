# HOD mock galaxies

`HOD.h` provides a local, header-only halo occupation and galaxy phase-space API.
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
Assembly-bias prescriptions, occupation fitting, and radius-dependent satellite
velocity models remain extensions; custom occupation callables already permit
user-supplied dependence on halo properties.

References:
[Zheng et al. (2007)](https://arxiv.org/abs/0703457),
[Halotools occupation conventions](https://halotools.readthedocs.io/en/stable/api/halotools.empirical_models.Zheng07Sats.html),
and the [NumPy distribution implementation](https://github.com/numpy/numpy/blob/main/numpy/random/src/distributions/distributions.c)
for the transformed-rejection algorithm convention.
