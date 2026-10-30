# Baryonification

Position-only post-processing of a three-dimensional gravity-only snapshot. The module builds radial mass
profiles around halos, matches enclosed masses to obtain a displacement, and applies the sum of halo
displacements at each particle's **original** position. It does not alter velocities, masses, IDs or the
simulation dynamics. SimulationCode is not coupled to this module.

Everything is in `FML::BARYONIFICATION`. The implementation is header-only, requires C++17, and uses the existing
FML KD-tree for periodic halo searches. It needs no GSL, FFTW, or Lua. Optional OpenMP parallelizes particle
queries and application. The mathematical profile/transport layer (`Profiles.h`) also works independently of FML.

## Model and scientific scope

The supplied model uses the profile family of
[Schneider et al. (2019), arXiv:1810.08629](https://arxiv.org/abs/1810.08629), equations 6 and 10--19, and the
enclosed-mass transport introduced by
[Schneider & Teyssier (2015), arXiv:1510.06034](https://arxiv.org/abs/1510.06034).

With `x=r/R200`:

- Reference halo: truncated NFW, proportional to
  `1 / [(c*x)*(1+c*x)^2 * (1+(x/epsilon)^2)^2]`, normalized to the input `M(<R200)=M200`.
- Gas: proportional to `(1+x/theta_co)^(-beta) * (1+(x/theta_ej)^2)^(-(7-beta)/2)`,
  with `beta=3-(M_c/M200)^mu`. This single extended gas component describes both the central gas and its outskirts;
  it is the 2019 parametrization, not the separate bound/ejected gas decomposition of the 2015 model.
- Central galaxy: `rho proportional to r^-2 exp[-(r/(2*R_h))^2]`. Its cumulative mass uses `erf` analytically.
- Collisionless matter: dark matter plus satellite/diffuse stars. Optional relaxation solves
  `rf/ri - 1 = A * [(Mi/Mf)^n - 1]`, with `Mi=M_reference(ri)` and
  `Mf=f_collisionless*Mi + M_gas(rf) + M_central(rf)` by bracketed bisection.
- Stellar fractions: `f_stars=A_star*(M1/M200)^eta_total` and
  `f_central=A_star*(M1/M200)^eta_central`. Gas gets `f_b-f_stars`; collisionless matter gets the remainder.
  Invalid budgets, including extrapolating these stellar laws to unsuitable low halo masses, throw rather than
  silently clipping the fractions.

**This is a finite-support implementation of that profile family, not a reproduction of the paper's calibrated
power-spectrum predictions.** The following choices are explicit:

1. All halo components are normalized at `ProfileNumerics::maximum_radius * R200` (default `30 R200`), not infinity.
   The reference determines the finite total halo mass, which generally exceeds M200. Changing the support changes
   the component normalization; check support convergence for the desired feedback parameters.
2. The supplied common background is uniform, with enclosed mass `4*pi*rho_background*r^3/3` in source and target.
   The paper's correlated two-halo term is not included in the built-in model. Custom source/target cumulative
   profiles can be passed to `RadialMapping` to use another prescription.
3. Both profiles have exactly the same outer mass. Displacements are zero outside the support. There is no
   arbitrary taper applied to the interior transport.
4. Overlapping halos contribute additive vectors, evaluated at the unchanged input coordinates. This is
   independent of halo order up to summation round-off, but does not exactly recover each isolated target profile
   in overlapping regions. Supply a **unique host-halo catalogue**, not hosts plus their subhalos as duplicate
   representations of the same mass.
5. Profile integration and inversion use monotone piecewise-linear cumulative tables. Increasing `radial_bins`
   (default 2048 positive radii, plus the origin and R200) controls interpolation and quadrature errors. `minimum_radius`
   (default `1e-6 R200`) controls the unresolved central interval, where interpolation is linear in enclosed mass.

Model defaults are illustrative. Validate the correction against a chosen hydrodynamical calibration and against
halo completeness, halo-mass/concentration errors, snapshot resolution, support, and radial table resolution.
No percent-level accuracy claim is made. This is not a hydrodynamical solver and provides no corrected velocities,
thermodynamics, galaxy population, or modified-gravity calibration.

## Usage

```cpp
#include <FML/Baryonification/Baryonification.h>
using namespace FML::BARYONIFICATION;

// Positions, R200, and boxsize must ALL use the same length unit.
// Here: comoving Mpc/h, mass Msun/h, density (Msun/h)/(Mpc/h)^3.
HaloProperties halo{1e14, 0.7, 5.0}; // M200, R200, concentration R200/rs
ModelParameters model;
model.gas_mass_scale = 1e14;
model.gas_ejection_radius = 4.0;
const double mean_density = 8.3e10;
auto profiles = make_halo_profiles(halo, model, mean_density);

std::vector<HaloDisplacement> halos;
halos.push_back({{50.0, 50.0, 50.0}, profiles.mapping()});
Baryonifier correction(std::move(halos), 100.0);

// T has get_pos() and get_ndim()==3. Preserves every field except position.
auto shifts = correction.displacements(particles.data(), particles.size()); // non-mutating
correction.apply(particles.data(), particles.size());                       // mutating
```

For ordinary FML snapshots with positions in `[0,1)`, use `boxsize=1` and divide all catalogue positions **and
radii** by the physical box length. Scale background density to mass per box-unit cubed. M200 remains in the
chosen mass unit, as do `gas_mass_scale` and `stellar_mass_scale`. Using physical rather than comoving radii is
also supported when positions and density are converted consistently. There is no implicit h or scale-factor
conversion, concentration relation, mass-definition conversion, or redshift evolution; supply these externally.
For a CDM+baryon-only field in a massive-neutrino cosmology, choose mass fractions and background for that same
field; do not silently count the neutrino mass in it. The API does not distinguish species and should initially
be applied to a single total CDM+baryon particle representation.

`EnclosedMassProfile(radii, masses)` accepts custom monotone cumulative tables starting at `(0,0)`.
`RadialMapping(source, target)` requires matching outer radii and total masses and computes
`target.mass(r_new)=source.mass(r_old)`. Centers and exterior particles have zero displacement.

## MPI, periodicity and memory

The halo catalogue **and prepared mappings must include every relevant halo on each rank**. The module performs
no halo exchange and no collectives. Each rank can process its local particles; the result is independent of the
particle partition. Replicating the catalogue is a simple initial implementation, not a distributed large-catalogue
solution. A mapping stores four double arrays (about 64 KiB at the default table resolution per halo, plus the
KD-tree and halo metadata). Particle application also allocates a three-double displacement array per local particle.

After correcting an `MPIParticles<T>` container, call `communicate_particles()` to restore its domain ownership:

```cpp
correction.apply(part.get_particles_ptr(), part.get_npart());
part.communicate_particles(); // collective; positions must be in [0,1) for MPIParticles
```

Periodic distances use the nearest halo image. Each halo support must be **strictly smaller than half the box**;
larger supports are rejected. Particles are wrapped into `[0,boxsize)` after application. The nonperiodic constructor
option (`Baryonifier(halos, boxsize, false)`) performs neither image searches nor wrapping.

The field is read-only after construction and can be queried concurrently. All numerical and ordinary particle
input validation happens before modifying particle positions. IDs, masses and velocities are not touched.

## Example and tests

```
cd FML/Baryonification/example
make
./baryonify              # profiles.csv and original/corrected particles.csv

cd ../tests
make run                # serial
make clean
make run USE_MPI=true USE_OMP=true NP_LIST="1 2 4"
```

Compiler paths and flags use `FML/Makefile.config` and optional `Makefile.local`. Clean before changing build options.
The new example is discovered by the existing top-level `make examples`. The tests are also included in the
library's `make test` loop, including its 1/2/4-rank runs.

Tests cover analytic enclosed-mass transport, identity/zero-baryon limits, component conservation, radial convergence,
relaxation, expected inward/outward responses, a periodic analytic brute-force reference, halo overlap and order,
MPI particle partitions, preservation of particle attributes, empty inputs, malformed profiles, invalid budgets,
non-finite inputs and multiple-translation-unit inclusion. These verify the numerical implementation, not a
hydrodynamical calibration.
