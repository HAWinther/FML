# Constants and units

`Units.h` is the common source of physical constants and unit conversions:

```cpp
#include <FML/Units/Units.h>

// Compile-time SI values; no object, initialization, MPI or external library is required.
static_assert(FML::UTILS::SI::c == 299792458.0, "speed of light in m/s");
constexpr double hubble_time_seconds = 1.0 / FML::UTILS::SI::H0_over_h;

// Physical constants and conversion factors in a selected runtime unit system.
FML::UTILS::ConstantsAndUnits units("Cosmology"); // Mpc, Hubble time, solar masses, kelvin
double electron_energy = units.m_e * units.c * units.c;
double electron_energy_ev = electron_energy / units.eV;
```

`FML::UTILS::SI` contains metre/second/kg/kelvin/coulomb units, commonly used derived and astronomical units,
fundamental constants, recombination energies and rates, and `H0_over_h` in inverse seconds. `epsilon_0`, `xhi0`
and `xhi1` are **energies in joules**, not numeric electronvolt values. `alpha` is dimensionless.

`ConstantsAndUnits` supports SI, Planck, ParticlePhysics, Cosmology, and user-specified unit factors. Its `m`, `s`,
`kg`, `K`, and `Co` describe the value of one SI unit in the chosen system; all derived constants are computed
from the shared SI table. Constructing another unit system never changes the SI table.

The cosmology solver's existing `FML::COSMOLOGY::Constants` namespace is an alias to `FML::UTILS::SI`, exposed by
`Cosmology/Common.h`. Existing `Constants::...` and `using namespace Constants` callers continue to work.
Numerical implementations live in shared Math, Spline, FileUtils and Timing modules. Cosmology aliases remain
in `Common.h`; including `Units.h`
alone does not pull in GSL, splines or the cosmology solver.

The consolidation retains all values previously used by the cosmology solver. `ConstantsAndUnits` now uses those
same values for Boltzmann's constant, eV, electron mass, Thomson cross section, and parsec instead of its older
rounded values. Its year is now the same **365.25-day Julian year** as the cosmology solver (previously 365.24 days).
Ionization energies inherit the common eV conversion. Existing calculations using the runtime unit class may
therefore change slightly; the cosmology compile-time constants are unchanged.

This standalone header continues to support C++14. In C++17 and later the constants are inline constexpr variables
and have a single address across translation units. Tests for constants, conversions, unit systems and linkage
are included in `FML/tests/test_utils.cpp` and `test_units_odr.cpp`.
