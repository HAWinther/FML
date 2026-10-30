# Cosmological timestep coefficients

`FML/NBody/TimeStepping.h` provides Quinn, Tassev, FastPM and BullFrog coefficients
without depending on SimulationCode, a particle type, a mesh or a gravity-model class.
It extracts the existing SimulationCode policies; it does not introduce another force
solver. GSL/ODESolver is needed for the Quinn/Tassev quadratures.

```cpp
#include <FML/NBody/TimeStepping.h>
using namespace FML::NBODY;

// Einstein-de Sitter example: E(a) = H(a)/H0, D1 = a, D2 = -3/7 a^2.
FML::COSMOLOGY::GrowthHistory growth(
    [](double a) { return a; },
    [](double) { return 1.0; },
    [](double a) { return -3.0 / 7.0 * a * a; },
    [](double) { return 2.0; });
TimestepOptions options;
options.method = TimestepMethod::FastPM;
options.spacing = ScaleFactorSpacing::GrowthFactor;
TimeStepping stepping(options,
                      [](double a) { return std::pow(a, -1.5); },
                      [](double a) { return 1.5 * a; },
                      growth);
auto steps = stepping.steps(0.02, 1.0, 10);
```

For each `Step`, positions initially correspond to `apos` and velocities to
`avel`. Evaluate the force at `apos`, apply

```
v = velocity_scale * v + delta_time_kick * force
x = x + delta_time_drift * v
```

and track the new position/velocity epochs `apos_new`/`avel_new`. Quinn, Tassev and
FastPM use KDK staggering, including a final kick-only synchronization. BullFrog
uses DKD staggering, beginning with a drift-only entry; its adjacent half-drifts
are combined between kicks. The sequence therefore contains `nsteps + 1` entries.
An entry with zero kick needs no force evaluation. A zero requested step count
returns an empty sequence.

Coordinates are comoving (usually in box units). The velocity convention is
`v = a^3 E(a) dx/da = a^2 dx/dt / H0`. The force callback supplies the time amplitude
of the force actually returned by the caller, e.g. `3/2 OmegaM mu(a) a` for the
SimulationCode convention. The force and growth normalizations must be consistent;
passing a physical acceleration with different units is not interchangeable.
This object generates coefficients; it does not move particles, handle COLA
residual forces, or cache endpoint force grids.

`scalefactors_KDK` exposes the KDK schedule. `deltatime_KDK` exposes the existing
Quinn/Tassev quadratures; use `steps` for complete FastPM/BullFrog coefficients.
All methods support linear, logarithmic, powerlaw and linear-growth-factor step
spacing. Growth-time inversion assumes continuously increasing D1 on the supplied
interval; endpoint checks reject decreasing/constant growth. The caller must
supply finite functions over the full requested interval. FastPM/BullFrog also
require nonzero linear momentum and force growth factors.

## Shared growth history

`FML/LPT/GrowthHistory.h` accepts independent functions for `D1(a)`, `f1(a)`,
`D2(a)` and `f2(a)`, where `fi = (a/Di) dDi/da`. D2 is signed; its rate does not
mean taking the real logarithm of a negative amplitude. FastPM only needs D1/f1;
BullFrog also needs D2/f2. Quinn/Tassev with nongrowth spacing need no growth
callbacks. Missing functions are reported when requested.

The interface exposes linear momentum `a^2 E f1 D1` and `dD2/dD1 = f2 D2/(f1 D1)`.
Growth amplitudes must share a normalization: scaling D1 by c requires scaling
D2 by c^2 and consistently normalizing the spatial LPT fields.

With GSL, use `growth_history_from_log_a_splines(D1, D2)` on the splines returned by
`compute_LPT_growth_factors_LCDM` or `compute_LPT_growth_factors_GeffGLCDM`.
The adapter owns copies of the spline data and obtains rates from their log(a)
derivatives. It does not extend their valid time range or change their initial
conditions. This also lets a model-independent LPT calculation supply timestep
coefficients without constructing SimulationCode cosmology/gravity objects.

These are scale-independent time amplitudes. They do not supply scale-dependent
nonlinear LPT kernels, species transfer functions or gauge transformations.
BullFrog retains the existing single-fluid 2LPT policy and its model assumptions;
this extraction is not a validation for arbitrary modified gravity or multi-fluid
models. SimulationCode's explicitly named `large_scale_growth_history` adapter keeps
its existing k=0 growth convention for scale-dependent models. Model callbacks
capture shared ownership so they remain alive for the coefficient generator.
