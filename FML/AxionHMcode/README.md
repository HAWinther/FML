# Native AxionHMcode

Standalone C++17 implementation of the mixed cold/ultralight-axion halo model of
[Vogt et al. (2022)](https://arxiv.org/abs/2209.13445), with the
[Dome et al. (2024)](https://arxiv.org/abs/2409.11469) mass-relation and smoothing calibration.
The module lives entirely in this folder. It uses GSL for special functions and the central-density
solve, and has its own example and tests. It has no runtime Python, axionCAMB or Boltzmann-solver dependency.

Supply **separate linear cold and axion auto-spectra at the requested redshift**, for the same
cosmology. Cold means CDM+baryons, defined relative to their own mean density. Suitable inputs
can be generated with axionCAMB or another ULA Boltzmann solver. FML's axion-like early-dark-energy
model describes different physics and does not supply these spectra.

## Build and run

From the repository root:

```sh
make -C FML/AxionHMcode/tests run
make -C FML/AxionHMcode/example
FML/AxionHMcode/example/example \
  FML/AxionHMcode/tests/reference/linear.txt /tmp/axion_nonlinear.txt \
  dome 0.67 0.05 0.2353 0.0317 3.162277660168379e-25 2 10
```

The example arguments are `input output basic|dome h Omega_b Omega_cdm Omega_axion mass_eV z k_max
[allow_extrapolation]`. Input columns: `k P_cold P_axion`; comment lines start with `#`.
The output contains `k P_linear_total P_total P_cold P_cold_axion P_axion`, and records whether
the calibration was extrapolated. All k are **h/Mpc**, powers are **(Mpc/h)^3**, and halo masses
are **Msun/h**. `k_max` selects output samples; the complete input spectrum is retained for integrals.

A separate CMake build is also available:

```sh
cmake -S FML/AxionHMcode -B /tmp/axionhm-build -DCMAKE_BUILD_TYPE=Release
cmake --build /tmp/axionhm-build -j4
ctest --test-dir /tmp/axionhm-build --output-on-failure
```

Use `GSL_ROOT_DIR` with CMake, or `GSL_INCLUDE` and `GSL_LIB` with Make, for a custom GSL install.
For older MacPorts GCC with a newer Xcode SDK, `-DCMAKE_OSX_SYSROOT=` avoids incompatible SDK
header overrides. No changes to the rest of FML's build configuration are needed.

## API

```cpp
#include <FML/AxionHMcode/AxionHMcode.h>
using A = FML::COSMOLOGY::AxionHMcode;
A::Cosmology c;
c.h = 0.67;
c.Omega_b = 0.05;
c.Omega_cdm = 0.2353;
c.Omega_axion = 0.0317;
c.mass_eV = 3.162277660168379e-25;
c.z = 2;
A::Input linear; // fill k, cold, axion at c.z, in the units above
A::Options options;
auto result = A::spectrum(c, linear, {0.01, 0.1, 1.0, 10.0}, options);
```

`Omega_cdm` excludes axions. The matter fraction used by the Dome calibration is
`f = Omega_axion/(Omega_b+Omega_cdm+Omega_axion)`, whereas axionCAMB's commonly used dark-matter
fraction is `Omega_axion/(Omega_cdm+Omega_axion)`. These fractions are different.

`Spectrum` returns nonlinear cold, cold--axion cross, axion, and total power, with

    P_total = (1-f)^2 P_cold + 2f(1-f) P_cold_axion + f^2 P_axion.

It also returns the reconstructed linear total, physical component one-halo terms, collapse
thresholds, cutoff mass, clustered fraction, smoothing parameters and per-halo diagnostics.
The one-halo component terms do not contain inverse clustered fractions, so a vanishing clustered
component is handled without division by zero. Zero axion density gives exactly the configured
cold halo model, with zero axion/cross spectra and no axion-profile calculations.

## Physics and options

The calculation includes cold-field variances and Sheth--Tormen abundance/bias, Bullock
formation concentrations, NFW Fourier profiles, a soliton-plus-NFW axion density profile,
the halo Jeans cutoff, clustered/unclustered axion contributions and cold--axion cross power.
The Dome variant adds its tabulated beta2(z,f), broken axion-mass/cold-mass relation, modified
collapse parameters and separate cold/cross transition fits, including the scale-dependent
large-scale floor of the cold smoothing parameter.

| Option | Default | Meaning |
|---|---|---|
| `version` | `Dome2024` | `Basic` selects the original mass assignment and collapse prescription |
| `transition` | `Recommended` | Dome uses calibrated smoothing; Basic uses an unsmoothed sum. `Sum` or `Smoothed` overrides this |
| `one_halo_damping` | true | Suppress the unphysical large-scale one-halo constant |
| `two_halo_damping` | false | Optional HMcode two-halo suppression |
| `halo_bloating` | false | Optional nu-dependent cold halo bloating |
| `full_two_halo` | false | Use the full bias/profile integrals with the unresolved-low-mass correction |
| `concentration_amplitude` | 4 | Minimum concentration; matches the pinned reference example with `concentration_param=False` |
| `mass_min`, `mass_max` | 10^7, 10^18 | Cold host mass bounds in Msun/h |
| `n_mass`, `n_radius` | 257, 2000 | Mass and radial integration nodes |
| `allow_extrapolation` | false | Permit extrapolation of the Dome redshift/fraction/mass calibration |
| `reference_density_cut` | true | Preserve the reference central-amplitude rejection criterion |
| `jeans_density` | `ReferenceCode` | Density convention used only in the Dome axion-mass relation, described below |

The calibrated Dome setup uses the simple two-halo term (`full_two_halo=false`); optional HMcode
bloating/damping and full two-halo integrals are numerical features, not new simulation calibrations.
The default cold prescription includes selected HMcode ingredients. Matching every HMcode-2020
feature, such as explicit BAO dewiggling, is outside this module's mixed-axion prescription.
The optional Dentler alpha_1, alpha_2, gamma_1, gamma_2 nuisance modifications in the reference
repository are not implemented.

## Domain and conventions

This implementation assumes a flat, late-time LCDM expansion with cold matter plus axions and
no massive neutrinos or baryonic feedback. It reconstructs cold--axion large-scale coherence
from the two auto-spectra as in the reference model; a general independent signed cross-transfer
or isocurvature mode is outside this prescription.

The Dome beta2 table covers z=1..8 and total-matter axion fractions f=0.01..0.3, with calibration
centered on m=10^-24.5 eV. By default the module enforces those bounds and the pivot mass.
Other masses or redshifts require `allow_extrapolation=true`; `Spectrum::extrapolated` records
this choice. A mass-dependent fitting formula is available in the model, but its evaluation
away from the calibration mass is still an extrapolation. Basic has no simulation-calibration
claim over a broad parameter space.

Linear k input must be strictly increasing, with at least 32 samples, cover k<=10^-4 through
k>=100 h/Mpc, and contain finite positive cold power and nonnegative axion power. Dense sampling
and convergence checks are needed in practice. The checked-in physical fixture has 1001 nodes
over 10^-5..10^3 h/Mpc, with a P~k^ns continuation below axionCAMB's first transfer sample.
Output k must lie inside the supplied range; no linear extrapolation is
performed. Numerical output at large k does not establish physical calibration there.

Two reference-code conventions are explicit:

1. **Jeans density.** The pinned Python implementation passes physical density
   `omega_m=Omega_m*h^2` into a Jeans-mass function whose formula takes `Omega_m` and multiplies
   by h^2 internally. `JeansDensity::ReferenceCode` reproduces that call, enabling faithful
   reference comparisons. `JeansDensity::OmegaMatter` consistently evaluates the formula with
   Omega_m. Changing this convention changes the prediction and has not been recalibrated to simulations.
2. **Central density.** The reference hybrid solve can stagnate because the last-intersection
   profile construction has plateaus/discontinuities. Its Fourier profile is normalized by its
   actual integrated mass, while the one-halo weights use the target axion mass relation. The native
   implementation follows that shape calculation and makes the final density normalization explicit:
   `profile_rescaling` converts the integrated profile to its target mass. `raw_profile_mass_error`
   records the pre-rescaling mismatch; `normalization_error` records the final normalization error.
   The initial-density guess and reference rejection `|amplitude-guess|>100` are also exposed.
   Rejected axion hosts remain in the cold term and are listed in `rejected_axion_hosts`.

Changing either convention or the reference rejection rule is a model variation, requiring its
own validation. The default reproduces the pinned public implementation's observable spectra.

## Validation

The tests compare all four nonlinear component spectra with
[axionHMcode](https://github.com/SophieMLV/axionHMcode) revision
`a85ba2679cba5abe68b4cfd1e49b7a52bdcda424`. Physical linear spectra were generated with
[axionCAMB](https://github.com/dgrin1/axionCAMB) revision
`891e779cc0bd422e49f97533e6c2fc761149737d`, for the cosmology in the API example.
The reference comparison uses 65 masses and 2000 radial nodes, for Basic and Dome, over
k=0.001..10 h/Mpc. The observed largest component difference is about 1.4e-6 relative;
the regression tolerance is 1e-4 (0.01%). Cutoff mass, clustered fraction, transition fits,
component reconstruction, profile normalization, calibration rejection and the exact zero-axion
limit are checked separately. At the same cosmology, doubling the production defaults from
257 to 513 mass nodes and 2000 to 4000 radius nodes changes total power by about 4.1e-5 relative
at the tested k values. This is numerical/reference validation at one cosmology, not an independent
N-body accuracy claim or a full parameter-space convergence survey.

The stock Python reference can emit warnings for fractional powers of negative cross one-halo
terms at larger k. Only finite positive reference outputs within the stated test range are retained.

Python is needed only to regenerate the fixtures:

```sh
python3 -m venv /tmp/axion-reference-env
/tmp/axion-reference-env/bin/pip install -r FML/AxionHMcode/tests/reference-requirements.txt
OMP_NUM_THREADS=1 /tmp/axion-reference-env/bin/python FML/AxionHMcode/tests/generate_reference.py \
  --reference-source /path/to/pinned/axionHMcode \
  --axioncamb /path/to/axionCAMB/camb --work-dir /tmp/axion-reference-work
```

Use `--reuse-linear` to keep the checked-in physical input and regenerate only the nonlinear
reference values. The script calls the reference Python implementation directly and does not
use the native C++ code. Stock axionCAMB was built serially with gfortran, `-O2
-ffree-line-length-none -fallow-argument-mismatch -fmax-stack-var-size=16384`; this avoids the
large automatic-stack allocations encountered with OpenMP in that older Fortran code.
