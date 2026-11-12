# HaloModel

Halo-model predictions for the matter power spectrum P(k, z), the mass function n(M, z) and the halo bias. All the
halo models in FML live here and share one set of building blocks.

| File | Contents |
|---|---|
| `HaloModelTools.h` | The building blocks: the variance sigma(R) and sigma_v (`LinearVariance`), the top-hat window, the NFW profile and its Fourier transform, Sheth–Tormen (and Tinker 2008, Watson 2013 in `MassFunctionFits.h`), Bullock et al. concentrations, the Mead (2017) and HMcode-2016 collapse fits, the HMcode transition |
| `Halomodel.h` | `HaloModel`: the general halo model (`Prescription::Standard`, deltac and DeltaVir from spherical collapse, also with modified gravity through mu(a)) and **HMcode-2016** (`Prescription::HMcode2016`) |
| `HMcode2016.h` | `HMcode2016`: HMcode-2016 with the interface of `HMcode2020` (a P_NL/P_L table and a one-redshift spectrum from a `BackgroundCosmology`), used by the solver (`nonlinear = hmcode2016`) |
| `HMcode2020.h` | `HMcode2020`: HMcode-2020 (Mead et al. 2021), used by the Einstein–Boltzmann solver (`nonlinear = hmcode2020`, `react`) |
| `HaloModelReaction.h` | `HaloModelReaction`: the halo-model reaction R(k) of ReACT (Cataneo et al. 2019; Bose et al. 2020, 2021) at one redshift: real (screened spherical collapse with a fifth force, `SphericalCollapse/ScreenedSphericalCollapse.h`) and pseudo one-halo terms, E and k* (from given one-loop values), massive neutrinos |
| `HaloModelBackground.h` | `cosmology_from_background`: the halo-model input from an FML `BackgroundCosmology` and the solver's P(k) (header only) |
| `PythonWrapper/` | A pybind11 module for `HaloModel` |

`AxionHMcode/` (the mixed cold/ultralight-axion halo model) uses the same building blocks.

Units: k in h/Mpc, R in Mpc/h, M in Msun/h, P(k) in (Mpc/h)^3, n in (h/Mpc)^3. x = ln a.

## The general halo model and HMcode-2016

```cpp
#include <FML/HaloModel/Halomodel.h>
using namespace FML::COSMOLOGY::HALOMODEL;

// The expansion history (and mu(x) for modified gravity) for the growth and spherical collapse, and the
// linear P(k, z = 0) in a file, scaled with the growth factor of the model
SphericalCollapseModel collapse(E, OmegaM_of_x, dlogE_dx, mu, w);
auto cosmology = cosmology_from_pofk_file(collapse, "pofk_lin.txt");

HaloModelOptions options;
options.prescription = Prescription::HMcode2016;   // or Standard
HaloModel hm(cosmology, options);
hm.compute_at_redshift(0.5);
double P = hm.pofk(1.0), P1h = hm.pofk_onehalo(1.0), n = hm.dndlogM(1e14);
const auto & s = hm.state();                       // deltac, DeltaVir, sigma8, neff, kstar, alpha, ...
```

With FML's Einstein–Boltzmann solver (scale-dependent growth and massive neutrinos from the solver):

```cpp
#include <FML/HaloModel/HaloModelBackground.h>
auto cosmology = cosmology_from_background(bg,
    [&](double x, double k) { return power.get_matter_power_spectrum(x, k); },     // (x, k [1/m]) -> m^3
    [&](double x, double k) { return power.get_matter_power_spectrum_cb(x, k); });
```

`HaloModelCosmology` is just the collapse model, Omega_m, f_nu and the callbacks P_m(x, k) and P_cb(x, k), so any
other source of linear spectra works too.

### Options

- `prescription`: `Standard` (deltac and DeltaVir from spherical collapse in the collapse model) or `HMcode2016`.
- Mass function and bias: Sheth–Tormen with `sheth_tormen` (a, p; the normalisation is exact), or any
  `multiplicity(nu, hm)` and `bias(nu, hm)`. The halo integrals are divided by int f dnu and int b f dnu, so a mass
  function that is not normalised does not change P(k).
- Profile: NFW with the analytic transform and `concentration_model` (`Bullock2001` with `bullock_A` and
  `bullock_mass_fraction`, or `PowerLaw`), or any `concentration(M, hm)`, or any real-space
  `density_profile(r, M, hm)` (4 pi r^3 rho / M, transformed numerically).
- `deltac_multiplier`, `DeltaVir_multiplier`, `concentration_multiplier` (also for HMcode-2016).
- HMcode-like modifications of the Standard model: `kstar` (one-halo damping, 0.01 h/Mpc by default), `alpha`
  (transition), `eta` (bloating), `f_damp` (two-halo damping). HMcode-2016 sets its own values at each redshift in
  `state()`; it never changes the options.
- Ranges and accuracy: `kmin`, `kmax`, `npts_k` (the P(k) splines; outside this range `pofk` throws),
  `accuracy_boost`, the spherical-collapse settings (`zmax_spherical_collapse`: Standard only works up to this z).

Change options with `set_options` (re-runs the growth and spherical collapse), then `compute_at_redshift` again.

### Massive neutrinos

The peak heights, mass function and concentrations use the cold-matter variance (P_cb), the masses the total matter
density, and the two-halo term the total-matter P(k). HMcode-2016 follows CAMB's `mead2016`: deltac and DeltaVir
have the Mead et al. (2016) neutrino factors, sigma8 and sigma_v in the fits are of the total matter, n_eff of the
cold matter, and the one-halo term has no (1 - f_nu)^2 factor. The Standard model multiplies the one-halo term by
(1 - f_nu)^2 (neutrinos do not cluster in haloes), as HMcode-2020.

### HMcode-2016 details

As in CAMB (halofit_version = mead2016): the fits of Mead et al. (2015) for deltac, DeltaVir, eta, f, k* and A =
3.13, alpha = 3.24 1.85^n_eff clamped to [0.5, 2], and the Dolag et al. (2004) correction of A with the power 1.5,
A (D(z=10)/D_LCDM(z=10))^1.5 (D normalised today, the GR growth of the model against LCDM with the same Omega_m).

## Halo-model reaction

```cpp
#include <FML/HaloModel/HaloModelReaction.h>
using namespace FML::COSMOLOGY::HALOMODEL;
HaloModelReaction::Cosmology c;                        // OmegaM (total), OmegaNu, target E(x) and dlogE/dx
c.OmegaM = 0.317;
HaloModelReaction::LinearSpectra S;                    // at the redshift: P (target total), Pcb, Pcb_lcdm
S.P = ...; S.Pcb = ...; S.Pcb_lcdm = ...;              // k [h/Mpc] -> P [(Mpc/h)^3]
auto F = [](double R_th) { return SPHERICALCOLLAPSE::fifth_force_hu_sawicki(1e-5, 1.0, 0.317, R_th); };
HaloModelReaction hm(c, a, S, F);                      // no F: GR / dark energy (E = 1)
hm.set_one_loop(P_spt_cb(k0), P_spt_pseudo(k0));       // modified gravity: k* from one-loop P at k0 = 0.06 h/Mpc
double R = hm.reaction(k);                             // P_NL = R P_NL,pseudo
```

The real halos collapse in the target background with the fifth force (mass and environment dependent); delta_c(M)
is extrapolated with the LCDM growth and nu = delta_c / sigma uses the LCDM cold spectrum, as in ReACT. The pseudo
halos are GR halos of the target P(k) in an LCDM background. `ReactionTable` tabulates R(z, k) for the solver
(`nonlinear = react`, `Cosmology/Spectra/PowerSpectrumReACT.cpp`). The one-loop P(k0) for k* comes from
`PerturbationTheory/MGOneLoop.h` (kernels solved in time with the model's mu, gamma2, gamma3; all redshifts in one
call, ~1 s). About 1 s per redshift for f(R) (30 collapses), 0.1 s for GR or dark energy.

## Validation

- `Cosmology/EinsteinBoltzmannSolver/tests/test_hmcode2016`: HMcode-2016 against CAMB 2.0.5 `mead2016` with the same
  linear spectra for LCDM, sum(m_nu) = 0.3 eV and w0 = -0.8, wa = 0.2, Omega_K = 0.02 at z = 0, 1, 2 and 31 k from
  0.01 to 10 h/Mpc: maximum differences 0.060% (LCDM), 0.030% (neutrinos), 0.050% (w0wa). Fixtures:
  `python/generate_hmcode2016_reference.py`.
- `Cosmology/EinsteinBoltzmannSolver/tests/test_hmcode2020`: HMcode-2020 against CAMB 1.5.4 `mead2020` (0.075%).
- `tests/test_halomodel_reaction`: `HaloModelReaction` against ReACT (ACTio-ReACTio a36dda02, `HALO.cpp` through
  `tests/reference/react/tools/halomodel_oracle.cpp`) for GR, f(R) (|fR0| = 1e-5), nDGP (Omega_rc = 0.1) and CPL with 0.3 eV
  neutrinos at z = 0 - 2: delta_c(M) and sigma(M) to 5e-5, E to 3e-5, k* to 3e-3 (with ReACT's one-loop values), the
  one-halo terms to 2e-3 and R to 1e-3 (k = 0.01 - 10 h/Mpc). Fully native (k* from `MGOneLoop.h`, with ReACT's
  f(R) gamma3): R to 2e-4 (f(R)) and 6e-4 (nDGP), k* to 3%. ReACT's f(R) gamma3 is 1/3 of the correct one; with the
  correct one R changes by up to 3e-3. With identical real and pseudo cosmologies R = 1 exactly (ReACT: within its
  numerical noise).
- `tests/test_screened_collapse`: the screened spherical collapse against ReACT (`SCOL.cpp`).
- `tests/test_halomodel`: the HMcode output in `example/`, the mass function against a direct computation, the
  profile transforms, the building blocks, and the error handling.

## References

Sheth & Tormen (1999); Bullock et al. (2001); Cooray & Sheth (2002); Dolag et al. (2004);
[Mead et al. (2015)](https://arxiv.org/abs/1505.07833); [Mead et al. (2016)](https://arxiv.org/abs/1602.02154);
[Mead (2017)](https://arxiv.org/abs/1606.05345); [Mead et al. (2021), HMcode-2020](https://arxiv.org/abs/2009.01858).
