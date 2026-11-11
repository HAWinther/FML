# Nonlinear matter power

The Einstein--Boltzmann solver accepts:

| `nonlinear` | Method |
|---|---|
| `none` | Linear power (default) |
| `halofit` | Existing Takahashi et al. (2012) Halofit |
| `halofit_bird` | Takahashi Halofit plus the updated Bird neutrino correction used by CLASS |
| `hmcode2016` | HMcode-2016 (as CAMB `mead2016`), including massive neutrinos |
| `hmcode2020` | Gravity-only HMcode-2020, including massive neutrinos |
| `react` | Optional official ReACT reaction times an HMcode-2020 matched GR pseudo spectrum |

For example, from `FML/Cosmology/EinsteinBoltzmannSolver`:

```sh
./cmb preset=planck2018_baseline nonlinear=halofit_bird compute_cells=false pert_k_max=1
./cmb preset=planck2018_baseline nonlinear=hmcode2020 compute_cells=false pert_k_max=1
```

All nonlinear methods add total-matter nonlinear power as the fourth column of `matter_power*.txt`.
The third column remains **linear** CDM+baryon power. With `lensing=true`, the same nonlinear ratio
multiplies the Weyl-potential power in the Limber part of the lensing-potential calculation. This is the
existing matter-to-Weyl prescription; no additional modified-gravity lensing relation is implied.

`react` requires an optional backend and has separate model and redshift limits. See
[ReACT build, API and scientific conventions](../../ReACT/README.md). Above its reaction range
(default z=2.5), the solver uses the matched pseudo baseline with R=1 as an explicit approximation.

## Bird extension

`Halofit::Version::Bird` selects the Takahashi-compatible Bird correction. The original constructor
default and `nonlinear=halofit` remain unchanged. For the one-redshift `Halofit::nonlinear` method,
the optional `f_nu` argument activates the correction; `h` is required to convert the input k grid
in 1/Mpc to the h/Mpc units of the quasi-linear correction. With `f_nu=0` this is exactly the
existing Takahashi result.

The correction changes beta, the rational part of the quasi-linear term, and the one-halo amplitude.
It omits the obsolete Omega_m-dependent neutrino amplitude term, following current CLASS.
The neutrino fraction is the present-day massive-neutrino density divided by total matter density.
Neutrino hierarchies and scale-dependent linear growth come from the supplied linear spectrum.

## HMcode-2020 library interface

`FML/HaloModel/HMcode2020.h` provides `FML::COSMOLOGY::HALOMODEL::HMcode2020`. It lives in `FML/HaloModel` with the
general halo model, which has HMcode-2016 as `Prescription::HMcode2016`, and both use the building blocks in
`HaloModelTools.h` (see [HaloModel/README.md](../../HaloModel/README.md)). `Cosmology/Spectra/HMcode.h` (the class
`FML::COSMOLOGY::HMcode`) still works but only forwards to it and will be removed.

`FML/HaloModel/HMcode2016.h` (`HALOMODEL::HMcode2016`, `nonlinear = hmcode2016`) has the same interface (table
constructor, `ratio`, one-redshift `spectrum`). It runs the general `HaloModel` with `Prescription::HMcode2016` at
each of the `n_z` redshifts (about 0.1 s each, so a few seconds for the default table) and accepts the same
backgrounds (lcdm, w0wa).

```cpp
using FML::COSMOLOGY::HALOMODEL::HMcode2020;
HMcode2020 hm(bg, P_total, P_cb);
double P_nonlinear = P_total(x, k) * hm.ratio(x, k);
auto terms = HMcode2020::spectrum(bg, P_total, P_cb, x, {0.1, 1.0, 10.0});
```

Both callbacks take `(ln(a), k[1/m])` and return power in m^3, for any positive k required by the
integrals. `P_cb` is defined relative to the mean CDM+baryon density. The single-redshift
`spectrum` method takes output k in **h/Mpc** and returns linear, one-halo, two-halo and total
nonlinear powers in **(Mpc/h)^3**, plus collapse and variance diagnostics.

The implementation includes:

- Mead (2017) growth-history-dependent collapse thresholds, including neutrino corrections;
- CDM+baryon variances for halo abundance, effective slope, bloating and damping parameters;
- Sheth--Tormen abundance and normalized NFW Fourier profiles with Bullock formation times;
- the redshift-dependent Dolag concentration correction;
- removal of unclustered neutrino mass from the one-halo term;
- BAO damping using Gaussian smoothing of P/P_EH in ln k;
- the 2020 one-halo cutoff, two-halo damping and smoothed transition.

Auxiliary collapse/concentration growth excludes radiation and treats neutrinos as matter, as in
HMcode. Actual linear spectra are evaluated separately at every requested time and retain
scale-dependent growth. The supported backgrounds are `lcdm` and smooth `w0wa` (including
curvature). Other models are rejected: replacing the linear spectrum does not specify screening,
interactions, axion profiles or other nonlinear physics. The baryonic-feedback variant
`HMcode2020_feedback` is not included.

Defaults: 256 masses from 1 to 10^18 Msun/h; logarithmic variance integration over
10^-5 to 10^7 h/Mpc with dln(k)=0.015; 512 output k points from 10^-5 to 100 h/Mpc;
60 time points from z=10 to 0. `HMcode::Options` controls the resolutions. The nonlinear
ratio is one above z_max or below k_min; it is held at the endpoint above k_max and for a>1.
These are numerical boundaries, not claims of physical calibration over the entire table.
The HMcode-2020 paper reports 2.5% RMS against simulations for k<10 h/Mpc and z<2,
with larger errors for some cosmologies. Reference-code agreement below does not reduce this
physical modelling uncertainty.

The solver extrapolates total and cold linear spectra beyond `pert_k_max` using the existing
Eisenstein--Hu broadband shape anchored to each computed spectrum. Use a sufficiently large
`pert_k_max` (at least 1/Mpc for precision work) and check convergence; the nonlinear integrals
need more k coverage than the requested output range.

## Validation

From `FML/Cosmology/EinsteinBoltzmannSolver`:

```sh
make -C tests test_hmcode2020 test_hmcode2016 test_nonlinear test_parameterfile USE_OMP=false
./tests/test_hmcode2020
./tests/test_hmcode2016
./tests/test_nonlinear
./tests/test_parameterfile
```

`test_hmcode2020` uses checked-in linear spectra and nonlinear references generated from the
same external runs, avoiding differences between Boltzmann solvers. HMcode is checked against
CAMB 1.5.4 `mead2020` for LCDM, sum(mnu)=0.3 eV, and w0=-0.8, wa=0.2, Omega_K=0.02 at
z=0,1,2 and 31 logarithmic k values from 0.01 to 10 h/Mpc. The tolerance is 0.1%; the measured
maximum difference is 0.075%. Doubling both mass and integration resolutions changes the LCDM
z=0 result by 0.036%. Bird agrees with CLASS (classy 3.2.0) within 0.051% at the same redshifts
and k points, using identical linear input. The test also checks the exact massless-neutrino limit.
`test_hmcode2016` checks HMcode-2016 (FML/HaloModel, built from the background with `HaloModelBackground.h`) against
CAMB 2.0.5 `mead2016` for the same three cosmologies (0.1% tolerance, at most 0.060%); fixtures from
`python/generate_hmcode2016_reference.py`. `test_nonlinear` exercises the solver's tables and output.

To regenerate the fixtures (requires CAMB, classy and NumPy):

```sh
OMP_NUM_THREADS=1 python3 python/generate_nonlinear_reference.py
```

References: [Takahashi et al.](https://arxiv.org/abs/1208.2701),
[Bird et al.](https://arxiv.org/abs/1109.4416), [Mead (2017)](https://arxiv.org/abs/1606.05345),
[HMcode-2020](https://arxiv.org/abs/2009.01858),
[CAMB implementation](https://github.com/cmbant/CAMB/blob/master/fortran/halofit.f90),
[CLASS implementation](https://github.com/lesgourg/class_public/blob/master/external/Halofit/halofit.c).
