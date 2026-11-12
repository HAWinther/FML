# Correlation-function multipoles

`CorrelationMultipoles.h` adds a shared, reusable conversion of power-spectrum multipoles to **xi_0(s), xi_2(s), xi_4(s)**. It consumes the existing LPT/EPT, TNS and EFTofLSS predictions; it does not recompute loops or change their bias, growth or IR-resummation conventions.

For the declared finite input band, the output is

```
xi_ell(s) = i^ell / (2 pi^2) integral[kmin,kmax] dk k^2 P_ell(k) W(k) j_ell(ks)
```

The real even-multipole phases are **+1, −1, +1**. P has length^3 units, k inverse-length and s length, so xi is dimensionless. The convention agrees with the Fourier-to-correlation operation in [velocileptors' LPT implementation](https://github.com/sfschen/velocileptors/blob/master/velocileptors/LPT/lpt_rsd_fftw.py). This layer uses the existing `FFTLog/SphericalBesselTransform.h`; the normalization and phase are also tested against analytic Gaussian transforms and independent direct quadrature.

## API

```cpp
#include <FML/PerturbationTheory/CorrelationMultipoles.h>
using namespace FML::PERTURBATION;
CorrelationOptions options;
options.fft_samples = 8192;       // default 4096; even, >=64, <=2^20
options.taper_fraction = 0.1;     // default: smooth EACH log-k endpoint over 10% of the band
options.damping_radius = 4.;     // example explicit exp[-(k R)^2] regulator; default R=0
CorrelationWorkspace workspace(k_output, separations, options);

// LPTRedshiftSpace::Multipoles / EPTRedshiftSpace::Multipoles:
auto xi = workspace.evaluate_spectrum(power);
// xi.s, xi.xi[0], xi.xi[1], xi.xi[2] correspond to ell=0,2,4.

// Or the array<vector<double>,3> returned by TNS / EFTofLSS:
auto xi_again = workspace.evaluate(power_array);
// Also: workspace.evaluate(p0, p2, p4).
```

The input k grid and requested separations must be finite, positive and strictly increasing. At least four input k samples and one separation are required. The k grid can be nonuniform; **signed** multipoles are interpolated linearly in their value versus ln k, not in ln P. Zeros and sign crossings are retained. Result vectors must have the same k grid as the workspace. `evaluate_spectrum` checks it explicitly; the array/vector overloads assume the caller uses the constructor's k grid and validate sizes/finite values.

Output uses a single caller-specified separation grid for all three multipoles. Values are interpolated linearly versus ln s from each Bessel order's own native FFTLog grid. Separation extrapolation is rejected. A workspace owns its fixed FFT plans, kernels, input/output interpolation maps, window weights and two reusable scratch vectors. Further evaluations reuse that setup; `evaluations()` counts successful calls. Copying is disabled; moving is supported, and evaluation of a moved-from workspace throws.

## Finite-band and ultraviolet choices

There is **no extrapolation outside the supplied k interval**, including no implicit power-law extension of perturbation theory beyond its usable range. With endpoint taper fraction t>0, the factor on either end is `u − sin(2 pi u)/(2 pi)` for u=0..1 across that fraction of the total ln-k span, and one in the interior. Allowed t is 0..0.5; t=0 disables tapering. An optional positive damping radius multiplies by `exp[-(k R)^2]`. These factors multiply and are applied equally to all multipoles.

These are deliberate filters of the supplied spectrum, not numerical guarantees for the infinite-band, unfiltered correlation function. Changing them changes the observable. An abrupt endpoint with taper disabled can ring; a taper extending into relevant modes can bias the desired correlation. The default taper is explicit and does **not** establish a universal reliable s range. Check input k support and sampling, FFT sampling, endpoints and regulator/taper variations for the model and separation range of interest. Signed linear interpolation can be the limiting error even when the FFT grid is very fine.

A sparse P(k) table limited to 0.01..0.5 h/Mpc is generally insufficient for a robust BAO correlation calculation. Generate model tables on a wider, sufficiently sampled **declared** interval, choose the UV prescription explicitly and check convergence. The example uses 0.001..1 h/Mpc, 512 power samples, R=4 Mpc/h and an endpoint taper to demonstrate the interface; those settings are not a calibrated production recommendation.

AP distortion already present in P_ell(k) is carried through the transform. Do not apply another AP transformation to the returned xi. Outputs are point values, without separation-shell averaging, survey-window convolution, wide-angle terms or estimator binning.

## Stochastic contact terms

An infinite-band constant P_0 is a contact distribution at s=0. Restricting/tapering/damping its band gives it a finite spatial profile. This API **does not silently discard** constants or polynomial powers: it transforms exactly the supplied terms with the declared filter. The Gaussian-filtered constant has an analytic regression test.

For smooth clustering-only predictions, set the stochastic parameters to zero *before* computing the supplied powers:

- LPT/EPT: `sn=sn2=sn4=0`.
- TNS: `N=0`.
- EFTofLSS: `Pshot=a0=a2=0`.

Physical counterterms proportional to k^2 P(k) are different from pure stochastic contacts and remain part of the chosen model. A finite-k stochastic transform may be intentional, but its filter dependence must then be included in the interpretation.

## Connections to the existing models

```cpp
// LPT: Lagrangian bias; generate k_output densely over an explicit interval.
auto plpt = lpt.multipoles(f, k_output, lagrangian_bias, apar, aperp);
CorrelationWorkspace lpt_xi(plpt.k, separations, options);
auto xlpt = lpt_xi.evaluate_spectrum(plpt);

// IR-resummed EPT: Eulerian bias.
auto pept = ept.multipoles(f, eulerian_bias, apar, aperp);
CorrelationWorkspace ept_xi(pept.k, separations, options);
auto xept = ept_xi.evaluate_spectrum(pept);

// TNS and EFTofLSS return arrays; their model k() supplies the grid.
CorrelationWorkspace tns_xi(tns.k(), separations, options);
auto xtns = tns_xi.evaluate(tns.multipoles(tns_parameters));
CorrelationWorkspace eft_xi(eft.k(), separations, options);
auto xeft = eft_xi.evaluate(eft.multipoles(eft_bias));
```

Each result remains the original model's prediction, including its approximations and bias convention. This conversion does not make the different models or their bias parameters interchangeable.

## Builds, threads and MPI

Requires C++17 and **double-precision FFTW3**, including when FML's mesh precision is float. The wrapper is header-only and does not need `FFTLog.cpp` or GSL. The new example can be built with

```
make -C FML/PerturbationTheory/example correlation_example USE_MPI=false USE_OMP=false
cd FML/PerturbationTheory/example
./correlation_example
```

By default it reads the module's bundled linear and no-wiggle reference spectra and writes `correlation_multipoles.csv`. Optional arguments are `linear_file nowiggle_file output_csv`; the two input spectra must have identical k grids. It is an IR EPT interface example with stochastic terms zero, not a forecast or a production fit.

Operations are **local**, with no MPI/grid collectives. They may run on a single selected rank, or independently on multiple parameter-point ranks. This is not a distributed 3D Fourier transform. Serialize workspace construction, destruction and move-assignment of prepared workspaces with all other FFTW planning in the process; use one workspace per concurrent caller. Separate preplanned workspaces execute concurrently in the OpenMP regression.

The model integration exposed an existing nonportable `vector<fftw_complex>` in the CLASS-PT no-wiggle path: fftw_complex is a C array type and Clang/libc++ rejects that container. It now uses aligned FFTW allocations and a plan with RAII cleanup, preserving the numerical construction. The original no-wiggle, REPT and CLASS-PT tests are rerun for this dependency change.

The normal module test Makefile includes `test_correlations` and `test_correlation_models` when FFTW is enabled; module CI discovers them through the existing test target. See [CORRELATION_RESULTS.md](CORRELATION_RESULTS.md) for actual results and validation boundaries.
