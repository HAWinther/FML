# Correlation multipoles: validation — 12 November 2026

The new `CorrelationWorkspace` produces finite-band, explicitly filtered xi_0/2/4 from the existing perturbative power models. All final runs listed here pass. Builds used fresh objects in scratch directories, with double FFTW transforms and no GSL dependency.

## Numerical checks

The analytic Gaussian test uses `P_ell(k)=k^ell exp(-a k^2)`:

```
xi_ell(s) = i^ell s^ell exp[-s^2/(4a)] / [2^(ell+3) pi^(3/2) a^(ell+3/2)]
```

It checks normalization, the negative quadrupole phase, all three output orders and the known change `a -> a+R^2` under Gaussian damping. A separate constant-power monopole test verifies the filtered stochastic profile. Amplitude/sign linearity, zero fields, physical unit conversion, input preservation, move behavior and safe rejection of malformed/nonfinite inputs are covered.

An independent reference integrates the declared signed-linear-in-ln-k interpolant with composite Simpson quadrature, splitting at **every input knot** and subdividing to resolve oscillations. Its Bessel functions use a small-argument series and recurrence; it does not use FFTs, Mellin coefficients, the production resampling maps or native transform grids. Phase-step refinement from 0.1 to 0.05 tests reference convergence. A nonuniform signed-spectrum case tests endpoint tapering, Gaussian damping, zero crossings and FFT sampling refinement.

For the Gaussian case, the direct integral of the finite piecewise-linear input differs from the analytic continuous Gaussian by 3.92e-8, and the FFT result differs from that direct integral by 3.94e-8 (absolute xi units). This is within the stated interpolation/transform tolerances, not a claim of an exact analytic Gaussian between the supplied knots. Tests use 2e-7 absolute transform tolerance for these fixtures and an explicit 5e-8 interpolation-to-analytic tolerance; arbitrary input spectra do not inherit these error bounds.

## Actual model connections

All model cases use 96 power samples over 0.003..0.6 h/Mpc, FFT width 8,192, a 10% log-end taper, R=4 Mpc/h and separations 25, 50, 80, 110, 140 Mpc/h. They compare the new FFT result to refined direct integration of **the same supplied finite-band multipoles and filters**. Maximum absolute error divided by the largest absolute reference xi across all three multipoles and separations:

| Model | Scaled maximum error |
|---|---:|
| One-loop LPT, with AP distortion | 1.14e-6 |
| IR-resummed EPT, with AP distortion | 2.24e-6 |
| TNS with FoG | 1.94e-6 |
| IR-resummed EFTofLSS | 2.14e-6 |

EPT and EFTofLSS use the bundled nontrivial no-wiggle reference, so their wiggle contributions and IR damping are present. Stochastic parameters are zero in these smooth-clustering cases; their separate filtered contact behavior is covered analytically. This tests the conversion and model interfaces, not the physical validity of perturbation theory at every k in the table. The acceptance bound is 5e-5 of the largest reference |xi|, with direct-reference refinement below 1e-6 of that scale. Near-zero individual xi values are not used as relative-error denominators.

## Build and runtime matrix

| Configuration | New core | Four-model integration |
|---|---:|---:|
| Serial Clang/libc++, double FFTW | 21 checks | 6 checks |
| OpenMP, Clang 17, 1 and 4 threads | 22 each | 6 each |
| MPI / FFTW-MPI at 1, 2 and 4 ranks | 21 each | Not a distributed-model test |
| Serial float-grid configuration | 21 | Not repeated; correlation transforms remain double |
| AddressSanitizer + UndefinedBehaviorSanitizer | 21 | 6 |

The MPI core test additionally constructs/evaluates a workspace on **rank zero only**, proving that the API itself has no collective requirement. Every rank can also evaluate independently. The OpenMP test executes separately preplanned workspaces concurrently.

The CLASS-PT no-wiggle allocation fix is checked by existing reference regressions: NoWiggle (16 checks), EFTofLSS including CLASS-PT split/IR references (15), and REPT (23), all passing in fresh Clang builds. NoWiggle and EFTofLSS also pass under ASan/UBSan. The original header's `vector<fftw_complex>` Clang/libc++ compile failure was reproduced in an isolated include overlay; replacing it with aligned, owned FFTW buffers and a plan changes no numerical formula. An initial GNU 15 baseline also passed the existing LPT (22) and REPT (23) tests; the LPT baseline uses GNU because that test's independent oracle calls `std::cyl_bessel_j`, unavailable in the installed Apple libc++.

The new test targets and the example built and ran through clean standalone Makefiles in scratch directories. The IR EPT example writes an 81-separation CSV; a rendered figure is included with the review. Sanitizers instrumented FML code, with external FFTW libraries uninstrumented. Full library CI was not run.

## Setup reuse timing

A diagnostic serial Clang run, using the precomputed EFTofLSS power table above and five separations, measured:

- Existing workspace: **1.11 ms per evaluation**, averaged over 100 calls.
- New workspace plus evaluation each time: **37.1 ms**, averaged over 20 calls.
- Independent long-double direct quadrature: **1.89 ms**, averaged over 10 calls.

Reusing setup is about 33 times faster than constructing it for every evaluation in this fixture. This does not speed up the underlying model's loop calculation, which is excluded from these timings. The direct comparison depends strongly on the number of separations and numerical tolerances; it is not a general FFT-versus-quadrature performance guarantee. Logs include the checksum and exact compiler/commands. No production scaling or total-memory forecast is made.

## Remaining limits

Finite-band and filtered predictions are intentional. Input support and sampling, FFT sampling, taper/regulator choices, contact prescriptions and the usable separation range need target-analysis convergence checks. This does not validate infinite-band extrapolation, survey-window/wide-angle effects, separation-shell averaging, posterior coverage, or a new cosmological calibration of the underlying models. No output accuracy bound is estimated automatically at runtime. See [CORRELATION_MULTIPOLES.md](CORRELATION_MULTIPOLES.md).
