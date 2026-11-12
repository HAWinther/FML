# Survey-window validation results

Validated 12 November 2026. These manufactured/reference experiments establish
numerical conventions and expose resolution limits. They do not certify arbitrary
masks or production mock recovery. See [README.md](README.md) for runnable commands,
fixtures, independent references and the perturbative-theory recipe.

## Matrix correctness and integration fix

The previous radial quadrature crossed changes of slope in the piecewise-linear Q
interpolation. Splitting each radial bin at its used interior `s` knot fixes this.
The comparison below uses the same anisotropic Gaussian Q table (ds=8), input bands
and output nodes before/after the split. Errors are max absolute entry difference
scaled to the largest entry of each reference matrix. The independent reference
projects the full angular product and refines radial quadrature from 24 to 32 nodes.

| Output / global IC | Before split | After split |
|---|---:|---:|
| Point / off | 8.91e-5 | 3.17e-14 |
| Point / on | 8.41e-5 | 3.13e-14 |
| Binned / off | 9.00e-5 | 3.06e-14 |
| Binned / on | 8.25e-5 | 3.00e-14 |

All input/output ell=0,2,4 blocks and input bands are checked, including signs and
mixing. Reference refinement changes matrices by at most 1.89e-15 (point) and
6.78e-16 (binned). IC is independently integrated; the corrected point monopole
at k=0 vanishes within the asserted 1e-11 matrix-scale tolerance. The second/first
singular-value ratio of the IC difference is 1.87e-15 (point), 4.90e-16 (binned),
consistent with the intended rank-one projection.

This fix increases radial nodes and temporary construction workspace by up to a
factor of two. It changes one-time matrix construction, not the matrix dimensions
or application cost. No speed or memory improvement is claimed. Finite quadrature
still requires convergence for strongly oscillatory kernels or broad bands.

Output-shell normalization now avoids subtracting nearly equal cubes. A width
1e-12 shell near k=.13 agrees with its midpoint prediction at tolerance 5e-10.
Malformed interpolation metadata, nonfinite input/output arithmetic and ten kinds
of rank-dependent estimator options are rejected before unsafe collective work.
Application checks finite results in its normal inner-product pass without a
second matrix scan.

## External smooth-model reference

Actual pypower 1.0.0, source commit
`8606a05b357357c7cc71a5324cc3fdbdda522875`, agrees with the independent analytic-input
Hankel/direct-angular result to **1.92e-6** of the largest reference signal.
Refining its FFTLog grid from 8192 to 16384 changes predictions by 5.69e-6 on the
same scale. Both use the supplied Q table, input ell=0,2,4 and wide-angle order zero.
This is a smooth-model, no-IC comparison; it is not a claim that pypower uses the
same top-hat input basis. Each FML band and IC are checked by independent quadrature.

The environment used Python 3.11, NumPy 1.26.4, SciPy 1.17.1 and Open MPI 5.0.11.
Exact commands, module/source/binary/input hashes and raw arrays accompany the
review artifacts; reference code was not patched.

## Binning and support convergence

The following errors use all three output multipoles and k=0,.01,.03,.06,.1,.15.
They are scaled absolute errors, not elementwise relative errors.

| Quantity varied | Value | Error |
|---|---:|---:|
| Input dk (analytic smooth-input reference) | 0.02 | 0.0133789 |
| Input dk (analytic smooth-input reference) | 0.01 | 0.00336775 |
| Input dk (analytic smooth-input reference) | 0.005 | 0.000843383 |
| Separation ds (analytic continuous-Q reference) | 8 | 0.0143069 |
| Separation ds (analytic continuous-Q reference) | 4 | 0.0036101 |
| Separation ds (analytic continuous-Q reference) | 2 | 0.000904626 |
| Separation ds (analytic continuous-Q reference) | 1 | 0.000226288 |
| Input kmax (difference from .4, dk=.005 fixed) | 0.08 | 0.0466715 |
| Input kmax (difference from .4, dk=.005 fixed) | 0.16 | 0.000212382 |
| Input kmax (difference from .4, dk=.005 fixed) | 0.24 | 4.02169e-10 |
| Input kmax (difference from .4, dk=.005 fixed) | 0.4 | 0 |

The input-dk check holds the Q table fixed; the separation-ds check holds the same
finite-band input fixed. Its independent continuous-Q quadrature is refined from
192 to 256 nodes and must agree to 1e-10. Thus the two sources of error are isolated.
Input support ending near the output range can miss window-mixed signal; the
required support and bin widths depend on the actual model and mask.

## Catalogue mesh and padding

The selection is a continuous anisotropic Gaussian, sampled as deterministic
quadrature weights; self pairs are retained and I2 is analytic. It is not a Poisson
mock. Exact weighted CIC self-cloud subtraction remains covered by direct-pair
workflow regressions. No Q0(0) rescaling is performed.

Common-interval maximum absolute Q errors, separation-bin lower edges >=48 through
192, compared to continuum shell averages (Q0(0)=1 analytically):

| LOS / mesh | Q0 | Q2 | Q4 | Q6 | Q8 |
|---|---:|---:|---:|---:|---:|
| fixed-n32 | 0.018464 | 0.017429 | 0.041657 | 0.050606 | 0.040972 |
| fixed-n64 | 0.0056945 | 0.0069973 | 0.010142 | 0.013898 | 0.016971 |
| fixed-n128 | 0.0011283 | 0.00071051 | 0.00069048 | 0.0017948 | 0.002991 |
| local-n32 | 0.018464 | 0.017662 | 0.039588 | 0.044773 | 0.033241 |
| local-n64 | 0.0056945 | 0.0070014 | 0.0099254 | 0.013026 | 0.014143 |
| local-n128 | 0.0011283 | 0.00069884 | 0.00071495 | 0.0017754 | 0.0022573 |

These are absolute errors in Q units, not fractional accuracies for the smaller
high-order multipoles. The local-LOS Sobol reference changes by at most 1.64e-4
between 2^17 and 2^19 samples; its Q0 agrees with analytic integration to 1.14e-10.
The common-interval accuracy improves under mesh refinement.

Small-scale errors are much larger and are retained in the machine-readable
summary. The isotropic-leakage subtraction does not resolve anisotropic lattice
mixing. The table below summarizes maximum errors over all multipoles:

| LOS / mesh | All nonempty bins | Bins starting beyond 3 cells | Empty bins |
|---|---:|---:|---:|
| fixed-n32 | 0.29886 | 0.050606 | 1 |
| fixed-n64 | 0.14216 | 0.051129 | 0 |
| fixed-n128 | 0.044284 | 0.021514 | 0 |
| local-n32 | 0.32824 | 0.044773 | 1 |
| local-n64 | 0.12441 | 0.049032 | 0 |
| local-n128 | 0.039325 | 0.020507 | 0 |

An empty bin returns Q=0 with `nvectors=0`; that value is not a measured continuum
window. There is no universal three-cell accuracy criterion. Choose the separation
interval and mesh using convergence of the actual multipoles or final prediction.

At fixed cell size 8, enlarging the aligned box from (L,N)=(512,64) to (768,96)
changes fixed/local-LOS Q by at most 8.89e-16 / 2.32e-14 respectively. The deliberately
underpadded (384,48) case rejects with the extent-plus-CIC check. This is an
image/padding test for this fixture, not arbitrary-mask support certification.
Local-LOS mesh outputs at 2 and 4 MPI ranks agree with serial within 4.06e-13.

## Build and regression coverage

All builds below compile the workflow and both new reference drivers from current
sources with C++17 and FFTW. There were no compiler warnings in the captured logs.

| Configuration | Workflow checks | Result |
|---|---|---|
| Apple Clang serial, double | 47 | Pass |
| Apple Clang serial, single-precision FFTW | 47 | Pass |
| Clang 17 OpenMP, 1 and 4 threads | 47 each | Pass |
| Apple Clang AddressSanitizer + UndefinedBehaviorSanitizer | 47 | Pass |
| Open MPI / Clang 17, 1 / 2 / 4 ranks | 47 / 66 / 66 | Pass |

The sanitizer build also runs the complete coarse point/binned IC/no-IC matrix
fixture; external FFTW libraries are not sanitizer-instrumented. The standalone
Makefile builds both drivers in a clean copied build directory. Python reference
scripts, all numerical assertions and the MPI catalogue checks pass.

Remaining work: target-survey mock recovery and noise calibration, convergence for
specific masks and theory models, small-scale anisotropic lattice accuracy, exact
mode-weighted output averages if needed, and survey constraints/wide-angle terms
beyond the declared local-plane-parallel global-IC model.
