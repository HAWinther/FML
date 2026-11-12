# Item 25 validation — 12 November 2026

## External numerical comparison

Eight weighted-catalogue cases passed against the actual installed pypower/pmesh estimator: fixed oblique/local first-point LOS, mesh widths 32/64 and CIC compensation off/on. Mode counts agree after the documented fixed-LOS floating-endpoint guard; mean k agrees to roundoff. The maximum raw multipole difference divided by the maximum raw reference monopole amplitude is **1.18e-15** across all cases (acceptance bound 2e-10). I2/alpha are checked independently. This comparison matches explicit normalization and transfer functions; it is not an assertion that the packages' defaults or shot-noise-corrected outputs coincide.

External sources and environment:

- pypower 1.0.0, cosmodesi commit `8606a05b357357c7cc71a5324cc3fdbdda522875`.
- pmesh 0.1.58, MP-Gadget commit `da7b574ab930836fac2410394592d62f3f04a68d` (the fork required by the tested pypower).
- pfft-python 0.1.24, mpsort 0.1.19, mpi4py 4.1.2, NumPy 1.26.4 and SciPy 1.17.1.
- Open MPI 5.0.11. Reference runs use one process; separate FML MPI checks use 2/4 ranks.

No external estimator source was patched. With pypower's default exact [-1,1] mu edges, the [1,2,3] fixed LOS dropped the two parallel modes in each of two bins: counts 240/1,024 versus FML/direct enumeration 242/1,026. A 16-double-epsilon endpoint guard restores the full sphere. The raw default-reference mismatch was preserved in the review evidence. This is an observed floating-boundary convention issue in the pinned reference, not a change to FML's estimator.

## Independent convergence

The unpainted catalogue DFT uses full signed modes and per-object LOS rather than CIC nodes. RMS raw multipole errors as a percentage of the continuum raw monopole RMS:

| Mesh | ell=0 | ell=2 | ell=4 |
|---|---:|---:|---:|
| 32^3 | 0.73555% | 1.51895% | 1.86896% |
| 64^3 | 0.04394% | 0.22835% | 0.49447% |
| 128^3 | 0.01868% | 0.04325% | 0.07493% |

The fixture's box is 1,200 length units; maximum k is 0.045 inverse-length units. These are numerical errors for this fixture, not fractional cosmological multipole-error forecasts. All weighted contact-noise vectors match the independently evaluated declared prescription. Common coordinate translation preserves the estimates. FML's 2/4-rank raw vectors differ from serial by at most 3.73e-10 in physical power units (roundoff relative to the raw power scale); corrected/noise vectors also pass.

## Regression and safety coverage

Fresh standalone survey workflow builds pass:

- Serial double: 38 checks.
- FFTW-MPI at 1/2/4 ranks: 38/47/47 checks.
- Serial single-precision FFTW: 38 checks.
- OpenMP at 1 and 4 threads: 38 checks each.
- AddressSanitizer plus UndefinedBehaviorSanitizer: 38 checks; FML code instrumented, external FFTW uninstrumented.

The new entry-point agreement check rejects mismatched mesh, observer, LOS mode, multipole count/order, bin count/values, I2 override, compensation or shifted-random presence **before** incompatible FFT schedules or reductions. Nonfinite normalization options are rejected, while finite <=0 continues to request the existing derived I2. The scientific force/painting/normalization/multipole formulas are unchanged. Existing tests include direct DFT, Poisson noise, window and reconstruction fixtures, so these dependent survey paths were exercised too.

The new catalogue driver also runs under the sanitizers. Its standalone Makefile is built in a clean scratch directory. Source/binary/input hashes, exact commands, reference package/module hashes, saved spectra and a convergence figure accompany the review artifact.

## Still open

These runs establish an external raw-estimator comparison, a declared-noise check and finite-CIC/LOS mesh convergence. They do not establish production mock recovery for an actual survey, masking/padding convergence for every geometry, near-/above-Nyquist accuracy, interlacing support, a cosmological covariance model, reconstruction-noise corrections, or an external pyrecon/window validation. A target-survey analysis must validate its own selection and noise conventions. No production scaling or speedup claim is made.
