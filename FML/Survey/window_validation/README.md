# Survey-window reference experiments

These optional NumPy/SciPy experiments validate `SurveyWindow.h` without changing
the production API. `window_reference.py --pypower` additionally runs actual
pypower `CorrelationFunctionSmoothWindow` / `PowerSpectrumSmoothWindowMatrix`
objects. Python and pypower are validation dependencies only.

## Build and run

From the repository root, with your normal FML compiler/FFTW configuration:

```sh
make -C FML/Survey/window_validation USE_MPI=false
python3 FML/Survey/window_validation/regression/window_reference.py \
  --binary FML/Survey/window_validation/survey_window_matrix --output /tmp/window-matrix
python3 FML/Survey/window_validation/regression/window_catalog.py \
  --binary FML/Survey/window_validation/survey_window_catalog --output /tmp/window-catalog
```

Install NumPy and SciPy in a separate environment. For the external comparison,
install [pypower](https://github.com/cosmodesi/pypower) with its supported dependencies
and add `--pypower` to the first command. See [RESULTS.md](RESULTS.md) for the exact
reference revision tested. MPI-enabled pypower needs a matching MPI runtime even
when its matrix calculation runs in one process.

For optional catalogue rank comparisons, build another copy with `USE_MPI=true`
and pass `--mpi-binary /path/to/mpi/survey_window_catalog --launcher 'mpirun'` to
`window_catalog.py`. Use your MPI launcher's normal host/oversubscription options.
The fixtures are small validation inputs: each rank reads the file and keeps its
index-modulo-rank share. This is not a production distributed catalogue reader.

Both scripts assert tolerances and save commands, generated inputs, CSV outputs,
reference arrays, summaries and runner/binary/input SHA-256 hashes. The catalogue
experiment writes roughly 20 MB per input and performs several ell=8 FFT estimates;
use a scratch output directory with sufficient space. MPI validation is optional;
it is not implied by a serial pass.

## Independent checks and conventions

The matrix fixture uses an axisymmetric Gaussian selection with widths
`(12,12,20)` in a consistent comoving length unit. Its autocorrelation is analytic.
The Python reference uses SciPy spherical Bessel functions, direct angular
projection (rather than the C++ triple-product helper), and refined Gauss quadrature
split at all interpolation knots. Every input/output ell=0,2,4 block and band is
checked for point and continuous k-squared-weighted output bins, with global IC
on and off. The IC difference must have rank one, and the point monopole row
at k=0 must vanish. This checks the declared local-plane-parallel global constraint,
not arbitrary survey-mean, radial-selection or wide-angle constraints.

Separate experiments vary input-band width, input k support and separation-table
spacing. They compare against analytic input Hankel transforms or the analytic
continuous window, so a finer C++ run is not used as the reference. Errors are
maximum absolute differences divided by the largest reference value over all
multipoles and output nodes; they are not relative errors of small entries.

The pypower comparison uses the same supplied interpolated Q table and smooth
Gaussian-times-polynomial input multipoles, wide-angle order zero and no IC.
Its FFTLog grid is refined from 8192 to 16384 samples. It tests phases, normalization
and angular mixing; its input interpolation/basis is not FML's piecewise-constant
band basis. Independent direct quadrature checks every FML band and the IC operator.
The explicit Q tail is zero at the end of the supplied separation range, where
the Gaussian is negligible; the first end uses constant extrapolation in both.

The catalogue fixture discretizes a **continuous** Gaussian selection on a spacing-2
quadrature grid, truncated at five standard deviations. It keeps mesh self pairs
(`subtract_self=false`) and uses analytic `I2`, with no renormalization to Q0(0).
It is not a Poisson mock or an empirical noise calibration. Exact CIC self-cloud
subtraction for weighted random objects is covered separately by the workflow's
direct-pair tests.

Fixed-LOS reference shell averages use angular/radial quadrature. The local LOS
reference integrates the conditional Gaussian pair distribution with scrambled
Sobol samples (seed 2718, 2^17 and 2^19 points). Meshes 32/64/128 share L=512,
separation edges 0..192 in steps of 8 and the same selection. Reports retain empty
bin counts and errors in all nonempty bins, beyond three cells for each mesh, and
on a common s>=48 interval. No fixed number of cells guarantees high-ell accuracy.
The padding test uses aligned meshes (L=512,N=64 vs L=768,N=96), keeping cell size
and sampling fixed. A deliberately underpadded L=384,N=48 case must reject.

## Applying this to perturbative theory

Construct the matrix once using sufficiently fine input edges and the measured
output edges. Evaluate your existing LPT/EPT, TNS or EFTofLSS `P0/P2/P4` model on
the input bands, then flatten all P0 values, all P2 values, then all P4 values into
`matrix.apply(theory_bandpowers)`. Midpoint model values are an approximation to
piecewise-constant input power: halve the input-band width and extend its k support
until the prediction stabilizes. Independently refine the Q table and catalogue
mesh; test padding at fixed physical cell size. Choose stochastic/noise terms in
the theory explicitly. The estimated catalogue noise subtraction and convolution
of a declared theory component have distinct conventions.

There is no need to transform the model through `CorrelationMultipoles.h` first:
the window matrix already performs the band-to-xi, angular-product and xi-to-power
operations. It does not include wide-angle corrections beyond the existing local
first-point convention or the exact finite-mesh output-mode distribution.
