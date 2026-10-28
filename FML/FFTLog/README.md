# FFTLog

FFTLog converts logarithmically sampled spectra into correlation functions and
general spherical-Bessel integrals. It requires double-precision FFTW, regardless
of the precision used by FML's FFTWGrid.

Existing calls still work. For repeated transforms, retain a workspace:

```cpp
using namespace FML::SOLVERS::FFTLog;
Workspace workspace; // FFTW_ESTIMATE by default; FFTW_MEASURE is also supported
auto xi = ComputeCorrelationFunction(k, pk, workspace);
auto next_xi = ComputeCorrelationFunction(k, next_pk, workspace);
auto multipole = ComputeXiLM(2, 4, k, pk, workspace, -1.0);
auto back = ComputePowerSpectrum(xi.first, xi.second, workspace);
// Complex Hankel transforms:
auto result = DiscreteHankelTransform(r, a, workspace, mu, q, kcrc, true);
```

The workspace owns two FFTW-aligned complex buffers and two plans. Plans are reused
at a fixed length; changing length replaces them. One coefficient set is cached by
length, Bessel order `mu`, bias `q`, logarithmic span `L`, requested `kcrc`, and
`noring`. Changing amplitudes alone reuses both plans and coefficients. Alternating
between geometries benefits from a workspace for each geometry. There is no global
cache. Returned vectors own their data and remain valid after subsequent calls.
The high-level real transforms fill the aligned input and normalize the output
directly, avoiding temporary complex integrand and result vectors.

The overloads without a workspace create a temporary one. All overloads preserve
caller inputs, including when using `FFTW_MEASURE`. Caller-supplied coefficients
must match the transform length and are read on every call. As in the original
API, supplying coefficients bypasses the no-ringing adjustment even if
`noring=true`; the caller must pass the `kcrc` used to compute those coefficients.
External coefficients do not replace the internal cache.

`prepare(N)` allocates buffers and builds plans before the first transform;
`clear()` releases them and resets statistics. Workspaces are movable, not copyable.
`size()` reports the prepared length, `buffer_bytes()` reports only the two aligned
buffers (`32*N` bytes), and `statistics()` reports cumulative plan and coefficient
build counts since the last clear. Coefficients, returned vectors, FFTW's internal
plan storage, and temporary storage while replacing plans are additional memory.

FFTW planning/destruction must be serialized with other FFTW planning in the
application. Construct/prepare, resize, clear, move-assign over an existing
workspace, and destroy workspaces outside concurrent planning. Use one workspace
per concurrent caller, prepare them serially, then execute transforms at those
prepared lengths concurrently. Plans inherit FFTW's thread configuration when
created; clear and prepare again to adopt a changed configuration. A workspace
cannot be shared by simultaneous calls.

Input grids retain the logarithmic-spacing requirement. Transforms reject lengths
below two or above `INT_MAX`, mismatched vectors, and invalid endpoint spans.
Interior logarithmic spacing is the caller's responsibility.
