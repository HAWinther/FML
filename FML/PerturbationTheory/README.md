# One-loop matter perturbation theory

Initial implementation of backlog item 35: EdS-kernel one-loop real-space matter spectra using FFTLog for `P22` and a linear FFT convolution for `P13`. An independent, FFT-free direct quadrature API supports reference calculations and explicitly supplied scale-dependent kernels. Existing `HybridBiasOneLoop*.h` LPT/CLEFT and modified-gravity calculations are unchanged. This module does not replace their time/scale-dependent kernels with EdS approximations.

## Definitions and units

Supply `P_linear(k,z)` in consistent length/volume units. The returned arrays are

```
linear = P_linear
loop = P22 + P13
total = linear + loop
```

Here `P13` includes the full factor `6 P_linear(k) integral P_linear(q) F3_sym(k,q,-q)`. No extra factor two should be added. The EdS `F2` is `5/7 + mu*(q/p+p/q)/2 + 2*mu^2/7`, with `mu=hat(q).hat(p)`. Redshift dependence enters through the supplied linear spectrum; the loops scale with its amplitude squared. These are ordinary SPT predictions, without EFT counterterms, tracer bias, redshift-space terms or IR resummation. Large corrections/negative total power can signal breakdown of the perturbative model; numerical agreement does not extend its physical validity.

## Checked fast path

```cpp
#include <FML/PerturbationTheory/FastOneLoop.h>
using namespace FML::PERTURBATION;
FastEdS workspace;
FastOptions options;
// k is strictly increasing, uniformly logarithmically spaced; at least 64 samples.
// The requested output interval must contain at least three input-grid modes.
auto result = workspace.evaluate(k, P_linear, output_kmin, output_kmax, options);
// result.k, linear, p22, p13, loop, total
// result.validation_k, reference_p22, reference_p13, max_reference_relative_error
```

Inputs must be finite, with positive wavenumbers and nonnegative powers. Only requested input-grid modes are returned. The spectrum is zero outside the supplied finite support; there is **no implicit power-law extrapolation or taper**. The default adds half an input length of zeros on each side for the Hankel transforms. `options.padding` selects a different number; zero is allowed. P13 uses a separately zero-padded *linear*, not circular, convolution and trapezoidal integration of the input log grid. Repeated calls at fixed dimensions/spacing reuse FFT plans and the P13 kernel transform.

The fast call always compares at least three logarithmically distributed output-grid modes (including both ends) with direct quadrature of an interpolant to the supplied spectrum. The interpolant is linear in power versus log wavenumber and zero outside support. Defaults are `reference_points=3`, `relative_tolerance=0.02`, `absolute_tolerance=0`, and the direct options below. Both `P22` and `P13` must separately satisfy `abs(error) <= atol + rtol*abs(reference)`. Invalid settings or failed checks throw; there is no silent clipping or EdS/direct substitution.

These are **spot checks, not a rigorous error bound between modes**. The returned diagnostic records their worst relative error. Increase checks, widen support, increase sampling, change padding and tighten/refine the reference integration for the intended spectrum/output interval. The finite domain and cancellation among P22 terms can cause inaccurate low-k tails; a tested Gaussian-cutoff case deliberately verifies that requesting such a tail fails. Very large padding can also amplify roundoff. Do not infer that any fixed padding or fraction of the grid is universally accurate. Use `direct_one_loop` when the fast setup cannot pass convergence tests. Altering reference tolerances does not improve numerical accuracy.

## Fast algorithm

Define `X_l^n(r) = integral dq q^(n+2) P(q) j_l(qr)/(2*pi^2)`. The configuration-space P22 integrand is

```
1219/735 (X_0^0)^2 + 1/3 X_0^-2 X_0^2 - 124/35 X_1^-1 X_1^1
+ 1342/1029 (X_2^0)^2 + 2/3 X_2^-2 X_2^2
- 16/35 X_3^-1 X_3^1 + 64/1715 (X_4^0)^2.
```

A final spherical transform gives `P22=4*pi integral dr r^2 j0(kr) xi22(r)`. The implementation uses the existing FFTLog workspace with zero bias and a common `kcrc=1` grid for every multipole; per-order shifts would prevent pointwise multiplication. It does not interpolate between different dual grids.

For P13, a bounded kernel `K(r)=r^2 Z(r)` gives

```
P13(k) = k^2 P(k)/(1008*pi^2) integral dlog(q) q P(q) K(q/k).
```

The equally spaced log grid makes this a Toeplitz convolution evaluated with FFTW. Moving `q` into the spectrum factor avoids an unnecessarily large `r*K(r)` kernel. `K(1)=-88`, `K(0)=-168`, `K(infinity)=-488/5`; small/large-r expansions avoid cancellation and the unity limit avoids `log(0)`. `convolution_plan_builds()` and `fft_buffer_bytes()` expose limited workspace diagnostics.

The FFT core costs O(N log N); fixed-count direct validation adds quadrature work. Storage is O(N), without a dense N-by-N kernel. The reported FFT buffer bytes include the FFTLog pair of complex buffers and the P13 pair; kernel coefficients, cached transformed kernel, padded input/product/output vectors, temporaries and FFTW internal planning memory are additional. This is not a total-memory budget. Workspaces own their buffers, have no global cache and must not be shared by concurrent calls.

## Direct path and custom kernels

```cpp
#include <FML/PerturbationTheory/OneLoop.h> // does not include or require FFTW
DirectOptions integration;
integration.order = 12;       // Gauss-Legendre order, 2..64
integration.log_step = 0.15;  // maximum log segment width
Spectrum P = [](double k) { return /* your linear power */; };
auto reference = direct_one_loop(output_k, P, qmin, qmax, integration);
```

Both internal legs use the explicitly truncated spectrum: zero outside `[qmin,qmax]`. P22 is integrated directly in `(q,p=|k-q|)` over the triangle domain, with composite logarithmic Gauss quadrature and splits at changing boundaries. The EdS P13 reference uses an independent one-dimensional quadrature with the analytic angular kernel, not FFT convolution. Output modes need not be logarithmic; output P(k) is also zero outside the declared support. Increase order and decrease log step to check integration convergence; no automatic quadrature error estimate is returned.

For scale-dependent/modified kernels, provide **both** callbacks and an explicit name distinct from `EdS`:

```cpp
Kernels model;
model.name = "my-model/version/redshift";
model.f2 = [](double q, double p, double mu_qp) { return /* fully symmetrized F2 */; };
model.f3_collapsed = [](double k, double q, double mu_kq) {
    return /* fully symmetrized F3(k-vector, q-vector, -q-vector) */;
};
auto prediction = direct_one_loop(output_k, P, qmin, qmax, integration, model);
```

F2 sees the two internal magnitudes and their mutual angle. F3 sees the external magnitude, loop magnitude and external–loop angle; its angular integral is performed numerically. This covers isotropic scalar kernels with that collapsed geometry, not arbitrary anisotropic or multi-fluid unequal-spectrum problems. The caller must provide the correct kernels and linear spectrum for the model. `FastEdS` deliberately accepts no custom kernels. Kernel callbacks and spectrum evaluations must be finite and may throw locally.

## Threads and MPI

These APIs are **local**, with no catalogue/grid collectives. MPI programs can distribute spectra or parameter points across ranks. Each FFT workspace must be prepared/exercised at its final sizes serially before concurrent execution; FFTW planning, destruction and geometry changes must be serialized with other FFTW planning in the process. Use one workspace per concurrent caller. A change in transform size may create plans. Tests execute separate preplanned workspaces concurrently. Direct quadrature and callbacks can be used concurrently when the callbacks support concurrent reads.

## Build and validation

The direct header requires only C++17. The fast header additionally requires the existing `FFTLog.cpp` and double-precision FFTW3, independent of FFTWGrid precision. Examples/tests use the shared FML Makefile configuration. **Clean objects between feature/compiler configurations.** From `tests/`:

```
make clean
make run USE_MPI=false USE_OMP=false USE_FFTW=false # analytic direct-only test
make clean
make run USE_MPI=false USE_OMP=false USE_FFTW=true
make clean
make run USE_MPI=true USE_OMP=true MPIRUN=mpirun NP_LIST="1 2 4"
```

`example/` prints a checked Gaussian-cutoff toy spectrum; it is not a cosmological forecast. The reference tests include two smooth spectral shapes, every returned mode for the Gaussian case, resolution changes, direct-integration refinement, analytic P22 k^4 and P13 displacement limits, amplitude scaling, custom-kernel normalization, FFT-plan reuse, thread execution and explicit rejection of an inaccurate low-k tail. The FFT-free test compares a constant-kernel convolution with the analytic intersection volume of two balls. The timing test compares a checked fast batch with direct evaluation of the same requested modes; measured speedups are environment/spectrum dependent.

Production follow-ups: validate cosmological transfer functions and desired k ranges, physical IR/UV-cut dependence and infrared cancellations; add analytic regularization, further supported bias/RSD terms and model-specific kernels before claiming a full FAST-PT replacement or an IR-resummed template library.

Primary references: [Schmittfull, Vlah & McDonald, FFT evaluation of one-loop terms](https://arxiv.org/abs/1603.04405), especially Eq. 25; [McEwen et al., FAST-PT](https://arxiv.org/abs/1603.04826), especially its P13 log-convolution construction. This is an independent implementation using FML's existing FFTLog, not imported FAST-PT code.
