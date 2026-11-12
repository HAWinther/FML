# One-loop matter perturbation theory

Contents: one-loop SPT matter spectra (`FastOneLoop.h`, `OneLoop.h`, below) and one-loop Lagrangian perturbation theory
of biased tracers in redshift space (`LPTRedshiftSpace.h`, `LagrangianQFunctions.h`, [last section](#redshift-space-lagrangian-perturbation-theory-of-biased-tracers)).

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

Inputs must be finite, with positive wavenumbers and nonnegative powers. Only requested input-grid modes are returned. The spectrum is zero outside the supplied finite support; there is **no implicit power-law extrapolation**. The default adds half an input length of zeros on each side for the Hankel transforms. `options.padding` selects a different number; zero is allowed.

For the P22 transforms the spectrum is smoothly tapered to zero over `options.taper` (default 0.1) of the logarithmic input range at each end, with the window `x - sin(2 pi x)/(2 pi)` of FAST-PT. P13 always uses the untapered spectrum. The FFTLog transforms of `k^{n+1.5} P(k)` (n = -2..2) of an abruptly truncated spectrum ring, and P22 is a cancellation among their products: for an Eisenstein-Hu LCDM spectrum on 1e-5..50 h/Mpc the untapered P22 was wrong by a factor ~5 at k = 0.1 h/Mpc and by 1e3-1e9 below 0.02 h/Mpc (the reference checks then throw). With the taper every mode in 0.02-1 h/Mpc agrees with direct quadrature to 2e-4. Keep the requested output range well inside the tapered ends; the reference checks compare with the untapered spectrum. P13 uses a separately zero-padded *linear*, not circular, convolution and trapezoidal integration of the input log grid. Repeated calls at fixed dimensions/spacing reuse FFT plans and the P13 kernel transform.

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

`example/` prints a checked Gaussian-cutoff toy spectrum; it is not a cosmological forecast. The direct quadrature agrees with FAST-PT (one_loop_dd) to 1e-3 for an Eisenstein-Hu LCDM spectrum at k = 0.02-1 h/Mpc (tested against stored FAST-PT 4.1.1 values; 4e-4 for a CLASS spectrum at 1e-3-1 h/Mpc), and the custom-kernel path with the EdS F3 from the standard recursion reproduces the analytic P13 kernel. The reference tests include two smooth spectral shapes, every returned mode for the Gaussian case, resolution changes, direct-integration refinement, analytic P22 k^4 and P13 displacement limits, amplitude scaling, custom-kernel normalization, FFT-plan reuse, thread execution and explicit rejection of an inaccurate low-k tail. The FFT-free test compares a constant-kernel convolution with the analytic intersection volume of two balls. The timing test compares a checked fast batch with direct evaluation of the same requested modes; measured speedups are environment/spectrum dependent.

Production follow-ups: validate cosmological transfer functions and desired k ranges, physical IR/UV-cut dependence and infrared cancellations; add analytic regularization, further supported bias/RSD terms and model-specific kernels before claiming a full FAST-PT replacement or an IR-resummed template library.

Primary references: [Schmittfull, Vlah & McDonald, FFT evaluation of one-loop terms](https://arxiv.org/abs/1603.04405), especially Eq. 25; [McEwen et al., FAST-PT](https://arxiv.org/abs/1603.04826), especially its P13 log-convolution construction. This is an independent implementation using FML's existing FFTLog, not imported FAST-PT code.

## Redshift-space Lagrangian perturbation theory of biased tracers

`LPTRedshiftSpace.h` gives the redshift-space power spectrum P(k, mu) and the multipoles P_0, P_2, P_4 of biased
tracers in one-loop LPT with the long-wavelength displacements resummed (the direct Lagrangian expansion of Chen,
Vlah, Castorina & White 2021, [arXiv:2012.04636](https://arxiv.org/abs/2012.04636)). It is a port of
[velocileptors](https://github.com/sfschen/velocileptors)' `LPT/lpt_rsd_fftw.py` (Chen, Vlah & White, MIT license);
`LagrangianQFunctions.h` ports its q-functions (`Utils/qfuncfft.py`) and `FFTLog/SphericalBesselTransform.h` its
FFTLog transform.

```cpp
#include <FML/PerturbationTheory/LPTRedshiftSpace.h>
using namespace FML::PERTURBATION;
LPTRedshiftSpace lpt(k, P_linear);               // k [h/Mpc], P [(Mpc/h)^3] at the redshift
LPTRedshiftSpace::Bias b;                         // Lagrangian b1, b2, bs, b3, alpha0..6, sn, sn2, sn4
b.b1 = 0.7; b.b2 = 0.3; b.bs = -0.2; b.alpha0 = 10; b.sn = 1000;
auto table = lpt.multipole_components(f, kout, apar, aperp);   // the expensive part (bias independent)
auto pell = LPTRedshiftSpace::combine(table, b);                // P_0, P_2, P_4: cheap for every bias
auto pkmu = lpt.pkmu(f, mu, kout, b);                           // P(k, mu) (no AP)
```

The bias expansion is delta_t = 1 + b1 delta + b2/2 delta^2 + bs s^2 + b3 O_3 (Lagrangian; the Eulerian linear bias is
1 + b1), with the counterterms (alpha0 + alpha2 mu^2 + alpha4 mu^4 + alpha6 mu^6) k^2 P_ct (P_ct the resummed linear
power, or the Zel'dovich one with `use_Pzel`) and the stochastic terms sn + sn2 k^2 mu^2 + sn4 k^4 mu^4. The
multipole tables have 19 columns (12 bias monomials, 4 counterterms, 3 stochastic terms) with the Alcock-Paczynski
parameters applied, so fitting the bias and nuisance parameters only needs `combine`. `pk_components` gives the 13
components of P(k, mu) (the bias monomials and P_ct). Options: the cutoff exp(-(k/cutoff)^2) of the input (10 h/Mpc),
the internal grid (`log10_kmin` = -4, `log10_kmax` = 1, `N` = 1024), the number of spherical Bessel orders and
hypergeometric terms of the angular integrals (`jn` = `nmax` = 10), an IR split `kIR` of the resummed displacements,
`one_loop`, `use_Pzel`.

Validation (`tests/test_lptrsd.cpp`):

- velocileptors with its settings (jn = nmax = 5, N = 256): the components and the multipole columns with AP, also
  with kIR and use_Pzel, to 7e-9 (fixture from `python/generate_lptrsd_reference.py`).
- The redshift-space Zel'dovich P(k, mu) (one_loop = false) against a direct integration over q, independent of the
  hypergeometric/FFTLog angular series: 0.02-0.12% (f = 0.8, k = 0.05-0.2 h/Mpc).
- f = 0 against the real-space CLEFT of `ComputePowerSpectra/HybridBiasOneLoop.h` (direct integration): 2e-4.
- The Kaiser limit (1 + b1 + f mu^2)^2 P_lin at k = 0.001 h/Mpc (log10_kmin = -5): 8e-5. Convergence in jn/nmax
  (1e-5) and ngauss (6e-5), the bias combination, isotropic AP, rejected options.

velocileptors' default settings are not converged for the angular series: with jn = nmax = 5, P_0 is off by 0.8% (1.5%)
and P_2 by 4% (1%) at k = 0.2 (0.3) h/Mpc, and ngauss = 3 gets P_4 only to ~1%. The defaults here are therefore jn =
nmax = 10, ngauss = 4 and N = 1024 (N = 256: 0.2-0.4% against the direct Zel'dovich integration). N must be even (odd N
gives 1-10% errors, also in velocileptors) and log10_kmax at most 1 (at smaller q the series loses all precision).
Valid for k >~ 0.005 h/Mpc with the default grid. Cost: ~0.8 s for the multipole tables at 50 k with the defaults
(0.1 s with velocileptors' settings). Not thread safe: one object per thread. The correlation-function multipoles
and the moment expansion of velocileptors are not ported.

## IR-resummed Eulerian perturbation theory in redshift space

`EPTRedshiftSpace.h` gives P(k, mu) and the multipoles P_0, P_2, P_4 of biased tracers in one-loop Eulerian
perturbation theory with IR resummation (velocileptors' "REPT", Chen, Vlah & White 2020,
[arXiv:2005.00523](https://arxiv.org/abs/2005.00523)): the one-loop P(k, mu) built from the velocity moments up to
the fourth, with the wiggle part of every term damped by exp(-k^2 Sigma^2 (1 + f (2 + f) mu^2) / 2), Sigma^2 =
X(r_bao) + Y(r_bao)/3 of the no-wiggle spectrum (r_bao = 110 Mpc/h), and the damped linear term compensated. The
one-loop spectra are computed with FFTLog from the k-expanded Lagrangian expansion and mapped to the Eulerian bias
basis (`EPTVelocityMoments`, also usable on its own for the unresummed EPT tables). A port of velocileptors'
`EPT/ept_fullresum_fftw.py` with `ept_fftw.py`, `velocity_moments_kexpanded_fftw.py` and `cleft_kexpanded_fftw.py`.

```cpp
#include <FML/PerturbationTheory/EPTRedshiftSpace.h>
using namespace FML::PERTURBATION;
EPTRedshiftSpace::Options opt;                   // output grid kmin = 0.01, kmax = 0.5, nk = 100 (log-spaced)
EPTRedshiftSpace rept(k, P_linear, {}, opt);    // {}: no-wiggle P by the DST method (NoWiggle.h); or pass P_nw(k)
EPTRedshiftSpace::Bias b;                        // Eulerian b1, b2, bs, b3, alpha0..6, sn, sn2, sn4
b.b1 = 1.7; b.b2 = 0.3; b.alpha0 = 10; b.sn = 1000;
auto table = rept.multipole_components(f, apar, aperp);   // 19 columns, bias independent
auto pell = EPTRedshiftSpace::combine(table, b);           // P_0, P_2, P_4 on rept.k()
auto pkmu = rept.pkmu(f, mu, b, apar, aperp, bFoG);        // P(k, mu)
```

The bias is Eulerian (P -> (b1 + f mu^2)^2 P_lin on large scales; LPTRedshiftSpace uses the Lagrangian b1 = b1_E - 1).
The counterterms (alpha0 + alpha2 mu^2 + alpha4 mu^4 + alpha6 mu^6) k^2 multiply the resummed linear power P_nw +
e^(-k^2 Sigma^2(mu)/2) P_w, the stochastic terms are sn + sn2 k^2 mu^2 + sn4 k^4 mu^4, and `pkmu` has velocileptors'
optional -bFoG k^4 mu^4 (b1 + f mu^2)^2 P_res term. The columns are the same 19 as in LPTRedshiftSpace. Without a
given no-wiggle spectrum it is computed on the internal grid with the discrete sine transform method of
`NoWiggle.h` (default) or, as in velocileptors' REPT, a Savitzky-Golay filter (`opt.nowiggle`); see below for why
the default differs (velocileptors also applies its cutoff exp(-(k/20)^2) twice to its filtered spectrum, here once).
Sigma^2 can be set directly (`sigma2_bao`).

Validation (`tests/test_rept.cpp`):

- velocileptors with its internal grid and the same no-wiggle spectrum: the Savitzky-Golay filter (6e-8, the
  accuracy of scipy's own coefficients),
  Sigma^2, the 31 EPT velocity-moment columns, the 19 columns of P(k, mu) with and without AP, P(k, mu) with bFoG and
  the multipole columns, all to ~1e-8 (fixture from `python/generate_rept_reference.py`).
- f = 0, b1 = 1, no wiggles against the direct one-loop SPT P_lin + P_22 + P_13 of `OneLoop.h`: 6e-5 for k = 0.02 -
  0.5 h/Mpc. The Kaiser limit at k = 0.001-0.003 h/Mpc (6e-4, the loop terms). No resummation two ways (P_nw = P,
  Sigma^2 = 0). Convergence of the internal grid (5e-5 of P_0) and of ngauss (4e-7).
- The no-wiggle method (a CAMB P(k), k <= 0.3 h/Mpc): the multipoles with the DST and the Gaussian-EH spectra agree to
  0.5% of P_0; with the Savitzky-Golay one they differ by up to 3%, which is why DST is the default.

velocileptors' internal grid (k = 1e-4 - 1e3 h/Mpc, N = 4000) is not converged: the k-expanded loop terms are
large cancelling powers of k, and with q <= 1e4 Mpc/h the real-space P is off by 0.2% at k = 0.3 h/Mpc and 0.5% at
k = 0.5 h/Mpc (0.7% in P(k, mu = 0.7)). The default grid is therefore 1e-5 - 1e3 h/Mpc with N = 6000. The angular
integrals are exact (l <= 4). Cost: ~0.3 s to construct (q-functions of P and of P_nw); the tables are cheap. Not
thread safe: one object per thread.

## Wiggle/no-wiggle splits

`NoWiggle.h` gives the smooth (no-wiggle) part P_nw of a linear P(k) (k in h/Mpc), with the BAO removed:

```cpp
#include <FML/PerturbationTheory/NoWiggle.h>
using namespace FML::PERTURBATION;
auto pnw_dst = nowiggle_dst(k, P);                  // discrete sine transform (no cosmology needed)
EisensteinHuParameters eh;                          // Omega_m, Omega_b, h, TCMB, n_s
auto pnw_gauss = nowiggle_gaussian(k, P, eh);       // Gaussian smoothing of P / P_EH,nw in ln k (sigma = 0.25)
auto pnw_sg = nowiggle_savgol(k, P);                // Savitzky-Golay filter in ln k
double T = eisenstein_hu_transfer(kh, eh), Tnw = eisenstein_hu_transfer_nowiggle(kh, eh);
```

- `nowiggle_dst`: the method of Hamann et al. 2010 / Wallisch 2018 (arXiv:1810.02800): the BAO bump in the discrete
  sine transform of ln(k P) on a linear k-grid (1e-3 - 10 h/Mpc, 2^16 points; P_nw = P outside) is cut out and
  interpolated over. A port of velocileptors' `Utils/pnw_dst.py`, reproduced to 1e-13, except that the bump detection
  ignores shallow extrema: velocileptors' choice of the cut depends on the sampling of P (3% changes in P_nw for the
  same P resampled, here < 0.5%). Manual cut indices `ii_l`, `ii_r` (a smooth P has no bump: an error).
- `nowiggle_gaussian`: P_ref times the Gaussian-smoothed P / P_ref (Vlah et al. 2016; HMcode-2020), with any smooth
  reference shape or the Eisenstein & Hu no-wiggle one.
- `nowiggle_savgol`: velocileptors' REPT default (order 4, window ln 7 in ln k).
- `nowiggle_classpt(k, P, h, kout)`: CLASS-PT's DST split (in 1/Mpc: 65536 points over 7e-5 - 7 /Mpc, harmonics
  120 - 239 removed), used by `EFTofLSS.h` to reproduce CLASS-PT's IR resummation.
- The Eisenstein & Hu (1998) transfer functions with and without the baryon wiggles (`Cosmology/Spectra/EisensteinHu.h`),
  equal to colossus' to rounding.

Validation (`tests/test_nowiggle.cpp`; references from `python/generate_nowiggle_reference.py`, including a
well-sampled CAMB P(k)): the ports above; a smooth P is returned to 9e-5 (DST), 0.5% (Savitzky-Golay) and 0.8%
(Gaussian with a reference shape for Omega_m = 0.25, n_s = 1 instead of 0.31, 0.97); the wiggles of the CAMB P,
measured as the largest residual of ln P_nw about a degree-10 polynomial in ln k for k = 0.05 - 0.4 h/Mpc, go from
4.9% to 0.7% (DST and Savitzky-Golay) and 0.05% (Gaussian-EH). DST and Gaussian-EH agree to 0.7% for k = 0.15 - 2
h/Mpc but differ by up to 5% around the first wiggle (k = 0.005 - 0.05), where the split is ill-defined.
The Savitzky-Golay filter leaves wiggles of ~2% at k = 0.1 - 0.3 h/Mpc. The Gaussian-EH method is the cleanest when
the cosmology is known; DST needs only P. The Eisenstein & Hu fits are those of `Cosmology/Spectra/EisensteinHu.h`
(header only), the one copy in FML, also used by HMcode-2020, FieldLevel and the Cosmology spectra.

## Direct loop integrals with redshift-space kernels

`SPTKernels.h` has the EdS kernels F_n, G_n (n <= 3, symmetrized, from the recursion) and the redshift-space kernels
Z_n of biased tracers (n <= 3, bias basis b1, b2/2 delta^2, bG2 G2, bGamma3 Gamma3) derived mechanically from the
redshift-space mapping delta_s = sum_m (1/m!) FT[(-i k_z u_z)^m (1 + delta_g)] rather than typed in (Z2 also in closed
form; the two agree to 1e-15). `LoopIntegrals.h` is the quadrature engine behind TNS and EFTofLSS: P22-type integrals
over (q, |k - q|) with log Gauss-Legendre segments broken at the kinematic points and Gauss-Chebyshev in the azimuth,
P13-type integrals as scale-free angular kernels on fixed r = q/k nodes, and the mu dependence as polynomials in mu^2
recovered from 5 mu nodes. P_lin is a cubic spline in ln P(ln k), zero outside its table: supply it over a wide range
(e.g. 1e-5 - 50 h/Mpc). Matter P22 and P13 agree with `OneLoop.h` to 1e-5 - 5e-5.

## TNS model

`TNS.h`: the TNS model (Taruya, Nishimichi & Saito 2010, [arXiv:1006.0699](https://arxiv.org/abs/1006.0699)) with
one-loop SPT and the McDonald & Roy nonlinear bias (as in BOSS fits, Beutler et al. 2014):

```cpp
#include <FML/PerturbationTheory/TNS.h>
using namespace FML::PERTURBATION;
TNS tns(k_lin, P_lin, k_out);                 // all loop tables (bias and f independent) on k_out
TNS::Parameters p;                            // f, b1, b2, bs2, b3nl, sigma_v [Mpc/h], N, fog, AB
p.f = 0.76; p.b1 = 2.0; p.b2 = 0.5; p.bs2 = -4.0 / 7.0 * (p.b1 - 1); p.b3nl = 32.0 / 315.0 * (p.b1 - 1);
p.sigma_v = 4.0; p.fog = TNS::FoG::Lorentzian;  // or Gaussian, None
auto pkmu = tns.pkmu(0.5, p);                 // P(k, mu)
auto pell = tns.multipoles(p);                // P_0, P_2, P_4 (Gauss-Legendre in mu)
auto & T = tns.tables();                      // P22/P13 of dd, dt, tt; bias terms; A_n, B_n by powers of mu
```

P_s = D_FoG(k mu f sigma_v) [P_g,dd + 2 f mu^2 P_g,dt + f^2 mu^4 P_tt + A + B] + N. A and B are tabulated as
coefficients of f^n b1^(3-n) mu^(2m) (A) and f^n b1^(4-n) mu^(2m) (B), so f and b1 vary freely without
recomputing loops. P_b22, P_b2s2, P_bs22 have their k -> 0 constants subtracted. Cost: ~0.2 s for 10 output k.

Validation (`tests/test_tns.cpp`, FAST-PT fixture from `python/generate_fastpt_reference.py`): the one-loop P_dd
(7e-5 of |P22| + |P13|), the nonlinear bias terms (Pd1d2, Pd2d2, Pd1s2, Pd2s2, Ps2s2: 3e-4; sig3nl 3e-4) and A + B by
powers of mu (2e-3 of the largest; FAST-PT's own accuracy) against FAST-PT 4.1.1; P22 and P13 of dd and tt against
`OneLoop.h` (6e-5; tt with custom G2, G3 kernels); P13 of dd, dt, tt against the analytic 1D kernels (9e-5); the
Kaiser limit; the multipoles without FoG against their exact Legendre moments; the FoG factors.

## EFTofLSS (CLASS-PT)

`EFTofLSS.h`: the one-loop EFTofLSS multipoles of biased tracers in redshift space with IR resummation, the model,
parameters and conventions of CLASS-PT (Ivanov, Simonovic & Zaldarriaga 2020; Chudaykin et al. 2020,
[arXiv:2004.10607](https://arxiv.org/abs/2004.10607)):

```cpp
#include <FML/PerturbationTheory/EFTofLSS.h>
using namespace FML::PERTURBATION;
EFTofLSS::Options o;
o.h = 0.6736;                                 // the ClassPT no-wiggle split works in 1/Mpc
o.r_bao = o.h * r_drag;                       // BAO scale in Mpc/h (CLASS-PT: the sound horizon at drag)
EFTofLSS eft(k_lin, P_lin, k_out, f, o);      // the 12 bias-monomial tables and counterterm templates
EFTofLSS::Bias b;                             // b1, b2, bG2, bGamma3, c0, c2, c4 [(Mpc/h)^2], b4 [(Mpc/h)^4], Pshot, a0, a2
b.b1 = 2.0; b.b2 = -0.5; b.bG2 = 0.3; b.c0 = 5; b.Pshot = 400;
auto pell = eft.multipoles(b);                // P_0, P_2, P_4 on k_out (a cheap linear combination)
auto & table = eft.multipole_table();         // table[l/2][monomial or CT0/CT2/CT4][ik]
```

The IR resummation reproduces CLASS-PT's (no-wiggle split, Sigma^2 and delta Sigma^2 below k_S = 0.2 h/Mpc, the
damping of the tree, matter loops, bias loops and counterterms; see the header); `o.nowiggle` can instead use the DST
split of `NoWiggle.h` or a given P_nw, and `o.ir_resummation = false` gives plain one-loop SPT. k in the counterterm
and stochastic terms is in h/Mpc: CLASS-PT's python wrapper uses the k of `initialize_output` (usually 1/Mpc) for b4,
a0, a2, so its b4 is h^2 times this one. The k -> 0 constant of the b2^2 term is subtracted (CLASS-PT's monopole
differs by a constant that depends on the k grid given to it). Cost: ~0.3 s for 10 output k.

Validation (`tests/test_eftoflss.cpp`, CLASS-PT fixture from `python/generate_classpt_reference.py`): every bias
monomial in l = 0, 2, 4 against CLASS-PT's pk_gg_l0/2/4 for k = 0.005 - 0.3 h/Mpc, with and without IR resummation, to
better than 1.3e-3 of the column's largest value (most to 1e-4); the counterterm templates and b4 to 2e-4; the
Kaiser limit; the bias combination; a given P_nw; convergence against a refined quadrature (5e-5 of P_0).

## One loop with modified-gravity kernels

`MGOneLoop.h`: the one-loop P(k) with the SPT kernels F1, F2, F3 solved numerically in time (Taruya et al. 2014;
Bose & Koyama 2016) for a background E(a), a linear modification mu(k, a) and the screening sources gamma2, gamma3:
GR on any background (`mg_lcdm`), Hu-Sawicki f(R) n = 1 (`mg_hu_sawicki`) and normal-branch DGP (`mg_ndgp`), or any
`MGKernelModel`. Used for k* of the halo-model reaction (`HaloModel/HaloModelReaction.h`).

```cpp
#include <FML/PerturbationTheory/MGOneLoop.h>
using namespace FML::PERTURBATION;
auto model = mg_hu_sawicki(1e-5, OmegaM, Omega_cb);          // or mg_ndgp(Omega_rc, ...), mg_lcdm(OmegaM, Omega_cb)
std::vector<double> a = {1.0 / 3.0, 0.5, 1.0};                // increasing
auto P = [&](double k, size_t i) { return P_lin_at_a_i(k); }; // the linear spectrum at a[i] (h/Mpc, (Mpc/h)^3)
auto Ploop = mg_one_loop(model, 0.06, a, P);                  // P_L + P22 + P13 at each a
```

The kernels are evolved for every loop node (P22 over q and p = |k - q|, P13 over q and the cosine; RK4 in log a
from a = 1e-5), so P0 = P_L / F1^2 with the model's own F1. ~1 s for one k at four redshifts on 4 threads.

Validation (`tests/test_mg_oneloop.cpp`): EdS = `OneLoop.h` (6e-4 of the loop); |fR0| = 1e-10 and Omega_rc = 1e-8
give the GR loop; the f(R) gamma2 and gamma3 reproduce the exact expansion of the field perturbation in the
homogeneous limit; against ReACT's one-loop P(k0 = 0.06) for its fixtures (z = 0 - 2): 4e-3 (ReACT's angular
quadrature misses ~4% of the loop at k0) and the real - pseudo difference to 0.7% (nDGP) and 3% (f(R), with ReACT's
gamma3, which is 1/3 of the correct one).
