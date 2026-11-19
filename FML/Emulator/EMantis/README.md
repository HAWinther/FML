# e-MANTIS emulators

`EMantis.h` is a C++ port of the prediction part of [e-MANTIS](https://gitlab.obspm.fr/e-mantis/e-mantis) 1.4.0
(I. Sáez-Casares et al., GPL-3.0-or-later): Gaussian-process emulators trained on N-body simulations. Header only, no
dependencies. Cite the papers when using them:

| Class | Emulates | Parameters | Paper |
| --- | --- | --- | --- |
| `NonLinearMGBoostEmulator("fR")` | B(k, a) = P_f(R) / P_ΛCDM, nonlinear, Hu-Sawicki n = 1 | Ω_b, Ω_m, h, n_s, σ8_ΛCDM (or A_s), log10\|fR0\| ∈ [4, 7] | [arXiv:2303.08899](https://arxiv.org/abs/2303.08899) |
| `NonLinearBoostEmulator("wCDM")` | P_nl(k, a) / P_lin(k, a) | Ω_b, Ω_m, h, n_s, σ8 (or A_s), w ∈ [-1.5, -0.5] | none yet; cite the package ([doi:10.5281/zenodo.7738362](https://doi.org/10.5281/zenodo.7738362)) |
| `HMFEmulator(model, mass_def, res_corr)` | dn/dlnM in (h/Mpc)³ and its std; `200c`, `500c`, `1000c`, FoF `b0.2` | as above, `fR` or `wCDM` | [arXiv:2410.05226](https://arxiv.org/abs/2410.05226) |
| `Sigma8Emulator("LCDM" / "wCDM")` | σ8 from A_s (used when A_s is given) | Ω_b, Ω_m, h, n_s, A_s (, w) | |

Ranges: 0.25 ≤ a ≤ 1, Ω_b ∈ [0.037, 0.062], Ω_m ∈ [0.155, 0.465], h ∈ [0.55, 0.85], n_s ∈ [0.72, 1.2],
σ8 ∈ [0.6083, 1.014]; k ∈ [0.029, 9.7] h/Mpc for the f(R) boost (extrapolated below) and [0.019, 10.1] h/Mpc for
the wCDM boost; masses in Msun/h,
range depending on a (`range("mass_halo", a)`). Each class has `range(name)`.

```cpp
#include <FML/Emulator/EMantis/EMantis.h>
using namespace FML::EMULATOR::EMANTIS;

Parameters p;                       // every emulator reads the parameters it needs
p.Omega_b = 0.049; p.Omega_m = 0.31; p.h = 0.68; p.n_s = 0.965;
p.sigma8 = 0.81;                    // sigma8_lcdm for f(R), or p.A_s = 2.1e-9 instead
p.logfR0 = 5.0;                     // |fR0| = 1e-5

NonLinearMGBoostEmulator fr("fR");
std::vector<double> B = fr.boost(p, a, k);           // k in h/Mpc; fr.kbins() if k is empty

HMFEmulator hmf("fR", "200c");                        // with the resolution correction (default)
std::vector<double> std;
std::vector<double> n = hmf.hmf(masses, p, a, false, &std);
```

The rules are those of the Python package: inputs out of range throw `EmulationRangeError` (with the parameter name),
the f(R) boost is extrapolated below the first k bin (at most the boost there, `extrapolate_k_low`), its cosmological
parameters other than fR0 can be clamped to the range (`extrapolate_cosmo`), and it is never below 1; the HMF is
extrapolated above the mass range only on request (not reliable). Between the scale factor nodes the predictions are
interpolated linearly in a (in log10 of the observable for the wCDM boost and the HMF). The classes are const after
construction, so they can be used from several threads. Loading takes ~5 ms (f(R) boost) to ~50 ms (wCDM boost), and
a prediction at 200 k values ~0.02-0.04 ms.

## Data and accuracy

`export_emantis.py` loads every shipped emulator with the Python package and writes `data/*.txt` (plain text, 15 MB):
the parameter ranges and scaled training parameters, per scale factor node the bins or B-spline knots and the PCA,
and per Gaussian process its Matérn kernel hyperparameters, normalisation and weights α = K⁻¹y (precomputed, so the
training targets are not needed). The std of the HMF needs the Cholesky factor of K, which is computed at load.

```sh
pip install emantis==1.4.0
python3 export_emantis.py data
```

`GPEmulator` reads any file in this format, so emulators trained with e-MANTIS' `custom_emulators` can be exported
with `write_emulator` from the script as well.

The tests (`tests/`, `make run`) compare with the Python package (`tests/make_emantis_reference.py` writes
`EMantisReference.h`). The agreement is ~1e-10 relative, and a few 1e-8 when A_s is converted to σ8: the GP
predictions are sums with strong cancellations (Σ|k_i α_i| is up to 10⁹ times the result for σ8), so this is the
floating point reproducibility of the emulators themselves; numpy's result moves by as much if the order of the
summation changes. This is far below the emulation errors (percent level).

Not ported: the deprecated `FofrBoost` / `fR_v1` emulator, training, cross-validation, and batching over many
cosmologies at once (call the classes in a loop; each call is cheap).
