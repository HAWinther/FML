# baccoemu emulators

`Bacco.h` is a C++ port of the prediction part of [baccoemu](https://bitbucket.org/rangulo/baccoemu) 2.3.0
(R. Angulo, M. Zennaro, G. Aricò, M. Pellejero Ibáñez, P. Burger; MIT licence, see [LICENSE](LICENSE)): neural-network
emulators of the matter power spectrum and of the Lagrangian bias expansion (real and redshift space) trained on the
BACCO simulations, with massive neutrinos and w0-wa dark energy. Header only; it uses FML's [NeuralNetwork](../NeuralNetwork/README.md) for the ReLU networks. Cite the papers
when using them:

| Function | Emulates | Paper |
| --- | --- | --- |
| `linear_pk`, `sigma8`, `sigma12`, `A_s` | linear P(k) of the cold (cdm + baryons) or total matter; σ8, σ12 | [arXiv:2104.14568](https://arxiv.org/abs/2104.14568) |
| `nonlinear_boost`, `nonlinear_pk` with `nonlinear_model = "Angulo2021"` (default) | P_nl / P_lin, cold or total matter | [arXiv:2004.06245](https://arxiv.org/abs/2004.06245) |
| the same with `nonlinear_model = "Arico2023"` | the same, wider parameter ranges | [arXiv:2303.05537](https://arxiv.org/abs/2303.05537) |
| `baryonic_boost` with `baryonic_model = "Arico2021"` (default) | S(k) = P_baryons / P_gravity-only (baryonification) | [arXiv:2011.15018](https://arxiv.org/abs/2011.15018) |
| the same with `baryonic_model = "Burger2025"`, and `MatterBispectrum` | S(k) of P(k), and of the bispectrum | [arXiv:2506.18974](https://arxiv.org/abs/2506.18974) |
| `no_wiggles_pk`, `smeared_bao_pk` | the no-wiggle and BAO-smeared P(k) used for the nonlinear boost | as the nonlinear boost |
| `LbiasExpansion`: `nonlinear_pnn`, `lpt_pk`, `galaxy_real_pk` | the 15 spectra of the 2nd order Lagrangian bias expansion (nonlinear, LPT); P_gg, P_gm | [arXiv:2101.12187](https://arxiv.org/abs/2101.12187), LPT [arXiv:2104.14568](https://arxiv.org/abs/2104.14568) |
| `LbiasExpansionRSD`: `nonlinear_pnn`, `lpt_pk`, `galaxy_pk`, `galaxy_2d_pk` | the same in redshift space (multipoles 0, 2, 4); galaxy multipoles and P(k, μ) with fingers of god and noise | [arXiv:2207.06437](https://arxiv.org/abs/2207.06437) (LPT: an emulator of velocileptors, [arXiv:2012.04636](https://arxiv.org/abs/2012.04636)) |
| `COEVOLUTION::` | coevolution relations of the bias parameters (haloes, galaxies) | [arXiv:2110.05408](https://arxiv.org/abs/2110.05408) |

```cpp
#include <FML/Emulator/Bacco/Bacco.h>
using namespace FML::EMULATOR::BACCO;

Parameters p;                         // NaN = not given
p.Omega_cold = 0.31;                  // Omega_cdm + Omega_b, or p.Omega_m (= Omega_cold + M_nu / (93.14 h^2))
p.Omega_b = 0.049;
p.sigma8_cold = 0.81;                 // or p.A_s (at k = 0.05/Mpc)
p.h = 0.67; p.n_s = 0.965;
p.M_nu = 0.06; p.w0 = -1.0; p.wa = 0.0;   // defaults 0, -1, 0 (baccoemu has none)

MatterPowerSpectrum::Options opt;     // the arguments of baccoemu.Matter_powerspectrum, same defaults
opt.nonlinear_model = "Arico2023";
MatterPowerSpectrum emu(opt);

auto plin = emu.linear_pk(p, a, k, /*cold=*/false);             // k in h/Mpc, P in (Mpc/h)^3; k empty: the bins
auto pnl  = emu.nonlinear_pk(p, a, k, /*cold=*/false);
p.M_c = 14; p.eta = -0.3; p.beta = -0.22; p.M1_z0_cen = 10.5;    // baryons (log10 masses in Msun/h)
p.theta_out = 0.25; p.theta_inn = -0.86; p.M_inn = 13.4;
auto S    = emu.baryonic_boost(p, a, k);
auto pbar = emu.nonlinear_pk(p, a, k, false, /*baryons=*/true);

MatterBispectrum bk;                  // Burger et al. 2025
auto r = bk.baryonic_boost(p, a, k1, k2, k3);                   // r.k (sorted triangles), r.boost, r.extrapolated

std::array<double, 4> bias = {b1, b2, bs2, blaplacian};          // Lagrangian
LbiasExpansion lb;                                               // real space, k <= 0.75 h/Mpc
auto pnn = lb.nonlinear_pnn(p, a, k);                            // pnn[n][ik], n = 11, 1d, 1d2, 1s2, 1n2, dd, ...
auto g   = lb.galaxy_real_pk(bias, p, a, k);                     // g.auto_pk, g.cross_pk
LbiasExpansionRSD lbz;                                           // redshift space, k <= 0.64 h/Mpc
LbiasExpansionRSD::RSDParameters rsd;                            // f_sat, lambda_FoG, optional noise
rsd.f_sat = 0.1; rsd.lambda_FoG = 2.5;                           // (epsilon_1, epsilon_2, epsilon_3, mean_num_dens)
auto ell = lbz.galaxy_pk(bias, rsd, p, a, k);                    // P_0, P_2, P_4
auto b2  = COEVOLUTION::gal_b2L_b1L(b1);                         // coevolution relations
```

The rules are those of the Python package. Parameters outside the training range of a network throw
`EmulationRangeError` (with the baccoemu name of the parameter); `ranges(which)` lists them. The linear and smeared-BAO
P(k) need k inside their bins; the nonlinear boost needs k below its largest bin and is 1 below its smallest; the
Arico2021 baryonic boost is extrapolated above its bins (not accurate) and 1 below. A linear P(k) can be given to
`nonlinear_boost` / `nonlinear_pk` (`k_lin`, `pk_lin`, covering [1e-3, 10] h/Mpc) instead of the emulated one. The
classes are const after construction (thread safe). Loading takes ~0.6 s (25 MB of text), `nonlinear_pk` ~2 ms.

Ranges (Angulo2021 / Arico2023): Ω_cold ∈ [0.23, 0.4] / [0.15, 0.47], σ8_cold ∈ [0.73, 0.9] / [0.4, 1.15],
Ω_b ∈ [0.04, 0.06] / [0.03, 0.07], n_s ∈ [0.92, 1.01] / [0.83, 1.1], h ∈ [0.6, 0.8] / [0.5, 0.9], M_ν ∈ [0, 0.4] eV,
w0 ∈ [-1.15, -0.85] / [-1.4, -0.6], wa ∈ [-0.3, 0.3] / [-0.5, 0.5], a ∈ [0.4, 1] / [0.275, 1.1]; k ∈ [0.01, 4.9] / [0.01, 9.8] h/Mpc.
The linear emulator covers a wider range (Ω_cold ∈ [0.06, 0.7], a ∈ [0.1, 1], k ∈ [1e-4, 50] h/Mpc), the Arico2021
baryonic boost k ∈ [0.011, 4.7] h/Mpc (other ranges as Angulo2021), Burger2025 k ∈ [0.018, 17.7] h/Mpc.

The bias expansion follows baccoemu exactly, including its choices: the spectra are interpolated in k with a cubic
spline (not-a-knot, as scipy's `interp1d(kind='cubic')`) in ln k, on P itself (baccoemu meant to interpolate some in
ln P, but its test `np.any(...) is False` is never true); in redshift space some LPT terms are replaced or held
constant above k = 0.2 h/Mpc as in `get_nonlinear_pnn`, and the galaxy multipoles are projected with baccoemu's
`pkmulti` (a 4th order polynomial fit in μ on μ = 0, 0.01, ..., 0.99 and its 8-point Gauss-Legendre rule).
Loading takes ~0.2 s (real space, 15 + 15 networks) and ~0.6 s (redshift space, 93 networks).

Not ported: the BAO smearing from a no-wiggle spectrum made by a DST of the linear P(k) (used by baccoemu only when
both `smeared_bao` and `no_wiggles` are off; here exactly one must be on), the bias emulators' `pk_lpt_in` /
`pk_bao_in` inputs, the late-time ΛCDM `Hubble_latetime_LCDM` / `comoving_radial_distance` helpers (FML has its own
background) and the baryon-fraction helper `get_baryon_fractions`.

## Data

The trained networks are not distributed with FML (they come from bacco.dipc.org, which serves them to baccoemu at
run time). Get them once with

```sh
cd FML/Emulator/Bacco
./get_data.sh               # all emulators: ~105 MB download, 98 MB in data/
./get_data.sh --matter      # only MatterPowerSpectrum and MatterBispectrum: ~10 MB download, 9 MB in data/
./get_data.sh --clean       # ... and delete the downloaded tarballs (weights/) afterwards
BACCO_URL=https://my.mirror/bacco ./get_data.sh   # from a mirror with the same files
```

It needs curl or wget and Python with numpy and h5py (`python3 -m pip install --user numpy h5py`; another Python with
`--python`). The script downloads the tarballs of baccoemu 2.3.0 into `weights/`, checks them against pinned SHA-256
sums (the server's certificate chain is incomplete, so, like baccoemu itself, the download does not verify the
certificate; the checksums verify the files), and converts them with `export_bacco.py` into `data/`. Both folders are ignored by git. Without the
data the constructors throw `MissingDataError`, whose message says how to get it; `data_available()` tells whether it
is there. If the data is elsewhere, call `FML::EMULATOR::BACCO::set_data_dir("path/to/data")` first.

The ReLU networks are written as `NAME.bin` (`FML_BACCO_NN 1`: a text header with the layer sizes and the input and
output scaling, i.e. the bounds scaling and baccoemu's output scaler, then the weights as little-endian float32, the
exact values of the Keras files); they are loaded into FML's [NeuralNetwork](../NeuralNetwork/README.md). With them
`NAME_details.txt` (parameter names, bounds, k bins, and the constant subtracted after exp() for the bias spectra).
The Burger et al. networks (CosmoPower activation (β + sigmoid(α x)(1 − β)) x with α, β per neuron) are text,
`FML_BACCO_COSMOPOWER 1` (weights with 9 digits, rounded back to float32 when loading).

## Accuracy

The tests (`tests/`, `make run`) compare 251 cases with baccoemu 2.3.0 (`tests/make_bacco_reference.py` writes
`BaccoReference.h`): every function, both nonlinear and both baryonic models, cold and total matter with neutrinos,
the no-wiggle and smeared-BAO paths, a given linear P(k), default bins and given k, the bias expansion in real and
redshift space (LPT, smeared BAO, the 15 nonlinear spectra, galaxy spectra, multipoles with and without noise, P(k, μ)),
and 18 out-of-range errors (the same parameter as in Python). The reference evaluates the same networks in float64,
and the agreement is ~1e-15 (relative; for the bias spectra, which cross zero, relative to the largest value).
baccoemu itself evaluates in float32, which moves its predictions by up to ~1.5e-6 (relative); the C++ evaluation is
in double. The accuracy of the emulators against the simulations is given in the papers.
