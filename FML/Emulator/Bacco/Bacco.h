#ifndef FML_EMULATOR_BACCO_H
#define FML_EMULATOR_BACCO_H

//=====================================================================================================================
// baccoemu in C++: neural-network emulators of the linear and nonlinear matter power spectrum (with massive
// neutrinos and w0-wa dark energy), its baryonic suppression and the baryonic suppression of the matter bispectrum,
// trained on the BACCO simulations.
//
//   baccoemu: R. Angulo, M. Zennaro, G. Arico, M. Pellejero Ibanez, P. Burger, https://bitbucket.org/rangulo/baccoemu
//   (MIT licence: see LICENSE in this folder; the notice must be kept with copies of the code and the data files)
//   Linear P(k) (cold and total matter):          Arico, Angulo & Zennaro 2021 (arXiv:2104.14568)
//   Nonlinear boost "Angulo2021":                 Angulo et al. 2021 (arXiv:2004.06245)
//   Nonlinear boost "Arico2023" (wider ranges):   Arico et al. 2023 (arXiv:2303.05537)
//   Baryonic boost of P(k) "Arico2021":           Arico et al. 2021 (arXiv:2011.15018)
//   Baryonic boost of P(k) and B(k) "Burger2025": Burger et al. 2025 (arXiv:2506.18974)
//   Lagrangian bias expansion, real space:        Zennaro et al. 2021 (arXiv:2101.12187); LPT: arXiv:2104.14568
//   Lagrangian bias expansion, redshift space:    Pellejero Ibanez et al. 2023 (arXiv:2207.06437)
//   Cite the papers when using the emulators.
//
// This is a port of the prediction part of baccoemu 2.3.0 (Matter_powerspectrum, Matter_bispectrum,
// Lbias_expansion, Lbias_expansion_RSD and the coevolution relations). The networks are converted by export_bacco.py
// to the files in data/. They are evaluated in double precision (baccoemu uses float32): the predictions agree with
// baccoemu evaluated in float64 to ~1e-14, and with baccoemu as it is (float32) to ~1e-6.
//
//   MatterPowerSpectrum   sigma8 / sigma12 / A_s, linear P(k) (cold or total matter), the no-wiggle and the
//                         BAO-smeared P(k), the nonlinear boost and P_nl(k), the baryonic boost (Arico2021 or
//                         Burger2025)
//   MatterBispectrum      the baryonic boost of the matter bispectrum (Burger2025)
//   LbiasExpansion        the 15 spectra of the second order Lagrangian bias expansion (nonlinear and LPT), the
//                         galaxy auto and galaxy-matter spectra, real space
//   LbiasExpansionRSD     the same in redshift space (multipoles 0, 2, 4), the galaxy multipoles and P(k, mu) with
//                         fingers of god and noise
//   COEVOLUTION::         the coevolution relations of the bias parameters (Zennaro et al. 2022)
//
// Conventions are baccoemu's: k in h/Mpc, P(k) in (Mpc/h)^3, Omega_cold = Omega_cdm + Omega_b,
// Omega_nu = M_nu / (93.14 h^2), sigma8_cold is the sigma8 of the cold matter at z = 0, A_s at k = 0.05 / Mpc.
// Inputs outside the training ranges throw EmulationRangeError (as the assertions in Python). Everything is const
// after construction (thread safe).
//
// The trained networks are not distributed with FML: run get_data.sh in this folder once (it downloads them from
// bacco.dipc.org and converts them with export_bacco.py into data/). Without them the constructors throw
// MissingDataError, which says how to get them.
//=====================================================================================================================

#include <FML/Emulator/NeuralNetwork/NeuralNetwork.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <fstream>
#include <map>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace FML {
    namespace EMULATOR {
        namespace BACCO {

            //=========================================================================================================
            // Where the data files are: by default the data/ folder next to this header
            //=========================================================================================================
#ifdef __FILE__
            inline std::string data_dir =
                std::string(__FILE__).substr(0, std::string(__FILE__).find_last_of("/\\") + 1) + "data/";
#else
            inline std::string data_dir = "data/";
#endif
            inline void set_data_dir(std::string dir) {
                if (not dir.empty() and dir.back() != '/')
                    dir += '/';
                data_dir = dir;
            }

            namespace detail {
                inline std::string to_str(double x) {
                    std::ostringstream s;
                    s << x;
                    return s.str();
                }
            } // namespace detail

            /// A parameter outside the training range of an emulator (an AssertionError in baccoemu). param is the
            /// baccoemu name (omega_cold, sigma8_cold, omega_baryon, ns, hubble, neutrino_mass, w0, wa, expfactor,
            /// M_c, eta, beta, M1_z0_cen, theta_inn, M_inn, theta_out)
            class EmulationRangeError : public std::runtime_error {
              public:
                std::string param;
                double value, min_value, max_value;
                EmulationRangeError(std::string emulator, std::string param_, double value_, double min_, double max_)
                    : std::runtime_error("baccoemu " + emulator + ": " + param_ + " = " + detail::to_str(value_) +
                                         " is outside the emulation range [" + detail::to_str(min_) + ", " +
                                         detail::to_str(max_) + "]"),
                      param(param_), value(value_), min_value(min_), max_value(max_) {}
            };

            /// The networks are not part of FML: this is thrown when a data file is missing, with how to get them
            class MissingDataError : public std::runtime_error {
              public:
                std::string filename;
                explicit MissingDataError(std::string file)
                    : std::runtime_error(
                          "baccoemu: the data file " + file +
                          " is missing. The baccoemu networks are not distributed with FML: download and convert them "
                          "once with\n    FML/Emulator/Bacco/get_data.sh            (all emulators, ~105 MB download)"
                          "\n    FML/Emulator/Bacco/get_data.sh --matter   (only MatterPowerSpectrum / "
                          "MatterBispectrum, ~10 MB)\n(needs curl or wget and Python with numpy and h5py). If the data "
                          "is elsewhere, call FML::EMULATOR::BACCO::set_data_dir(\"path/to/data\") first."),
                      filename(file) {}
            };

            /// True if the data files of the matter emulators (all = true: of all the emulators) are present
            inline bool data_available(bool all = false) {
                auto exists = [](const std::string & f) { return bool(std::ifstream(data_dir + f)); };
                bool ok = exists("sigma8_emu_2.0.2.bin") and exists("nonlinear_emu_1.0.3.bin") and
                          exists("baryon_boost_Burger2025_bk.txt");
                if (all)
                    ok = ok and exists("lbias_nonlinear_field14.bin") and exists("lbias_rsd_nonlinear_field14_ell4.bin");
                return ok;
            }

            //=========================================================================================================
            /// The parameters. NaN means not given. Give Omega_cold or Omega_m (or both, consistently) and sigma8_cold
            /// or A_s (or both, consistently); the other one is computed with the sigma8 emulator when needed. The
            /// baryon parameters are only used by the baryonic boosts (masses are log10 in Msun/h, as in baccoemu).
            /// baccoemu has no defaults; here M_nu = 0, w0 = -1, wa = 0.
            //=========================================================================================================
            struct Parameters {
                double Omega_cold{NAN};  ///< Omega_cdm + Omega_b
                double Omega_m{NAN};     ///< Omega_cold + Omega_nu
                double Omega_b{NAN};     ///< Omega_b
                double sigma8_cold{NAN}; ///< sigma8 of cdm + baryons (linear, z = 0)
                double A_s{NAN};         ///< primordial amplitude at k = 0.05 / Mpc
                double h{NAN};           ///< H0 / (100 km/s/Mpc)
                double n_s{NAN};         ///< spectral index
                double M_nu{0.0};        ///< sum of the neutrino masses in eV
                double w0{-1.0};         ///< dark energy w(a) = w0 + wa (1 - a)
                double wa{0.0};
                // Baryons (Arico et al. 2021 / Burger et al. 2025)
                double M_c{NAN};       ///< log10 of the halo mass where half the gas is retained
                double eta{NAN};       ///< log10 of the extent of the ejected gas
                double beta{NAN};      ///< log10 of the slope of the retained gas fraction
                double M1_z0_cen{NAN}; ///< log10 of the characteristic halo mass of the central galaxies
                double theta_out{NAN}; ///< log10 of the outer slope of the hot gas profile (Arico2021)
                double theta_inn{NAN}; ///< log10 of the inner radius of the hot gas profile
                double M_inn{NAN};     ///< log10 of the mass at which the hot gas profile changes slope (Arico2021)
            };

            //=========================================================================================================
            /// The bounds and k bins of a network (data/NAME_details.txt)
            //=========================================================================================================
            struct Details {
                std::vector<std::string> keys;
                std::vector<std::array<double, 2>> bounds;
                std::vector<double> k;
                double subtract{0.0}; ///< subtracted after exp() (bias expansion emulators)

                void load(const std::string & filename) {
                    std::ifstream in(filename);
                    if (not in)
                        throw MissingDataError(filename);
                    std::stringstream tok;
                    std::string line;
                    while (std::getline(in, line))
                        tok << line.substr(0, line.find('#')) << "\n";
                    auto expect = [&](const std::string & w) {
                        std::string s;
                        if (not(tok >> s) or s != w)
                            throw std::runtime_error("baccoemu: expected '" + w + "' in " + filename);
                    };
                    int version, n;
                    expect("FML_BACCO_DETAILS");
                    tok >> version;
                    expect("keys");
                    tok >> n;
                    keys.resize(size_t(n));
                    for (auto & s : keys)
                        tok >> s;
                    expect("bounds");
                    bounds.resize(size_t(n));
                    for (auto & b : bounds)
                        tok >> b[0] >> b[1];
                    expect("k");
                    tok >> n;
                    k.resize(size_t(n));
                    for (auto & x : k)
                        tok >> x;
                    if (not tok)
                        throw std::runtime_error("baccoemu: bad file " + filename);
                    std::string w;
                    if (tok >> w and w == "subtract")
                        tok >> subtract;
                }
            };

            //=========================================================================================================
            /// A network with the CosmoPower activation (b + sigmoid(a x) (1 - b)) x with a and b per neuron
            /// (Burger et al. 2025; data/baryon_boost_Burger2025_*.txt). Inputs are named (parameters()).
            //=========================================================================================================
            class CosmoPowerNetwork {
              public:
                void load(const std::string & filename) {
                    std::ifstream in(filename);
                    if (not in)
                        throw MissingDataError(filename);
                    std::stringstream tok;
                    std::string line;
                    while (std::getline(in, line))
                        tok << line.substr(0, line.find('#')) << "\n";
                    auto expect = [&](const std::string & w) {
                        std::string s;
                        if (not(tok >> s) or s != w)
                            throw std::runtime_error("baccoemu: expected '" + w + "' in " + filename);
                    };
                    auto read_vec = [&](const std::string & name, size_t n) {
                        expect(name);
                        std::vector<double> v(n);
                        for (auto & x : v)
                            tok >> x;
                        return v;
                    };
                    int version, n;
                    expect("FML_BACCO_COSMOPOWER");
                    tok >> version;
                    expect("parameters");
                    tok >> n;
                    params.resize(size_t(n));
                    for (auto & s : params)
                        tok >> s;
                    expect("sizes");
                    tok >> n;
                    sizes.resize(size_t(n));
                    for (auto & s : sizes)
                        tok >> s;
                    if (sizes.size() < 2 or size_t(sizes.front()) != params.size())
                        throw std::runtime_error("baccoemu: bad sizes in " + filename);
                    param_mean = read_vec("param_mean", params.size());
                    param_std = read_vec("param_std", params.size());
                    feature_mean = read_vec("feature_mean", size_t(sizes.back()));
                    feature_std = read_vec("feature_std", size_t(sizes.back()));
                    expect("scaling");
                    tok >> scaling_division >> scaling_subtraction;
                    expect("ranges");
                    tok >> n;
                    for (int i = 0; i < n; i++) {
                        std::string key;
                        double lo, hi;
                        tok >> key >> lo >> hi;
                        ranges[key] = {lo, hi};
                    }
                    const size_t nl = sizes.size() - 1;
                    W.resize(nl);
                    b.resize(nl);
                    alpha.resize(nl - 1);
                    beta.resize(nl - 1);
                    for (size_t l = 0; l < nl; l++) {
                        expect("layer");
                        tok >> n;
                        W[l].resize(size_t(sizes[l]) * size_t(sizes[l + 1]));
                        for (auto & x : W[l]) { // float32 in baccoemu, 9 digits in the file: the exact value
                            tok >> x;
                            x = double(float(x));
                        }
                        b[l].resize(size_t(sizes[l + 1]));
                        for (auto & x : b[l]) {
                            tok >> x;
                            x = double(float(x));
                        }
                        if (l + 1 < nl) {
                            alpha[l] = read_vec("alpha", size_t(sizes[l + 1]));
                            beta[l] = read_vec("beta", size_t(sizes[l + 1]));
                        }
                    }
                    if (not tok)
                        throw std::runtime_error("baccoemu: bad file " + filename);
                }

                /// The input names, in order
                const std::vector<std::string> & parameters() const { return params; }
                /// The training ranges, by name
                const std::map<std::string, std::array<double, 2>> & parameter_ranges() const { return ranges; }

                /// The (rescaled) prediction at x (inputs in the order of parameters())
                std::vector<double> operator()(const std::vector<double> & x) const {
                    if (x.size() != params.size())
                        throw std::invalid_argument("baccoemu CosmoPowerNetwork: the input has the wrong size");
                    std::vector<double> a(x.size()), z;
                    for (size_t j = 0; j < x.size(); j++)
                        a[j] = (x[j] - param_mean[j]) / param_std[j];
                    for (size_t l = 0; l + 1 < sizes.size(); l++) {
                        const size_t nin = size_t(sizes[l]), nout = size_t(sizes[l + 1]);
                        z.assign(nout, 0.0);
                        for (size_t o = 0; o < nout; o++) {
                            double s = 0.0;
                            const double * w = W[l].data() + o * nin;
                            for (size_t i = 0; i < nin; i++)
                                s += w[i] * a[i];
                            s += b[l][o];
                            if (l + 2 < sizes.size()) {
                                const double sig = 1.0 / (1.0 + std::exp(-alpha[l][o] * s));
                                s = (beta[l][o] + sig * (1.0 - beta[l][o])) * s;
                            }
                            z[o] = s;
                        }
                        a.swap(z);
                    }
                    for (size_t o = 0; o < a.size(); o++)
                        a[o] = (a[o] * feature_std[o] + feature_mean[o]) * scaling_division + scaling_subtraction;
                    return a;
                }

              private:
                std::vector<std::string> params;
                std::vector<int> sizes;
                std::vector<double> param_mean, param_std, feature_mean, feature_std;
                double scaling_division{1.0}, scaling_subtraction{0.0};
                std::map<std::string, std::array<double, 2>> ranges;
                std::vector<std::vector<double>> W, b, alpha, beta;
            };

            namespace detail {

                /// scipy.interpolate.interp1d(x, y, kind='linear', fill_value='extrapolate') at xi (x increasing)
                inline double interp_extrap(const std::vector<double> & x, const std::vector<double> & y, double xi) {
                    const size_t n = x.size();
                    size_t i = size_t(std::lower_bound(x.begin(), x.end(), xi) - x.begin()); // np.searchsorted, left
                    i = std::min(std::max(i, size_t(1)), n - 1);
                    const double slope = (y[i] - y[i - 1]) / (x[i] - x[i - 1]);
                    return slope * (xi - x[i - 1]) + y[i - 1];
                }

                /// Linear interpolation of y in x, with fill outside [x0, xn]
                inline double interp_fill(const std::vector<double> & x,
                                          const std::vector<double> & y,
                                          double xi,
                                          double fill) {
                    if (xi < x.front() or xi > x.back() or std::isnan(xi))
                        return fill;
                    return interp_extrap(x, y, xi);
                }

                inline std::vector<double> log_of(const std::vector<double> & v) {
                    std::vector<double> r(v.size());
                    for (size_t i = 0; i < v.size(); i++)
                        r[i] = std::log(v[i]);
                    return r;
                }

                /// exp(interp(log x, log y)(log xi)) with extrapolation (the log-log interpolation of baccoemu)
                inline std::vector<double> interp_loglog(const std::vector<double> & x,
                                                         const std::vector<double> & y,
                                                         const std::vector<double> & xi) {
                    const auto lx = log_of(x), ly = log_of(y);
                    std::vector<double> r(xi.size());
                    for (size_t i = 0; i < xi.size(); i++)
                        r[i] = std::exp(interp_extrap(lx, ly, std::log(xi[i])));
                    return r;
                }

                /// numpy.trapezoid(y, x)
                inline double trapezoid(const std::vector<double> & y, const std::vector<double> & x) {
                    double s = 0.0;
                    for (size_t i = 0; i + 1 < x.size(); i++)
                        s += (x[i + 1] - x[i]) * (y[i + 1] + y[i]) / 2.0;
                    return s;
                }

                /// scipy.interpolate.interp1d(x, y, kind='cubic', fill_value='extrapolate'): the not-a-knot cubic
                /// spline, extrapolated with the end pieces (x increasing, at least 4 points)
                class CubicSpline {
                  public:
                    CubicSpline(const std::vector<double> & x_, const std::vector<double> & y_) : x(x_), y(y_) {
                        const size_t n = x.size();
                        if (n < 4 or y.size() != n)
                            throw std::invalid_argument("baccoemu CubicSpline: need >= 4 points");
                        std::vector<double> dx(n - 1), slope(n - 1);
                        for (size_t i = 0; i + 1 < n; i++) {
                            dx[i] = x[i + 1] - x[i];
                            slope[i] = (y[i + 1] - y[i]) / dx[i];
                        }
                        // The tridiagonal system for the slopes s (as scipy's CubicSpline, bc_type = 'not-a-knot')
                        std::vector<double> lo(n, 0.0), di(n, 0.0), up(n, 0.0), b(n, 0.0);
                        for (size_t i = 1; i + 1 < n; i++) {
                            lo[i] = dx[i];
                            di[i] = 2.0 * (dx[i - 1] + dx[i]);
                            up[i] = dx[i - 1];
                            b[i] = 3.0 * (dx[i] * slope[i - 1] + dx[i - 1] * slope[i]);
                        }
                        double d = x[2] - x[0];
                        di[0] = dx[1];
                        up[0] = d;
                        b[0] = ((dx[0] + 2.0 * d) * dx[1] * slope[0] + dx[0] * dx[0] * slope[1]) / d;
                        d = x[n - 1] - x[n - 3];
                        di[n - 1] = dx[n - 3];
                        lo[n - 1] = d;
                        b[n - 1] = (dx[n - 2] * dx[n - 2] * slope[n - 3] + (2.0 * d + dx[n - 2]) * dx[n - 3] * slope[n - 2]) / d;
                        // Gaussian elimination with partial pivoting on the band (rows i, i+1)
                        s = solve_tridiagonal(lo, di, up, b);
                        c2.resize(n - 1);
                        c3.resize(n - 1);
                        for (size_t i = 0; i + 1 < n; i++) {
                            c2[i] = (3.0 * slope[i] - 2.0 * s[i] - s[i + 1]) / dx[i];
                            c3[i] = (s[i] + s[i + 1] - 2.0 * slope[i]) / (dx[i] * dx[i]);
                        }
                    }
                    double operator()(double xi) const {
                        size_t i = size_t(std::upper_bound(x.begin(), x.end(), xi) - x.begin());
                        i = std::min(std::max(i, size_t(1)), x.size() - 1) - 1;
                        const double t = xi - x[i];
                        return y[i] + t * (s[i] + t * (c2[i] + t * c3[i]));
                    }

                  private:
                    std::vector<double> x, y, s, c2, c3;
                    static std::vector<double> solve_tridiagonal(std::vector<double> lo,
                                                                 std::vector<double> di,
                                                                 std::vector<double> up,
                                                                 std::vector<double> b) {
                        // Dense band elimination with partial pivoting (a pivot swap adds one super-diagonal)
                        const size_t n = di.size();
                        std::vector<double> up2(n, 0.0);
                        for (size_t i = 0; i + 1 < n; i++) {
                            if (std::abs(lo[i + 1]) > std::abs(di[i])) {
                                std::swap(lo[i + 1], di[i]);
                                std::swap(di[i + 1], up[i]);
                                if (i + 2 < n)
                                    std::swap(up[i + 1], up2[i]);
                                std::swap(b[i], b[i + 1]);
                            }
                            const double m = lo[i + 1] / di[i];
                            di[i + 1] -= m * up[i];
                            if (i + 2 < n)
                                up[i + 1] -= m * up2[i];
                            b[i + 1] -= m * b[i];
                        }
                        std::vector<double> r(n);
                        for (size_t i = n; i-- > 0;) {
                            double v = b[i];
                            if (i + 1 < n)
                                v -= up[i] * r[i + 1];
                            if (i + 2 < n)
                                v -= up2[i] * r[i + 2];
                            r[i] = v / di[i];
                        }
                        return r;
                    }
                };

                /// The cubic spline of y(ln x_in) at ln x_out
                inline std::vector<double> interp_cubic_logx(const std::vector<double> & x_in,
                                                             const std::vector<double> & y,
                                                             const std::vector<double> & x_out) {
                    CubicSpline sp(log_of(x_in), y);
                    std::vector<double> r(x_out.size());
                    for (size_t i = 0; i < r.size(); i++)
                        r[i] = sp(std::log(x_out[i]));
                    return r;
                }

                /// numpy.polyfit(x, y, deg): least squares on the column-scaled Vandermonde matrix (Householder QR).
                /// Returns the coefficients, highest power first
                inline std::vector<double> polyfit(const std::vector<double> & x, const std::vector<double> & y, int deg) {
                    const size_t m = x.size(), nc = size_t(deg + 1);
                    std::vector<std::vector<double>> A(m, std::vector<double>(nc));
                    for (size_t j = 0; j < m; j++)
                        for (size_t c = 0; c < nc; c++)
                            A[j][c] = std::pow(x[j], double(deg) - double(c));
                    std::vector<double> scale(nc, 0.0), rhs = y;
                    for (size_t c = 0; c < nc; c++) {
                        for (size_t j = 0; j < m; j++)
                            scale[c] += A[j][c] * A[j][c];
                        scale[c] = std::sqrt(scale[c]);
                        for (size_t j = 0; j < m; j++)
                            A[j][c] /= scale[c];
                    }
                    for (size_t c = 0; c < nc; c++) { // Householder
                        double norm = 0.0;
                        for (size_t j = c; j < m; j++)
                            norm += A[j][c] * A[j][c];
                        norm = std::sqrt(norm);
                        const double alpha = A[c][c] > 0 ? -norm : norm;
                        std::vector<double> v(m, 0.0);
                        for (size_t j = c; j < m; j++)
                            v[j] = A[j][c];
                        v[c] -= alpha;
                        double vv = 0.0;
                        for (size_t j = c; j < m; j++)
                            vv += v[j] * v[j];
                        if (vv == 0.0)
                            continue;
                        for (size_t cc = c; cc < nc; cc++) {
                            double d = 0.0;
                            for (size_t j = c; j < m; j++)
                                d += v[j] * A[j][cc];
                            for (size_t j = c; j < m; j++)
                                A[j][cc] -= 2.0 * d / vv * v[j];
                        }
                        double d = 0.0;
                        for (size_t j = c; j < m; j++)
                            d += v[j] * rhs[j];
                        for (size_t j = c; j < m; j++)
                            rhs[j] -= 2.0 * d / vv * v[j];
                    }
                    std::vector<double> coef(nc);
                    for (size_t c = nc; c-- > 0;) {
                        double v = rhs[c];
                        for (size_t cc = c + 1; cc < nc; cc++)
                            v -= A[c][cc] * coef[cc];
                        coef[c] = v / A[c][c];
                    }
                    for (size_t c = 0; c < nc; c++)
                        coef[c] /= scale[c];
                    return coef;
                }

                /// baccoemu.utils.pkmulti (mode 'reimann_sum', pmulti_interp 'polyfit'): the multipoles 0, 2, 4 of
                /// P(k, mu) given on mu (pk2d[imu][ik]), from a 4th order polynomial fit in mu integrated with the
                /// 8-point Gauss-Legendre rule of baccoemu
                inline std::array<std::vector<double>, 3> pkmulti(const std::vector<double> & mu,
                                                                  const std::vector<std::vector<double>> & pk2d,
                                                                  size_t nk) {
                    static const double roots[8] = {0.09501251, 0.28160355, 0.45801678, 0.61787624,
                                                    0.75540441, 0.8656312,  0.94457502, 0.98940093};
                    static const double weights[8] = {0.189450610455069, 0.182603415044924, 0.169156519395003,
                                                      0.149595988816577, 0.124628971255534, 0.095158511682493,
                                                      0.062253523938648, 0.027152459411754};
                    std::array<std::vector<double>, 3> mom, multi;
                    for (auto & m : mom)
                        m.assign(nk, 0.0);
                    for (size_t ik = 0; ik < nk; ik++) {
                        std::vector<double> xm, ym;
                        for (size_t j = 0; j < mu.size(); j++)
                            if (pk2d[j][ik] != 0.0) {
                                xm.push_back(mu[j]);
                                ym.push_back(pk2d[j][ik]);
                            }
                        if (xm.size() <= 1)
                            continue;
                        const int deg = int(std::min<size_t>(4, xm.size()));
                        const auto cc = polyfit(xm, ym, deg);
                        auto p_at = [&](double m) {
                            double r = 0.0;
                            for (size_t i = 0; i < cc.size(); i++)
                                r += cc[i] * std::pow(m, double(deg) - double(i));
                            return r;
                        };
                        for (int ell = 0; ell < 3; ell++) {
                            double r = 0.0;
                            for (int io = 0; io < 8; io++)
                                r += weights[io] * std::pow(roots[io], 2 * ell) * p_at(roots[io]);
                            mom[size_t(ell)][ik] = r;
                        }
                    }
                    for (auto & m : multi)
                        m.assign(nk, 0.0);
                    for (size_t ik = 0; ik < nk; ik++) {
                        multi[0][ik] = mom[0][ik];
                        multi[1][ik] = (5.0 / 2.0) * (3.0 * mom[1][ik] - mom[0][ik]);
                        multi[2][ik] = (9.0 / 8.0) * (35 * mom[2][ik] - 30.0 * mom[1][ik] + 3.0 * mom[0][ik]);
                    }
                    return multi;
                }

                /// Read a ReLU network written by export_bacco.py (FML_BACCO_NN 1: a text header, then the weights
                /// and biases as little-endian float32, layer by layer, in the order of NeuralNetwork::parameters())
                inline FML::EMULATOR::NEURALNETWORK::NeuralNetwork load_network(const std::string & filename) {
                    using namespace FML::EMULATOR::NEURALNETWORK;
                    std::ifstream in(filename, std::ios::binary);
                    if (not in)
                        throw MissingDataError(filename);
                    std::stringstream tok;
                    std::string line;
                    size_t nfloat = 0;
                    while (std::getline(in, line)) {
                        line = line.substr(0, line.find('#'));
                        if (line.rfind("float32_le", 0) == 0) {
                            nfloat = size_t(std::stoull(line.substr(10)));
                            break;
                        }
                        tok << line << "\n";
                    }
                    auto expect = [&](const std::string & w) {
                        std::string x;
                        if (not(tok >> x) or x != w)
                            throw std::runtime_error("baccoemu: expected '" + w + "' in " + filename);
                    };
                    auto read_vec = [&](const std::string & w, size_t n) {
                        expect(w);
                        std::vector<double> v(n);
                        for (auto & x : v)
                            tok >> x;
                        return v;
                    };
                    int version, n;
                    std::string act;
                    expect("FML_BACCO_NN");
                    tok >> version;
                    expect("activation");
                    tok >> act;
                    expect("sizes");
                    tok >> n;
                    std::vector<int> sizes(size_t(std::max(n, 0)));
                    for (auto & v : sizes)
                        tok >> v;
                    if (not tok or sizes.size() < 2)
                        throw std::runtime_error("baccoemu: bad header in " + filename);
                    NeuralNetwork net(sizes, activation_from_name(act));
                    const auto im = read_vec("input_mean", size_t(sizes.front()));
                    const auto is = read_vec("input_std", size_t(sizes.front()));
                    const auto om = read_vec("output_mean", size_t(sizes.back()));
                    const auto os = read_vec("output_std", size_t(sizes.back()));
                    if (not tok or nfloat != net.num_parameters())
                        throw std::runtime_error("baccoemu: bad header in " + filename);
                    net.set_input_scaling(im, is);
                    net.set_output_scaling(om, os);
                    std::vector<unsigned char> raw(4 * nfloat);
                    in.read(reinterpret_cast<char *>(raw.data()), std::streamsize(raw.size()));
                    if (size_t(in.gcount()) != raw.size())
                        throw std::runtime_error("baccoemu: truncated file " + filename);
                    auto & w = net.parameters();
                    for (size_t i = 0; i < nfloat; i++) {
                        const uint32_t u = uint32_t(raw[4 * i]) | (uint32_t(raw[4 * i + 1]) << 8) |
                                           (uint32_t(raw[4 * i + 2]) << 16) | (uint32_t(raw[4 * i + 3]) << 24);
                        float f;
                        std::memcpy(&f, &u, 4);
                        w[i] = double(f);
                    }
                    return net;
                }

                /// A ReLU network with its details
                struct KerasEmulator {
                    std::string name;
                    FML::EMULATOR::NEURALNETWORK::NeuralNetwork net;
                    Details details;
                    void load(const std::string & n) {
                        name = n;
                        details.load(data_dir + n + "_details.txt");
                        net = load_network(data_dir + n + ".bin");
                        if (size_t(net.input_dim()) != details.keys.size())
                            throw std::runtime_error("baccoemu: the network and details of " + n + " do not match");
                    }
                    bool loaded() const { return not name.empty(); }
                };

                inline void check_k_increasing(const std::vector<double> & k, const std::string & what) {
                    for (size_t i = 0; i < k.size(); i++)
                        if (not(k[i] > 0.0) or (i > 0 and not(k[i] > k[i - 1])))
                            throw std::invalid_argument("baccoemu " + what + ": k must be positive and increasing");
                }
            } // namespace detail

            //=========================================================================================================
            /// The matter power spectrum emulators (baccoemu.Matter_powerspectrum). The options choose the networks
            /// to load, as the arguments of the Python class (same defaults).
            ///
            ///   MatterPowerSpectrum emu;                         // or emu(options)
            ///   Parameters p; p.Omega_cold = 0.31; p.Omega_b = 0.049; p.sigma8_cold = 0.81; p.h = 0.67; p.n_s = 0.965;
            ///   auto pk = emu.nonlinear_pk(p, a, k);             // P_nl(k) in (Mpc/h)^3 for k in h/Mpc
            ///   auto S  = emu.baryonic_boost(p, a, k);           // with the baryon parameters set in p
            //=========================================================================================================
            class MatterPowerSpectrum {
              public:
                struct Options {
                    bool linear{true};                          ///< the linear emulators (cold and total matter)
                    bool sigma8{true};                          ///< the sigma8 emulator (sigma8 <-> A_s)
                    bool nonlinear_boost{true};                 ///< the nonlinear boost
                    std::string nonlinear_model{"Angulo2021"};  ///< "Angulo2021" or "Arico2023" (wider ranges)
                    bool smeared_bao{false};                    ///< the BAO-smeared P(k) from its own emulator, or
                    bool no_wiggles{true};                      ///< ... from the no-wiggle emulator (one of the two)
                    bool baryonic_boost{true};                  ///< the baryonic boost
                    std::string baryonic_model{"Arico2021"};    ///< "Arico2021" or "Burger2025"
                };

                MatterPowerSpectrum() : MatterPowerSpectrum(Options{}) {}
                explicit MatterPowerSpectrum(const Options & opt) : options(opt) {
                    if ((opt.smeared_bao and opt.no_wiggles) or (opt.nonlinear_boost and not opt.smeared_bao and
                                                                  not opt.no_wiggles))
                        throw std::invalid_argument("baccoemu: set exactly one of smeared_bao and no_wiggles (the "
                                                    "BAO smearing from the linear P(k) by a DST is not ported)");
                    if (opt.sigma8)
                        sigma8_emu.load("sigma8_emu_2.0.2");
                    if (opt.linear) {
                        linear_cold.load("cold_matter_linear_emu_1.0.2");
                        linear_tot.load("total_matter_linear_emu_1.0.2");
                    }
                    if (opt.smeared_bao)
                        smeared_bao_emu.load("smeared_bao_emu_1.0.2");
                    if (opt.no_wiggles)
                        no_wiggles_emu.load("no_wiggles_emu_1.0.2");
                    if (opt.nonlinear_boost) {
                        if (opt.nonlinear_model == "Angulo2021")
                            nonlinear_emu.load("nonlinear_emu_1.0.3");
                        else if (opt.nonlinear_model == "Arico2023")
                            nonlinear_emu.load("nonlinear_emu_2.0.1");
                        else
                            throw std::invalid_argument("baccoemu: nonlinear_model must be Angulo2021 or Arico2023");
                    }
                    if (opt.baryonic_boost) {
                        if (opt.baryonic_model == "Arico2021")
                            baryon_emu.load("baryon_emu_1.0.2");
                        else if (opt.baryonic_model == "Burger2025") {
                            burger = std::make_shared<CosmoPowerNetwork>();
                            burger->load(data_dir + "baryon_boost_Burger2025_pk.txt");
                        } else
                            throw std::invalid_argument("baccoemu: baryonic_model must be Arico2021 or Burger2025");
                    }
                }

                const Options & get_options() const { return options; }

                /// The k bins (h/Mpc) of the emulators: used when k is empty
                const std::vector<double> & kbins_linear() const { return need(linear_cold, "linear").details.k; }
                const std::vector<double> & kbins_nonlinear() const {
                    return need(nonlinear_emu, "nonlinear").details.k;
                }
                const std::vector<double> & kbins_no_wiggles() const {
                    return need(no_wiggles_emu, "no_wiggles").details.k;
                }
                const std::vector<double> & kbins_smeared_bao() const {
                    return need(smeared_bao_emu, "smeared_bao").details.k;
                }
                /// The k bins of the Arico2021 baryonic boost, or the default 100 log-spaced k of Burger2025
                std::vector<double> kbins_baryon() const {
                    if (burger) {
                        const auto r = burger->parameter_ranges().at("k");
                        std::vector<double> k(100);
                        const double l0 = std::log10(r[0]), l1 = std::log10(r[1]);
                        const double step = (l1 - l0) / double(k.size() - 1);
                        for (size_t i = 0; i < k.size(); i++) // np.geomspace
                            k[i] = std::pow(10.0, double(i) * step + l0);
                        k.front() = r[0];
                        k.back() = r[1];
                        return k;
                    }
                    return need(baryon_emu, "baryon").details.k;
                }

                /// The parameter names and ranges of an emulator: "linear", "sigma8", "nonlinear", "no_wiggles",
                /// "smeared_bao", "baryon" (baccoemu names)
                std::vector<std::pair<std::string, std::array<double, 2>>> ranges(const std::string & which) const {
                    std::vector<std::pair<std::string, std::array<double, 2>>> r;
                    if (which == "baryon" and burger) {
                        for (auto & p : burger->parameter_ranges())
                            r.push_back({burger_name(p.first), p.second});
                        return r;
                    }
                    const auto & e = need(emulator(which), which);
                    for (size_t i = 0; i < e.details.keys.size(); i++)
                        r.push_back({e.details.keys[i], e.details.bounds[i]});
                    return r;
                }

                //-----------------------------------------------------------------------------------------------------
                // sigma8, sigma12, A_s (get_sigma8, get_sigma12, get_A_s)
                //-----------------------------------------------------------------------------------------------------

                /// sigma8 of the cold (cdm + baryons) or total matter at z = 0
                double sigma8(const Parameters & p, bool cold = true) const { return sigmaR(p, cold ? 0 : 1); }
                /// sigma12 (R = 12 Mpc) of the cold or total matter at z = 0
                double sigma12(const Parameters & p, bool cold = true) const { return sigmaR(p, cold ? 2 : 3); }
                /// A_s (for sigma8_cold, or as given)
                double A_s(const Parameters & p) const {
                    auto q = resolve(p, false, true);
                    check(linear_cold, q, 1.0);
                    return q.A_s;
                }

                //-----------------------------------------------------------------------------------------------------
                // Linear P(k) (get_linear_pk). k empty: kbins_linear(). k must be inside [kmin, kmax] of the bins.
                //-----------------------------------------------------------------------------------------------------
                std::vector<double> linear_pk(const Parameters & p,
                                              double a,
                                              const std::vector<double> & k = {},
                                              bool cold = true) const {
                    need(cold ? linear_cold : linear_tot, "linear");
                    auto q = resolve(p, false, false);
                    check(cold ? linear_cold : linear_tot, q, a);
                    return linear_pk_resolved(resolve(p, false, true), a, k, cold);
                }

                //-----------------------------------------------------------------------------------------------------
                // The no-wiggle P(k) of the cold matter (get_no_wiggles_pk). k empty: kbins_no_wiggles(). The linear
                // P(k) is the emulator's unless k_lin, pk_lin are given (then interpolated log-log to k).
                //-----------------------------------------------------------------------------------------------------
                std::vector<double> no_wiggles_pk(const Parameters & p,
                                                  double a,
                                                  const std::vector<double> & k = {},
                                                  const std::vector<double> & k_lin = {},
                                                  const std::vector<double> & pk_lin = {}) const {
                    return no_wiggles_resolved(p, resolve(p, false, false), a, k, k_lin, pk_lin);
                }

                //-----------------------------------------------------------------------------------------------------
                // The BAO-smeared P(k) of the cold matter from its emulator (get_smeared_bao_pk; options.smeared_bao).
                // k empty: kbins_smeared_bao(). k must be inside the bins.
                //-----------------------------------------------------------------------------------------------------
                std::vector<double> smeared_bao_pk(const Parameters & p,
                                                   double a,
                                                   const std::vector<double> & k = {}) const {
                    auto q = resolve(p, true, false);
                    return smeared_bao_resolved(q, a, k);
                }

                //-----------------------------------------------------------------------------------------------------
                // The nonlinear boost P_nl / P_lin (get_nonlinear_boost) of the cold (cdm + baryons) or total matter.
                // k empty: kbins_nonlinear(); otherwise k <= kmax of the bins, and the boost is 1 below the bins.
                // The linear P(k) (cold or total as `cold`) is the emulator's unless k_lin, pk_lin are given (they
                // must cover [1e-3, 10] h/Mpc; for the total matter the cold linear P(k) is still emulated).
                //-----------------------------------------------------------------------------------------------------
                std::vector<double> nonlinear_boost(const Parameters & p,
                                                    double a,
                                                    const std::vector<double> & k = {},
                                                    bool cold = true,
                                                    const std::vector<double> & k_lin = {},
                                                    const std::vector<double> & pk_lin = {}) const {
                    return evaluate_nonlinear(p, a, k, cold, k_lin, pk_lin).first;
                }

                /// The nonlinear P(k) (get_nonlinear_pk), times the baryonic boost if baryons (at the same k)
                std::vector<double> nonlinear_pk(const Parameters & p,
                                                 double a,
                                                 const std::vector<double> & k = {},
                                                 bool cold = true,
                                                 bool baryons = false,
                                                 const std::vector<double> & k_lin = {},
                                                 const std::vector<double> & pk_lin = {}) const {
                    auto r = evaluate_nonlinear(p, a, k, cold, k_lin, pk_lin).second;
                    if (baryons) {
                        const auto S = baryonic_boost(p, a, k.empty() ? kbins_nonlinear() : k);
                        for (size_t i = 0; i < r.size(); i++)
                            r[i] *= S[i];
                    }
                    return r;
                }

                //-----------------------------------------------------------------------------------------------------
                // The baryonic boost S(k) = P_baryons / P_gravity-only (get_baryonic_boost). k empty: kbins_baryon().
                // Arico2021: linear in ln k, extrapolated above the bins (not accurate) and 1 below. Burger2025:
                // evaluated at every k (trained on [0.018, 17.7] h/Mpc). Parameters: Arico2021 all of them;
                // Burger2025 Omega_cold, Omega_b, sigma8_cold, M_c, eta, beta, M1_z0_cen, theta_inn.
                //-----------------------------------------------------------------------------------------------------
                std::vector<double> baryonic_boost(const Parameters & p,
                                                   double a,
                                                   const std::vector<double> & k = {}) const {
                    if (burger)
                        return burger_boost(p, a, k.empty() ? kbins_baryon() : k);
                    const auto & e = need(baryon_emu, "baryon");
                    auto q = resolve(p, true, false);
                    check(e, q, a);
                    auto y = e.net(input(e, q, a));
                    for (auto & v : y)
                        v = std::exp(v);
                    if (k.empty())
                        return y;
                    const auto lk = detail::log_of(e.details.k);
                    std::vector<double> r(k.size());
                    for (size_t i = 0; i < k.size(); i++)
                        r[i] = k[i] < e.details.k.front() ? 1.0 : detail::interp_extrap(lk, y, std::log(k[i]));
                    return r;
                }

                //-----------------------------------------------------------------------------------------------------
                // Internals, public for MatterBispectrum and the tests
                //-----------------------------------------------------------------------------------------------------

                /// The parameters with Omega_cold, sigma8_cold and A_s filled in (as baccoemu's _get_parameters)
                struct Resolved {
                    Parameters p;
                    double Omega_cold, sigma8_cold, A_s;
                };

                Resolved resolve(const Parameters & p, bool need_sigma8, bool need_A_s) const {
                    Resolved q{p, p.Omega_cold, p.sigma8_cold, p.A_s};
                    const double Onu = p.M_nu / 93.14 / (p.h * p.h);
                    if (std::isnan(p.Omega_cold) and std::isnan(p.Omega_m))
                        throw std::invalid_argument("baccoemu: give Omega_cold or Omega_m");
                    if (not std::isnan(p.Omega_cold) and not std::isnan(p.Omega_m) and
                        not(std::abs(p.Omega_m - (p.Omega_cold + Onu)) < 1e-4))
                        throw std::invalid_argument("baccoemu: Omega_cold and Omega_m are inconsistent");
                    if (std::isnan(q.Omega_cold))
                        q.Omega_cold = p.Omega_m - Onu;
                    q.p.Omega_cold = q.Omega_cold;
                    if (std::isnan(p.sigma8_cold) and std::isnan(p.A_s))
                        throw std::invalid_argument("baccoemu: give sigma8_cold or A_s");
                    if (not std::isnan(p.sigma8_cold) and not std::isnan(p.A_s)) {
                        const double s8 = sigma_raw(q.p, p.A_s, 0);
                        if (not(std::abs(p.sigma8_cold - s8) < 1e-4))
                            throw std::invalid_argument("baccoemu: sigma8_cold and A_s are inconsistent");
                    }
                    if (need_sigma8 and std::isnan(q.sigma8_cold))
                        q.sigma8_cold = sigma_raw(q.p, p.A_s, 0);
                    if (need_A_s and std::isnan(q.A_s)) {
                        const double ref = 2e-9;
                        q.A_s = std::pow(p.sigma8_cold / sigma_raw(q.p, ref, 0), 2) * ref;
                    }
                    q.p.sigma8_cold = q.sigma8_cold;
                    q.p.A_s = q.A_s;
                    return q;
                }

              private:
                Options options;
                detail::KerasEmulator sigma8_emu, linear_cold, linear_tot, smeared_bao_emu, no_wiggles_emu,
                    nonlinear_emu, baryon_emu;
                std::shared_ptr<CosmoPowerNetwork> burger;

                const detail::KerasEmulator & emulator(const std::string & which) const {
                    if (which == "linear")
                        return linear_cold;
                    if (which == "sigma8")
                        return sigma8_emu;
                    if (which == "nonlinear")
                        return nonlinear_emu;
                    if (which == "no_wiggles")
                        return no_wiggles_emu;
                    if (which == "smeared_bao")
                        return smeared_bao_emu;
                    if (which == "baryon")
                        return baryon_emu;
                    throw std::invalid_argument("baccoemu: unknown emulator " + which);
                }

              public:
                // Helpers, also used by the bias expansion emulators
                static const detail::KerasEmulator & need(const detail::KerasEmulator & e, const std::string & what) {
                    if (not e.loaded())
                        throw std::invalid_argument("baccoemu: the " + what + " emulator is not loaded (Options)");
                    return e;
                }

                static double value(const Parameters & p, const std::string & key, double a) {
                    if (key == "omega_cold")
                        return p.Omega_cold;
                    if (key == "sigma8_cold")
                        return p.sigma8_cold;
                    if (key == "omega_baryon")
                        return p.Omega_b;
                    if (key == "ns")
                        return p.n_s;
                    if (key == "hubble")
                        return p.h;
                    if (key == "neutrino_mass")
                        return p.M_nu;
                    if (key == "w0")
                        return p.w0;
                    if (key == "wa")
                        return p.wa;
                    if (key == "expfactor")
                        return a;
                    if (key == "M_c")
                        return p.M_c;
                    if (key == "eta")
                        return p.eta;
                    if (key == "beta")
                        return p.beta;
                    if (key == "M1_z0_cen")
                        return p.M1_z0_cen;
                    if (key == "theta_out")
                        return p.theta_out;
                    if (key == "theta_inn")
                        return p.theta_inn;
                    if (key == "M_inn")
                        return p.M_inn;
                    throw std::invalid_argument("baccoemu: unknown parameter " + key);
                }

                static std::string burger_name(const std::string & key) {
                    if (key == "Omega_m")
                        return "omega_cold";
                    if (key == "Omega_b")
                        return "omega_baryon";
                    if (key == "sigma8")
                        return "sigma8_cold";
                    return key;
                }

                /// The bounds check of baccoemu (all keys, in order)
                static void check_values(const detail::KerasEmulator & e, const Parameters & p, double a) {
                    for (size_t i = 0; i < e.details.keys.size(); i++) {
                        const double v = value(p, e.details.keys[i], a);
                        if (std::isnan(v))
                            throw std::invalid_argument("baccoemu " + e.name + ": the parameter " +
                                                        e.details.keys[i] + " is needed");
                        if (not(v >= e.details.bounds[i][0] and v <= e.details.bounds[i][1]))
                            throw EmulationRangeError(e.name, e.details.keys[i], v, e.details.bounds[i][0],
                                                      e.details.bounds[i][1]);
                    }
                }
                static void check(const detail::KerasEmulator & e, const Resolved & q, double a) {
                    check_values(e, q.p, a);
                }

                static std::vector<double> input(const detail::KerasEmulator & e, const Resolved & q, double a) {
                    std::vector<double> x(e.details.keys.size());
                    for (size_t i = 0; i < x.size(); i++)
                        x[i] = value(q.p, e.details.keys[i], a);
                    return x;
                }

              private:

                /// The 4 outputs of the sigma8 emulator (sigma8 cold, total, sigma12 cold, total) for A_s
                double sigma_raw(const Parameters & p, double As, int index) const {
                    const auto & e = need(sigma8_emu, "sigma8");
                    check_values(e, p, 1.0);
                    std::vector<double> x(e.details.keys.size());
                    for (size_t i = 0; i < x.size(); i++)
                        x[i] = value(p, e.details.keys[i], 1.0);
                    return e.net(x)[size_t(index)] * std::sqrt(As / 1.e-9);
                }

                double sigmaR(const Parameters & p, int index) const {
                    auto q = resolve(p, false, true);
                    return sigma_raw(q.p, q.A_s, index);
                }

                std::vector<double> linear_pk_resolved(const Resolved & q,
                                                       double a,
                                                       const std::vector<double> & k,
                                                       bool cold) const {
                    const auto & e = need(cold ? linear_cold : linear_tot, "linear");
                    if (std::isnan(q.p.n_s))
                        throw std::invalid_argument("baccoemu linear: the parameter ns is needed");
                    check(e, q, a);
                    auto y = e.net(input(e, q, a));
                    const auto & ke = e.details.k;
                    const double As_fixed = 1.e-9, ns_fixed = 1.0, k_pivot = 0.05;
                    for (size_t i = 0; i < y.size(); i++)
                        y[i] = std::exp(y[i]) *
                               (q.A_s / As_fixed * std::pow(ke[i] / k_pivot * q.p.h, q.p.n_s - ns_fixed));
                    if (k.empty())
                        return y;
                    detail::check_k_increasing(k, "linear_pk");
                    if (k.back() > ke.back() or k.front() < ke.front())
                        throw std::invalid_argument("baccoemu linear_pk: k must be inside [" +
                                                    detail::to_str(ke.front()) + ", " + detail::to_str(ke.back()) +
                                                    "] h/Mpc");
                    bool negative = false;
                    for (double v : y)
                        negative = negative or v < 0;
                    if (not negative)
                        return detail::interp_loglog(ke, y, k);
                    const auto lk = detail::log_of(ke);
                    std::vector<double> r(k.size());
                    for (size_t i = 0; i < k.size(); i++)
                        r[i] = detail::interp_extrap(lk, y, std::log(k[i]));
                    return r;
                }

                std::vector<double> smeared_bao_resolved(const Resolved & q,
                                                         double a,
                                                         const std::vector<double> & k) const {
                    const auto & e = need(smeared_bao_emu, "smeared_bao");
                    check(e, q, a);
                    auto y = e.net(input(e, q, a));
                    for (auto & v : y)
                        v = std::exp(v);
                    if (k.empty())
                        return y;
                    const auto & ke = e.details.k;
                    detail::check_k_increasing(k, "smeared_bao_pk");
                    if (k.back() > ke.back() or k.front() < ke.front())
                        throw std::invalid_argument("baccoemu smeared_bao_pk: k must be inside [" +
                                                    detail::to_str(ke.front()) + ", " + detail::to_str(ke.back()) +
                                                    "] h/Mpc");
                    return detail::interp_loglog(ke, y, k);
                }

                std::vector<double> no_wiggles_resolved(const Parameters & p,
                                                        const Resolved & q,
                                                        double a,
                                                        const std::vector<double> & k,
                                                        const std::vector<double> & k_lin,
                                                        const std::vector<double> & pk_lin) const {
                    const auto & e = need(no_wiggles_emu, "no_wiggles");
                    check(e, q, a);
                    std::vector<double> y = e.net(input(e, q, a));
                    const auto & ke = e.details.k;
                    for (size_t i = 0; i < y.size(); i++)
                        if (ke[i] > 0.8)
                            y[i] = 0.0;
                    std::vector<double> kk = k.empty() ? ke : k, ratio(kk.size());
                    if (k.empty()) {
                        for (size_t i = 0; i < y.size(); i++)
                            ratio[i] = std::exp(y[i]);
                    } else {
                        detail::check_k_increasing(k, "no_wiggles_pk");
                        const auto lk = detail::log_of(ke);
                        for (size_t i = 0; i < kk.size(); i++)
                            ratio[i] = std::exp(detail::interp_fill(lk, y, std::log(kk[i]), 0.0));
                    }
                    std::vector<double> plin;
                    if (not k_lin.empty() and not pk_lin.empty()) {
                        if (k_lin.size() != pk_lin.size())
                            throw std::invalid_argument("baccoemu no_wiggles_pk: k_lin and pk_lin differ in size");
                        plin = (k_lin == kk) ? pk_lin : detail::interp_loglog(k_lin, pk_lin, kk);
                    } else {
                        plin = linear_pk_resolved(std::isnan(q.A_s) ? resolve(p, false, true) : q, a, kk, true);
                    }
                    for (size_t i = 0; i < kk.size(); i++)
                        ratio[i] *= plin[i];
                    return ratio;
                }

                /// _evaluate_nonlinear: (boost, P_nl)
                std::pair<std::vector<double>, std::vector<double>>
                evaluate_nonlinear(const Parameters & p,
                                   double a,
                                   const std::vector<double> & k,
                                   bool cold,
                                   std::vector<double> k_lin,
                                   std::vector<double> pk_lin) const {
                    const auto & e = need(nonlinear_emu, "nonlinear");
                    auto q = resolve(p, true, false);
                    check(e, q, a);
                    if (pk_lin.empty() or not cold)
                        q = resolve(p, true, true);
                    auto Q = e.net(input(e, q, a));
                    for (auto & v : Q)
                        v = std::exp(v);
                    const auto & knl = e.details.k;

                    if (pk_lin.empty()) {
                        k_lin = need(linear_cold, "linear").details.k;
                        pk_lin = linear_pk_resolved(q, a, {}, cold);
                    } else {
                        if (k_lin.size() != pk_lin.size())
                            throw std::invalid_argument("baccoemu nonlinear: k_lin and pk_lin differ in size");
                        detail::check_k_increasing(k_lin, "nonlinear (k_lin)");
                        if (k_lin.front() > 1e-3 or k_lin.back() < 10)
                            throw std::invalid_argument("baccoemu nonlinear: the linear P(k) must cover k in "
                                                        "[1e-3, 10] h/Mpc");
                    }
                    std::vector<double> k_lin_cold = k_lin, pk_lin_cold = pk_lin;
                    if (not cold)
                        pk_lin_cold = linear_pk_resolved(q, a, k_lin, true);

                    const auto pk_lin_emu_cold = detail::interp_loglog(k_lin_cold, pk_lin_cold, knl);
                    std::vector<double> pk_lin_emu_tot;
                    if (not cold)
                        pk_lin_emu_tot = detail::interp_loglog(k_lin, pk_lin, knl);

                    std::vector<double> pk_smeared;
                    if (options.smeared_bao) {
                        pk_smeared = smeared_bao_resolved(q, a, knl);
                    } else {
                        const auto pk_nw = no_wiggles_resolved(p, q, a, knl, k_lin_cold, pk_lin_cold);
                        std::vector<double> integrand(k_lin_cold.size());
                        for (size_t i = 0; i < integrand.size(); i++)
                            integrand[i] = k_lin_cold[i] * pk_lin_cold[i];
                        const double sigma_star_2 =
                            detail::trapezoid(integrand, detail::log_of(k_lin_cold)) / (3 * M_PI * M_PI);
                        const double k_star_2 = 1 / sigma_star_2;
                        pk_smeared.resize(knl.size());
                        for (size_t i = 0; i < knl.size(); i++) {
                            const double G = std::exp(-0.5 * (knl[i] * knl[i] / k_star_2));
                            pk_smeared[i] = pk_lin_emu_cold[i] * G + pk_nw[i] * (1 - G);
                        }
                    }

                    std::vector<double> boost(knl.size());
                    if (cold) {
                        for (size_t i = 0; i < knl.size(); i++)
                            boost[i] = Q[i] * pk_smeared[i] / pk_lin_emu_cold[i];
                    } else {
                        const double omega_nu = p.M_nu / 93.14 / (p.h * p.h);
                        const double omega_matter = not std::isnan(p.Omega_cold) ? p.Omega_cold + omega_nu : p.Omega_m;
                        const double f_nu = omega_nu / omega_matter;
                        for (size_t i = 0; i < knl.size(); i++) {
                            const double add = pk_lin_emu_tot[i] - (1 - f_nu) * (1 - f_nu) * pk_lin_emu_cold[i];
                            boost[i] = (Q[i] * pk_smeared[i] * ((1 - f_nu) * (1 - f_nu)) + add) / pk_lin_emu_tot[i];
                        }
                    }

                    if (k.empty()) {
                        const auto & plin = cold ? pk_lin_emu_cold : pk_lin_emu_tot;
                        std::vector<double> pk(knl.size());
                        for (size_t i = 0; i < knl.size(); i++)
                            pk[i] = boost[i] * plin[i];
                        return {boost, pk};
                    }
                    detail::check_k_increasing(k, "nonlinear");
                    if (k.back() > knl.back())
                        throw std::invalid_argument("baccoemu nonlinear: the maximum k of the emulator is " +
                                                    detail::to_str(knl.back()) + " h/Mpc");
                    const auto lk = detail::log_of(knl);
                    std::vector<double> b(k.size());
                    for (size_t i = 0; i < k.size(); i++)
                        b[i] = k[i] < knl.front() ? 1.0 : detail::interp_extrap(lk, boost, std::log(k[i]));
                    auto pk = detail::interp_loglog(k_lin, pk_lin, k);
                    for (size_t i = 0; i < k.size(); i++)
                        pk[i] *= b[i];
                    return {b, pk};
                }

                std::vector<double> burger_boost(const Parameters & p, double a, const std::vector<double> & k) const {
                    const auto & net = *burger;
                    auto q = resolve(p, true, false);
                    std::vector<double> x(net.parameters().size());
                    size_t ik = x.size();
                    for (size_t j = 0; j < x.size(); j++) {
                        const auto & name = net.parameters()[j];
                        if (name == "k") {
                            ik = j;
                            continue;
                        }
                        const std::string key = burger_name(name);
                        x[j] = value(q.p, key, a);
                        const auto r = net.parameter_ranges().at(name);
                        if (std::isnan(x[j]))
                            throw std::invalid_argument("baccoemu Burger2025: the parameter " + key + " is needed");
                        if (not(x[j] >= r[0] and x[j] <= r[1]))
                            throw EmulationRangeError("Burger2025", key, x[j], r[0], r[1]);
                    }
                    std::vector<double> S(k.size());
                    for (size_t i = 0; i < k.size(); i++) {
                        x[ik] = k[i];
                        S[i] = net(x)[0];
                    }
                    return S;
                }

                friend class MatterBispectrum;
            };

            //=========================================================================================================
            /// The baryonic boost of the matter bispectrum B_baryons / B_gravity-only (baccoemu.Matter_bispectrum,
            /// Burger et al. 2025). Parameters: Omega_cold (or Omega_m with h and M_nu), Omega_b, sigma8_cold, M_c,
            /// eta, beta, M1_z0_cen, theta_inn. Trained on k in [0.018, 17.7] h/Mpc.
            //=========================================================================================================
            class MatterBispectrum {
              public:
                struct Result {
                    std::vector<std::array<double, 3>> k; ///< the triangles, sorted k1 <= k2 <= k3
                    std::vector<double> boost;
                    std::vector<bool> extrapolated; ///< a k outside the training range
                };

                MatterBispectrum() {
                    net = std::make_shared<CosmoPowerNetwork>();
                    net->load(data_dir + "baryon_boost_Burger2025_bk.txt");
                }

                /// The boost for the triangles (k1[i], k2[i], k3[i]); as in baccoemu each is sorted and those that are
                /// not triangles (k1 + k2 < k3) are dropped
                Result baryonic_boost(const Parameters & p,
                                      double a,
                                      const std::vector<double> & k1,
                                      const std::vector<double> & k2,
                                      const std::vector<double> & k3) const {
                    if (k1.size() != k2.size() or k1.size() != k3.size())
                        throw std::invalid_argument("baccoemu MatterBispectrum: k1, k2, k3 differ in size");
                    if (std::isnan(p.sigma8_cold))
                        throw std::invalid_argument("baccoemu MatterBispectrum: give sigma8_cold");
                    Parameters q = p;
                    if (std::isnan(q.Omega_cold)) {
                        if (std::isnan(q.Omega_m))
                            throw std::invalid_argument("baccoemu MatterBispectrum: give Omega_cold or Omega_m");
                        q.Omega_cold = q.Omega_m - q.M_nu / 93.14 / (q.h * q.h);
                    }
                    std::vector<double> x(net->parameters().size());
                    std::array<size_t, 3> ik{};
                    for (size_t j = 0; j < x.size(); j++) {
                        const auto & name = net->parameters()[j];
                        if (name == "k1" or name == "k2" or name == "k3") {
                            ik[size_t(name[1] - '1')] = j;
                            continue;
                        }
                        const std::string key = MatterPowerSpectrum::burger_name(name);
                        x[j] = MatterPowerSpectrum::value(q, key, a);
                        const auto r = net->parameter_ranges().at(name);
                        if (std::isnan(x[j]))
                            throw std::invalid_argument("baccoemu MatterBispectrum: the parameter " + key +
                                                        " is needed");
                        if (not(x[j] >= r[0] and x[j] <= r[1]))
                            throw EmulationRangeError("Burger2025 bispectrum", key, x[j], r[0], r[1]);
                    }
                    const auto kr = net->parameter_ranges().at("k");
                    Result res;
                    for (size_t i = 0; i < k1.size(); i++) {
                        std::array<double, 3> t{k1[i], k2[i], k3[i]};
                        std::sort(t.begin(), t.end());
                        if (not(t[0] + t[1] >= t[2]))
                            continue;
                        for (int d = 0; d < 3; d++)
                            x[ik[size_t(d)]] = t[size_t(d)];
                        res.k.push_back(t);
                        res.boost.push_back((*net)(x)[0]);
                        bool ext = false;
                        for (double v : t)
                            ext = ext or v < kr[0] or v > kr[1];
                        res.extrapolated.push_back(ext);
                    }
                    return res;
                }

              private:
                std::shared_ptr<CosmoPowerNetwork> net;
            };

            //=========================================================================================================
            /// The coevolution relations of the Lagrangian bias parameters of Zennaro et al. 2022 (arXiv:2110.05408),
            /// as baccoemu.utils.coevolution_relations: haloes (b1L from the peak height nu) and galaxies
            //=========================================================================================================
            namespace COEVOLUTION {
                inline double halo_b1L_nu(double nu) { return -0.00951 * nu * nu * nu + 0.4873 * nu * nu - 0.1395 * nu - 0.4383; }
                inline double halo_b2L_b1L(double b) { return -0.09143 * b * b * b + 0.7093 * b * b - 0.2607 * b - 0.3469; }
                inline double halo_bs2L_b1L(double b) { return 0.02278 * b * b * b - 0.005503 * b * b - 0.5904 * b - 0.1174; }
                inline double halo_blL_b1L(double b) { return -0.6971 * b * b * b + 0.7892 * b * b + 0.5882 * b - 0.1072; }
                inline double gal_b2L_b1L(double b) { return 0.01677 * b * b * b - 0.005116 * b * b + 0.4279 * b - 0.1635; }
                inline double gal_bs2L_b1L(double b) { return -0.3605 * b * b * b + 0.5649 * b * b - 0.1412 * b - 0.01318; }
                inline double gal_blL_b1L(double b) { return 0.2298 * b * b * b - 2.096 * b * b + 0.7816 * b - 0.1545; }
            } // namespace COEVOLUTION

            namespace detail {
                /// The 15 products of the bias operators (1, delta, delta^2, s^2, nabla^2 delta), i <= j, in the
                /// order of the 15 spectra P_ij: auto = sum_{ij} (2 - [i == j]) b_i b_j P_ij with b_0 = 1
                inline double combine_bias(const std::array<double, 4> & bias,
                                           const std::vector<std::vector<double>> & pnn,
                                           size_t ik) {
                    const double b[5] = {1.0, bias[0], bias[1], bias[2], bias[3]};
                    double r = 0.0;
                    size_t n = 0;
                    for (size_t i = 0; i < 5; i++)
                        for (size_t j = i; j < 5; j++, n++)
                            r += b[i] * b[j] * (i != j ? 2 : 1) * pnn[n][ik];
                    return r;
                }
            } // namespace detail

            //=========================================================================================================
            /// The 15 spectra of the second order Lagrangian bias expansion in real space (baccoemu.Lbias_expansion):
            /// nonlinear (Zennaro et al. 2021, arXiv:2101.12187) and LPT (Arico et al. 2021, arXiv:2104.14568), for
            /// the operators (1, delta, delta^2, s^2, nabla^2 delta): P_11, P_1d, P_1d2, P_1s2, P_1n2, P_dd, P_dd2,
            /// P_ds2, P_dn2, P_d2d2, P_d2s2, P_d2n2, P_s2s2, P_s2n2, P_n2n2 of the cold matter, in (Mpc/h)^3, for
            /// k <= 0.75 h/Mpc. bias = (b1, b2, bs2, blaplacian) are Lagrangian.
            //=========================================================================================================
            class LbiasExpansion {
              public:
                LbiasExpansion() : mps(matter_options()) {
                    for (int n = 0; n < 15; n++) {
                        lpt[size_t(n)].load("lbias_lpt_field" + std::to_string(n));
                        nonlinear[size_t(n)].load("lbias_nonlinear_field" + std::to_string(n));
                    }
                }

                /// The k bins (h/Mpc) of the nonlinear emulator, used when k is empty
                const std::vector<double> & kbins() const { return nonlinear[0].details.k; }
                /// The k bins of the LPT emulator
                const std::vector<double> & kbins_lpt() const { return lpt[0].details.k; }

                /// The 15 LPT spectra (get_lpt_pk); k <= 0.75 h/Mpc, cubic spline in ln k
                std::vector<std::vector<double>> lpt_pk(const Parameters & p,
                                                        double a,
                                                        const std::vector<double> & k = {}) const {
                    auto q = mps.resolve(p, true, false);
                    MatterPowerSpectrum::check(lpt[14], q, a);
                    if (not k.empty()) {
                        detail::check_k_increasing(k, "lpt_pk");
                        if (k.back() > 0.75)
                            throw std::invalid_argument("baccoemu Lbias lpt_pk: the maximum k is 0.75 h/Mpc");
                    }
                    std::vector<std::vector<double>> P(15);
                    for (size_t n = 0; n < 15; n++) {
                        auto y = lpt[n].net(MatterPowerSpectrum::input(lpt[n], q, a));
                        if (not k.empty())
                            y = detail::interp_cubic_logx(lpt[n].details.k, y, k);
                        for (auto & v : y)
                            v = std::exp(v) - lpt[n].details.subtract;
                        P[n] = y;
                    }
                    return P;
                }

                /// The BAO-smeared P(k) of the cold matter (get_smeared_bao_pk): the matter smeared-BAO emulator,
                /// cubic spline in (ln k, ln P); k in [1e-3, 30] h/Mpc
                std::vector<double> smeared_bao_pk(const Parameters & p,
                                                   double a,
                                                   const std::vector<double> & k = {}) const {
                    auto y = mps.smeared_bao_pk(p, a);
                    if (k.empty())
                        return y;
                    detail::check_k_increasing(k, "smeared_bao_pk");
                    if (k.back() > 30. or k.front() < 1e-3)
                        throw std::invalid_argument("baccoemu Lbias smeared_bao_pk: k must be in [1e-3, 30] h/Mpc");
                    auto r = detail::interp_cubic_logx(mps.kbins_smeared_bao(), detail::log_of(y), k);
                    for (auto & v : r)
                        v = std::exp(v);
                    return r;
                }

                /// The 15 nonlinear spectra (get_nonlinear_pnn); k empty: kbins(); k <= 0.75 h/Mpc, cubic spline in
                /// ln k (in P, as baccoemu does in practice)
                std::vector<std::vector<double>> nonlinear_pnn(const Parameters & p,
                                                               double a,
                                                               const std::vector<double> & k = {}) const {
                    const auto & e = nonlinear[0];
                    auto q = mps.resolve(p, true, false);
                    MatterPowerSpectrum::check(e, q, a);
                    const auto & ke = e.details.k;
                    auto pk_lpt = lpt_pk(p, a, ke);
                    const auto pk_bao = smeared_bao_pk(p, a, ke);
                    for (size_t i = 0; i < ke.size(); i++) {
                        pk_lpt[0][i] = pk_lpt[1][i] = pk_lpt[5][i] = pk_bao[i];
                        pk_lpt[4][i] = pk_lpt[8][i] = -pk_bao[i] * (ke[i] * ke[i]);
                        pk_lpt[14][i] = pk_bao[i] * std::pow(ke[i], 4);
                    }
                    static const std::array<int, 6> n_log{0, 1, 5, 9, 12, 14};
                    std::vector<std::vector<double>> P(15);
                    const auto x = MatterPowerSpectrum::input(e, q, a);
                    for (size_t n = 0; n < 15; n++) {
                        auto y = nonlinear[n].net(x);
                        const bool is_log = std::find(n_log.begin(), n_log.end(), int(n)) != n_log.end();
                        const bool is_switch = n == 10 or n == 11;
                        const auto & lpt_term = pk_lpt[is_switch ? n + 2 : n];
                        for (size_t i = 0; i < y.size(); i++) {
                            y[i] = (is_log ? std::exp(y[i]) : y[i]) * lpt_term[i];
                            if (is_switch and i < 25)
                                y[i] = y[i] * pk_lpt[n][i] / lpt_term[i];
                        }
                        P[n] = y;
                    }
                    if (k.empty())
                        return P;
                    detail::check_k_increasing(k, "nonlinear_pnn");
                    if (k.back() > 0.75)
                        throw std::invalid_argument("baccoemu Lbias nonlinear_pnn: the maximum k is 0.75 h/Mpc");
                    for (auto & v : P)
                        v = detail::interp_cubic_logx(ke, v, k);
                    return P;
                }

                /// The galaxy auto and galaxy-matter cross spectra (get_galaxy_real_pk) for bias = (b1, b2, bs2,
                /// blaplacian)
                struct GalaxyPk {
                    std::vector<double> k, auto_pk, cross_pk;
                };
                GalaxyPk galaxy_real_pk(const std::array<double, 4> & bias,
                                        const Parameters & p,
                                        double a,
                                        const std::vector<double> & k = {}) const {
                    const auto pnn = nonlinear_pnn(p, a, k);
                    GalaxyPk r;
                    r.k = k.empty() ? kbins() : k;
                    const double b[5] = {1.0, bias[0], bias[1], bias[2], bias[3]};
                    for (size_t i = 0; i < r.k.size(); i++) {
                        r.auto_pk.push_back(detail::combine_bias(bias, pnn, i));
                        double c = 0.0;
                        for (size_t j = 0; j < 5; j++)
                            c += b[j] * pnn[j][i];
                        r.cross_pk.push_back(c);
                    }
                    return r;
                }

                /// The matter emulator used for sigma8 (A_s input) and the smeared-BAO P(k)
                const MatterPowerSpectrum & matter() const { return mps; }

              private:
                MatterPowerSpectrum mps;
                std::array<detail::KerasEmulator, 15> lpt, nonlinear;
                static MatterPowerSpectrum::Options matter_options() {
                    MatterPowerSpectrum::Options o;
                    o.linear = o.nonlinear_boost = o.baryonic_boost = o.no_wiggles = false;
                    o.smeared_bao = o.sigma8 = true;
                    return o;
                }
            };

            //=========================================================================================================
            /// The 15 spectra of the Lagrangian bias expansion in redshift space (baccoemu.Lbias_expansion_RSD,
            /// Pellejero Ibanez et al. 2023, arXiv:2207.06437; the LPT spectra are an emulator of velocileptors,
            /// Chen et al. 2021): the multipoles l = 0, 2, 4 of the 15 spectra (as LbiasExpansion), and the galaxy
            /// multipoles and P(k, mu) with fingers of god and noise. k <= 0.64 h/Mpc.
            //=========================================================================================================
            class LbiasExpansionRSD {
              public:
                /// The redshift-space parameters of get_galaxy_pk: f_sat, lambda_FoG, and the noise
                /// (epsilon_1 + epsilon_2 k^2 [+ epsilon_3 k^2 mu^2]) / mean_num_dens if epsilon_1 is given
                struct RSDParameters {
                    double f_sat{NAN}, lambda_FoG{NAN};
                    double epsilon_1{NAN}, epsilon_2{NAN}, epsilon_3{NAN}, mean_num_dens{NAN};
                };
                using Multipoles = std::array<std::vector<std::vector<double>>, 3>; ///< [l/2][n][ik]

                LbiasExpansionRSD() : mps(matter_options()) {
                    for (int ell = 0; ell < 3; ell++) {
                        const std::string L = std::to_string(2 * ell);
                        bao[size_t(ell)].load("lbias_rsd_smeared_bao_ell" + L);
                        for (int n = 0; n < 15; n++) {
                            lpt[size_t(ell)][size_t(n)].load("lbias_rsd_lpt_field" + std::to_string(n) + "_ell" + L);
                            nonlinear[size_t(ell)][size_t(n)].load("lbias_rsd_nonlinear_field" + std::to_string(n) +
                                                                   "_ell" + L);
                        }
                    }
                }

                /// The k bins (h/Mpc) of the nonlinear emulator, used when k is empty
                const std::vector<double> & kbins() const { return nonlinear[0][0].details.k; }
                const std::vector<double> & kbins_lpt() const { return lpt[0][0].details.k; }

                /// The multipoles of the 15 LPT spectra (get_lpt_pk); k <= max of kbins_lpt(), cubic in ln k
                Multipoles lpt_pk(const Parameters & p, double a, const std::vector<double> & k = {}) const {
                    auto q = mps.resolve(p, true, false);
                    MatterPowerSpectrum::check(lpt[2][14], q, a);
                    const auto & ke = lpt[0][0].details.k;
                    if (not k.empty()) {
                        detail::check_k_increasing(k, "lpt_pk");
                        if (k.back() > ke.back())
                            throw std::invalid_argument("baccoemu Lbias RSD lpt_pk: the maximum k is " +
                                                        detail::to_str(ke.back()) + " h/Mpc");
                    }
                    Multipoles P;
                    for (size_t ell = 0; ell < 3; ell++)
                        for (size_t n = 0; n < 15; n++) {
                            const auto & e = lpt[ell][n];
                            auto y = e.net(MatterPowerSpectrum::input(e, q, a));
                            if (not k.empty())
                                y = detail::interp_cubic_logx(ke, y, k);
                            for (auto & v : y)
                                v = std::exp(v) - e.details.subtract;
                            P[ell].push_back(y);
                        }
                    return P;
                }

                /// The multipoles l = 0, 2, 4 of the BAO-smeared P(k) (get_smeared_bao_pk); k in [1e-3, 30] h/Mpc,
                /// cubic in ln k
                std::array<std::vector<double>, 3> smeared_bao_pk(const Parameters & p,
                                                                  double a,
                                                                  const std::vector<double> & k = {}) const {
                    auto q = mps.resolve(p, true, false);
                    MatterPowerSpectrum::check(bao[2], q, a);
                    std::array<std::vector<double>, 3> P;
                    for (size_t ell = 0; ell < 3; ell++) {
                        const auto & e = bao[ell];
                        auto y = e.net(MatterPowerSpectrum::input(e, q, a));
                        for (auto & v : y)
                            v = std::exp(v) - e.details.subtract;
                        P[ell] = y;
                    }
                    if (k.empty())
                        return P;
                    detail::check_k_increasing(k, "smeared_bao_pk");
                    if (k.back() > 30. or k.front() < 1e-3)
                        throw std::invalid_argument("baccoemu Lbias RSD smeared_bao_pk: k must be in [1e-3, 30] h/Mpc");
                    for (size_t ell = 0; ell < 3; ell++)
                        P[ell] = detail::interp_cubic_logx(bao[ell].details.k, P[ell], k);
                    return P;
                }

                /// The multipoles of the 15 nonlinear spectra (get_nonlinear_pnn); k empty: kbins()
                Multipoles nonlinear_pnn(const Parameters & p, double a, const std::vector<double> & k = {}) const {
                    const auto & e0 = nonlinear[2][14];
                    auto q = mps.resolve(p, true, false);
                    MatterPowerSpectrum::check(e0, q, a);
                    const auto & ke = nonlinear[0][0].details.k;
                    auto L = lpt_pk(q.p, a, ke);
                    const auto B = smeared_bao_pk(q.p, a, ke);
                    const size_t nk = ke.size();
                    for (size_t ell = 0; ell < 3; ell++)
                        for (size_t n : {4, 8, 13})
                            for (auto & v : L[ell][n])
                                v = -v;
                    L[1][9] = L[1][7];
                    size_t i0 = 0;
                    while (i0 < nk and not(ke[i0] > 0.2))
                        i0++;
                    for (size_t n : {1, 3, 4, 6, 7, 9, 10, 11, 12, 13, 14})
                        if (i0 < nk) {
                            const double v = L[2][n][i0];
                            for (size_t i = i0; i < nk; i++)
                                if (ke[i] > 0.2)
                                    L[2][n][i] = v;
                        }
                    for (size_t ell = 0; ell < 3; ell++)
                        L[ell][0] = B[ell];
                    for (size_t ell = 0; ell < 2; ell++) {
                        double sum = 0.0;
                        int cnt = 0;
                        for (size_t i = 0; i < nk; i++)
                            if (ke[i] < 0.05) {
                                sum += L[ell][1][i] / B[ell][i];
                                cnt++;
                            }
                        const double mean = sum / cnt;
                        for (size_t i = 0; i < nk; i++)
                            L[ell][1][i] = mean * B[ell][i];
                    }
                    for (size_t i = 0; i < nk; i++) {
                        L[1][4][i] = -(ke[i] * ke[i]) * L[1][1][i];
                        L[0][14][i] = std::pow(ke[i], 4) * L[0][5][i];
                    }
                    Multipoles P;
                    const auto x = MatterPowerSpectrum::input(e0, q, a);
                    for (size_t ell = 0; ell < 3; ell++)
                        for (size_t n = 0; n < 15; n++) {
                            if (ell == 0 and n == 10) {
                                P[ell].push_back(L[0][10]);
                                continue;
                            }
                            auto y = nonlinear[ell][n].net(x);
                            for (size_t i = 0; i < nk; i++)
                                y[i] *= L[ell][n][i];
                            P[ell].push_back(y);
                        }
                    if (k.empty())
                        return P;
                    detail::check_k_increasing(k, "nonlinear_pnn");
                    if (k.back() > ke.back())
                        throw std::invalid_argument("baccoemu Lbias RSD nonlinear_pnn: the maximum k is " +
                                                    detail::to_str(ke.back()) + " h/Mpc");
                    for (size_t ell = 0; ell < 3; ell++)
                        for (auto & v : P[ell])
                            v = detail::interp_cubic_logx(ke, v, k);
                    return P;
                }

                /// The galaxy P(k, mu) with fingers of god and noise (get_galaxy_2Dpk; epsilon_3 is not used there):
                /// result[imu][ik]. mu empty: 0, 0.01, ..., 0.99
                std::vector<std::vector<double>> galaxy_2d_pk(const std::array<double, 4> & bias,
                                                              const RSDParameters & r,
                                                              const Parameters & p,
                                                              double a,
                                                              const std::vector<double> & k = {},
                                                              std::vector<double> mu = {}) const {
                    if (mu.empty())
                        mu = default_mu();
                    const auto kk = k.empty() ? kbins() : k;
                    auto pk2d = pk2d_fog(bias, r, nonlinear_pnn(p, a, k), kk, mu);
                    if (not std::isnan(r.epsilon_1))
                        for (size_t j = 0; j < mu.size(); j++)
                            for (size_t ik = 0; ik < kk.size(); ik++)
                                pk2d[j][ik] += 1 / r.mean_num_dens * (r.epsilon_1 + r.epsilon_2 * (kk[ik] * kk[ik]));
                    return pk2d;
                }

                /// The galaxy multipoles l = 0, 2, 4 (get_galaxy_pk): P(k, mu) from the multipoles of the 15
                /// spectra, with fingers of god and noise, projected back with baccoemu's pkmulti
                std::array<std::vector<double>, 3> galaxy_pk(const std::array<double, 4> & bias,
                                                            const RSDParameters & r,
                                                            const Parameters & p,
                                                            double a,
                                                            const std::vector<double> & k = {}) const {
                    const auto mu = default_mu();
                    const auto kk = k.empty() ? kbins() : k;
                    auto pk2d = pk2d_fog(bias, r, nonlinear_pnn(p, a, k), kk, mu);
                    if (not std::isnan(r.epsilon_1))
                        for (size_t j = 0; j < mu.size(); j++)
                            for (size_t ik = 0; ik < kk.size(); ik++) {
                                const double k2 = kk[ik] * kk[ik];
                                pk2d[j][ik] += std::isnan(r.epsilon_3)
                                                   ? 1 / r.mean_num_dens * (r.epsilon_1 + r.epsilon_2 * k2)
                                                   : 1 / r.mean_num_dens * (r.epsilon_1 + r.epsilon_2 * k2 +
                                                                            r.epsilon_3 * k2 * mu[j] * mu[j]);
                            }
                    return detail::pkmulti(mu, pk2d, kk.size());
                }

                const MatterPowerSpectrum & matter() const { return mps; }

              private:
                MatterPowerSpectrum mps;
                std::array<std::array<detail::KerasEmulator, 15>, 3> lpt, nonlinear;
                std::array<detail::KerasEmulator, 3> bao;

                static MatterPowerSpectrum::Options matter_options() {
                    MatterPowerSpectrum::Options o;
                    o.linear = o.nonlinear_boost = o.baryonic_boost = o.no_wiggles = o.smeared_bao = false;
                    o.sigma8 = true;
                    return o;
                }
                static std::vector<double> default_mu() { // np.arange(0, 1, 0.01)
                    std::vector<double> mu(100);
                    for (size_t i = 0; i < mu.size(); i++)
                        mu[i] = double(i) * 0.01;
                    return mu;
                }
                static std::vector<std::vector<double>> pk2d_fog(const std::array<double, 4> & bias,
                                                                 const RSDParameters & r,
                                                                 const Multipoles & pnn,
                                                                 const std::vector<double> & k,
                                                                 const std::vector<double> & mu) {
                    if (std::isnan(r.f_sat) or std::isnan(r.lambda_FoG))
                        throw std::invalid_argument("baccoemu Lbias RSD: give f_sat and lambda_FoG");
                    if (not std::isnan(r.epsilon_1) and (std::isnan(r.epsilon_2) or std::isnan(r.mean_num_dens)))
                        throw std::invalid_argument("baccoemu Lbias RSD: epsilon_1 needs epsilon_2 and mean_num_dens");
                    std::vector<std::vector<double>> pk2d(mu.size(), std::vector<double>(k.size()));
                    for (size_t ik = 0; ik < k.size(); ik++) {
                        const double p0 = detail::combine_bias(bias, pnn[0], ik);
                        const double p2 = detail::combine_bias(bias, pnn[1], ik);
                        const double p4 = detail::combine_bias(bias, pnn[2], ik);
                        for (size_t j = 0; j < mu.size(); j++) {
                            const double m = mu[j], m2 = m * m;
                            const double L2 = 1.5 * m2 - 0.5, L4 = (35.0 * m2 * m2 - 30.0 * m2 + 3.0) / 8.0;
                            const double l2 = r.lambda_FoG * r.lambda_FoG, km = k[ik] * m;
                            const double fog = (1 - r.f_sat) + r.f_sat * l2 / (l2 + km * km);
                            pk2d[j][ik] = (p0 + p2 * L2 + p4 * L4) * (fog * fog);
                        }
                    }
                    return pk2d;
                }
            };

        } // namespace BACCO
    } // namespace EMULATOR
} // namespace FML

#endif
