#ifndef FML_EMULATOR_EMANTIS_H
#define FML_EMULATOR_EMANTIS_H

// SPDX-License-Identifier: GPL-3.0-or-later
//=====================================================================================================================
// e-MANTIS in C++: emulators of the nonlinear matter power spectrum and the halo mass function in f(R) gravity and
// wCDM, built from Gaussian processes trained on N-body simulations.
//
// Licence. This file is a modified version (a C++ port, FML, H. A. Winther, November 2026) of the prediction code of
// e-MANTIS 1.4.0 by I. Saez-Casares and the e-MANTIS authors (https://gitlab.obspm.fr/e-mantis/e-mantis), which is
// distributed under the GNU General Public License, version 3 or later. This file, export_emantis.py, the files in
// data/ (exported from e-MANTIS) and the tests are therefore also distributed under the GNU General Public License,
// version 3 or (at your option) any later version. This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
// See the GNU General Public License in COPYING (in this folder) or https://www.gnu.org/licenses/ for more details.
// FML is GPL-3.0-or-later as well (see LICENCE at the top of the repository); see README.md.
//
// Cite the papers when using the emulators:
//   P(k) f(R) boost: I. Saez-Casares, Y. Rasera & B. Li, "The e-MANTIS emulator: fast predictions of the non-linear
//                    matter power spectrum in f(R)CDM cosmology", MNRAS 527, 7242 (2024), arXiv:2303.08899,
//                    doi:10.1093/mnras/stad3343
//   HMF:             I. Saez-Casares, Y. Rasera, T. R. G. Richardson & P.-S. Corasaniti, "The e-MANTIS emulator: fast
//                    and accurate predictions of the halo mass function in f(R)CDM and wCDM cosmologies",
//                    A&A 691, A323 (2024), arXiv:2410.05226, doi:10.1051/0004-6361/202450193
//   The wCDM P_nl/P_lin boost has no paper yet (e-MANTIS 1.4.0); cite the package (doi:10.5281/zenodo.7738362)
//
// This is a port of the prediction part of e-MANTIS 1.4.0 (the Python package). The trained emulators are exported by
// export_emantis.py to the plain text files in data/ (GP weights precomputed). The predictions agree with the
// Python package to ~1e-10 (a few 1e-8 when A_s is converted to sigma8): the GP sums cancel strongly, so this is
// the floating point reproducibility of the emulators themselves (numpy's own result moves by as much if the
// summation order changes). Available:
//
//   NonLinearMGBoostEmulator  B(k, a) = P_f(R)(k) / P_LCDM(k), nonlinear, Hu-Sawicki n = 1, with the same initial
//                             conditions (sigma8_lcdm is the sigma8 of the LCDM model with the same A_s)
//   NonLinearBoostEmulator    P_nl(k) / P_lin(k) in wCDM
//   HMFEmulator               the halo mass function dn/dlnM in (h/Mpc)^3 (fR or wCDM; 200c, 500c, 1000c or FoF
//                             b = 0.2), with its std
//   Sigma8Emulator            sigma8 from A_s (LCDM or wCDM), used when A_s is given instead of sigma8
//   GPEmulator                the generic emulator (any GaussianProcessEmulator1D / 1Dx1D exported by the script)
//
// Units: k in h/Mpc, halo masses in Msun/h. Out of range inputs throw EmulationRangeError (as in Python), unless an
// extrapolation option allows them. Everything is const after construction (thread safe).
//=====================================================================================================================

#include <algorithm>
#include <cmath>
#include <fstream>
#include <limits>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace FML {
    namespace EMULATOR {
        namespace EMANTIS {

            //=========================================================================================================
            // Where the data files are: by default the data/ folder next to this header
            //=========================================================================================================
#ifdef __FILE__
            inline std::string data_dir = std::string(__FILE__).substr(0, std::string(__FILE__).find_last_of("/\\") + 1) + "data/";
#else
            inline std::string data_dir = "data/";
#endif
            inline void set_data_dir(std::string dir) {
                if (not dir.empty() and dir.back() != '/')
                    dir += '/';
                data_dir = dir;
            }

            /// An input outside the emulation range (EmulationRangeError in Python)
            class EmulationRangeError : public std::runtime_error {
              public:
                std::string param;
                double value, min_value, max_value;
                EmulationRangeError(std::string param_, double value_, double min_, double max_, std::string note = "")
                    : std::runtime_error("e-MANTIS: " + param_ + " = " + to_str(value_) + " is outside the emulation range [" +
                                         to_str(min_) + ", " + to_str(max_) + "]" + (note.empty() ? "" : ". " + note)),
                      param(param_), value(value_), min_value(min_), max_value(max_) {}

              private:
                static std::string to_str(double x) {
                    std::ostringstream s;
                    s.precision(10);
                    s << x;
                    return s.str();
                }
            };

            /// The cosmological parameters. Each emulator reads the ones it needs (others are ignored):
            ///   fR:   Omega_b, Omega_m, h, n_s, logfR0 = log10|fR0|, and sigma8 (= sigma8_lcdm) or A_s
            ///   wCDM: Omega_b, Omega_m, h, n_s, w, and sigma8 or A_s
            /// Give exactly one of sigma8 and A_s (the other NaN)
            struct Parameters {
                double Omega_b = std::numeric_limits<double>::quiet_NaN();
                double Omega_m = std::numeric_limits<double>::quiet_NaN();
                double h = std::numeric_limits<double>::quiet_NaN();
                double n_s = std::numeric_limits<double>::quiet_NaN();
                double sigma8 = std::numeric_limits<double>::quiet_NaN();
                double A_s = std::numeric_limits<double>::quiet_NaN();
                double w = -1.0;
                double logfR0 = std::numeric_limits<double>::quiet_NaN();
            };

            enum class Transform { lin, log10 };

            namespace detail {
                inline double transform(double x, Transform t) { return t == Transform::lin ? x : std::log10(x); }
                inline double inverse_transform(double x, Transform t) { return t == Transform::lin ? x : std::pow(10.0, x); }
                // Std of X = inverse_transform(Y) for Y ~ N(mean, var) (exact; log-normal for log10)
                inline double inverse_transform_std(double mean, double var, Transform t) {
                    if (t == Transform::lin)
                        return std::sqrt(var);
                    const double ln10 = std::log(10.0);
                    const double s2 = var * ln10 * ln10;
                    if (s2 <= 0.0)
                        return 0.0;
                    return std::exp(mean * ln10 + s2 + 0.5 * std::log(-std::expm1(-s2)));
                }
                inline Transform transform_from_name(const std::string & s) {
                    if (s == "lin")
                        return Transform::lin;
                    if (s == "log10")
                        return Transform::log10;
                    throw std::runtime_error("e-MANTIS: unknown transform " + s);
                }
                // y1 + (x0 - x1) (y2 - y1) / (x2 - x1), the form of e-MANTIS' lin_interp
                inline double lin_interp(double x0, double x1, double x2, double y1, double y2) {
                    return y1 + (x0 - x1) * (y2 - y1) / (x2 - x1);
                }
                inline void check_range(double value, double vmin, double vmax, const std::string & name, bool check_low = true,
                                        bool check_high = true, const std::string & note = "") {
                    if (not std::isfinite(value))
                        throw std::invalid_argument("e-MANTIS: input " + name + " is not finite");
                    if ((check_low and value < vmin) or (check_high and value > vmax))
                        throw EmulationRangeError(name, value, vmin, vmax, note);
                }

                // Reads the tokens of a file, skipping comment lines starting with '#'
                class TokenReader {
                    std::istringstream in;
                    std::string filename;

                  public:
                    explicit TokenReader(const std::string & file) : filename(file) {
                        std::ifstream f(file);
                        if (not f.good())
                            throw std::runtime_error("e-MANTIS: cannot open " + file +
                                                     " (set the data folder with FML::EMULATOR::EMANTIS::set_data_dir)");
                        std::string line, all;
                        while (std::getline(f, line)) {
                            const auto p = line.find_first_not_of(" \t");
                            if (p != std::string::npos and line[p] == '#')
                                continue;
                            all += line;
                            all += '\n';
                        }
                        in.str(all);
                    }
                    std::string word() {
                        std::string s;
                        if (not(in >> s))
                            throw std::runtime_error("e-MANTIS: unexpected end of " + filename);
                        return s;
                    }
                    void expect(const std::string & key) {
                        const std::string s = word();
                        if (s != key)
                            throw std::runtime_error("e-MANTIS: expected '" + key + "' in " + filename + ", got '" + s + "'");
                    }
                    double number() {
                        // strtod reads inf/nan too
                        const std::string s = word();
                        char * end = nullptr;
                        const double x = std::strtod(s.c_str(), &end);
                        if (end == s.c_str() or *end != '\0')
                            throw std::runtime_error("e-MANTIS: expected a number in " + filename + ", got '" + s + "'");
                        return x;
                    }
                    long integer() {
                        const double x = number();
                        if (x != std::floor(x) or x < 0)
                            throw std::runtime_error("e-MANTIS: expected a non-negative integer in " + filename);
                        return long(x);
                    }
                    std::vector<double> numbers(size_t n) {
                        std::vector<double> v(n);
                        for (auto & x : v)
                            x = number();
                        return v;
                    }
                    const std::string & name() const { return filename; }
                };

                // One Gaussian process: ConstantKernel(amp) * Matern(nu, length_scale) [+ WhiteKernel(noise)],
                // as scikit-learn's GaussianProcessRegressor with normalize_y
                struct GaussianProcess {
                    double nu{2.5};    // 0.5, 1.5, 2.5, or < 0 for RBF
                    double amp{1.0};   // ConstantKernel
                    double noise{0.0}; // WhiteKernel noise level (0 if none)
                    double y_mean{0.0}, y_std{1.0};
                    std::vector<double> length_scale;
                    std::vector<double> Xs; // the training parameters / length_scale (as scikit-learn: X / ls, then distances)
                    std::vector<double> alpha; // K^-1 (y - y_mean) / y_std
                    std::vector<double> reg;   // the regularisation added to the diagonal of K (per training point)
                    std::vector<double> L;     // Cholesky factor of K (row major, lower), only for the std

                    double correlation(double d) const {
                        if (nu < 0)
                            return std::exp(-0.5 * d * d);
                        if (nu == 0.5)
                            return std::exp(-d);
                        if (nu == 1.5) {
                            const double x = std::sqrt(3.0) * d;
                            return (1.0 + x) * std::exp(-x);
                        }
                        const double x = std::sqrt(5.0) * d;
                        return (1.0 + x + x * x / 3.0) * std::exp(-x);
                    }
                };

                struct Node {
                    double value{0.0};
                    bool bspline{false};
                    std::vector<double> bins;  // binned: the bins; bspline: the knots
                    int degree{0};             // bspline degree
                    std::vector<double> bins_tr; // binned: the transformed bins (x_interp_type)
                    // PCA (n_pca = 0 if none)
                    int n_pca{0}, n_features{0};
                    std::vector<double> pca_scaler_mean, pca_scaler_scale, pca_mean, components; // components: n_pca x n_features
                    std::vector<GaussianProcess> gps;
                    std::vector<double> gp_std_factor; // empty if none
                    // The range of the bin variable (untransformed)
                    double bin_min{0.0}, bin_max{0.0};
                    size_t n_latent() const { return n_pca > 0 ? size_t(n_features) : gps.size(); }
                };

                // Cholesky decomposition (lower, in place, row major). Throws if not positive definite
                inline void cholesky(std::vector<double> & A, size_t n) {
                    for (size_t j = 0; j < n; j++) {
                        double s = A[j * n + j];
                        for (size_t k = 0; k < j; k++)
                            s -= A[j * n + k] * A[j * n + k];
                        if (not(s > 0.0))
                            throw std::runtime_error("e-MANTIS: the kernel matrix is not positive definite");
                        const double d = std::sqrt(s);
                        A[j * n + j] = d;
                        for (size_t i = j + 1; i < n; i++) {
                            double t = A[i * n + j];
                            for (size_t k = 0; k < j; k++)
                                t -= A[i * n + k] * A[j * n + k];
                            A[i * n + j] = t / d;
                        }
                        for (size_t k = j + 1; k < n; k++)
                            A[j * n + k] = 0.0;
                    }
                }

                // The B-spline basis functions nonzero on interval l at x (scipy's _deBoor_D with m = 0): h[0..k]
                // multiply the coefficients c[l - k .. l]. Outside the base interval the end polynomials are used
                // (scipy's BSpline(extrapolate=True))
                inline size_t bspline_interval(const std::vector<double> & t, int k, double x) {
                    const size_t n = t.size() - k - 1;
                    if (x < t[k])
                        return k;
                    if (x >= t[n])
                        return n - 1;
                    size_t l = std::upper_bound(t.begin(), t.end(), x) - t.begin() - 1;
                    return std::min(std::max(l, size_t(k)), n - 1);
                }
                inline void bspline_basis(const std::vector<double> & t, int k, double x, size_t l, std::vector<double> & h) {
                    h.assign(k + 1, 0.0);
                    std::vector<double> hh(k + 1);
                    h[0] = 1.0;
                    for (int j = 1; j <= k; j++) {
                        for (int i = 0; i < j; i++)
                            hh[i] = h[i];
                        h[0] = 0.0;
                        for (int n = 1; n <= j; n++) {
                            const size_t ind = l + n;
                            const double xb = t[ind], xa = t[ind - j];
                            if (xb == xa) {
                                h[n] = 0.0;
                                continue;
                            }
                            const double w = hh[n - 1] / (xb - xa);
                            h[n - 1] += w * (xb - x);
                            h[n] = w * (x - xa);
                        }
                    }
                }
            } // namespace detail

            //=========================================================================================================
            /// A Gaussian process emulator exported from e-MANTIS (GaussianProcessEmulator1D or 1Dx1D): an observable
            /// y(x) on bins (or B-spline coefficients) for each node (scale factor), each output (or PCA component)
            /// emulated by a GP in the cosmological parameters. Between nodes the transformed predictions are
            /// interpolated linearly. predict() returns the observable at the given x (the original bins if x is
            /// empty, binned data only)
            //=========================================================================================================
            class GPEmulator {
              public:
                GPEmulator() = default;
                explicit GPEmulator(const std::string & filename, bool with_std = false) { load(filename, with_std); }

                void load(const std::string & filename, bool with_std = false);

                /// The prediction for one cosmology (params in the order of param_names()) at the node variable
                /// node_var (ignored if there are no nodes). std_out (if not null) gets the std of the prediction
                /// (needs with_std = true at load)
                std::vector<double> predict(const std::vector<double> & params,
                                            double node_var,
                                            const std::vector<double> & x = {},
                                            bool extrapolate_x_low = false,
                                            bool extrapolate_x_high = false,
                                            std::vector<double> * std_out = nullptr) const;

                /// The prediction for parameters given by name
                std::vector<double> predict(const std::map<std::string, double> & params,
                                            double node_var,
                                            const std::vector<double> & x = {},
                                            bool extrapolate_x_low = false,
                                            bool extrapolate_x_high = false,
                                            std::vector<double> * std_out = nullptr) const {
                    return predict(params_from_map(params), node_var, x, extrapolate_x_low, extrapolate_x_high, std_out);
                }

                std::vector<double> params_from_map(const std::map<std::string, double> & params) const {
                    std::vector<double> p(names.size());
                    for (size_t i = 0; i < names.size(); i++) {
                        auto it = params.find(names[i]);
                        if (it == params.end())
                            throw std::invalid_argument("e-MANTIS: the parameter '" + names[i] + "' is missing");
                        p[i] = it->second;
                    }
                    for (auto & kv : params)
                        if (std::find(names.begin(), names.end(), kv.first) == names.end())
                            throw std::invalid_argument("e-MANTIS: the parameter '" + kv.first + "' is not supported");
                    return p;
                }

                const std::string & name() const { return emu_name; }
                const std::vector<std::string> & param_names() const { return names; }
                size_t param_index(const std::string & name) const {
                    auto it = std::find(names.begin(), names.end(), name);
                    if (it == names.end())
                        throw std::invalid_argument("e-MANTIS: unknown parameter '" + name + "'");
                    return it - names.begin();
                }
                /// The range of a parameter, of the node variable or of the bin variable (at node_var; the
                /// intersection of the ranges of the two neighbouring nodes in between nodes)
                std::pair<double, double> range(const std::string & name, double node_var = std::nan("")) const;
                bool has_nodes() const { return node_variable != "none"; }
                std::vector<double> node_values() const {
                    std::vector<double> v;
                    for (auto & n : nodes)
                        v.push_back(n.value);
                    return v;
                }
                /// The bins (binned) or knots (bspline) of node i
                const std::vector<double> & bins(size_t i = 0) const { return nodes.at(i).bins; }
                const std::string & bin_variable() const { return bin_variable_name; }
                const std::string & node_variable_name() const { return node_variable; }
                /// Other key-value pairs of the file header (kind, model, mass_def, ...)
                std::string info(const std::string & key) const {
                    auto it = header.find(key);
                    return it == header.end() ? "" : it->second;
                }

              private:
                std::string emu_name;
                std::map<std::string, std::string> header;
                bool bspline{false};
                std::string bin_variable_name;
                Transform x_transform{Transform::lin}, y_transform{Transform::lin};
                std::vector<std::string> names;
                std::vector<double> pmin, pmax, scaler_mean, scaler_scale;
                size_t n_train{0};
                std::vector<double> X; // n_train x n_params (scaled)
                std::string node_variable{"none"};
                Transform node_transform{Transform::lin};
                bool constant_bin_range{true};
                std::vector<detail::Node> nodes;
                bool has_std{false};

                // At one node: the latent mean (transformed observable at x) and, if var != nullptr, its variance
                void predict_node(const detail::Node & node,
                                  const std::vector<double> & z,
                                  const std::vector<double> & x,
                                  std::vector<double> & latent,
                                  std::vector<double> * var) const;
                // The weights of the bin map (latent at the features -> at x): for each x the (index, weight) pairs
                void bin_map_weights(const detail::Node & node,
                                     double x,
                                     std::vector<std::pair<size_t, double>> & w) const;
                std::pair<size_t, size_t> neighbour_nodes(double v) const;
            };

            inline void GPEmulator::load(const std::string & filename, bool with_std) {
                detail::TokenReader r(filename);
                r.expect("FML_EMANTIS");
                if (r.integer() != 1)
                    throw std::runtime_error("e-MANTIS: unsupported format version in " + filename);
                r.expect("name");
                emu_name = r.word();
                header.clear();
                std::string key;
                while ((key = r.word()) != "data")
                    header[key] = r.word();
                const std::string type = r.word();
                if (type != "binned" and type != "bspline")
                    throw std::runtime_error("e-MANTIS: unknown data type " + type);
                bspline = type == "bspline";
                bin_variable_name = r.word();
                x_transform = detail::transform_from_name(r.word());
                y_transform = detail::transform_from_name(r.word());

                r.expect("params");
                const size_t np = r.integer();
                names.resize(np);
                pmin.resize(np);
                pmax.resize(np);
                for (size_t i = 0; i < np; i++) {
                    names[i] = r.word();
                    pmin[i] = r.number();
                    pmax[i] = r.number();
                }
                r.expect("scaler_mean");
                scaler_mean = r.numbers(np);
                r.expect("scaler_scale");
                scaler_scale = r.numbers(np);
                r.expect("train");
                n_train = r.integer();
                X = r.numbers(n_train * np);

                r.expect("nodes");
                node_variable = r.word();
                node_transform = detail::transform_from_name(r.word());
                constant_bin_range = r.integer() != 0;
                const size_t nn = r.integer();
                nodes.assign(nn, {});
                for (auto & node : nodes) {
                    r.expect("node");
                    node.value = r.number();
                    node.bspline = bspline;
                    if (bspline) {
                        r.expect("bspline");
                        node.degree = int(r.integer());
                        node.bins = r.numbers(r.integer());
                        const size_t nk = node.bins.size();
                        node.bin_min = detail::inverse_transform(node.bins[node.degree], x_transform);
                        node.bin_max = detail::inverse_transform(node.bins[nk - node.degree - 1], x_transform);
                        if (node.bin_min > node.bin_max)
                            std::swap(node.bin_min, node.bin_max);
                    } else {
                        r.expect("bins");
                        node.bins = r.numbers(r.integer());
                        node.bins_tr.resize(node.bins.size());
                        for (size_t i = 0; i < node.bins.size(); i++)
                            node.bins_tr[i] = detail::transform(node.bins[i], x_transform);
                        if (not node.bins.empty()) {
                            node.bin_min = *std::min_element(node.bins.begin(), node.bins.end());
                            node.bin_max = *std::max_element(node.bins.begin(), node.bins.end());
                        }
                    }
                    r.expect("pca");
                    node.n_pca = int(r.integer());
                    node.n_features = int(r.integer());
                    if (node.n_pca > 0) {
                        r.expect("pca_scaler_mean");
                        node.pca_scaler_mean = r.numbers(node.n_features);
                        r.expect("pca_scaler_scale");
                        node.pca_scaler_scale = r.numbers(node.n_features);
                        r.expect("pca_mean");
                        node.pca_mean = r.numbers(node.n_features);
                        node.components = r.numbers(size_t(node.n_pca) * node.n_features);
                    }
                    r.expect("gps");
                    const size_t ngp = r.integer();
                    const bool has_factor = r.integer() != 0;
                    if (has_factor) {
                        r.expect("gp_std_factor");
                        node.gp_std_factor = r.numbers(ngp);
                    }
                    if (node.n_pca > 0 and ngp != size_t(node.n_pca))
                        throw std::runtime_error("e-MANTIS: the number of GPs differs from the number of PCA components in " + filename);
                    node.gps.resize(ngp);
                    for (auto & gp : node.gps) {
                        r.expect("gp");
                        gp.nu = r.number();
                        gp.amp = r.number();
                        gp.noise = r.number();
                        gp.y_mean = r.number();
                        gp.y_std = r.number();
                        gp.length_scale = r.numbers(np);
                        gp.Xs.resize(n_train * np);
                        for (size_t j = 0; j < n_train; j++)
                            for (size_t p = 0; p < np; p++)
                                gp.Xs[j * np + p] = X[j * np + p] / gp.length_scale[p];
                        r.expect("alpha");
                        gp.alpha = r.numbers(n_train);
                        r.expect("reg");
                        const size_t nreg = r.integer();
                        gp.reg = r.numbers(nreg);
                        if (nreg == 1)
                            gp.reg.assign(n_train, gp.reg[0]);
                        else if (nreg != n_train)
                            throw std::runtime_error("e-MANTIS: wrong number of regularisation values in " + filename);
                    }
                    // Sanity checks of the sizes
                    const size_t nlat = node.n_latent();
                    if (not bspline and not node.bins.empty() and node.bins.size() != nlat)
                        throw std::runtime_error("e-MANTIS: the number of bins differs from the number of outputs in " + filename);
                    if (bspline and node.bins.size() != nlat + node.degree + 1)
                        throw std::runtime_error("e-MANTIS: the number of knots does not match the B-spline in " + filename);
                }
                r.expect("end");

                // The Cholesky factors of the kernel matrices (for the std)
                has_std = with_std;
                if (with_std) {
                    for (auto & node : nodes) {
                        for (auto & gp : node.gps) {
                            gp.L.assign(n_train * n_train, 0.0);
                            for (size_t i = 0; i < n_train; i++) {
                                for (size_t j = 0; j <= i; j++) {
                                    double d2 = 0.0;
                                    for (size_t p = 0; p < np; p++) {
                                        const double d = gp.Xs[i * np + p] - gp.Xs[j * np + p];
                                        d2 += d * d;
                                    }
                                    double kij = gp.amp * gp.correlation(std::sqrt(d2));
                                    if (i == j)
                                        kij += gp.noise + gp.reg[i];
                                    gp.L[i * n_train + j] = gp.L[j * n_train + i] = kij;
                                }
                            }
                            detail::cholesky(gp.L, n_train);
                        }
                    }
                }
            }

            inline std::pair<size_t, size_t> GPEmulator::neighbour_nodes(double v) const {
                // As e-MANTIS: searchsorted(side = right), the upper node clamped to the last one
                size_t i = std::upper_bound(nodes.begin(), nodes.end(), v,
                                            [](double a, const detail::Node & n) { return a < n.value; }) -
                           nodes.begin();
                i = std::min(i, nodes.size() - 1);
                return {i - 1, i};
            }

            inline std::pair<double, double> GPEmulator::range(const std::string & name, double node_var) const {
                if (name == bin_variable_name) {
                    if (not has_nodes() or constant_bin_range)
                        return {nodes[0].bin_min, nodes[0].bin_max};
                    if (std::isnan(node_var))
                        throw std::invalid_argument("e-MANTIS: the range of " + name + " depends on " + node_variable);
                    detail::check_range(node_var, nodes.front().value, nodes.back().value, node_variable);
                    for (auto & n : nodes)
                        if (n.value == node_var)
                            return {n.bin_min, n.bin_max};
                    auto nb = neighbour_nodes(node_var);
                    return {std::max(nodes[nb.first].bin_min, nodes[nb.second].bin_min),
                            std::min(nodes[nb.first].bin_max, nodes[nb.second].bin_max)};
                }
                if (has_nodes() and name == node_variable)
                    return {nodes.front().value, nodes.back().value};
                const size_t i = param_index(name);
                return {pmin[i], pmax[i]};
            }

            inline void GPEmulator::bin_map_weights(const detail::Node & node,
                                                    double x,
                                                    std::vector<std::pair<size_t, double>> & w) const {
                w.clear();
                const double xt = detail::transform(x, x_transform);
                if (node.bspline) {
                    const int k = node.degree;
                    const size_t l = detail::bspline_interval(node.bins, k, xt);
                    std::vector<double> h;
                    detail::bspline_basis(node.bins, k, xt, l, h);
                    for (int a = 0; a <= k; a++)
                        w.push_back({l - k + a, h[a]});
                    return;
                }
                // Linear interpolation in the transformed x, extrapolated linearly with the end segments
                // (scipy's make_interp_spline(k = 1))
                const auto & t = node.bins_tr;
                const size_t n = t.size();
                if (n == 1) {
                    w.push_back({0, 1.0});
                    return;
                }
                size_t i = std::upper_bound(t.begin(), t.end(), xt) - t.begin();
                i = std::min(std::max(i, size_t(1)), n - 1);
                const double f = (xt - t[i - 1]) / (t[i] - t[i - 1]);
                w.push_back({i - 1, 1.0 - f});
                w.push_back({i, f});
            }

            inline void GPEmulator::predict_node(const detail::Node & node,
                                                 const std::vector<double> & z,
                                                 const std::vector<double> & x,
                                                 std::vector<double> & latent,
                                                 std::vector<double> * var) const {
                const size_t np = names.size();
                const size_t ngp = node.gps.size();
                std::vector<double> gmean(ngp), gvar(var ? ngp : 0);
                std::vector<double> kvec(n_train), v(n_train), zs(np);
                for (size_t g = 0; g < ngp; g++) {
                    const auto & gp = node.gps[g];
                    for (size_t p = 0; p < np; p++)
                        zs[p] = z[p] / gp.length_scale[p];
                    double s = 0.0;
                    for (size_t j = 0; j < n_train; j++) {
                        double d2 = 0.0;
                        for (size_t p = 0; p < np; p++) {
                            const double d = zs[p] - gp.Xs[j * np + p];
                            d2 += d * d;
                        }
                        kvec[j] = gp.amp * gp.correlation(std::sqrt(d2));
                        s += kvec[j] * gp.alpha[j];
                    }
                    gmean[g] = gp.y_std * s + gp.y_mean;
                    if (var) {
                        // v = L^-1 k, var = k(x, x) - v.v
                        double vv = 0.0;
                        for (size_t i = 0; i < n_train; i++) {
                            double t = kvec[i];
                            for (size_t k = 0; k < i; k++)
                                t -= gp.L[i * n_train + k] * v[k];
                            v[i] = t / gp.L[i * n_train + i];
                            vv += v[i] * v[i];
                        }
                        double gv = gp.amp + gp.noise - vv;
                        if (gv < 0.0)
                            gv = 0.0;
                        gv *= gp.y_std * gp.y_std;
                        if (not node.gp_std_factor.empty())
                            gv *= node.gp_std_factor[g] * node.gp_std_factor[g];
                        gvar[g] = gv;
                    }
                }

                // The latent features (transformed observable at the data bins / B-spline coefficients)
                const size_t nlat = node.n_latent();
                std::vector<double> feat(nlat);
                if (node.n_pca > 0) {
                    for (size_t f = 0; f < nlat; f++) {
                        double s = node.pca_mean[f];
                        for (size_t g = 0; g < ngp; g++)
                            s += gmean[g] * node.components[g * nlat + f];
                        feat[f] = s * node.pca_scaler_scale[f] + node.pca_scaler_mean[f];
                    }
                } else {
                    feat = gmean;
                }

                // Linear operator from the GP outputs to feature f: op(g, f)
                auto op = [&](size_t g, size_t f) -> double {
                    if (node.n_pca > 0)
                        return node.components[g * nlat + f] * node.pca_scaler_scale[f];
                    return g == f ? 1.0 : 0.0;
                };

                if (x.empty()) {
                    latent = feat;
                    if (var) {
                        var->assign(nlat, 0.0);
                        for (size_t f = 0; f < nlat; f++)
                            for (size_t g = 0; g < ngp; g++) {
                                const double o = op(g, f);
                                (*var)[f] += gvar[g] * o * o;
                            }
                    }
                    return;
                }
                latent.assign(x.size(), 0.0);
                if (var)
                    var->assign(x.size(), 0.0);
                std::vector<std::pair<size_t, double>> w;
                std::vector<double> opx(ngp);
                for (size_t i = 0; i < x.size(); i++) {
                    bin_map_weights(node, x[i], w);
                    double s = 0.0;
                    for (auto & iw : w)
                        s += feat[iw.first] * iw.second;
                    latent[i] = s;
                    if (var) {
                        std::fill(opx.begin(), opx.end(), 0.0);
                        for (auto & iw : w)
                            for (size_t g = 0; g < ngp; g++)
                                opx[g] += op(g, iw.first) * iw.second;
                        double sv = 0.0;
                        for (size_t g = 0; g < ngp; g++)
                            sv += gvar[g] * opx[g] * opx[g];
                        (*var)[i] = sv;
                    }
                }
            }

            inline std::vector<double> GPEmulator::predict(const std::vector<double> & params,
                                                           double node_var,
                                                           const std::vector<double> & x,
                                                           bool extrapolate_x_low,
                                                           bool extrapolate_x_high,
                                                           std::vector<double> * std_out) const {
                if (nodes.empty())
                    throw std::runtime_error("e-MANTIS: the emulator is not loaded");
                const size_t np = names.size();
                if (params.size() != np)
                    throw std::invalid_argument("e-MANTIS: " + emu_name + " expects " + std::to_string(np) + " parameters");
                if (std_out and not has_std)
                    throw std::invalid_argument("e-MANTIS: load the emulator with with_std = true to get the std");
                if (bspline and x.empty())
                    throw std::invalid_argument("e-MANTIS: give the " + bin_variable_name + " values (B-spline data)");

                // The nodes to use
                size_t ilow = 0, iup = 0;
                bool at_node = true;
                if (has_nodes()) {
                    detail::check_range(node_var, nodes.front().value, nodes.back().value, node_variable);
                    auto it = std::find_if(nodes.begin(), nodes.end(), [&](const detail::Node & n) { return n.value == node_var; });
                    if (it != nodes.end()) {
                        ilow = iup = it - nodes.begin();
                    } else {
                        auto nb = neighbour_nodes(node_var);
                        ilow = nb.first;
                        iup = nb.second;
                        at_node = false;
                    }
                }
                if (x.empty() and nodes[ilow].bins != nodes[iup].bins)
                    throw std::invalid_argument("e-MANTIS: the nodes have different bins: give the " + bin_variable_name + " values");

                // Range checks (in the order of e-MANTIS): the bins at each node used, then the parameters
                for (size_t in : {ilow, iup}) {
                    for (double xi : x) {
                        detail::check_range(xi, nodes[in].bin_min, nodes[in].bin_max, bin_variable_name, not extrapolate_x_low,
                                            not extrapolate_x_high);
                        if (x_transform == Transform::log10 and not(xi > 0.0))
                            throw std::invalid_argument("e-MANTIS: " + bin_variable_name + " must be positive");
                    }
                }

                for (size_t i = 0; i < np; i++)
                    detail::check_range(params[i], pmin[i], pmax[i], names[i]);

                std::vector<double> z(np);
                for (size_t i = 0; i < np; i++)
                    z[i] = (params[i] - scaler_mean[i]) / scaler_scale[i];

                std::vector<double> lat_low, var_low;
                predict_node(nodes[ilow], z, x, lat_low, std_out ? &var_low : nullptr);
                std::vector<double> result(lat_low.size());
                if (at_node) {
                    for (size_t i = 0; i < result.size(); i++)
                        result[i] = detail::inverse_transform(lat_low[i], y_transform);
                    if (std_out) {
                        std_out->resize(result.size());
                        for (size_t i = 0; i < result.size(); i++)
                            (*std_out)[i] = detail::inverse_transform_std(lat_low[i], var_low[i], y_transform);
                    }
                    return result;
                }
                std::vector<double> lat_up, var_up;
                predict_node(nodes[iup], z, x, lat_up, std_out ? &var_up : nullptr);
                const double xt = detail::transform(node_var, node_transform);
                const double xl = detail::transform(nodes[ilow].value, node_transform);
                const double xu = detail::transform(nodes[iup].value, node_transform);
                for (size_t i = 0; i < result.size(); i++)
                    result[i] = detail::inverse_transform(detail::lin_interp(xt, xl, xu, lat_low[i], lat_up[i]), y_transform);
                if (std_out) {
                    const double wu = detail::lin_interp(xt, xl, xu, 0.0, 1.0);
                    const double wl = 1.0 - wu;
                    std_out->resize(result.size());
                    for (size_t i = 0; i < result.size(); i++) {
                        const double m = wl * lat_low[i] + wu * lat_up[i];
                        const double v = wl * wl * var_low[i] + wu * wu * var_up[i];
                        (*std_out)[i] = detail::inverse_transform_std(m, v, y_transform);
                    }
                }
                return result;
            }

            //=========================================================================================================
            /// sigma8 as a function of A_s: model "LCDM" (Omega_b, Omega_m, h, n_s, A_s) or "wCDM" (also w)
            //=========================================================================================================
            class Sigma8Emulator {
              public:
                Sigma8Emulator() = default;
                explicit Sigma8Emulator(const std::string & model) { load(model); }
                void load(const std::string & model) {
                    if (model != "LCDM" and model != "wCDM")
                        throw std::invalid_argument("e-MANTIS: the sigma8 model must be LCDM or wCDM");
                    model_name = model;
                    emu.load(data_dir + "sigma8_" + model + ".txt");
                }
                /// sigma8 for the parameters p (p.A_s, not p.sigma8, is used)
                double sigma8(const Parameters & p) const {
                    check_ranges(p);
                    return emu.predict(param_vector(p), 0.0)[0];
                }
                /// The range checks of the conversion A_s -> sigma8 (A_s and the other parameters, in the sigma8
                /// emulator ranges)
                void check_ranges(const Parameters & p) const {
                    const std::string note = "This is the range of the sigma8 emulator used to convert A_s into sigma8.";
                    auto r = emu.range("ln1e10As");
                    detail::check_range(p.A_s, std::exp(r.first) * 1e-10, std::exp(r.second) * 1e-10, "A_s", true, true, note);
                    auto v = param_vector(p);
                    for (size_t i = 0; i < v.size(); i++) {
                        if (emu.param_names()[i] == "ln1e10As")
                            continue;
                        auto ri = emu.range(emu.param_names()[i]);
                        detail::check_range(v[i], ri.first, ri.second, emu.param_names()[i], true, true, note);
                    }
                }
                std::pair<double, double> range(const std::string & name) const { return emu.range(name); }
                const GPEmulator & emulator() const { return emu; }

              private:
                std::string model_name;
                GPEmulator emu;
                std::vector<double> param_vector(const Parameters & p) const {
                    std::map<std::string, double> m{{"Omega_b", p.Omega_b}, {"Omega_m", p.Omega_m}, {"h", p.h},
                                                    {"n_s", p.n_s},         {"ln1e10As", std::log(p.A_s * 1e10)}};
                    if (model_name == "wCDM")
                        m["w"] = p.w;
                    return emu.params_from_map(m);
                }
            };

            namespace detail {
                // The parameter vector of an fR or wCDM emulator (sigma8 from A_s if A_s is given)
                inline std::vector<double> cosmo_vector(const GPEmulator & emu,
                                                        const std::string & model,
                                                        const Parameters & p,
                                                        const Sigma8Emulator & s8) {
                    const bool has_s8 = not std::isnan(p.sigma8), has_As = not std::isnan(p.A_s);
                    if (has_s8 == has_As)
                        throw std::invalid_argument("e-MANTIS: give exactly one of sigma8 and A_s");
                    const double sigma8 = has_As ? s8.sigma8(p) : p.sigma8;
                    std::map<std::string, double> m{{"Omega_b", p.Omega_b}, {"Omega_m", p.Omega_m}, {"h", p.h}, {"n_s", p.n_s}};
                    if (model == "fR") {
                        m["sigma8_lcdm"] = sigma8;
                        m["logfR0"] = p.logfR0;
                    } else {
                        m["sigma8"] = sigma8;
                        m["w"] = p.w;
                    }
                    return emu.params_from_map(m);
                }
            } // namespace detail

            //=========================================================================================================
            /// The nonlinear f(R) boost B(k, a) = P_f(R)(k, a) / P_LCDM(k, a) (Hu-Sawicki n = 1), for
            /// Omega_b, Omega_m, h, n_s, logfR0 = log10|fR0| in [4, 7] and sigma8 = sigma8_lcdm (or A_s), at
            /// 0.25 <= a <= 1 and k <= ~9.7 h/Mpc. As e-MANTIS' NonLinearMGBoostEmulator('fR'):
            ///   extrapolate_k_low:  below the first bin k0 the extrapolated boost, at most the boost at k0
            ///                       (otherwise k < k0 throws)
            ///   extrapolate_cosmo:  the parameters other than logfR0 are clamped to the emulation range
            ///   the boost is never below 1
            //=========================================================================================================
            class NonLinearMGBoostEmulator {
              public:
                NonLinearMGBoostEmulator() = default;
                explicit NonLinearMGBoostEmulator(const std::string & model) { load(model); }
                void load(const std::string & model = "fR") {
                    if (model != "fR")
                        throw std::invalid_argument("e-MANTIS: the MG boost model must be fR");
                    emu.load(data_dir + "pk_nl_mg_boost_fR.txt");
                    s8.load("LCDM");
                }

                /// The boost at the wavenumbers k (h/Mpc); the default bins (kbins()) if k is empty
                std::vector<double> boost(const Parameters & p,
                                          double a,
                                          std::vector<double> k = {},
                                          bool extrapolate_k_low = true,
                                          bool extrapolate_cosmo = false) const {
                    Parameters q = p;
                    if (not std::isnan(q.A_s)) {
                        if (not std::isnan(q.sigma8))
                            throw std::invalid_argument("e-MANTIS: give exactly one of sigma8 and A_s");
                        q.sigma8 = s8.sigma8(q);
                        q.A_s = std::numeric_limits<double>::quiet_NaN();
                    }
                    auto v = detail::cosmo_vector(emu, "fR", q, s8);
                    if (extrapolate_cosmo) {
                        for (size_t i = 0; i < v.size(); i++) {
                            if (not std::isfinite(v[i]))
                                throw std::invalid_argument("e-MANTIS: input " + emu.param_names()[i] + " is not finite");
                            if (emu.param_names()[i] == "logfR0")
                                continue;
                            auto r = emu.range(emu.param_names()[i]);
                            v[i] = std::min(std::max(v[i], r.first), r.second);
                        }
                    }
                    const double k0 = emu.range("k").first;
                    std::vector<size_t> low;
                    if (extrapolate_k_low and not k.empty()) {
                        for (size_t i = 0; i < k.size(); i++)
                            if (k[i] < k0)
                                low.push_back(i);
                        if (not low.empty())
                            k.push_back(k0);
                    }
                    auto b = emu.predict(v, a, k, extrapolate_k_low, false);
                    if (not low.empty()) {
                        const double b0 = b.back();
                        b.pop_back();
                        for (size_t i : low)
                            b[i] = std::min(b[i], b0);
                    }
                    for (auto & x : b)
                        x = std::max(x, 1.0);
                    return b;
                }
                const std::vector<double> & kbins() const { return emu.bins(0); }
                std::vector<double> aexp_nodes() const { return emu.node_values(); }
                std::pair<double, double> range(const std::string & name) const { return emu.range(name); }
                const GPEmulator & emulator() const { return emu; }

              private:
                GPEmulator emu;
                Sigma8Emulator s8;
            };

            //=========================================================================================================
            /// The nonlinear boost P_nl(k, a) / P_lin(k, a) in wCDM (Omega_b, Omega_m, h, n_s, w, sigma8 or A_s)
            //=========================================================================================================
            class NonLinearBoostEmulator {
              public:
                NonLinearBoostEmulator() = default;
                explicit NonLinearBoostEmulator(const std::string & model) { load(model); }
                void load(const std::string & model = "wCDM") {
                    if (model != "wCDM")
                        throw std::invalid_argument("e-MANTIS: the nonlinear boost model must be wCDM");
                    emu.load(data_dir + "pk_nl_boost_wCDM.txt");
                    s8.load("wCDM");
                }
                /// P_nl / P_lin at the wavenumbers k (h/Mpc); the default bins (kbins()) if k is empty
                std::vector<double> boost(const Parameters & p, double a, const std::vector<double> & k = {}) const {
                    return emu.predict(detail::cosmo_vector(emu, "wCDM", p, s8), a, k);
                }
                const std::vector<double> & kbins() const { return emu.bins(0); }
                std::vector<double> aexp_nodes() const { return emu.node_values(); }
                std::pair<double, double> range(const std::string & name) const { return emu.range(name); }
                const GPEmulator & emulator() const { return emu; }

              private:
                GPEmulator emu;
                Sigma8Emulator s8;
            };

            //=========================================================================================================
            /// The halo mass function for model "fR" or "wCDM", mass definition "200c", "500c", "1000c" or "b0.2"
            /// (FoF), with or without the resolution correction at low masses. Masses in Msun/h. The std of the
            /// prediction can be returned. Above the mass range it is extrapolated only with extrapolate_mass_high
            /// (not reliable, as in e-MANTIS)
            //=========================================================================================================
            class HMFEmulator {
              public:
                HMFEmulator() = default;
                HMFEmulator(const std::string & model, const std::string & mass_def, bool use_resolution_correction = true) {
                    load(model, mass_def, use_resolution_correction);
                }
                void load(const std::string & model, const std::string & mass_def, bool use_resolution_correction = true) {
                    if (model != "fR" and model != "wCDM")
                        throw std::invalid_argument("e-MANTIS: the HMF model must be fR or wCDM");
                    if (mass_def != "200c" and mass_def != "500c" and mass_def != "1000c" and mass_def != "b0.2")
                        throw std::invalid_argument("e-MANTIS: unsupported mass definition " + mass_def +
                                                    " (200c, 500c, 1000c, b0.2)");
                    model_name = model;
                    emu.load(data_dir + "hmf_" + model + "_" + mass_def + "_res_corr_" + (use_resolution_correction ? "1" : "0") +
                                 ".txt",
                             true);
                    s8.load(model == "fR" ? "LCDM" : "wCDM");
                }
                /// The halo mass function dn/dlnM in (h/Mpc)^3 at the masses (Msun/h); its std in std_out if not null
                std::vector<double> hmf(const std::vector<double> & mass_halo,
                                        const Parameters & p,
                                        double a,
                                        bool extrapolate_mass_high = false,
                                        std::vector<double> * std_out = nullptr) const {
                    return emu.predict(detail::cosmo_vector(emu, model_name, p, s8), a, mass_halo, false, extrapolate_mass_high,
                                       std_out);
                }
                std::vector<double> aexp_nodes() const { return emu.node_values(); }
                /// The range of the mass (at the scale factor a), a parameter or aexp
                std::pair<double, double> range(const std::string & name, double a = std::nan("")) const {
                    return emu.range(name, a);
                }
                const GPEmulator & emulator() const { return emu; }

              private:
                std::string model_name;
                GPEmulator emu;
                Sigma8Emulator s8;
            };

        } // namespace EMANTIS
    } // namespace EMULATOR
} // namespace FML

#endif
