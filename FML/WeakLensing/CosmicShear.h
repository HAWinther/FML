#ifndef FML_WEAK_LENSING_COSMIC_SHEAR_HEADER
#define FML_WEAK_LENSING_COSMIC_SHEAR_HEADER
// Tomographic cosmic shear for survey likelihoods (KiDS, DES, ...), built on the projection primitives of
// WeakLensing.h (same units: comoving Mpc/h, k in h/Mpc, P in (Mpc/h)^3, angles in radians):
//
//   * Limber spectra of shear and NLA intrinsic alignments,
//       C_ell^ij = S(ell) int dchi / chi^2 [ q_i q_j + (q_i p_j + p_i q_j) F + p_i p_j F^2 ] P((ell+1/2)/chi, z)
//     (GG, GI, II), q_i the lensing kernel of WeakLensing.h (times an optional Sigma(z) for modified
//     gravity), p_i = n_i(z) dz/dchi the normalised source distribution, F(z) the NLA amplitude
//       F(z) = -A_IA C1 rho_crit Omega_m / D(z) [(1+z)/(1+z_pivot)]^eta_IA  (C1 rho_crit = 0.013877),
//     S(ell) = (ell-1) ell (ell+1) (ell+2)/(ell+1/2)^4 if spin_factor (else 1);
//   * photometric-redshift shifts n_i(z) -> n_i(z - delta_z_i) and multiplicative shear bias
//     C_ell^ij -> (1 + m_i)(1 + m_j) C_ell^ij;
//   * the flat-sky correlation functions xi_+-(theta) = int dell ell/(2 pi) C_ell J_{0,4}(ell theta), at points or
//     averaged over angular bins;
//   * logarithmic COSEBIs (Schneider, Eifler & Krause 2010, arXiv:1002.2136; as KiDS-1000, Asgari et al. 2021),
//       E_n = 1/2 int dtheta theta [T_+n xi_+ + T_-n xi_-],  B_n = 1/2 int dtheta theta [T_+n xi_+ - T_-n xi_-],
//     = int dell ell/(2 pi) C^E_ell W_n(ell) with W_n(ell) = int dtheta theta T_+n(theta) J_0(ell theta).
//
// The matter power spectrum is the caller's (nonlinear, total matter, including any baryonic feedback).
// Standalone C++17 like WeakLensing.h; the Bessel functions are POSIX j0/jn (glibc, macOS libm).
#include <FML/WeakLensing/WeakLensing.h>

#include <math.h>

#include <algorithm>
#include <cmath>
#include <complex>
#include <string>
#include <vector>

namespace FML::WEAKLENSING {

    namespace detail {
        /// Gauss-Legendre nodes and weights on [-1, 1]
        inline void gauss_legendre(int order, std::vector<double> & x, std::vector<double> & w) {
            require(order >= 1 && order <= 512, "CosmicShear: Gauss-Legendre order must be 1..512");
            x.assign(order, 0.0);
            w.assign(order, 0.0);
            for (int i = 0; i < (order + 1) / 2; ++i) {
                double r = std::cos(M_PI * (i + .75) / (order + .5)), dp = 0;
                for (int it = 0; it < 100; ++it) {
                    double p0 = 1, p1 = 0;
                    for (int j = 1; j <= order; ++j) {
                        const double p2 = p1;
                        p1 = p0;
                        p0 = ((2 * j - 1) * r * p1 - (j - 1) * p2) / j;
                    }
                    dp = order * (r * p0 - p1) / (r * r - 1);
                    const double dr = p0 / dp;
                    r -= dr;
                    if (std::abs(dr) < 4e-16)
                        break;
                }
                double p0 = 1, p1 = 0;
                for (int j = 1; j <= order; ++j) {
                    const double p2 = p1;
                    p1 = p0;
                    p0 = ((2 * j - 1) * r * p1 - (j - 1) * p2) / j;
                }
                dp = order * (r * p0 - p1) / (r * r - 1);
                x[i] = -r;
                x[order - 1 - i] = r;
                w[i] = w[order - 1 - i] = 2 / ((1 - r * r) * dp * dp);
            }
        }
        /// Composite Gauss-Legendre rule on npanel equal panels of [a, b] (appended to x, w)
        inline void panel_rule(double a, double b, int npanel, int order, std::vector<double> & x,
                               std::vector<double> & w) {
            std::vector<double> gx, gw;
            gauss_legendre(order, gx, gw);
            for (int p = 0; p < npanel; p++) {
                const double lo = a + (b - a) * p / npanel, hi = a + (b - a) * (p + 1) / npanel;
                for (int i = 0; i < order; i++) {
                    x.push_back(0.5 * (lo + hi) + 0.5 * (hi - lo) * gx[i]);
                    w.push_back(0.5 * (hi - lo) * gw[i]);
                }
            }
        }
    } // namespace detail

    // =============================================================================
    // Source redshift distributions and systematics
    // =============================================================================

    /// Tabulated dN/dz (any normalisation), linear between the nodes and zero outside.
    struct RedshiftDistribution {
        std::vector<double> z, n;
    };

    /// n(z) -> n(z - delta_z): a positive shift moves the sources to higher redshift (CosmoSIS photoz_bias,
    /// additive mode, as KiDS and DES). The part below z = 0 is cut off. The result is refined to steps of at
    /// most dz_max by linear interpolation of the input (the kernels treat the nodes as quadrature points).
    inline RedshiftDistribution shifted_distribution(const RedshiftDistribution & in, double delta_z, double dz_max) {
        detail::require(in.z.size() >= 2 && in.z.size() == in.n.size(), "CosmicShear: invalid n(z) table");
        detail::require(std::isfinite(delta_z) && std::isfinite(dz_max) && dz_max > 0, "CosmicShear: invalid shift");
        for (std::size_t i = 0; i < in.z.size(); i++)
            detail::require(std::isfinite(in.z[i]) && std::isfinite(in.n[i]) && in.n[i] >= 0 &&
                                (i == 0 || in.z[i] > in.z[i - 1]),
                            "CosmicShear: n(z) must have increasing z and n >= 0");
        RedshiftDistribution out;
        for (std::size_t i = 0; i + 1 < in.z.size(); i++) {
            double z0 = in.z[i] + delta_z, z1 = in.z[i + 1] + delta_z, n0 = in.n[i], n1 = in.n[i + 1];
            if (z1 <= 1e-8) // (also a sliver above z = 0: its nodes would coincide in distance)
                continue;
            if (z0 < 0) { // cut at z = 0
                n0 = n0 + (n1 - n0) * (-z0) / (z1 - z0);
                z0 = 0;
            }
            const int m = std::max(1, int(std::ceil((z1 - z0) / dz_max - 1e-9)));
            for (int j = (out.z.empty() ? 0 : 1); j <= m; j++) {
                const double t = double(j) / m;
                out.z.push_back(z0 + t * (z1 - z0));
                out.n.push_back(n0 + t * (n1 - n0));
            }
        }
        detail::require(out.z.size() >= 2, "CosmicShear: n(z) shifted below z = 0");
        return out;
    }

    /// Nonlinear alignment model (Hirata & Seljak 2004, Bridle & King 2007) with C1 = 5e-14 h^-2 M_sun^-1 Mpc^3:
    /// C1 rho_crit = 0.013877 (CosmoSIS' la_model as used by KiDS-1000 and DES, and CCL; some papers round it to
    /// 0.0134). D(z) is the linear growth normalised to 1 today.
    struct NLA {
        double A = 0.0, eta = 0.0, z_pivot = 0.62, C1_rho_crit = 5e-14 * 2.77536627e11;
        double amplitude(double z, double omega_m, double growth) const {
            if (A == 0.0)
                return 0.0;
            return -A * C1_rho_crit * omega_m / growth * std::pow((1 + z) / (1 + z_pivot), eta);
        }
    };

    struct ShearModel {
        FlatGeometry geometry;     // Omega_m (total matter today), c/H0 [Mpc/h], z(chi)
        Function chi_of_z;         // comoving distance [Mpc/h]
        PowerSpectrum power;       // nonlinear total-matter P(k, z); taken as 0 outside [k_min, k_max]
        Function growth;           // D(z)/D(0), for the NLA amplitude (needed when A_IA != 0)
        Function lensing_strength; // Sigma(z) multiplying the lensing kernel at the lens; empty: 1 (GR)
    };

    struct ShearSystematics {
        std::vector<double> delta_z; // per bin; empty: 0
        std::vector<double> m;       // per bin; empty: 0
        NLA ia;
    };

    struct ShearSettings {
        double dz_max = 0.002;  // refinement of n(z)
        double dchi_max = 20.0; // radial panel width [Mpc/h] ...
        double dlnchi = 0.1;    // ... and at most this fraction of chi (high k at small chi)
        int order = 6;          // Gauss-Legendre points per panel
        bool spin_factor = false; // (ell+2)!/(ell-2)!/(ell+1/2)^4 (CCL: yes; CosmoSIS/KiDS/DES: no)
    };

    /// C_ell for the pairs i <= j (index pair(i, j)), at the given ell (>= 2). GI is the sum of both orderings,
    /// total = (1 + m_i)(1 + m_j)(GG + GI + II).
    struct ShearSpectra {
        std::vector<double> ell;
        std::size_t nbins = 0;
        std::vector<std::vector<double>> GG, GI, II, total; // [pair][ell]
        std::size_t pair(std::size_t i, std::size_t j) const {
            if (i > j)
                std::swap(i, j);
            detail::require(j < nbins, "CosmicShear: bin index out of range");
            return i * nbins - i * (i + 1) / 2 + j;
        }
        std::size_t npairs() const { return nbins * (nbins + 1) / 2; }
    };

    inline ShearSpectra shear_spectra(const std::vector<RedshiftDistribution> & nz,
                                      const ShearModel & model,
                                      const ShearSystematics & sys,
                                      const std::vector<double> & ell,
                                      const ShearSettings & settings = {}) {
        const std::size_t nb = nz.size();
        detail::require(nb > 0, "CosmicShear: no source bins");
        detail::require(sys.delta_z.empty() || sys.delta_z.size() == nb, "CosmicShear: delta_z per bin");
        detail::require(sys.m.empty() || sys.m.size() == nb, "CosmicShear: m per bin");
        detail::require(bool(model.chi_of_z), "CosmicShear: missing chi(z)");
        detail::require(settings.dchi_max > 0 && settings.dlnchi > 0 && settings.order >= 2 && settings.order <= 64,
                        "CosmicShear: invalid settings");
        for (std::size_t i = 0; i < ell.size(); i++)
            detail::require(std::isfinite(ell[i]) && ell[i] >= 2 && (i == 0 || ell[i] > ell[i - 1]),
                            "CosmicShear: ell must be increasing and >= 2");
        model.geometry.validate();
        model.power.validate();
        const bool ia = sys.ia.A != 0.0;
        detail::require(!ia || bool(model.growth), "CosmicShear: NLA needs the growth factor");

        // Source bins: lensing efficiency (Sources) and the normalised p(chi) = n(z) dz/dchi
        std::vector<Sources> sources;
        std::vector<RadialSelection> windows;
        double chi_max = 0.0;
        for (std::size_t b = 0; b < nb; b++) {
            const auto d = shifted_distribution(nz[b], sys.delta_z.empty() ? 0.0 : sys.delta_z[b], settings.dz_max);
            const auto weights = detail::masses(d.z, d.n);
            std::vector<double> chi(d.z.size()), n_chi(d.z.size());
            for (std::size_t i = 0; i < d.z.size(); i++) {
                // (an interpolated chi(z) may be ~ -1e-13 at z = 0)
                chi[i] = d.z[i] == 0.0 ? 0.0 : std::max(0.0, detail::finite(model.chi_of_z(d.z[i])));
                const double h = 1e-4 * std::max(1.0, d.z[i]);
                const double zl = std::max(0.0, d.z[i] - h), zr = d.z[i] + h;
                const double dchidz = (model.chi_of_z(zr) - model.chi_of_z(zl)) / (zr - zl);
                detail::require(dchidz > 0, "CosmicShear: chi(z) must increase");
                n_chi[i] = d.n[i] / dchidz;
            }
            detail::grid(chi);
            // A source at chi = 0 does not lens; move it off the observer (Sources rejects it)
            auto chi_src = chi;
            if (chi_src[0] == 0.0)
                chi_src[0] = 1e-10 * chi_src[1];
            sources.emplace_back(chi_src, weights);
            windows.emplace_back(chi, n_chi);
            chi_max = std::max(chi_max, chi.back());
        }
        detail::require(model.geometry.redshift(chi_max) <= model.power.z_max,
                        "CosmicShear: P(k, z) does not cover the source redshifts");

        // Radial nodes: Gauss-Legendre on panels of width min(dchi_max, dlnchi chi), the first from 0
        std::vector<double> edges{0.0, std::min(chi_max, 1e-3 * settings.dchi_max)};
        while (edges.back() < chi_max)
            edges.push_back(
                std::min(chi_max, edges.back() + std::min(settings.dchi_max, settings.dlnchi * edges.back())));
        std::vector<double> chi_n, w_n;
        for (std::size_t e = 0; e + 1 < edges.size(); e++)
            detail::panel_rule(edges[e], edges[e + 1], 1, settings.order, chi_n, w_n);

        const std::size_t N = chi_n.size();
        std::vector<double> z_n(N), F_n(N, 0.0);
        std::vector<std::vector<double>> q(nb, std::vector<double>(N)), p(nb, std::vector<double>(N));
        for (std::size_t k = 0; k < N; k++) {
            z_n[k] = model.geometry.redshift(chi_n[k]);
            const double sigma = model.lensing_strength ? detail::finite(model.lensing_strength(z_n[k])) : 1.0;
            if (ia)
                F_n[k] = sys.ia.amplitude(z_n[k], model.geometry.omega_m, detail::finite(model.growth(z_n[k])));
            for (std::size_t b = 0; b < nb; b++) {
                q[b][k] = model.geometry.kernel(sources[b], chi_n[k]) * sigma;
                p[b][k] = windows[b](chi_n[k]);
            }
        }

        ShearSpectra out;
        out.ell = ell;
        out.nbins = nb;
        const std::size_t np = out.npairs();
        out.GG.assign(np, std::vector<double>(ell.size(), 0.0));
        out.GI = out.GG;
        out.II = out.GG;
        out.total = out.GG;
        for (std::size_t l = 0; l < ell.size(); l++) {
            const double L = ell[l];
            std::vector<long double> gg(np, 0), gi(np, 0), ii(np, 0);
            for (std::size_t k = 0; k < N; k++) {
                const double kk = (L + 0.5) / chi_n[k];
                if (kk < model.power.k_min || kk > model.power.k_max)
                    continue;
                const double P = model.power(kk, z_n[k]) * w_n[k] / (chi_n[k] * chi_n[k]);
                for (std::size_t i = 0; i < nb; i++)
                    for (std::size_t j = i; j < nb; j++) {
                        const std::size_t ij = out.pair(i, j);
                        gg[ij] += q[i][k] * q[j][k] * P;
                        if (ia) {
                            gi[ij] += (q[i][k] * p[j][k] + p[i][k] * q[j][k]) * F_n[k] * P;
                            ii[ij] += p[i][k] * p[j][k] * F_n[k] * F_n[k] * P;
                        }
                    }
            }
            const double S = settings.spin_factor ? (L - 1) * L * (L + 1) * (L + 2) / std::pow(L + 0.5, 4) : 1.0;
            for (std::size_t i = 0; i < nb; i++)
                for (std::size_t j = i; j < nb; j++) {
                    const std::size_t ij = out.pair(i, j);
                    const double mi = sys.m.empty() ? 0.0 : sys.m[i], mj = sys.m.empty() ? 0.0 : sys.m[j];
                    out.GG[ij][l] = S * double(gg[ij]);
                    out.GI[ij][l] = S * double(gi[ij]);
                    out.II[ij][l] = S * double(ii[ij]);
                    out.total[ij][l] = (1 + mi) * (1 + mj) * (out.GG[ij][l] + out.GI[ij][l] + out.II[ij][l]);
                }
        }
        return out;
    }

    // =============================================================================
    // Correlation functions
    // =============================================================================

    namespace detail {
        inline double bessel_J(int nu, double x) { return nu == 0 ? ::j0(x) : ::jn(nu, x); }

        /// Natural cubic spline y(x), x increasing
        class CubicSpline {
            std::vector<double> x_, y_, y2_;

          public:
            CubicSpline() = default;
            CubicSpline(std::vector<double> x, std::vector<double> y) : x_(std::move(x)), y_(std::move(y)) {
                const std::size_t n = x_.size();
                require(n >= 2 && y_.size() == n, "CosmicShear: spline needs two points");
                y2_.assign(n, 0.0);
                std::vector<double> u(n, 0.0);
                for (std::size_t i = 1; i + 1 < n; i++) {
                    const double sig = (x_[i] - x_[i - 1]) / (x_[i + 1] - x_[i - 1]);
                    const double p = sig * y2_[i - 1] + 2.0;
                    y2_[i] = (sig - 1.0) / p;
                    u[i] = (y_[i + 1] - y_[i]) / (x_[i + 1] - x_[i]) - (y_[i] - y_[i - 1]) / (x_[i] - x_[i - 1]);
                    u[i] = (6.0 * u[i] / (x_[i + 1] - x_[i - 1]) - sig * u[i - 1]) / p;
                }
                for (std::size_t k = n - 1; k-- > 0;)
                    y2_[k] = y2_[k] * y2_[k + 1] + u[k];
            }
            double operator()(double x) const {
                std::size_t hi = std::size_t(std::upper_bound(x_.begin(), x_.end(), x) - x_.begin());
                hi = std::clamp<std::size_t>(hi, 1, x_.size() - 1);
                const std::size_t lo = hi - 1;
                const double h = x_[hi] - x_[lo], a = (x_[hi] - x) / h, b = (x - x_[lo]) / h;
                return a * y_[lo] + b * y_[hi] + ((a * a * a - a) * y2_[lo] + (b * b * b - b) * y2_[hi]) * h * h / 6.0;
            }
        };

        /// C_ell interpolated in ln ell (of ln C_ell when C_ell > 0 everywhere), zero outside the table
        class EllSpline {
            CubicSpline s_;
            bool log_ = true;
            double lo_ = 0, hi_ = 0;

          public:
            EllSpline(const std::vector<double> & ell, const std::vector<double> & cl) {
                require(ell.size() >= 2 && ell.size() == cl.size(), "CosmicShear: C_ell table");
                std::vector<double> x, y;
                for (double c : cl)
                    log_ = log_ && c > 0;
                for (std::size_t i = 0; i < ell.size(); i++) {
                    x.push_back(std::log(ell[i]));
                    y.push_back(log_ ? std::log(cl[i]) : cl[i]);
                }
                s_ = CubicSpline(x, y);
                lo_ = ell.front();
                hi_ = ell.back();
            }
            double operator()(double ell) const {
                if (ell < lo_ || ell > hi_)
                    return 0.0;
                const double v = s_(std::log(ell));
                return log_ ? std::exp(v) : v;
            }
        };

        /// Nodes and weights for int_{ell_min}^{ell_max} dell f(ell) J(ell theta): panels of width dlnell in
        /// ln ell up to ell theta = pi, then panels of half a period (pi/theta) of the Bessel function.
        inline void hankel_nodes(double theta, double ell_min, double ell_max, int order, double dlnell,
                                 std::vector<double> & ell, std::vector<double> & w) {
            ell.clear();
            w.clear();
            const double ell1 = std::clamp(M_PI / theta, ell_min, ell_max);
            if (ell1 > ell_min) {
                const double L = std::log(ell1 / ell_min);
                const int np = std::max(1, int(std::ceil(L / dlnell)));
                std::vector<double> x, ww;
                panel_rule(std::log(ell_min), std::log(ell1), np, order, x, ww);
                for (std::size_t i = 0; i < x.size(); i++) {
                    ell.push_back(std::exp(x[i]));
                    w.push_back(ww[i] * ell.back());
                }
            }
            if (ell_max > ell1) {
                const double h = M_PI / theta;
                const int np = std::max(1, int(std::ceil((ell_max - ell1) / h - 1e-9)));
                panel_rule(ell1, ell_max, np, order, ell, w);
            }
        }
    } // namespace detail

    struct HankelSettings {
        int order = 8;        // Gauss-Legendre points per panel
        double dlnell = 0.05; // panel width in ln ell below ell theta = pi
    };

    /// int_{ell_min}^{ell_max} dell ell/(2 pi) C(ell) J_nu(ell theta) for a callable C(ell) (theta in radians)
    inline double hankel(const Function & C, double ell_min, double ell_max, double theta, int nu,
                         const HankelSettings & settings = {}) {
        detail::require(theta > 0 && ell_min > 0 && ell_max > ell_min, "CosmicShear: invalid Hankel transform");
        std::vector<double> ell, w;
        detail::hankel_nodes(theta, ell_min, ell_max, settings.order, settings.dlnell, ell, w);
        long double sum = 0;
        for (std::size_t i = 0; i < ell.size(); i++)
            sum += w[i] * ell[i] * C(ell[i]) * detail::bessel_J(nu, ell[i] * theta);
        return double(sum) / (2 * M_PI);
    }

    /// xi_+-(theta) [pair][theta] of ShearSpectra::total, theta in radians
    struct ShearCorrelation {
        std::vector<double> theta;
        std::vector<std::vector<double>> plus, minus;
    };

    /// xi_+(theta) = int dell ell/(2 pi) C_ell J_0(ell theta), xi_- with J_4 (flat sky), over the ell range of
    /// the spectra (C_ell interpolated in ln ell; choose ell up to ~1e5 for arcminute scales)
    inline ShearCorrelation shear_correlation(const ShearSpectra & cl, const std::vector<double> & theta,
                                              const HankelSettings & settings = {}, bool with_minus = true) {
        std::vector<detail::EllSpline> spl;
        for (const auto & c : cl.total)
            spl.emplace_back(cl.ell, c);
        ShearCorrelation out;
        out.theta = theta;
        out.plus.assign(spl.size(), std::vector<double>(theta.size(), 0.0));
        out.minus = out.plus;
        std::vector<double> ell, w;
        for (std::size_t t = 0; t < theta.size(); t++) {
            detail::require(theta[t] > 0, "CosmicShear: theta must be > 0");
            detail::hankel_nodes(theta[t], cl.ell.front(), cl.ell.back(), settings.order, settings.dlnell, ell, w);
            std::vector<double> j0(ell.size()), j4(ell.size());
            for (std::size_t i = 0; i < ell.size(); i++) {
                const double x = ell[i] * theta[t];
                j0[i] = w[i] * ell[i] / (2 * M_PI) * ::j0(x);
                j4[i] = with_minus ? w[i] * ell[i] / (2 * M_PI) * ::jn(4, x) : 0.0;
            }
            for (std::size_t p = 0; p < spl.size(); p++) {
                long double sp = 0, sm = 0;
                for (std::size_t i = 0; i < ell.size(); i++) {
                    const double c = spl[p](ell[i]);
                    sp += c * j0[i];
                    sm += c * j4[i];
                }
                out.plus[p][t] = double(sp);
                out.minus[p][t] = double(sm);
            }
        }
        return out;
    }

    /// xi_+- averaged over angular bins [edges[b], edges[b+1]] with weight theta (pair counts of a uniform
    /// survey), by Gauss-Legendre with n_sub points per bin; theta of the result = the weighted bin centres.
    inline ShearCorrelation shear_correlation_binned(const ShearSpectra & cl, const std::vector<double> & edges,
                                                     int n_sub = 8, const HankelSettings & settings = {}) {
        detail::grid(edges, false);
        std::vector<double> gx, gw, th, wt;
        detail::gauss_legendre(n_sub, gx, gw);
        for (std::size_t b = 0; b + 1 < edges.size(); b++)
            for (int i = 0; i < n_sub; i++) {
                const double lo = edges[b], hi = edges[b + 1];
                th.push_back(0.5 * (lo + hi) + 0.5 * (hi - lo) * gx[i]);
                wt.push_back(0.5 * (hi - lo) * gw[i] * th.back());
            }
        const auto fine = shear_correlation(cl, th, settings);
        ShearCorrelation out;
        const std::size_t nb = edges.size() - 1;
        out.plus.assign(fine.plus.size(), std::vector<double>(nb, 0.0));
        out.minus = out.plus;
        for (std::size_t b = 0; b < nb; b++) {
            double norm = 0, tc = 0;
            for (int i = 0; i < n_sub; i++) {
                norm += wt[b * n_sub + i];
                tc += wt[b * n_sub + i] * th[b * n_sub + i];
            }
            out.theta.push_back(tc / norm);
            for (std::size_t p = 0; p < fine.plus.size(); p++) {
                double sp = 0, sm = 0;
                for (int i = 0; i < n_sub; i++) {
                    sp += wt[b * n_sub + i] * fine.plus[p][b * n_sub + i];
                    sm += wt[b * n_sub + i] * fine.minus[p][b * n_sub + i];
                }
                out.plus[p][b] = sp / norm;
                out.minus[p][b] = sm / norm;
            }
        }
        return out;
    }

    // =============================================================================
    // Logarithmic COSEBIs (Schneider, Eifler & Krause 2010)
    // =============================================================================

    /// T_+n(theta) = t_n(z), z = ln(theta/theta_min), a polynomial of degree n + 1 in z with positive leading
    /// coefficient N_n, zero outside [theta_min, theta_max], with
    ///   int dtheta theta T_+n = int dtheta theta^3 T_+n = 0,   1/Delta theta int dtheta T_+m T_+n = delta_mn
    /// (SEK10 eqs. 4, 31). SEK10 solve for the monomial coefficients with ~40 digits; here T_+n is expanded in
    /// the polynomials orthonormal for the measure dtheta on [theta_min, theta_max] (three-term recurrence from
    /// the Stieltjes procedure), where the conditions are a well-conditioned Gram-Schmidt step in double.
    ///   T_-n(theta) = T_+n(theta) + int_{theta_min}^theta dt t T_+n(t) (4/theta^2 - 12 t^2/theta^4)  (SEK10 eq. 3)
    class LogCOSEBIs {
        double tmin_, tmax_, Z_;
        int nmax_;
        std::vector<double> alpha_, sqrtbeta_; // orthonormal recurrence
        std::vector<double> lead_;             // leading coefficient of q_j
        std::vector<std::vector<double>> t_;   // T_+n = sqrt(Delta theta) sum_j t_[n-1][j] q_j(z)
        std::vector<double> gz_, gw_;          // Gauss-Legendre in z on [0, Z] for the theta integrals
        std::vector<double> ell_, well_;       // ell quadrature of E_n = int dell ell/(2 pi) C_ell W_n(ell)
        std::vector<std::vector<double>> Wtab_; // [n-1][ell node] W_n at ell_ (times well_ ell/(2 pi))

        // q_0..q_{m}(z)
        void q_values(double z, int m, std::vector<double> & q) const {
            q.resize(m + 1);
            q_values(z, m, q.data());
        }
        void q_values(double z, int m, double * q) const {
            q[0] = 1.0 / sqrtbeta_[0];
            if (m >= 1)
                q[1] = (z - alpha_[0]) * q[0] / sqrtbeta_[1];
            for (int j = 1; j < m; j++)
                q[j + 1] = ((z - alpha_[j]) * q[j] - sqrtbeta_[j] * q[j - 1]) / sqrtbeta_[j + 1];
        }

      public:
        /// theta in radians. E_n integrates C_ell over [ell_min, ell_max] against W_n(ell), tabulated here (a few
        /// seconds for ell_max = 1e5; ell_max <= 0: no table, E() unavailable).
        LogCOSEBIs(double theta_min, double theta_max, int n_max, double ell_min = 1.0, double ell_max = 1e5,
                   int quadrature_points = 256)
            : tmin_(theta_min), tmax_(theta_max), nmax_(n_max) {
            detail::require(theta_min > 0 && theta_max > theta_min && n_max >= 1 && n_max <= 30,
                            "CosmicShear: invalid COSEBI range");
            Z_ = std::log(tmax_ / tmin_);
            const int D = nmax_ + 2; // degrees 0..n_max+1
            // Discretised measure dtheta = theta_min e^z dz
            std::vector<double> z, mu;
            detail::panel_rule(0.0, Z_, 8, std::max(8, quadrature_points / 8), z, mu);
            for (std::size_t k = 0; k < z.size(); k++)
                mu[k] *= tmin_ * std::exp(z[k]);
            const std::size_t M = z.size();
            // Stieltjes / Lanczos with full reorthogonalisation
            std::vector<std::vector<double>> Q;
            double b0 = 0;
            for (double m : mu)
                b0 += m;
            sqrtbeta_.push_back(std::sqrt(b0));
            Q.emplace_back(M, 1.0 / sqrtbeta_[0]);
            lead_.push_back(1.0 / sqrtbeta_[0]);
            for (int j = 0; j + 1 < D; j++) {
                double a = 0;
                for (std::size_t k = 0; k < M; k++)
                    a += mu[k] * z[k] * Q[j][k] * Q[j][k];
                alpha_.push_back(a);
                std::vector<double> r(M);
                for (std::size_t k = 0; k < M; k++)
                    r[k] = (z[k] - a) * Q[j][k] - (j > 0 ? sqrtbeta_[j] * Q[j - 1][k] : 0.0);
                for (int pass = 0; pass < 2; pass++)
                    for (int i = 0; i <= j; i++) {
                        double c = 0;
                        for (std::size_t k = 0; k < M; k++)
                            c += mu[k] * r[k] * Q[i][k];
                        for (std::size_t k = 0; k < M; k++)
                            r[k] -= c * Q[i][k];
                    }
                double nr = 0;
                for (std::size_t k = 0; k < M; k++)
                    nr += mu[k] * r[k] * r[k];
                sqrtbeta_.push_back(std::sqrt(nr));
                for (auto & v : r)
                    v /= sqrtbeta_.back();
                Q.push_back(r);
                lead_.push_back(lead_.back() / sqrtbeta_.back());
            }
            alpha_.push_back(0.0);
            // Constraint vectors <theta, q_j>, <theta^3, q_j>
            std::vector<double> A(D, 0.0), B(D, 0.0);
            for (int j = 0; j < D; j++)
                for (std::size_t k = 0; k < M; k++) {
                    const double th = tmin_ * std::exp(z[k]);
                    A[j] += mu[k] * th * Q[j][k];
                    B[j] += mu[k] * th * th * th * Q[j][k];
                }
            for (int n = 1; n <= nmax_; n++) {
                const int d = n + 2;
                std::vector<std::vector<double>> U;
                auto add = [&](std::vector<double> v) {
                    v.resize(d, 0.0);
                    for (int pass = 0; pass < 2; pass++)
                        for (const auto & u : U) {
                            double c = 0;
                            for (int i = 0; i < d; i++)
                                c += u[i] * v[i];
                            for (int i = 0; i < d; i++)
                                v[i] -= c * u[i];
                        }
                    double nv = 0;
                    for (double x : v)
                        nv += x * x;
                    return std::make_pair(v, std::sqrt(nv));
                };
                for (const auto & c : {std::vector<double>(A.begin(), A.begin() + d),
                                       std::vector<double>(B.begin(), B.begin() + d)}) {
                    auto [v, nv] = add(c);
                    for (auto & x : v)
                        x /= nv;
                    U.push_back(v);
                }
                for (const auto & tm : t_) {
                    auto [v, nv] = add(tm);
                    for (auto & x : v)
                        x /= nv;
                    U.push_back(v);
                }
                std::vector<double> e(d, 0.0);
                e[d - 1] = 1.0;
                auto [v, nv] = add(e);
                for (auto & x : v)
                    x /= (v[d - 1] > 0 ? nv : -nv);
                t_.push_back(v);
            }
            detail::panel_rule(0.0, Z_, 32, 16, gz_, gw_);
            if (ell_max > 0)
                tabulate_W(ell_min, ell_max);
        }

        /// Quadrature nodes in ell (panels of width 0.05 in ln ell up to pi/theta_max, then half periods
        /// pi/theta_max of the oscillation of W_n) and W_n(ell) on them
        void tabulate_W(double ell_min, double ell_max) {
            detail::require(ell_min > 0 && ell_max > ell_min, "CosmicShear: invalid COSEBI ell range");
            detail::hankel_nodes(tmax_, ell_min, ell_max, 8, 0.05, ell_, well_);
            Wtab_.assign(nmax_, std::vector<double>(ell_.size(), 0.0));
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 64)
#endif
            for (std::size_t k = 0; k < ell_.size(); k++) {
                std::vector<double> W;
                W_all(ell_[k], W);
                for (int n = 0; n < nmax_; n++)
                    Wtab_[n][k] = W[n] * well_[k] * ell_[k] / (2 * M_PI);
            }
        }

        /// T_+n(theta) for n = 1..n_max at z = ln(theta/theta_min) (T: n_max values)
        void T_plus_all(double z, double * T) const {
            double q[34];
            q_values(z, nmax_ + 1, q);
            const double norm = std::sqrt(tmax_ - tmin_);
            for (int n = 1; n <= nmax_; n++) {
                const double * tn = t_[n - 1].data();
                double s = 0;
                for (int j = 0; j <= n + 1; j++)
                    s += tn[j] * q[j];
                T[n - 1] = norm * s;
            }
        }

        /// W_n(ell) for n = 1..n_max. Below ell theta = 30 Gauss-Legendre on panels of width min(0.05 theta,
        /// pi/ell); above, J_0(x) = Re[sqrt(2/(pi x)) (P + i Q) e^{i(x - pi/4)}] with the Hankel asymptotic series
        /// for P, Q (error < 1e-10 at x >= 30) and Filon's method on panels of width 0.005 theta: the smooth
        /// amplitude theta T_+n sqrt(2/(pi x)) (P + i Q) is interpolated quadratically and the product with the
        /// oscillation integrated exactly, so the cost does not grow with ell.
        void W_all(double ell, std::vector<double> & W) const {
            W.assign(nmax_, 0.0);
            static const auto gl8 = [] {
                std::pair<std::vector<double>, std::vector<double>> r;
                detail::gauss_legendre(8, r.first, r.second);
                return r;
            }();
            const auto & gx = gl8.first;
            const auto & gw = gl8.second;
            double T[32];
            const double xa = 30.0, tb = std::clamp(xa / ell, tmin_, tmax_);
            // (a) direct
            for (double lo = tmin_; lo < tb;) {
                const double hi = std::min(tb, lo + std::min(0.05 * lo, M_PI / ell));
                for (int i = 0; i < 8; i++) {
                    const double t = 0.5 * (lo + hi) + 0.5 * (hi - lo) * gx[i];
                    const double f = 0.5 * (hi - lo) * gw[i] * t * ::j0(ell * t);
                    T_plus_all(std::log(t / tmin_), T);
                    for (int n = 0; n < nmax_; n++)
                        W[n] += f * T[n];
                }
                lo = hi;
            }
            if (tb >= tmax_)
                return;
            // (b) Filon with the asymptotic Bessel function
            using cplx = std::complex<double>;
            auto amplitude = [&](double t, cplx * g) {
                const double x = ell * t, ix2 = 1.0 / (x * x);
                const double P = 1.0 + ix2 * (-9.0 / 128 + ix2 * (3675.0 / 32768 + ix2 * (-2401245.0 / 4194304)));
                const double Q =
                    (1.0 / x) * (-1.0 / 8 + ix2 * (75.0 / 1024 + ix2 * (-59535.0 / 262144 + ix2 * (57972915.0 / 33554432))));
                const cplx a = t * std::sqrt(2.0 / (M_PI * x)) * cplx(P, Q);
                T_plus_all(std::log(t / tmin_), T);
                for (int n = 0; n < nmax_; n++)
                    g[n] = a * T[n];
            };
            const int np = std::max(1, int(std::ceil(std::log(tmax_ / tb) / 0.005)));
            const double r = std::pow(tmax_ / tb, 1.0 / np);
            cplx gl[32], gm[32], gr[32];
            double lo = tb;
            amplitude(lo, gl);
            for (int p = 0; p < np; p++) {
                const double hi = (p + 1 == np) ? tmax_ : lo * r, m = 0.5 * (lo + hi), h = 0.5 * (hi - lo);
                amplitude(m, gm);
                amplitude(hi, gr);
                const double a = ell * h, sa = std::sin(a), ca = std::cos(a);
                double I0, I1, I2; // int_{-h}^{h} u^k e^{i ell u} du = I0, i I1, I2
                if (a < 0.1) {
                    const double a2 = a * a;
                    I0 = 2 * h * (1 - a2 / 6 + a2 * a2 / 120);
                    I1 = 2 * h * h * a * (1.0 / 3 - a2 / 30 + a2 * a2 / 840);
                    I2 = 2 * h * h * h * (1.0 / 3 - a2 / 10 + a2 * a2 / 168);
                } else {
                    I0 = 2 * h * sa / a;
                    I1 = 2 * h * h * (sa - a * ca) / (a * a);
                    I2 = 2 * h * h * h * (a * a * sa + 2 * a * ca - 2 * sa) / (a * a * a);
                }
                const cplx phase = std::exp(cplx(0.0, ell * m - M_PI / 4));
                for (int n = 0; n < nmax_; n++) {
                    const cplx g1 = (gr[n] - gl[n]) / (2 * h), g2 = (gr[n] - 2.0 * gm[n] + gl[n]) / (2 * h * h);
                    W[n] += std::real(phase * (gm[n] * I0 + g1 * cplx(0.0, I1) + g2 * I2));
                }
                std::copy(gr, gr + nmax_, gl);
                lo = hi;
            }
        }

        int n_max() const { return nmax_; }
        double theta_min() const { return tmin_; }
        double theta_max() const { return tmax_; }

        /// t_n(z) for any z (the polynomial; T_+n is zero outside [0, z_max])
        double t_plus_of_z(int n, double z) const {
            detail::require(n >= 1 && n <= nmax_, "CosmicShear: COSEBI n out of range");
            std::vector<double> q;
            q_values(z, n + 1, q);
            double s = 0;
            for (int j = 0; j <= n + 1; j++)
                s += t_[n - 1][j] * q[j];
            return std::sqrt(tmax_ - tmin_) * s;
        }
        double T_plus(int n, double theta) const {
            if (theta < tmin_ || theta > tmax_)
                return 0.0;
            return t_plus_of_z(n, std::log(theta / tmin_));
        }
        double T_minus(int n, double theta) const {
            if (theta < tmin_)
                return 0.0;
            const double zu = std::log(std::min(theta, tmax_) / tmin_);
            std::vector<double> x, w;
            detail::panel_rule(0.0, zu, 4, 16, x, w);
            long double a = 0, b = 0; // int dt t T_+, int dt t^3 T_+
            for (std::size_t k = 0; k < x.size(); k++) {
                const double t = tmin_ * std::exp(x[k]), f = w[k] * t * t * t_plus_of_z(n, x[k]);
                a += f;
                b += f * t * t;
            }
            return T_plus(n, theta) + 4.0 * double(a) / (theta * theta) - 12.0 * double(b) / std::pow(theta, 4);
        }
        /// N_n (leading coefficient of t_n(z)) and the n + 1 roots r_ni of t_n(z) = N_n prod_i (z - r_ni)
        double normalization(int n) const { return std::sqrt(tmax_ - tmin_) * t_[n - 1][n + 1] * lead_[n + 1]; }
        std::vector<double> roots(int n) const {
            std::vector<double> r;
            const double lo = -2.0, hi = Z_ + 2.0;
            const int ns = 4000;
            double z0 = lo, f0 = t_plus_of_z(n, z0);
            for (int i = 1; i <= ns; i++) {
                double z1 = lo + (hi - lo) * i / ns, f1 = t_plus_of_z(n, z1);
                if (f0 == 0.0)
                    r.push_back(z0);
                else if (f0 * f1 < 0) {
                    double a = z0, b = z1, fa = f0;
                    for (int it = 0; it < 200; it++) {
                        const double c = 0.5 * (a + b), fc = t_plus_of_z(n, c);
                        if ((fc < 0) == (fa < 0)) {
                            a = c;
                            fa = fc;
                        } else
                            b = c;
                        if (b - a < 1e-15 * std::max(1.0, std::abs(a)))
                            break;
                    }
                    r.push_back(0.5 * (a + b));
                }
                z0 = z1;
                f0 = f1;
            }
            return r;
        }

        /// W_n(ell) = int dtheta theta T_+n(theta) J_0(ell theta), by brute-force Gauss-Legendre on panels of width
        /// min(0.02 theta, pi/(2 ell)) (a check of the tabulated W_all)
        double W(int n, double ell) const {
            detail::require(n >= 1 && n <= nmax_, "CosmicShear: COSEBI n out of range");
            std::vector<double> gx, gw;
            detail::gauss_legendre(10, gx, gw);
            long double s = 0;
            for (double lo = tmin_; lo < tmax_;) {
                const double hi = std::min(tmax_, lo + std::min(0.02 * lo, 0.5 * M_PI / ell));
                for (int i = 0; i < 10; i++) {
                    const double t = 0.5 * (lo + hi) + 0.5 * (hi - lo) * gx[i];
                    s += 0.5 * (hi - lo) * gw[i] * t * T_plus(n, t) * ::j0(ell * t);
                }
                lo = hi;
            }
            return double(s);
        }

        /// E_n, B_n from xi_+-(theta): 1/2 int dtheta theta [T_+n xi_+ +- T_-n xi_-]
        void EB(const Function & xi_plus, const Function & xi_minus, std::vector<double> & E,
                std::vector<double> & B) const {
            E.assign(nmax_, 0.0);
            B.assign(nmax_, 0.0);
            for (std::size_t k = 0; k < gz_.size(); k++) {
                const double t = tmin_ * std::exp(gz_[k]), w = 0.5 * gw_[k] * t * t; // dtheta theta = t^2 dz
                const double xp = xi_plus(t), xm = xi_minus(t);
                for (int n = 1; n <= nmax_; n++) {
                    const double tp = t_plus_of_z(n, gz_[k]), tm = T_minus(n, t);
                    E[n - 1] += w * (tp * xp + tm * xm);
                    B[n - 1] += w * (tp * xp - tm * xm);
                }
            }
        }

        /// E_n of each pair [pair][n-1] from the spectra: E_n = int dell ell/(2 pi) C_ell W_n(ell) (C_ell
        /// interpolated as in shear_correlation, zero outside its ell range)
        std::vector<std::vector<double>> E(const ShearSpectra & cl) const {
            detail::require(!Wtab_.empty(), "CosmicShear: COSEBIs constructed without the W_n table");
            std::vector<std::vector<double>> out(cl.total.size(), std::vector<double>(nmax_, 0.0));
            for (std::size_t p = 0; p < cl.total.size(); p++) {
                detail::EllSpline spl(cl.ell, cl.total[p]);
                std::vector<long double> e(nmax_, 0);
                for (std::size_t k = 0; k < ell_.size(); k++) {
                    const double c = spl(ell_[k]);
                    if (c == 0.0)
                        continue;
                    for (int n = 0; n < nmax_; n++)
                        e[n] += c * Wtab_[n][k];
                }
                for (int n = 0; n < nmax_; n++)
                    out[p][n] = double(e[n]);
            }
            return out;
        }
    };

} // namespace FML::WEAKLENSING
#endif
