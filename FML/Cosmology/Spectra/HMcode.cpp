#include <FML/Cosmology/Spectra/HMcode.h>
#include <gsl/gsl_sf_expint.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>
#include <utility>

namespace FML::COSMOLOGY {
    using namespace Constants;
    namespace {
        double top_hat(double y) {
            if (std::abs(y) < 0.05) {
                const double y2 = y * y;
                return 1.0 - y2 / 10.0 + y2 * y2 / 280.0 - y2 * y2 * y2 / 15120.0;
            }
            return 3.0 * (std::sin(y) - y * std::cos(y)) / (y * y * y);
        }

        // Radiation-free auxiliary growth used by the Mead (2017) collapse fits and Bullock concentrations.
        // Massive neutrinos count as matter here, as in HMcode; scale-dependent growth of the actual spectra
        // is supplied by P_m(x,k) and P_cb(x,k), never inferred from this auxiliary growth function.
        struct Growth {
            std::vector<double> xs, g, G;
            double om, ok;
            const BackgroundCosmology & bg;
            bool lcdm;
            Growth(const BackgroundCosmology & b, bool flat_lcdm) :
                om(b.rhoB(0) + b.rhoCDM(0) + b.rhoNcdm(0)),
                ok(flat_lcdm ? 0.0 : b.OmegaK_of_x(0)), bg(b), lcdm(flat_lcdm) {
                xs = Utils::linspace(std::log(1e-4), 0.0, 1025);
                const double ai = std::exp(xs.front());
                std::array<double, 3> y{ai, ai, ai}; // g, dg/dln(a), int g dln(a)
                for (size_t i = 0; i < xs.size(); ++i) {
                    g.push_back(y[0]);
                    G.push_back(y[2]);
                    if (i + 1 == xs.size())
                        break;
                    const double x = xs[i], dx = xs[i + 1] - x;
                    auto rhs = [&](double xx, const std::array<double, 3> & v) {
                        const double m = om * std::exp(-3 * xx), curv = ok * std::exp(-2 * xx);
                        const double de = lcdm ? 1.0 - om : bg.rhoDE(xx);
                        const double dep = lcdm ? 0.0 : bg.drhoDEdx(xx);
                        const double e2 = m + curv + de;
                        const double hp = 0.5 * (-3 * m - 2 * curv + dep) / e2;
                        return std::array<double, 3>{v[1], -(2 + hp) * v[1] + 1.5 * m / e2 * v[0], v[0]};
                    };
                    auto add = [](auto a, const auto & b, double f) {
                        for (int j = 0; j < 3; ++j)
                            a[j] += f * b[j];
                        return a;
                    };
                    const auto k1 = rhs(x, y), k2 = rhs(x + dx / 2, add(y, k1, dx / 2));
                    const auto k3 = rhs(x + dx / 2, add(y, k2, dx / 2)), k4 = rhs(x + dx, add(y, k3, dx));
                    for (int j = 0; j < 3; ++j)
                        y[j] += dx / 6 * (k1[j] + 2 * k2[j] + 2 * k3[j] + k4[j]);
                }
            }
            double at(const std::vector<double> & v, double x) const {
                if (x <= xs.front())
                    return v.front() * std::exp(x - xs.front());
                const double t = (std::min(x, 0.0) - xs.front()) / (xs[1] - xs[0]);
                const size_t i = std::min(size_t(t), xs.size() - 2);
                return v[i] + (t - i) * (v[i + 1] - v[i]);
            }
            double formation_x(double target) const {
                if (target <= g.front())
                    return xs.front() + std::log(target / g.front());
                const auto it = std::lower_bound(g.begin(), g.end(), target);
                if (it == g.end())
                    return 0.0;
                const size_t i = size_t(it - g.begin()) - 1;
                return xs[i] + (target - g[i]) / (g[i + 1] - g[i]) * (xs[i + 1] - xs[i]);
            }
            double omega_m(double x) const {
                const double m = om * std::exp(-3 * x);
                return m / (m + ok * std::exp(-2 * x) + (lcdm ? 1 - om : bg.rhoDE(x)));
            }
        };

        void check_background(const BackgroundCosmology & bg) {
            // Restrict the calibrated implementation to standard GR matter + smooth DE backgrounds.
            // Other linear spectra cannot specify nonlinear screening or halo physics by themselves.
            const auto & model = bg.model_name();
            if ((model != "lcdm" && model != "w0wa") || bg.gravity_name() != "GR")
                throw std::invalid_argument("HMcode-2020 supports lcdm and w0wa backgrounds only");
            const double cb = bg.rhoB(0) + bg.rhoCDM(0);
            if (!(cb > 0))
                throw std::invalid_argument("HMcode-2020 requires positive CDM+baryon density");
        }

        double nfw(double k, double rv, double c) {
            const double q = k * rv / c, v = q * c, qc = q + v;
            if (std::abs(v) < 1e-3)
                return 1.0;
            return (std::sin(q) * (gsl_sf_Si(qc) - gsl_sf_Si(q)) +
                    std::cos(q) * (gsl_sf_Ci(qc) - gsl_sf_Ci(q)) - std::sin(v) / qc) /
                   (std::log1p(c) - c / (1 + c));
        }

        // Eisenstein & Hu (1998), equations 29--31. The broadband shape is divided out before
        // Gaussian smoothing, so the smoothing removes BAO without changing the turnover or slope.
        double smooth_shape(const BackgroundCosmology & bg, double kh) {
            const auto & p = bg.get_params();
            const double om = (bg.rhoB(0) + bg.rhoCDM(0) + bg.rhoNcdm(0)) * p.h * p.h;
            const double ob = bg.rhoB(0) * p.h * p.h, fb = ob / om, theta = p.TCMB / 2.7;
            const double s = 44.5 * std::log(9.83 / om) / std::sqrt(1 + 10 * std::pow(ob, 0.75));
            const double ag = 1 - 0.328 * std::log(431 * om) * fb + 0.38 * std::log(22.3 * om) * fb * fb;
            const double gamma = om / p.h * (ag + (1 - ag) / (1 + std::pow(0.43 * kh * p.h * s, 4)));
            const double q = kh * theta * theta / gamma;
            const double L = std::log(2 * M_E + 1.8 * q), C = 14.2 + 731 / (1 + 62.5 * q);
            return std::pow(kh, p.n_s) * std::pow(L / (L + C * q * q), 2);
        }

        HMcode::Spectrum calculate(const BackgroundCosmology & bg, const HMcode::LinearPower & Pm,
                                   const HMcode::LinearPower & Pcb, double x, const std::vector<double> & k,
                                   int nm, double dlnk, const Growth & growth, const Growth & reference) {
            if (!std::isfinite(x) || x > 0.0 || x < std::log(1e-4) || nm < 64 ||
                !(dlnk > 0 && dlnk <= 0.1) || k.empty())
                throw std::invalid_argument("HMcode: require -ln(10000) <= x <= 0, n_mass >= 64, 0 < dlnk <= 0.1");
            for (double kk : k)
                if (!(kk > 0) || !std::isfinite(kk))
                    throw std::invalid_argument("HMcode: k must be finite and positive");
            const double h = bg.get_params().h, length = Mpc / h, volume = length * length * length;
            auto power = [&](const auto & callback, double kh) {
                const double p = callback(x, kh / length) / volume;
                if (!(p > 0) || !std::isfinite(p))
                    throw std::invalid_argument("HMcode: linear power must be finite and positive");
                return p;
            };
            const double om = growth.om, fnu = bg.rhoNcdm(0) / om, rho = 2.775e11 * om;
            const double a = std::exp(x), gx = growth.at(growth.g, x), Gx = growth.at(growth.G, x);
            auto fit = [&](double p0, double p1, double p2, double p3) {
                return p0 + p1 * (1 - gx / a) + p2 * std::pow(1 - gx / a, 2) + p3 * (1 - Gx / a);
            };
            const double logom = std::log10(growth.omega_m(x));
            HMcode::Spectrum out;
            out.delta_c = 3.0 / 20 * std::pow(12 * M_PI, 2.0 / 3) * (1 - 0.041 * fnu) *
                          (1 + fit(-0.0069, -0.0208, 0.0312, 0.0021) * logom +
                           fit(0.0001, -0.0647, -0.0417, 0.0646));
            out.Delta_v = 18 * M_PI * M_PI * (1 + 0.763 * fnu) *
                          (1 + fit(-0.79, -10.17, 2.51, 6.51) * logom +
                           fit(-1.89, 0.38, 18.8, -15.87) * logom * logom);
            if (!(out.delta_c > 0 && out.Delta_v > 0))
                throw std::runtime_error("HMcode: non-positive collapse threshold; background outside fit domain");

            // Integrals in ln k, with sufficient UV coverage for the minimum halo mass (1 Msun/h).
            const auto logs = Utils::linspace(std::log(1e-5), std::log(1e7),
                                              int(std::ceil(std::log(1e12) / dlnk)) + 1);
            const double dl = logs[1] - logs[0];
            std::vector<double> ki, weights;
            double sigv2 = 0;
            for (size_t i = 0; i < logs.size(); ++i) {
                const double kk = std::exp(logs[i]), w = dl * (i == 0 || i + 1 == logs.size() ? 0.5 : 1.0);
                ki.push_back(kk);
                weights.push_back(w * kk * kk * kk * power(Pcb, kk) / (2 * M_PI * M_PI));
                sigv2 += w * kk * power(Pm, kk) / (6 * M_PI * M_PI);
            }
            auto sigma = [&](double R) {
                double s = 0;
                for (size_t j = 0; j < ki.size(); ++j)
                    s += weights[j] * std::pow(top_hat(ki[j] * R), 2);
                return std::sqrt(s);
            };
            out.sigma8_cb = sigma(8.0);
            std::vector<double> mass(nm), radius(nm), nu(nm), rv(nm), conc(nm), weight(nm);
            double dolag = 1;
            const double xinf = -std::log(11.0);
            if (x > xinf)
                dolag = growth.at(growth.g, xinf) / reference.at(reference.g, xinf) *
                        reference.at(reference.g, x) / gx;
            for (int i = 0; i < nm; ++i) {
                mass[i] = std::exp(std::log(1e18) * i / (nm - 1));
                radius[i] = std::cbrt(3 * mass[i] / (4 * M_PI * rho));
                nu[i] = out.delta_c / sigma(radius[i]);
                rv[i] = radius[i] / std::cbrt(out.Delta_v);
                const double target = gx * out.delta_c / sigma(radius[i] * std::cbrt(0.01));
                const double xf = target >= gx ? x : growth.formation_x(target);
                conc[i] = 5.196 * std::exp(x - xf) * dolag;
            }
            // sigma(Rnl) = delta_c. Outside the mass interval use the end radius, as in HMcode's
            // high-z/low-amplitude fallback. This still allows the small one-halo contribution to vanish.
            double logRnl = std::log(radius.front());
            if (nu.front() < 1 && nu.back() > 1)
                logRnl = Utils::find_root([&](double lR) { return sigma(std::exp(lR)) - out.delta_c; },
                                          std::log(radius.front()), std::log(radius.back()), 1e-7, 1e-7);
            else if (nu.back() <= 1)
                logRnl = std::log(radius.back());
            const double eps = 0.01;
            out.n_eff = -3 - (std::log(sigma(std::exp(logRnl + eps))) -
                              std::log(sigma(std::exp(logRnl - eps)))) / eps;
            out.n_eff = std::clamp(out.n_eff, bg.get_params().n_s - 4, bg.get_params().n_s);
            // Sheth--Tormen mass fraction g(nu) dnu. Integrating directly in nu avoids a noisy dln sigma/dln M.
            for (int i = 0; i < nm; ++i) {
                const double q = 0.707 * nu[i] * nu[i];
                const double gst = 0.21616 * (1 + std::pow(q, -0.3)) * std::exp(-q / 2);
                const double dnu = 0.5 * ((i > 0 ? nu[i] - nu[i - 1] : 0) +
                                         (i + 1 < nm ? nu[i + 1] - nu[i] : 0));
                weight[i] = gst * mass[i] / rho * dnu * (1 - fnu) * (1 - fnu);
            }
            const double s8 = out.sigma8_cb;
            const double ks = 0.05618 * std::pow(s8, -1.013), kd = 0.05699 * std::pow(s8, -1.089);
            const double fd = std::clamp(0.2696 * std::pow(s8, 0.9403), 1e-3, 0.99);
            const double eta = 0.1281 * std::pow(s8, -0.3644);
            const double alpha = std::clamp(1.875 * std::pow(1.603, out.n_eff), 0.5, 2.0);
            for (double kk : k) {
                const double pl = power(Pm, kk);
                // Gaussian smoothing of P/P_EH in ln k (HMcode footnote 7), width 0.25.
                // Outside the BAO interval the linear spectrum is used directly.
                double psmooth = pl;
                if (kk > 0.005 && kk < 5.0) {
                    double sum = 0, norm = 0;
                    for (int j = -40; j <= 40; ++j) {
                        const double t = j * 0.025, kj = kk * std::exp(t), w = std::exp(-0.5 * t * t / (0.25 * 0.25));
                        sum += w * power(Pm, kj) / smooth_shape(bg, kj);
                        norm += w;
                    }
                    psmooth = smooth_shape(bg, kk) * sum / norm;
                }
                const double pdw = pl + (-std::expm1(-kk * kk * sigv2)) * (psmooth - pl);
                const double damp2 = 1 / (1 + std::pow(kd / kk, 2.853));
                const double p2 = pdw * (1 - fd * damp2);
                double p1 = 0;
                for (int i = 0; i < nm; ++i)
                    p1 += weight[i] * std::pow(nfw(kk * std::pow(nu[i], eta), rv[i], conc[i]), 2);
                p1 /= 1 + std::pow(ks / kk, 4);
                out.linear.push_back(pl);
                out.one_halo.push_back(p1);
                out.two_halo.push_back(p2);
                out.nonlinear.push_back(std::pow(std::pow(p1, alpha) + std::pow(p2, alpha), 1 / alpha));
            }
            return out;
        }
    } // namespace

    HMcode::HMcode(const BackgroundCosmology & bg, LinearPower Pm, LinearPower Pcb) :
        HMcode(bg, std::move(Pm), std::move(Pcb), Options{}) {}

    HMcode::HMcode(const BackgroundCosmology & bg, LinearPower Pm, LinearPower Pcb, Options opt) {
        check_background(bg);
        if (!(opt.z_max > 0 && opt.z_max < 1e4) || opt.n_z < 4 || opt.n_k < 4 ||
            !(opt.k_min > 0 && opt.k_max > opt.k_min))
            throw std::invalid_argument("HMcode: invalid time or wavenumber table options");
        h = bg.get_params().h;
        x_min = -std::log1p(opt.z_max);
        lnk_min = std::log(opt.k_min);
        lnk_max = std::log(opt.k_max);
        const auto xs = Utils::linspace(x_min, 0.0, opt.n_z);
        const auto lk = Utils::linspace(lnk_min, lnk_max, opt.n_k);
        std::vector<double> k;
        for (double l : lk)
            k.push_back(std::exp(l));
        const Growth growth(bg, false), reference(bg, true);
        std::vector<std::vector<double>> ratios;
        for (double x : xs) {
            const auto p = calculate(bg, Pm, Pcb, x, k, opt.n_mass, opt.dlnk_integral, growth, reference);
            std::vector<double> r;
            for (size_t i = 0; i < k.size(); ++i)
                r.push_back(std::log(p.nonlinear[i] / p.linear[i]));
            ratios.push_back(std::move(r));
        }
        log_ratio.create(xs, lk, ratios, "HMcode-2020 ln(P_NL/P_L)");
    }

    HMcode::Spectrum HMcode::spectrum(const BackgroundCosmology & bg, const LinearPower & Pm,
                                     const LinearPower & Pcb, double x, const std::vector<double> & k,
                                     int n_mass, double dlnk) {
        check_background(bg);
        return calculate(bg, Pm, Pcb, x, k, n_mass, dlnk, Growth(bg, false), Growth(bg, true));
    }

    double HMcode::ratio(double x, double k) const {
        if (!(k > 0) || !std::isfinite(k) || !std::isfinite(x))
            throw std::invalid_argument("HMcode::ratio requires finite x and positive finite k");
        if (x < x_min || std::log(k * Mpc / h) < lnk_min)
            return 1.0;
        return std::exp(log_ratio(std::min(x, 0.0), std::min(std::log(k * Mpc / h), lnk_max)));
    }
} // namespace FML::COSMOLOGY
