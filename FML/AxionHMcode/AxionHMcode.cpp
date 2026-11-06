#include <FML/AxionHMcode/AxionHMcode.h>
#include <gsl/gsl_sf_expint.h>
#include <gsl/gsl_multiroots.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>

namespace FML::COSMOLOGY {
    namespace {
        using Model = AxionHMcode;
        using Vec = std::vector<double>;
        constexpr double pi = 3.14159265358979323846;
        void need(bool b, const char * s) {
            if (!b) throw std::invalid_argument(std::string("AxionHMcode: ") + s);
        }
        double rho(double om) { return 3e4 / (8 * pi * 4.300917270069977e-9) * om; }
        Vec geom(double a, double b, int n) {
            Vec v(n);
            for (int i = 0; i < n; ++i) v[i] = std::exp(std::log(a) + std::log(b / a) * i / (n - 1));
            v.front() = a; v.back() = b;
            return v;
        }
        // Nonuniform composite Simpson rule, including the final-interval correction for an even node count.
        Vec simpson_weights(const Vec & x) {
            const size_t n = x.size();
            Vec w(n, 0);
            if (n == 2) { w[0] = w[1] = (x[1] - x[0]) / 2; return w; }
            for (size_t i = 0; i + 2 < n; i += 2) {
                const double a = x[i + 1] - x[i], b = x[i + 2] - x[i + 1], s = a + b;
                w[i] += s / 6 * (2 - b / a);
                w[i + 1] += s * s * s / (6 * a * b);
                w[i + 2] += s / 6 * (2 - a / b);
            }
            if (n % 2 == 0) {
                const double a = x[n - 2] - x[n - 3], b = x[n - 1] - x[n - 2];
                w[n - 1] += (2 * b * b + 3 * a * b) / (6 * (a + b));
                w[n - 2] += (b * b + 3 * a * b) / (6 * a);
                w[n - 3] -= b * b * b / (6 * a * (a + b));
            }
            return w;
        }
        template <class F> double root(F f, double a, double b) {
            double fa = f(a), fb = f(b);
            if (!std::isfinite(fa) || !std::isfinite(fb) || fa * fb > 0)
                throw std::runtime_error("AxionHMcode: root is not bracketed; increase linear/mass coverage or check model parameters");
            if (fa == 0) return a;
            if (fb == 0) return b;
            for (int i = 0; i < 100; ++i) {
                const double m = (a + b) / 2, fm = f(m);
                if (std::abs(b - a) < 1e-10 || fm == 0) return m;
                if ((fm > 0) == (fa > 0)) { a = m; fa = fm; } else b = m;
            }
            return (a + b) / 2;
        }
        double window(double x) {
            if (std::abs(x) < 0.05) {
                const double t = x * x;
                return 1 - t / 10 + t * t / 280 - t * t * t / 15120;
            }
            return 3 * (std::sin(x) - x * std::cos(x)) / (x * x * x);
        }
        double sinc(double x) { return std::abs(x) < 1e-4 ? 1 - x * x / 6 : std::sin(x) / x; }
        double norm(double c) { return std::log1p(c) - c / (1 + c); }
        double nfw(double k, double r, double c) {
            if (k * r < 1e-4) return 1;
            const double q = k * r / c, v = q * c, t = q + v;
            return (std::cos(q) * (gsl_sf_Ci(t) - gsl_sf_Ci(q)) +
                    std::sin(q) * (gsl_sf_Si(t) - gsl_sf_Si(q)) - std::sin(v) / t) / norm(c);
        }
        // The reference model defines this auxiliary growth integral with z_upper=100 and 2000 nodes.
        double growth(double z, double om) {
            if (z == 100) return 0;
            const double dz = (100 - z) / 1999;
            double sum = 0;
            for (int i = 0; i < 2000; ++i) {
                const double zp = 1 + z + i * dz;
                sum += (i == 0 || i == 1999 ? 0.5 : 1) * zp / std::pow(om * zp * zp * zp + 1 - om, 1.5);
            }
            return 2.5 * om * std::sqrt(om * std::pow(1 + z, 3) + 1 - om) * sum * dz;
        }
        // Accumulated growth with the reference's finite upper bound z=10000. The growing/decaying-mode
        // initial conditions reproduce its nested integral without repeated numerical double integration.
        double accumulated_growth(double z, double om) {
            const double xi = -std::log(10001.0), xf = -std::log1p(z), ai = std::exp(xi);
            const int n = std::max(1, int(std::ceil((xf - xi) / 0.003)));
            const double dx = (xf - xi) / n;
            std::array<double, 3> y{0, 2.5 * om / (ai * ai * (om / std::pow(ai, 3) + 1 - om)), 0};
            auto rhs = [&](double x, const auto & v) {
                const double m = om * std::exp(-3 * x), frac = m / (m + 1 - om);
                return std::array<double, 3>{v[1], -(2 - 1.5 * frac) * v[1] + 1.5 * frac * v[0], v[0]};
            };
            auto add = [](auto a, const auto & b, double f) {
                for (int j = 0; j < 3; ++j) a[j] += f * b[j];
                return a;
            };
            for (int i = 0; i < n; ++i) {
                const double x = xi + i * dx;
                const auto p = rhs(x, y), q = rhs(x + dx / 2, add(y, p, dx / 2));
                const auto r = rhs(x + dx / 2, add(y, q, dx / 2)), s = rhs(x + dx, add(y, r, dx));
                for (int j = 0; j < 3; ++j) y[j] += dx / 6 * (p[j] + 2 * q[j] + 2 * r[j] + s[j]);
            }
            return y[2];
        }
        struct Linear {
            const Model::Input & in;
            Vec weights, dk;
            explicit Linear(const Model::Input & i) : in(i), weights(i.k.size()), dk(simpson_weights(i.k)) {
                for (size_t j = 0; j < i.k.size(); ++j) {
                    const double d = 0.5 * ((j ? std::log(i.k[j] / i.k[j - 1]) : 0) +
                                            (j + 1 < i.k.size() ? std::log(i.k[j + 1] / i.k[j]) : 0));
                    weights[j] = d * std::pow(i.k[j], 3) * i.cold[j] / (2 * pi * pi);
                }
            }
            double sigma(double R) const {
                double v = 0;
                for (size_t j = 0; j < weights.size(); ++j) v += weights[j] * std::pow(window(R * in.k[j]), 2);
                return std::sqrt(v);
            }
            double derivative(double R) const {
                double d = 0;
                for (size_t j = 0; j < weights.size(); ++j) {
                    const double k = in.k[j], x = R * k;
                    if (x <= 1e-3) continue;
                    const double a = std::sin(x) * (1 - 3 / (x * x)) + 3 * std::cos(x) / x;
                    const double b = std::sin(x) - x * std::cos(x);
                    d += dk[j] * a * b * in.cold[j] / (k * k);
                }
                return 3 * d / (pi * pi * std::pow(sigma(R), 2) * std::pow(R, 4));
            }
            double at(double k, const Vec & p) const {
                auto it = std::upper_bound(in.k.begin(), in.k.end(), k);
                const size_t j = it == in.k.end() ? in.k.size() - 2 : size_t(it - in.k.begin()) - 1;
                const double f = std::log(k / in.k[j]) / std::log(in.k[j + 1] / in.k[j]);
                if (p[j] == 0 || p[j + 1] == 0) return p[j] + f * (p[j + 1] - p[j]);
                return std::exp(std::log(p[j]) + f * std::log(p[j + 1] / p[j]));
            }
        };
        struct Context {
            const Model::Cosmology & c;
            const Model::Options & opt;
            Linear linear;
            double om, oc, fa, dc, dv, alpha, s8, eta, kstar;
            Context(const Model::Cosmology & cc, const Model::Input & in, const Model::Options & o) :
                c(cc), opt(o), linear(in), om(cc.Omega_b + cc.Omega_cdm + cc.Omega_axion),
                oc(cc.Omega_b + cc.Omega_cdm), fa(cc.Omega_axion / om) {
                const double a = 1 / (1 + c.z), g = growth(c.z, om) / a, G = accumulated_growth(c.z, om) / a;
                const double omz = om / (om + (1 - om) * std::pow(a, 3)), l = std::log10(omz);
                auto f = [&](double p, double q, double r, double s) { return p + q * (1 - g) + r * (1 - g) * (1 - g) + s * (1 - G); };
                const bool dome = opt.version == Model::Version::Dome2024;
                dc = 1.686 * (1 + f(-.0069, -.0208, .0312, .0021) * l + f(.0001, -.0647, -.0417, .0646));
                dv = 177.7 * (1 + f(-.79, -10.17, 2.51, 6.51) * l + f(-1.89, .38, 18.8, -15.87) * l * l);
                if (dome) { dc *= 1 - .041 * fa; dv *= 1 + .763 * fa; }
                s8 = linear.sigma(8);
                eta = opt.halo_bloating ? .1281 * std::pow(s8, -.3644) : 0;
                kstar = .056188 * std::pow(s8, -1.013);
                double neff = -3;
                auto eq = [&](double lr) { return std::log(linear.sigma(std::exp(lr)) / dc); };
                if (eq(-10) * eq(1) < 0) {
                    const double R = std::exp(root(eq, -10, 1));
                    if (R > 1e-5) neff = -3 - 3 * linear.derivative(R);
                }
                alpha = 1.875 * std::pow(1.603, neff);
            }
            double radius(double M) const { return std::cbrt(3 * M / (4 * pi * rho(oc))); }
            double concentration(double M) const {
                const double nu = dc / linear.sigma(radius(.01 * M));
                double zf = c.z;
                if (nu < 1) {
                    const double target = growth(c.z, om) * nu;
                    zf = root([&](double z) { return growth(z, om) - target; }, c.z, 100);
                }
                return opt.concentration_amplitude * (1 + zf) / (1 + c.z);
            }
            double mass_function(double M) const {
                const double R = radius(M), nu = dc / linear.sigma(R), q = .707 * nu * nu;
                const double A = std::sqrt(2 * .707) / (std::sqrt(pi) + std::tgamma(.2) / std::pow(2, .3));
                const double st = A * nu * (1 + std::pow(q, -.3)) * std::exp(-q / 2);
                return rho(oc) / (2 * M * M) * st * std::abs(linear.derivative(R));
            }
            double bias(double M) const {
                const double nu = dc / linear.sigma(radius(M)), q = .707 * nu * nu;
                return 1 + (q - 1 + .6 / (1 + std::pow(q, .3))) / dc;
            }
            double rv(double M) const { return radius(M) / std::cbrt(dv); }
        };
        struct Profile {
            Model::Halo halo;
            Vec r, weights, soliton, envelope, normalized_shell;
            double rs{}, cnorm{}, prefactor{}, ax_to_cold{};
            bool density_cut{};
            Profile(double M, double Ma, double jeans, double beta, const Context & ctx) {
                halo.cold_mass = M; halo.axion_mass = Ma;
                halo.concentration = ctx.concentration(M); halo.virial_radius = ctx.rv(M);
                const auto & c = ctx.c;
                const double velocity = std::sqrt(1.3271244e20 * M * ctx.om / ctx.oc / (halo.virial_radius * 3.086e22));
                halo.core_radius = 2 * pi * 1.0545718176461565e-34 / (7.5 * c.mass_eV * 1.78266269594644e-36 * velocity) /
                                   3.086e22 * c.h / std::sqrt(1 + c.z);
                rs = halo.virial_radius / halo.concentration;
                ax_to_cold = c.Omega_axion / ctx.oc;
                cnorm = rho(ctx.oc) * ctx.dv * std::pow(halo.concentration, 3) / (3 * norm(halo.concentration));
                prefactor = (1 + c.z) * .019 * std::pow(c.mass_eV / 1e-22, -2) * c.h * c.h * 1e18 /
                            std::pow(halo.core_radius * 1e3 / c.h, 4);
                // Reproduce the reference's amplitude rejection rule. Its initial NFW-core estimate uses
                // B=5.2 even when the actual profile uses B=4, and its mass relation acts on int rho*r^2 dr.
                const double guess_c = halo.concentration * 5.2 / ctx.opt.concentration_amplitude;
                double core_integral = 0;
                for (int j = 0; j <= 512; ++j) {
                    const double x = j / 512.0;
                    core_integral += (j == 0 || j == 512 ? 1 : j % 2 ? 4 : 2) * x * x / std::pow(1 + .091 * x * x, 8);
                }
                core_integral *= prefactor * std::pow(halo.core_radius, 3) / (3 * 512);
                const double nfw_core = M / (4 * pi) * norm(halo.core_radius * guess_c / halo.virial_radius) / norm(guess_c);
                const double initial_mass = ax_to_cold * nfw_core * (ctx.opt.version == Model::Version::Dome2024 ?
                                              std::pow(1 + jeans / nfw_core, -beta) : 1);
                halo.central_density_guess = initial_mass / core_integral;
                need(std::isfinite(prefactor) && prefactor > 0 && std::isfinite(halo.central_density_guess) &&
                     halo.central_density_guess > 0, "profile normalization is outside the numerical domain");
                r = geom(1e-15, halo.virial_radius, ctx.opt.n_radius);
                weights = simpson_weights(r);
                for (double rr : r) { soliton.push_back(unit_soliton(rr)); envelope.push_back(nfw_density(rr)); }
                double unit_mass = 0;
                for (size_t i = 0; i < r.size(); ++i) unit_mass += 4 * pi * weights[i] * r[i] * r[i] * soliton[i];
                // The reference uses a hybrid nonlinear solve from this core estimate, rather than a
                // monotone bracket. The last-intersection construction has discontinuities and plateaus;
                // a positive finite approximate solution still defines a usable normalized Fourier shape.
                struct State { Profile * p; double mass, best_amp, error; } state{this, Ma, Ma / unit_mass, std::numeric_limits<double>::infinity()};
                gsl_multiroot_function function{
                    [](const gsl_vector * x, void * data, gsl_vector * f) -> int {
                        auto & s = *static_cast<State *>(data);
                        const double amp = gsl_vector_get(x, 0), err = s.p->profile_mass(amp) / s.mass - 1;
                        if (!std::isfinite(err)) return GSL_EBADFUNC;
                        if (amp > 0 && std::abs(err) < s.error) { s.error = std::abs(err); s.best_amp = amp; }
                        gsl_vector_set(f, 0, err);
                        return GSL_SUCCESS;
                    }, 1, &state};
                gsl_vector * initial = gsl_vector_alloc(1);
                gsl_vector_set(initial, 0, halo.central_density_guess);
                auto * solver = gsl_multiroot_fsolver_alloc(gsl_multiroot_fsolver_hybrids, 1);
                int status = gsl_multiroot_fsolver_set(solver, &function, initial);
                for (int j = 0; j < 200 && status == GSL_SUCCESS; ++j) {
                    status = gsl_multiroot_fsolver_iterate(solver);
                    if (state.error < 1e-10 || gsl_multiroot_test_delta(solver->dx, solver->x, 0, 1.49012e-8) == GSL_SUCCESS) break;
                }
                gsl_multiroot_fsolver_free(solver);
                gsl_vector_free(initial);
                if (!std::isfinite(state.error)) throw std::runtime_error("AxionHMcode: central-density solver found no finite profile");
                const double amp = state.best_amp;
                halo.central_density_parameter = amp;
                density_cut = ctx.opt.reference_density_cut && std::abs(amp - halo.central_density_guess) > 100;
                const double match = matching_radius(amp);
                double mass = 0;
                for (size_t i = 0; i < r.size(); ++i) {
                    const double density = match > 0 && r[i] > match ? envelope[i] : amp * soliton[i];
                    const double shell = 4 * pi * weights[i] * r[i] * r[i] * density;
                    normalized_shell.push_back(shell); mass += shell;
                }
                if (!(mass > 0) || !std::isfinite(mass)) throw std::runtime_error("AxionHMcode: invalid axion profile mass");
                halo.raw_profile_mass_error = mass / Ma - 1;
                halo.profile_rescaling = Ma / mass;
                for (double & w : normalized_shell) w /= mass;
                halo.normalization_error = std::accumulate(normalized_shell.begin(), normalized_shell.end(), 0.0) - 1;
            }
            double unit_soliton(double rr) const { return prefactor / std::pow(1 + .091 * std::pow(rr / halo.core_radius, 2), 8); }
            double nfw_density(double rr) const { const double x = rr / rs; return ax_to_cold * cnorm / (x * std::pow(1 + x, 2)); }
            double matching_radius(double amp) const {
                int last = -1;
                double previous = amp * soliton[0] - envelope[0];
                for (size_t i = 1; i < r.size(); ++i) {
                    const double v = amp * soliton[i] - envelope[i];
                    if ((v > 0) != (previous > 0)) last = int(i) - 1;
                    previous = v;
                }
                if (last < 0) return -1;
                return r[last];
            }
            double profile_mass(double amp) const {
                const double match = matching_radius(amp);
                double sum = 0;
                for (size_t i = 0; i < r.size(); ++i)
                    sum += 4 * pi * weights[i] * r[i] * r[i] * (match > 0 && r[i] > match ? envelope[i] : amp * soliton[i]);
                return sum;
            }
            double fourier(double k) const {
                double sum = 0;
                for (size_t i = 0; i < r.size(); ++i) sum += normalized_shell[i] * sinc(k * r[i]);
                return sum;
            }
        };
        double combine(double a, double b, double alpha) {
            if (alpha == 1) return a + b;
            if (a < 0 || b < 0) throw std::runtime_error("AxionHMcode: negative halo term cannot be used in the smoothed transition");
            const double s = std::max(a, b);
            return s == 0 ? 0 : s * std::pow(std::pow(a / s, alpha) + std::pow(b / s, alpha), 1 / alpha);
        }
    }

    double AxionHMcode::beta2(double z, double f, bool extrapolate) {
        need(std::isfinite(z) && z >= 0 && z < 100 && std::isfinite(f) && f >= 0 && f < 1, "beta2 requires 0<=z<100 and 0<=f<1");
        need(extrapolate || (z >= 1 && z <= 8 && f >= .01 && f <= .3), "Dome beta2 calibration requires z in [1,8], Omega_a/Omega_m in [0.01,0.3]");
        const std::array<double, 4> fs{.01, .1, .2, .3};
        const double data[8][4] = {{1.27,.98,.61,.30},{1.22,.87,.59,.32},{1.32,.93,.66,.40},{1.37,.96,.71,.45},
                                  {1.43,1.04,.75,.51},{1.46,1.02,.79,.58},{1.48,1.04,.77,.63},{1.52,1.04,.78,.62}};
        const int i = std::clamp(int(std::floor(z)) - 1, 0, 6);
        const int j = std::clamp(int(std::upper_bound(fs.begin(), fs.end(), f) - fs.begin()) - 1, 0, 2);
        const double t = z - (i + 1), u = (f - fs[j]) / (fs[j + 1] - fs[j]);
        return (1 - t) * ((1 - u) * data[i][j] + u * data[i][j + 1]) + t * ((1 - u) * data[i + 1][j] + u * data[i + 1][j + 1]);
    }

    double AxionHMcode::jeans_mass(double z, double om, double h, double m) {
        need(std::isfinite(z) && z >= 0 && std::isfinite(om) && om > 0 && std::isfinite(h) && h > 0 && std::isfinite(m) && m > 0,
             "invalid Jeans-mass inputs");
        const double G = 6.674e-29 / std::pow(3.086e16, 3);
        const double rhoc = 3 * std::pow(100 / 3.086e19, 2) / (8 * pi * G) / 2e30;
        const double k = 66.5 * std::pow(1 + z, -.25) * std::pow(om * h * h / .12, .25) * std::sqrt(m / 1e-22);
        return 4 * pi / 3 * std::pow(pi / k * h, 3) * rhoc * om;
    }

    void AxionHMcode::validate(const Cosmology & c, const Input & in, const Options & o) {
        need(std::isfinite(c.h) && c.h > 0 && std::isfinite(c.z) && c.z >= 0 && c.z < 100, "need h>0 and 0<=z<100");
        need(std::isfinite(c.Omega_b) && c.Omega_b >= 0 && std::isfinite(c.Omega_cdm) && c.Omega_cdm >= 0 &&
             std::isfinite(c.Omega_axion) && c.Omega_axion >= 0, "densities must be finite and nonnegative");
        const double oc = c.Omega_b + c.Omega_cdm, om = oc + c.Omega_axion, f = c.Omega_axion / om;
        need(oc > 0 && om < 1 && std::isfinite(c.mass_eV) && c.mass_eV > 0, "need positive cold density, Omega_m<1 and positive axion mass");
        need(o.n_mass >= 33 && o.n_radius >= 256 && std::isfinite(o.mass_min) && std::isfinite(o.mass_max) &&
             o.mass_min > 0 && o.mass_max > o.mass_min && std::isfinite(o.concentration_amplitude) && o.concentration_amplitude > 0,
             "invalid mass/radius resolution or concentration");
        need(in.k.size() >= 32 && in.k.size() == in.cold.size() && (f == 0 || in.k.size() == in.axion.size()), "linear arrays must have matching lengths >=32");
        for (size_t i = 0; i < in.k.size(); ++i) {
            need(std::isfinite(in.k[i]) && in.k[i] > 0 && (i == 0 || in.k[i] > in.k[i - 1]), "k must be positive, finite and strictly increasing");
            need(std::isfinite(in.cold[i]) && in.cold[i] > 0, "cold power must be finite and positive");
            if (f > 0) need(std::isfinite(in.axion[i]) && in.axion[i] >= 0, "axion power must be finite and nonnegative");
        }
        need(in.k.front() <= 1e-4 && in.k.back() >= 100, "linear input must cover k<=1e-4 through k>=100 h/Mpc");
        if (o.version == Version::Dome2024 && f > 0 && !o.allow_extrapolation) {
            beta2(c.z, f);
            need(std::abs(std::log10(c.mass_eV) + 24.5) < 1e-6,
                 "Dome calibration is centered on m=10^-24.5 eV; set allow_extrapolation explicitly for other masses");
        }
    }

    AxionHMcode::Spectrum AxionHMcode::spectrum(const Cosmology & c, const Input & in, const Vec & k) {
        return spectrum(c, in, k, Options{});
    }

    AxionHMcode::Spectrum AxionHMcode::spectrum(const Cosmology & c, const Input & in, const Vec & k, const Options & o) {
        validate(c, in, o);
        need(!k.empty(), "output k grid is empty");
        for (double v : k) need(std::isfinite(v) && v >= in.k.front() && v <= in.k.back(), "output k must lie inside the linear input range");
        Context ctx(c, in, o);
        Spectrum out;
        out.k = k; out.sigma8_cold = ctx.s8; out.delta_c = ctx.dc; out.Delta_vir = ctx.dv;
        const bool dome = o.version == Version::Dome2024;
        const bool smoothing = o.transition == Transition::Smoothed || (o.transition == Transition::Recommended && dome);
        out.extrapolated = dome && ctx.fa > 0 && (c.z < 1 || c.z > 8 || ctx.fa < .01 || ctx.fa > .3 || std::abs(std::log10(c.mass_eV) + 24.5) >= 1e-6);
        out.alpha_cold = out.alpha_cross = 1;
        if (smoothing) {
            out.alpha_cold = ctx.alpha;
            if (dome) {
                out.alpha_cold /= 1 + .124 * std::pow(1e-24 / c.mass_eV, .045) * std::pow(ctx.fa, .226) * std::pow(1 + c.z, 1.13);
                out.alpha_cross = ctx.alpha / (1 + .0487 * std::pow(1e-24 / c.mass_eV, .045) * std::pow(ctx.fa, .224) * std::pow(1 + c.z, 2.21));
            }
        }
        const Vec M = geom(o.mass_min, o.mass_max, o.n_mass), mw = simpson_weights(M);
        Vec abundance, bias, conc, rv, bloating;
        for (double m : M) {
            abundance.push_back(ctx.mass_function(m)); bias.push_back(ctx.bias(m));
            conc.push_back(ctx.concentration(m)); rv.push_back(ctx.rv(m));
            bloating.push_back(std::pow(ctx.dc / ctx.linear.sigma(ctx.radius(m)), ctx.eta));
        }
        Vec axM, axAbundance, axBias, axC, axR, axBloat;
        std::vector<Profile> profiles;
        if (ctx.fa > 0) {
            auto ratio = [&](double lm) {
                const double m = std::pow(10, lm), cc = ctx.concentration(m);
                const double kj = 1424.6287862379945 * std::pow(c.mass_eV / 1e-22, 2.0 / 3) *
                    std::pow(norm(cc) / (cc * cc) * 100 / (std::log(11) - 10.0 / 11), -1.0 / 3) *
                    std::pow(m / c.h / 1e10, 1.0 / 9) * std::pow(ctx.oc * c.h * c.h / .12, 2.0 / 9) /
                    c.h * std::cbrt(1 + c.z);
                return pi / kj / ctx.rv(m) - 1;
            };
            double lower = 7, upper = 17;
            while (ratio(lower) < 0 && lower > -10) lower -= 3;
            while (ratio(upper) > 0 && upper < 30) upper += 3;
            out.cutoff_mass = std::pow(10, root(ratio, lower, upper));
            out.beta2 = dome ? beta2(c.z, ctx.fa, o.allow_extrapolation) : 1;
            const double jm = jeans_mass(c.z, ctx.om * (o.jeans_density == JeansDensity::ReferenceCode ? c.h * c.h : 1), c.h, c.mass_eV);
            for (size_t i = 0; i < M.size(); ++i) {
                if (!dome && M[i] < out.cutoff_mass) continue;
                const double ma = c.Omega_axion / ctx.oc * M[i] * (dome ? std::pow(1 + jm / M[i], -out.beta2) : 1);
                profiles.emplace_back(M[i], ma, jm, out.beta2, ctx);
                if (profiles.back().density_cut || std::abs(profiles.back().halo.normalization_error) > 1e-5) {
                    out.rejected_axion_hosts.push_back(M[i]);
                    profiles.pop_back();
                    continue;
                }
                out.haloes.push_back(profiles.back().halo);
                axM.push_back(M[i]); axAbundance.push_back(abundance[i]); axBias.push_back(bias[i]);
                axC.push_back(conc[i]); axR.push_back(rv[i]); axBloat.push_back(bloating[i]);
            }
        }
        Vec aw;
        if (profiles.size() >= 2) aw = simpson_weights(axM);
        else if (!profiles.empty()) throw std::runtime_error("AxionHMcode: axion halo interval has fewer than two mass samples");
        double fc = 0;
        for (size_t i = 0; i < profiles.size(); ++i)
            fc += aw[i] * axAbundance[i] * axBias[i] * profiles[i].halo.axion_mass / rho(c.Omega_axion);
        if (fc < -1e-8 || fc > 1.01) {
            double cold_integral = 0;
            for (size_t i = 0; i < M.size(); ++i) cold_integral += mw[i] * abundance[i] * M[i] * bias[i] / rho(ctx.oc);
            throw std::runtime_error("AxionHMcode: clustered fraction=" + std::to_string(fc) +
                ", cold bias integral=" + std::to_string(cold_integral) + ", rejected hosts=" +
                std::to_string(out.rejected_axion_hosts.size()) + "; check mass/k convergence");
        }
        fc = std::clamp(fc, 0.0, 1.0);
        out.clustered_fraction = fc;
        for (double kk : k) {
            const double pc = ctx.linear.at(kk, in.cold), pa = ctx.fa == 0 ? 0 : ctx.linear.at(kk, in.axion);
            const double damp1 = o.one_halo_damping ? 1 / (1 + std::pow(ctx.kstar / kk, 4)) : 1;
            double p1c = 0, cold_bias = 0, mass_bias = 0, p1x = 0, p1a = 0, ax_bias = 0;
            for (size_t i = 0; i < M.size(); ++i) {
                const double u = nfw(kk * bloating[i], rv[i], conc[i]);
                p1c += mw[i] * abundance[i] * M[i] * M[i] * u * u / std::pow(rho(ctx.oc), 2);
                cold_bias += mw[i] * abundance[i] * M[i] * bias[i] * u / rho(ctx.oc);
                mass_bias += mw[i] * abundance[i] * M[i] * bias[i] / rho(ctx.oc);
            }
            cold_bias += nfw(kk * bloating[0], rv[0], conc[0]) * (1 - mass_bias);
            for (size_t i = 0; i < profiles.size(); ++i) {
                const double ma = profiles[i].halo.axion_mass, ua = profiles[i].fourier(kk);
                const double uc = nfw(kk * axBloat[i], axR[i], axC[i]), aweight = aw[i] * axAbundance[i];
                p1x += aweight * axM[i] * ma * uc * ua / (rho(ctx.oc) * rho(c.Omega_axion));
                p1a += aweight * ma * ma * ua * ua / std::pow(rho(c.Omega_axion), 2);
                ax_bias += aweight * ma * axBias[i] * ua / rho(c.Omega_axion);
            }
            p1c *= damp1; p1x *= damp1; p1a *= damp1;
            double p2 = pc;
            if (o.two_halo_damping) {
                const double kd = .05699 * std::pow(ctx.s8, -1.089), fd = .2696 * std::pow(ctx.s8, .9403);
                p2 *= 1 - fd / (1 + std::pow(kd / kk, 2.853));
            }
            double ac = out.alpha_cold;
            if (smoothing && dome && ctx.fa >= .01) {
                const double alow = std::max(ac, 1.1 / (1 + c.z));
                ac = alow + (ac - alow) / (1 + std::exp(std::clamp((.5 - kk) / .1, -700.0, 700.0)));
            }
            const double pcc = combine(p1c, p2 * (o.full_two_halo ? cold_bias * cold_bias : 1), ac);
            const double px_cluster = combine(p1x, p2 * (o.full_two_halo ? cold_bias * ax_bias : fc), out.alpha_cross);
            const double pca = ctx.fa == 0 ? 0 : px_cluster + (1 - fc) * std::sqrt(pa * pcc);
            const double pax_cluster = p1a + p2 * (o.full_two_halo ? ax_bias * ax_bias : fc * fc);
            const double paa = ctx.fa == 0 ? 0 : std::pow(std::sqrt(std::max(0.0, pax_cluster)) + (1 - fc) * std::sqrt(pa), 2);
            const double total = (1 - ctx.fa) * (1 - ctx.fa) * pcc + 2 * ctx.fa * (1 - ctx.fa) * pca + ctx.fa * ctx.fa * paa;
            if (!(total > 0) || !std::isfinite(total)) throw std::runtime_error("AxionHMcode: non-finite or non-positive nonlinear power");
            out.cold.push_back(pcc); out.cold_axion.push_back(pca); out.axion.push_back(paa); out.total.push_back(total);
            out.cold_one_halo.push_back(p1c); out.cross_one_halo.push_back(p1x); out.axion_one_halo.push_back(p1a);
            out.linear_total.push_back(std::pow((1 - ctx.fa) * std::sqrt(pc) + ctx.fa * std::sqrt(pa), 2));
        }
        return out;
    }
} // namespace FML::COSMOLOGY
