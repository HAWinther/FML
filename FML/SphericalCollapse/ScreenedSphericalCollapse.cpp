#include "ScreenedSphericalCollapse.h"

#include <FML/Math/Math.h>
#include <FML/ODESolver/ODESolver.h>

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

namespace FML {
    namespace COSMOLOGY {
        namespace SPHERICALCOLLAPSE {

            using DVector = FML::SOLVERS::ODESOLVER::DVector;
            using ODESolver = FML::SOLVERS::ODESOLVER::ODESolver;

            FifthForce fifth_force_none() {
                return [](double, double, double) { return 0.0; };
            }

            FifthForce fifth_force_unscreened(Func1 muofx) {
                return [muofx](double x, double, double) { return muofx(x) - 1.0; };
            }

            FifthForce fifth_force_hu_sawicki(double fR0, double n, double OmegaM, double R_th) {
                if (not(fR0 >= 0.0) or not(n > 0.0) or not(OmegaM > 0.0 and OmegaM < 1.0) or not(R_th > 0.0))
                    throw std::invalid_argument("fifth_force_hu_sawicki: need fR0 >= 0, n > 0, 0 < OmegaM < 1, R_th > 0");
                const double OmegaL = 1.0 - OmegaM;
                // (H0 R_th / c)^2 OmegaM: 2 Phi_N a y with Phi_N = G M / R the Newtonian potential at the surface
                const double two_phi_ay = std::pow(R_th / c_over_H0_Mpch, 2) * OmegaM;
                // |f_R(rho)| in the large-curvature limit: R ~ rho_m + 4 rho_Lambda
                auto fR = [=](double x, double one_plus_delta) {
                    const double curvature = OmegaM * std::exp(-3.0 * x) * one_plus_delta + 4.0 * OmegaL;
                    return fR0 * std::pow((OmegaM + 4.0 * OmegaL) / curvature, n + 1.0);
                };
                return [=](double x, double delta, double delta_env) {
                    if (fR0 == 0.0)
                        return 0.0;
                    const double a = std::exp(x);
                    const double y = std::pow(1.0 + delta, -1.0 / 3.0); // R / (a R_th)
                    const double dRR = (fR(x, 1.0 + delta_env) - fR(x, 1.0 + delta)) * a * y / two_phi_ay;
                    if (dRR <= 0.0)
                        return 0.0;
                    if (dRR >= 1.0)
                        return 1.0 / 3.0;
                    // The mass fraction in the thin shell, times the unscreened 1/3
                    return (1.0 - std::pow(1.0 - dRR, 3)) / 3.0;
                };
            }

            FifthForce fifth_force_ndgp(double Omega_rc, const SphericalCollapseModel & model) {
                if (not(Omega_rc >= 0.0))
                    throw std::invalid_argument("fifth_force_ndgp: need Omega_rc >= 0");
                const Func1 E = model.Eofx, dlogE = model.logEprimeofx, Om = model.OmegaMofx;
                return [=](double x, double delta, double) {
                    if (Omega_rc == 0.0)
                        return 0.0;
                    const double Hrc = E(x) / (2.0 * std::sqrt(Omega_rc));
                    const double beta = 1.0 + 2.0 * Hrc * (1.0 + dlogE(x) / 3.0);
                    const double eps = 8.0 * Hrc * Hrc * Om(x) * delta / (9.0 * beta * beta);
                    // (sqrt(1 + eps) - 1) / eps -> 1/2 for small eps (also for eps < 0, an underdensity)
                    const double g = std::abs(eps) < 1e-6 ? 0.5 - eps / 8.0 : (std::sqrt(1.0 + eps) - 1.0) / eps;
                    return 2.0 / (3.0 * beta) * g;
                };
            }

            double lagrangian_radius(double M, double OmegaM) {
                // rho_crit,0 = 2.77536627e11 h^2 Msun / Mpc^3
                const double rho = 2.77536627e11 * OmegaM;
                return std::cbrt(3.0 * M / (4.0 * M_PI * rho));
            }

            double environment_collapse_threshold(double OmegaM, double z) {
                return 3.0 / 20.0 * std::pow(12.0 * M_PI, 2.0 / 3.0) *
                       (1.0 - 0.0123 * std::log10(1.0 + (1.0 / OmegaM - 1.0) / std::pow(1.0 + z, 3)));
            }

            double environment_most_probable(double sigma2_8, double dsigma2dR_8, double OmegaM, double z) {
                if (not(sigma2_8 > 0.0) or not(dsigma2dR_8 < 0.0))
                    throw std::invalid_argument("environment_most_probable: need sigma^2(8) > 0 and dsigma^2/dR < 0");
                // Lombriser et al. 2013: the distribution of the linear density zeta of spheres of Eulerian radius
                // R_E = 5 Mpc/h (with the mapping to Lagrangian radii of a spherical collapse with threshold dc),
                // gamma = -d log sigma^2 / d log R / 3 at R = 8 Mpc/h
                const double dc = environment_collapse_threshold(OmegaM, z);
                const double gamma = -8.0 / (3.0 * sigma2_8) * dsigma2dR_8;
                const double omega = gamma * dc;
                const double beta = std::pow(5.0 / 8.0, 3.0 / dc) / std::pow(sigma2_8, 1.0 / omega);
                auto P = [=](double zeta) {
                    const double s = 1.0 - zeta / dc;
                    return (1.0 + (omega - 1.0) * zeta / dc) * std::pow(s, -omega / 2.0 - 1.0) *
                           std::exp(-std::pow(beta, omega) / 2.0 * zeta * zeta / std::pow(s, omega));
                };
                // The maximum on (-dc, 0.99 dc): a grid, then golden-section refinement
                const int n = 400;
                double best = 0.0, Pbest = -1.0;
                for (int i = 0; i <= n; i++) {
                    const double zeta = -dc + 1.99 * dc * i / n;
                    const double p = P(zeta);
                    if (p > Pbest) {
                        Pbest = p;
                        best = zeta;
                    }
                }
                double lo = std::max(-dc, best - 1.99 * dc / n), hi = std::min(0.99 * dc, best + 1.99 * dc / n);
                const double r = (std::sqrt(5.0) - 1.0) / 2.0;
                double c = hi - r * (hi - lo), d = lo + r * (hi - lo);
                for (int it = 0; it < 200 and hi - lo > 1e-12; it++) {
                    if (P(c) > P(d))
                        hi = d;
                    else
                        lo = c;
                    c = hi - r * (hi - lo);
                    d = lo + r * (hi - lo);
                }
                return 0.5 * (lo + hi);
            }

            namespace {
                // Linear growth from xini to x with model.muofx, for a growing mode delta ~ a at xini
                double linear_growth_ratio(const SphericalCollapseModel & model, double xini, double x) {
                    ODESolver ode;
                    ode.set_accuracy(1e-3, 1e-12, 1e-8);
                    FML::SOLVERS::ODESOLVER::ODEFunction deriv = [&model](double xx, const double * y, double * dydx) {
                        return model.ode_system_linear_growth(xx, y, dydx);
                    };
                    ode.solve(deriv, {xini, x}, {1.0, 1.0});
                    return ode.get_final_data()[0];
                }

                // Halo (0, 1), linear halo (2, 3), environment (4, 5)
                struct Solution {
                    double xcollapsed;
                    DVector x, delta, delta_prime, delta_lin, delta_env;
                };

                Solution evolve(const SphericalCollapseModel & model,
                                const FifthForce & F,
                                double delta_ini,
                                double delta_env_ini,
                                double xcollapse,
                                const ScreenedCollapseOptions & opt,
                                bool get_arrays) {
                    Solution s;
                    s.xcollapsed = xcollapse;
                    const double dinf = opt.delta_infinity;
                    DVector x_array = FML::MATH::linspace(std::exp(opt.xini), std::exp(xcollapse), opt.npts);
                    for (auto & x : x_array)
                        x = std::log(x);
                    FML::SOLVERS::ODESOLVER::ODEFunction deriv = [&](double x, const double * y, double * dydx) {
                        const double E1 = model.logEprimeofx(x), Om = model.OmegaMofx(x), mu = model.muofx(x);
                        // Environment (unscreened, mu)
                        if (y[4] > dinf) {
                            dydx[4] = dydx[5] = 0.0;
                        } else {
                            dydx[4] = y[5];
                            dydx[5] = -(2.0 + E1) * y[5] + 4.0 / 3.0 * y[5] * y[5] / (1.0 + y[4]) +
                                      1.5 * Om * mu * y[4] * (1.0 + y[4]);
                        }
                        // Linear halo
                        dydx[2] = y[3];
                        dydx[3] = -(2.0 + E1) * y[3] + 1.5 * Om * mu * y[2];
                        // Halo with the fifth force; frozen once collapsed
                        if (y[0] > dinf) {
                            if (x < s.xcollapsed)
                                s.xcollapsed = x;
                            dydx[0] = dydx[1] = 0.0;
                            return GSL_SUCCESS;
                        }
                        const double f = F(x, y[0], y[4]);
                        dydx[0] = y[1];
                        dydx[1] = -(2.0 + E1) * y[1] + 4.0 / 3.0 * y[1] * y[1] / (1.0 + y[0]) +
                                  1.5 * Om * (1.0 + f) * y[0] * (1.0 + y[0]);
                        return GSL_SUCCESS;
                    };
                    // Growing mode with the EdS second order term (delta = delta_L + 17/21 delta_L^2)
                    auto ic = [](double d) { return std::pair{d + 17.0 / 21.0 * d * d, d + 34.0 / 21.0 * d * d}; };
                    const auto [h0, h1] = ic(delta_ini);
                    const auto [e0, e1] = ic(delta_env_ini);
                    ODESolver ode;
                    ode.set_accuracy(1e-3, 1e-12, 1e-6);
                    ode.solve(deriv, x_array, {h0, h1, delta_ini, delta_ini, e0, e1}, gsl_odeiv2_step_rkck);
                    if (get_arrays) {
                        s.x = x_array;
                        s.delta = ode.get_data_by_component(0);
                        s.delta_prime = ode.get_data_by_component(1);
                        s.delta_lin = ode.get_data_by_component(2);
                        s.delta_env = ode.get_data_by_component(4);
                    }
                    return s;
                }
            } // namespace

            ScreenedCollapseResult screened_spherical_collapse(const SphericalCollapseModel & model,
                                                               const FifthForce & F,
                                                               double xcollapse,
                                                               double delta_env_lin,
                                                               const ScreenedCollapseOptions & opt) {
                if (not(xcollapse > opt.xini + 1.0) or opt.npts < 10)
                    throw std::invalid_argument("screened_spherical_collapse: need xcollapse > xini + 1, npts >= 10");
                ScreenedCollapseResult r;
                r.xcollapse = xcollapse;
                r.delta_env_ini = delta_env_lin / linear_growth_ratio(model, opt.xini, xcollapse);

                // delta_ini: delta_ini = 0 never collapses, delta_ini = 1 collapses at once
                auto condition = [&](double delta_ini) {
                    return evolve(model, F, delta_ini, r.delta_env_ini, xcollapse, opt, false).xcollapsed - xcollapse;
                };
                if (not(condition(1.0) < 0.0))
                    throw std::runtime_error("screened_spherical_collapse: no collapse before xcollapse = " +
                                             std::to_string(xcollapse) + " even for delta_ini = 1");
                r.delta_ini = FML::MATH::find_root_bisection(condition, {0.0, 1.0}, opt.epsilon_delta);

                // The solution (the halo is just short of collapsing at xcollapse)
                const auto s = evolve(model, F, r.delta_ini, r.delta_env_ini, xcollapse, opt, true);
                r.deltac = s.delta_lin.back();
                // Splines up to the last point before the halo is frozen (beyond delta_infinity)
                size_t n = s.x.size();
                while (n > 4 and s.delta[n - 1] > opt.delta_infinity)
                    n--;
                const DVector x(s.x.begin(), s.x.begin() + n);
                const DVector d(s.delta.begin(), s.delta.begin() + n), dp(s.delta_prime.begin(), s.delta_prime.begin() + n),
                    de(s.delta_env.begin(), s.delta_env.begin() + n);
                r.delta_of_x = Spline(x, d, "delta_halo(loga)");
                r.delta_env_of_x = Spline(x, de, "delta_env(loga)");
                const Spline delta_prime(x, dp, "ddelta/dloga(loga)");
                const double xmax = x.back();

                // Turnaround: d log r / d log a = 1 - delta' / (3 (1 + delta)) = 0
                auto turnaround = [&](double xx) { return 1.0 - delta_prime(xx) / 3.0 / (1.0 + r.delta_of_x(xx)); };
                r.xta = FML::MATH::find_root_bisection(turnaround, {opt.xini, xmax}, 1e-10);

                // Virialization: 2 E_kin + E_pot = 0 with E_pot from the background acceleration of the smooth
                // components and (1 + delta + F delta) OmegaM(x) of the matter (as SphericalCollapseModel, with
                // mu -> 1 + F for the mass excess)
                auto virial = [&](double xx) {
                    const double delta = r.delta_of_x(xx), dprime = delta_prime(xx);
                    const double Om = model.OmegaMofx(xx);
                    const double smooth = -2.0 * (1.0 + model.logEprimeofx(xx)) - Om;
                    const double f = F(xx, delta, r.delta_env_of_x(xx));
                    const double dlogr = 1.0 - dprime / 3.0 / (1.0 + delta);
                    // Common factor (3/10) (E r)^2 dropped
                    return 2.0 * dlogr * dlogr - (smooth + Om * (1.0 + delta + f * delta));
                };
                // The first sign change after turnaround
                double xs = r.xta;
                const double sign = virial(xs) > 0 ? 1.0 : -1.0;
                while (virial(xs) * sign > 0.0) {
                    if (xs >= xmax)
                        throw std::runtime_error("screened_spherical_collapse: no virialization before the collapse");
                    xs = std::min(xs + 0.005, xmax);
                }
                r.xvir = FML::MATH::find_root_bisection(virial, {r.xta, xs}, 1e-10);
                r.delta_vir = r.delta_of_x(r.xvir);
                r.DeltaVir = (1.0 + r.delta_vir) * std::exp(3.0 * (xcollapse - r.xvir));
                r.F_vir = F(r.xvir, r.delta_vir, r.delta_env_of_x(r.xvir));
                return r;
            }

        } // namespace SPHERICALCOLLAPSE
    } // namespace COSMOLOGY
} // namespace FML
