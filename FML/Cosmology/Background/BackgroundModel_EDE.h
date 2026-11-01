#ifndef FML_COSMOLOGY_BACKGROUND_BACKGROUNDMODEL_EDE_H
#define FML_COSMOLOGY_BACKGROUND_BACKGROUNDMODEL_EDE_H

// =====================================================================================================================
// Early dark energy: the standard species + an axion-like scalar field (ScalarFieldSpecies, Species.h) + Lambda, GR
//
//   EarlyDarkEnergySpecies    v = mu (1 - cos(phi/f))^n (Poulin et al. 2019, Smith et al. 2020), from rest at
//                             theta_i = phi_i / f, given by (f_EDE, log10 z_c, theta_i, n): f_EDE the maximum of
//                             rho_EDE / rho_tot, z_c where it is reached. (ln mu, ln f) by a damped Newton iteration
//                             on (ln f_max, x_peak), with a bare Lambda from closure (iterated). Needs n > 1 (and
//                             n > 2 for z_c before matter-radiation equality). SimulationCode's zc is 10^log10_zc.
//   make_ede(p, f_EDE, log10_zc, theta_i, n, r)
// =====================================================================================================================

#include <FML/Cosmology/Background/BackgroundParameters.h>
#include <FML/Cosmology/Background/ComponentBackground.h>
#include <FML/Cosmology/Background/FieldEquation_GeneralRelativity.h>
#include <FML/Cosmology/Background/Species.h>
#include <FML/Math/NumericalMethods.h>
#include <FML/ODESolver/ODESolver.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace FML::COSMOLOGY::BACKGROUND {

    // -----------------------------------------------------------------------------------------------------------------
    // Early dark energy
    // -----------------------------------------------------------------------------------------------------------------

    class EarlyDarkEnergySpecies : public ScalarFieldSpecies {
      public:
        /// others: all other species (not the cosmological constant: it is solved for here and given by
        /// Omega_Lambda_bare()); [x_start, x_end]: the range of the solution
        EarlyDarkEnergySpecies(double f_EDE,
                               double log10_zc,
                               double theta_i,
                               double n,
                               SpeciesList others,
                               double Omega_k,
                               double x_start,
                               double x_end)
            : ScalarFieldSpecies(std::move(others), Omega_k), fEDE(f_EDE), x_c(-std::log1p(std::pow(10.0, log10_zc))),
              theta_i(theta_i), n(n) {
            const std::string msg = "[early dark energy] ";
            if (not(fEDE > 0.0 and fEDE < 1.0))
                throw std::invalid_argument(msg + "Need 0 < f_EDE < 1");
            if (not(theta_i > 0.0 and theta_i < M_PI))
                throw std::invalid_argument(msg + "Need 0 < theta_i < pi");
            // The oscillating field has w = (n-1)/(n+1): for n <= 1 its fraction never decays, for 1 < n <= 2 it grows
            // in the radiation era and peaks only after equality (z_c in the radiation era then has no solution and
            // the shooting fails). The Einstein-Boltzmann solver requires n > 2, SimulationCode n > 1 (as here)
            if (not(n > 1.0))
                throw std::invalid_argument(msg + "Need n > 1 (the f_EDE, z_c parametrization)");
            if (not(x_c > -18.0 and x_c < -1.0))
                throw std::invalid_argument(msg + "Need 0.5 < log10(z_c) < 7.5");
            if (not(x_start < x_c - 3.0 and x_end >= 0.0))
                throw std::invalid_argument(msg + "The solution must start well before z_c and reach today");

            const double rho_o0 = rho_others(0.0); // with Omega_L = 0
            OmegaL = 1.0 - rho_o0;

            // Damped Newton on (ln mu, ln f) for (ln f_max, x_peak) = (ln f_EDE, x_c), on the solution up to x_c + 3;
            // then Omega_L from closure, and again until Omega_L converges
            const double u_i = 1.0 - std::cos(theta_i), s_i = std::sin(theta_i);
            for (int closure = 0; closure < 5; closure++) {
                auto residual = [&](double lnmu, double lnf, double & g1, double & g2) {
                    mu_ = std::exp(lnmu);
                    f_ = std::exp(lnf);
                    solve(f_ * theta_i, 0.0, x_start, x_c + 3.0);
                    double fm, xp;
                    peak(fm, xp);
                    g1 = std::log(fm / fEDE);
                    g2 = xp - x_c;
                };
                // Initial guess: the frozen field has rho = mu u_i^n, and it starts rolling when the force term of
                // theta_xx = -3 mu n u^(n-1) sin(theta) / (f^2 E^2) is ~ 3 theta_i
                const double E2c = rho_others(x_c);
                double lnmu = std::log(fEDE * E2c / std::pow(u_i, n));
                double lnf = 0.5 * std::log(std::exp(lnmu) * n * std::pow(u_i, n - 1.0) * s_i / (theta_i * E2c));
                double g1, g2;
                residual(lnmu, lnf, g1, g2);
                for (int it = 0;; it++) {
                    if (std::abs(g1) < 1e-10 and std::abs(g2) < 1e-9)
                        break;
                    if (it > 40)
                        throw std::runtime_error(msg + "No solution for f_EDE and z_c (the shooting failed)");
                    const double h = 1e-5;
                    double a1, a2, b1, b2;
                    residual(lnmu + h, lnf, a1, a2);
                    residual(lnmu, lnf + h, b1, b2);
                    const double J11 = (a1 - g1) / h, J21 = (a2 - g2) / h, J12 = (b1 - g1) / h, J22 = (b2 - g2) / h;
                    const double det = J11 * J22 - J12 * J21;
                    const double d1 = -(J22 * g1 - J12 * g2) / det, d2 = -(-J21 * g1 + J11 * g2) / det;
                    const double norm = std::hypot(g1, g2);
                    double t = std::min(1.0, 1.0 / std::max(std::abs(d1), std::abs(d2))), n1 = 0.0, n2 = 0.0;
                    for (int ls = 0; ls < 30; ls++, t *= 0.5) {
                        residual(lnmu + t * d1, lnf + t * d2, n1, n2);
                        if (std::isfinite(n1) and std::isfinite(n2) and std::hypot(n1, n2) < norm)
                            break;
                    }
                    lnmu += t * d1;
                    lnf += t * d2;
                    g1 = n1;
                    g2 = n2;
                }
                mu_ = std::exp(lnmu);
                f_ = std::exp(lnf);
                solve(f_ * theta_i, 0.0, x_start, x_end);
                const double OmegaL_new = 1.0 - rho_o0 - rho(0.0);
                const bool done = std::abs(OmegaL_new - OmegaL) < 1e-15;
                OmegaL = OmegaL_new;
                if (done)
                    break;
            }
            solve(f_ * theta_i, 0.0, x_start, x_end);
        }
        std::string name() const override { return "early_dark_energy"; }
        std::string info() const override {
            double fm, xp;
            peak(fm, xp);
            std::ostringstream s;
            s << "early dark energy: V = m^2 f^2 (1 - cos(phi/f))^n, n = " << n << ", theta_i = " << theta_i
              << "; f_EDE = " << fm << " at z_c = " << std::expm1(-xp) << "; f = " << f_
              << " M_pl, m = " << mass_over_H0() << " H0, Omega_Lambda = " << OmegaL;
            if (std::isfinite(x_fluid()))
                s << "; a fluid with w = " << w_oscillating() << " for a > " << std::exp(x_fluid());
            return s.str();
        }

        double v(double phi) const override { return mu_ * std::pow(1.0 - std::cos(phi / f_), n); }
        double v_phi(double phi) const override {
            const double th = phi / f_;
            return mu_ * n * std::pow(1.0 - std::cos(th), n - 1.0) * std::sin(th) / f_;
        }
        double v_phiphi(double phi) const override {
            const double th = phi / f_, u = 1.0 - std::cos(th), s = std::sin(th);
            double r = std::pow(u, n - 1.0) * std::cos(th);
            if (n != 1.0)
                r += (n - 1.0) * std::pow(u, n - 2.0) * s * s;
            return mu_ * n * r / (f_ * f_);
        }
        bool has_minimum() const override { return true; }
        double phi_min() const override { return 0.0; }
        double w_oscillating() const override { return (n - 1.0) / (n + 1.0); }

        /// mu = m^2 f^2 / (3 H0^2 M_pl^2), f [M_pl], m / H0 = sqrt(3 mu) / f
        double mu() const { return mu_; }
        double f() const { return f_; }
        double mass_over_H0() const { return std::sqrt(3.0 * mu_) / f_; }
        /// The maximum of rho_EDE / rho_tot and the x where it is reached (from the stored solution)
        void fraction_peak(double & f_max, double & x_peak) const { peak(f_max, x_peak); }

      private:
        double fEDE, x_c, theta_i, n;
        double mu_{0.0}, f_{1.0};
    };

    /// Standard species + early dark energy + Lambda (from E(0) = 1), GR
    inline ComponentBackgroundPtr make_ede(const BackgroundParameters & p,
                                           double f_EDE,
                                           double log10_zc,
                                           double theta_i,
                                           double n,
                                           BackgroundRange r = {}) {
        auto s = standard_species(p);
        const auto [x0, x1] = ScalarFieldSpecies::solution_range(r);
        auto e = std::make_shared<EarlyDarkEnergySpecies>(f_EDE, log10_zc, theta_i, n, s, p.Omega_k, x0, x1);
        s.push_back(e);
        s.push_back(std::make_shared<PowerLawSpecies>("lambda", Component::DarkEnergy, e->Omega_Lambda_bare(), 0.0));
        return std::make_shared<const ComponentBackground>("ede", p, s, std::make_shared<GeneralRelativity>(), r);
    }

} // namespace FML::COSMOLOGY::BACKGROUND

#endif
