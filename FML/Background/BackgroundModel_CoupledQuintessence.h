#ifndef FML_BACKGROUND_BACKGROUNDMODEL_COUPLEDQUINTESSENCE_H
#define FML_BACKGROUND_BACKGROUNDMODEL_COUPLEDQUINTESSENCE_H

// =====================================================================================================================
// Coupled quintessence (Wetterich 1995, Amendola 2000), the background: a scalar field phi (in reduced Planck masses)
// with V = 3 H0^2 M_pl^2 V0 exp(-lambda phi) that exchanges energy with the cold dark matter, whose particles have the
// mass m(phi) ~ exp(-beta phi). With x = ln a, densities in units of rho_c0 and v = V/(3 H0^2 M_pl^2):
//   rho_c = C exp(-3x - beta phi),     rho_c' = -(3 + beta phi_x) rho_c
//   phi_xx = -(3 + dlnE/dx) phi_x - 3 v_phi/E^2 + 3 beta rho_c/E^2
//   E^2 = (rho_o + rho_c + v) / (1 - phi_x^2/6)          (rho_o: all other species + Omega_k e^{-2x})
//   rho_phi = E^2 phi_x^2/6 + v,  p_phi = E^2 phi_x^2/6 - v,  rho_phi' = -E^2 phi_x^2 + beta phi_x rho_c
//   dlnE/dx = (rho_o' - 3 rho_c - E^2 phi_x^2) / (2 E^2)       (the exchange cancels in the total)
// beta is Amendola's sqrt(2/3) beta. The field starts at rest at x = min(-25, x_min) (in the radiation era); V0 and
// C are found by a Newton iteration so that rho_c(0) = Omega_cdm and rho_phi(0) = 1 - Omega_k - (the other species
// today).
// beta = 0 is uncoupled quintessence, beta = lambda = 0 is LCDM. The two species (CoupledCDMSpecies: Component::CDM,
// CoupledFieldSpecies: Component::DarkEnergy) share the solution (CoupledDarkSector); the field equations are GR.
// =====================================================================================================================

#include <FML/Background/BackgroundParameters.h>
#include <FML/Background/ComponentBackground.h>
#include <FML/Background/FieldEquation_GeneralRelativity.h>
#include <FML/Background/Species.h>
#include <FML/ODESolver/ODESolver.h>
#include <FML/Spline/Spline.h>

#include <algorithm>
#include <cmath>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace FML::COSMOLOGY::BACKGROUND {

    /// The background solution of the coupled field and CDM
    class CoupledDarkSector {
      public:
        /// others: all species except CDM and the field. Solves on [min(-25, r.x_min), max(r.x_max, 0) + 1]
        CoupledDarkSector(double lambda,
                          double beta,
                          ComponentBackground::SpeciesList others,
                          double Omega_k,
                          double Omega_c,
                          double Omega_phi,
                          BackgroundRange r = {})
            : lambda(lambda), beta(beta), C(Omega_c), Omega_k(Omega_k), others(std::move(others)) {
            const std::string msg = "[CoupledQuintessence] ";
            if (not(Omega_phi > 0.0) or not(Omega_c > 0.0))
                throw std::invalid_argument(msg + "Need Omega_c > 0 and Omega_phi > 0");
            x_ini = std::min(-25.0, r.x_min);
            x_fin = std::max(r.x_max, 0.0) + 1.0;
            // Newton iteration in (ln V0, ln C) for rho_phi(0) = Omega_phi and rho_c(0) = Omega_c (the Jacobian by
            // finite differences; rho_c(0) depends on phi(0), which depends on both)
            const auto residuals = [&](double u, double w, double & r1, double & r2) {
                V0 = std::exp(u);
                C = std::exp(w);
                solve();
                r1 = std::log(rho_phi(0.0) / Omega_phi);
                r2 = std::log(rho_c(0.0) / Omega_c);
            };
            double u = std::log(Omega_phi), w = std::log(Omega_c);
            for (int it = 0;; it++) {
                double r1, r2;
                residuals(u, w, r1, r2);
                if (std::abs(r1) < 1e-13 and std::abs(r2) < 1e-13)
                    break;
                if (it > 50 or not std::isfinite(r1) or not std::isfinite(r2))
                    throw std::runtime_error(msg + "The background iteration did not converge");
                const double h = 1e-6;
                double a11, a21, a12, a22;
                residuals(u + h, w, a11, a21);
                residuals(u, w + h, a12, a22);
                a11 = (a11 - r1) / h;
                a21 = (a21 - r2) / h;
                a12 = (a12 - r1) / h;
                a22 = (a22 - r2) / h;
                const double det = a11 * a22 - a12 * a21;
                double du = -(a22 * r1 - a12 * r2) / det, dw = -(a11 * r2 - a21 * r1) / det;
                const double step = std::max({std::abs(du), std::abs(dw), 1.0});
                u += du / step;
                w += dw / step;
            }
        }

        double phi(double x) const { return x < x_ini ? 0.0 : phi_spline(x); }
        double dphidx(double x) const { return x < x_ini ? 0.0 : dphi_spline(x); }
        double v(double f) const { return V0 * std::exp(-lambda * f); }
        double v_phi(double f) const { return -lambda * v(f); }
        double v_phiphi(double f) const { return lambda * lambda * v(f); }
        /// d^2phi/dx^2 from the field equation, and dlnE/dx
        double ddphidx(double x) const {
            const double f = phi(x), df = dphidx(x);
            const double e2 = E2(x, f, df), rc = C * std::exp(-3.0 * x - beta * f);
            return -(3.0 + dlnE(x, f, df)) * df + 3.0 * lambda * v(f) / e2 + 3.0 * beta * rc / e2;
        }
        double dlnEdx(double x) const { return dlnE(x, phi(x), dphidx(x)); }
        double rho_c(double x) const { return C * std::exp(-3.0 * x - beta * phi(x)); }
        double drho_c(double x) const { return -(3.0 + beta * dphidx(x)) * rho_c(x); }
        double E2(double x) const { return E2(x, phi(x), dphidx(x)); }
        double rho_phi(double x) const {
            const double f = phi(x), df = dphidx(x);
            return E2(x, f, df) * df * df / 6.0 + v(f);
        }
        double p_phi(double x) const {
            const double f = phi(x), df = dphidx(x);
            return E2(x, f, df) * df * df / 6.0 - v(f);
        }
        double drho_phi(double x) const {
            const double df = dphidx(x);
            return -E2(x) * df * df + beta * df * rho_c(x);
        }
        double x_start() const { return x_ini; }
        double x_end() const { return x_fin; }

        const double lambda, beta;
        double V0{0.0}, C; ///< v = V0 e^{-lambda phi}, rho_c = C e^{-3x - beta phi}

      private:
        double Omega_k;
        ComponentBackground::SpeciesList others;
        double x_ini{-25.0}, x_fin{10.0};
        FML::INTERPOLATION::SPLINE::GSLSpline phi_spline, dphi_spline;

        double rho_o(double x) const {
            double r = Omega_k * std::exp(-2.0 * x);
            for (const auto & s : others)
                r += s->rho(x);
            return r;
        }
        double drho_o(double x) const {
            double r = -2.0 * Omega_k * std::exp(-2.0 * x);
            for (const auto & s : others)
                r += s->drhodx(x);
            return r;
        }
        double E2(double x, double f, double df) const {
            return (rho_o(x) + C * std::exp(-3.0 * x - beta * f) + v(f)) / (1.0 - df * df / 6.0);
        }
        double dlnE(double x, double f, double df) const {
            const double e2 = E2(x, f, df), rc = C * std::exp(-3.0 * x - beta * f);
            return (drho_o(x) - 3.0 * rc - e2 * df * df) / (2.0 * e2);
        }
        void solve() {
            FML::SOLVERS::ODESOLVER::ODEFunction rhs = [&](double x, const double * y, double * dydx) {
                const double f = y[0], df = y[1];
                const double e2 = E2(x, f, df), rc = C * std::exp(-3.0 * x - beta * f);
                dydx[0] = df;
                dydx[1] = -(3.0 + dlnE(x, f, df)) * df + 3.0 * lambda * v(f) / e2 + 3.0 * beta * rc / e2;
                return GSL_SUCCESS;
            };
            const int n = int(std::ceil((x_fin - x_ini) * 400.0)) + 1;
            std::vector<double> xs(n);
            for (int i = 0; i < n; i++)
                xs[i] = x_ini + (x_fin - x_ini) * i / double(n - 1);
            FML::SOLVERS::ODESOLVER::ODESolver ode(1e-4, 1e-14, 1e-12);
            ode.solve(rhs, xs, {0.0, 0.0}, gsl_odeiv2_step_rk8pd);
            phi_spline.create(xs, ode.get_data_by_component(0), "CoupledQuintessence phi(x)");
            dphi_spline.create(xs, ode.get_data_by_component(1), "CoupledQuintessence dphi/dx(x)");
        }
    };

    /// The CDM of coupled quintessence, rho_c = C e^{-3x - beta phi}
    class CoupledCDMSpecies : public Species {
      public:
        explicit CoupledCDMSpecies(std::shared_ptr<const CoupledDarkSector> d) : d(std::move(d)) {}
        std::string name() const override { return "cdm (coupled)"; }
        Component component() const override { return Component::CDM; }
        double rho(double x) const override { return d->rho_c(x); }
        double p(double) const override { return 0.0; }
        double drhodx(double x) const override { return d->drho_c(x); }
        /// rho_c'' = [(3 + beta phi_x)^2 - beta phi_xx] rho_c
        double d2rhodx2(double x) const override {
            const double a = 3.0 + d->beta * d->dphidx(x);
            return (a * a - d->beta * d->ddphidx(x)) * d->rho_c(x);
        }

      private:
        std::shared_ptr<const CoupledDarkSector> d;
    };

    /// The scalar field of coupled quintessence
    class CoupledFieldSpecies : public Species {
      public:
        explicit CoupledFieldSpecies(std::shared_ptr<const CoupledDarkSector> d) : d(std::move(d)) {}
        std::string name() const override { return "coupled_quintessence"; }
        Component component() const override { return Component::DarkEnergy; }
        double rho(double x) const override { return d->rho_phi(x); }
        double p(double x) const override { return d->p_phi(x); }
        double drhodx(double x) const override { return d->drho_phi(x); }
        /// rho_phi'' = -(E^2' phi_x^2 + 2 E^2 phi_x phi_xx) + beta (phi_xx rho_c + phi_x rho_c')
        double d2rhodx2(double x) const override {
            const double df = d->dphidx(x), ddf = d->ddphidx(x), e2 = d->E2(x), de2 = 2.0 * d->dlnEdx(x) * e2;
            const double rc = d->rho_c(x), drc = d->drho_c(x);
            return -(de2 * df * df + 2.0 * e2 * df * ddf) + d->beta * (ddf * rc + df * drc);
        }
        std::string info() const override {
            std::ostringstream s;
            s << "coupled quintessence: lambda = " << d->lambda << ", beta = " << d->beta << ", V0 = " << d->V0
              << ", Omega0 = " << rho(0.0);
            return s.str();
        }
        const CoupledDarkSector & sector() const { return *d; }

      private:
        std::shared_ptr<const CoupledDarkSector> d;
    };

    /// Coupled quintessence: the standard species with coupled CDM (Omega_cdm today) + the field (closure), GR
    inline ComponentBackgroundPtr
    make_coupled_quintessence(const BackgroundParameters & p, double lambda, double beta, BackgroundRange r = {}) {
        const auto standard = standard_species(p);
        ComponentBackground::SpeciesList others;
        for (const auto & s : standard)
            if (s->component() != Component::CDM)
                others.push_back(s);
        const double Omega_phi = 1.0 - p.Omega_k - density_today(standard);
        auto d = std::make_shared<const CoupledDarkSector>(lambda, beta, others, p.Omega_k, p.Omega_cdm, Omega_phi, r);
        ComponentBackground::SpeciesList s;
        for (const auto & sp : standard)
            s.push_back(sp->component() == Component::CDM ? std::make_shared<CoupledCDMSpecies>(d) : sp);
        s.push_back(std::make_shared<CoupledFieldSpecies>(d));
        return std::make_shared<const ComponentBackground>("coupled_quintessence",
                                                           p,
                                                           s,
                                                           std::make_shared<GeneralRelativity>(),
                                                           r);
    }

} // namespace FML::COSMOLOGY::BACKGROUND

#endif
