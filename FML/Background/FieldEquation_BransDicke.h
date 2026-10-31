#ifndef FML_BACKGROUND_FIELDEQUATION_BRANSDICKE_H
#define FML_BACKGROUND_FIELDEQUATION_BRANSDICKE_H

// =====================================================================================================================
// Jordan-Brans-Dicke gravity with a cosmological constant (the background; Jordan frame):
//   S = (1/16 pi G) int sqrt(-g) [phi R - (omega/phi) (d phi)^2] + S_matter
// G is the "bare" constant of the action. The coupling in the Friedmann equation is G/phi, and the constant measured
// in a Cavendish experiment is G_eff = G phi_* / phi with phi_* = (4 + 2 omega)/(3 + 2 omega).
//
// Background (x = ln a, F = phi, F_x = dphi/dx, g = G/G_*, densities rho, p in units of rho_c0 = 3 H0^2/(8 pi G_*),
// see the density parameter definition below):
//   E^2 (F + F_x - omega F_x^2/(6 F)) = g rho + F Omega_k e^{-2x}                        (Friedmann)
//   F_xx = -(3 + dlnE/dx) F_x + 3 g (rho - 3p) / ((3 + 2 omega) E^2)                      (box phi = 8 pi G T/(3+2w))
// Writing N = g rho + F Omega_k e^{-2x}, D = F + F_x - omega F_x^2/(6F), y = F_x/F, q = 1 - omega y/3 and
// S = 3 g (rho - 3p)/(3 + 2 omega), the derivative of the Friedmann equation with F_xx eliminated gives
//   dlnE/dx [2 D - q F_x] = (N'/N) D - F_x - q (S/E^2 - 3 F_x) - omega F_x y^2/6.
// The field starts at rest deep in the radiation era (where rho - 3p ~ 0, the decaying mode ~ 1/a dropped). Its
// initial value and Omega_Lambda are found by iteration so that phi(0) = phi_today and E(0) = 1:
//   phi_today = phi_* / GeffG_today        (GeffG_today = G_eff/G at a = 1; 1: the Newtonian limit is G today)
//   Omega_Lambda = (D(0) - phi_today Omega_k)/g - (all other species today)
// The scalar field has no species: Omega_i(x) = rho_i/E^2 of the matter species and Lambda, their sum is not 1.
// omega -> infinity is GR.
//
// Density parameter definition (SimulationCode's cosmology_JBD_density_parameter_definition): which constant G_* the
// density parameters Omega_i = 8 pi G_* rho_i / (3 H0^2) use, and which of G and G/phi_* the radiation densities
// computed from T_CMB (and the massive neutrinos from their masses) are taken to have used for the SI value of G:
//   hi-class (0, default)  G_* = G         G_SI = G        (the Einstein-Boltzmann solver; with GeffG_today = 1 the
//                                                          Cavendish constant today is G_SI)
//   Gtoday   (1)           G_* = G/phi(0)  G_SI = G/phi_*  (flat: sum of the Omega_i + Omega_phi = 1 today)
//   Gnewton  (2)           G_* = G/phi_*   G_SI = G/phi_*  (SimulationCode's "G_N"; = Gtoday for GeffG_today = 1)
//   Gbare    (3)           G_* = G         G_SI = G/phi_*
// So g = G/G_* = 1, phi_today, phi_*, 1 and the radiation and massive neutrino densities are multiplied by phi_*/g
// (1 for hi-class). Gtoday, Gnewton and Gbare describe the same physics with differently normalised Omega_i (Omega_b,
// Omega_cdm are read in the chosen definition); hi-class differs from them by the factor phi_* in the radiation.
// (SimulationCode instead keeps its input Omega_mnu and rescales the neutrino mass; here the masses are physical.)
// =====================================================================================================================

#include <FML/Background/BackgroundParameters.h>
#include <FML/Background/ComponentBackground.h>
#include <FML/Background/FieldEquations.h>
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

    enum class BransDickeDensityDefinition { HiClass = 0, Gtoday = 1, Gnewton = 2, Gbare = 3 };

    class BransDickeFieldEquations : public FieldEquations {
      public:
        /// species: all species except Lambda (in the units of the density definition). Solves the field on
        /// [min(-25, r.x_min), max(r.x_max, 0) + 1] and sets Omega_Lambda (Omega_Lambda()). g = G/G_*
        BransDickeFieldEquations(double omega,
                                 double phi_today,
                                 double g,
                                 ComponentBackground::SpeciesList species,
                                 double Omega_k,
                                 BackgroundRange r = {})
            : omega(omega), phi0(phi_today), g(g), Omega_k(Omega_k), list(std::move(species)) {
            const std::string msg = "[BransDicke] ";
            if (not(omega > -1.5) or not std::isfinite(omega))
                throw std::invalid_argument(msg + "Need omega > -3/2");
            if (not(phi0 > 0.0) or not(g > 0.0))
                throw std::invalid_argument(msg + "Need phi_today > 0 and g > 0");
            x_ini = std::min(-25.0, r.x_min);
            x_fin = std::max(r.x_max, 0.0) + 1.0;
            const double rho_o = sum_today();
            // Iterate: F_i so that F(0) = phi_today, Omega_Lambda so that E(0) = 1
            double F_i = phi0;
            OmegaL = (1.0 - phi0 * Omega_k) / g - rho_o;
            for (int it = 0;; it++) {
                const double Fe = solve(F_i);
                const double dF = dF_spline(0.0);
                const double OmegaL_new = (Fe + dF - omega * dF * dF / (6.0 * Fe) - phi0 * Omega_k) / g - rho_o;
                const bool done = std::abs(Fe / phi0 - 1.0) < 1e-14 and std::abs(OmegaL_new - OmegaL) < 1e-14;
                F_i *= phi0 / Fe;
                OmegaL = OmegaL_new;
                if (done)
                    break;
                if (it > 200)
                    throw std::runtime_error(msg + "The background iteration did not converge");
            }
            solve(F_i);
        }

        std::string name() const override { return "jbd"; }
        std::string info() const override {
            std::ostringstream s;
            s << "Jordan-Brans-Dicke (omega = " << omega << ", phi_ini = " << phi(x_ini) << ", phi_0 = " << phi(0.0)
              << ", phi_* = " << phi_star() << ", G/G_* = " << g << ", Omega_Lambda = " << OmegaL << ")";
            return s.str();
        }
        void init(const ComponentBackground & bg) override {
            if (bg.parameters().Omega_k != Omega_k)
                throw std::invalid_argument("[BransDicke] Omega_k of the field equations and the background differ");
            if (bg.x_min() < x_ini or bg.x_max() > x_fin)
                throw std::invalid_argument("[BransDicke] The range is outside the field solution");
        }
        double E2(double x, const ComponentBackground & bg) const override {
            return E2_of(x, phi(x), dphidx(x), bg.rho_total(x));
        }
        double dE2dx(double x, const ComponentBackground & bg) const override {
            double p = 0.0;
            for (const auto & s : bg.species())
                p += s->p(x);
            const double F = phi(x), dF = dphidx(x), rho = bg.rho_total(x);
            return 2.0 * dlnE(x, F, dF, rho, bg.drho_total(x), p) * E2_of(x, F, dF, rho);
        }

        /// The scalar field and dphi/dx (frozen before the start of the solution)
        double phi(double x) const { return F_spline(std::max(x, x_ini)); }
        double dphidx(double x) const { return x < x_ini ? 0.0 : dF_spline(x); }
        double phi_star() const { return (4.0 + 2.0 * omega) / (3.0 + 2.0 * omega); }
        /// G_eff / G_eff(today) = phi(0)/phi(x): the Cavendish constant relative to today
        double Geff_over_Geff_today(double x) const { return phi0 / phi(x); }
        double Omega_Lambda() const { return OmegaL; }
        double get_omega() const { return omega; }
        double G_over_Gstar() const { return g; }

      private:
        double omega, phi0, g, Omega_k, OmegaL{0.0};
        ComponentBackground::SpeciesList list;
        double x_ini{-25.0}, x_fin{10.0};
        FML::INTERPOLATION::SPLINE::GSLSpline F_spline, dF_spline;

        double sum_today() const {
            double r = 0.0;
            for (const auto & s : list)
                r += s->rho(0.0);
            return r;
        }
        /// rho, rho', p of the species and Lambda
        void sums(double x, double & rho, double & drho, double & p) const {
            rho = OmegaL;
            drho = 0.0;
            p = -OmegaL;
            for (const auto & s : list) {
                rho += s->rho(x);
                drho += s->drhodx(x);
                p += s->p(x);
            }
        }
        double E2_of(double x, double F, double dF, double rho) const {
            return (g * rho + F * Omega_k * std::exp(-2.0 * x)) / (F + dF - omega * dF * dF / (6.0 * F));
        }
        /// dlnE/dx given phi, dphi/dx and the matter (see the header)
        double dlnE(double x, double F, double dF, double rho, double drho, double p) const {
            const double k = Omega_k * std::exp(-2.0 * x);
            const double N = g * rho + F * k, dN = g * drho + k * (dF - 2.0 * F);
            const double y = dF / F, D = F + dF - omega * dF * y / 6.0, E2 = N / D;
            const double S = 3.0 * g * (rho - 3.0 * p) / (3.0 + 2.0 * omega), q = 1.0 - omega * y / 3.0;
            return ((dN / N) * D - dF - q * (S / E2 - 3.0 * dF) - omega * dF * y * y / 6.0) / (2.0 * D - q * dF);
        }
        /// Integrate from rest at F_i; sets the splines and returns F(0)
        double solve(double F_i) {
            FML::SOLVERS::ODESOLVER::ODEFunction rhs = [&](double x, const double * Y, double * dY) {
                double rho, drho, p;
                sums(x, rho, drho, p);
                const double F = Y[0], dF = Y[1];
                const double E2 = E2_of(x, F, dF, rho), L = dlnE(x, F, dF, rho, drho, p);
                dY[0] = dF;
                dY[1] = -(3.0 + L) * dF + 3.0 * g * (rho - 3.0 * p) / ((3.0 + 2.0 * omega) * E2);
                return GSL_SUCCESS;
            };
            const int n = int(std::ceil((x_fin - x_ini) * 400.0)) + 1;
            std::vector<double> xs(n);
            for (int i = 0; i < n; i++)
                xs[i] = x_ini + (x_fin - x_ini) * i / double(n - 1);
            FML::SOLVERS::ODESOLVER::ODESolver ode(1e-4, 1e-15, 1e-12);
            ode.solve(rhs, xs, {F_i, 0.0}, gsl_odeiv2_step_rk8pd);
            F_spline.create(xs, ode.get_data_by_component(0), "BransDicke phi(x)");
            dF_spline.create(xs, ode.get_data_by_component(1), "BransDicke dphi/dx(x)");
            return F_spline(0.0);
        }
    };

} // namespace FML::COSMOLOGY::BACKGROUND

#endif
