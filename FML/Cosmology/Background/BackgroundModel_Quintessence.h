#ifndef FML_COSMOLOGY_BACKGROUND_BACKGROUNDMODEL_QUINTESSENCE_H
#define FML_COSMOLOGY_BACKGROUND_BACKGROUNDMODEL_QUINTESSENCE_H

// =====================================================================================================================
// Quintessence: the standard species + a scalar field (ScalarFieldSpecies, Species.h) as the dark energy, GR
//
//   QuintessenceSpecies       v = V0 u(phi): exponential, inverse power law, PNGB, hilltop (QuintessencePotential).
//                             V0 by shooting (Brent on ln V0) so that rho_phi(0) closes the budget. No bare Lambda.
//   make_quintessence(p, QuintessencePotential, phi_ini = 0, dphidlna_ini = 0, r)
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
    // Quintessence
    // -----------------------------------------------------------------------------------------------------------------

    /// The shape u(phi) of a quintessence potential v = V0 u(phi), phi in M_pl:
    ///   exponential        u = exp(-lambda phi)                (lambda = 0: a cosmological constant)
    ///   inverse power law  u = phi^-alpha                      (Ratra-Peebles; tracker; needs phi > 0)
    ///   pngb               u = 1 + cos(phi / f)                (pseudo Nambu-Goldstone boson; minima at (2k+1) pi f)
    ///   hilltop            u = (1 - (|phi| / mu)^p)^2          (top at phi = 0, minima at +-mu)
    struct QuintessencePotential {
        enum class Type { Exponential, InversePowerLaw, PNGB, Hilltop };
        Type type{Type::Exponential};
        double lambda{1.0}, alpha{1.0}, f{1.0}, mu{1.0}, p{2.0};

        static QuintessencePotential exponential(double lambda) {
            QuintessencePotential V;
            V.type = Type::Exponential;
            V.lambda = lambda;
            return V;
        }
        static QuintessencePotential inverse_power_law(double alpha) {
            QuintessencePotential V;
            V.type = Type::InversePowerLaw;
            V.alpha = alpha;
            return V;
        }
        static QuintessencePotential pngb(double f) {
            QuintessencePotential V;
            V.type = Type::PNGB;
            V.f = f;
            return V;
        }
        static QuintessencePotential hilltop(double mu, double p) {
            QuintessencePotential V;
            V.type = Type::Hilltop;
            V.mu = mu;
            V.p = p;
            return V;
        }

        double u(double phi) const {
            switch (type) {
                case Type::Exponential:
                    return std::exp(-lambda * phi);
                case Type::InversePowerLaw:
                    return phi > 0.0 ? std::pow(phi, -alpha) : std::numeric_limits<double>::infinity();
                case Type::PNGB:
                    return 1.0 + std::cos(phi / f);
                default: {
                    const double s = 1.0 - std::pow(std::abs(phi) / mu, p);
                    return s * s;
                }
            }
        }
        double u_phi(double phi) const {
            switch (type) {
                case Type::Exponential:
                    return -lambda * std::exp(-lambda * phi);
                case Type::InversePowerLaw:
                    return phi > 0.0 ? -alpha * std::pow(phi, -alpha - 1.0) : 0.0;
                case Type::PNGB:
                    return -std::sin(phi / f) / f;
                default: {
                    if (phi == 0.0)
                        return 0.0;
                    const double y = std::abs(phi) / mu;
                    return -2.0 * (1.0 - std::pow(y, p)) * p * std::pow(y, p - 1.0) / mu * (phi > 0.0 ? 1.0 : -1.0);
                }
            }
        }
        double u_phiphi(double phi) const {
            switch (type) {
                case Type::Exponential:
                    return lambda * lambda * std::exp(-lambda * phi);
                case Type::InversePowerLaw:
                    return phi > 0.0 ? alpha * (alpha + 1.0) * std::pow(phi, -alpha - 2.0) : 0.0;
                case Type::PNGB:
                    return -std::cos(phi / f) / (f * f);
                default: {
                    // u = s^2, s = 1 - y^p, y = |phi| / mu: u'' = 2 s'^2 + 2 s s''
                    const double y = std::abs(phi) / mu, s = 1.0 - std::pow(y, p);
                    const double ds = p * std::pow(y, p - 1.0) / mu, dds = -p * (p - 1.0) * std::pow(y, p - 2.0) / (mu * mu);
                    return 2.0 * ds * ds + 2.0 * s * dds;
                }
            }
        }
        std::string name() const {
            switch (type) {
                case Type::Exponential:
                    return "exponential";
                case Type::InversePowerLaw:
                    return "inverse_power_law";
                case Type::PNGB:
                    return "pngb";
                default:
                    return "hilltop";
            }
        }
        std::string info() const {
            std::ostringstream s;
            switch (type) {
                case Type::Exponential:
                    s << "V0 exp(-lambda phi), lambda = " << lambda;
                    break;
                case Type::InversePowerLaw:
                    s << "V0 phi^-alpha, alpha = " << alpha;
                    break;
                case Type::PNGB:
                    s << "V0 (1 + cos(phi/f)), f = " << f << " M_pl";
                    break;
                default:
                    s << "V0 (1 - (|phi|/mu)^p)^2, mu = " << mu << " M_pl, p = " << p;
            }
            return s.str();
        }
        void validate() const {
            const std::string msg = "[QuintessencePotential] ";
            if (not std::isfinite(lambda) or not std::isfinite(alpha))
                throw std::invalid_argument(msg + "lambda and alpha must be finite");
            if (type == Type::PNGB and not(f > 0.0))
                throw std::invalid_argument(msg + "pngb needs f > 0");
            if (type == Type::Hilltop and not(mu > 0.0 and p > 0.0))
                throw std::invalid_argument(msg + "hilltop needs mu > 0 and p > 0");
        }
    };

    class QuintessenceSpecies : public ScalarFieldSpecies {
      public:
        /// others: all other species; Omega_phi: the density today it must have; [x_start, x_end]: the range of the
        /// solution (ScalarFieldSpecies::solution_range); phi_ini, dphidlna_ini: the initial conditions at x_start
        QuintessenceSpecies(QuintessencePotential potential,
                            double phi_ini,
                            double dphidlna_ini,
                            SpeciesList others,
                            double Omega_k,
                            double Omega_phi,
                            double x_start,
                            double x_end)
            : ScalarFieldSpecies(std::move(others), Omega_k), V(potential), phi_ini(phi_ini),
              dphi_ini(dphidlna_ini) {
            const std::string msg = "[quintessence] ";
            V.validate();
            if (not(Omega_phi > 0.0))
                throw std::invalid_argument(msg + "Need Omega_phi > 0 today");
            if (not(V.u(phi_ini) > 0.0) or not std::isfinite(V.u(phi_ini)))
                throw std::invalid_argument(msg + "The potential must be positive and finite at phi_ini (" +
                                            V.name() + (V.type == QuintessencePotential::Type::InversePowerLaw ?
                                                            ": needs phi_ini > 0)" :
                                                            ")"));
            // The minimum the field rolls to (PNGB: the nearest in the direction of the slope; hilltop: on its side)
            if (V.type == QuintessencePotential::Type::PNGB)
                min_at = (2.0 * std::floor(phi_ini / (2.0 * M_PI * V.f)) + 1.0) * M_PI * V.f;
            if (V.type == QuintessencePotential::Type::Hilltop)
                min_at = (phi_ini > 0.0 or (phi_ini == 0.0 and dphidlna_ini >= 0.0)) ? V.mu : -V.mu;

            // Shooting on ln V0: ln(rho_phi(0) / Omega_phi), increasing with V0 (while the field does not oscillate)
            auto g = [&](double lnV0) {
                V0 = std::exp(lnV0);
                solve(phi_ini, dphi_ini, x_start, x_end);
                return std::log(rho(0.0) / Omega_phi);
            };
            // rho_phi(0) is not monotonic in V0 once the field oscillates before today: bracket within e^+-24 of the
            // frozen-field value, then Brent
            const std::string nosol = msg + "No potential amplitude V0 gives Omega_phi = " + std::to_string(Omega_phi) +
                                      " today (the field rolls or oscillates too early: a flatter potential or start "
                                      "nearer the top)";
            double lo = std::log(Omega_phi / V.u(phi_ini)) - 1.0, hi = lo + 2.0;
            double glo = g(lo), ghi = g(hi);
            for (int i = 0; i < 12 and glo > 0.0; i++) {
                hi = lo;
                ghi = glo;
                lo -= 2.0;
                glo = g(lo);
            }
            for (int i = 0; i < 12 and ghi < 0.0; i++) {
                lo = hi;
                glo = ghi;
                hi += 2.0;
                ghi = g(hi);
            }
            if (not(glo <= 0.0 and ghi >= 0.0))
                throw std::runtime_error(nosol);
            const double lnV0 = glo == 0.0 ? lo : (ghi == 0.0 ? hi : FML::MATH::find_root(g, lo, hi, 1e-15, 1e-15));
            if (not(std::abs(g(lnV0)) < 1e-10))
                throw std::runtime_error(nosol + " (rho_phi(0) jumps with V0 at the switch to the oscillating fluid)");
        }
        std::string name() const override { return "quintessence"; }
        std::string info() const override {
            std::ostringstream s;
            s << "quintessence: V = " << V.info() << ", V0 = " << V0 << " rho_c0, phi_ini = " << phi_ini
              << " M_pl, dphi/dlna_ini = " << dphi_ini << "; Omega0 = " << rho(0.0) << ", w0 = " << w(0.0)
              << ", phi0 = " << phi(0.0) << " M_pl";
            if (std::isfinite(x_fluid()))
                s << "; a fluid with w = 0 for a > " << std::exp(x_fluid());
            return s.str();
        }

        double v(double phi) const override { return V0 * V.u(phi); }
        double v_phi(double phi) const override { return V0 * V.u_phi(phi); }
        double v_phiphi(double phi) const override { return V0 * V.u_phiphi(phi); }
        bool has_minimum() const override {
            return V.type == QuintessencePotential::Type::PNGB or V.type == QuintessencePotential::Type::Hilltop;
        }
        double phi_min() const override { return min_at; }
        /// Both minima are quadratic
        double w_oscillating() const override { return 0.0; }

        double V0_value() const { return V0; }
        const QuintessencePotential & potential() const { return V; }

      private:
        QuintessencePotential V;
        double phi_ini, dphi_ini;
        double V0{1.0}, min_at{0.0};
    };

    /// Standard species + quintessence (V0 from E(0) = 1), GR. phi_ini [M_pl] and dphidlna_ini at the start of the
    /// solution, a = min(1e-10, a_min) (dphidlna_ini = 0: frozen by Hubble friction)
    inline ComponentBackgroundPtr make_quintessence(const BackgroundParameters & p,
                                                    const QuintessencePotential & V,
                                                    double phi_ini = 0.0,
                                                    double dphidlna_ini = 0.0,
                                                    BackgroundRange r = {}) {
        auto s = standard_species(p);
        const double Omega_phi = 1.0 - p.Omega_k - density_today(s);
        const auto [x0, x1] = ScalarFieldSpecies::solution_range(r);
        s.push_back(std::make_shared<QuintessenceSpecies>(V, phi_ini, dphidlna_ini, s, p.Omega_k, Omega_phi, x0, x1));
        return std::make_shared<const ComponentBackground>(
            "quintessence", p, s, std::make_shared<GeneralRelativity>(), r);
    }

} // namespace FML::COSMOLOGY::BACKGROUND

#endif
