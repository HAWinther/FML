#ifndef FML_COSMOLOGY_BACKGROUND_BACKGROUNDMODEL_CUBICGALILEON_H
#define FML_COSMOLOGY_BACKGROUND_BACKGROUNDMODEL_CUBICGALILEON_H

// =====================================================================================================================
// Cubic Galileon (Deffayet, Esposito-Farese & Vikman 2009; Barreira et al. 2013) on its tracker solution, flat, without
// a cosmological constant, the background:
//   S = int sqrt(-g) [ R/(2 kappa) + c2 X - g3 X box phi ] + S_m,   X = -(1/2)(d phi)^2,  c2 = -1
// On the tracker H dphi/dt = xi is constant (an attractor), so rho_phi = xi^2/(2 kappa H^2) and the expansion has no
// free parameter beyond those of LCDM (rho: all other species, units of rho_c0):
//   E^2 = [rho + sqrt(rho^2 + 4 Omega_phi)] / 2,   Omega_phi = 1 - (all other species today)
//   rho_phi = E^2 - rho = 2 Omega_phi / (rho + s),  s = sqrt(rho^2 + 4 Omega_phi)    (no cancellation for rho >> 1)
//   rho_phi' = -rho' rho_phi / s,  p_phi = -rho_phi - rho_phi'/3
// (xi = sqrt(6 Omega_phi) H0^2, g3 = 1/(3 xi), kappa = 1). The scalar is a species (CubicGalileonSpecies) and the
// Friedmann equation that of GR: E^2 = rho + rho_phi is the tracker expansion above.
// =====================================================================================================================

#include <FML/Cosmology/Background/BackgroundParameters.h>
#include <FML/Cosmology/Background/ComponentBackground.h>
#include <FML/Cosmology/Background/FieldEquation_GeneralRelativity.h>
#include <FML/Cosmology/Background/Species.h>

#include <cmath>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>

namespace FML::COSMOLOGY::BACKGROUND {

    class CubicGalileonSpecies : public Species {
      public:
        /// others: all other species (flat universe); Omega_phi = the Galileon today
        CubicGalileonSpecies(ComponentBackground::SpeciesList others, double Omega_phi)
            : others(std::move(others)), Om(Omega_phi) {
            if (not(Om > 0.0))
                throw std::invalid_argument("[CubicGalileon] Need Omega_phi > 0");
        }
        std::string name() const override { return "cubic_galileon"; }
        Component component() const override { return Component::DarkEnergy; }
        double rho(double x) const override {
            const double r = rho_others(x);
            return 2.0 * Om / (r + std::sqrt(r * r + 4.0 * Om));
        }
        double drhodx(double x) const override {
            double r = 0.0, dr = 0.0;
            for (const auto & s : others) {
                r += s->rho(x);
                dr += s->drhodx(x);
            }
            const double s = std::sqrt(r * r + 4.0 * Om);
            return -dr * 2.0 * Om / ((r + s) * s);
        }
        double p(double x) const override { return -rho(x) - drhodx(x) / 3.0; }
        /// rho_phi'' = -(r'' q + r' q')/s + r' q s'/s^2 with q = rho_phi, q' = -r' q/s, s' = r r'/s
        double d2rhodx2(double x) const override {
            double r = 0.0, dr = 0.0, ddr = 0.0;
            for (const auto & s : others) {
                r += s->rho(x);
                dr += s->drhodx(x);
                ddr += s->d2rhodx2(x);
            }
            const double s = std::sqrt(r * r + 4.0 * Om), q = 2.0 * Om / (r + s);
            const double dq = -dr * q / s, ds = r * dr / s;
            return -(ddr * q + dr * dq) / s + dr * q * ds / (s * s);
        }
        double Omega_phi() const { return Om; }
        /// H dphi/dt on the tracker [H0^2, phi in reduced Planck masses]
        double xi() const { return std::sqrt(6.0 * Om); }
        std::string info() const override {
            std::ostringstream s;
            s << "cubic Galileon (tracker): Omega0 = " << Om << ", w0 = " << p(0.0) / rho(0.0);
            return s.str();
        }

      private:
        ComponentBackground::SpeciesList others;
        double Om;
        double rho_others(double x) const {
            double r = 0.0;
            for (const auto & s : others)
                r += s->rho(x);
            return r;
        }
    };

    /// The cubic Galileon on its tracker (flat): the standard species + the Galileon (closure), GR
    inline ComponentBackgroundPtr make_cubic_galileon(const BackgroundParameters & p, BackgroundRange r = {}) {
        if (p.Omega_k != 0.0)
            throw std::invalid_argument("[make_cubic_galileon] only implemented for a flat universe");
        auto s = standard_species(p);
        const double Om = 1.0 - density_today(s);
        s.push_back(std::make_shared<CubicGalileonSpecies>(s, Om));
        return std::make_shared<const ComponentBackground>("cubic_galileon",
                                                           p,
                                                           s,
                                                           std::make_shared<GeneralRelativity>(),
                                                           r);
    }

} // namespace FML::COSMOLOGY::BACKGROUND

#endif
