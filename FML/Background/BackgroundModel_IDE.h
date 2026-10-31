#ifndef FML_BACKGROUND_BACKGROUNDMODEL_IDE_H
#define FML_BACKGROUND_BACKGROUNDMODEL_IDE_H

// =====================================================================================================================
// Interacting dark energy (Valiviita, Majerotto & Maartens 2008; Gavela et al. 2009): energy exchange between the cold
// dark matter and the dark energy, Q = xi H rho_DE (xi > 0: energy flows to the dark matter):
//   rho_DE' = -(3 (1 + w) + xi) rho_DE        ->  rho_DE = Omega_DE a^{-3(1 + w) - xi}
//   rho_c'  = -3 rho_c + xi rho_DE            ->  rho_c  = a^{-3} [Omega_c + xi Omega_DE (1 - a^{-(3w + xi)})/(3w +
//   xi)]
// (x = ln a, ' = d/dx, units of rho_c0; Omega_DE from closure). The total is conserved, the field equations are GR.
// rho_c must stay positive until today (with xi < 0 it vanishes at some time in the future).
// =====================================================================================================================

#include <FML/Background/BackgroundParameters.h>
#include <FML/Background/ComponentBackground.h>
#include <FML/Background/FieldEquation_GeneralRelativity.h>
#include <FML/Background/Species.h>

#include <cmath>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>

namespace FML::COSMOLOGY::BACKGROUND {

    /// The dark energy of interacting dark energy, rho = Omega_DE a^{-3(1 + w) - xi}, p = w rho
    class InteractingDarkEnergySpecies : public Species {
      public:
        InteractingDarkEnergySpecies(double Omega_DE, double w, double xi)
            : Omega0(Omega_DE), w(w), xi(xi), L(3.0 * (1.0 + w) + xi) {}
        std::string name() const override { return "ide"; }
        Component component() const override { return Component::DarkEnergy; }
        double rho(double x) const override { return Omega0 * std::exp(-L * x); }
        double p(double x) const override { return w * rho(x); }
        double drhodx(double x) const override { return -L * rho(x); }
        std::string info() const override {
            std::ostringstream s;
            s << "interacting dark energy (Q = xi H rho_DE): Omega0 = " << Omega0 << ", w = " << w << ", xi = " << xi;
            return s.str();
        }
        const double Omega0, w, xi;

      private:
        double L;
    };

    /// The cold dark matter of interacting dark energy
    class InteractingCDMSpecies : public Species {
      public:
        InteractingCDMSpecies(double Omega_c, std::shared_ptr<const InteractingDarkEnergySpecies> de)
            : Omega_c(Omega_c), de(std::move(de)) {}
        std::string name() const override { return "cdm (interacting)"; }
        Component component() const override { return Component::CDM; }
        /// rho_c a^3 = Omega_c + xi Omega_DE G(x),  G = int_0^x e^{-s x'} dx' = -expm1(-s x)/s,  s = 3w + xi
        double rho(double x) const override {
            const double s = 3.0 * de->w + de->xi;
            const double G = (s == 0.0) ? x : -std::expm1(-s * x) / s;
            return std::exp(-3.0 * x) * (Omega_c + de->xi * de->Omega0 * G);
        }
        double p(double) const override { return 0.0; }
        double drhodx(double x) const override { return -3.0 * rho(x) + de->xi * de->rho(x); }

      private:
        double Omega_c;
        std::shared_ptr<const InteractingDarkEnergySpecies> de;
    };

    /// Interacting dark energy: the standard species with the interacting CDM (Omega_cdm today) + the dark energy
    /// (closure), GR
    inline ComponentBackgroundPtr
    make_ide(const BackgroundParameters & p, double w, double xi, BackgroundRange r = {}) {
        const auto standard = standard_species(p);
        const double OmegaDE = 1.0 - p.Omega_k - density_today(standard);
        auto de = std::make_shared<const InteractingDarkEnergySpecies>(OmegaDE, w, xi);
        auto cdm = std::make_shared<const InteractingCDMSpecies>(p.Omega_cdm, de);
        for (double x = r.x_min; x <= 0.0; x += 0.01)
            if (not(cdm->rho(x) > 0.0))
                throw std::invalid_argument("[make_ide] rho_cdm < 0 at a = " + std::to_string(std::exp(x)) +
                                            " (too much energy flows out of the dark matter)");
        ComponentBackground::SpeciesList s;
        for (const auto & sp : standard)
            s.push_back(sp->component() == Component::CDM ? cdm : sp);
        s.push_back(de);
        return std::make_shared<const ComponentBackground>("ide", p, s, std::make_shared<GeneralRelativity>(), r);
    }

} // namespace FML::COSMOLOGY::BACKGROUND

#endif
