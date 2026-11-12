#ifndef FML_COSMOLOGY_BACKGROUND_BACKGROUNDMODEL_DARKSCATTERING_H
#define FML_COSMOLOGY_BACKGROUND_BACKGROUNDMODEL_DARKSCATTERING_H

// =====================================================================================================================
// Dark scattering (Simpson 2010; Baldi & Simpson 2015): elastic scattering of dark matter off the dark energy. Only
// momentum is exchanged, so the background is wCDM:
//   rho_DE = Omega_DE a^{-3(1 + w)}
// The cross section xi = sigma_D/m_c and the sound speed cs2 only affect the perturbations (make_dark_scattering does
// not take them; the registry accepts ds_xi, ds_cs2 so that parameter sets are interchangeable).
//
// dark_scattering_cdm_drag(bg, xi, x): the drag of the dark matter per ln a, Gamma_c / (aH) = (1 + w) c xi rho_DE / H
// (rho_DE physical, xi in m^2/kg; dark_scattering_xi_SI converts from barn/GeV): the rate in the dark-matter velocity
// equation v_c' = ... - Gamma_c/(aH) (v_c - v_DE) of the Einstein-Boltzmann solver (SpeciesPerturbation_DarkScattering
// in Perturbations/SpeciesPerturbation_DarkEnergyFluid.h) and the velocity friction of the N-body dark matter, where the
// smooth dark energy does not move (v_DE = 0; SimulationCode's DarkScattering cosmology).
// =====================================================================================================================

#include <FML/Cosmology/Background/BackgroundParameters.h>
#include <FML/Cosmology/Background/ComponentBackground.h>
#include <FML/Cosmology/Background/FieldEquation_GeneralRelativity.h>
#include <FML/Cosmology/Background/Species.h>
#include <FML/Units/Units.h>

#include <cmath>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>

namespace FML::COSMOLOGY::BACKGROUND {

    /// Dark scattering: wCDM, the dark energy rho = Omega_DE a^{-3(1 + w)} from closure, GR
    inline ComponentBackgroundPtr
    make_dark_scattering(const BackgroundParameters & p, double w, BackgroundRange r = {}) {
        auto s = standard_species(p);
        const double OmegaDE = 1.0 - p.Omega_k - density_today(s);
        s.push_back(
            std::make_shared<PowerLawSpecies>("dark_energy (w)", Component::DarkEnergy, OmegaDE, 3.0 * (1.0 + w)));
        return std::make_shared<const ComponentBackground>("dark_scattering",
                                                           p,
                                                           s,
                                                           std::make_shared<GeneralRelativity>(),
                                                           r);
    }

    /// xi = sigma_D / m_c from barn/GeV (the parameter ds_xi) to m^2/kg
    inline double dark_scattering_xi_SI(double xi_barn_per_GeV) {
        return xi_barn_per_GeV * 1e-28 / FML::UTILS::SI::GeV_over_c2;
    }

    /// The drag rate of the dark matter per ln a, Gamma_c / (aH) = (1 + w) c xi rho_DE / H, at x = ln a on a dark
    /// scattering background (its one dark-energy species), xi in m^2/kg
    inline double dark_scattering_cdm_drag(const ComponentBackground & bg, double xi_SI, double x) {
        using namespace FML::UTILS::SI;
        const Species * de = nullptr;
        for (const auto & s : bg.species())
            if (s->component() == Component::DarkEnergy) {
                if (de)
                    throw std::invalid_argument("dark_scattering_cdm_drag: more than one dark energy species");
                de = s.get();
            }
        if (not de)
            throw std::invalid_argument("dark_scattering_cdm_drag: no dark energy species");
        const double h = bg.parameters().h;
        const double rho_DE = de->rho(x) * rho_crit_over_h2 * h * h; // kg/m^3
        const double H = H0_over_h * h * bg.E(x);                       // 1/s
        return (1.0 + de->w(x)) * c * xi_SI * rho_DE / H;
    }

} // namespace FML::COSMOLOGY::BACKGROUND

#endif
