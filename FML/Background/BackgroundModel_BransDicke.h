#ifndef FML_BACKGROUND_BACKGROUNDMODEL_BRANSDICKE_H
#define FML_BACKGROUND_BACKGROUNDMODEL_BRANSDICKE_H

// =====================================================================================================================
// Jordan-Brans-Dicke with a cosmological constant: the standard species (in the units of the density-parameter
// definition) + Lambda from E(0) = 1, with the Brans-Dicke field equations (FieldEquation_BransDicke.h, which explains
// the parameters).
//   make_jbd(p, omega, GeffG_today = 1, density definition = HiClass, r)
// =====================================================================================================================

#include <FML/Background/BackgroundParameters.h>
#include <FML/Background/ComponentBackground.h>
#include <FML/Background/FieldEquation_BransDicke.h>
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

    /// Jordan-Brans-Dicke with Lambda: the standard species (in the units of the density definition) + Lambda from
    /// E(0) = 1. GeffG_today = G_eff/G today (phi(0) = phi_*/GeffG_today)
    inline ComponentBackgroundPtr
    make_jbd(const BackgroundParameters & p,
             double omega,
             double GeffG_today = 1.0,
             BransDickeDensityDefinition definition = BransDickeDensityDefinition::HiClass,
             BackgroundRange r = {}) {
        if (not(GeffG_today > 0.0))
            throw std::invalid_argument("[make_jbd] Need GeffG_today > 0");
        if (not(omega > -1.5))
            throw std::invalid_argument("[make_jbd] Need omega > -3/2");
        const double phi_star = (4.0 + 2.0 * omega) / (3.0 + 2.0 * omega), phi0 = phi_star / GeffG_today;
        double g = 1.0, radiation = 1.0;
        switch (definition) {
            case BransDickeDensityDefinition::HiClass:
                break;
            case BransDickeDensityDefinition::Gtoday:
                g = phi0;
                radiation = phi_star / phi0;
                break;
            case BransDickeDensityDefinition::Gnewton:
                g = phi_star;
                break;
            case BransDickeDensityDefinition::Gbare:
                radiation = phi_star;
                break;
            default:
                throw std::invalid_argument("[make_jbd] Unknown density parameter definition");
        }
        auto s = standard_species(p);
        if (radiation != 1.0)
            for (auto & sp : s)
                if (sp->component() == Component::Photons or sp->component() == Component::MasslessNeutrinos or
                    sp->component() == Component::MassiveNeutrinos)
                    sp = std::make_shared<ScaledSpecies>(sp, radiation);
        auto eqs = std::make_shared<BransDickeFieldEquations>(omega, phi0, g, s, p.Omega_k, r);
        s.push_back(std::make_shared<PowerLawSpecies>("lambda", Component::DarkEnergy, eqs->Omega_Lambda(), 0.0));
        return std::make_shared<const ComponentBackground>("jbd", p, s, eqs, r);
    }

} // namespace FML::COSMOLOGY::BACKGROUND

#endif
