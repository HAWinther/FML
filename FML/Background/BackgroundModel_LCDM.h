#ifndef FML_BACKGROUND_BACKGROUNDMODEL_LCDM_H
#define FML_BACKGROUND_BACKGROUNDMODEL_LCDM_H

// =====================================================================================================================
// LCDM: the standard species + a cosmological constant (Omega_Lambda from E(0) = 1), general relativity.
//   make_lcdm(p, r)
// =====================================================================================================================

#include <FML/Background/BackgroundParameters.h>
#include <FML/Background/ComponentBackground.h>
#include <FML/Background/FieldEquation_GeneralRelativity.h>
#include <FML/Background/Species.h>

#include <cmath>
#include <memory>
#include <stdexcept>
#include <string>

namespace FML::COSMOLOGY::BACKGROUND {

    /// Standard species + Lambda (from E(0) = 1), GR
    inline ComponentBackgroundPtr make_lcdm(const BackgroundParameters & p, BackgroundRange r = {}) {
        auto s = standard_species(p);
        const double OmegaL = 1.0 - p.Omega_k - density_today(s);
        s.push_back(std::make_shared<PowerLawSpecies>("lambda", Component::DarkEnergy, OmegaL, 0.0));
        return std::make_shared<const ComponentBackground>("lcdm", p, s, std::make_shared<GeneralRelativity>(), r);
    }

} // namespace FML::COSMOLOGY::BACKGROUND

#endif
