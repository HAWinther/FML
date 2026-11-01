#ifndef FML_COSMOLOGY_BACKGROUND_BACKGROUNDMODEL_W0WA_H
#define FML_COSMOLOGY_BACKGROUND_BACKGROUNDMODEL_W0WA_H

// =====================================================================================================================
// w0wa: the standard species + dark energy with w = w0 + wa (1 - a) (Omega_DE from E(0) = 1), general relativity.
//   make_w0wa(p, w0, wa, r)
// =====================================================================================================================

#include <FML/Cosmology/Background/BackgroundParameters.h>
#include <FML/Cosmology/Background/ComponentBackground.h>
#include <FML/Cosmology/Background/FieldEquation_GeneralRelativity.h>
#include <FML/Cosmology/Background/Species.h>

#include <cmath>
#include <memory>
#include <stdexcept>
#include <string>

namespace FML::COSMOLOGY::BACKGROUND {

    /// Standard species + w0wa dark energy (from E(0) = 1), GR
    inline ComponentBackgroundPtr make_w0wa(const BackgroundParameters & p, double w0, double wa, BackgroundRange r = {}) {
        auto s = standard_species(p);
        const double OmegaDE = 1.0 - p.Omega_k - density_today(s);
        s.push_back(std::make_shared<CPLDarkEnergy>(OmegaDE, w0, wa));
        return std::make_shared<const ComponentBackground>("w0wa", p, s, std::make_shared<GeneralRelativity>(), r);
    }

} // namespace FML::COSMOLOGY::BACKGROUND

#endif
