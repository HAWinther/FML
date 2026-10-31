#ifndef FML_BACKGROUND_BACKGROUNDMODEL_DARKSCATTERING_H
#define FML_BACKGROUND_BACKGROUNDMODEL_DARKSCATTERING_H

// =====================================================================================================================
// Dark scattering (Simpson 2010; Baldi & Simpson 2015): elastic scattering of dark matter off the dark energy. Only
// momentum is exchanged, so the background is wCDM:
//   rho_DE = Omega_DE a^{-3(1 + w)}
// The cross section xi = sigma_D/m_c and the sound speed cs2 only affect the perturbations (make_dark_scattering does
// not take them; the registry accepts ds_xi, ds_cs2 so that parameter sets are interchangeable).
//
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

} // namespace FML::COSMOLOGY::BACKGROUND

#endif
