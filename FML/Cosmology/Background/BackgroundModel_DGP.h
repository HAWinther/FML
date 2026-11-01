#ifndef FML_COSMOLOGY_BACKGROUND_BACKGROUNDMODEL_DGP_H
#define FML_COSMOLOGY_BACKGROUND_BACKGROUNDMODEL_DGP_H

// =====================================================================================================================
// DGP: the standard species + Lambda with the DGP field equations (FieldEquation_DGP.h).
//
// make_dgp(p, Omega_rc, branch, lcdm_background): standard species + Lambda (from E(0) = 1 with the DGP Friedmann
// equation). Omega_rc = 0 on the self-accelerating branch: the value without dark energy.
// =====================================================================================================================

#include <FML/Cosmology/Background/BackgroundParameters.h>
#include <FML/Cosmology/Background/ComponentBackground.h>
#include <FML/Cosmology/Background/FieldEquation_DGP.h>
#include <FML/Cosmology/Background/FieldEquations.h>
#include <FML/Cosmology/Background/FieldEquation_GeneralRelativity.h>
#include <FML/Cosmology/Background/Species.h>

#include <cmath>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>

namespace FML::COSMOLOGY::BACKGROUND {

    inline ComponentBackgroundPtr make_dgp(const BackgroundParameters & p,
                                           double Omega_rc,
                                           int branch,
                                           bool lcdm_background = false,
                                           BackgroundRange r = {}) {
        if (p.Omega_k != 0.0)
            throw std::invalid_argument("[make_dgp] only implemented for a flat universe");
        auto s = standard_species(p);
        const double rho0 = density_today(s);
        if (lcdm_background) {
            s.push_back(std::make_shared<PowerLawSpecies>("lambda", Component::DarkEnergy, 1.0 - rho0, 0.0));
        } else {
            if (Omega_rc == 0.0) {
                if (branch != 1)
                    throw std::invalid_argument("[make_dgp] Omega_rc = 0 (no dark energy) needs branch = +1");
                Omega_rc = 0.25 * (1.0 - rho0) * (1.0 - rho0);
            }
            const double OmegaL = DGPFieldEquations::OmegaLambda(Omega_rc, branch, rho0);
            if (std::abs(OmegaL) > 1e-14)
                s.push_back(std::make_shared<PowerLawSpecies>("lambda", Component::DarkEnergy, OmegaL, 0.0));
        }
        return std::make_shared<const ComponentBackground>(
            "dgp", p, s, std::make_shared<DGPFieldEquations>(Omega_rc, branch, lcdm_background), r);
    }

} // namespace FML::COSMOLOGY::BACKGROUND

#endif
