#ifndef FML_COSMOLOGY_BACKGROUND_FIELDEQUATION_GENERALRELATIVITY_H
#define FML_COSMOLOGY_BACKGROUND_FIELDEQUATION_GENERALRELATIVITY_H

// =====================================================================================================================
// The field equations of general relativity: E^2 = sum_i rho_i / rho_c0 + Omega_k e^{-2x}
// =====================================================================================================================

#include <FML/Cosmology/Background/ComponentBackground.h>
#include <FML/Cosmology/Background/FieldEquations.h>

#include <string>

namespace FML::COSMOLOGY::BACKGROUND {

    class GeneralRelativity : public FieldEquations {
      public:
        std::string name() const override { return "gr"; }
        std::string info() const override { return "general relativity"; }
        double E2(double x, const ComponentBackground & bg) const override { return bg.rho_total(x) + bg.rho_k(x); }
        double dE2dx(double x, const ComponentBackground & bg) const override {
            return bg.drho_total(x) + bg.drho_k(x);
        }
    };

} // namespace FML::COSMOLOGY::BACKGROUND

#endif
