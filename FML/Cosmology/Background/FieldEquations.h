#ifndef FML_COSMOLOGY_BACKGROUND_FIELDEQUATIONS_H
#define FML_COSMOLOGY_BACKGROUND_FIELDEQUATIONS_H

// =====================================================================================================================
// FieldEquations: the theory of gravity of a ComponentBackground, i.e. how the species source the expansion (the
// Friedmann equation).
//
// Background: E^2(x) = (H/H0)^2 and dE^2/dx, given the species of the background (ComponentBackground::rho_total,
// drho_total, rho(component, x), Omega_k). In GR
//   E^2 = sum_i rho_i / rho_c0 + Omega_k e^{-2x}.
// A theory with its own background degrees of freedom (Brans-Dicke's scalar, ...) solves them in init(), which the
// ComponentBackground constructor calls once before anything else is evaluated.
// =====================================================================================================================

#include <string>

namespace FML::COSMOLOGY::BACKGROUND {

    class ComponentBackground;

    class FieldEquations {
      public:
        virtual ~FieldEquations() = default;
        virtual std::string name() const = 0;
        virtual std::string info() const { return name(); }

        /// Called once by the ComponentBackground constructor (solve background fields, check the model)
        virtual void init(const ComponentBackground & /*bg*/) {}
        /// (H/H0)^2 and its x-derivative
        virtual double E2(double x, const ComponentBackground & bg) const = 0;
        virtual double dE2dx(double x, const ComponentBackground & bg) const = 0;
    };

} // namespace FML::COSMOLOGY::BACKGROUND

#endif
