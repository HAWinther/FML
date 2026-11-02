#ifndef FML_COSMOLOGY_PERTURBATIONS_QUASISTATICGRAVITY_BRANSDICKE_H
#define FML_COSMOLOGY_PERTURBATIONS_QUASISTATICGRAVITY_BRANSDICKE_H

// =====================================================================================================================
// Jordan-Brans-Dicke gravity in the quasi-static limit (QuasiStaticGravity.h), on its background
// (Background/FieldEquation_BransDicke.h: phi(x), omega, g = G/G_*), in units of the G_* of the density parameters:
//   mu = g phi_* / phi(a),   phi_* = (4 + 2 omega)/(3 + 2 omega),   Sigma = g / phi(a)
// (G_eff = G phi_*/phi is the Cavendish constant; the lensing potential couples to G/phi). Scale independent. The
// Einstein-Boltzmann solver solves the full perturbations (PerturbedFieldEquation_BransDicke.h); this is their
// sub-horizon limit, used by SimulationCode's GravityModel_JBD.
// =====================================================================================================================

#include <FML/Cosmology/Background/FieldEquation_BransDicke.h>
#include <FML/Cosmology/Perturbations/QuasiStaticGravity.h>

#include <stdexcept>
#include <string>

namespace FML::COSMOLOGY::PERTURBATIONS {

    class QuasiStaticGravity_BransDicke : public QuasiStaticGravity {
      public:
        QuasiStaticGravity_BransDicke(const BACKGROUND::Background & bg, const BACKGROUND::BransDickeFieldEquations & bd)
            : QuasiStaticGravity(bg), bd(bd) {}
        /// From a background with Brans-Dicke field equations (throws otherwise)
        explicit QuasiStaticGravity_BransDicke(const BACKGROUND::ComponentBackground & bg)
            : QuasiStaticGravity(bg), bd(field_equations(bg)) {}
        std::string name() const override { return "Brans-Dicke"; }
        std::string info() const override {
            return "Brans-Dicke (omega = " + std::to_string(bd.get_omega()) +
                   ", G/G_* = " + std::to_string(bd.G_over_Gstar()) + "; quasi-static)";
        }

        double mu(double x, double /*koverH0*/) const override {
            return bd.G_over_Gstar() * bd.phi_star() / bd.phi(x);
        }
        double Sigma(double x, double /*koverH0*/) const override { return bd.G_over_Gstar() / bd.phi(x); }

        const BACKGROUND::BransDickeFieldEquations & field_equations() const { return bd; }

      private:
        const BACKGROUND::BransDickeFieldEquations & bd;
        static const BACKGROUND::BransDickeFieldEquations & field_equations(const BACKGROUND::ComponentBackground & bg) {
            const auto * bd = bg.field_equations_as<BACKGROUND::BransDickeFieldEquations>();
            if (not bd)
                throw std::invalid_argument("[QuasiStaticGravity_BransDicke] The background is not Brans-Dicke");
            return *bd;
        }
    };

} // namespace FML::COSMOLOGY::PERTURBATIONS

#endif
