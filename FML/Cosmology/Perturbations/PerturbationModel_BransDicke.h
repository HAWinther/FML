#ifndef FML_COSMOLOGY_PERTURBATIONS_PERTURBATIONMODEL_BRANSDICKE_H
#define FML_COSMOLOGY_PERTURBATIONS_PERTURBATIONMODEL_BRANSDICKE_H

// =====================================================================================================================
// Jordan-Brans-Dicke with Lambda: the standard species (StandardSpeciesPerturbations; Lambda smooth) and the perturbed
// Brans-Dicke field equations (PerturbedFieldEquation_BransDicke.h), on the background of make_jbd
// (Background/BackgroundModel_BransDicke.h, whose BransDickeFieldEquations are kept as a typed pointer).
//   make_jbd_perturbations(p, omega, GeffG_today = 1, density definition = HiClass, r)
// Flat only (as the solver).
// =====================================================================================================================

#include <FML/Cosmology/Background/BackgroundModel_BransDicke.h>
#include <FML/Cosmology/Background/BackgroundParameters.h>
#include <FML/Cosmology/Background/ComponentBackground.h>
#include <FML/Cosmology/Background/FieldEquation_BransDicke.h>
#include <FML/Cosmology/Perturbations/PerturbationModel.h>
#include <FML/Cosmology/Perturbations/PerturbedFieldEquation_BransDicke.h>

#include <memory>
#include <stdexcept>
#include <string>

namespace FML::COSMOLOGY::PERTURBATIONS {

    class PerturbationModel_BransDicke : public PerturbationModel {
      public:
        explicit PerturbationModel_BransDicke(BACKGROUND::ComponentBackgroundPtr background)
            : PerturbationModel(std::move(background)), standard(*bg),
              bd(bg->field_equations_as<BACKGROUND::BransDickeFieldEquations>()) {
            if (not bd)
                throw std::invalid_argument("[PerturbationModel_BransDicke] The background is not Brans-Dicke");
            if (bg->parameters().Omega_k != 0.0)
                throw std::invalid_argument("[PerturbationModel_BransDicke] Only implemented for a flat universe");
        }
        std::string name() const override { return "jbd"; }
        const BACKGROUND::BransDickeFieldEquations & field_equations() const { return *bd; }

        SpeciesPerturbationList species_perturbations(const PerturbationContext & ctx) const override {
            check(ctx);
            SpeciesPerturbationList list;
            standard.add_to(ctx, list);
            return list;
        }
        std::unique_ptr<PerturbedFieldEquation>
        field_equation_perturbations(const PerturbationContext & ctx) const override {
            check(ctx);
            return std::make_unique<PerturbedFieldEquation_BransDicke>(*bd, ctx);
        }

      private:
        StandardSpeciesPerturbations standard;
        const BACKGROUND::BransDickeFieldEquations * bd;
    };

    inline std::shared_ptr<const PerturbationModel_BransDicke> make_jbd_perturbations(
        const BACKGROUND::BackgroundParameters & p,
        double omega,
        double GeffG_today = 1.0,
        BACKGROUND::BransDickeDensityDefinition definition = BACKGROUND::BransDickeDensityDefinition::HiClass,
        BACKGROUND::BackgroundRange r = {}) {
        return std::make_shared<const PerturbationModel_BransDicke>(
            BACKGROUND::make_jbd(p, omega, GeffG_today, definition, r));
    }

} // namespace FML::COSMOLOGY::PERTURBATIONS

#endif
