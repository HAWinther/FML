#ifndef FML_PERTURBATIONS_PERTURBATIONMODEL_LCDM_H
#define FML_PERTURBATIONS_PERTURBATIONMODEL_LCDM_H

// =====================================================================================================================
// LCDM: the standard species (CDM, massless and massive neutrinos; StandardSpeciesPerturbations) in general relativity
// (PerturbedFieldEquation_GeneralRelativity). Lambda is smooth.
//   make_lcdm_perturbations(p, r)    with its background make_lcdm(p, r)
// =====================================================================================================================

#include <FML/Background/BackgroundModel_LCDM.h>
#include <FML/Background/BackgroundParameters.h>
#include <FML/Background/ComponentBackground.h>
#include <FML/Background/FieldEquation_GeneralRelativity.h>
#include <FML/Perturbations/PerturbationModel.h>
#include <FML/Perturbations/PerturbedFieldEquation_GeneralRelativity.h>

#include <memory>
#include <stdexcept>
#include <string>

namespace FML::COSMOLOGY::PERTURBATIONS {

    class PerturbationModel_LCDM : public PerturbationModel {
      public:
        explicit PerturbationModel_LCDM(BACKGROUND::ComponentBackgroundPtr background)
            : PerturbationModel(std::move(background)), standard(*bg) {
            if (not bg->field_equations_as<BACKGROUND::GeneralRelativity>())
                throw std::invalid_argument("[PerturbationModel_LCDM] The background is not general relativity");
        }
        std::string name() const override { return "lcdm"; }

        SpeciesPerturbationList species_perturbations(const PerturbationContext & ctx) const override {
            check(ctx);
            SpeciesPerturbationList list;
            standard.add_to(ctx, list);
            return list;
        }
        std::unique_ptr<PerturbedFieldEquation>
        field_equation_perturbations(const PerturbationContext & ctx) const override {
            check(ctx);
            return std::make_unique<PerturbedFieldEquation_GeneralRelativity>(ctx);
        }

      private:
        StandardSpeciesPerturbations standard;
    };

    inline std::shared_ptr<const PerturbationModel_LCDM>
    make_lcdm_perturbations(const BACKGROUND::BackgroundParameters & p, BACKGROUND::BackgroundRange r = {}) {
        return std::make_shared<const PerturbationModel_LCDM>(BACKGROUND::make_lcdm(p, r));
    }

} // namespace FML::COSMOLOGY::PERTURBATIONS

#endif
