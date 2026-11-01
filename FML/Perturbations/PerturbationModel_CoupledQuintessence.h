#ifndef FML_PERTURBATIONS_PERTURBATIONMODEL_COUPLEDQUINTESSENCE_H
#define FML_PERTURBATIONS_PERTURBATIONMODEL_COUPLEDQUINTESSENCE_H

// =====================================================================================================================
// Coupled quintessence (Background/BackgroundModel_CoupledQuintessence.h): the standard species except the CDM
// (StandardSpeciesPerturbations), and the coupled CDM and field in one block (SpeciesPerturbation_CoupledQuintessence.h,
// on the background's CoupledDarkSector), in general relativity.
//   make_coupled_quintessence_perturbations(p, lambda, beta, r)
// =====================================================================================================================

#include <FML/Background/BackgroundModel_CoupledQuintessence.h>
#include <FML/Background/ComponentBackground.h>
#include <FML/Background/FieldEquation_GeneralRelativity.h>
#include <FML/Perturbations/PerturbationModel.h>
#include <FML/Perturbations/PerturbedFieldEquation_GeneralRelativity.h>
#include <FML/Perturbations/SpeciesPerturbation_CoupledQuintessence.h>

#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace FML::COSMOLOGY::PERTURBATIONS {

    class PerturbationModel_CoupledQuintessence : public PerturbationModel {
      public:
        explicit PerturbationModel_CoupledQuintessence(BACKGROUND::ComponentBackgroundPtr background)
            : PerturbationModel(std::move(background)), field(find_field(*bg)),
              standard(*bg, {find_species<BACKGROUND::CoupledCDMSpecies>(*bg)}) {
            if (not bg->field_equations_as<BACKGROUND::GeneralRelativity>())
                throw std::invalid_argument(
                    "[PerturbationModel_CoupledQuintessence] The background is not general relativity");
        }
        std::string name() const override { return "coupled_quintessence"; }
        const BACKGROUND::CoupledDarkSector & sector() const { return field->sector(); }

        SpeciesPerturbationList species_perturbations(const PerturbationContext & ctx) const override {
            check(ctx);
            SpeciesPerturbationList list;
            standard.add_to(ctx, list);
            list.push_back(std::make_unique<SpeciesPerturbation_CoupledQuintessence>(field->sector(), ctx));
            return list;
        }
        std::unique_ptr<PerturbedFieldEquation>
        field_equation_perturbations(const PerturbationContext & ctx) const override {
            check(ctx);
            return std::make_unique<PerturbedFieldEquation_GeneralRelativity>(ctx);
        }

      private:
        const BACKGROUND::CoupledFieldSpecies * field;
        StandardSpeciesPerturbations standard;

        template <class T>
        static const T * find_species(const BACKGROUND::ComponentBackground & bg) {
            for (const auto & s : bg.species())
                if (const auto * t = dynamic_cast<const T *>(s.get()))
                    return t;
            throw std::invalid_argument("[PerturbationModel_CoupledQuintessence] The background is not coupled "
                                        "quintessence");
        }
        static const BACKGROUND::CoupledFieldSpecies * find_field(const BACKGROUND::ComponentBackground & bg) {
            return find_species<BACKGROUND::CoupledFieldSpecies>(bg);
        }
    };

    inline std::shared_ptr<const PerturbationModel_CoupledQuintessence> make_coupled_quintessence_perturbations(
        const BACKGROUND::BackgroundParameters & p, double lambda, double beta, BACKGROUND::BackgroundRange r = {}) {
        return std::make_shared<const PerturbationModel_CoupledQuintessence>(
            BACKGROUND::make_coupled_quintessence(p, lambda, beta, r));
    }

} // namespace FML::COSMOLOGY::PERTURBATIONS

#endif
