#ifndef FML_PERTURBATIONS_PERTURBATIONMODEL_EDE_H
#define FML_PERTURBATIONS_PERTURBATIONMODEL_EDE_H

// =====================================================================================================================
// Early dark energy (Background/BackgroundModel_EDE.h): the standard species (StandardSpeciesPerturbations), the
// axion-like field (SpeciesPerturbation_ScalarField.h, named "early dark energy", with the effective fluid of
// Poulin et al. 2018 once the background replaces the oscillating field by a fluid; see there) and a smooth
// cosmological constant, in general relativity.
//   make_ede_perturbations(p, f_EDE, log10_zc, theta_i, n, r)
// =====================================================================================================================

#include <FML/Background/BackgroundModel_EDE.h>
#include <FML/Background/ComponentBackground.h>
#include <FML/Background/FieldEquation_GeneralRelativity.h>
#include <FML/Perturbations/PerturbationModel.h>
#include <FML/Perturbations/PerturbedFieldEquation_GeneralRelativity.h>
#include <FML/Perturbations/SpeciesPerturbation_ScalarField.h>

#include <memory>
#include <stdexcept>
#include <string>

namespace FML::COSMOLOGY::PERTURBATIONS {

    class PerturbationModel_EDE : public PerturbationModel {
      public:
        explicit PerturbationModel_EDE(BACKGROUND::ComponentBackgroundPtr background)
            : PerturbationModel(std::move(background)), standard(*bg) {
            if (not bg->field_equations_as<BACKGROUND::GeneralRelativity>())
                throw std::invalid_argument("[PerturbationModel_EDE] The background is not general relativity");
            for (const auto & s : bg->species())
                if (const auto * e = dynamic_cast<const BACKGROUND::EarlyDarkEnergySpecies *>(s.get()))
                    field = e;
            if (not field)
                throw std::invalid_argument("[PerturbationModel_EDE] The background has no early dark energy");
        }
        std::string name() const override { return "ede"; }
        const BACKGROUND::EarlyDarkEnergySpecies & early_dark_energy() const { return *field; }

        SpeciesPerturbationList species_perturbations(const PerturbationContext & ctx) const override {
            check(ctx);
            SpeciesPerturbationList list;
            standard.add_to(ctx, list);
            list.push_back(std::make_unique<SpeciesPerturbation_ScalarField>(*field, "early dark energy", ctx));
            return list;
        }
        std::unique_ptr<PerturbedFieldEquation>
        field_equation_perturbations(const PerturbationContext & ctx) const override {
            check(ctx);
            return std::make_unique<PerturbedFieldEquation_GeneralRelativity>(ctx);
        }

      private:
        StandardSpeciesPerturbations standard;
        const BACKGROUND::EarlyDarkEnergySpecies * field = nullptr;
    };

    inline std::shared_ptr<const PerturbationModel_EDE> make_ede_perturbations(const BACKGROUND::BackgroundParameters & p,
                                                                               double f_EDE,
                                                                               double log10_zc,
                                                                               double theta_i,
                                                                               double n,
                                                                               BACKGROUND::BackgroundRange r = {}) {
        return std::make_shared<const PerturbationModel_EDE>(BACKGROUND::make_ede(p, f_EDE, log10_zc, theta_i, n, r));
    }

} // namespace FML::COSMOLOGY::PERTURBATIONS

#endif
