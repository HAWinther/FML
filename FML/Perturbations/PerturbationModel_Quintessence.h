#ifndef FML_PERTURBATIONS_PERTURBATIONMODEL_QUINTESSENCE_H
#define FML_PERTURBATIONS_PERTURBATIONMODEL_QUINTESSENCE_H

// =====================================================================================================================
// Quintessence (all four potentials of Background/BackgroundModel_Quintessence.h): the standard species
// (StandardSpeciesPerturbations) and the scalar field (SpeciesPerturbation_ScalarField.h, named "quintessence", with
// the fluid mode once a PNGB or hilltop field oscillates fast) in general relativity. The exponential potential with
// lambda = 0 is a cosmological constant and gets no perturbations (as the Einstein-Boltzmann solver).
//   make_quintessence_perturbations(p, potential, phi_ini, dphidlna_ini, r)
// =====================================================================================================================

#include <FML/Background/BackgroundModel_Quintessence.h>
#include <FML/Background/ComponentBackground.h>
#include <FML/Background/FieldEquation_GeneralRelativity.h>
#include <FML/Perturbations/PerturbationModel.h>
#include <FML/Perturbations/PerturbedFieldEquation_GeneralRelativity.h>
#include <FML/Perturbations/SpeciesPerturbation_ScalarField.h>

#include <memory>
#include <stdexcept>
#include <string>

namespace FML::COSMOLOGY::PERTURBATIONS {

    class PerturbationModel_Quintessence : public PerturbationModel {
      public:
        explicit PerturbationModel_Quintessence(BACKGROUND::ComponentBackgroundPtr background)
            : PerturbationModel(std::move(background)), standard(*bg) {
            if (not bg->field_equations_as<BACKGROUND::GeneralRelativity>())
                throw std::invalid_argument("[PerturbationModel_Quintessence] The background is not general relativity");
            for (const auto & s : bg->species())
                if (const auto * q = dynamic_cast<const BACKGROUND::QuintessenceSpecies *>(s.get()))
                    field = q;
            if (not field)
                throw std::invalid_argument("[PerturbationModel_Quintessence] The background has no quintessence");
            const auto & V = field->potential();
            perturbed = not(V.type == BACKGROUND::QuintessencePotential::Type::Exponential and V.lambda == 0.0);
        }
        std::string name() const override { return "quintessence"; }
        const BACKGROUND::QuintessenceSpecies & quintessence() const { return *field; }

        SpeciesPerturbationList species_perturbations(const PerturbationContext & ctx) const override {
            check(ctx);
            SpeciesPerturbationList list;
            standard.add_to(ctx, list);
            if (perturbed)
                list.push_back(std::make_unique<SpeciesPerturbation_ScalarField>(*field, "quintessence", ctx));
            return list;
        }
        std::unique_ptr<PerturbedFieldEquation>
        field_equation_perturbations(const PerturbationContext & ctx) const override {
            check(ctx);
            return std::make_unique<PerturbedFieldEquation_GeneralRelativity>(ctx);
        }

      private:
        StandardSpeciesPerturbations standard;
        const BACKGROUND::QuintessenceSpecies * field = nullptr;
        bool perturbed = true;
    };

    inline std::shared_ptr<const PerturbationModel_Quintessence>
    make_quintessence_perturbations(const BACKGROUND::BackgroundParameters & p,
                                    const BACKGROUND::QuintessencePotential & V,
                                    double phi_ini = 0.0,
                                    double dphidlna_ini = 0.0,
                                    BACKGROUND::BackgroundRange r = {}) {
        return std::make_shared<const PerturbationModel_Quintessence>(
            BACKGROUND::make_quintessence(p, V, phi_ini, dphidlna_ini, r));
    }

} // namespace FML::COSMOLOGY::PERTURBATIONS

#endif
