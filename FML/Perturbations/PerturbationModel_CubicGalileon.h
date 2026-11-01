#ifndef FML_PERTURBATIONS_PERTURBATIONMODEL_CUBICGALILEON_H
#define FML_PERTURBATIONS_PERTURBATIONMODEL_CUBICGALILEON_H

// =====================================================================================================================
// The cubic Galileon on its tracker (Background/BackgroundModel_CubicGalileon.h): the standard species
// (StandardSpeciesPerturbations) with the braided scalar in the perturbed field equations
// (PerturbedFieldEquation_CubicGalileon.h; the Galileon species has no perturbations of its own). Flat.
//   make_cubic_galileon_perturbations(p, r)
// =====================================================================================================================

#include <FML/Background/BackgroundModel_CubicGalileon.h>
#include <FML/Background/ComponentBackground.h>
#include <FML/Perturbations/PerturbationModel.h>
#include <FML/Perturbations/PerturbedFieldEquation_CubicGalileon.h>

#include <memory>
#include <stdexcept>
#include <string>

namespace FML::COSMOLOGY::PERTURBATIONS {

    class PerturbationModel_CubicGalileon : public PerturbationModel {
      public:
        explicit PerturbationModel_CubicGalileon(BACKGROUND::ComponentBackgroundPtr background)
            : PerturbationModel(std::move(background)), standard(*bg) {
            int n = 0;
            for (const auto & s : bg->species())
                if (const auto * g = dynamic_cast<const BACKGROUND::CubicGalileonSpecies *>(s.get())) {
                    galileon = g;
                    n++;
                }
            if (n != 1)
                throw std::invalid_argument("cubic Galileon: the model needs exactly one CubicGalileonSpecies");
            if (bg->parameters().Omega_k != 0.0)
                throw std::invalid_argument("cubic Galileon: only implemented for a flat universe");
        }
        std::string name() const override { return "cubic_galileon"; }
        const BACKGROUND::CubicGalileonSpecies & galileon_species() const { return *galileon; }

        SpeciesPerturbationList species_perturbations(const PerturbationContext & ctx) const override {
            check(ctx);
            SpeciesPerturbationList list;
            standard.add_to(ctx, list);
            return list;
        }
        std::unique_ptr<PerturbedFieldEquation>
        field_equation_perturbations(const PerturbationContext & ctx) const override {
            check(ctx);
            return std::make_unique<PerturbedFieldEquation_CubicGalileon>(*galileon, ctx);
        }

      private:
        StandardSpeciesPerturbations standard;
        const BACKGROUND::CubicGalileonSpecies * galileon = nullptr;
    };

    inline std::shared_ptr<const PerturbationModel_CubicGalileon>
    make_cubic_galileon_perturbations(const BACKGROUND::BackgroundParameters & p, BACKGROUND::BackgroundRange r = {}) {
        return std::make_shared<const PerturbationModel_CubicGalileon>(BACKGROUND::make_cubic_galileon(p, r));
    }

} // namespace FML::COSMOLOGY::PERTURBATIONS

#endif
