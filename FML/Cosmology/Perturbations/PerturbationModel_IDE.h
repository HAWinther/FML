#ifndef FML_COSMOLOGY_PERTURBATIONS_PERTURBATIONMODEL_IDE_H
#define FML_COSMOLOGY_PERTURBATIONS_PERTURBATIONMODEL_IDE_H

// =====================================================================================================================
// Interacting dark energy (Background/BackgroundModel_IDE.h): the standard species except the CDM
// (StandardSpeciesPerturbations), and the interacting CDM with the dark-energy fluid in one block
// (SpeciesPerturbation_InteractingDarkEnergy, SpeciesPerturbation_DarkEnergyFluid.h), in general relativity. Needs
// w != -1 and cs2 > 0; with check_stability the doom factor d = xi / (3 (1 + w)) must be <= 0 (else the dark-energy
// perturbations have a large-scale instability: xi and 1 + w need opposite signs), as the Einstein-Boltzmann solver.
//   make_ide_perturbations(p, w, xi, cs2, check_stability, r)
// =====================================================================================================================

#include <FML/Cosmology/Background/BackgroundModel_IDE.h>
#include <FML/Cosmology/Background/ComponentBackground.h>
#include <FML/Cosmology/Background/FieldEquation_GeneralRelativity.h>
#include <FML/Cosmology/Perturbations/PerturbationModel.h>
#include <FML/Cosmology/Perturbations/PerturbedFieldEquation_GeneralRelativity.h>
#include <FML/Cosmology/Perturbations/SpeciesPerturbation_DarkEnergyFluid.h>

#include <cmath>
#include <memory>
#include <stdexcept>
#include <string>

namespace FML::COSMOLOGY::PERTURBATIONS {

    class PerturbationModel_IDE : public PerturbationModel {
      public:
        PerturbationModel_IDE(BACKGROUND::ComponentBackgroundPtr background, double cs2, bool check_stability = true)
            : PerturbationModel(std::move(background)), cdm(find<BACKGROUND::InteractingCDMSpecies>(*bg)),
              de(find<BACKGROUND::InteractingDarkEnergySpecies>(*bg)), standard(*bg, {cdm}), cs2(cs2) {
            if (not bg->field_equations_as<BACKGROUND::GeneralRelativity>())
                throw std::invalid_argument("[PerturbationModel_IDE] The background is not general relativity");
            if (std::abs(1.0 + de->w) < 1e-6)
                throw std::invalid_argument("interacting dark energy: need w != -1");
            if (not(cs2 > 0.0))
                throw std::invalid_argument("interacting dark energy: need cs2 > 0");
            if (check_stability and doom() > 0.0)
                throw std::invalid_argument(
                    "ide: doom factor d = xi/(3(1 + w)) = " + std::to_string(doom()) +
                    " > 0: large-scale instability of the dark-energy perturbations (xi and 1 + w "
                    "need opposite signs; ide_stability_check = 0 to run anyway)");
        }
        std::string name() const override { return "ide"; }
        /// The doom factor xi / (3 (1 + w))
        double doom() const { return de->xi / (3.0 * (1.0 + de->w)); }

        SpeciesPerturbationList species_perturbations(const PerturbationContext & ctx) const override {
            check(ctx);
            SpeciesPerturbationList list;
            standard.add_to(ctx, list);
            list.push_back(std::make_unique<SpeciesPerturbation_InteractingDarkEnergy>(*cdm, *de, cs2, ctx));
            return list;
        }
        std::unique_ptr<PerturbedFieldEquation>
        field_equation_perturbations(const PerturbationContext & ctx) const override {
            check(ctx);
            return std::make_unique<PerturbedFieldEquation_GeneralRelativity>(ctx);
        }

      private:
        const BACKGROUND::InteractingCDMSpecies * cdm;
        const BACKGROUND::InteractingDarkEnergySpecies * de;
        StandardSpeciesPerturbations standard;
        double cs2;

        template <class T>
        static const T * find(const BACKGROUND::ComponentBackground & bg) {
            for (const auto & s : bg.species())
                if (const auto * t = dynamic_cast<const T *>(s.get()))
                    return t;
            throw std::invalid_argument("[PerturbationModel_IDE] The background is not interacting dark energy");
        }
    };

    inline std::shared_ptr<const PerturbationModel_IDE> make_ide_perturbations(const BACKGROUND::BackgroundParameters & p,
                                                                               double w,
                                                                               double xi,
                                                                               double cs2 = 1.0,
                                                                               bool check_stability = true,
                                                                               BACKGROUND::BackgroundRange r = {}) {
        return std::make_shared<const PerturbationModel_IDE>(BACKGROUND::make_ide(p, w, xi, r), cs2, check_stability);
    }

} // namespace FML::COSMOLOGY::PERTURBATIONS

#endif
