#ifndef FML_PERTURBATIONS_PERTURBATIONMODEL_HUSAWICKY_H
#define FML_PERTURBATIONS_PERTURBATIONMODEL_HUSAWICKY_H

// =====================================================================================================================
// Hu-Sawicky f(R): the standard species (StandardSpeciesPerturbations) on the LCDM background (to O(|f_R0|)) with
// the quasi-static perturbed field equations of PerturbedFieldEquation_HuSawicky.h (|f_R0| >= 0, n > 0). Flat, as the
// Einstein-Boltzmann solver.
//   make_hu_sawicky_perturbations(p, fR0, n, r)
// =====================================================================================================================

#include <FML/Background/BackgroundModel_LCDM.h>
#include <FML/Background/ComponentBackground.h>
#include <FML/Background/FieldEquation_GeneralRelativity.h>
#include <FML/Perturbations/PerturbationModel.h>
#include <FML/Perturbations/PerturbationModel_MuSigma.h>
#include <FML/Perturbations/PerturbedFieldEquation_HuSawicky.h>

#include <memory>
#include <stdexcept>
#include <string>

namespace FML::COSMOLOGY::PERTURBATIONS {

    class PerturbationModel_HuSawicky : public PerturbationModel {
      public:
        PerturbationModel_HuSawicky(BACKGROUND::ComponentBackgroundPtr background, double fR0, double n)
            : PerturbationModel(std::move(background)), standard(*bg), fR0(fR0), n(n) {
            if (not bg->field_equations_as<BACKGROUND::GeneralRelativity>())
                throw std::invalid_argument("[PerturbationModel_HuSawicky] The background is not general relativity");
            check_musigma_background(*bg, "Hu-Sawicky f(R)");
            PerturbedFieldEquation_HuSawicky(fR0, n, context()); // checks fR0, n
        }
        std::string name() const override { return "fr"; }

        SpeciesPerturbationList species_perturbations(const PerturbationContext & ctx) const override {
            check(ctx);
            SpeciesPerturbationList list;
            standard.add_to(ctx, list);
            return list;
        }
        std::unique_ptr<PerturbedFieldEquation>
        field_equation_perturbations(const PerturbationContext & ctx) const override {
            check(ctx);
            return std::make_unique<PerturbedFieldEquation_HuSawicky>(fR0, n, ctx);
        }

      private:
        StandardSpeciesPerturbations standard;
        double fR0, n;
    };

    inline std::shared_ptr<const PerturbationModel_HuSawicky> make_hu_sawicky_perturbations(
        const BACKGROUND::BackgroundParameters & p, double fR0, double n = 1.0, BACKGROUND::BackgroundRange r = {}) {
        return std::make_shared<const PerturbationModel_HuSawicky>(BACKGROUND::make_lcdm(p, r), fR0, n);
    }

} // namespace FML::COSMOLOGY::PERTURBATIONS

#endif
