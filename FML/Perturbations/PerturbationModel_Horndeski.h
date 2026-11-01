#ifndef FML_PERTURBATIONS_PERTURBATIONMODEL_HORNDESKI_H
#define FML_PERTURBATIONS_PERTURBATIONMODEL_HORNDESKI_H

// =====================================================================================================================
// Horndeski in the alpha parameterization of hi_class (alpha_T = 0): the standard species (StandardSpeciesPerturbations;
// the dark energy, Lambda or w0wa, is the effective fluid of the scalar and has no perturbations of its own) with the
// perturbed field equations of PerturbedFieldEquation_HorndeskiAlpha.h, on a GR background (LCDM or w0wa expansion, as
// the Einstein-Boltzmann solver's horndeski model). The alphas (StandardAlphas, or any AlphaFunctions) and the
// stability check (HorndeskiAlphaBackground) are set up once, here. Flat.
//   make_horndeski_perturbations(p, type, cK, cB, cM, M2_ini, w0, wa, check_stability, r)
// =====================================================================================================================

#include <FML/Background/BackgroundModel_LCDM.h>
#include <FML/Background/BackgroundModel_w0wa.h>
#include <FML/Background/ComponentBackground.h>
#include <FML/Perturbations/PerturbationModel.h>
#include <FML/Perturbations/PerturbedFieldEquation_HorndeskiAlpha.h>

#include <memory>
#include <stdexcept>
#include <string>

namespace FML::COSMOLOGY::PERTURBATIONS {

    class PerturbationModel_Horndeski : public PerturbationModel {
      public:
        PerturbationModel_Horndeski(BACKGROUND::ComponentBackgroundPtr background,
                                    std::shared_ptr<AlphaFunctions> alphas,
                                    bool check_stability = true)
            : PerturbationModel(std::move(background)), standard(*bg),
              hd(std::make_shared<const HorndeskiAlphaBackground>(std::move(alphas), *bg, check_stability)) {}
        std::string name() const override { return "horndeski"; }
        const HorndeskiAlphaBackground & alpha_background() const { return *hd; }

        SpeciesPerturbationList species_perturbations(const PerturbationContext & ctx) const override {
            check(ctx);
            SpeciesPerturbationList list;
            standard.add_to(ctx, list);
            return list;
        }
        std::unique_ptr<PerturbedFieldEquation>
        field_equation_perturbations(const PerturbationContext & ctx) const override {
            check(ctx);
            return std::make_unique<PerturbedFieldEquation_HorndeskiAlpha>(hd, ctx);
        }

      private:
        StandardSpeciesPerturbations standard;
        std::shared_ptr<const HorndeskiAlphaBackground> hd;
    };

    /// The hi_class parameterization type (0 propto_omega, 1 propto_scale, 2 constant_alphas) as a StandardAlphas
    inline std::shared_ptr<StandardAlphas>
    make_standard_alphas(double type, double cK, double cB, double cM, double M2_ini) {
        if (type != 0.0 and type != 1.0 and type != 2.0)
            throw std::invalid_argument(
                "horndeski: hd_parametrization must be 0 (propto_omega), 1 (propto_scale) or 2 (constant)");
        return std::make_shared<StandardAlphas>(StandardAlphas::Type(int(type)), cK, cB, cM, M2_ini);
    }

    /// The model of the Einstein-Boltzmann solver: LCDM (w0 = -1, wa = 0) or w0wa expansion
    inline std::shared_ptr<const PerturbationModel_Horndeski>
    make_horndeski_perturbations(const BACKGROUND::BackgroundParameters & p,
                                 double type,
                                 double cK,
                                 double cB,
                                 double cM,
                                 double M2_ini = 1.0,
                                 double w0 = -1.0,
                                 double wa = 0.0,
                                 bool check_stability = true,
                                 BACKGROUND::BackgroundRange r = {}) {
        auto bg = (w0 == -1.0 and wa == 0.0) ? BACKGROUND::make_lcdm(p, r) : BACKGROUND::make_w0wa(p, w0, wa, r);
        return std::make_shared<const PerturbationModel_Horndeski>(
            bg, make_standard_alphas(type, cK, cB, cM, M2_ini), check_stability);
    }

} // namespace FML::COSMOLOGY::PERTURBATIONS

#endif
