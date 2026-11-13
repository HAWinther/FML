#ifndef FML_COSMOLOGY_PERTURBATIONS_PERTURBATIONMODEL_QUASISTATIC_H
#define FML_COSMOLOGY_PERTURBATIONS_PERTURBATIONMODEL_QUASISTATIC_H

// =====================================================================================================================
// A theory of gravity given by its quasi-static limit (QuasiStaticGravity.h): the standard species
// (StandardSpeciesPerturbations) on the LCDM background with the mu-Sigma perturbed field equations
// (PerturbedFieldEquation_MuSigma.h) of the theory's mu(k, a) and Sigma(k, a). It is the linear model of the
// chameleon-like theories whose background is LCDM to a good approximation (the scalar's energy density is
// O(beta^2 phi^2), negligible) and whose fifth force has a range much shorter than the horizon (the quasi-static
// limit; as MGCAMB, and the force of SimulationCode):
//   symmetron   QuasiStaticGravity_Symmetron.h   sym_assb, sym_beta, sym_L_mpch, sym_N, sym_M
//   dilaton     QuasiStaticGravity_Dilaton.h     dil_A2, dil_beta0, dil_r, dil_xi
//   mbeta       QuasiStaticGravity_MBeta.h       mb_beta0, mb_n_beta, mb_m0, mb_n_m
// (Hu-Sawicky f(R), PerturbationModel_HuSawicky.h, is the same with its own field equation for the tensor modes.)
// The tensors are those of GR (the conformal factor is 1 + O(beta phi)). Flat, with dark energy. The modification
// grows with k up to the scalar's mass: set the k range of the perturbations high enough for P(k).
//   make_quasistatic_perturbations(name, background, factory)
// =====================================================================================================================

#include <FML/Cosmology/Background/BackgroundModel_LCDM.h>
#include <FML/Cosmology/Background/ComponentBackground.h>
#include <FML/Cosmology/Background/FieldEquation_GeneralRelativity.h>
#include <FML/Cosmology/Perturbations/PerturbationModel.h>
#include <FML/Cosmology/Perturbations/PerturbationModel_MuSigma.h>
#include <FML/Cosmology/Perturbations/PerturbedFieldEquation_MuSigma.h>
#include <FML/Cosmology/Perturbations/QuasiStaticGravity.h>

#include <functional>
#include <memory>
#include <stdexcept>
#include <string>

namespace FML::COSMOLOGY::PERTURBATIONS {

    class PerturbationModel_QuasiStatic : public PerturbationModel {
      public:
        /// Makes the theory on a background (that of the perturbations' context)
        using Factory = std::function<QuasiStaticGravityPtr(const BACKGROUND::Background &)>;

        PerturbationModel_QuasiStatic(BACKGROUND::ComponentBackgroundPtr background,
                                      std::string model_name,
                                      Factory factory)
            : PerturbationModel(std::move(background)), standard(*bg), model_name(std::move(model_name)),
              factory(std::move(factory)) {
            if (not bg->field_equations_as<BACKGROUND::GeneralRelativity>())
                throw std::invalid_argument("[PerturbationModel_QuasiStatic] " + this->model_name +
                                            ": the background is not general relativity");
            check_musigma_background(*bg, this->model_name);
            this->factory(*bg); // checks the parameters
        }
        std::string name() const override { return model_name; }

        SpeciesPerturbationList species_perturbations(const PerturbationContext & ctx) const override {
            check(ctx);
            SpeciesPerturbationList list;
            standard.add_to(ctx, list);
            return list;
        }
        std::unique_ptr<PerturbedFieldEquation>
        field_equation_perturbations(const PerturbationContext & ctx) const override {
            check(ctx);
            return std::make_unique<PerturbedFieldEquation_MuSigma>(ctx, factory(ctx.background));
        }

      private:
        StandardSpeciesPerturbations standard;
        std::string model_name;
        Factory factory;
    };

    inline std::shared_ptr<const PerturbationModel_QuasiStatic>
    make_quasistatic_perturbations(std::string name,
                                   BACKGROUND::ComponentBackgroundPtr bg,
                                   PerturbationModel_QuasiStatic::Factory factory) {
        return std::make_shared<const PerturbationModel_QuasiStatic>(std::move(bg),
                                                                     std::move(name),
                                                                     std::move(factory));
    }

} // namespace FML::COSMOLOGY::PERTURBATIONS

#endif
