#ifndef FML_COSMOLOGY_PERTURBATIONS_PERTURBATIONMODEL_MUSIGMA_H
#define FML_COSMOLOGY_PERTURBATIONS_PERTURBATIONMODEL_MUSIGMA_H

// =====================================================================================================================
// mu-Sigma (Planck/DES): the standard species (StandardSpeciesPerturbations) on the LCDM background with the
// perturbed field equations of PerturbedFieldEquation_MuSigma.h, mu = 1 + mu0 Omega_DE(a)/Omega_DE,0 and Sigma = 1 +
// Sigma0 Omega_DE(a)/Omega_DE,0. Flat; needs dark energy and 1 + 2 Sigma0 - mu0 >= 0 (2 Sigma - mu >= 0), as the
// Einstein-Boltzmann solver.
//   make_musigma_perturbations(p, mu0, Sigma0, r)
// =====================================================================================================================

#include <FML/Cosmology/Background/BackgroundModel_LCDM.h>
#include <FML/Cosmology/Background/ComponentBackground.h>
#include <FML/Cosmology/Background/FieldEquation_GeneralRelativity.h>
#include <FML/Cosmology/Perturbations/PerturbationModel.h>
#include <FML/Cosmology/Perturbations/PerturbedFieldEquation_MuSigma.h>

#include <memory>
#include <stdexcept>
#include <string>

namespace FML::COSMOLOGY::PERTURBATIONS {

    /// The checks of the mu-Sigma equations on a background (flat, with dark energy)
    inline void check_musigma_background(const BACKGROUND::ComponentBackground & bg, const std::string & name) {
        if (bg.parameters().Omega_k != 0.0)
            throw std::invalid_argument(name + ": only implemented for a flat universe");
        if (not(bg.rho(BACKGROUND::Component::DarkEnergy, 0.0) > 0.0))
            throw std::invalid_argument(name + ": needs dark energy (Omega_DE > 0)");
    }

    class PerturbationModel_MuSigma : public PerturbationModel {
      public:
        PerturbationModel_MuSigma(BACKGROUND::ComponentBackgroundPtr background, double mu0, double Sigma0)
            : PerturbationModel(std::move(background)), standard(*bg), mu0(mu0), Sigma0(Sigma0) {
            if (not bg->field_equations_as<BACKGROUND::GeneralRelativity>())
                throw std::invalid_argument("[PerturbationModel_MuSigma] The background is not general relativity");
            check_musigma_background(*bg, "mu-Sigma");
            // Omega_DE(a)/Omega_DE,0 is in [0, 1] in the past, so 2 Sigma - mu >= 0 iff 1 + 2 Sigma0 - mu0 >= 0
            if (1.0 + 2.0 * Sigma0 - mu0 < 0.0)
                throw std::invalid_argument("mu-Sigma: need 1 + 2 Sigma0 - mu0 >= 0 (2 Sigma - mu >= 0)");
        }
        std::string name() const override { return "musigma"; }

        SpeciesPerturbationList species_perturbations(const PerturbationContext & ctx) const override {
            check(ctx);
            SpeciesPerturbationList list;
            standard.add_to(ctx, list);
            return list;
        }
        std::unique_ptr<PerturbedFieldEquation>
        field_equation_perturbations(const PerturbationContext & ctx) const override {
            check(ctx);
            return std::make_unique<PerturbedFieldEquation_MuSigma>(ctx, mu0, Sigma0);
        }

      private:
        StandardSpeciesPerturbations standard;
        double mu0, Sigma0;
    };

    inline std::shared_ptr<const PerturbationModel_MuSigma> make_musigma_perturbations(
        const BACKGROUND::BackgroundParameters & p, double mu0, double Sigma0, BACKGROUND::BackgroundRange r = {}) {
        return std::make_shared<const PerturbationModel_MuSigma>(BACKGROUND::make_lcdm(p, r), mu0, Sigma0);
    }

} // namespace FML::COSMOLOGY::PERTURBATIONS

#endif
