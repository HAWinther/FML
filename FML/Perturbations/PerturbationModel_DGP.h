#ifndef FML_PERTURBATIONS_PERTURBATIONMODEL_DGP_H
#define FML_PERTURBATIONS_PERTURBATIONMODEL_DGP_H

// =====================================================================================================================
// DGP: the standard species (StandardSpeciesPerturbations; Lambda smooth) with the quasi-static perturbed field
// equations of PerturbedFieldEquation_DGP.h, on the background of make_dgp (Background/BackgroundModel_DGP.h, whose
// DGPFieldEquations are kept as a typed pointer). Flat. As the Einstein-Boltzmann solver, a negative cosmological
// constant (allowed, but most likely not intended) is warned about once per process.
//   make_dgp_perturbations(p, Omega_rc, branch, lcdm_background, r)
// =====================================================================================================================

#include <FML/Background/BackgroundModel_DGP.h>
#include <FML/Background/ComponentBackground.h>
#include <FML/Background/FieldEquation_DGP.h>
#include <FML/Perturbations/PerturbationModel.h>
#include <FML/Perturbations/PerturbedFieldEquation_DGP.h>

#include <atomic>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>

namespace FML::COSMOLOGY::PERTURBATIONS {

    class PerturbationModel_DGP : public PerturbationModel {
      public:
        explicit PerturbationModel_DGP(BACKGROUND::ComponentBackgroundPtr background)
            : PerturbationModel(std::move(background)), standard(*bg),
              dgp(bg->field_equations_as<BACKGROUND::DGPFieldEquations>()) {
            if (not dgp)
                throw std::invalid_argument("[PerturbationModel_DGP] The background is not DGP");
            if (bg->parameters().Omega_k != 0.0)
                throw std::invalid_argument("[PerturbationModel_DGP] Only implemented for a flat universe");
            const double OmL = bg->rho(BACKGROUND::Component::DarkEnergy, 0.0);
            static std::atomic<bool> warned{false};
            if (not dgp->get_lcdm_background() and OmL < -1e-12 and not warned.exchange(true))
                std::cerr << "Warning: dgp: Omega_Lambda = " << OmL << " < 0 for Omega_rc = " << dgp->get_Omega_rc()
                          << (dgp->get_branch() == 1 ? " (self-accelerating branch; Omega_rc = 0 selects the value "
                                                       "without dark energy)\n" :
                                                       "\n");
        }
        std::string name() const override { return "dgp"; }
        const BACKGROUND::DGPFieldEquations & field_equations() const { return *dgp; }

        SpeciesPerturbationList species_perturbations(const PerturbationContext & ctx) const override {
            check(ctx);
            SpeciesPerturbationList list;
            standard.add_to(ctx, list);
            return list;
        }
        std::unique_ptr<PerturbedFieldEquation>
        field_equation_perturbations(const PerturbationContext & ctx) const override {
            check(ctx);
            return std::make_unique<PerturbedFieldEquation_DGP>(*dgp, ctx);
        }

      private:
        StandardSpeciesPerturbations standard;
        const BACKGROUND::DGPFieldEquations * dgp;
    };

    inline std::shared_ptr<const PerturbationModel_DGP> make_dgp_perturbations(const BACKGROUND::BackgroundParameters & p,
                                                                               double Omega_rc,
                                                                               int branch,
                                                                               bool lcdm_background = false,
                                                                               BACKGROUND::BackgroundRange r = {}) {
        return std::make_shared<const PerturbationModel_DGP>(BACKGROUND::make_dgp(p, Omega_rc, branch, lcdm_background, r));
    }

} // namespace FML::COSMOLOGY::PERTURBATIONS

#endif
