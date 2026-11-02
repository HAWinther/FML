#ifndef FML_COSMOLOGY_PERTURBATIONS_PERTURBEDFIELDEQUATION_DGP_H
#define FML_COSMOLOGY_PERTURBATIONS_PERTURBEDFIELDEQUATION_DGP_H

// =====================================================================================================================
// DGP braneworld gravity (the background: Background/FieldEquation_DGP.h, whose Omega_rc and branch eps are read here)
// in the quasi-static approximation (Lue, Scoccimarro & Starkman 2004; Koyama & Maartens 2006): the brane-bending mode
// strengthens (nDGP) or weakens (sDGP) the force on matter and does not change the lensing potential,
//   mu = 1 + 1/(3 beta),  beta = 1 - 2 eps (H r_c/c)(1 + Hdot/(3 H^2)) = 1 - eps (E + E'/3) / sqrt(Omega_rc),  Sigma = 1
// from QuasiStaticGravity_DGP.h (E of the background: the DGP Friedmann equation, or LCDM with the LCDM-background
// option), with the mu-Sigma equations (PerturbedFieldEquation_MuSigma.h). The quasi-static approximation is used on
// all scales (as in most Boltzmann-code implementations).
// =====================================================================================================================

#include <FML/Cosmology/Background/FieldEquation_DGP.h>
#include <FML/Cosmology/Perturbations/PerturbationTypes.h>
#include <FML/Cosmology/Perturbations/PerturbedFieldEquation_MuSigma.h>
#include <FML/Cosmology/Perturbations/QuasiStaticGravity_DGP.h>

#include <cmath>
#include <memory>
#include <sstream>
#include <string>

namespace FML::COSMOLOGY::PERTURBATIONS {

    class PerturbedFieldEquation_DGP : public PerturbedFieldEquation_MuSigma {
      public:
        PerturbedFieldEquation_DGP(const BACKGROUND::DGPFieldEquations & dgp, const PerturbationContext & ctx)
            : PerturbedFieldEquation_MuSigma(ctx,
                                             std::make_shared<const QuasiStaticGravity_DGP>(
                                                 ctx.background, 0.5 / std::sqrt(dgp.get_Omega_rc()), dgp.get_branch())),
              lcdm_bg(dgp.get_lcdm_background()) {}
        std::string name() const override { return qs->name(); }
        std::string info() const override {
            std::ostringstream os;
            os << qs->info() << (lcdm_bg ? " (LCDM background)" : "");
            return os.str();
        }
        double beta(double x) const { return static_cast<const QuasiStaticGravity_DGP &>(*qs).beta(x); }

      private:
        bool lcdm_bg;
    };

} // namespace FML::COSMOLOGY::PERTURBATIONS

#endif
