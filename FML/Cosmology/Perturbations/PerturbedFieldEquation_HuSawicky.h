#ifndef FML_COSMOLOGY_PERTURBATIONS_PERTURBEDFIELDEQUATION_HUSAWICKY_H
#define FML_COSMOLOGY_PERTURBATIONS_PERTURBEDFIELDEQUATION_HUSAWICKY_H

// =====================================================================================================================
// Hu-Sawicky f(R) gravity (Hu & Sawicki 2007), S = int sqrt(-g) [R + f(R)] / (16 pi G) + S_m, in the high-curvature
// regime fixed by |f_R0| and n, on the (LCDM) background (the O(|f_R0|) corrections to the expansion are neglected, as
// MGCAMB). The quasi-static perturbations (Pogosian & Silvestri 2008)
//   mu(k, a) = (1 + 4 Q) / ((1 + 3 Q)(1 + f_R)),   Sigma(a) = 1 / (1 + f_R),   Q = (k/a)^2 f_RR / (1 + f_R)
// are those of QuasiStaticGravity_HuSawicky.h (f_R, f_RR and R from there), with the mu-Sigma equations
// (PerturbedFieldEquation_MuSigma.h). Tensors: M*^2 = 1 + f_R, alpha_M = -(n+1) f_R dlnR/dx / (1 + f_R) (dlnR/dx by a
// central difference, step 1e-4). The modification grows with k: set the k range of the perturbations high enough for
// P(k).
// =====================================================================================================================

#include <FML/Cosmology/Perturbations/PerturbationTypes.h>
#include <FML/Cosmology/Perturbations/PerturbedFieldEquation_MuSigma.h>
#include <FML/Cosmology/Perturbations/QuasiStaticGravity_HuSawicky.h>
#include <FML/Units/Units.h>

#include <cmath>
#include <memory>
#include <string>

namespace FML::COSMOLOGY::PERTURBATIONS {

    class PerturbedFieldEquation_HuSawicky : public PerturbedFieldEquation_MuSigma {
      public:
        PerturbedFieldEquation_HuSawicky(double fR0, double n, const PerturbationContext & ctx)
            : PerturbedFieldEquation_MuSigma(ctx,
                                             std::make_shared<const QuasiStaticGravity_HuSawicky>(ctx.background, fR0, n)),
              hs(static_cast<const QuasiStaticGravity_HuSawicky &>(*qs)) {}
        std::string name() const override { return "Hu-Sawicky f(R)"; }

        /// Ricci scalar [1/m^2], f_R, f_RR [m^2] and the Compton wavelength [m] of the background
        double R(double x) const { return 3.0 * ctx.H0 * ctx.H0 * hs.R(x) / (c() * c()); }
        double fR(double x) const { return hs.fR(x); }
        double fRR(double x) const { return -(hs.get_n() + 1.0) * fR(x) / R(x); }
        double compton_wavelength(double x) const { return hs.compton_wavelength(x) * c() / ctx.H0; }

        TensorCoefficients tensor(double x) const override {
            const double h = 1e-4;
            const double dlnR = (std::log(hs.R(x + h)) - std::log(hs.R(x - h))) / (2.0 * h);
            const double f = fR(x);
            return {1.0 + f, -(hs.get_n() + 1.0) * f * dlnR / (1.0 + f), 0.0};
        }

      private:
        const QuasiStaticGravity_HuSawicky & hs;
        static double c() { return FML::UTILS::SI::c; }
    };

} // namespace FML::COSMOLOGY::PERTURBATIONS

#endif
