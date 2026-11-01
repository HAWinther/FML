#ifndef FML_COSMOLOGY_PERTURBATIONS_PERTURBEDFIELDEQUATION_GENERALRELATIVITY_H
#define FML_COSMOLOGY_PERTURBATIONS_PERTURBEDFIELDEQUATION_GENERALRELATIVITY_H

// =====================================================================================================================
// General relativity (Newtonian gauge, spatial curvature K, k^2 the Laplacian eigenvalue), one variable Phi:
//   Psi  = -Phi - 3 shear / (c^2 k^2 s_2)
//   Phi' = Psi - (ck/Hp)^2 (1 - 3K/k^2) Phi / 3 + delta rho / (2 Hp^2)
// with the Moments of all species (times H0^2 a^2 / rho_c0). The 1/s_2 converts the symmetric-convention quadrupoles
// to the physical shear (s_2 = 1 when flat). Tensors: M*^2 = 1, alpha_M = alpha_T = 0.
// =====================================================================================================================

#include <FML/Cosmology/Perturbations/PerturbationTypes.h>
#include <FML/Cosmology/Perturbations/PerturbedFieldEquation.h>
#include <FML/Units/Units.h>

#include <string>

namespace FML::COSMOLOGY::PERTURBATIONS {

    class PerturbedFieldEquation_GeneralRelativity : public PerturbedFieldEquation {
      public:
        explicit PerturbedFieldEquation_GeneralRelativity(const PerturbationContext & ctx)
            : PerturbedFieldEquation(ctx) {}
        std::string name() const override { return "GR"; }
        MetricAtX metric(const ModeAtX & b,
                         const double *,
                         const double * y,
                         const Moments & total,
                         double * dydx) const override {
            const double c = FML::UTILS::SI::c, Phi = y[0];
            MetricAtX m;
            m.Phi = Phi;
            m.Psi = -Phi - 3.0 * total.shear / (c * c * b.k * b.k * b.s2);
            m.dPhi = m.Psi - b.ckH * b.ckH * b.curv * Phi / 3.0 + total.drho / (2.0 * b.Hp * b.Hp);
            dydx[0] = m.dPhi;
            return m;
        }
    };

} // namespace FML::COSMOLOGY::PERTURBATIONS

#endif
