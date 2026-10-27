#ifndef FML_COSMOLOGY_MODELS_GENERALRELATIVITY_H
#define FML_COSMOLOGY_MODELS_GENERALRELATIVITY_H

// =============================================================================
// General relativity.
//   Background:     E^2 = sum_i rho_i / rho_c0 + Omega_K e^{-2x}
//   Perturbations (Newtonian gauge, spatial curvature K, k^2 the Laplacian eigenvalue):
//     Psi  = -Phi - 3 shear / (c^2 k^2 s_2)
//     Phi' = Psi - (ck/Hp)^2 (1 - 3K/k^2) Phi / 3 + delta rho / (2 Hp^2)
//   with the Moments of all species (times H0^2 a^2 / rho_c0). The 1/s_2 converts the
//   symmetric-convention quadrupoles to the physical shear (s_2 = 1 when flat).
// =============================================================================

#include <FML/Cosmology/Background/Gravity.h>

namespace FML::COSMOLOGY {

class GeneralRelativity : public Gravity {
  public:
    std::string name() const override { return "GR"; }
    double E2(double x, const BackgroundCosmology& bg) const override;
    double dE2dx(double x, const BackgroundCosmology& bg) const override;
    double ddE2ddx(double x, const BackgroundCosmology& bg) const override;
    MetricAtX metric(const ModeAtX& b, const double* cache, const double* y, const Moments& total,
                     double* dydx) const override;
};

} // namespace FML::COSMOLOGY

#endif
