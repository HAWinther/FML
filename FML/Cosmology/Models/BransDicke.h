#ifndef FML_COSMOLOGY_MODELS_BRANSDICKE_H
#define FML_COSMOLOGY_MODELS_BRANSDICKE_H

// =============================================================================
// Jordan-Brans-Dicke gravity (flat), in the Jordan frame with a cosmological constant:
//   S = (1/16 pi G) int sqrt(-g) [phi R - (omega/phi) (d phi)^2] + S_matter
// Field equations (kappa = 8 pi G):
//   phi G_mn = kappa T_mn + (omega/phi)(d_m phi d_n phi - g_mn (d phi)^2/2) + nabla_m nabla_n phi - g_mn box phi
//   box phi  = kappa T / (3 + 2 omega)
// phi today is fixed so that the gravitational constant measured in a Cavendish experiment,
// G_eff = G (4 + 2 omega)/((3 + 2 omega) phi), equals G:  phi_0 = (4 + 2 omega)/(3 + 2 omega).
//
// Background (x = ln a, F = phi, F_x = dphi/dx, densities in units of rho_c0 = 3 H0^2/(8 pi G)):
//   E^2 (F + F_x - omega F_x^2/(6 F)) = rho
//   F_xx = -(3 + dlnE/dx) F_x + 3 (rho - 3p) / ((3 + 2 omega) E^2)
// The field starts at rest deep in the radiation era (where rho - 3p ~ 0). Its initial value and
// Omega_Lambda are found by iteration so that F(0) = phi_0 and E(0) = 1.
//
// Perturbations (Newtonian gauge; derived in python/derive_brans_dicke.py), variables Phi, dF = delta phi and
// w = dF_x - F_x Psi (which removes Psi' from the field equation); Moments times H0^2 a^2 / rho_c0:
//   Psi    = -Phi - dF/F - 3 shear / (c^2 k^2 F)                                          (traceless ij)
//   Phi_x  = [3 drho/Hp^2 + omega F_x w/F - omega dF F_x^2/(2 F^2) + 3 (2 F Psi - dF) + 3 (F_x Psi - w)
//             - (ck/Hp)^2 (2 F Phi + dF)] / (6 F + 3 F_x)                                    (00)
//   dF_x   = w + F_x Psi
//   w_x    = -(2 + Hp'/Hp) w - (ck/Hp)^2 dF + B Psi - 3 F_x Phi_x + 3 (drho - 3 dp) / ((3 + 2 omega) Hp^2)
// with B = F_xx + (2 + Hp'/Hp) F_x. The momentum constraint (0i),
//   -omega dF F_x/F + 2 F Psi + dF - 2 F Phi_x + F_x Psi - dF_x = 3 flux / ((ck/Hp) Hp^2),
// is not used and serves as a test. omega -> infinity is GR.
// =============================================================================

#include <FML/Cosmology/Background/Gravity.h>
#include <FML/Cosmology/Common.h>

#include <memory>
#include <vector>

namespace FML::COSMOLOGY {

class BransDickeGravity : public Gravity {
  public:
    // species: all species except Lambda. Solves the background and sets Omega_Lambda (OmegaLambda()).
    BransDickeGravity(double omega, const std::vector<std::shared_ptr<const Species>>& species);
    std::string name() const override { return "Brans-Dicke"; }
    std::string info() const override;
    void init_background(const BackgroundCosmology& bg) override;

    double OmegaLambda() const { return OmegaL; }
    double phi(double x) const;
    double dphidx(double x) const;
    double phi_today() const { return (4.0 + 2.0 * omega) / (3.0 + 2.0 * omega); }
    // G_eff / G for non-relativistic matter on small scales: (4 + 2 omega)/((3 + 2 omega) phi)
    double G_eff(double x) const { return phi_today() / phi(x); }

    double E2(double x, const BackgroundCosmology& bg) const override;
    double dE2dx(double x, const BackgroundCosmology& bg) const override;
    double ddE2ddx(double x, const BackgroundCosmology& bg) const override;

    int n_vars() const override { return 3; }
    enum { F_, FX, B_, N_CACHE };
    int cache_size() const override { return N_CACHE; }
    void update_cache(const BackgroundCosmology& bg, const ModeAtX& b, double* cache) const override;
    double G_eff_initial() const override { return 1.0 / phi(x_ini); }
    void initial_conditions(const ModeAtX& b, const InitialData& ic, double* y) const override;
    MetricAtX metric(const ModeAtX& b, const double* cache, const double* y, const Moments& total,
                     double* dydx) const override;
    // delta phi / phi and (d delta phi/dx) / phi
    std::vector<std::string> output_names() const override { return {"delta_phi_BD", "ddelta_phi_BD_dx"}; }
    // Tensors: M*^2 = phi, alpha_M = dln phi/dx
    TensorCoefficients tensor(double x, const BackgroundCosmology& /*bg*/) const override {
      return {phi(x), dphidx(x) / phi(x), 0.0};
    }
    void outputs(const ModeAtX& b, const double* cache, const double* y, const MetricAtX& m, double* out) const override;

  private:
    double omega, OmegaL = 0.0;
    std::vector<std::shared_ptr<const Species>> species;
    double x_ini = -25.0, x_fin = 10.0;
    Spline1D F_spline, dF_spline, L_spline;   // F, F_x, dlnE/dx

    // dlnE/dx from the Friedmann equation and the field equation, given rho, rho', p (with Lambda)
    double dlnE(double F, double dF, double rho, double drho, double p) const;
    void sums(double x, double& rho, double& drho, double& p) const;   // species + Lambda
    // Integrate from rest at F_i; returns F(0) and sets the splines
    double solve(double F_i);
};

} // namespace FML::COSMOLOGY

#endif
