#ifndef FML_COSMOLOGY_MODELS_EARLYDARKENERGY_H
#define FML_COSMOLOGY_MODELS_EARLYDARKENERGY_H

// =============================================================================
// Early dark energy (Poulin, Smith, Karwal & Kamionkowski 2019; Smith et al. 2020; Hill et al. 2020): an
// axion-like scalar field (models/ScalarField.h) with
//   V(phi) = m^2 f^2 (1 - cos(phi/f))^n = 3 H0^2 M_pl^2 mu (1 - cos theta)^n,   theta = phi/f
// frozen at theta_i until H drops to ~ m, after which it oscillates about the minimum and its energy
// redshifts as a^{-6n/(n+1)} (w -> (n-1)/(n+1)). Plus a cosmological constant (from closure, E(0) = 1).
//
// The model is specified by the phenomenological parameters (as in CLASS_EDE / AxiCLASS):
//   f_EDE = max_z rho_EDE/rho_tot,  z_c the redshift of this maximum,  theta_i,  n.
// The potential parameters mu (= m^2 f^2 / (3 H0^2 M_pl^2)) and f (in M_pl) are found by a Newton iteration
// on (ln mu, ln f) on the background solution, and Omega_Lambda by iterating the closure. This needs n > 2
// (for n <= 2 the oscillating field does not dilute faster than radiation and its fraction has no peak at z_c).
// =============================================================================

#include <FML/Cosmology/Models/ScalarField.h>

namespace FML::COSMOLOGY {

class EarlyDarkEnergySpecies : public ScalarFieldSpecies {
  public:
    // others: all other species (no cosmological constant: it is kept by this species and given by OmegaL())
    EarlyDarkEnergySpecies(double f_EDE, double log10_zc, double theta_i, double n,
                           const std::vector<std::shared_ptr<const Species>>& others, double OmegaK);
    std::string name() const override { return "early dark energy"; }
    std::string info() const override;

    double v(double phi) const override;
    double v_phi(double phi) const override;
    double v_phiphi(double phi) const override;

    double mu() const { return mu_; }
    double f() const { return f_; }
    // m / H0 = sqrt(3 mu) / f
    double mass_over_H0() const { return std::sqrt(3.0 * mu_) / f_; }
    double fraction(double x) const { return rho(x) / E2(x); }
    // max rho_EDE/rho_tot and the x where it is reached (from the stored solution)
    void peak(double& f_max, double& x_peak) const;

  private:
    double fEDE, x_c, theta_i, n;
    double mu_ = 0.0, f_ = 1.0;
    double dx = 0.002;     // spacing of the stored solution
    void solve_to(double x_end);
};

} // namespace FML::COSMOLOGY

#endif
