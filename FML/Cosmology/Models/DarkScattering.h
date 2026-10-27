#ifndef FML_COSMOLOGY_MODELS_DARKSCATTERING_H
#define FML_COSMOLOGY_MODELS_DARKSCATTERING_H

// =============================================================================
// Dark scattering (Simpson 2010; Baldi & Simpson 2015): elastic scattering of dark matter off a dark-energy
// fluid with constant equation of state w, with cross section per dark-matter mass xi = sigma_D / m_c.
// The background is wCDM (no energy exchange): rho_DE ~ a^{-3(1+w)}. Only the momenta are exchanged:
//   theta_c'  = -Hp theta_c + k^2 Psi + Gamma_c (theta_DE - theta_c),     Gamma_c  = (1 + w) rho_DE/rho_c Gamma_DE
//   theta_DE' = ...                    - Gamma_DE (theta_DE - theta_c),    Gamma_DE = a c xi rho_c
// (conformal time; physical densities), which conserves the total momentum.
//
// Dark energy: a fluid with rest-frame sound speed cs2 (default 1) and c_a^2 = w, Newtonian gauge, with
// u = theta/(ck), x = ln a:
//   delta_DE' = -(1 + w)(ck/Hp u + 3 Phi') - 3 (cs2 - w) [delta_DE + 3 (1 + w) u/(ck/Hp)]
//   u'        = -(1 - 3 cs2) u + (ck/Hp)[cs2 delta_DE/(1 + w) + Psi] - (Gamma_DE/Hp)(u - u_c)
//   delta_c'  = (ck/Hp) v_c - 3 Phi',   v_c' = -v_c - (ck/Hp) Psi + (Gamma_c/Hp)(v_DE - v_c),  v = -u
// delta p_DE = rho_DE [cs2 delta_DE + 3 (1 + w)(cs2 - w) u/(ck/Hp)].
// While Gamma_DE/Hp > 200 (early times) the dark-energy velocity is locked to that of the dark matter
// (mode 0; its momentum is negligible then, (1 + w) rho_DE << rho_c), afterwards both are evolved (mode 1).
// xi = 0 is wCDM with dark-energy perturbations. Requires w != -1.
// =============================================================================

#include <FML/Cosmology/Background/Species.h>

#include <cmath>

namespace FML::COSMOLOGY {

class Perturbations;

class FluidDarkEnergy : public Species {
  public:
    // xi in m^2/kg; rho_c0 the critical density today [kg/m^3] (to convert xi rho_c into a rate)
    FluidDarkEnergy(double Omega0, double w, double cs2, double xi, double rho_crit0);
    std::string name() const override { return "dark energy fluid"; }
    Role role() const override { return Role::DarkEnergy; }
    std::string info() const override;
    double rho(double x) const override { return Omega0 * std::exp(-3.0 * (1.0 + w) * x); }
    double p(double x) const override { return w * rho(x); }
    double drhodx(double x) const override { return -3.0 * (1.0 + w) * rho(x); }
    double ddrhoddx(double x) const override { return 9.0 * (1.0 + w) * (1.0 + w) * rho(x); }
    // Perturbations of dark matter and dark energy together (the dark matter species must not have its own)
    std::unique_ptr<SpeciesPerturbations> perturbations(const Perturbations& pert) const override;

    double Omega0, w, cs2, xi, rho_crit0;
};

} // namespace FML::COSMOLOGY

#endif
