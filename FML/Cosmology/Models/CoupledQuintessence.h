#ifndef FML_COSMOLOGY_MODELS_COUPLEDQUINTESSENCE_H
#define FML_COSMOLOGY_MODELS_COUPLEDQUINTESSENCE_H

// =============================================================================
// Coupled quintessence (Wetterich 1995, Amendola 2000): a scalar field phi (reduced Planck masses) with
// V = 3 H0^2 M_pl^2 V0 exp(-lambda phi) that exchanges energy with the cold dark matter, whose particles
// have the mass m(phi) ~ exp(-beta phi):
//   rho_c' = -(3 + beta phi_x) rho_c,     phi_xx = -(3 + dlnE/dx) phi_x - 3 v_phi/E^2 + 3 beta rho_c/E^2
// (x = ln a, densities in units of rho_c0, v = V/(3 H0^2 M_pl^2)); E^2 = (rho_o + rho_c + v)/(1 - phi_x^2/6).
// beta here is Amendola's sqrt(2/3) beta. The field starts at rest in the radiation era; V0 and the CDM
// normalisation are found by iteration so that Omega_c and Omega_phi today are those of the parameters
// (OmegaCDM and closure). beta = 0 is the quintessence model, beta = lambda = 0 LCDM.
//
// Perturbations (Newtonian gauge), CDM and field in one block, (delta_c, v_c, dphi, w = dphi_x - phi_x Psi):
//   delta_c' = (ck/Hp) v_c - 3 Phi' - beta dphi_x                 (number conservation, rho_c = m(phi) n)
//   v_c'     = -(1 - beta phi_x) v_c - (ck/Hp) Psi + beta (ck/Hp) dphi    (fifth force + drag)
//   dphi_x   = w + phi_x Psi
//   w'       = -(2 + Hp'/Hp) w - (ck/Hp)^2 dphi - 3 v_phiphi dphi/E^2 - 3 phi_x Phi'
//              - 3 (v_phi - beta rho_c) Psi/E^2 + 3 beta rho_c delta_c/E^2
// On small scales the CDM feels G_eff = G (1 + 2 beta^2). The field's density, pressure and momentum are
// those of an uncoupled field (Quintessence.h); the exchange cancels in the total.
// =============================================================================

#include <FML/Cosmology/Background/Species.h>
#include <FML/Cosmology/Common.h>

#include <cmath>
#include <memory>
#include <vector>

namespace FML::COSMOLOGY {

// The background solution shared by the two species
class CoupledDarkSector {
  public:
    // others: all species except CDM and dark energy
    CoupledDarkSector(double lambda, double beta, const std::vector<std::shared_ptr<const Species>>& others,
                      double OmegaK, double Omega_c, double Omega_phi);

    double phi(double x) const;
    double dphi(double x) const;
    double ddphi(double x) const;
    double v(double f) const { return V0 * std::exp(-lambda * f); }
    double v_phi(double f) const { return -lambda * v(f); }
    double v_phiphi(double f) const { return lambda * lambda * v(f); }
    double rho_c(double x) const;
    double rho_phi(double x) const;
    double p_phi(double x) const;
    double E2(double x) const;
    double dlnE(double x) const;
    double lambda, beta;
    double V0 = 0.0, C = 0.0;   // rho_c = C exp(-3x - beta phi)

  private:
    double OmegaK;
    std::vector<std::shared_ptr<const Species>> others;
    double x_ini = -25.0, x_fin = 10.0;
    Spline1D phi_spline, dphi_spline;

    double rho_o(double x) const;
    double drho_o(double x) const;
    double E2(double x, double f, double df) const;
    double dlnE(double x, double f, double df) const;
    double ddphi(double x, double f, double df) const;
    void solve();
};

class CoupledCDM : public Species {
  public:
    explicit CoupledCDM(std::shared_ptr<const CoupledDarkSector> d) : d(std::move(d)) {}
    std::string name() const override { return "cdm (coupled)"; }
    Role role() const override { return Role::CDM; }
    double rho(double x) const override { return d->rho_c(x); }
    double p(double) const override { return 0.0; }
    double drhodx(double x) const override;
    double ddrhoddx(double x) const override;
    // The perturbations are part of the field's block (CoupledField)

  private:
    std::shared_ptr<const CoupledDarkSector> d;
};

class CoupledField : public Species {
  public:
    explicit CoupledField(std::shared_ptr<const CoupledDarkSector> d) : d(std::move(d)) {}
    std::string name() const override { return "coupled quintessence"; }
    Role role() const override { return Role::DarkEnergy; }
    std::string info() const override;
    double rho(double x) const override { return d->rho_phi(x); }
    double p(double x) const override { return d->p_phi(x); }
    double drhodx(double x) const override;
    double ddrhoddx(double x) const override;
    std::unique_ptr<SpeciesPerturbations> perturbations(const Perturbations& pert) const override;
    const CoupledDarkSector& sector() const { return *d; }

  private:
    std::shared_ptr<const CoupledDarkSector> d;
};

} // namespace FML::COSMOLOGY

#endif
