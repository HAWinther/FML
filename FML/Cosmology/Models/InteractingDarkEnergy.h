#ifndef FML_COSMOLOGY_MODELS_INTERACTINGDARKENERGY_H
#define FML_COSMOLOGY_MODELS_INTERACTINGDARKENERGY_H

// =============================================================================
// Interacting dark energy (Valiviita, Majerotto & Maartens 2008; Gavela et al. 2009, 2010): a dark-energy fluid
// with constant w and rest-frame sound speed cs2 exchanging energy with the cold dark matter,
//   nabla_mu T^mu_nu(c) = Q_nu,   nabla_mu T^mu_nu(DE) = -Q_nu,   Q^mu = Q u_c^mu,   Q = xi H rho_DE
// (the transfer is along the dark-matter 4-velocity, so the dark matter feels no drag). Background
// (x = ln a; xi > 0: energy flows to the dark matter):
//   rho_DE = Omega_DE a^{-3(1+w) - xi}
//   rho_c  = a^{-3} [Omega_c + xi Omega_DE (1 - a^{-(3w+xi)})/(3w + xi)]
// The perturbed rate uses the local expansion of the dark matter, H -> Theta_c/3, so that in Newtonian gauge
// (conformal time, our Phi = -Phi_MB):
//   delta Q / Q = delta_DE - Psi + (theta_c - 3 Phi_MB') / (3 Hp)
// and, with E = (theta_c - 3 Phi_MB')/(3 Hp), u = theta_DE/(ck), v_c = -theta_c/(ck), d = xi/(3(1 + w)):
//   delta_c'  = (ck/Hp) v_c - 3 Phi' + xi (rho_DE/rho_c) (delta_DE - delta_c + E)
//   v_c'      = -v_c - (ck/Hp) Psi                                         (no momentum transfer)
// (the last terms of u' make the total momentum conserved: theta_DE' = ... + xi Hp [theta_DE - theta_c/(1 + w)])
//   delta_DE' = -(1 + w)((ck/Hp) u + 3 Phi') - 3 (cs2 - w)[delta_DE + 3 (1 + w)(1 + d) u/(ck/Hp)] - xi E
//   u'        = [3 cs2 - 1 + 3 d (cs2 - w)] u + (ck/Hp)[cs2 delta_DE/(1 + w) + Psi] + xi u + 3 d v_c
// with delta p_DE = rho_DE [cs2 delta_DE + 3 (1 + w)(1 + d)(cs2 - w) u/(ck/Hp)] (the rest-frame transformation
// uses rho_DE' including the transfer). The large-scale instability of these models (the "doom factor"
// d > 0) is avoided for d <= 0, i.e. xi and 1 + w of opposite signs. xi = 0 is wCDM.
// =============================================================================

#include <FML/Cosmology/Background/Species.h>

#include <cmath>
#include <memory>

namespace FML::COSMOLOGY {

    class Perturbations;

    class InteractingDarkEnergy : public Species {
      public:
        InteractingDarkEnergy(double Omega_DE, double Omega_c, double w, double xi, double cs2);
        std::string name() const override { return "interacting dark energy"; }
        Role role() const override { return Role::DarkEnergy; }
        std::string info() const override;
        double rho(double x) const override { return Omega0 * std::exp(-L * x); }
        double p(double x) const override { return w * rho(x); }
        double drhodx(double x) const override { return -L * rho(x); }
        double ddrhoddx(double x) const override { return L * L * rho(x); }
        // Perturbations of dark matter and dark energy together (the dark matter species must not have its own)
        std::unique_ptr<SpeciesPerturbations> perturbations(const Perturbations & pert) const override;

        double doom() const { return xi / (3.0 * (1.0 + w)); }
        double Omega0, Omega_c, w, xi, cs2;
        double L; // -dln rho_DE/dx = 3 (1 + w) + xi
    };

    // The dark matter of the model (background only; its perturbations are in InteractingDarkEnergy's block)
    class InteractingCDM : public Species {
      public:
        explicit InteractingCDM(std::shared_ptr<const InteractingDarkEnergy> de) : de(std::move(de)) {}
        std::string name() const override { return "cdm (interacting)"; }
        Role role() const override { return Role::CDM; }
        double rho(double x) const override;
        double p(double) const override { return 0.0; }
        double drhodx(double x) const override { return -3.0 * rho(x) + de->xi * de->rho(x); }
        double ddrhoddx(double x) const override { return -3.0 * drhodx(x) + de->xi * de->drhodx(x); }

      private:
        std::shared_ptr<const InteractingDarkEnergy> de;
    };

} // namespace FML::COSMOLOGY

#endif
