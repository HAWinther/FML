#ifndef FML_COSMOLOGY_MODELS_DGP_H
#define FML_COSMOLOGY_MODELS_DGP_H

// =============================================================================
// DGP braneworld gravity (Dvali, Gabadadze & Porrati 2000), flat, with the crossover scale r_c,
//   Omega_rc = c^2 / (4 r_c^2 H0^2).
// Background (modified Friedmann equation on the brane):
//   H^2 - eps H c/r_c = 8 pi G rho / 3   ->   E = eps sqrt(Omega_rc) + sqrt(Omega_rc + rho/rho_c0)
//   eps = +1: self-accelerating branch (sDGP; accelerates without dark energy)
//   eps = -1: normal branch (nDGP; needs dark energy, here Lambda)
// Omega_Lambda follows from E(0) = 1:  Omega_Lambda = 1 - 2 eps sqrt(Omega_rc) - Omega_m - Omega_r.
//
// With lcdm_background = true the expansion is instead that of LCDM (GR with Lambda from closure) and only
// the perturbations are modified, with beta computed from that H(a): the "nDGP with LCDM background" used in
// most N-body work (Schmidt 2009), where the dark energy that makes up the difference is not perturbed.
//
// Perturbations in the quasi-static approximation (Lue, Scoccimarro & Starkman 2004; Koyama & Maartens 2006):
// the brane-bending mode strengthens (nDGP) or weakens (sDGP) the force on matter, and does not change
// the lensing potential:
//   mu = 1 + 1/(3 beta),  beta = 1 - 2 eps (H r_c/c) (1 + Hdot/(3 H^2)) = 1 - eps E (1 + dlnH/dx / 3) / sqrt(Omega_rc)
//   Sigma = 1
// (the mu-Sigma equations of MuSigma.h). The quasi-static approximation is not valid on super-horizon
// scales, where it is nevertheless used here (as in most Boltzmann-code implementations).
// =============================================================================

#include <FML/Cosmology/Models/MuSigma.h>

#include <cmath>

namespace FML::COSMOLOGY {

    class DGPGravity : public MuSigmaGravity {
      public:
        DGPGravity(double Omega_rc, int branch, bool lcdm_background = false);
        std::string name() const override { return branch > 0 ? "sDGP" : "nDGP"; }
        std::string info() const override;
        void init_background(const BackgroundCosmology & bg) override;

        double E2(double x, const BackgroundCosmology & bg) const override;
        double dE2dx(double x, const BackgroundCosmology & bg) const override;
        double ddE2ddx(double x, const BackgroundCosmology & bg) const override;
        MuSigma mu_sigma(double x, double k, const BackgroundCosmology & bg) const override;
        double beta(double x, const BackgroundCosmology & bg) const;

        // Omega_Lambda from E(0) = 1, given the other densities today
        static double OmegaLambda(double Omega_rc, int branch, double rho_other0) {
            return 1.0 - 2.0 * branch * std::sqrt(Omega_rc) - rho_other0;
        }

      private:
        double Omega_rc, sq; // sq = sqrt(Omega_rc)
        int branch;
        bool lcdm_bg;
        // E and its first two derivatives
        void E_derivs(double x, const BackgroundCosmology & bg, double & E, double & dE, double & ddE) const;
    };

} // namespace FML::COSMOLOGY

#endif
