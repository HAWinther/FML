#ifndef FML_COSMOLOGY_MODELS_MUSIGMA_H
#define FML_COSMOLOGY_MODELS_MUSIGMA_H

// =============================================================================
// Phenomenological modified gravity with two functions of time, mu(a) and Sigma(a), in the form used by
// Planck (2015 XIV, 2018 VI) and DES (Y1/Y3 extensions). With the comoving density rho Delta of all species
// (Moments: rho Delta = delta rho + 3 (Hp/ck) flux) and in the Planck sign convention for the spatial
// potential (Phi_P = -Phi here):
//   k^2 Psi           = -4 pi G a^2 mu    rho Delta
//   k^2 (Phi_P + Psi) = -8 pi G a^2 Sigma rho Delta
// mu modifies the force on non-relativistic matter (growth), Sigma the lensing (Weyl) potential.
// Anisotropic stress enters as in GR (it is negligible when mu, Sigma differ from 1).
//
// In terms of the GR potential Phi_GR = (3/2) H0^2 a^2 rho Delta / (rho_c0 c^2 k^2) and S = 3 shear/(c^2 k^2):
//   Psi = -mu Phi_GR - S,     Phi = (2 Sigma - mu) Phi_GR
// Planck/DES time dependence:  mu = 1 + mu0 Omega_DE(a)/Omega_DE,0,  Sigma = 1 + Sigma0 Omega_DE(a)/Omega_DE,0
// (mu0 and Sigma0 are DES's mu0 and Sigma0, and Planck's mu0 - 1 and Sigma0 - 1).
//
// Evolution: the ODE variable is Phi_GR. Computing it from rho Delta directly would lose all precision on
// super-horizon scales (rho Delta ~ (k/Hp)^2 delta rho), so it is evolved with the equation that follows
// from energy-momentum conservation of all species (valid for any metric; d/dx, all densities times H0^2 a^2):
//   (rho Delta)' = -rho Delta - (ck/Hp) flux + 3 (Hp'/Hp - 1) flux/(ck/Hp) - 2 shear + 3 (rho + p) (Psi - Phi')
// With Phi' = (2 Sigma - mu)' Phi_GR + (2 Sigma - mu) Phi_GR' this is linear in Phi_GR', which gives
//   Phi_GR' [1 + 3 A (rho+p)(2 Sigma - mu)] = -Phi_GR + A [-(ck/Hp) flux + 3 (Hp'/Hp - 1) flux/(ck/Hp) - 2 shear
//                                             + 3 (rho+p) Psi - 3 (rho+p) (2 Sigma - mu)' Phi_GR]
// with A = 3/(2 c^2 k^2). For mu = Sigma = 1 this is the GR momentum constraint, Phi' = Psi - G with
// G = 3 flux/(2 (ck/Hp) Hp^2). The code evolves the GR energy constraint (as GeneralRelativity) plus the
// difference of the two expressions above, so that mu = Sigma = 1 gives exactly GR; the two GR forms only
// differ by violations of the Poisson constraint (tight coupling, initial conditions), which this one damps.
// Requires a flat universe and 2 Sigma - mu >= 0 (else Phi_GR' is singular for some k).
//
// mu and Sigma may depend on k (each mode is evolved separately, the derivative of 2 Sigma - mu is at fixed k).
// Subclasses (DGP, Hu-Sawicky f(R)) override mu_sigma() and, if needed, the background.
// =============================================================================

#include <FML/Cosmology/Models/GeneralRelativity.h>

namespace FML::COSMOLOGY {

    class MuSigmaGravity : public GeneralRelativity {
      public:
        MuSigmaGravity(double mu0 = 0.0, double Sigma0 = 0.0) : mu0(mu0), Sigma0(Sigma0) {}
        std::string name() const override { return "mu-Sigma"; }
        std::string info() const override;
        void init_background(const BackgroundCosmology & bg) override;

        // mu, Sigma and d(2 Sigma - mu)/dx (at fixed k) at x for the mode k [1/m] (k = 0: scale-independent value)
        struct MuSigma {
            double mu = 1.0, Sigma = 1.0, dg = 0.0;
        };
        virtual MuSigma mu_sigma(double x, double k, const BackgroundCosmology & bg) const;

        enum { MU, SIGMA, DG, RPP, DHPH, N_CACHE };
        int cache_size() const override { return N_CACHE; }
        void update_cache(const BackgroundCosmology & bg, const ModeAtX & b, double * cache) const override;
        MetricAtX metric(const ModeAtX & b,
                         const double * cache,
                         const double * y,
                         const Moments & total,
                         double * dydx) const override;

      protected:
        double mu0, Sigma0;
    };

} // namespace FML::COSMOLOGY

#endif
