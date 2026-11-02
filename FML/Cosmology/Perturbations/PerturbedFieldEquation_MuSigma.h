#ifndef FML_COSMOLOGY_PERTURBATIONS_PERTURBEDFIELDEQUATION_MUSIGMA_H
#define FML_COSMOLOGY_PERTURBATIONS_PERTURBEDFIELDEQUATION_MUSIGMA_H

// =====================================================================================================================
// Phenomenological modified gravity with two functions mu(k, a) and Sigma(k, a), in the form used by Planck (2015 XIV,
// 2018 VI) and DES. With the comoving density
// rho Delta of all species (Moments: rho Delta = delta rho + 3 (Hp/ck) flux) and the Planck sign of the spatial
// potential (Phi_P = -Phi here):
//   k^2 Psi           = -4 pi G a^2 mu    rho Delta
//   k^2 (Phi_P + Psi) = -8 pi G a^2 Sigma rho Delta
// mu modifies the force on non-relativistic matter (growth), Sigma the lensing (Weyl) potential. Anisotropic stress
// enters as in GR. In terms of Phi_GR = (3/2) H0^2 a^2 rho Delta / (rho_c0 c^2 k^2) and S = 3 shear / (c^2 k^2):
//   Psi = -mu Phi_GR - S,     Phi = (2 Sigma - mu) Phi_GR
// The ODE variable is Phi_GR, evolved with the equation that follows from energy-momentum conservation of all species
// (d/dx, densities times H0^2 a^2, A = 3 / (2 c^2 k^2)):
//   Phi_GR' [1 + 3 A (rho+p)(2 Sigma - mu)] = -Phi_GR + A [-(ck/Hp) flux + 3 (Hp'/Hp - 1) flux/(ck/Hp) - 2 shear
//                                             + 3 (rho+p) Psi - 3 (rho+p) (2 Sigma - mu)' Phi_GR]
// to which the GR energy constraint minus its momentum-constraint form is added, so that mu = Sigma = 1 is exactly the
// GR code (the two differ only by violations of the Poisson constraint, which this damps). Requires a flat universe
// and 2 Sigma - mu >= 0 (checked by the models).
//
// mu and Sigma are those of a QuasiStaticGravity object (QuasiStaticGravity.h: the one definition of the theory's
// quasi-static limit, shared with SimulationCode); mu_sigma(x, k) gives mu, Sigma and d(2 Sigma - mu)/dx at fixed k
// [1/m]. The default is the Planck/DES form (QuasiStaticGravity_MuSigma.h)
//   mu = 1 + mu0 Omega_DE(a)/Omega_DE,0,   Sigma = 1 + Sigma0 Omega_DE(a)/Omega_DE,0
// and DGP and Hu-Sawicky f(R) pass theirs (PerturbedFieldEquation_DGP.h, _HuSawicky.h). Cache: mu, Sigma,
// d(2 Sigma - mu)/dx, H0^2 a^2 (rho + p) of all species, Hp'/Hp.
// =====================================================================================================================

#include <FML/Cosmology/Perturbations/PerturbationTypes.h>
#include <FML/Cosmology/Perturbations/PerturbedFieldEquation.h>
#include <FML/Cosmology/Perturbations/QuasiStaticGravity_MuSigma.h>
#include <FML/Units/Units.h>

#include <memory>
#include <stdexcept>
#include <string>

namespace FML::COSMOLOGY::PERTURBATIONS {

    class PerturbedFieldEquation_MuSigma : public PerturbedFieldEquation {
      public:
        /// The quasi-static theory qs (on ctx.background)
        PerturbedFieldEquation_MuSigma(const PerturbationContext & ctx, QuasiStaticGravityPtr qs)
            : PerturbedFieldEquation(ctx), qs(std::move(qs)) {}
        /// Planck/DES
        PerturbedFieldEquation_MuSigma(const PerturbationContext & ctx, double mu0 = 0.0, double Sigma0 = 0.0)
            : PerturbedFieldEquation_MuSigma(
                  ctx, std::make_shared<const QuasiStaticGravity_MuSigma>(ctx.background, mu0, Sigma0)) {}
        std::string name() const override { return "mu-Sigma"; }
        std::string info() const override { return qs->info(); }

        /// The quasi-static theory
        const QuasiStaticGravity & quasi_static() const { return *qs; }

        using MuSigma = QuasiStaticGravity::MuSigma;
        /// mu, Sigma and d(2 Sigma - mu)/dx (at fixed k) at x for the mode k [1/m] (k = 0: the large-scale value)
        MuSigma mu_sigma(double x, double k) const { return qs->mu_sigma(x, k * FML::UTILS::SI::c / ctx.H0); }

        enum { MU, SIGMA, DG, RPP, DHPH, N_CACHE };
        int cache_size() const override { return N_CACHE; }
        void update_cache(const ModeAtX & b, double * cache) const override {
            const MuSigma m = mu_sigma(b.x, b.k);
            cache[MU] = m.mu;
            cache[SIGMA] = m.Sigma;
            cache[DG] = m.dg;
            cache[RPP] = b.H0a2 * (ctx.background.rho_total(b.x) + ctx.background.p_total(b.x));
            cache[DHPH] = b.dHpH;
        }
        MetricAtX metric(const ModeAtX & b,
                         const double * cache,
                         const double * y,
                         const Moments & total,
                         double * dydx) const override {
            const double c = FML::UTILS::SI::c;
            const double P = y[0]; // Phi_GR
            const double mu = cache[MU], g = 2.0 * cache[SIGMA] - mu, dg = cache[DG], rpp = cache[RPP];
            const double c2k2 = c * c * b.k * b.k, A = 1.5 / c2k2;
            const double F = total.flux, ckH = b.ckH;
            MetricAtX m;
            m.Psi = -mu * P - 3.0 * total.shear / c2k2;
            m.Phi = g * P;
            const double num = -P + A * (-ckH * F + 3.0 * (cache[DHPH] - 1.0) * F / ckH - 2.0 * total.shear +
                                         3.0 * rpp * m.Psi - 3.0 * rpp * dg * P);
            const double den = 1.0 + 3.0 * A * rpp * g;
            if (not(den > 0.0))
                throw std::runtime_error(name() + ": singular Phi_GR equation (2 Sigma - mu < 0)");
            // GR: the same equation gives Psi_GR - G (G = 3 flux / (2 (ck/Hp) Hp^2)); replaced by the energy-constraint
            // form of the GR code
            const double Psi_GR = -P - 3.0 * total.shear / c2k2, G = 1.5 * F / (ckH * b.Hp * b.Hp);
            const double dP_GR = Psi_GR - ckH * ckH * P / 3.0 + total.drho / (2.0 * b.Hp * b.Hp);
            dydx[0] = dP_GR + (num / den - (Psi_GR - G));
            m.dPhi = dg * P + g * dydx[0];
            return m;
        }

      protected:
        QuasiStaticGravityPtr qs;
    };

} // namespace FML::COSMOLOGY::PERTURBATIONS

#endif
