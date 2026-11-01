#ifndef FML_PERTURBATIONS_SPECIESPERTURBATION_MASSLESSNEUTRINOS_H
#define FML_PERTURBATIONS_SPECIESPERTURBATION_MASSLESSNEUTRINOS_H

// =====================================================================================================================
// Massless neutrinos (free streaming): the Boltzmann hierarchy Nu_0..Nu_L (L = settings.lmax_nu), with the curvature
// factors s_l (= 1 when flat):
//   Nu_0' = -(ck/Hp) Nu_1 - Phi'
//   Nu_1' = (ck/3Hp) (Nu_0 - 2 s_2 Nu_2 + Psi)
//   Nu_l' = ck/((2l+1) Hp) (l s_l Nu_{l-1} - (l+1) s_{l+1} Nu_{l+1})
//   Nu_L' = (ck/Hp) s_L Nu_{L-1} - c (L+1) cot_K(eta)/Hp Nu_L                    (truncation)
// Moments: delta rho = 4 rho Nu_0, delta p = delta rho / 3, flux = 4 rho Nu_1, shear = 4 rho Nu_2.
// Initial conditions: free_streaming_initial_conditions (SpeciesPerturbation.h). Outputs Nu_0, Nu_1, Nu_2.
// Built from the background species it perturbs (any Species: only rho(x) is used). The model leaves it out when
// Omega_nu = 0 or settings.neutrinos = false (PerturbationModel.h).
// =====================================================================================================================

#include <FML/Background/Species.h>
#include <FML/Perturbations/PerturbationTypes.h>
#include <FML/Perturbations/SpeciesPerturbation.h>

#include <stdexcept>
#include <string>
#include <vector>

namespace FML::COSMOLOGY::PERTURBATIONS {

    class SpeciesPerturbation_MasslessNeutrinos : public SpeciesPerturbation {
      public:
        SpeciesPerturbation_MasslessNeutrinos(const BACKGROUND::Species & nu, const PerturbationContext & ctx)
            : species(nu), ctx(ctx), L(ctx.settings.lmax_nu) {
            if (L < 3)
                throw std::invalid_argument("[SpeciesPerturbation_MasslessNeutrinos] Need lmax_nu >= 3");
        }
        std::string name() const override { return "massless neutrinos"; }
        std::string info() const override { return "massless neutrinos (lmax_nu = " + std::to_string(L) + ")"; }
        int n_vars(int) const override { return L + 1; }
        int cache_size() const override { return 1; }
        void update_cache(const ModeAtX & b, double * cache) const override { cache[0] = b.H0a2 * species.rho(b.x); }
        void initial_conditions(const ModeAtX & b, const InitialData & ic, double * y) const override {
            const auto N = free_streaming_initial_conditions(ctx, b, ic, L);
            for (int ell = 0; ell <= L; ell++)
                y[ell] = N[ell];
        }
        Moments moments(const ModeAtX &, const double * cache, int, const double * N) const override {
            Moments m;
            m.drho = 4.0 * cache[0] * N[0];
            m.dp = m.drho / 3.0;
            m.flux = 4.0 * cache[0] * N[1];
            m.shear = 4.0 * cache[0] * N[2];
            return m;
        }
        void
        rhs(const ModeAtX & b, const double *, int, const MetricAtX & m, const double * N, double * dN) const override {
            const double ckH = b.ckH, k = b.k;
            dN[0] = -ckH * N[1] - m.dPhi;
            dN[1] = ckH / 3.0 * (N[0] - 2.0 * ctx.s_ell(k, 2) * N[2] + m.Psi);
            for (int ell = 2; ell < L; ell++)
                dN[ell] = ckH / (2.0 * ell + 1.0) *
                          (ell * ctx.s_ell(k, ell) * N[ell - 1] - (ell + 1.0) * ctx.s_ell(k, ell + 1) * N[ell + 1]);
            dN[L] = ckH * ctx.s_ell(k, L) * N[L - 1] - (L + 1.0) * b.cotK_H * N[L];
        }
        /// Only Nu_0..Nu_2 are stored (all that enter the Einstein equations and the outputs)
        std::vector<std::string> output_names() const override { return {"Nu_0", "Nu_1", "Nu_2"}; }
        void outputs(const ModeAtX &, const double *, int, const double * N, double * out) const override {
            out[0] = N[0];
            out[1] = N[1];
            out[2] = N[2];
        }
        double free_streaming_density(double x) const override { return species.rho(x); }

      private:
        const BACKGROUND::Species & species;
        PerturbationContext ctx;
        int L;
    };

} // namespace FML::COSMOLOGY::PERTURBATIONS

#endif
