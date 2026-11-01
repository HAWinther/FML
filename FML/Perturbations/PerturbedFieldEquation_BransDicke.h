#ifndef FML_PERTURBATIONS_PERTURBEDFIELDEQUATION_BRANSDICKE_H
#define FML_PERTURBATIONS_PERTURBEDFIELDEQUATION_BRANSDICKE_H

// =====================================================================================================================
// Jordan-Brans-Dicke gravity (flat; the background: Background/FieldEquation_BransDicke.h, whose phi(x), dphi/dx,
// omega and g = G/G_* are read here):
//   phi G_mn = 8 pi G T_mn + (omega/phi)(d_m phi d_n phi - g_mn (d phi)^2/2) + nabla_m nabla_n phi - g_mn box phi
//   box phi  = 8 pi G T / (3 + 2 omega)
// Perturbations (Newtonian gauge; derived in Cosmology/EinsteinBoltzmannSolver/python/derive_brans_dicke.py), with
// F = phi, F_x = dphi/dx and the variables Phi, dF = delta phi and w = dF_x - F_x Psi (which removes Psi' from the
// field equation); Moments times H0^2 a^2 / rho_c0, and since they are in units of rho_c0 = 3 H0^2/(8 pi G_*) the
// matter sources carry g = G/G_* (1 in the hi-class definition of the density parameters, the solver's):
//   Psi    = -Phi - dF/F - 3 g shear / (c^2 k^2 F)                                               (traceless ij)
//   Phi_x  = [3 g drho/Hp^2 + omega F_x w/F - omega dF F_x^2/(2 F^2) + 3 (2 F Psi - dF) + 3 (F_x Psi - w)
//             - (ck/Hp)^2 (2 F Phi + dF)] / (6 F + 3 F_x)                                          (00)
//   dF_x   = w + F_x Psi
//   w_x    = -(2 + Hp'/Hp) w - (ck/Hp)^2 dF + B Psi - 3 F_x Phi_x + 3 g (drho - 3 dp) / ((3 + 2 omega) Hp^2)
// with B = F_xx + (2 + Hp'/Hp) F_x = 3 g (rho - 3p) / ((3 + 2 omega) E^2) (rho, p: all species including Lambda). The
// momentum constraint (0i),
//   -omega dF F_x/F + 2 F Psi + dF - 2 F Phi_x + F_x Psi - dF_x = 3 g flux / ((ck/Hp) Hp^2),
// is not used. Initial conditions: Phi, and dF = w = 0 (the field is frozen in the radiation era, where its source
// rho - 3p vanishes). G_eff_initial = g / phi(x_ini) (the coupling of the Friedmann equation at the start, where the
// field is at rest). Tensors: M*^2 = phi/g, alpha_M = dln phi/dx, alpha_T = 0. Outputs delta_phi_BD = dF/F and
// ddelta_phi_BD_dx = dF_x/F. omega -> infinity is GR.
// =====================================================================================================================

#include <FML/Background/FieldEquation_BransDicke.h>
#include <FML/Perturbations/PerturbationTypes.h>
#include <FML/Perturbations/PerturbedFieldEquation.h>
#include <FML/Units/Units.h>

#include <stdexcept>
#include <string>
#include <vector>

namespace FML::COSMOLOGY::PERTURBATIONS {

    class PerturbedFieldEquation_BransDicke : public PerturbedFieldEquation {
      public:
        PerturbedFieldEquation_BransDicke(const BACKGROUND::BransDickeFieldEquations & bd,
                                          const PerturbationContext & ctx)
            : PerturbedFieldEquation(ctx), bd(bd), omega(bd.get_omega()), g(bd.G_over_Gstar()) {
            if (ctx.background.parameters().Omega_k != 0.0)
                throw std::invalid_argument("[PerturbedFieldEquation_BransDicke] Only implemented for a flat universe");
        }
        std::string name() const override { return "Brans-Dicke"; }
        std::string info() const override {
            return "Brans-Dicke (omega = " + std::to_string(omega) + ", G/G_* = " + std::to_string(g) + ")";
        }

        int n_vars() const override { return 3; }
        enum { F_, FX, B_, N_CACHE };
        int cache_size() const override { return N_CACHE; }
        void update_cache(const ModeAtX & b, double * cache) const override {
            const auto & bg = ctx.background;
            const double E = bg.E(b.x), p = bg.p_total(b.x);
            cache[F_] = bd.phi(b.x);
            cache[FX] = bd.dphidx(b.x);
            cache[B_] = 3.0 * g * (bg.rho_total(b.x) - 3.0 * p) / ((3.0 + 2.0 * omega) * E * E);
        }
        double G_eff_initial() const override { return g / bd.phi_initial(); }
        void initial_conditions(const ModeAtX &, const InitialData & ic, double * y) const override {
            y[0] = ic.Phi;
            y[1] = 0.0;
            y[2] = 0.0;
        }
        MetricAtX metric(const ModeAtX & b,
                         const double * cache,
                         const double * y,
                         const Moments & total,
                         double * dydx) const override {
            const double c = FML::UTILS::SI::c;
            const double F = cache[F_], Fx = cache[FX], B = cache[B_];
            const double Phi = y[0], dF = y[1], w = y[2];
            const double Hp2 = b.Hp * b.Hp, ckH2 = b.ckH * b.ckH;
            MetricAtX m;
            m.Phi = Phi;
            m.Psi = -Phi - dF / F - 3.0 * g * total.shear / (c * c * b.k * b.k * F);
            const double Psi = m.Psi;
            m.dPhi = (3.0 * g * total.drho / Hp2 + omega * Fx * w / F - omega * dF * Fx * Fx / (2.0 * F * F) +
                      3.0 * (2.0 * F * Psi - dF) + 3.0 * (Fx * Psi - w) - ckH2 * (2.0 * F * Phi + dF)) /
                     (6.0 * F + 3.0 * Fx);
            dydx[0] = m.dPhi;
            dydx[1] = w + Fx * Psi;
            dydx[2] = -(2.0 + b.dHpH) * w - ckH2 * dF + B * Psi - 3.0 * Fx * m.dPhi +
                      3.0 * g * (total.drho - 3.0 * total.dp) / ((3.0 + 2.0 * omega) * Hp2);
            return m;
        }
        TensorCoefficients tensor(double x) const override {
            const double F = bd.phi(x);
            return {F / g, bd.dphidx(x) / F, 0.0};
        }
        std::vector<std::string> output_names() const override { return {"delta_phi_BD", "ddelta_phi_BD_dx"}; }
        void outputs(const ModeAtX &,
                     const double * cache,
                     const double * y,
                     const MetricAtX & m,
                     double * out) const override {
            out[0] = y[1] / cache[F_];
            out[1] = (y[2] + cache[FX] * m.Psi) / cache[F_];
        }

      private:
        const BACKGROUND::BransDickeFieldEquations & bd;
        double omega, g;
    };

} // namespace FML::COSMOLOGY::PERTURBATIONS

#endif
