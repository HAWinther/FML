#ifndef FML_PERTURBATIONS_SPECIESPERTURBATION_MASSIVENEUTRINOS_H
#define FML_PERTURBATIONS_SPECIESPERTURBATION_MASSIVENEUTRINOS_H

// =====================================================================================================================
// Massive neutrinos: one Fermi-Dirac species (neutrino + antineutrino), f0(q) = 1/(e^q + 1), with q the comoving
// momentum in units of the neutrino temperature, M(x) = m c^2 a / (k_B T_nu0) and eps = sqrt(q^2 + M^2).
//
// Mode 0: the perturbation Psi_l(q) of the distribution, f = f0(q) [1 + Psi(q)], for n_q_ncdm Gauss-Laguerre momenta
// and l = 0..L (L = lmax_ncdm), Ma & Bertschinger 1995 in the Newtonian gauge (phi_MB = -Phi, psi_MB = Psi):
//   Psi_0' = -(ck/Hp)(q/eps) Psi_1 + Phi' dln f0/dln q
//   Psi_1' = (ck/3Hp)(q/eps)(Psi_0 - 2 s_2 Psi_2) - (ck/3Hp)(eps/q) Psi dln f0/dln q
//   Psi_l' = (ck/Hp)(q/eps)/(2l+1) (l s_l Psi_{l-1} - (l+1) s_{l+1} Psi_{l+1})
//   Psi_L' = (ck/Hp)(q/eps) s_L Psi_{L-1} - c (L+1) cot_K(eta)/Hp Psi_L                       (truncation)
// Moments (eqs. 55-56 of Ma & Bertschinger), with C = rho / I_rho (rho = C I_rho(M), I_rho = int q^2 eps f0 dq):
//   delta rho      = C sum_i w_i q^2 eps Psi_0,   delta p = C sum_i w_i q^4/(3 eps) Psi_0
//   (rho+p) theta  = C ck sum_i w_i q^3 Psi_1          (flux = (rho+p) theta / (ck))
//   (rho+p) sigma  = (2/3) C sum_i w_i q^4/eps Psi_2   (shear = (3/2)(rho+p) sigma)
// (massless: Psi_l = -dln f0/dln q Nu_l, and these reduce to 4 rho Nu_l). Initial conditions (relativistic):
// Psi_l(q) = -dln f0/dln q N_l (times eps/q for l = 1), N_l from free_streaming_initial_conditions.
//
// Mode 1, from k eta = ncdm_fluid_trigger on (the switch point; deep inside the horizon): the fluid equations of
// CLASS's ncdmfa_CLASS (Lesgourgues & Tram 2011) for F = (delta, theta/(ck), sigma):
//   delta' = -(1 + w)(ck/Hp theta/(ck) + 3 Phi') - 3 (c_a^2 - w) delta
//   [theta/(ck)]' = -(1 - 3 c_a^2) theta/(ck) + (ck/Hp)[c_a^2 delta/(1 + w) - sigma + Psi]
//   sigma' = -3 [2/3 - c_a^2 - pseudo_p/(3p) + c/(Hp eta)] sigma
//            + (8/3) c_vis^2/(1 + w) s_2 (ck/Hp theta/(ck) + 6 Phi'),   c_vis^2 = 3 w c_a^2
// with c_a^2 = w (5 - pseudo_p/p) / (3 (1 + w)) the adiabatic sound speed and pseudo_p the pseudo-pressure
// (dp/dx = -(5p - pseudo_p)); delta p = c_a^2 delta rho. (As in the solver, the c/(Hp eta) term is the flat-space
// one, also with curvature.)
// Outputs: delta_ncdm = delta rho / rho, v_ncdm with (rho + p) v = -flux.
//
// The background (rho, p, the pseudo-pressure, M(x), C) comes from the MassiveNeutrinoSpecies (times the factor of a
// ScaledSpecies wrapping it; w, c_a^2 and pseudo_p/p do not depend on it). MassiveNeutrinoQuadrature: the
// Gauss-Laguerre momentum nodes and weights of the hierarchy (weight e^{-q}, f0 folded into the weights), as the
// solver's Cosmology/Background/MassiveNeutrinos.
// =====================================================================================================================

#include <FML/Background/Species.h>
#include <FML/Math/NumericalMethods.h>
#include <FML/Perturbations/PerturbationTypes.h>
#include <FML/Perturbations/SpeciesPerturbation.h>
#include <FML/Units/Units.h>

#include <gsl/gsl_integration.h>

#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

namespace FML::COSMOLOGY::PERTURBATIONS {

    /// Nodes q_i and weights w_i (Gauss-Laguerre, f0 included) with int_0^inf F(q) f0(q) dq ~ sum_i w_i F(q_i)
    struct MassiveNeutrinoQuadrature {
        std::vector<double> q, w;

        explicit MassiveNeutrinoQuadrature(int n) {
            if (n < 1)
                throw std::invalid_argument("[MassiveNeutrinoQuadrature] Need n >= 1");
            gsl_integration_fixed_workspace * ws =
                gsl_integration_fixed_alloc(gsl_integration_fixed_laguerre, n, 0.0, 1.0, 0.0, 0.0);
            const double * x = gsl_integration_fixed_nodes(ws);
            const double * wl = gsl_integration_fixed_weights(ws);
            q.assign(x, x + n);
            w.resize(n);
            // Laguerre weight e^{-q}: int F f0 dq = int e^{-q} [F e^q f0] dq, and e^q f0 = 1/(1 + e^{-q})
            for (int i = 0; i < n; i++)
                w[i] = wl[i] / (1.0 + std::exp(-x[i]));
            gsl_integration_fixed_free(ws);
        }
        int size() const { return int(q.size()); }

        /// d ln f0 / d ln q = -q e^q / (e^q + 1)
        static double dlnf0_dlnq(double q) { return -q / (1.0 + std::exp(-q)); }
    };

    class SpeciesPerturbation_MassiveNeutrinos : public SpeciesPerturbation {
      public:
        /// nu: the background species; factor: its densities are multiplied by this (a ScaledSpecies wrapping it)
        SpeciesPerturbation_MassiveNeutrinos(const BACKGROUND::MassiveNeutrinoSpecies & nu,
                                             const PerturbationContext & ctx,
                                             double factor = 1.0)
            : species(nu), ctx(ctx), f(factor), L(ctx.settings.lmax_ncdm), nq(ctx.settings.n_q_ncdm) {
            if (L < 3 or nq < 1)
                throw std::invalid_argument(
                    "[SpeciesPerturbation_MassiveNeutrinos] Need lmax_ncdm >= 3 and n_q_ncdm >= 1");
            if (not(f > 0.0))
                throw std::invalid_argument("[SpeciesPerturbation_MassiveNeutrinos] Need factor > 0");
            const MassiveNeutrinoQuadrature quad(nq);
            q = quad.q;
            wq = quad.w;
            for (double qi : q)
                dlnf0.push_back(MassiveNeutrinoQuadrature::dlnf0_dlnq(qi));
        }

        std::string name() const override { return "massive neutrinos"; }
        std::string info() const override {
            return "massive neutrinos (" + std::to_string(nq) + " momenta, lmax_ncdm = " + std::to_string(L) + ")";
        }

        /// k eta = ncdm_fluid_trigger (x_start if already there, none if not reached by x_end)
        std::vector<double> switch_points(double k) const override {
            const auto & s = ctx.settings;
            const double trig = s.ncdm_fluid_trigger;
            if (k * ctx.eta(s.x_end) <= trig)
                return {};
            if (k * ctx.eta(s.x_start) >= trig)
                return {s.x_start};
            return {FML::MATH::find_root([&](double x) { return k * ctx.eta(x) - trig; }, s.x_start, s.x_end)};
        }
        int n_vars(int mode) const override { return mode == 0 ? nq * (L + 1) : 3; }
        /// ~200 variables: the dense linear algebra of an implicit solver makes it ~20x slower
        bool prefers_explicit(int mode) const override { return mode == 0; }

        enum { M_, C_, RHO, P_, W, CA2, PSEUDO, N_CACHE };
        int cache_size() const override { return N_CACHE; }
        void update_cache(const ModeAtX & b, double * cc) const override {
            const double x = b.x;
            const double rho = species.rho(x), p = species.p(x);
            cc[M_] = species.M_of_x(x);
            cc[C_] = b.H0a2 * f * species.prefactor() * std::exp(-4.0 * x);
            cc[RHO] = b.H0a2 * f * rho;
            cc[P_] = b.H0a2 * f * p;
            cc[W] = p / rho;
            cc[PSEUDO] = species.pseudo_p(x) / p;
            cc[CA2] = cc[W] / (3.0 * (1.0 + cc[W])) * (5.0 - cc[PSEUDO]);
        }

        void initial_conditions(const ModeAtX & b, const InitialData & ic, double * y) const override {
            const auto N = free_streaming_initial_conditions(ctx, b, ic, L);
            const double M = species.M_of_x(b.x);
            for (int i = 0; i < nq; i++) {
                const double eps_q = std::sqrt(q[i] * q[i] + M * M) / q[i];
                for (int ell = 0; ell <= L; ell++)
                    y[i * (L + 1) + ell] = -dlnf0[i] * N[ell] * (ell == 1 ? eps_q : 1.0);
            }
        }

        void switch_mode(const ModeAtX & b, const double * cc, int, const double * P, double * F) const override {
            const Moments m = moments(b, cc, 0, P);
            const double rpp = cc[RHO] + cc[P_];
            F[0] = m.drho / cc[RHO];
            F[1] = m.flux / rpp;
            F[2] = m.shear / (1.5 * rpp);
        }

        Moments moments(const ModeAtX &, const double * cc, int mode, const double * P) const override {
            Moments m;
            if (mode == 1) {
                const double rpp = cc[RHO] + cc[P_];
                m.drho = cc[RHO] * P[0];
                m.dp = cc[CA2] * m.drho; // adiabatic, as CLASS's fluid approximation
                m.flux = rpp * P[1];
                m.shear = 1.5 * rpp * P[2];
                return m;
            }
            const double M = cc[M_];
            for (int i = 0; i < nq; i++) {
                const double qi = q[i], eps = std::sqrt(qi * qi + M * M), w = wq[i];
                const double * Y = P + i * (L + 1);
                m.drho += w * qi * qi * eps * Y[0];
                m.dp += w * qi * qi * qi * qi / (3.0 * eps) * Y[0];
                m.flux += w * qi * qi * qi * Y[1];
                m.shear += w * qi * qi * qi * qi / eps * Y[2];
            }
            m.drho *= cc[C_];
            m.dp *= cc[C_];
            m.flux *= cc[C_];
            m.shear *= cc[C_];
            return m;
        }

        void rhs(const ModeAtX & b, const double * cc, int mode, const MetricAtX & met, const double * P, double * dP)
            const override {
            const double k = b.k, s2 = ctx.s_ell(k, 2), dPhi = met.dPhi, Psi = met.Psi;
            if (mode == 1) {
                const double w = cc[W], ca2 = cc[CA2], cvis2 = 3.0 * w * ca2;
                dP[0] = -(1.0 + w) * (b.ckH * P[1] + 3.0 * dPhi) - 3.0 * (ca2 - w) * P[0];
                dP[1] = -(1.0 - 3.0 * ca2) * P[1] + b.ckH * (ca2 * P[0] / (1.0 + w) - P[2] + Psi);
                dP[2] = -3.0 * (2.0 / 3.0 - ca2 - cc[PSEUDO] / 3.0 + FML::UTILS::SI::c / (b.Hp * b.eta)) * P[2] +
                        8.0 / 3.0 * cvis2 / (1.0 + w) * s2 * (b.ckH * P[1] + 6.0 * dPhi);
                return;
            }
            const double M = cc[M_];
            for (int i = 0; i < nq; i++) {
                const double qi = q[i], eps = std::sqrt(qi * qi + M * M);
                const double kq = b.ckH * qi / eps, dlnf = dlnf0[i];
                const double * Y = P + i * (L + 1);
                double * dY = dP + i * (L + 1);
                dY[0] = -kq * Y[1] + dPhi * dlnf;
                dY[1] = kq / 3.0 * (Y[0] - 2.0 * s2 * Y[2]) - b.ckH * eps / (3.0 * qi) * Psi * dlnf;
                for (int ell = 2; ell < L; ell++)
                    dY[ell] = kq / (2.0 * ell + 1.0) *
                              (ell * ctx.s_ell(k, ell) * Y[ell - 1] - (ell + 1.0) * ctx.s_ell(k, ell + 1) * Y[ell + 1]);
                dY[L] = kq * ctx.s_ell(k, L) * Y[L - 1] - (L + 1.0) * b.cotK_H * Y[L];
            }
        }

        std::vector<std::string> output_names() const override { return {"delta_ncdm", "v_ncdm"}; }
        void outputs(const ModeAtX & b, const double * cc, int mode, const double * P, double * out) const override {
            const Moments m = moments(b, cc, mode, P);
            out[0] = m.drho / cc[RHO];
            out[1] = -m.flux / (cc[RHO] + cc[P_]);
        }
        double free_streaming_density(double x) const override { return f * species.rho(x); }

      private:
        const BACKGROUND::MassiveNeutrinoSpecies & species;
        PerturbationContext ctx;
        double f;
        int L, nq;
        std::vector<double> q, wq, dlnf0; ///< momentum nodes, weights (incl. f0), dln f0/dln q
    };

} // namespace FML::COSMOLOGY::PERTURBATIONS

#endif
