#ifndef FML_PERTURBATIONS_PERTURBEDFIELDEQUATION_CUBICGALILEON_H
#define FML_PERTURBATIONS_PERTURBEDFIELDEQUATION_CUBICGALILEON_H

// =====================================================================================================================
// The cubic Galileon on its tracker (the background: Background/BackgroundModel_CubicGalileon.h, whose
// CubicGalileonSpecies is read here), S = int sqrt(-g) [R/(2 kappa) + c2 X - g3 X box phi] + S_m, c2 = -1, with
// xi = H dphi/dt = sqrt(6 Omega_phi) H0^2 and g3 = 1/(3 xi) (kappa = 1), as the Einstein-Boltzmann solver
// (Cosmology/Models/CubicGalileon.h). The scalar mixes with the metric, so its perturbations are part of the field
// equations (the Galileon species has none of its own).
//
// Perturbations (Newtonian gauge; derived in Cosmology/EinsteinBoltzmannSolver/python/derive_cubic_galileon.py):
// variables Phi, dphi and u = dphi' - phi' Psi (conformal time in 1/H0), with phi_x = xi / E^2:
//   Psi    = -Phi - 3 shear/(c^2 k^2)                       (no gravitational slip from the cubic term)
//   Phi'   from the 00 Einstein equation,  u' from the scalar field equation, in which Phi_P'' and Psi' are eliminated
//          with the derivative of the 0i equation and total matter momentum conservation.
// The 0i equation itself is not used (a test). Initial conditions dphi = u = 0 (phi_x ~ 1/E^2 is negligible early).
// Outputs delta_phi_gal, u_gal. Cache: phi_x, phi_xx, H0^2 a^2 (rho + p) of all species but the Galileon. Flat.
// =====================================================================================================================

#include <FML/Background/BackgroundModel_CubicGalileon.h>
#include <FML/Perturbations/PerturbationTypes.h>
#include <FML/Perturbations/PerturbedFieldEquation.h>

#include <cmath>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace FML::COSMOLOGY::PERTURBATIONS {

    class PerturbedFieldEquation_CubicGalileon : public PerturbedFieldEquation {
      public:
        PerturbedFieldEquation_CubicGalileon(const BACKGROUND::CubicGalileonSpecies & galileon,
                                             const PerturbationContext & ctx)
            : PerturbedFieldEquation(ctx), xi_(std::sqrt(6.0 * galileon.Omega_phi())), g3_(1.0 / (3.0 * xi_)) {
            if (ctx.background.parameters().Omega_k != 0.0)
                throw std::invalid_argument(name() + ": only implemented for a flat universe");
            for (const auto & s : ctx.background.species())
                if (s.get() != &galileon)
                    others.push_back(s.get());
        }
        std::string name() const override { return "cubic Galileon"; }
        std::string info() const override {
            std::ostringstream os;
            os << "cubic Galileon (tracker, c2 = -1, xi = H dphi/dt = " << xi_ << " H0^2, g3 = " << g3_ << ")";
            return os.str();
        }

        /// H dphi/dt on the tracker [H0^2, field in reduced Planck masses], g3, c2 and phi_x = xi / E^2
        double xi() const { return xi_; }
        double g3() const { return g3_; }
        double c2() const { return -1.0; }
        double phi_x(double E2) const { return xi_ / E2; }

        int n_vars() const override { return 3; }
        enum { PHIX, PHIXX, RPP, N_CACHE };
        int cache_size() const override { return N_CACHE; }
        void update_cache(const ModeAtX & b, double * cache) const override {
            const auto & bg = ctx.background;
            const double E = bg.E(b.x), E2 = E * E, dE2 = 2.0 * E2 * bg.dlogEdx(b.x);
            cache[PHIX] = phi_x(E2);
            cache[PHIXX] = -cache[PHIX] * dE2 / E2;
            double rpp = 0.0;
            for (const auto * s : others)
                rpp += s->rho(b.x) + s->p(b.x);
            cache[RPP] = b.H0a2 * rpp;
        }
        void initial_conditions(const ModeAtX &, const InitialData & ic, double * y) const override {
            y[0] = ic.Phi;
            y[1] = 0.0; // the field perturbation is negligible early (phi_x ~ 1/E^2)
            y[2] = 0.0;
        }

        /// Equations from python/derive_cubic_galileon.py, in units H0 = c = kappa = 1 and conformal time: a,
        /// H = Hp/H0, Hd = dH/dtau, F1 = phi', F2 = phi'', k = ck/H0, P = Phi_P = -Phi, D = dphi, U = u, matter:
        /// drho = a^2 delta rho, dp = a^2 delta p, Q = a^2 (rho+p) theta/k^2, Sig = (3/2) a^2 (rho+p) sigma,
        /// R = a^2 (rho+p) (3 times the Moments / H0^2)
        MetricAtX metric(const ModeAtX & b,
                         const double * cache,
                         const double * y,
                         const Moments & total,
                         double * dydx) const override {
            const double a = b.a, H0sq = b.H0a2 / (a * a), H0 = std::sqrt(H0sq);
            const double H = b.Hp / H0, Hd = H * H * b.dHpH;
            const double px = cache[PHIX], pxx = cache[PHIXX];
            const double F1 = H * px, F2 = H * H * (b.dHpH * px + pxx);
            const double k = b.ckH * H;
            const double drho = 3.0 * total.drho / H0sq, dp = 3.0 * total.dp / H0sq, Sig = 3.0 * total.shear / H0sq;
            const double Q = 3.0 * total.flux / (H0sq * k), R = 3.0 * cache[RPP] / H0sq;
            const double c2 = -1.0, g3 = g3_;
            const double P = -y[0], D = y[1], U = y[2];
            const double Psi = P - Sig / (k * k);
            const double a2 = a * a, a4 = a2 * a2, F12 = F1 * F1, F13 = F12 * F1, F14 = F12 * F12, k2 = k * k,
                         g32 = g3 * g3;

            // 00 Einstein equation
            const double P1 = (D * F12 * g3 * k2 - 3.0 * F13 * H * Psi * g3 + 9.0 * F12 * H * U * g3 +
                               F1 * U * a2 * c2 + 6.0 * H * H * Psi * a2 + 2.0 * P * a2 * k2 + a2 * drho) /
                              (3.0 * (F13 * g3 - 2.0 * H * a2));
            // scalar field equation
            const double U1 =
                (-36.0 * D * F14 * H * H * g32 + 9.0 * D * F14 * Hd * g32 + 18.0 * D * F13 * F2 * H * g32 -
                 6.0 * D * F13 * H * a2 * c2 * g3 + 3.0 * D * F12 * F2 * a2 * c2 * g3 -
                 4.0 * D * F1 * H * a2 * g3 * k2 - 4.0 * D * F2 * a2 * g3 * k2 - 2.0 * D * a4 * c2 * k2 +
                 9.0 * F14 * F1 * H * Psi * g32 + 21.0 * F14 * H * U * g32 + 3.0 * F14 * Psi * a2 * c2 * g3 -
                 6.0 * F13 * F2 * U * g32 + 3.0 * F13 * U * a2 * c2 * g3 + 12.0 * F12 * H * H * Psi * a2 * g3 +
                 30.0 * F12 * H * P1 * a2 * g3 - 12.0 * F12 * H * Q * a2 * g3 + 6.0 * F12 * Hd * Psi * a2 * g3 +
                 3.0 * F12 * Psi * R * a2 * g3 - 2.0 * F12 * Psi * a2 * g3 * k2 - 2.0 * F12 * Sig * a2 * g3 +
                 3.0 * F12 * a2 * dp * g3 + 24.0 * F1 * F2 * H * Psi * a2 * g3 + 12.0 * F1 * F2 * P1 * a2 * g3 +
                 4.0 * F1 * H * Psi * a4 * c2 - 12.0 * F1 * Hd * U * a2 * g3 + 6.0 * F1 * P1 * a4 * c2 -
                 12.0 * F2 * H * U * a2 * g3 + 2.0 * F2 * Psi * a4 * c2 - 4.0 * H * U * a4 * c2) /
                (3.0 * F14 * g32 + 12.0 * F1 * H * a2 * g3 + 2.0 * a4 * c2);

            MetricAtX m;
            m.Phi = y[0];
            m.Psi = Psi;
            dydx[0] = -P1 / H; // d/dx = (1/H) d/dtau
            dydx[1] = (U + F1 * Psi) / H;
            dydx[2] = U1 / H;
            m.dPhi = dydx[0];
            return m;
        }
        std::vector<std::string> output_names() const override { return {"delta_phi_gal", "u_gal"}; }
        void outputs(const ModeAtX &, const double *, const double * y, const MetricAtX &, double * out) const override {
            out[0] = y[1];
            out[1] = y[2];
        }

      private:
        double xi_, g3_;
        std::vector<const BACKGROUND::Species *> others;
    };

} // namespace FML::COSMOLOGY::PERTURBATIONS

#endif
