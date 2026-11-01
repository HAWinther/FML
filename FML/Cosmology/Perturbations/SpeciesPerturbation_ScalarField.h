#ifndef FML_COSMOLOGY_PERTURBATIONS_SPECIESPERTURBATION_SCALARFIELD_H
#define FML_COSMOLOGY_PERTURBATIONS_SPECIESPERTURBATION_SCALARFIELD_H

// =====================================================================================================================
// A minimally coupled scalar field (the background: BACKGROUND::ScalarFieldSpecies, Species.h: quintessence and early
// dark energy), V = 3 H0^2 M_pl^2 v(phi), phi in M_pl, x = ln a, phi_x = dphi/dx.
//
// Mode 0, the Klein-Gordon equation (Newtonian gauge, Mukhanov): with dphi the field perturbation and w = dphi_x - phi_x Psi (which removes Psi'),
//   dphi_x = w + phi_x Psi
//   w_x    = -(2 + Hp'/Hp) w - (ck/Hp)^2 dphi - 3 v_phiphi dphi / E^2 - 3 phi_x Phi' - 3 v_phi Psi / E^2
// Moments (times H0^2 a^2 / rho_c0):
//   delta rho = Hp^2 phi_x w / 3 + H0^2 a^2 v_phi dphi,   delta p = Hp^2 phi_x w / 3 - H0^2 a^2 v_phi dphi
//   flux = (ck/Hp) Hp^2 phi_x dphi / 3,   no anisotropic stress
// Initial conditions dphi = w = 0 (the field is frozen; the adiabatic mode is approached quickly).
//
// Mode 1, the oscillating field as a fluid. A potential with a minimum lets the background replace the field by a
// fluid with the cycle-averaged w = w_osc (= (n-1)/(n+1) for V ~ |phi - phi_min|^(2n)) once it oscillates fast (m / H >
// ~125; ScalarFieldSpecies::x_fluid()). From there the background has no field to perturb, so the perturbations switch
// to the same effective fluid at the same x (a switch point of this species): variables delta = delta rho / rho and
// u = theta / (ck), constant w = c_a^2 and the effective rest-frame sound speed of an oscillating field (as Poulin,
// Smith, Grin, Karwal & Kamionkowski 2018 for V ~ phi^(2n); n = 1 is the ultralight-axion case),
//   c_s^2(k, x) = [2 (n - 1) W + (ck/Hp)^2] / [2 (n + 1) W + (ck/Hp)^2],   W = (varpi / H)^2,
// with varpi the angular frequency of the oscillation (cosmic time): at the switch from the period of the actual
// potential at the switch density, T = 2 sum_(both sides) int_0^A dD / sqrt(6 (rho - v(phi_min +- D))) / H0, and
// after it varpi ~ rho^((n - 1)/(2n)) (amplitude scaling of a power-law potential; constant for n = 1). The fluid
// equations are those of a dark-energy fluid (SpeciesPerturbation_DarkEnergyFluid.h, without exchange):
//   delta' = -(1 + w)((ck/Hp) u + 3 Phi') - 3 (c_s^2 - w)[delta + 3 (1 + w) u / (ck/Hp)]
//   u'     = -(1 - 3 c_s^2) u + (ck/Hp)[c_s^2 delta / (1 + w) + Psi]
//   delta p = rho [c_s^2 delta + 3 (1 + w)(c_s^2 - w) u / (ck/Hp)],  flux = (1 + w) rho u
// The switch converts (dphi, w) into the cycle-averaged fluid. At the switch the field is near a crossing of the
// minimum, and (dphi, w) is split into a shift along the trajectory (epsilon: phi_x, phi_xx) and a change of the
// oscillation amplitude at fixed phase (eta: for V ~ D^(2n), D = phi - phi_min, the direction (D, n phi_x)),
//   dphi = epsilon phi_x + eta D,   w = epsilon phi_xx + eta n phi_x.
// Only the shift carries momentum on average: (1 + w_osc) rho (ck/Hp) epsilon (its instantaneous value oscillates
// with the kinetic energy), so u = (ck/Hp) epsilon. The density is set by the comoving density perturbation
// rho Delta = delta rho + 3 flux / (ck/Hp), which is continuous: it is what the constraint k^2 Phi ~ rho Delta of the
// (continuous) metric holds with, and it does not oscillate (a shift has rho Delta = 0 at every phase; the field's
// amplitude perturbation is 2 n rho eta for V ~ D^(2n)), so
//   delta = rho Delta_field / rho - 3 (1 + w_osc) epsilon
// (the adiabatic mode, a pure shift, goes over exactly; the instantaneous delta rho of a shift, -6 rho epsilon at a
// crossing, would be wrong by a factor 2 / (1 + w_osc)). This keeps the Einstein constraints exact through the switch;
// the neglected terms are O(H / m) < 1e-2 of the field's perturbation, whose share of the total is the field's
// (small) fraction of the density then. A field that is a fluid already at
// x_start starts in the fluid (one mode) with the adiabatic delta = -3/2 (1 + w) Psi, u = (ck/2Hp) Psi.
// The Einstein-Boltzmann solver used to integrate the field (and its perturbations) through the oscillations to the end;
// for early dark energy with n >= 2.5 and the usual z_c the field never oscillates fast enough to switch (m / H stays
// below ~125 as varpi ~ rho^((n-1)/2n) falls), so the two agree; see the README for the cases that switch.
//
// Outputs delta_phi (dphi; 0 in the fluid), delta_de (delta rho / rho) and q_de (flux / rho = (1 + w) theta / (ck), the
// momentum density). Cache: phi_x, v_phi, v_phiphi, 3/E^2, H0^2 a^2 rho, phi_xx, phi - phi_min and c_s^2 (fluid).
// =====================================================================================================================

#include <FML/Cosmology/Background/Species.h>
#include <FML/Cosmology/Perturbations/PerturbationTypes.h>
#include <FML/Cosmology/Perturbations/SpeciesPerturbation.h>

#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace FML::COSMOLOGY::PERTURBATIONS {

    class SpeciesPerturbation_ScalarField : public SpeciesPerturbation {
      public:
        /// name: the species' name in the solver ("quintessence", "early dark energy")
        SpeciesPerturbation_ScalarField(const BACKGROUND::ScalarFieldSpecies & field,
                                        std::string name,
                                        const PerturbationContext & ctx)
            : q(field), label(std::move(name)), x_fluid(field.x_fluid()) {
            switches = std::isfinite(x_fluid) and x_fluid < ctx.settings.x_end;
            fluid_start = switches and x_fluid <= ctx.settings.x_start;
            if (switches) {
                w_osc = q.w_oscillating();
                n_pow = (1.0 + w_osc) / (1.0 - w_osc);
                if (not(n_pow >= 1.0) or not std::isfinite(n_pow))
                    throw std::invalid_argument("[" + label + "] The oscillating fluid needs 0 <= w_osc < 1");
                rho_switch = q.rho(x_fluid);
                varpi_switch = angular_frequency(rho_switch);
            }
        }
        std::string name() const override { return label; }

        std::vector<double> switch_points(double) const override {
            if (switches and not fluid_start)
                return {x_fluid};
            return {};
        }
        /// The fluid mode (1, or 0 if the field is a fluid from the start)
        bool fluid(int mode) const { return fluid_start or mode == 1; }
        int n_vars(int) const override { return 2; }

        enum { DPHI, VPHI, VPP, E2INV3, RHO, DDPHI, DMIN, CS2, N_CACHE };
        int cache_size() const override { return N_CACHE; }
        void update_cache(const ModeAtX & b, double * cc) const override {
            const double phi = q.phi(b.x);
            cc[DPHI] = q.dphidx(b.x);
            cc[VPHI] = q.v_phi(phi);
            cc[VPP] = q.v_phiphi(phi);
            cc[E2INV3] = 3.0 * b.H0a2 / (b.Hp * b.Hp);
            cc[RHO] = b.H0a2 * q.rho(b.x);
            cc[DDPHI] = switches ? q.ddphidx(b.x) : 0.0;
            cc[DMIN] = switches ? phi - q.phi_min() : 0.0;
            cc[CS2] = 0.0;
            if (switches and b.x > x_fluid) {
                // (varpi / H)^2 with varpi ~ rho^((n-1)/(2n)), H = H0 E = Hp / a
                const double E = b.Hp / std::sqrt(b.H0a2);
                const double varpi = varpi_switch * std::pow(cc[RHO] / (b.H0a2 * rho_switch), (n_pow - 1.0) / (2.0 * n_pow));
                const double W = varpi * varpi / (E * E), k2 = b.ckH * b.ckH;
                cc[CS2] = (2.0 * (n_pow - 1.0) * W + k2) / (2.0 * (n_pow + 1.0) * W + k2);
            }
        }
        void initial_conditions(const ModeAtX & b, const InitialData & ic, double * y) const override {
            if (fluid_start) {
                y[0] = -1.5 * (1.0 + w_osc) * ic.Psi;
                y[1] = 0.5 * b.ckH * ic.Psi;
                return;
            }
            y[0] = y[1] = 0.0;
        }
        void switch_mode(const ModeAtX & b, const double * cc, int, const double * yo, double * yn) const override {
            const double fx = cc[DPHI], fxx = cc[DDPHI], D = cc[DMIN];
            const double det = n_pow * fx * fx - D * fxx;
            if (not(std::abs(det) > 0.0) or not std::isfinite(det))
                throw std::runtime_error("[" + label + "] The switch to the oscillating fluid (x = " +
                                         std::to_string(x_fluid) + ") is too close to the start of the perturbations");
            const double eps = (yo[0] * n_pow * fx - D * yo[1]) / det;
            const Moments m = moments(b, cc, 0, yo);
            yn[0] = (m.drho + 3.0 * m.flux / b.ckH) / cc[RHO] - 3.0 * (1.0 + w_osc) * eps;
            yn[1] = b.ckH * eps;
        }

        Moments moments(const ModeAtX & b, const double * cc, int mode, const double * y) const override {
            Moments m;
            if (fluid(mode)) {
                const double w = w_osc, cs2 = cc[CS2];
                m.drho = cc[RHO] * y[0];
                m.dp = cc[RHO] * (cs2 * y[0] + 3.0 * (1.0 + w) * (cs2 - w) * y[1] / b.ckH);
                m.flux = (1.0 + w) * cc[RHO] * y[1];
                return m;
            }
            const double kin = b.Hp * b.Hp * cc[DPHI] * y[1] / 3.0, pot = b.H0a2 * cc[VPHI] * y[0];
            m.drho = kin + pot;
            m.dp = kin - pot;
            m.flux = b.ckH * b.Hp * b.Hp * cc[DPHI] * y[0] / 3.0;
            return m;
        }
        void rhs(const ModeAtX & b, const double * cc, int mode, const MetricAtX & m, const double * y, double * dydx)
            const override {
            if (fluid(mode)) {
                const double w = w_osc, cs2 = cc[CS2], ckH = b.ckH, d = y[0], u = y[1];
                dydx[0] = -(1.0 + w) * (ckH * u + 3.0 * m.dPhi) - 3.0 * (cs2 - w) * (d + 3.0 * (1.0 + w) * u / ckH);
                dydx[1] = -(1.0 - 3.0 * cs2) * u + ckH * (cs2 * d / (1.0 + w) + m.Psi);
                return;
            }
            const double dphi = y[0], w = y[1];
            dydx[0] = w + cc[DPHI] * m.Psi;
            dydx[1] = -(2.0 + b.dHpH) * w - (b.ckH * b.ckH + cc[E2INV3] * cc[VPP]) * dphi - 3.0 * cc[DPHI] * m.dPhi -
                      cc[E2INV3] * cc[VPHI] * m.Psi;
        }
        std::vector<std::string> output_names() const override { return {"delta_phi", "delta_de", "q_de"}; }
        void outputs(const ModeAtX & b, const double * cc, int mode, const double * y, double * out) const override {
            const Moments m = moments(b, cc, mode, y);
            out[0] = fluid(mode) ? 0.0 : y[0];
            out[1] = m.drho / cc[RHO];
            out[2] = m.flux / cc[RHO];
        }

        /// Where the field becomes a fluid (+infinity if not on the range of the perturbations) and its angular
        /// frequency there [H0]
        double x_switch() const { return switches ? x_fluid : std::numeric_limits<double>::infinity(); }
        double angular_frequency_at_switch() const { return varpi_switch; }

        /// The angular frequency [H0] of the oscillation about the minimum with (kinetic + potential) density rho
        double angular_frequency(double rho) const {
            const double fmin = q.phi_min();
            double T = 0.0;
            for (double side : {1.0, -1.0}) {
                // the turning point A: v(phi_min + side A) = rho
                double lo = 0.0, hi = 1e-6;
                for (int i = 0; i < 200 and q.v(fmin + side * hi) < rho; i++)
                    hi *= 2.0;
                if (not(q.v(fmin + side * hi) >= rho))
                    throw std::runtime_error("[" + label + "] No turning point of the oscillation");
                for (int i = 0; i < 200; i++) {
                    const double mid = 0.5 * (lo + hi);
                    (q.v(fmin + side * mid) < rho ? lo : hi) = mid;
                }
                const double A = 0.5 * (lo + hi);
                // int_0^A dD / sqrt(6 (rho - v)), D = A (1 - s^2): int_0^1 2 A s ds / sqrt(6 (rho - v)), Gauss-Legendre
                double I = 0.0;
                for (int i = 0; i < 32; i++) {
                    const double s = 0.5 * (1.0 + gl_x[i]), D = A * (1.0 - s * s);
                    const double r = rho - q.v(fmin + side * D);
                    if (r > 0.0)
                        I += 0.5 * gl_w[i] * 2.0 * A * s / std::sqrt(6.0 * r);
                }
                T += 2.0 * I;
            }
            return 2.0 * M_PI / T;
        }

      private:
        const BACKGROUND::ScalarFieldSpecies & q;
        std::string label;
        double x_fluid;
        bool switches = false, fluid_start = false;
        double w_osc = 0.0, n_pow = 1.0, rho_switch = 0.0, varpi_switch = 0.0;

        // 32-point Gauss-Legendre on [-1, 1]
        static constexpr double gl_x[32] = {
            -0.9972638618494816, -0.9856115115452684, -0.9647622555875064, -0.9349060759377397, -0.8963211557660521,
            -0.8493676137325700, -0.7944837959679424, -0.7321821187402897, -0.6630442669302152, -0.5877157572407623,
            -0.5068999089322294, -0.4213512761306353, -0.3318686022821276, -0.2392873622521371, -0.1444719615827965,
            -0.0483076656877383, 0.0483076656877383,  0.1444719615827965,  0.2392873622521371,  0.3318686022821276,
            0.4213512761306353,  0.5068999089322294,  0.5877157572407623,  0.6630442669302152,  0.7321821187402897,
            0.7944837959679424,  0.8493676137325700,  0.8963211557660521,  0.9349060759377397,  0.9647622555875064,
            0.9856115115452684,  0.9972638618494816};
        static constexpr double gl_w[32] = {
            0.0070186100094701, 0.0162743947309057, 0.0253920653092621, 0.0342738629130214, 0.0428358980222267,
            0.0509980592623762, 0.0586840934785355, 0.0658222227763618, 0.0723457941088485, 0.0781938957870703,
            0.0833119242269467, 0.0876520930044038, 0.0911738786957639, 0.0938443990808046, 0.0956387200792749,
            0.0965400885147278, 0.0965400885147278, 0.0956387200792749, 0.0938443990808046, 0.0911738786957639,
            0.0876520930044038, 0.0833119242269467, 0.0781938957870703, 0.0723457941088485, 0.0658222227763618,
            0.0586840934785355, 0.0509980592623762, 0.0428358980222267, 0.0342738629130214, 0.0253920653092621,
            0.0162743947309057, 0.0070186100094701};
    };

} // namespace FML::COSMOLOGY::PERTURBATIONS

#endif
