#ifndef FML_COSMOLOGY_PERTURBATIONS_SPECIESPERTURBATION_COUPLEDQUINTESSENCE_H
#define FML_COSMOLOGY_PERTURBATIONS_SPECIESPERTURBATION_COUPLEDQUINTESSENCE_H

// =====================================================================================================================
// Coupled quintessence (Wetterich 1995, Amendola 2000; the background: Background/BackgroundModel_CoupledQuintessence.h,
// whose CoupledDarkSector is read here): the cold dark matter (m ~ exp(-beta phi)) and the scalar field (V = 3 H0^2
// M_pl^2 V0 exp(-lambda phi)) in one block.
// Newtonian gauge, x = ln a, variables (delta_c, v_c, dphi, w = dphi_x - phi_x Psi):
//   delta_c' = (ck/Hp) v_c - 3 Phi' - beta dphi_x                 (number conservation, rho_c = m(phi) n)
//   v_c'     = -(1 - beta phi_x) v_c - (ck/Hp) Psi + beta (ck/Hp) dphi    (fifth force + drag)
//   dphi_x   = w + phi_x Psi
//   w'       = -(2 + Hp'/Hp) w - (ck/Hp)^2 dphi - 3 v_phiphi dphi/E^2 - 3 phi_x Phi'
//              - 3 (v_phi - beta rho_c) Psi/E^2 + 3 beta rho_c delta_c/E^2
// The field's density, pressure and momentum are those of an uncoupled field (SpeciesPerturbation_ScalarField.h); the
// exchange cancels in the total. Initial conditions: CDM adiabatic (or IsoValues delta_c, v_c), dphi = w = 0. Outputs
// delta_cdm, v_cdm, delta_phi, delta_de.
// =====================================================================================================================

#include <FML/Cosmology/Background/BackgroundModel_CoupledQuintessence.h>
#include <FML/Cosmology/Perturbations/PerturbationTypes.h>
#include <FML/Cosmology/Perturbations/SpeciesPerturbation.h>

#include <string>
#include <vector>

namespace FML::COSMOLOGY::PERTURBATIONS {

    class SpeciesPerturbation_CoupledQuintessence : public SpeciesPerturbation {
      public:
        SpeciesPerturbation_CoupledQuintessence(const BACKGROUND::CoupledDarkSector & sector,
                                                const PerturbationContext & /*ctx*/)
            : d(sector) {}
        std::string name() const override { return "coupled CDM + quintessence"; }
        int n_vars(int) const override { return 4; }

        enum { RC, DPHI, VPHI, VPP, G3, BRC, RPHI, N_CACHE };
        int cache_size() const override { return N_CACHE; }
        void update_cache(const ModeAtX & b, double * cc) const override {
            const double f = d.phi(b.x), rc = d.rho_c(b.x);
            cc[RC] = b.H0a2 * rc;
            cc[DPHI] = d.dphidx(b.x);
            cc[VPHI] = d.v_phi(f);
            cc[VPP] = d.v_phiphi(f);
            cc[G3] = 3.0 * b.H0a2 / (b.Hp * b.Hp); // 3/E^2
            cc[BRC] = d.beta * rc;
            cc[RPHI] = b.H0a2 * d.rho_phi(b.x);
        }
        void initial_conditions(const ModeAtX & b, const InitialData & ic, double * y) const override {
            // CDM: the isocurvature values if set (the field has no isocurvature mode of its own and starts at rest)
            y[0] = ic.iso ? ic.iso->delta_c : -1.5 * ic.Psi;
            y[1] = ic.iso ? ic.iso->v_c : -0.5 * b.ckH * ic.Psi;
            y[2] = y[3] = 0.0;
        }
        Moments moments(const ModeAtX & b, const double * cc, int, const double * y) const override {
            Moments m;
            const double kin = b.Hp * b.Hp * cc[DPHI] * y[3] / 3.0, pot = b.H0a2 * cc[VPHI] * y[2];
            m.drho = cc[RC] * y[0] + kin + pot;
            m.dp = kin - pot;
            m.flux = -cc[RC] * y[1] + b.ckH * b.Hp * b.Hp * cc[DPHI] * y[2] / 3.0;
            return m;
        }
        void rhs(const ModeAtX & b, const double * cc, int, const MetricAtX & m, const double * y, double * dydx)
            const override {
            const double beta = d.beta, ckH = b.ckH, Fx = cc[DPHI];
            const double dc = y[0], vc = y[1], dphi = y[2], w = y[3];
            const double dphi_x = w + Fx * m.Psi;
            dydx[0] = ckH * vc - 3.0 * m.dPhi - beta * dphi_x;
            dydx[1] = -(1.0 - beta * Fx) * vc - ckH * m.Psi + beta * ckH * dphi;
            dydx[2] = dphi_x;
            dydx[3] = -(2.0 + b.dHpH) * w - (ckH * ckH + cc[G3] * cc[VPP]) * dphi - 3.0 * Fx * m.dPhi -
                      cc[G3] * (cc[VPHI] - cc[BRC]) * m.Psi + cc[G3] * cc[BRC] * dc;
        }
        std::vector<std::string> output_names() const override {
            return {"delta_cdm", "v_cdm", "delta_phi", "delta_de"};
        }
        void outputs(const ModeAtX & b, const double * cc, int, const double * y, double * out) const override {
            out[0] = y[0];
            out[1] = y[1];
            out[2] = y[2];
            out[3] = (b.Hp * b.Hp * cc[DPHI] * y[3] / 3.0 + b.H0a2 * cc[VPHI] * y[2]) / cc[RPHI];
        }

      private:
        const BACKGROUND::CoupledDarkSector & d;
    };

} // namespace FML::COSMOLOGY::PERTURBATIONS

#endif
