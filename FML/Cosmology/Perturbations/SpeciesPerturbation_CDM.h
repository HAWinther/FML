#ifndef FML_COSMOLOGY_PERTURBATIONS_SPECIESPERTURBATION_CDM_H
#define FML_COSMOLOGY_PERTURBATIONS_SPECIESPERTURBATION_CDM_H

// =====================================================================================================================
// Cold dark matter (pressureless, no anisotropic stress), variables delta and v (Newtonian gauge):
//   delta' = (ck/Hp) v - 3 Phi',   v' = -v - (ck/Hp) Psi
// Adiabatic initial conditions: delta = -3/2 Psi, v = -(ck/2Hp) Psi (isocurvature: IsoValues delta_c, v_c).
// Moments: delta rho = rho delta, flux = -rho v. Outputs delta_cdm, v_cdm.
// Built from the background species it perturbs (any Species: only rho(x) is used).
// =====================================================================================================================

#include <FML/Cosmology/Background/Species.h>
#include <FML/Cosmology/Perturbations/PerturbationTypes.h>
#include <FML/Cosmology/Perturbations/SpeciesPerturbation.h>

#include <string>
#include <vector>

namespace FML::COSMOLOGY::PERTURBATIONS {

    class SpeciesPerturbation_CDM : public SpeciesPerturbation {
      public:
        SpeciesPerturbation_CDM(const BACKGROUND::Species & cdm, const PerturbationContext & /*ctx*/) : species(cdm) {}
        std::string name() const override { return "cdm"; }
        int n_vars(int) const override { return 2; }
        int cache_size() const override { return 1; }
        void update_cache(const ModeAtX & b, double * cache) const override { cache[0] = b.H0a2 * species.rho(b.x); }
        void initial_conditions(const ModeAtX & b, const InitialData & ic, double * y) const override {
            if (ic.iso) {
                y[0] = ic.iso->delta_c;
                y[1] = ic.iso->v_c;
                return;
            }
            y[0] = -1.5 * ic.Psi;
            y[1] = -0.5 * b.ckH * ic.Psi;
        }
        Moments moments(const ModeAtX &, const double * cache, int, const double * y) const override {
            Moments m;
            m.drho = cache[0] * y[0];
            m.flux = -cache[0] * y[1];
            return m;
        }
        void rhs(const ModeAtX & b, const double *, int, const MetricAtX & m, const double * y, double * dydx)
            const override {
            dydx[0] = b.ckH * y[1] - 3.0 * m.dPhi;
            dydx[1] = -y[1] - b.ckH * m.Psi;
        }
        std::vector<std::string> output_names() const override { return {"delta_cdm", "v_cdm"}; }
        void outputs(const ModeAtX &, const double *, int, const double * y, double * out) const override {
            out[0] = y[0];
            out[1] = y[1];
        }

      private:
        const BACKGROUND::Species & species;
    };

} // namespace FML::COSMOLOGY::PERTURBATIONS

#endif
