#ifndef FML_PERTURBATIONS_PERTURBEDFIELDEQUATION_H
#define FML_PERTURBATIONS_PERTURBEDFIELDEQUATION_H

// =====================================================================================================================
// PerturbedFieldEquation: the linear perturbations of the field equations of a model (the perturbation part of the
// solver's Gravity, Cosmology/Background/Gravity.h), built from the PerturbationContext (and the background's
// FieldEquations it perturbs) instead of receiving the background at each call.
//
// Newtonian gauge, conventions of PerturbationTypes.h. The field equations own n_vars() variables of the ODE state
// (GR: Phi; Brans-Dicke: Phi and the scalar field perturbation). From these and the summed Moments of all species
// (photons and baryons included) metric() gives Phi, Psi and Phi', which drive the species, and the x-derivatives of
// its own variables. In GR (PerturbedFieldEquation_GeneralRelativity.h):
//   Psi  = -Phi - 3 shear / (c^2 k^2 s_2)
//   Phi' = Psi - (ck/Hp)^2 (1 - 3K/k^2) Phi / 3 + delta rho / (2 Hp^2)
// Background quantities needed per x go in a cache (cache_size() doubles per mode, update_cache()).
//
// G_eff_initial(): the gravitational coupling of the Friedmann equation at the initial time in units of the G of the
// density parameters (enters the neutrino shear of the adiabatic initial conditions). tensor(x): the coefficients of
// the tensor-mode equation (PerturbationTypes.h).
// =====================================================================================================================

#include <FML/Perturbations/PerturbationTypes.h>

#include <string>
#include <vector>

namespace FML::COSMOLOGY::PERTURBATIONS {

    class PerturbedFieldEquation {
      public:
        explicit PerturbedFieldEquation(const PerturbationContext & ctx) : ctx(ctx) {}
        virtual ~PerturbedFieldEquation() = default;
        virtual std::string name() const = 0;
        virtual std::string info() const { return name(); }

        virtual int n_vars() const { return 1; }
        virtual int cache_size() const { return 0; }
        virtual void update_cache(const ModeAtX & /*b*/, double * /*cache*/) const {}
        /// Effective gravitational constant of the Friedmann equation at the initial time, in units of G
        virtual double G_eff_initial() const { return 1.0; }
        /// Variables at the initial time (default: y[0] = Phi)
        virtual void initial_conditions(const ModeAtX & /*b*/, const InitialData & ic, double * y) const {
            y[0] = ic.Phi;
        }
        /// Phi, Psi, Phi' from the variables y and the total Moments; dydx: the derivatives of y
        virtual MetricAtX metric(const ModeAtX & b,
                                 const double * cache,
                                 const double * y,
                                 const Moments & total,
                                 double * dydx) const = 0;
        /// Tensor-mode coefficients at x
        virtual TensorCoefficients tensor(double /*x*/) const { return {}; }

        /// Extra quantities stored as functions of (x, k)
        virtual std::vector<std::string> output_names() const { return {}; }
        virtual void outputs(const ModeAtX & /*b*/,
                             const double * /*cache*/,
                             const double * /*y*/,
                             const MetricAtX & /*m*/,
                             double * /*out*/) const {}

      protected:
        PerturbationContext ctx;
    };

} // namespace FML::COSMOLOGY::PERTURBATIONS

#endif
