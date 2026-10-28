#ifndef FML_COSMOLOGY_BACKGROUND_GRAVITY_H
#define FML_COSMOLOGY_BACKGROUND_GRAVITY_H

// =============================================================================
// Gravity: how the species source the expansion and the metric perturbations.
//
// Background: E^2(x) = (H/H0)^2 and its first two x-derivatives. The species densities are available
// through the BackgroundCosmology (rho_species, drho_species, ddrho_species, rhoK for the curvature).
// A theory with extra background degrees of freedom keeps them itself (e.g. solved by the model's build
// function, see models/BransDicke.h).
//
// Perturbations (Newtonian gauge, conventions of Perturbations.h): the gravity owns n_vars() variables of
// the ODE state (GR: Phi; Brans-Dicke: Phi and the scalar field perturbation). From these and the summed
// Moments of all species (photons and baryons included) metric() gives Phi, Psi and Phi', which drive the
// species, and the x-derivatives of its own variables. In GR (GeneralRelativity.h):
//   Psi  = -Phi - 3 shear / (c^2 k^2 s_2)
//   Phi' = Psi - (ck/Hp)^2 (1 - 3K/k^2) Phi / 3 + delta rho / (2 Hp^2)
// Background quantities needed per x go in a cache (cache_size() doubles per mode, update_cache()).
//
// Tensor modes (Tensors.h) obey, in conformal time,
//   h'' + (2 + alpha_M) Hp h' + (1 + alpha_T) k^2 h = (2 a^2 / M*^2) Pi_T   (Pi_T: anisotropic stress)
// with M*^2 the effective Planck mass in units of 1/(8 pi G) and alpha_M = dln M*^2/dx. tensor() gives these
// (GR: M*^2 = 1, alpha_M = alpha_T = 0).
// =============================================================================

#include <FML/Cosmology/Background/Species.h>

#include <string>
#include <vector>

namespace FML::COSMOLOGY {

    class BackgroundCosmology;

    struct TensorCoefficients {
        double M2 = 1.0, alpha_M = 0.0, alpha_T = 0.0;
    };

    class Gravity {
      public:
        virtual ~Gravity() = default;
        virtual std::string name() const = 0;
        virtual std::string info() const { return name(); }

        // Called once by the BackgroundCosmology constructor (e.g. to check the model)
        virtual void init_background(const BackgroundCosmology & /*bg*/) {}
        virtual double E2(double x, const BackgroundCosmology & bg) const = 0;
        virtual double dE2dx(double x, const BackgroundCosmology & bg) const = 0;
        virtual double ddE2ddx(double x, const BackgroundCosmology & bg) const = 0;

        // --- Perturbations ---------------------------------------------------------
        virtual int n_vars() const { return 1; }
        virtual int cache_size() const { return 0; }
        virtual void update_cache(const BackgroundCosmology & /*bg*/, const ModeAtX & /*b*/, double * /*cache*/) const {
        }
        // Effective gravitational constant in the Friedmann equation at the initial time, in units of G
        // (enters the neutrino shear of the adiabatic initial conditions)
        virtual double G_eff_initial() const { return 1.0; }
        // Variables at the initial time (adiabatic; default: y[0] = Phi)
        virtual void initial_conditions(const ModeAtX & /*b*/, const InitialData & ic, double * y) const {
            y[0] = ic.Phi;
        }
        // Phi, Psi, Phi' from the gravity variables y and the total Moments; dydx: derivatives of y
        virtual MetricAtX metric(const ModeAtX & b,
                                 const double * cache,
                                 const double * y,
                                 const Moments & total,
                                 double * dydx) const = 0;
        // Tensor-mode coefficients at time x (see above)
        virtual TensorCoefficients tensor(double /*x*/, const BackgroundCosmology & /*bg*/) const { return {}; }

        // Extra quantities stored as functions of (x, k) (Perturbations::get)
        virtual std::vector<std::string> output_names() const { return {}; }
        virtual void outputs(const ModeAtX & /*b*/,
                             const double * /*cache*/,
                             const double * /*y*/,
                             const MetricAtX & /*m*/,
                             double * /*out*/) const {}
    };

} // namespace FML::COSMOLOGY

#endif
