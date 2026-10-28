#ifndef FML_COSMOLOGY_BACKGROUND_SPECIES_H
#define FML_COSMOLOGY_BACKGROUND_SPECIES_H

// =============================================================================
// Species: the matter and energy components of a model.
//
// A Species describes one component in the background (energy density and pressure as functions of
// x = ln a, in units of today's critical density rho_c0) and, optionally, its linear perturbations
// (a SpeciesPerturbations object). A model (Model.h) is a list of species plus a theory of gravity
// (Gravity.h). Photons and baryons are special: they are coupled by Thomson scattering and their
// perturbations (with tight coupling, polarization and the line-of-sight sources) are part of the core of
// Perturbations; their Species only describe the background.
//
// Adding a species: derive from Species (background) and, if it clusters, from SpeciesPerturbations,
// and return the latter from Species::perturbations(). The perturbation equations are written in
// Newtonian gauge with the conventions of Perturbations.h; a species communicates with the metric only
// through its Moments (density, momentum density and anisotropic stress) and receives Phi, Psi, Phi'.
// See models/StandardSpecies.h for CDM, massless and massive neutrinos.
// =============================================================================

#include <limits>
#include <memory>
#include <string>
#include <vector>

namespace FML::COSMOLOGY {

    class Perturbations;

    class SpeciesPerturbations;

    class Species {
      public:
        // What a species is, so that the named accessors of BackgroundCosmology (rhoB, rhoCDM, ...) and
        // the recombination code can find it. New kinds of matter use Other.
        enum class Role { Baryons, Photons, CDM, MasslessNeutrinos, MassiveNeutrinos, DarkEnergy, Other };

        virtual ~Species() = default;
        virtual std::string name() const = 0;
        virtual Role role() const { return Role::Other; }

        // Energy density and pressure in units of rho_c0 (pressure: rho_c0 c^2), and x-derivatives of rho
        virtual double rho(double x) const = 0;
        virtual double p(double x) const = 0;
        virtual double drhodx(double x) const = 0;
        virtual double ddrhoddx(double x) const = 0;
        double w(double x) const { return p(x) / rho(x); }

        // Free-streaming radiation at early times (neutrinos): enters f_nu in the adiabatic initial conditions
        virtual bool free_streaming() const { return false; }

        // Perturbations of this species, or nullptr if it is smooth (or handled by the core of Perturbations).
        // Called once by the Perturbations constructor.
        virtual std::unique_ptr<SpeciesPerturbations> perturbations(const Perturbations &) const { return nullptr; }

        virtual std::string info() const { return name(); }
    };

    // -----------------------------------------------------------------------------
    // Perturbations of a species
    // -----------------------------------------------------------------------------

    // Background quantities at one x for one mode k, computed once per x by Perturbations and shared by
    // all species. Densities are multiplied by H0^2 a^2 (units 1/s^2), as they appear in the Einstein equations.
    struct ModeAtX {
        double x = std::numeric_limits<double>::quiet_NaN();
        double k = 0.0;          // [1/m]
        double s2 = 1.0;         // curvature factor s_2 of this mode (Perturbations::s_ell)
        double curv = 1.0;       // (k^2 - 3K)/k^2
        double a, Hp, dHpH, ckH; // a, Hp [1/s], Hp'/Hp, ck/Hp
        double dtau, ddtau, R;   // tau', tau'', R = 4 rho_gamma / (3 rho_b)
        double eta;              // conformal time [m]
        double cotK_H;           // c cot_K(eta) / Hp: free-streaming truncation rate (= c / (Hp eta) when flat)
        double H0a2;             // H0^2 a^2 [1/s^2]
        double H0a2_rho_b, H0a2_rho_gamma;
    };

    // Contributions to the Einstein equations, all times H0^2 a^2 / rho_c0 (summed over species):
    //   drho  = delta rho
    //   dp    = delta p (pressure perturbation; enters only theories that couple to the trace, e.g. Brans-Dicke)
    //   flux  = (rho + p) theta / (ck) = -(rho + p) v         (v as for CDM: delta_cdm' = (ck/Hp) v - 3 Phi')
    //   shear = (3/2) (rho + p) sigma                         (= 4 rho Theta_2 for photons)
    struct Moments {
        double drho = 0.0, dp = 0.0, flux = 0.0, shear = 0.0;
        Moments & operator+=(const Moments & m) {
            drho += m.drho;
            dp += m.dp;
            flux += m.flux;
            shear += m.shear;
            return *this;
        }
    };

    // Metric at one x: Phi (evolved), Psi and Phi' (from the Einstein equations, see Gravity.h)
    struct MetricAtX {
        double Phi = 0.0, Psi = 0.0, dPhi = 0.0;
    };

    // Explicit initial values of the standard species (isocurvature modes, Perturbations.h), in the variables of
    // the code: photon Theta_0, Theta_1, baryon and CDM delta, v, free-streaming neutrino N_0, N_1, N_2
    struct IsoValues {
        double Theta0 = 0, Theta1 = 0, delta_b = 0, v_b = 0, delta_c = 0, v_c = 0, N0 = 0, N1 = 0, N2 = 0;
    };

    // Initial conditions (superhorizon, radiation domination): adiabatic with curvature perturbation R = 1, or, if
    // iso is set, explicit values (species not covered by IsoValues start from the adiabatic form with Psi, Phi)
    struct InitialData {
        const IsoValues * iso = nullptr;
        double Psi, Phi;
        double f_nu;        // fraction of the radiation density in free-streaming species
        double OmNu;        // a^4 rho of the free-streaming species [rho_c0]
        double H0;          // [1/s]
        double G_eff = 1.0; // effective gravitational constant / G at the initial time (Gravity::G_eff_initial)
    };

    // One object per species and run, shared by all k-modes (and threads): it must not keep per-mode state.
    // Per-x background quantities go in a cache (cache_size() doubles per mode) filled by update_cache().
    //
    // A species can switch between approximation schemes ("modes", e.g. a Boltzmann hierarchy and a fluid
    // approximation): mode 0 is used from the initial time and the mode increases by one at each of the
    // switch_points(k). Each mode has its own number of variables, and switch_mode converts the variables.
    class SpeciesPerturbations {
      public:
        virtual ~SpeciesPerturbations() = default;
        virtual std::string name() const = 0;

        virtual std::vector<double> switch_points(double /*k*/) const { return {}; }
        virtual int n_vars(int mode) const = 0;
        // The ODE system in this mode is large and not stiff: an explicit method is faster (method_ncdm)
        virtual bool prefers_explicit(int /*mode*/) const { return false; }

        virtual int cache_size() const { return 0; }
        virtual void update_cache(const ModeAtX & /*b*/, double * /*cache*/) const {}

        // Mode-0 variables at the initial time
        virtual void initial_conditions(const ModeAtX & b, const InitialData & ic, double * y) const = 0;
        // Variables of new_mode from those of new_mode - 1 at the switch point
        virtual void switch_mode(const ModeAtX & /*b*/,
                                 const double * /*cache*/,
                                 int /*new_mode*/,
                                 const double * /*y_old*/,
                                 double * /*y_new*/) const {}

        virtual Moments moments(const ModeAtX & b, const double * cache, int mode, const double * y) const = 0;
        virtual void rhs(const ModeAtX & b,
                         const double * cache,
                         int mode,
                         const MetricAtX & m,
                         const double * y,
                         double * dydx) const = 0;

        // Quantities stored as functions of (x, k), e.g. delta_cdm, v_cdm (Perturbations::get(name, x, k))
        virtual std::vector<std::string> output_names() const = 0;
        virtual void
        outputs(const ModeAtX & b, const double * cache, int mode, const double * y, double * out) const = 0;

        virtual std::string info() const { return name(); }
    };

} // namespace FML::COSMOLOGY

#endif
