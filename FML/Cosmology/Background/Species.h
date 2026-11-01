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
// TRANSITIONAL: the solver takes the background from FML/Background and the perturbations from FML/Perturbations for
// every model (BackgroundCosmology.h) and no longer uses these classes. They remain as the references of the tests of
// FML/Background and FML/Perturbations (an explicit Model of BackgroundCosmology is background only; its species'
// perturbations are made by the tests with a Perturbations object as the host). New species belong in FML/Background
// and FML/Perturbations. The aliases of the perturbation data types below are used by the solver.
//
// Adding a species: derive from Species (background) and, if it clusters, from SpeciesPerturbations,
// and return the latter from Species::perturbations(). The perturbation equations are written in
// Newtonian gauge with the conventions of Perturbations.h; a species communicates with the metric only
// through its Moments (density, momentum density and anisotropic stress) and receives Phi, Psi, Phi'.
// See models/StandardSpecies.h for CDM, massless and massive neutrinos.
// =============================================================================

#include <FML/Perturbations/PerturbationTypes.h>

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

    // The data exchanged between the species, the gravity and the core of Perturbations are the types of
    // FML/Perturbations (PerturbationTypes.h; identical to the solver's former structs):
    //   ModeAtX     background quantities at one x for one mode k (densities times H0^2 a^2, units 1/s^2)
    //   Moments     contributions to the Einstein equations (drho, dp, flux, shear) times H0^2 a^2 / rho_c0
    //   MetricAtX   Phi (evolved), Psi and Phi' (from the field equations, see Gravity.h)
    //   IsoValues   explicit initial values of the standard species (isocurvature modes)
    //   InitialData initial conditions (adiabatic with R = 1, or isocurvature)
    using ModeAtX = PERTURBATIONS::ModeAtX;
    using Moments = PERTURBATIONS::Moments;
    using MetricAtX = PERTURBATIONS::MetricAtX;
    using IsoValues = PERTURBATIONS::IsoValues;
    using InitialData = PERTURBATIONS::InitialData;

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
