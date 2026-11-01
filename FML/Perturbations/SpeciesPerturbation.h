#ifndef FML_PERTURBATIONS_SPECIESPERTURBATION_H
#define FML_PERTURBATIONS_SPECIESPERTURBATION_H

// =====================================================================================================================
// SpeciesPerturbation: the linear perturbations of one species of the background (the solver's
// SpeciesPerturbations interface, Cosmology/Background/Species.h).
//
// A species owns a block of variables of the ODE state and gives their initial conditions, their right-hand side
// given the metric (Phi, Psi, Phi'), and its Moments (delta rho, delta p, flux, shear: PerturbationTypes.h), through
// which alone it sources the metric. Photons and baryons are not species here: they are coupled by Thomson scattering
// and stay in the core of the solver.
//
// One object per species and run, shared by all k-modes (and threads): it must not keep per-mode state. Per-x
// background quantities go in a cache (cache_size() doubles per mode) filled by update_cache().
//
// A species can switch between approximation schemes ("modes", e.g. a Boltzmann hierarchy and a fluid
// approximation): mode 0 is used from the initial time and the mode increases by one at each of the switch_points(k).
// Each mode has its own number of variables, and switch_mode converts the variables.
//
// free_streaming_density(x): the density [rho_c0] the species contributes to the free-streaming radiation (neutrinos,
// 0 otherwise), from which the solver computes f_nu and OmNu of the adiabatic initial conditions (the solver's
// Species::free_streaming()).
//
// free_streaming_initial_conditions: the adiabatic multipoles N_0..N_lmax of free-streaming radiation, shared by the
// massless and massive neutrinos:
//   N_0 = -Psi/2,  N_1 = (ck/6Hp) Psi,
//   N_2 = -s_2 (c k a)^2 Phi / (12 H0^2 G_eff OmNu) / (1 + 5/(2 f_nu)),  N_l = ck s_l/((2l+1) Hp) N_{l-1}
// (the physical shear of the flat-space solution in the symmetric convention with s_2; OmNu = a^4 rho of all
// free-streaming species, G_eff in units of G). Isocurvature: N_0, N_1, N_2 from IsoValues, the higher ones 0.
// =====================================================================================================================

#include <FML/Perturbations/PerturbationTypes.h>
#include <FML/Units/Units.h>

#include <cmath>
#include <string>
#include <vector>

namespace FML::COSMOLOGY::PERTURBATIONS {

    class SpeciesPerturbation {
      public:
        virtual ~SpeciesPerturbation() = default;
        virtual std::string name() const = 0;
        virtual std::string info() const { return name(); }

        /// The x where the mode increases by one (k in 1/m)
        virtual std::vector<double> switch_points(double /*k*/) const { return {}; }
        /// Number of variables in a mode
        virtual int n_vars(int mode) const = 0;
        /// The ODE system in this mode is large and not stiff: an explicit method is faster
        virtual bool prefers_explicit(int /*mode*/) const { return false; }

        virtual int cache_size() const { return 0; }
        virtual void update_cache(const ModeAtX & /*b*/, double * /*cache*/) const {}

        /// Mode-0 variables at the initial time
        virtual void initial_conditions(const ModeAtX & b, const InitialData & ic, double * y) const = 0;
        /// Variables of new_mode from those of new_mode - 1 at the switch point
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

        /// Quantities stored as functions of (x, k), e.g. delta_cdm, v_cdm
        virtual std::vector<std::string> output_names() const = 0;
        virtual void
        outputs(const ModeAtX & b, const double * cache, int mode, const double * y, double * out) const = 0;

        /// rho / rho_c0 of the free-streaming radiation this species is (0 if it is not)
        virtual double free_streaming_density(double /*x*/) const { return 0.0; }
    };

    /// The adiabatic (or isocurvature) multipoles N_0..N_lmax of free-streaming radiation (see the header)
    inline std::vector<double> free_streaming_initial_conditions(const PerturbationContext & ctx,
                                                                 const ModeAtX & b,
                                                                 const InitialData & ic,
                                                                 int lmax) {
        const double k = b.k, c = FML::UTILS::SI::c;
        std::vector<double> N(lmax + 1, 0.0);
        if (ic.iso) {
            N[0] = ic.iso->N0;
            N[1] = ic.iso->N1;
            N[2] = ic.iso->N2;
            return N;
        }
        N[0] = -0.5 * ic.Psi;
        N[1] = b.ckH / 6.0 * ic.Psi;
        N[2] = -ctx.s_ell(k, 2) * std::pow(c * k * b.a, 2) * ic.Phi / (12.0 * ic.H0 * ic.H0 * ic.G_eff * ic.OmNu) /
               (1.0 + 2.5 / ic.f_nu);
        for (int ell = 3; ell <= lmax; ell++)
            N[ell] = b.ckH * ctx.s_ell(k, ell) / (2.0 * ell + 1.0) * N[ell - 1];
        return N;
    }

} // namespace FML::COSMOLOGY::PERTURBATIONS

#endif
