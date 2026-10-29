#ifndef FML_COSMOLOGY_PERTURBATIONS_PERTURBATIONS_H
#define FML_COSMOLOGY_PERTURBATIONS_PERTURBATIONS_H

// =============================================================================
// Linear perturbations in Newtonian gauge, ds^2 = -(1 + 2 Psi) dt^2 + a^2 (1 + 2 Phi) dx^2
// (conventions of Dodelson / Callin 2006). Time variable x = ln a, k in 1/m (SI).
//
// Structure: the core evolves the photon-baryon plasma (baryons coupled to the photon temperature
// and polarization hierarchies by Thomson scattering, with tight coupling) and the potential Phi. All other
// species (CDM, neutrinos, ...) come from the model (Species.h): each contributes its own block of variables,
// equations, initial conditions and stored outputs, and the Gravity of the model gives Psi and Phi' from
// the summed Moments (density, momentum, anisotropic stress) of all species. See models/StandardSpecies.cpp
// for CDM and the neutrinos, and models/GeneralRelativity.cpp for the Einstein equations.
//
// Physics of the core:
//   * Baryons (delta_b, v_b), photon temperature multipoles Theta_0..Theta_lmax, photon polarization
//     multipoles Theta^P_0..Theta^P_lmax, metric potentials Phi, Psi. Pi = Theta_2 + Theta^P_0 + Theta^P_2.
//     (options.polarization = false gives the temperature-only system with Pi = Theta_2.)
//   * Adiabatic initial conditions deep in the radiation era, normalised to a primordial
//     curvature perturbation R = 1: Psi = -1/(3/2 + 2 f_nu/5), Phi = -(1 + 2 f_nu/5) Psi,
//     f_nu = rho_fs / (rho_gamma + rho_fs) with rho_fs the free-streaming species (neutrinos).
//   * Tight coupling approximation (Theta_0, Theta_1 only; higher multipoles algebraic)
//     until |tau'| < 10, |ck/(Hp tau')| > 0.1, or z < 1700, then the full hierarchy.
//   * Spatial curvature K (from the background): the curvature factors s_l in all hierarchies, k^2 - 3K
//     in the Einstein equations, cot_K truncation, and modes labelled by q = sqrt(k^2 + K) (see s_ell).
//
// Outputs (2D splines in (x, k)): delta_b, v_b, Phi, Psi, dPhi/dx, Theta_ell and Theta^P_ell for
// ell = 0..lmax, Pi, the outputs of the species (get(name, x, k): delta_cdm, v_cdm, Nu_0..2, delta_ncdm,
// v_ncdm, ...), and the source functions
//   S_T(k,x) = g (Theta_0 + Psi + Pi/4) + e^{-tau} (Psi' - Phi')
//              - 1/(ck) d/dx(Hp g v_b) + 3/(4 c^2 k^2) d/dx[Hp d/dx(Hp g Pi)]
//   S_E(k,x) = 3 g Pi / (4 [k (eta0 - eta)]^2)
// The numerator 3 g Pi / 4 of S_E is splined (smooth); the 1/[k(eta0-eta)]^2 factor is
// applied on evaluation (it diverges today, where g = 0).
//
// The integration is split into segments with fixed equations (tight coupling or not, and the
// approximation scheme of each species, e.g. massive-neutrino hierarchy or fluid), each solved with its
// own ODE method.
// =============================================================================

#include <FML/Cosmology/Background/BackgroundCosmology.h>
#include <FML/Cosmology/Recombination/RecombinationHistory.h>
#include <FML/Cosmology/Background/Species.h>
#include <FML/Cosmology/Common.h>

#include <array>
#include <limits>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace FML::COSMOLOGY {

    struct PerturbationOptions {
        // k-grid [1/m], logarithmically spaced. With curvature these are the bounds on
        // q = sqrt(k^2 + K) (see Perturbations::s_ell); q = k when flat.
        double k_min = 0.00005 / Constants::Mpc;
        double k_max = 0.3 / Constants::Mpc;
        int n_k = 400;
        double dk_max = 0.0; // > 0: log-linear grid, spacing min(k dlnk, dk_max) with dlnk = ln(k_max/k_min)/(n_k-1)
                             // (the sources oscillate in k with period ~2 pi/r_s, so a linear spacing at large
                             //  k needs far fewer modes than a log grid; n_k then only sets the low-k spacing)

        // x-grid on which results are stored: coarse before recombination, fine around it
        double x_start = -18.0;
        double x_end = 0.0;
        int n_x_early = 300;   // [x_start, -8]
        int n_x_recomb = 1200; // [-8, -6]
        int n_x_late = 600;    // [-6, x_end]

        int lmax = 8; // Highest photon multipole in the full hierarchy

        // Tight coupling criteria
        double tc_tau_min = 10.0; // end TC when |tau'| < this
        // or when |ck / (Hp tau')| > this (CLASS's tau_c/tau_k = 0.01; the first-order tight-coupling error grows
        // with k/kappa': 0.1 left TT and EE 0.2-0.4% low at l = 1000-2500 against CLASS)
        double tc_k_over_tau = 0.01;
        double tc_z_max = 1700.0; // or at the latest when z < this (start of recombination)

        // ODE accuracy and methods. After tight coupling the system is stiff (|tau'| ~ 1e3 while
        // ck/Hp is large), so the full hierarchy uses an implicit BDF method by default (~30x faster
        // than explicit Runge-Kutta at k = 0.3/Mpc for the same accuracy).
        double ode_abserr = 1e-10;
        double ode_relerr = 1e-8;
        ODEStepper method_tc = gsl_odeiv2_step_rk8pd;
        ODEStepper method_full = gsl_odeiv2_step_msbdf;
        ODEStepper method_ncdm = gsl_odeiv2_step_rk8pd; // after tight coupling, when a species prefers an
                                                        // explicit method (massive-neutrino hierarchy)

        // Initial conditions: ic_adiabatic times the adiabatic mode (curvature perturbation R = 1) plus, for each
        // isocurvature mode m (0 = CDI, 1 = BI, 2 = NID, 3 = NIV; entropy S_m = 1), iso_amplitude[m]
        // (k/k_pivot)^iso_tilt[m] (a k-dependent mixture; see Isocurvature.h). Isocurvature modes need GR and the
        // standard species.
        double ic_adiabatic = 1.0;
        std::array<double, 4> iso_amplitude{{0.0, 0.0, 0.0, 0.0}};
        std::array<double, 4> iso_tilt{{0.0, 0.0, 0.0, 0.0}};

        bool polarization = true;             // Photon polarization hierarchy
        bool store_source_components = false; // Also store the individual terms of S_T (SW, ISW, Doppler, quadrupole)
        bool neutrinos = true;                // Massless neutrino hierarchy (only active if Omega_nu > 0)
        int lmax_nu = 22;   // Highest neutrino multipole. Relative to 30: C_ell < 2e-4, P(k < 0.3/Mpc) < 5e-4
                            // (lmax_nu = 12 gives 2e-4 in C_ell but 0.35% in P(k) at k = 0.3/Mpc)
        int lmax_ncdm = 17; // Highest massive-neutrino multipole (CLASS default 17)
        int n_q_ncdm = 8;   // Momentum nodes (Gauss-Laguerre) for the massive neutrinos
        double ncdm_fluid_trigger = 31.0; // k eta above which the massive neutrinos are a fluid (CLASS default;
                                          // 1e30: always the full hierarchy, much slower)
    };

    class Perturbations {
      public:
        Perturbations(const BackgroundCosmology & cosmo,
                      const RecombinationHistory & rec,
                      PerturbationOptions options = PerturbationOptions());
        ~Perturbations();

        void solve();

        // Quantities of the core as functions of (x, k), k in 1/m
        double get_delta_b(double x, double k) const { return check(delta_b_spline)(x, k); }
        double get_v_b(double x, double k) const { return check(v_b_spline)(x, k); }
        double get_Phi(double x, double k) const { return check(Phi_spline)(x, k); }
        double get_dPhidx(double x, double k) const { return check(dPhidx_spline)(x, k); }
        double get_Psi(double x, double k) const { return check(Psi_spline)(x, k); }
        double get_Pi(double x, double k) const { return check(Pi_spline)(x, k); }
        double get_Theta(int ell, double x, double k) const;
        double get_ThetaP(int ell, double x, double k) const; // requires polarization

        // Outputs of the species and the gravity by name (output_names), e.g. "delta_cdm"
        double get(const std::string & name, double x, double k) const;
        bool has(const std::string & name) const { return output_index.count(name) > 0; }
        const std::vector<std::string> & get_output_names() const { return output_names; }
        // Shorthands for the standard species
        double get_delta_cdm(double x, double k) const { return get("delta_cdm", x, k); }
        double get_v_cdm(double x, double k) const { return get("v_cdm", x, k); }
        double get_Nu(int ell, double x, double k) const; // massless neutrinos, ell = 0, 1, 2
        bool has_neutrinos() const { return has("Nu_0"); }
        // Massive neutrinos: delta = delta rho / rho, and the velocity v defined like v_cdm,
        // (rho + p) v = -(rho + p) theta / (ck)
        double get_delta_ncdm(double x, double k) const { return get("delta_ncdm", x, k); }
        double get_v_ncdm(double x, double k) const { return get("v_ncdm", x, k); }
        bool has_massive_nu() const { return has("delta_ncdm"); }

        double get_Source_T(double x, double k) const { return check(Source_T_spline)(x, k); }
        // Terms of S_T (requires options.store_source_components):
        //   SW = g (Theta_0 + Psi),  ISW = e^{-tau} (Psi' - Phi'),  Doppler = -1/(ck) d(Hp g v_b)/dx,
        //   Quadrupole = g Pi/4 + 3/(4 c^2 k^2) d/dx[Hp d/dx(Hp g Pi)]
        enum class SourceTerm { All = 0, SW = 1, ISW = 2, Doppler = 3, Quadrupole = 4 };
        double get_Source_T_term(SourceTerm term, double x, double k) const;
        double get_Source_E(double x, double k) const; // 3 g Pi / (4 [k(eta0 - eta)]^2)
        // Part of S_T that multiplies the radial function Phi_ell itself, g (Theta_0 + Psi) + e^{-tau}(Psi' - Phi')
        // (SW + ISW). The Doppler and quadrupole terms are -g v_b and g Pi / 2 times the first and second
        // derivative radial functions; S_T above has them integrated by parts, which is only valid when flat.
        double get_Source_T0(double x, double k) const { return check(Source_T0_spline)(x, k); }
        double get_Source_E_numerator(double x, double k) const { return check(Source_E_num_spline)(x, k); }
        bool has_polarization() const { return options.polarization; }

        // Where tight coupling ends for a given k (computed on demand)
        double x_tight_coupling_end(double k) const;

        const std::vector<double> & get_k_array() const { return k_array; }
        const std::vector<double> & get_x_array() const { return x_array; }
        const PerturbationOptions & get_options() const { return options; }
        const BackgroundCosmology & get_cosmo() const { return cosmo; }
        const RecombinationHistory & get_rec() const { return rec; }
        // Perturbed species of the model (those whose Species::perturbations() is not null)
        const std::vector<std::unique_ptr<SpeciesPerturbations>> & get_species() const { return species; }

        void info() const;
        // Write all quantities as a function of x for one k [1/m]
        void output(double k, const std::string & filename) const;

        // -------------------------------------------------------------------------
        // Spatial curvature K (from the background). k is the Laplacian eigenvalue, -nabla^2 -> k^2,
        // and q^2 = k^2 + K labels the modes (q = nu sqrt|K|; in a closed universe the k-grid only
        // contains the physical modes, integer nu >= 3). The free-streaming couplings between multipoles l-1 and l
        // carry the factor
        //   s_l = sqrt(1 - (l^2 - 1) K / k^2) = sqrt(q^2 - l^2 K) / k,
        // in the symmetric convention used for all hierarchies here (CLASS uses the same s_l).
        // In this convention the physical photon/neutrino shear is Theta_2 / s_2. In a closed
        // universe s_l = 0 for l >= nu and the hierarchy terminates by itself.
        // -------------------------------------------------------------------------
        double get_K() const { return K; }
        double s_ell(double k, int ell) const {
            if (K == 0.0)
                return 1.0;
            const double s2 = 1.0 - (ell * ell - 1.0) * K / (k * k);
            return s2 > 0.0 ? std::sqrt(s2) : 0.0;
        }

      private:
        const BackgroundCosmology & cosmo;
        const RecombinationHistory & rec;
        PerturbationOptions options;
        double K = 0.0; // spatial curvature [1/m^2], see s_ell

        // Perturbed species, the Species they belong to, and their stored outputs
        std::vector<std::unique_ptr<SpeciesPerturbations>> species;
        std::vector<const Species *> species_bg;
        std::vector<std::string> output_names; // of the species, then of the gravity
        std::map<std::string, int> output_index;
        int n_gravity_outputs = 0;

        // Layout of the ODE state vector: the core (delta_b, v_b, the variables of the gravity (GR: Phi),
        // Theta_0.., Theta^P_0..), then one block per species in the given mode
        struct Layout {
            static constexpr int delta_b = 0, v_b = 1, metric = 2;
            int Theta = 3;
            bool tc = true;
            int n_Theta = 0, ThetaP = 0, n_ThetaP = 0;
            std::vector<int> mode, offset;
            int n_tot = 0;
        };
        Layout layout(bool tc, const std::vector<int> & modes) const;

        // Background at one x for mode k (shared by the core and the species), with the per-species caches.
        // The implicit ODE solver evaluates the RHS many times at the same x (finite-difference Jacobian),
        // so this is recomputed only when x changes.
        struct ModeState {
            ModeAtX b;
            std::vector<std::vector<double>> cache;
            std::vector<double> gcache; // of the gravity
        };
        ModeState make_mode(double k) const;
        void update(double x, ModeState & st) const;

        // Phi, Psi and Phi' from the Moments of the core (photons with quadrupole Theta2, baryons) and the species;
        // dydx_metric: derivatives of the gravity variables
        MetricAtX
        metric(const ModeState & st, const Layout & lay, const double * y, double Theta2, double * dydx_metric) const;

        // Initial conditions and right-hand sides
        std::vector<double> initial_conditions(ModeState & st) const;
        // the isocurvature mode m (0..3) with S = 1, in the variables of the code
        IsoValues isocurvature_values(int m, double x, double k, double & Phi, double & Psi) const;
        std::vector<double>
        change_layout(ModeState & st, const Layout & from, const Layout & to, const std::vector<double> & y) const;
        int rhs_tight_coupling(const ModeState & st, const Layout & lay, const double * y, double * dydx) const;
        int rhs_full(const ModeState & st, const Layout & lay, const double * y, double * dydx) const;
        void rhs_species(const ModeState & st,
                         const Layout & lay,
                         const MetricAtX & m,
                         const double * y,
                         double * dydx) const;

        // Multipoles in tight coupling (algebraic in Theta_1): fills Theta_0..lmax and
        // Theta^P_0..lmax (zeros without polarization). Returns Pi.
        double multipoles_tc(double x,
                             double k,
                             double Theta0,
                             double Theta1,
                             std::vector<double> & Theta,
                             std::vector<double> & ThetaP) const;
        // Theta_2 in tight coupling
        double Theta2_tc(double x, double k, double Theta1) const;

        std::vector<double> k_array, x_array;
        bool solved = false;

        Spline2D delta_b_spline, v_b_spline;
        Spline2D Phi_spline, dPhidx_spline, Psi_spline, Pi_spline, Source_T_spline, Source_E_num_spline,
            Source_T0_spline;
        Spline2D S_SW_spline, S_ISW_spline, S_Doppler_spline, S_Quad_spline;
        std::vector<Spline2D> Theta_spline, ThetaP_spline;
        std::vector<Spline2D> output_spline; // species outputs, in the order of output_names

        const Spline2D & check(const Spline2D & s) const;
        std::vector<double> make_x_grid() const;
    };

} // namespace FML::COSMOLOGY

#endif
