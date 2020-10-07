#ifndef LINEARPERTURBATIONS_HEADER
#define LINEARPERTURBATIONS_HEADER
#ifdef USE_OMP
#include <omp.h>
#endif
#include <algorithm>
#include <chrono>
#include <fstream>
#include <memory>
#include <vector>

#include <FML/Cosmology/BackgroundCosmology/BackgroundCosmology.h>
#include <FML/Cosmology/RecombinationHistory/RecombinationHistory.h>
#include <FML/Math/Math.h>
#include <FML/ODESolver/ODESolver.h>
#include <FML/Spline/Spline.h>
#include <FML/Timing/Timings.h>
#include <FML/Global/Global.h> // We only need ThisTask

namespace FML {
    namespace COSMOLOGY {

        // Accuracy settings for solving the tight-coupling ODEs
#define FIDUCIAL_HSTART_ODE_TIGHT 1e-3
#define FIDUCIAL_ABSERR_ODE_TIGHT 1e-10
#define FIDUCIAL_RELERR_ODE_TIGHT 1e-10
#define FIDUCIAL_HSTART_ODE_FULL 1e-3
#define FIDUCIAL_ABSERR_ODE_FULL 1e-10
#define FIDUCIAL_RELERR_ODE_FULL 1e-10

        using BackgroundCosmology = FML::COSMOLOGY::BackgroundCosmology;
        using RecombinationHistory = FML::COSMOLOGY::RecombinationHistory;
        using ODESolver = FML::SOLVERS::ODESOLVER::ODESolver;
        using ODEFunction = FML::SOLVERS::ODESOLVER::ODEFunction;
        using ODEFunctionJacobian = FML::SOLVERS::ODESOLVER::ODEFunctionJacobian;
        using Spline = FML::INTERPOLATION::SPLINE::Spline;
        using Spline2D = FML::INTERPOLATION::SPLINE::Spline2D;
        using DVector = std::vector<double>;
        using DVector2D = std::vector<DVector>;

        /// Struct for keeping track over what quantities we have at what index when solving the linear
        /// Einstein-Boltzmann system.
        typedef struct _PerturbationSystemInfo {
            int n_scalar{5};

            // Scalar quantities
            int index_delta_cdm{0};
            int index_delta_b{1};
            int index_v_cdm{2};
            int index_v_b{3};
            int index_Phi{4};

            // Number of multipoles
            int n_ell_theta;
            int n_ell_theta_p;
            int n_ell_nu;

            // Total number of perturbations
            int n_tot;

            // Start index of multipoles
            int index_theta_start;
            int index_theta_p_start;
            int index_nu_start;

            void debug_print() {
                int count = 0;
                for (int i = 0; i < n_scalar; i++)
                    std::cout << std::setw(5) << count++ << " Scalar\n";
                for (int i = 0; i < n_ell_theta; i++)
                    std::cout << std::setw(5) << count++ << " Theta" << std::to_string(i) << "\n";
                for (int i = 0; i < n_ell_theta_p; i++)
                    std::cout << std::setw(5) << count++ << " Theta_p" << std::to_string(i) << "\n";
                for (int i = 0; i < n_ell_nu; i++)
                    std::cout << std::setw(5) << count++ << " Nu" << std::to_string(i) << "\n";
                std::cout << "\n";
                assert(count == n_tot);
            }

            // Init with fiducial values
            _PerturbationSystemInfo() : _PerturbationSystemInfo(10, 10, 10) {}

            // Set up system
            _PerturbationSystemInfo(int n_ell_theta, int n_ell_theta_p, int n_ell_nu)
                : n_ell_theta(n_ell_theta), n_ell_theta_p(n_ell_theta_p), n_ell_nu(n_ell_nu),
                  n_tot(n_scalar + n_ell_theta + n_ell_theta_p + n_ell_nu), index_theta_start(n_scalar),
                  index_theta_p_start(index_theta_start + n_ell_theta),
                  index_nu_start(index_theta_p_start + n_ell_theta_p) {
                assert(n_ell_theta >= 2);
                assert(n_ell_theta_p >= 0);
                assert(n_ell_nu >= 0);
            }
        } PerturbationSystemInfo;

        /// Class for solving the linear perturbations (LCDM) in temperature, baryon, CDM, massless neutrinos etc. and
        /// computing source functions needed for computing power-spectra. Holds transfer functions.
        class Perturbations {
          private:
            // The cosmology to use
            std::shared_ptr<const BackgroundCosmology> cosmo{nullptr};

            // The recombination history to use
            std::shared_ptr<const RecombinationHistory> rec{nullptr};

            // The scales we integrate over (log-spacing)
            int n_k_total{};
            double k_min{};
            double k_max{};

            // Start and end of the time-integration
            DVector x_array_integration;
            int n_x_total{};
            double x_start{};
            double x_end{};

            // Spline ell=0,1,2 only or everything when we integrate perturbations
            bool pert_spline_all_ells{false};

            // Splines of scalar perturbations quantities
            Spline2D delta_cdm_spline{"delta_cdm_spline"};
            Spline2D delta_b_spline{"delta_b_spline"};
            Spline2D v_cdm_spline{"v_cdm_spline"};
            Spline2D v_b_spline{"v_b_spline"};
            Spline2D Phi_spline{"Phi_spline"};
            Spline2D Pi_spline{"Pi_spline"};
            Spline2D Psi_spline{"Psi_spline"};
            Spline2D dPidx_spline{"dPidx_spline"};
            // Gauge invariant curvature perturbation
            Spline2D zeta_spline{"zetaCurvPert_spline"};
            Spline2D dzetadx_spline{"dzetaCurvPertdx_spline"};

            // Source functions
            Spline2D ST_spline{"ST_spline"};
            Spline2D SE_spline{"SE_spline"};
            Spline2D SN_spline{"SN_spline"};

            // Source functions for individual contributions to the temperature source
            Spline2D SW_spline{"SW_spline"};
            Spline2D ISW_spline{"ISW_spline"};
            Spline2D VEL_spline{"VEL_spline"};
            Spline2D POL_spline{"POL_spline"};

            // Splines of mulipole quantities
            std::vector<Spline2D> Theta_spline{};
            std::vector<Spline2D> Theta_p_spline{};
            std::vector<Spline2D> Nu_spline{};

            // Internal methods
            double get_tight_coupling_time(const double k) const;

            DVector set_ic_after_tight_coupling(const DVector & y_tight_coupling, const double x, const double k) const;

            DVector set_ic(const double x, const double k) const;

            DVector set_all_perturbations_in_tight_coupling(const DVector & y_tight_coupling,
                                                            const double x,
                                                            const double k) const;

            int rhs_tight_coupling_ode(double x, double k, const double * y, double * dydx);
            int rhs_full_ode(double x, double k, const double * y, double * dydx);
            int rhs_jacobian_full(double x, double k, const double * y, double * dfdy, double * dfdt);

            // Steps computed in solve()
            void integrate_perturbations();
            void compute_source_functions();

            // For keeping timings
            mutable FML::UTILS::Timings timer;

          public:
            PerturbationSystemInfo psinfo;
            PerturbationSystemInfo psinfo_tight_coupling;

            Perturbations(){};
            Perturbations(std::shared_ptr<BackgroundCosmology> cosmo,
                          std::shared_ptr<RecombinationHistory> rec,
                          ParameterMap & p);
            Perturbations & operator=(const Perturbations & rhs) = default;
            Perturbations & operator=(Perturbations && other) = default;
            Perturbations(const Perturbations & rhs) = default;
            Perturbations(Perturbations && rhs) = default;
            ~Perturbations() = default;

            /// Do all the solving
            void solve();

            /// Show some info
            void info() const;

            /// Output perturbation quantities to file
            void output_perturbations(const double k, const std::string filename) const;
            /// Output transfer functions to file
            void output_transfer(const double x, const std::string filename) const;

            // Get the quantities we have integrated
            /// CDM density contrast in the newtonian gauge of x=log(a) and k
            double get_delta_cdm(const double x, const double k) const;
            /// Baryon density contrast in the newtonian gauge of x=log(a) and k
            double get_delta_b(const double x, const double k) const;
            /// CDM velocity in the newtonian gauge of x=log(a) and k
            double get_v_cdm(const double x, const double k) const;
            /// Baryon velocity in the newtonian gauge of x=log(a) and k
            double get_v_b(const double x, const double k) const;
            /// Space-space newtonian potential Phi in the newtonian gauge of x=log(a) and k
            double get_Phi(const double x, const double k) const;
            /// Time-time newtonian potential Psi in the newtonian gauge of x=log(a) and k
            double get_Psi(const double x, const double k) const;
            /// Photon aniotrop stress Pi in the newtonian gauge of x=log(a) and k
            double get_Pi(const double x, const double k) const;
            /// Photon perturbation Theta in the newtonian gauge of x=log(a) and k
            double get_Theta(const double x, const double k, const int ell) const;
            /// Photon polarisation perturbation Theta_p in the newtonian gauge of x=log(a) and k
            double get_Theta_p(const double x, const double k, const int ell) const;
            /// Neutrino perturbation Nu in the newtonian gauge of x=log(a) and k
            double get_Nu(const double x, const double k, const int ell) const;

            /// Gauge invariant matter density constrast
            double get_Delta_M(const double x, const double k) const;

            // Source functions
            /// Source function for LOS integrals for temperature of x=log(a) and k
            double get_Source_T(const double x, const double k) const;
            /// Source function for LOS integrals for E-mode polarisation of x=log(a) and k
            double get_Source_E(const double x, const double k) const;
            /// Source function for LOS integrals for CMB lensing of x=log(a) and k
            double get_Source_L(const double x, const double k) const;
            /// Source function for LOS integrals for massless neutrinos of x=log(a) and k
            double get_Source_N(const double x, const double k) const;

            // Individual contributions to temperature source
            /// Source function for the Sachs-Wolfe effect in the photon LOS integral of x=log(a) and k
            double get_Source_SW_T(const double x, const double k) const;
            /// Source function for the Integrated Sachs-Wolfe effect in the photon LOS integral of x=log(a) and k
            double get_Source_ISW_T(const double x, const double k) const;
            /// Source function for the doppler term in the photon LOS integral of x=log(a) and k
            double get_Source_VEL_T(const double x, const double k) const;
            /// Source function for the polarisation term in the photon LOS integral of x=log(a) and k
            double get_Source_POL_T(const double x, const double k) const;

            // Transfer functions
            double get_transfer_zeta(const double x, const double k) const;
            /// Transfer function for the gamma term needed for including relativistic terms in N-body of x = log(a) and
            /// k (same as CAMB gives)
            double get_transfer_gammaNbody(const double x, const double k) const;
            /// Transfer function for the CDM density contrast of x = log(a) and k (same as CAMB gives)
            double get_transfer_Delta_cdm(const double x, const double k) const;
            /// Transfer function for the baryon density contrast of x = log(a) and k (same as CAMB gives)
            double get_transfer_Delta_b(const double x, const double k) const;
            /// Transfer function for the baryon+CDM density contrast of x = log(a) and k (same as CAMB gives)
            double get_transfer_Delta_cb(const double x, const double k) const;
            /// Transfer function for the total matter density contrast of x = log(a) and k (same as CAMB gives)
            double get_transfer_Delta_M(const double x, const double k) const;
            /// Transfer function for the photon density contrast of x = log(a) and k (same as CAMB gives)
            double get_transfer_Delta_R(const double x, const double k) const;
            /// Transfer function for the neutrino density contrast of x = log(a) and k (same as CAMB gives)
            double get_transfer_Delta_Nu(const double x, const double k) const;
            /// Transfer function for the total relativistic density contrast of x = log(a) and k (same as CAMB gives)
            double get_transfer_Delta_Rtot(const double x, const double k) const;
            /// Transfer function for the CDM velocity of x = log(a) and k (same as CAMB gives)
            double get_transfer_v_cdm(const double x, const double k) const;
            /// Transfer function for the baryon velocity of x = log(a) and k (same as CAMB gives)
            double get_transfer_v_b(const double x, const double k) const;
            /// Transfer function for the photon velocity of x = log(a) and k (same as CAMB gives)
            double get_transfer_v_R(const double x, const double k) const;
            /// Transfer function for the massless neutrino velocity of x = log(a) and k (same as CAMB gives)
            double get_transfer_v_Nu(const double x, const double k) const;
            /// Transfer function for the baryon-CDM relative velocity of x = log(a) and k (same as CAMB gives)
            double get_transfer_v_b_v_c(const double x, const double k) const;
            /// Transfer function for the space-space metric potential Psi of x = log(a) and k (same as CAMB gives)
            double get_transfer_Phi(const double x, const double k) const;
            /// Transfer function for the time-time metric potential Phi of x = log(a) and k (same as CAMB gives)
            double get_transfer_Psi(const double x, const double k) const;
            /// Transfer function for the Weyl potential of x = log(a) and k (same as CAMB gives)
            double get_transfer_Weyl(const double x, const double k) const;

            double get_kmin() const;
            double get_kmax() const;
            double get_x_start() const;
            double get_x_end() const;

            void output_test(const double x, const std::string filename) const;
        };
    } // namespace COSMOLOGY
} // namespace FML

#endif
