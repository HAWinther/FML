#ifndef POWERSPECTRUM_HEADER
#define POWERSPECTRUM_HEADER
#ifdef USE_OMP
#include <omp.h>
#endif
#include <algorithm>
#include <chrono>
#include <fstream>
#include <functional>
#include <numeric>
#include <utility>
#ifdef USE_FFTW
#include <fftw3.h>
#endif

#include <FML/Cosmology/BackgroundCosmology/BackgroundCosmology.h>
#include <FML/Cosmology/LinearPerturbations/Perturbations.h>
#include <FML/Cosmology/RecombinationHistory/RecombinationHistory.h>
#include <FML/Math/Math.h>
#include <FML/ODESolver/ODESolver.h>
#include <FML/Spline/Spline.h>
#include <FML/Timing/Timings.h>
#include <FML/Global/Global.h> // We only need ThisTask
#ifdef USE_FFTW
#include <FML/FFTLog/FFTLog.h>
#endif

namespace FML {
    namespace COSMOLOGY {

        // Accuracy settings for the LOS integral
#define FIDUCIAL_HSTART_ODE_LOS 1e-3
#define FIDUCIAL_ABSERR_ODE_LOS 1e-6
#define FIDUCIAL_RELERR_ODE_LOS 1e-6

        // Accuracy settings for Cell integral
#define FIDUCIAL_HSTART_ODE_CELL 1e-6
#define FIDUCIAL_ABSERR_ODE_CELL 1e-12
#define FIDUCIAL_RELERR_ODE_CELL 1e-12

        using BackgroundCosmology = FML::COSMOLOGY::BackgroundCosmology;
        using RecombinationHistory = FML::COSMOLOGY::RecombinationHistory;
        using Perturbations = FML::COSMOLOGY::Perturbations;
        using ODESolver = FML::SOLVERS::ODESOLVER::ODESolver;
        using ODEFunction = FML::SOLVERS::ODESOLVER::ODEFunction;
        using ODEFunctionJacobian = FML::SOLVERS::ODESOLVER::ODEFunctionJacobian;
        using Spline = FML::INTERPOLATION::SPLINE::Spline;
        using Spline2D = FML::INTERPOLATION::SPLINE::Spline2D;
        using DVector = std::vector<double>;
        using DVector2D = std::vector<DVector>;

        /// Computing power-spectra (matter, CMB, correlation functions etc.) in linear perturbation theory
        class PowerSpectrum {
          private:
            // The cosmology to use
            std::shared_ptr<const BackgroundCosmology> cosmo{nullptr};

            // The recombination history to use
            std::shared_ptr<const RecombinationHistory> rec{nullptr};

            // The perturbations data to use
            std::shared_ptr<const Perturbations> pert{nullptr};

            // Parameters defining the primordial power-spectrum
            double A_s{2e-9};
            double n_s{0.96};
            double kpivot{0.05 / Constants.Mpc};

            // These are gotten from the Perturbations object
            double kmin{};
            double kmax{};

            // The time where we compute Theta_ell(k), Cell(k)
            double x_cell{0.0};

            // This is eta and e^tau at the time x_cell we are to compute the Cell's at
            // Gotten from the cosmology object and kept here to ensure we use it consistently
            double eta0{};
            double exptau0{};

            // Accuracy settings
            int bessel_nsamples_per_osc{16};
            int los_integration_nsamples_per_osc{8};
            int los_integration_loga_nsamples{300};
            int cell_nsamples_per_osc{16};

            // The ells's we compute Theta_ell and Cell for
            // We will shrink this to ell_max
            int ell_max{2000};
            DVector ells{};

            // What Cells to compute (if data is availiable)
            bool compute_temperature_cells{true};
            bool compute_polarization_cells{false};
            bool compute_neutrino_cells{false};
            bool compute_lensing_cells{false};
            bool compute_corr_function{false};

            // Splines of bessel-functions and theta_ell(k)
            std::vector<Spline> j_ell_splines{};
            Spline2D thetaT_ell_of_k_spline{"thetaT_ell_of_k_spline"};
            Spline2D thetaE_ell_of_k_spline{"thetaE_ell_of_k_spline"};
            Spline2D lens_ell_of_k_spline{"lens_ell_of_k_spline"};
            Spline2D Nu_ell_of_k_spline{"Nu_ell_of_k_spline"};

            // Splines with the power-spectra
            Spline cell_TT_spline{"cell_TT_spline"};
            Spline cell_TE_spline{"cell_TE_spline"};
            Spline cell_EE_spline{"cell_EE_spline"};
            Spline cell_LL_spline{"cell_LL_spline"};
            Spline cell_NN_spline{"cell_NN_spline"};
            Spline cell_TL_spline{"cell_TL_spline"};

            Spline index_of_ells_spline{"index_of_ells_spline"};

            // Splines of the correlation functions
            Spline2D xi_CDM_spline{"xi_CDM_spline"};
            Spline2D xi_B_spline{"xi_B_spline"};
            Spline2D xi_R_spline{"xi_R_spline"};
            Spline2D xi_Nu_spline{"xi_Nu_spline"};
            Spline2D xi_M_spline{"xi_M_spline"};
            
            // For keeping timings
            mutable FML::UTILS::Timings timer;

          public:
            PowerSpectrum() = delete;
            PowerSpectrum(std::shared_ptr<BackgroundCosmology> cosmo,
                          std::shared_ptr<RecombinationHistory> rec,
                          std::shared_ptr<Perturbations> pert,
                          ParameterMap & p);
            PowerSpectrum & operator=(const PowerSpectrum & rhs) = default;
            PowerSpectrum & operator=(PowerSpectrum && other) = default;
            PowerSpectrum(const PowerSpectrum & rhs) = default;
            PowerSpectrum(PowerSpectrum && rhs) = default;
            ~PowerSpectrum() = default;

            /// Do all the solving
            void solve();
            /// Show some info
            void info() const;

            /// Make bessel-function splines that we need
            void generate_bessel_function_splines(double xmax, int nsamples_per_osc);

            /// Do all the LOS integrals we need
            void line_of_sight_integration(DVector & k_array);

            /// Compute LOS integral to get F_ell for any given source function
            DVector2D line_of_sight_integration_single(DVector & x_array,
                                                       DVector & k_array,
                                                       std::function<double(double, double)> & source_function,
                                                       std::function<double(double, double)> & aux_norm);

            /// Compute Cell for any given quantity
            DVector solve_for_cell_single(DVector & log_k_array,
                                          std::function<double(double, int)> & integrand,
                                          double accuracy_limit);

            /// Compute and spline all xi(r,x) - the fourier transforms of the power-spectra
            void compute_all_correlation_functions(double xmin = -8.0, double xmax = 0.0, int nx = 50);

            /// The primordial power-spectrum P(k)
            double primordial_power_spectrum(double k) const;

            /// The dimensionless primordial power-spectrum Delta = 2pi^2/k^3 P(k)
            double primordial_power_spectrum_dimless(double k) const;

            // Get the various power spectra: standard format l(l+1)/2pi Cell in muK^2 for photons
            /// l(l+1)/2pi Cell in muK^2 for photon temperature
            double get_cell_TT(double ell) const;
            /// l(l+1)/2pi Cell in muK^2 for photon temperature cross E-mode
            double get_cell_TE(double ell) const;
            /// l(l+1)/2pi Cell in muK^2 for photon E-mode
            double get_cell_EE(double ell) const;
            /// l(l+1)/2pi Cell in muK^2 for lensing potential
            double get_cell_LL(double ell) const;
            /// l(l+1)/2pi Cell in muK^2 for neutrinos
            double get_cell_NN(double ell) const;

            // Get the various correlation functions
            double get_corr_func(double x, double r, std::string type) const;
            /// Total matter correlation function of x = log(a) and r
            double get_corr_func_M(double x, double r) const;
            /// CDM correlation function of x = log(a) and r
            double get_corr_func_CDM(double x, double r) const;
            /// Baryon correlation function of x = log(a) and r
            double get_corr_func_B(double x, double r) const;
            /// Baryon+CDM correlation function of x = log(a) and r
            double get_corr_func_CB(double x, double r) const;
            /// Photon correlation function of x = log(a) and r
            double get_corr_func_R(double x, double r) const;
            /// Masseless neutrino correlation function of x = log(a) and r
            double get_corr_func_Nu(double x, double r) const;
            /// Total relativistic correlation function of x = log(a) and r
            double get_corr_func_Rtot(double x, double r) const;

            /// Get P(k,x) for a given x = log(a) and k. The type is:
            /// B (baryons), CDM, CB (CDM+baryons), R (photons), Nu (neutrionos), Rtot (total radiation)
            /// or M (total matter). The latter is the same as get_matter_power_spectrum
            double get_power_spectrum(double x, double k, std::string type) const;
            
            /// Get total matter power-spectrum of x = log(a) and k
            double get_matter_power_spectrum(double x, double k) const;
            std::pair<DVector, DVector> get_power_spectrum_array(double x, int npts, std::string type) const;

            /// Get spherical bessel-function from the splines we made
            double get_j_ell(int ell, double x) { return j_ell_splines[int(index_of_ells_spline(ell))](x); }

            /// Lensing potential source function
            double lensing_source(double x, double x_observer = 0.0) const;

            /// Outputs (k, P(k)) in units of h/Mpc and (Mpc/h)^3
            void output_matter_power_spectrum(double x, std::string filename) const;

            /// Outputs (r, xi(r)) in units of Mpc/h and 1
            void output_correlation_function(double x, std::string filename) const;

            /// Output Cl (TT,EE,TE) in units of l(l+1)/2pi (muK)^2
            void output_angular_power_spectra(std::string filename) const;

            /// Output the LOS integral quantities we have computed
            void output_theta_ell(std::string filename) const;

            double los_photons(int i, double k, std::function<double(double, double)> & source_function);
        };

        // Methods for computing correlation functions
#ifdef USE_FFTW
        std::pair<DVector, DVector> correlation_function_single_fftw(std::function<double(double)> & Delta_P,
                                                                     double rmin = 1.0 * Constants.Mpc,
                                                                     double rmax = 512.0 * Constants.Mpc,
                                                                     int ngrid_min = 8192);

        std::pair<DVector, DVector> correlation_function_single_fftlog(std::function<double(double)> & Delta_P,
                                                                       double rmin = 1e-3 * Constants.Mpc,
                                                                       double rmax = 1e3 * Constants.Mpc,
                                                                       int ngrid = 8192);
#endif
        double correlation_function_single(double r, std::function<double(double)> & Delta_P, double kmin, double kmax);

    } // namespace COSMOLOGY
} // namespace FML

#endif
