#ifndef _BACKGROUNDCOSMOLOGY_HEADER
#define _BACKGROUNDCOSMOLOGY_HEADER
#include <algorithm>
#include <chrono>
#include <fstream>
#include <functional>
#include <utility>

#include <FML/Math/Math.h>
#include <FML/ODESolver/ODESolver.h>
#include <FML/ParameterMap/ParameterMap.h>
#include <FML/Spline/Spline.h>
#include <FML/Units/Units.h>

namespace FML {

    /// This namespace deals with various cosmology specific things: background evolution, perturbation theory,
    /// recombination history etc.
    namespace COSMOLOGY {

        // Accuracy settings for ODE solver for the eta ODE
#define FIDUCIAL_COSMO_HSTART_ODE_ETA 1e-3
#define FIDUCIAL_COSMO_ABSERR_ODE_ETA 1e-10
#define FIDUCIAL_COSMO_RELERR_ODE_ETA 1e-10

        // Number of points and x-range for the splines we make
#define FIDUCIAL_COSMO_NPTS_SPLINES 10000
#define FIDUCIAL_COSMO_X_START (std::log(1e-15))
#define FIDUCIAL_COSMO_X_END (std::log(100.0))

        // Type aliases
        using ParameterMap = FML::UTILS::ParameterMap;
        using ODEFunction = FML::SOLVERS::ODESOLVER::ODEFunction;
        using ODESolver = FML::SOLVERS::ODESOLVER::ODESolver;
        using Spline = FML::INTERPOLATION::SPLINE::Spline;
        using DVector = std::vector<double>;

        // Global units
        extern FML::UTILS::ConstantsAndUnits Constants;

        /// Computing the background evolution of our Universe (LCDM). Holds various functions related to the
        /// background: Hubble, distances, growth functions etc.
        class BackgroundCosmology {
          private:
            // The Hubble parameter today in units of 1/sec
            double h{0.7};
            double H0{};

            // Density parameters
            double OmegaB{0.05};
            double OmegaCDM{0.25};
            double OmegaLambda{0.7};
            double OmegaR{};
            double OmegaNu{};
            double OmegaRtot{};
            double OmegaM{};
            double OmegaK{0.0};

            // Other parameters
            double Neff{3.046};
            double TCMB{2.7255};

            std::string name{"FiducialCosmology"};

            // The conformal time today - set after we solve for eta
            double eta0{};

            // Curvature K = -OmegaK0 H0^2/c^2
            double K{};

            // Splines
            Spline H_spline{"H_spline"};
            Spline Hp_spline{"Hp_spline"};
            Spline dHdx_spline{"dHdx_spline"};
            Spline dHpdx_spline{"dHpdx_spline"};
            Spline w_spline{"w_spline"};
            Spline eta_of_x_spline{"eta_of_x_spline"};
            Spline cosmic_time_of_x_spline{"cosmic_time_of_x_spline"};
            Spline D1_spline{"D1_LPT_spline"};
            Spline D2_spline{"D2_LPT_spline"};
            Spline chi_of_x_spline{"chi_of_x_spline"};
            Spline x_of_chi_spline{"x_of_chi_spline"};

            // Parameters defining the range and number of points
            int n_pts_splines{FIDUCIAL_COSMO_NPTS_SPLINES};
            double x_min_background{FIDUCIAL_COSMO_X_START};
            double x_max_background{FIDUCIAL_COSMO_X_END};

            // Internal solve methods
            void compute_growth_factors();
            void compute_conformal_time();
            void compute_background();

          public:
            BackgroundCosmology(){};
            BackgroundCosmology(const ParameterMap & p);
            BackgroundCosmology & operator=(BackgroundCosmology && other) = default;
            BackgroundCosmology & operator=(const BackgroundCosmology & rhs) = default;
            BackgroundCosmology(const BackgroundCosmology & rhs) = default;
            BackgroundCosmology(BackgroundCosmology && rhs) = default;
            ~BackgroundCosmology() = default;

            // Solve everything
            void solve();
            void info() const;
            void output(const std::string filename) const;

            // Distance measures
            double dL_of_x(double x) const;
            double dA_of_x(double x) const;
            double r_of_x(double x) const;
            double r_of_chi(double x) const;
            double chi_of_x(double x) const;
            double x_of_chi(double x) const;
            double eta_of_x(double x) const;
            double detadx_of_x(double x) const;

            // Hubble factors
            double H_of_x(double x) const;
            double dHdx_of_x(double x) const;
            double ddHddx_of_x(double x) const;
            double Hp_of_x(double x) const;
            double dHpdx_of_x(double x) const;
            double ddHpddx_of_x(double x) const;

            // Density parameters
            double get_OmegaB(double x = 0.0) const;
            double get_OmegaM(double x = 0.0) const;
            double get_OmegaR(double x = 0.0) const;
            double get_OmegaRtot(double x = 0.0) const;
            double get_OmegaNu(double x = 0.0) const;
            double get_OmegaCDM(double x = 0.0) const;
            double get_OmegaLambda(double x = 0.0) const;
            double get_OmegaK(double x = 0.0) const;
            double get_OmegaMnu(double x = 0.0) const;
            double get_weff(double x) const;
            double get_dweffdx(double x) const;

            // LPT growth-factors normalized to unity today
            double get_D1_LPT(double x) const;
            double get_D2_LPT(double x) const;
            double get_dD1dx_LPT(double x) const;
            double get_dD2dx_LPT(double x) const;

            // Other stuff
            double get_cosmic_time(double x = 0.0) const;
            double get_TCMB(double x = 0.0) const;
            double get_Tnu(double x = 0.0) const;
            double get_H0() const;
            double get_h() const;
            double get_K() const;
            double get_Neff() const;
            std::string get_name() const;
        };
    } // namespace COSMOLOGY
} // namespace FML

#endif
