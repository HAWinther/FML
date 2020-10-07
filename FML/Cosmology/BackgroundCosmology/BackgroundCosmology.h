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
#include <FML/Global/Global.h> // Just for ThisTask

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

        // Global units (SI by default)
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

            /// Solve everything for the background
            void solve();
            /// Show some info
            void info() const;
            /// Output some data related to the background
            void output(const std::string filename) const;

            // Distance measures
            /// Luminoisity distance of x = log(a)
            double dL_of_x(double x) const;
            /// Angular diameterdistance of x = log(a)
            double dA_of_x(double x) const;
            double r_of_x(double x) const;
            double r_of_chi(double x) const;
            /// Comoving distance of x = log(a)
            double chi_of_x(double x) const;
            /// Inverse function of comoving distance of x = log(a)
            double x_of_chi(double x) const;
            /// Conformal time of x = log(a)
            double eta_of_x(double x) const;
            /// Derivative of conformal time of x = log(a)
            double detadx_of_x(double x) const;

            // Hubble factors
            /// Hubble function of of x = log(a)
            double H_of_x(double x) const;
            /// Derivative of Hubble function of x = log(a)
            double dHdx_of_x(double x) const;
            /// Second derivative of Hubble function of x = log(a)
            double ddHddx_of_x(double x) const;
            /// Conformal Hubble function Hp = aH of x = log(a)
            double Hp_of_x(double x) const;
            /// Derivative of conformal Hubble function Hp = aH of x = log(a)
            double dHpdx_of_x(double x) const;
            /// Second derivative of conformal Hubble function Hp = aH of x = log(a)
            double ddHpddx_of_x(double x) const;

            // Density parameters
            /// Baryon density parameter of x = log(a)
            double get_OmegaB(double x = 0.0) const;
            /// Total matter density parameter of x = log(a)
            double get_OmegaM(double x = 0.0) const;
            /// Radiation density parameter of x = log(a)
            double get_OmegaR(double x = 0.0) const;
            /// Total relativistic  density parameter of x = log(a)
            double get_OmegaRtot(double x = 0.0) const;
            /// Massless neutrino density parameter of x = log(a)
            double get_OmegaNu(double x = 0.0) const;
            /// CDM density parameter of x = log(a)
            double get_OmegaCDM(double x = 0.0) const;
            /// Dark energy density parameter of x = log(a)
            double get_OmegaLambda(double x = 0.0) const;
            /// Curvature density parameter of x = log(a)
            double get_OmegaK(double x = 0.0) const;
            /// Massive neutrino density parameter of x = log(a)
            double get_OmegaMnu(double x = 0.0) const;
            
            /// Effective equatin of state of x = log(a)
            double get_weff(double x) const;
            /// Derivative of effective equatin of state of x = log(a)
            double get_dweffdx(double x) const;

            // LPT growth-factors normalized to unity today
            /// 1LPT growth factor of x = log(a) with D=1 today
            double get_D1_LPT(double x) const;
            /// 2LPT growth factor of x = log(a)
            double get_D2_LPT(double x) const;
            /// Derivative of 1LPT growth factor of x = log(a)
            double get_dD1dx_LPT(double x) const;
            /// Derivative of 2LPT growth factor of x = log(a)
            double get_dD2dx_LPT(double x) const;

            // Other stuff
            /// Cosmic time t of x = log(a)
            double get_cosmic_time(double x = 0.0) const;
            /// Temperature of the CMB of x = log(a)
            double get_TCMB(double x = 0.0) const;
            /// Temperature of the neutrinos of x = log(a)
            double get_Tnu(double x = 0.0) const;
            /// Hubble parameter today
            double get_H0() const;
            /// Little h
            double get_h() const;
            /// Curvature K
            double get_K() const;
            /// Effective number of relativistic degrees of freedom
            double get_Neff() const;
            /// Name of the cosmology
            std::string get_name() const;
        };
    } // namespace COSMOLOGY
} // namespace FML

#endif
