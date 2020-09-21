#ifndef RECOMBINATION_HISTORY_HEADER
#define RECOMBINATION_HISTORY_HEADER
#include <chrono>
#include <cmath>
#include <fstream>
#include <functional>
#include <iostream>
#include <memory>
#include <utility>
#include <vector>
#ifdef USE_OMP
#include <omp.h>
#endif

#include <FML/Cosmology/BackgroundCosmology/BackgroundCosmology.h>
#include <FML/Math/Math.h>
#include <FML/ODESolver/ODESolver.h>
#include <FML/Spline/Spline.h>
#ifdef USE_RECFAST
#include "Recfast++.h"
#endif

namespace FML {
    namespace COSMOLOGY {

        // Accuracy settings for ODE solver for the tau ODE
#define FIDUCIAL_HSTART_ODE_TAU 1e-6
#define FIDUCIAL_ABSERR_ODE_TAU 1e-12
#define FIDUCIAL_RELERR_ODE_TAU 1e-12

        // Accuracy settings for ODE solver for Peebles ODE for Xe
#define FIDUCIAL_HSTART_ODE_PEEBLES 1e-6
#define FIDUCIAL_ABSERR_ODE_PEEBLES 1e-12
#define FIDUCIAL_RELERR_ODE_PEEBLES 1e-12

        // Type aliases
        using BackgroundCosmology = FML::COSMOLOGY::BackgroundCosmology;
        using ParameterMap = FML::UTILS::ParameterMap;
        using ODEFunction = FML::SOLVERS::ODESOLVER::ODEFunction;
        using ODESolver = FML::SOLVERS::ODESOLVER::ODESolver;
        using Spline = FML::INTERPOLATION::SPLINE::Spline;
        using DVector = std::vector<double>;

        /// For computing the recombination history of our Universe. Optical depth, free electron fraction, baryon
        /// temperature, reionization, etc.
        class RecombinationHistory {
          private:
            // The cosmology to use
            std::shared_ptr<const BackgroundCosmology> cosmo{nullptr};

            // Helium fraction
            double Yp{0.24};

            // Use recfast for Xe, Tb instead?
            bool userecfast{false};

            // This is the same factor as used in Recfast
            double rec_fudge_factor{1.14};

            // Reinonization parameters
            bool reionization{true};
            double z_reion{11.0};
            double delta_z_reion{0.5};
            bool helium_reionization{true};
            double z_helium_reion{3.5};
            double delta_z_helium_reion{0.5};

            // The start and end points for recombination arrays
            double x_start_rec_array{-25.0};
            double x_end_rec_array{std::log(1.0)};

            // Numbers of points of Xe,ne array
            int npts_Xe_array{1000};

            // Number of points for tau and visibility functions arrays
            int npts_tau_before_reion{1000};
            int npts_tau_during_reion{100};
            int npts_tau_after_reion{100};

            // X_e for when Saha equation needs to be replaced by the Peebles equation
            // Putting this to 0 means always using Saha
            double Xe_saha_limit{0.99};

            // The time when Xe = 0.5. Computed after we solve for Xe
            double x_recombination{};
            // The time when Xe = 0.5 with Saha. Computed after we solve for Xe
            double x_recombination_saha{};
            // The time when tau_noreion = 1.0. Computed after we solve for tau
            double x_star{};
            // The time when tau_saha_noreion = 1.0. Computed after we solve for tau
            double x_star_saha{};
            // The time when the visibility function peaks: g'=0. Computed after we solve
            // for tau
            double x_star2{};
            // The time when tau_baryon = 1.0. Computed after we solve for tau
            double x_drag{};

            // Splines contained in this class
            Spline Xe_of_x_spline{"Xe_of_x_spline"};
            Spline tau_of_x_spline{"tau_of_x_spline"};
            Spline g_tilde_of_x_spline{"g_tilde_of_x_spline"};
            Spline tau_of_x_noreion_spline{"tau_of_x_noreion_spline"};
            Spline g_tilde_of_x_noreion_spline{"g_tilde_of_x_noreion_spline"};
            Spline tau_baryon_noreion_of_x_spline{"tau_baryon_noreion_of_x_spline"};
            Spline sound_horizon_of_x_spline{"sound_horizon_of_x_spline"};
            Spline Tb_spline{"Temp_baryon_of_x_spline"};
            Spline cs2_baryon_spline{"cs2_baryon_of_x_spline"};
            Spline kd_of_x_spline{"kD damping scale"};

            // Just Saha splines
            Spline Xe_of_x_saha_spline{"Xe_of_x_saha_spline"};
            Spline tau_of_x_saha_spline{"tau_of_x_saha_spline"};
            Spline tau_of_x_saha_noreion_spline{"tau_of_x_saha_noreion_spline"};

            // Splines of derivatives
            Spline dgdx_tilde_of_x_spline{"dgdx_tilde_of_x_spline"};
            Spline ddgddx_tilde_of_x_spline{"ddgddx_tilde_of_x_spline"};
            Spline dtaudx_of_x_spline{"dtaudx_of_x_spline"};

            // Internal methods
            std::pair<double, double> electron_fraction_from_saha_equation_with_helium(double x) const;
            std::pair<double, double> electron_fraction_from_saha_equation_without_helium(double x) const;
            int rhs_peebles_ode(double x, const double * y, double * dydx);
            double Xe_reionization_factor_of_x(double x) const;

            // The steps in solve
            void solve_number_density_electrons();
            void solve_for_optical_depth_tau();
            void solve_extra();

          public:
            RecombinationHistory(){};
            RecombinationHistory(std::shared_ptr<BackgroundCosmology> cosmo, ParameterMap & p);
            RecombinationHistory & operator=(const RecombinationHistory & rhs) = default;
            RecombinationHistory & operator=(RecombinationHistory && other) = default;
            RecombinationHistory(const RecombinationHistory & rhs) = default;
            RecombinationHistory(RecombinationHistory && rhs) = default;
            ~RecombinationHistory() = default;

            void solve();
            void info() const;
            void output(const std::string filename) const;

            double tau_of_x(double x) const;
            double tau_of_x_saha(double x) const;
            double tau_of_x_saha_noreion(double x) const;
            double dtaudx_of_x(double x) const;
            double ddtauddx_of_x(double x) const;
            double g_tilde_of_x(double x) const;
            double dgdx_tilde_of_x(double x) const;
            double ddgddx_tilde_of_x(double x) const;
            double Xe_of_x(double x) const;
            double Xe_of_x_saha(double x) const;
            double ne_of_x(double x) const;
            double ne_of_x_saha(double x) const;
            double Xe_of_x_noreion(double x) const;
            double Xe_of_x_saha_noreion(double x) const;
            double ne_of_x_noreion(double x) const;
            double ne_of_x_saha_noreion(double x) const;

            double get_Tbaryon(double x) const;
            double get_baryon_sound_speed_squared(double x) const;
            double get_sound_speed_squared(double x) const;
            double get_sound_horizon(double x) const;
            double get_xstar() const;
            double get_xdrag() const;
            double get_Yp() const;
            double get_x_start_rec_array() const;

#ifdef USE_RECFAST
            std::pair<Vector, Vector> run_recfast(Vector & x_array);
#endif
        };
    } // namespace COSMOLOGY
} // namespace FML

#endif
