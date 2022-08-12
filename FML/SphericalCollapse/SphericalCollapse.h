#ifndef SPHERICALCOLLAPSE_INCLUDE
#define SPHERICALCOLLAPSE_INCLUDE

#include <FML/ODESolver/ODESolver.h>
#include <FML/Spline/Spline.h>
#include <FML/Math/Math.h>
#include <FML/Timing/Timings.h>
#include <FML/SphericalCollapse/SphericalCollapseModel.h>
#include <cmath>
#include <tuple>
#include <iostream>
#include <fstream>

namespace FML {

  namespace COSMOLOGY {

    namespace SPHERICALCOLLAPSE {

      using Timer = FML::UTILS::Timings;
      using DVector = FML::SOLVERS::ODESOLVER::DVector;
      using DVector2D = FML::SOLVERS::ODESOLVER::DVector2D;
      using ODEFunction = FML::SOLVERS::ODESOLVER::ODEFunction;
      using ODEFunctionJacobian = FML::SOLVERS::ODESOLVER::ODEFunctionJacobian;
      using ODESolver = FML::SOLVERS::ODESOLVER::ODESolver;
      using Spline = FML::INTERPOLATION::SPLINE::Spline;
        
      //===================================================================================
      /// @brief Computes the growth factor (not normalized) and growth rate for a given
      /// model and returns splines with \f$D(x)\f$ and \f$f(x)\f$ where \f$x=\log(a)\f$
      ///
      /// @param[in] model The spherical-collapse model
      /// @param[in, out] growthfactor_spline Spline of the growthfactor \f$D(\log a)\f$
      /// @param[in, out] growthrate_spline Spline of the growth-rate \f$d\log D(\log a)/d\log a\f$
      /// @param[in] xini The time \f$x=\log(a)\f$ we start the integration. Fiducial \f$a=10^{-5}\f$.
      /// @param[in] xend The time we end the integration.
      /// @param[in] npts Number of points (linear between xini and xend) to compute it on.
      ///
      //===================================================================================
      void compute_growthfactor(
          const SphericalCollapseModel & model,
          Spline & growthfactor_spline,
          Spline & growthrate_spline,
          const double xini = std::log(1e-5),
          const double xend = 0.0,
          const int npts = 1000);

      /// This is the dimensionless quantity \f$r(x)/R\f$ (the radius of the tophat) for spherical collapse evolution, 
      double roverR_of_delta_and_x(
          double delta, 
          double x, 
          double RH0 = 1.0, 
          double delta_ini = 0.0);

      /// Evolve the spherical collapse equation + linear
      /// for a given set of IC delta_ini over a given range
      /// [xini,xcollapse] and return the arrays + the x-value
      /// where delta(x) > delta_infinity (i.e. it has collapsed
      /// completely). If it haven't collapsed return xcollapse
      /// This value can be used to more easily figure out delta_ini
      /// that corresponds to collapse at exactly xcollapse via
      /// root finding
      double evolve_spherical_collapse_equations(
          const SphericalCollapseModel & model,
          const double delta_ini, 
          DVector & x_array,
          DVector & delta_array,
          DVector & delta_prime_array,
          DVector & delta_lin_array,
          DVector & delta_lin_prime_array,
          const bool get_arrays = true,
          const double xini = std::log(1e-3), 
          const double xcollapse = std::log(1.0),
          const int npts = 1000,
          const double delta_infinity = 1e16);

      /// Do root finding to figure out the delta_ini that gives collapse at x=xcollapse
      double find_delta_ini(
          const SphericalCollapseModel & model,
          const double xini = std::log(1e-3), 
          const double xcollapse = std::log(1.0),
          const int npts = 1000,
          const double epsilon_delta = 1e-8);

      /// Extract useful times from spherical collapse evolution
      /// like turnaround, virialization and non-linear onset
      auto compute_times(
          const double xini,
          const double xcollapse,
          const SphericalCollapseModel & model,
          const Spline & delta_spline, 
          const Spline & delta_prime_spline) -> std::tuple<double, double, double, double>;

      /// Do spherical collapse for all times
      void compute_sphericalcollapse_splines(
          const SphericalCollapseModel & model,
          Spline & deltac_of_x_spline,
          Spline & DeltaVir_of_x_spline,
          Spline & xnl_of_x_spline,
          Spline & xta_of_x_spline,
          Spline & xvir_of_x_spline,
          Spline & delta_ini_of_x_spline,
          const double xini = std::log(1e-3),
          const int npts = 1000,
          bool verbose = false);

      //===================================================================================
      /// Class for doing general spherical collapse calculations, i.e. evolve
      /// the collapse and then turn the initial conditions such that collapse happens
      /// at a given redshift. Then extract quantities like non-linear onset, turnaround,
      /// virialization and \f$\delta_c\f$ and virial overdennsity \f$\Delta_{\rm vir}\f$
      //===================================================================================
      class SphericalCollapse {
        private:

          SphericalCollapseModel spc;

          //===========================================
          // Accuracy
          //===========================================
          bool verbose{false};
          int npts_spherical_collapse{1000};
          double xini_spherical_collapse{std::log(1e-5)};

        public:
          //===========================================
          // Construction
          //===========================================
          SphericalCollapse(
              SphericalCollapseModel spc,
              bool verbose = false,
              int npts_spherical_collapse = 1000,
              double xini_spherical_collapse = std::log(1e-5)) : 
            spc(spc),
            verbose(verbose),
            npts_spherical_collapse(npts_spherical_collapse),
            xini_spherical_collapse(xini_spherical_collapse){}

          //===========================================
          /// @brief Do the spherical collapse at a single redshift and get \f$\delta(x)\f$, \f$\delta_{\rm lin}(x)\f$ together
          /// with turnaround time, virialization time, non-linear time, initial overdensity,
          /// critial overdensity \f$\delta_c\f$ and virial overdensity. The times are \f$x=\log a\f$.
          //===========================================
          void run_single_redshift(
              double z,
              Spline & delta_of_x_spline,
              Spline & delta_prime_of_x_spline,
              Spline & delta_lin_of_x_spline,
              double & deltac,
              double & DeltaVir,
              double & xnl,
              double & xta,
              double & xvir,
              double & delta_ini,
              bool use_provided_delta_ini){

            // Time we want the collapse to happen
            const double xcollapse = std::log(1.0/(1.0+z));

            // Find initial density. If use_provided_delta_ini then we evolve the equations
            // with the provided IC (but then deltac, DeltaVir etc do not really make sense)
            if(not use_provided_delta_ini)
              delta_ini = find_delta_ini(spc, xini_spherical_collapse, xcollapse, npts_spherical_collapse);

            // Perform spherical collapse calculation
            DVector x_array, delta_array, delta_prime_array, delta_lin_array, delta_lin_prime_array;
            evolve_spherical_collapse_equations(
                spc, 
                delta_ini,
                x_array,
                delta_array, 
                delta_prime_array, 
                delta_lin_array, 
                delta_lin_prime_array, 
                true,
                xini_spherical_collapse, 
                xcollapse, 
                npts_spherical_collapse);

            // Make splines
            delta_of_x_spline = Spline(x_array, delta_array, "delta(loga)");
            delta_prime_of_x_spline = Spline(x_array, delta_prime_array, "ddelta/dloga(loga)");
            delta_lin_of_x_spline = Spline(x_array, delta_lin_array, "deltaLin(loga)");

            // Compute quantities
            const auto times = compute_times(xini_spherical_collapse, xcollapse, spc, delta_of_x_spline, delta_prime_of_x_spline);
            xta  = std::get<0>(times);
            xvir = std::get<1>(times);
            xnl  = std::get<2>(times);
            DeltaVir = (1.0+delta_of_x_spline(xvir))*std::exp(3.0*xcollapse-3.0*xvir);
            deltac = delta_lin_of_x_spline(xcollapse);
          }

          //===========================================
          /// @brief Perform the spherical collapse calculation for a given model with collapse at 
          /// \f$x\f$ for several values of \f$x\f$ and fill resulting splines. 
          ///
          /// @param[in, out] deltac_of_x_spline The critical linear extrapolated overdensity as function of \f$x=\log a\f$.
          /// @param[in, out] DeltaVir_of_x_spline The virial overdensity as function of \f$x=\log a\f$.
          /// @param[in, out] xta_of_x_spline The turnaround radius as function of \f$x=\log a\f$.
          /// @param[in, out] xnl_of_x_spline The non-linear time as function of \f$x=\log a\f$.
          /// @param[in, out] xvir_of_x_spline The virialization time as function of \f$x=\log a\f$.
          /// @param[in, out] delta_ini_of_x_spline The initial density contrast as function of \f$x=\log a\f$.
          /// @param[in, out] growthfactor_of_x_spline The growthfactor \f$D\f$ as function of \f$x=\log a\f$.
          /// @param[in, out] growthrate_of_x_spline The growthrate \f$d\log D/d\log a\f$ as function of \f$x=\log a\f$.
          //===========================================
          void run_at_all_redshifts(
              Spline & deltac_of_x_spline,
              Spline & DeltaVir_of_x_spline,
              Spline & xta_of_x_spline,
              Spline & xnl_of_x_spline,
              Spline & xvir_of_x_spline,
              Spline & delta_ini_of_x_spline,
              Spline & growthfactor_of_x_spline,
              Spline & growthrate_of_x_spline){
            compute_sphericalcollapse_splines(
                spc, 
                deltac_of_x_spline, 
                DeltaVir_of_x_spline, 
                xnl_of_x_spline,
                xta_of_x_spline,
                xvir_of_x_spline,
                delta_ini_of_x_spline,
                xini_spherical_collapse, 
                npts_spherical_collapse,
                verbose);

            compute_growthfactor(
                spc,
                growthfactor_of_x_spline,
                growthrate_of_x_spline);
          }
      };

    }
  }
}

#endif
