#include "SphericalCollapse.h"

namespace FML {

  namespace COSMOLOGY {

    namespace SPHERICALCOLLAPSE {

      //================================================
      // Computes the growth factor (not normalized) and growth rate
      //================================================
      void compute_growthfactor(
          const SphericalCollapseModel & model,
          Spline & growthfactor_of_x_spline,
          Spline & growthrate_of_x_spline,
          const double xini,
          const double xend,
          const int npts){

        auto x_array = FML::MATH::linspace(xini, xend, npts);
        ODEFunction deriv = [&model](double x, const double * y, double * dydx) {
          return model.ode_system_linear_growth(x, y, dydx);
        };

        // EdS D ~ a so D'=D
        DVector y_initial{std::exp(xini), std::exp(xini)};
        ODESolver ode;
        ode.solve(deriv, x_array, y_initial);
        const auto growthfactor_array = ode.get_data_by_component(0);
        const auto delta_prime_lin_array = ode.get_data_by_component(1);
        auto growthrate_array = delta_prime_lin_array;
        for(size_t i = 0; i < growthfactor_array.size(); i++){
          growthrate_array[i] /= growthfactor_array[i];
        }
        growthfactor_of_x_spline = Spline(x_array, growthfactor_array, "Growthfactor D(loga)"); 
        growthrate_of_x_spline = Spline(x_array, growthrate_array, "Growthrate f(loga)"); 
      }

      //================================================
      // This is the dimensionless quantity r(x)/R 
      // for spherical collapse evolution
      //================================================
      double roverR_of_delta_and_x(
          double delta, 
          double x, 
          double RH0, 
          double delta_ini) { 
        return (RH0) * std::pow((1.0+delta_ini)/(1.0+delta),1.0/3.0)*std::exp(x); 
      };

      //===========================================================
      // Evolve the spherical collapse equation + linear
      // for a given set of IC delta_ini over a given range
      // [xini,xcollapse] and return the arrays + the x-value
      // where delta(x) > delta_infinity (i.e. it has collapsed
      // completely). If it haven't collapsed return xcollapse
      // This value can be used to more easily figure out delta_ini
      // that corresponds to collapse at exactly xcollapse via
      // root finding
      //===========================================================
      double evolve_spherical_collapse_equations(
          const SphericalCollapseModel & model,
          const double delta_ini, 
          DVector & x_array,
          DVector & delta_array,
          DVector & delta_prime_array,
          DVector & delta_lin_array,
          DVector & delta_lin_prime_array,
          const bool get_arrays,
          const double xini, 
          const double xcollapse,
          const int npts,
          const double delta_infinity) {

        // ODE solver parameters
        const double hstart = 1e-3;
        const double epsilon_abs = 1e-6;
        const double epsilon_rel = 1e-6;

        // The first x for which delta > delta_infinity
        double xminover = xcollapse;

        // Set x array (linear in 'a' seems to work well)
        x_array = FML::MATH::linspace(std::exp(xini),std::exp(xcollapse),npts);
        for(auto & x : x_array) x = std::log(x); 

        ODEFunction deriv = [&model,&xminover,&delta_infinity](double x, const double * y, double * dydx) -> int {
          // Check if we have collapsed
          if(y[0] > delta_infinity){
            if(x < xminover) xminover = x;
            dydx[0] = dydx[1] = 0.0;
            return GSL_SUCCESS;
          }
          return model.ode_system_non_linear_growth(x, y, dydx);
        };

        // The growing mode has D ~ a = exp(x) so D' = D
        DVector y_initial{delta_ini, delta_ini, delta_ini, delta_ini};
        ODESolver ode;
        ode.set_accuracy(hstart, epsilon_abs, epsilon_rel);
        ode.solve(deriv, x_array, y_initial, gsl_odeiv2_step_rkck);

        // Fetch solution arrays
        if(get_arrays){
          delta_array = ode.get_data_by_component(0);
          delta_prime_array = ode.get_data_by_component(1);
          delta_lin_array = ode.get_data_by_component(2);
          delta_lin_prime_array = ode.get_data_by_component(3);
        }
        return xminover;
      }

      //===========================================================
      // Do root finding to figure out the delta_ini that gives
      // collapse at x=xcollapse
      //===========================================================
      double find_delta_ini(
          const SphericalCollapseModel & model,
          const double xini, 
          const double xcollapse,
          const int npts,
          const double epsilon_delta){

        auto condition = [&](double delta_ini){
          DVector x_array, delta_array, delta_prime_array, delta_lin_array, delta_lin_prime_array;
          const double xminover = evolve_spherical_collapse_equations(
              model, 
              delta_ini, 
              x_array,
              delta_array, 
              delta_prime_array, 
              delta_lin_array, 
              delta_lin_prime_array, 
              false,
              xini, 
              xcollapse, 
              npts);
          return xminover - xcollapse;
        };
        return FML::MATH::find_root_bisection(condition, {0.0, 1.0}, epsilon_delta);
      }

      //================================================
      // Extract useful times from spherical collapse evolution
      // like turnaround, virialization and non-linear onset
      //================================================
      auto compute_times(
          const double xini,
          const double xcollapse,
          const SphericalCollapseModel & model,
          const Spline & delta_spline, 
          const Spline & delta_prime_spline) -> std::tuple<double, double, double, double> {

        // Accuracy in x for finding times
        const double epsilon_x = 1e-8;

        // Compute turnaround time
        auto turnaround_condition = [&delta_spline, &delta_prime_spline](double x){
          const double delta = delta_spline(x), delta_prime = delta_prime_spline(x);
          return 1.0 - delta_prime/3.0/(1.0+delta);
        };
        const double xturnaround = FML::MATH::find_root_bisection(turnaround_condition, {xini, xcollapse}, epsilon_x);

        // Compute virialization time
        auto virial_condition = [&model, &delta_spline, &delta_prime_spline](double x) {
          const double delta = delta_spline(x), delta_prime = delta_prime_spline(x);
          const double roverR = roverR_of_delta_and_x(delta, x);
          return model.virial_condition(delta, delta_prime, roverR, x);
        };
        // If we have several times where 2T+W=0 pick the one with largest redshift
        double x_search = xturnaround;
        const double deltax = 0.01;
        const int sign = virial_condition(x_search) > 0 ? 1 : -1;
        while(true){
          const int sign1 = virial_condition(x_search) > 0 ? 1 : -1;
          if(sign * sign1 == -1) break;
          x_search += deltax;
        }
        const double xvir = FML::MATH::find_root_bisection(virial_condition, {xturnaround, x_search}, epsilon_x);

        // Compute non-linear time
        auto nonlinear_condition = [&model, &delta_spline, &delta_prime_spline](double x){
          const double delta = delta_spline(x), delta_prime = delta_prime_spline(x);
          const double roverR = roverR_of_delta_and_x(delta, x);
          return model.nonlinear_condition(delta, delta_prime, roverR, x);;
        };
        const double xnl = FML::MATH::find_root_bisection(nonlinear_condition, {xini, xcollapse}, epsilon_x);

        // Compute time for which r=rturnaround/2
        const double roverR_turnaround = roverR_of_delta_and_x(delta_spline(xturnaround), xturnaround);
        auto halfturnaround_condition = [&roverR_turnaround, &delta_spline](double x) {
          const double delta = delta_spline(x);
          const double roverR = roverR_of_delta_and_x(delta, x);
          return roverR - roverR_turnaround/2.0;
        };
        const double xhalfturnaround = FML::MATH::find_root_bisection(halfturnaround_condition, {xturnaround, xcollapse}, epsilon_x);

        return std::tuple(xturnaround, xvir, xnl, xhalfturnaround);
      }

      //================================================
      // Runs spherical collapse for all redshifts
      // and makes splines of deltac(z), ...
      // and other stuff
      //================================================
      void compute_sphericalcollapse_splines(
          const SphericalCollapseModel & model,
          Spline & deltac_of_x_spline,
          Spline & DeltaVir_of_x_spline,
          Spline & xnl_of_x_spline,
          Spline & xta_of_x_spline,
          Spline & xvir_of_x_spline,
          Spline & delta_ini_of_x_spline,
          const double xini,
          const int npts,
          bool verbose){

        // Run spherical collapse and extract splines
        auto run_and_generate_delta_splines = [](
            const double xini,
            const double xcollapse,
            const int npts,
            const SphericalCollapseModel & model,
            Spline & delta_spline, 
            Spline & delta_prime_spline, 
            Spline & delta_lin_spline,
            double delta_ini = -1.0){
          if(delta_ini < 0.0)
            delta_ini = find_delta_ini(model, xini, xcollapse, npts);
          DVector x_array, delta_array, delta_prime_array, delta_lin_array, delta_lin_prime_array;
          evolve_spherical_collapse_equations(
              model, 
              delta_ini,
              x_array,
              delta_array, 
              delta_prime_array, 
              delta_lin_array, 
              delta_lin_prime_array, 
              true,
              xini, 
              xcollapse, 
              npts);
          delta_spline = Spline(x_array, delta_array, "delta(loga)");
          delta_prime_spline = Spline(x_array, delta_prime_array, "ddelta/dloga(loga)");
          delta_lin_spline = Spline(x_array, delta_lin_array, "deltaLin(loga)");
          return delta_ini;
        };

        // Range we want deltac(z) etc on
        const double zini = 10.0;
        const double zend = 0.0;
        const int npts_z = 100;
        const auto xcollapse_array = FML::MATH::linspace(std::log(1.0/(1.0+zini)),std::log(1.0/(1.0+zend)),npts_z);

        DVector deltac_array(xcollapse_array.size());
        DVector DeltaVir_array(xcollapse_array.size());
        DVector xta_array(xcollapse_array.size());
        DVector xvir_array(xcollapse_array.size());
        DVector xnl_array(xcollapse_array.size());
        DVector delta_ini_array(xcollapse_array.size());

#ifdef USE_OMP
#pragma omp parallel for schedule(dynamic,1)
#endif
        for(size_t i = 0; i < xcollapse_array.size(); i++){
          auto xcollapse = xcollapse_array[i];

          // Generate solution
          Spline delta_spline;
          Spline delta_prime_spline;
          Spline delta_lin_spline;
          const double delta_ini = run_and_generate_delta_splines(
              xini, 
              xcollapse, 
              npts, 
              model, 
              delta_spline, 
              delta_prime_spline, 
              delta_lin_spline);

          // Compute quantities
          const auto times = compute_times(xini, xcollapse, model, delta_spline, delta_prime_spline);
          const double xta  = std::get<0>(times);
          const double xvir = std::get<1>(times);
          const double xnl  = std::get<2>(times);
          const double xta2  = std::get<3>(times);
          const double DeltaVir = (1.0+delta_spline(xvir))*std::exp(3.0*xcollapse-3.0*xvir);
          const double deltac = delta_lin_spline(xcollapse);

          if(xcollapse == 0.0 and verbose){
            std::cout << "\n#=====================================================\n";
            std::cout << "# For collapse at z=0 we get:\n";
            std::cout << "#=====================================================\n";
            std::cout << " deltac       = " << deltac << "\n";
            std::cout << " DeltaVir     = " << DeltaVir << "\n";
            std::cout << " Turnaround r = " << roverR_of_delta_and_x(delta_spline(xta),xta,1.0,0.0) << "\n";
            std::cout << " Virial     r = " << roverR_of_delta_and_x(delta_spline(xvir),xvir,1.0,0.0) << "\n";
            std::cout << " rta/2        = " << roverR_of_delta_and_x(delta_spline(xta2),xta2,1.0,0.0) << "\n";
            std::cout << "#=====================================================\n";
          }

          xta_array[i] = xta;
          xnl_array[i] = xnl;
          xvir_array[i] = xvir;
          deltac_array[i] = deltac;
          DeltaVir_array[i] = DeltaVir;
          delta_ini_array[i] = delta_ini;

          // In this case the problem is not well posed so return nan
          if(model.muofx(xcollapse) <= 0.0){
            throw std::runtime_error("Spherical collapse. Negative Geff/G encountere. Problem not well posed!");
          }
        }
        xta_of_x_spline = Spline(xcollapse_array, xta_array, "loga_turnaround(loga_collapse)");
        xnl_of_x_spline = Spline(xcollapse_array, xnl_array, "loga_nonlinear(loga_collapse)");
        xvir_of_x_spline = Spline(xcollapse_array, xvir_array, "loga_virialization(loga_collapse)");
        deltac_of_x_spline = Spline(xcollapse_array, deltac_array, "deltac(loga_collapse)");
        DeltaVir_of_x_spline = Spline(xcollapse_array, DeltaVir_array, "DeltaVir(loga_collapse)");
        delta_ini_of_x_spline = Spline(xcollapse_array, delta_ini_array, "delta_ini(loga_collapse)");
      }

    }
  }
}
