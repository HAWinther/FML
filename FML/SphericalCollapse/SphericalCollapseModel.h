#ifndef SPHERICALCOLLAPSEMODEL
#define SPHERICALCOLLAPSEMODEL
#include <FML/ODESolver/ODESolver.h>
#include <FML/Math/Math.h>
#include <functional>
#include <iostream>
#include <cmath>

namespace FML {
  namespace COSMOLOGY {
    namespace SPHERICALCOLLAPSE {

      using ODEFunction = FML::SOLVERS::ODESOLVER::ODEFunction;
      using Func1 = std::function<double(double)>;

      //============================================================
      /// The class contains all the info about the model
      /// needed to run the spherical collapse and halomodel
      /// Cosmology/gravity functions provided using lambda functions
      //============================================================
      class SphericalCollapseModel {
        public:

          //============================================================
          // Cosmological functions. Below x = log(a). w(x) is equation of state
          // E(x) == H(x)/H0, mu(x) == Geff/G, logEprimeofx = dlogE/dx
          //============================================================
          Func1 Eofx;
          Func1 OmegaMofx;
          Func1 logEprimeofx;
          Func1 muofx;
          Func1 wofx;

          //============================================================
          // Cosmological parameters
          //============================================================
          double OmegaM{0.3};

          SphericalCollapseModel(){
            set_model_lcdm();
          }
          SphericalCollapseModel(
              Func1 Eofx,
              Func1 OmegaMofx,
              Func1 logEprimeofx,
              Func1 muofx,
              Func1 wofx) :
            Eofx(Eofx),
            OmegaMofx(OmegaMofx),
            logEprimeofx(logEprimeofx),
            muofx(muofx),
            wofx(wofx),
            OmegaM(OmegaMofx(0.0)) {}

          //============================================================
          /// Virial condition. Virialization when this quantity is zero
          /// Standard condition 2*Ek+Ep = 0 where Ek is kinetic and Ep is
          /// potential energy Ep/M of a spherical tophat overdensity
          //============================================================
          double virial_condition(
              double delta, 
              double delta_prime, 
              double roverR, 
              double x) const;

          //============================================================
          /// Define the non-linear time when this quantity is zero
          /// Using delta-1 = 0
          //============================================================
          double nonlinear_condition(
              double delta, 
              [[maybe_unused]] double delta_prime, 
              [[maybe_unused]] double roverR, 
              [[maybe_unused]] double x) const;

          //============================================================
          /// Print useful info
          //============================================================
          void info() const;

          //============================================================
          /// Set the model to be lcdm
          //============================================================
          void set_model_lcdm(){
            Eofx = Eofx_lcdm;
            OmegaMofx = OmegaMofx_lcdm;
            logEprimeofx = logEprimeofx_lcdm;
            muofx = muofx_lcdm;
            wofx = wofx_lcdm;
          }

          //============================================================
          /// Define the linear ODE system
          /// 0 = delta_lin
          /// 1 = ddeltadloga_lin
          //============================================================
          int ode_system_linear_growth(double x, const double * y, double * dydx) const {
            const double mu = muofx(x);
            const double om = OmegaMofx(x);
            const double dlogEdloga = logEprimeofx(x);
            const double delta = y[0];
            const double ddeltadx = y[1];

            // Growth ODE
            dydx[0] = ddeltadx;
            dydx[1] = -(2.0+dlogEdloga)*ddeltadx + 1.5*om*mu*delta;
            return GSL_SUCCESS;
          }

          //============================================================
          /// Define the non-linear + linear ODE system for spherical
          /// collapse
          /// 0 = delta
          /// 1 = ddeltadloga
          /// 2 = delta_lin
          /// 3 = ddelta_lindloga
          //============================================================
          int ode_system_non_linear_growth(double x, const double * y, double * dydx) const {
            const double mu = muofx(x);
            const double om = OmegaMofx(x);
            const double dlogEdloga = logEprimeofx(x);
            const double delta = y[0];
            const double ddeltadx = y[1];
            const double delta_lin = y[2];
            const double ddeltadx_lin = y[3];

            // Spherical collapse
            dydx[0] = ddeltadx;
            dydx[1] = -(2.0+dlogEdloga)*ddeltadx + 1.5*om*mu*delta*(1+delta) + 4.0/3.0*ddeltadx*ddeltadx/(1.0+delta);

            // Linear evolution
            dydx[2] = ddeltadx_lin;
            dydx[3] = -(2.0+dlogEdloga)*ddeltadx_lin + 1.5*om*mu*delta_lin;
            return GSL_SUCCESS;
          }

          //============================================================
          // LCDM functions which can be used if none is provided
          //============================================================
          Func1 Eofx_lcdm =
            [&](double x){ return std::sqrt(OmegaM*std::exp(-3*x) + 1.0 - OmegaM); };
          Func1 OmegaMofx_lcdm = 
            [&](double x){ return OmegaM*std::exp(-3*x)/(OmegaM*std::exp(-3*x) + 1.0 - OmegaM); };
          Func1 logEprimeofx_lcdm = 
            [&](double x){ return -1.5*OmegaMofx(x); };
          Func1 muofx_lcdm = 
            [&]([[maybe_unused]]double x){ return 1.0; };
          Func1 wofx_lcdm = 
            [&]([[maybe_unused]]double x){ return -1.0; };

      };

    }
  }
}

#endif
