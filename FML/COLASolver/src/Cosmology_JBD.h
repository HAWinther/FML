#ifndef COSMOLOGY_JBD_HEADER
#define COSMOLOGY_JBD_HEADER

#include <FML/FFTWGrid/FFTWGrid.h>
#include <FML/Global/Global.h>
#include <FML/LPT/DisplacementFields.h>
#include <FML/ODESolver/ODESolver.h>
#include <FML/ParameterMap/ParameterMap.h>
#include <FML/Spline/Spline.h>
#include <FML/Math/Math.h>

#include "Cosmology.h"

//========================================================================
// Some implementation notes on JBD
// Code by Herman Slettmoen and HAW (UiO)
//========================================================================
// 
//========================================================================
// 1. Gravitational constant:
//========================================================================
// In JBD Newtons constant is G_N = G / phi_star where phi_star = (4+2w)/(3+2w) and
// G is the parameter in the action (sometimes called the "bare" gravitational constant).
// If we want the correct newtonian limit we should have phi_today = phi_star
// We leave this choice free by introducing the parameter GeffG_today == phi_star / phi_today
// such that if GeffG_today = 1 then phi_today = phi_star
//
//========================================================================
// 2. Density parameter definitions:
//========================================================================
// We allow for different definitions (density_parameter_definition = hi-class, Gnewton, Gbare, Gtoday)
//    Omega == 8piG_* rho / 3H0^2
// For Gnewton: G_* = G/phi_*, for Gbare: G_* = G, for Gtoday = G / phi(0) and then its hi-class (which is very similar to Gnewton)
// With Gtoday then Sum Omega == 1 today.
// The only exception above is curvature OmegaK = -k/H0^2 in all cases which is as usual and OmegaPhi = -(dphi/dloga) + w/6 (dphi/dloga)^2.
// NB: for times other than a=1 the function get_Omega currently returns the same expression as in GR, i.e. without any 1/phi factor.
//
// The Poisson equation using these definitions is:
//    D^2 Phi = const * Omega * (G/G_*) * phi_star / phi
//
//========================================================================
// 3. Solving background
//========================================================================
// The algorithm used below is a single shoot-match to fit phi_ini while adjusting
// OmegaLambda using closure condition. We also have the opportunity to do a double
// shoot-match (for both OmegaLambda and phi_ini) which is a bit slower.
// The algorithm givse us a solution with the desired phi_today and which satisfy H(a=1) == H0
//
//========================================================================
//
//========================================================================
// 4. Connection to hi-class
//========================================================================
// 
// If hi-class is run with M_pl_today_smg = ... and normalize_G_NR = no then we should use:
// cosmology_JBD_wBD = 100.0
// cosmology_JBD_GeffG_today = (4+2*cosmology_JBD_wBD)/(3+2*cosmology_JBD_wBD)/M_pl_today_smg
// cosmology_JBD_density_parameter_definition = "hi-class"
//
// If we run hi-class with M_pl_today_smg and normalize_G_NR = yes then we need
// cosmology_JBD_wBD = 100.0
// cosmology_JBD_GeffG_today = 1.0
// cosmology_JBD_density_parameter_definition = "hi-class"
//
//========================================================================

class CosmologyJBD final : public Cosmology {
  public:
    CosmologyJBD() : Cosmology(1e-9, 10.0, 2500) { name = "JBD"; }

    //========================================================================
    // Read the parameters we need
    //========================================================================
    void read_parameters(ParameterMap & param) override {
        Cosmology::read_parameters(param);
        wBD = param.get<double>("cosmology_JBD_wBD");
        GeffG_today = param.get<double>("cosmology_JBD_GeffG_today");
        density_parameter_definition = param.get<std::string>("cosmology_JBD_density_parameter_definition");
    }

    //========================================================================
    // Initialize cosmology by bisecting/shooting for phi_ini and OmegaLambda
    // that gives desired Geff/G today and satisfies closure condition E0 == 1
    //========================================================================
    void init() override {

        // Parent class initialization (makes splines for neutrinos etc.)
        Cosmology::init();
       
        // The phi value we need cosmologically today for newtons constant to be newtons constant
        phi_star = (4 + 2*wBD) / (3 + 2*wBD);

        // The phi-value we shoot for today
        const double phi_today = phi_star / GeffG_today;

        // Select the density parameter definition to use
        // The radiation density parameters have been computed using G_newton
        // i.e. Omega = 8piG_Nrho/3H0^2 = 8piGrho/3H0^2phi_star
        // so we need to rescale them if we use a different definition
        if(density_parameter_definition == "Gtoday") {
          // Omega0 = 8piG_today rho0/3H0^2= 8piG rho0/3H0^2phi_today
          OmegaR    *= phi_star / phi_today;
          OmegaNu   *= phi_star / phi_today;
          OmegaRtot *= phi_star / phi_today;
          Mnu_eV /= phi_star / phi_today;
          G_over_Gdensityparams = phi_today;
        } else if(density_parameter_definition == "Gbare") {
          // Omega0 = 8piG rho0/3H0^2
          OmegaR    *= phi_star;
          OmegaNu   *= phi_star;
          OmegaRtot *= phi_star;
          Mnu_eV /= phi_star;
          G_over_Gdensityparams = 1.0;
        } else if(density_parameter_definition == "Gnewton") {
          // Omega0 = 8piG rho0/3H0^2phi_star = 8piG_N rho0/3H0^2
          OmegaR    *= 1.0;
          OmegaNu   *= 1.0;
          OmegaRtot *= 1.0;
          Mnu_eV /= 1.0;
          G_over_Gdensityparams = phi_star;
        } else if(density_parameter_definition == "hi-class") {
          // Omega0 = 8piG rho0/3H0^2 ~ 8piG_N rho0/3H0^2
          // Basically assuming phi_star = 1 (but only in the density parameters)
          G_over_Gdensityparams = 1.0;
        } else {
          throw std::runtime_error("JBD::init do not recognize density_parameter_definition (Gbare, Gtoday, Gnewton, hi-class)\n");
        }

        // Outside these ranges the solver does not converge
        FML::assert_mpi( !(wBD < -0.5 and wBD > -6.0), "JBD::init Not a valid value for wBD. If you want to try other values remove this test");

        if (FML::ThisTask == 0) {
            std::cout << "JBD::init Trying to find phi_ini and OmegaLambda that gives "
                      << "phi_today = " << std::fixed << std::setprecision(8) << phi_today << ", "
                      << "E_today = " << 1.0 << ":\n";
        }

        // Shoot-match to figure out correct phi_ini and OmegaLambda
        // Limits for OmegaLambda (broad to allow for extreme cases like w ~ 0)
        double OmegaLambda_lo = OmegaLambda_limits_lo;
        double OmegaLambda_hi = OmegaLambda_limits_hi;
        for (int iter_lambda = 0; iter_lambda < nmax_iter_phi; iter_lambda++) {
          
            // Try new OmegaLambda between bisection limits
            OmegaLambda = (OmegaLambda_lo + OmegaLambda_hi)/2.0;
           
            // Limits for phi (extra broad)
            double phi_ini_lo = phi_ini_limits_lo;
            double phi_ini_hi = phi_ini_limits_hi;
            double phi_today_current;
            for (int iter_phi = 0; iter_phi < nmax_iter_phi; iter_phi++) {
                // Try new phi_ini between bisection limits
                phi_ini = (phi_ini_lo + phi_ini_hi) / 2.0; 

                // Solve cosmology for current (phi_ini, OmegaLambda)
                solve_current();
                phi_today_current = phi_of_a(1.0);

                // Check for convergence (phi_today == phi_today_current)
                bool converged_phi_today = std::fabs(phi_today_current/phi_today-1.0) < epsilon_convergence_phi;
                if (converged_phi_today) {
                    if (FML::ThisTask == 0) {
                        std::cout << "JBD::init Guessing for phi converged in " << iter_phi << " iterations " << std::fixed << std::setprecision(8)
                          << "phi_ini = " << phi_ini << " phi_today = " << phi_today_current << "\n";
                    }

                    if(single_shoot_algorithm){
                      bool converged_E_today = std::fabs(HoverH0_of_a(1.0) - 1.0) < epsilon_convergence_E;
                      if (converged_E_today) break;
                    } else {
                      // Shoot-match for phi is converged
                      break;
                    }
                }

                // Refine guess for phi_ini from bisection limits
                if (phi_today_current < phi_today) {
                    // Underhit, so increase next guess
                    phi_ini_lo = phi_ini; 
                } else {
                    // Overhit, so decrease next guess
                    phi_ini_hi = phi_ini; 
                }
            
                // The single shoot algorithm solves it in one go
                if(single_shoot_algorithm){
                  double dlogphi_dloga = dlogphi_dloga_of_a(1.0);
                  OmegaLambda = (1.0 - OmegaK + dlogphi_dloga - wBD/6.0 * dlogphi_dloga * dlogphi_dloga) * (phi_today / G_over_Gdensityparams) 
                    - (OmegaR + this->get_rhoNu_exact(1.0) + Omegab + OmegaCDM);
                }

                if(iter_phi == nmax_iter_phi - 1) throw std::runtime_error("JBD::init Shoot-match for phi did not converge!\n");
            }
            
            if (FML::ThisTask == 0) {
                std::cout << "JBD::init Guess #" << std::setiosflags(std::ios::right) << std::setw(2) << iter_lambda << std::setiosflags(std::ios::left) << ": "
                          << std::fixed << std::setprecision(8)
                          << "OmegaLambda = " << OmegaLambda << ", E_today = " << HoverH0_of_a(1.0) << "\n";
            }
            
            // Converged if we do single shoot, so stop; the cosmology is now initialized and ready-to-use
            if(single_shoot_algorithm)
              return;

            // Refine guess for OmegaLambda from bisection limits
            if(HoverH0_of_a(1.0) < 1.0) {
                OmegaLambda_lo = OmegaLambda;
            } else {
                OmegaLambda_hi = OmegaLambda;
            }
                
            // Check for convergence (E0 = H(a=1)/H0 == 1.0)
            bool converged_E_today = std::fabs(HoverH0_of_a(1.0) - 1.0) < epsilon_convergence_E;
            if (converged_E_today) {
                if (FML::ThisTask == 0) {
                    std::cout << "JBD::init Guessing for lambda converged in " << iter_lambda << " iterations\n";
                }
                // Converged, so stop; the cosmology is now initialized and ready-to-use
                return; 
            }
        }
           
        throw std::runtime_error("JBD::init Shoot-match for lambda did not converge!");
    };

    //========================================================================
    // Solve cosmology (with current values of phi_ini and OmegaLambda)
    //========================================================================
    void solve_current() {

        auto E_func = [&](double loga, double logphi, double dlogphi_dloga) {
            double a = std::exp(loga);
            double phi = std::exp(logphi);
            return std::sqrt(((OmegaR / (a*a*a*a) + this->get_rhoNu_exact(a) + (Omegab+OmegaCDM) / (a*a*a) + OmegaLambda) * (G_over_Gdensityparams / phi) 
                  + OmegaK / (a*a)) / (1.0 + dlogphi_dloga - wBD/6.0 * dlogphi_dloga * dlogphi_dloga)); 
        };

        // ODE system for scalar field phi
        // a^3E d/dloga [a^3 E dphi/dlog a] = 3a^3/(3+2*wBD) * (OmegaM + 4*OmegaLambda*a^3)
        // made into first order system
        // y0' = d(logphi) / dloga
        // y1' = d(a^3*E*phi*dlogphi/dloga) / dloga = 3/(3+2*wBD) * (OmegaM + 4*OmegaLambda*a^3) / E
        FML::SOLVERS::ODESOLVER::ODEFunction deriv = [&](double loga, const double * y, double * dy_dloga) {
            double a = std::exp(loga);
            double logphi = y[0];
            double phi    = std::exp(logphi);

            // dlogphi_dloga is solution of y1 == a^3 * E(dlogphi_dloga) * phi * dlogphi_dloga
            // Squaring it and expanding E2(dlogphi_dloga) it becomes a quadratic equation that can be solved exactly. 
            // NB: + in quadratic formula picks out correct solution
            // NB: we are calling E with dlogphi_dloga = 0 to extract the "numerator" of the E-expression
            double A = pow(a * a * a * (phi / G_over_Gdensityparams) * E_func(loga, logphi, 0.0), 2) + wBD/6.0 * y[1]*y[1];
            double B = -y[1]*y[1];
            double C = -y[1]*y[1];
            double dlogphi_dloga = (-B + std::sqrt(B*B - 4*A*C)) / (2*A); 

            dy_dloga[0] = dlogphi_dloga;
            dy_dloga[1] = 3.0 / (3.0 + 2.0*wBD) * 
              ( // Radiation has rho - 3p = 0 so is not included
               (OmegaCDM + Omegab) + 
               (get_rhoNu_exact(a) - 3.0*get_pNu_exact(a)) * (a * a * a) + 
               4.0 * OmegaLambda * (a * a * a)
              ) / E_func(loga, logphi, dlogphi_dloga);
            return GSL_SUCCESS;
        };
        

        // Integrate scalar field phi
        // Fiducial choice is to assume dlogphi_dloga == 0 at early times in radiation era
        // (i.e. neglecting the unphysical diverging mode in the approximate analytical solution phi = A + B/a)
        FML::SOLVERS::ODESOLVER::ODESolver ode;
        DVector loga_arr = FML::MATH::linspace(std::log(alow), std::log(ahigh), npts_loga);
        DVector y_ini{std::log(phi_ini), dlogphi_dloga_ini}; 
        ode.solve(deriv, loga_arr, y_ini);
        DVector logphi_arr = ode.get_data_by_component(0);

        // Spline logphi(loga)
        logphi_of_loga_spline.create(loga_arr, logphi_arr, "JBD logphi(loga)");

        // Spline logE(loga)
        DVector logE_arr(npts_loga);
        for (int i = 0; i < npts_loga; i++) {
            logE_arr[i] = std::log(E_func(loga_arr[i], logphi_of_loga_spline(loga_arr[i]), logphi_of_loga_spline.deriv_x(loga_arr[i])));
        }
        logE_of_loga_spline.create(loga_arr, logE_arr, "JBD logE(loga)");
    }

    //========================================================================
    // Spline evaluation wrappers (for Hubble function and scalar field)
    //========================================================================
    double HoverH0_of_a(double a) const override { return std::exp(logE_of_loga_spline(std::log(a))); }
    double dlogHdloga_of_a(double a) const override { return logE_of_loga_spline.deriv_x(std::log(a)); }
    double phi_of_a(double a) const { return std::exp(logphi_of_loga_spline(std::log(a))); } 
    double dlogphi_dloga_of_a(double a) const { return logphi_of_loga_spline.deriv_x(std::log(a)); }
    
    //========================================================================
    // JBD specific functions
    //========================================================================
    double get_OmegaPhi(double a = 1.0) const {
      double OmegaPhi = -dlogphi_dloga_of_a(a) + wBD/6.0 * dlogphi_dloga_of_a(a) * dlogphi_dloga_of_a(a);
      return OmegaPhi;
    }
    double get_wBD() const { return wBD; }
    double get_phi_star() const { return phi_star; }
    double get_G_over_Gdensityparams() const { return G_over_Gdensityparams; }

    // Override and extend parent info and output functions with additional parameters and scalar field
    void info() const override {
        Cosmology::info();
        if (FML::ThisTask == 0) {
            std::cout << "# wBD                     : " << wBD << "\n";
            std::cout << "# GeffG_today             : " << GeffG_today << "\n";
            std::cout << "# phi_ini                 : " << phi_ini << "\n";
            std::cout << "# z_ini                   : " << 1.0/alow-1.0 << "\n";
            std::cout << "# dlogphi_dloga_ini       : " << dlogphi_dloga_ini << "\n";
            std::cout << "# DensityParamDefinition  : " << density_parameter_definition << "\n";
            std::cout << "# OmegaPhi+Lambda         : " << get_OmegaPhi(1.0) + OmegaLambda << "\n";
            std::cout << "# Updated radiation density parameters: \n";
            std::cout << "# OmegaNu                 : " << OmegaNu << "\n";
            std::cout << "# OmegaR                  : " << OmegaR << "\n";
            std::cout << "# OmegaRtot               : " << OmegaRtot << "\n";
            std::cout << "# Gravitational constant over G_newton at different times:\n";
            std::cout << "# Radiation era           : " << phi_star / phi_ini << "\n";
            std::cout << "# Mat-Rad equality        : " << phi_star / phi_of_a(OmegaRtot/OmegaM) << "\n";
            std::cout << "# Recombination           : " << phi_star / phi_of_a(1e-3) << "\n";
            std::cout << "# Mat-DE equality         : " << phi_star / phi_of_a(std::pow(OmegaM/OmegaLambda, 1.0/3.0)) << "\n";
            std::cout << "# Today                   : " << phi_star / phi_of_a(1.0) << "\n";
            std::cout << "#=====================================================\n";
            std::cout << "\n";
        }
    }
    void output_header(std::ofstream & fp) const override {
        Cosmology::output_header(fp);
        fp << ' '; output_element(fp, "phi");
        fp << ' '; output_element(fp, "dlogphi/dloga");
    }
    void output_row(std::ofstream & fp, double a) const override {
        Cosmology::output_row(fp, a);
        fp << ' '; output_element(fp, phi_of_a(a));
        fp << ' '; output_element(fp, dlogphi_dloga_of_a(a));
    }
    
  protected:
    //========================================================================
    // Parameters specific to the JBD model
    //========================================================================
    double wBD;                    // Independent. GR recovered as w -> infty
    double GeffG_today{1.0};       // Independent. G_today / G_Newton. phi_today = phi_*/GeffG_today with phi_* = (4+2w)/(3+2w)
    double phi_star;               // phi_* = (4+3w)/(3+2w) Value of phi(0) s.t. Gmatter = G_N today
    double phi_ini;                // Initial value for phi at a = alow
    double dlogphi_dloga_ini{0.0}; // Initial value for dphidloga at a = alow

    // The density parameter definition we use:
    // Gtoday   : Omega = 8piGrho / 3H0^2phi(0)
    // Gbare    : Omega = 8piGrho/3H0^2
    // Gnewton  : Omega = 8piGrho/3H0^2phi_star
    // hi-class : Similar to Gnewton apart from a phi_star ~ 1 factor
    // natural implies Sum Omega == 1 today. 
    // GG_over_Gdensityparams = G/G_* where G_* is the "G" used to define the density parameters
    std::string density_parameter_definition{"Gtoday"};
    double G_over_Gdensityparams{1.0}; 

    // Shoot-match interval for OmegaLambda
    // If for some case one is interested in very small 
    // values of wBD <~ -0.5 one might need to increase this
    // as OmegaLambda -> infty as wBD -> -1.5
    double OmegaLambda_limits_lo = 0.0;
    double OmegaLambda_limits_hi = 1e2;

    // Shoot-match interval for phi_ini
    double phi_ini_limits_lo = 0.0;
    double phi_ini_limits_hi = 1e2;
    
    // Convergence criteria for shoot-match
    double epsilon_convergence_phi = 1e-6;
    double epsilon_convergence_E = 1e-6;
    int nmax_iter_phi = 1000;
    int nmax_iter_lambda = 1000;

    // Use faster algorithm to solve background or solver
    // but maybe more roubust
    bool single_shoot_algorithm{true};

    //========================================================================
    // Splines for the Hubble function (E = H/H0) and JBD scalar field phi
    //========================================================================
    Spline logE_of_loga_spline{"JBD logE(loga)"};
    Spline logphi_of_loga_spline{"JBD logphi(dloga)"};
};
#endif
