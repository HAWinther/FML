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

class CosmologyJBD final : public Cosmology {
  public:
    CosmologyJBD() : Cosmology(1e-10, 1e0, 1000) { name = "JBD"; }

    //========================================================================
    // Read the parameters we need
    //========================================================================
    void read_parameters(ParameterMap & param) override {
        Cosmology::read_parameters(param);
        wBD = param.get<double>("cosmology_JBD_wBD");
        GeffG_today = param.get<double>("cosmology_JBD_GeffG_today");
    }

    //========================================================================
    // Initialize cosmology by bisecting/shooting for phi_ini and OmegaLambda
    // that gives desired Geff/G today and satisfies closure condition E0 == 1
    //========================================================================
    void init() override {
        double phi_today_target = (4+2*wBD) / (3+2*wBD) / GeffG_today; // arXiv:2010.15278 equation (17)
        double phi_ini_lo = 0.0;
        double phi_ini_hi = phi_today_target;
        OmegaLambda = 0.0; // initial arbitrary guess for OmegaLambda (stupidly chosen to emphasize it is a *guess*; smarter guesses gives <1 fewer iterations)

        if (FML::ThisTask == 0) {
            std::cout << "JBD::init Guessing   phi_ini = ??????????, OmegaLambda = ????? that gives "
                      << "phi_today = " << std::fixed << std::setprecision(8) << phi_today_target << ", "
                      << "E_today = " << 1.0 << ":\n";
        }

        // Here          (and     in hiclass): input h and Omegas except OmegaL, enforce E0 == 1, and output OmegaL
        // Alternatively (and not in hiclass): input all Omegas but not h,       relax   E0 == 1, and output h

        for (int iter = 1; iter < 100; iter++) {
            phi_ini = (phi_ini_lo + phi_ini_hi) / 2.0; // try new phi_ini between bisection limits

            // Solve cosmology for current (phi_ini, OmegaLambda)
            init_current();
            double phi_today = phi_of_a(1.0);

            if (FML::ThisTask == 0) {
                std::cout << "JBD::init Guess #" << std::setiosflags(std::ios::right) << std::setw(2) << iter << std::setiosflags(std::ios::left) << ": "
                          << std::fixed << std::setprecision(8) // 8 decimals in all following numbers
                          << "phi_ini = " << phi_ini << ", OmegaLambda = " << OmegaLambda << " gives "
                          << "phi_today = " << phi_today << ", E_today = " << HoverH0_of_a(1.0) << "\n";
            }

            // Check for convergence (phi_today == phi_today_target and E0 == 1)
            bool converged_phi_today = std::fabs(phi_today - phi_today_target) < 1e-9; // require 8=9-1 correct decimals
            bool converged_E_today   = std::fabs(HoverH0_of_a(1.0) - 1.0)      < 1e-9; // require 8=9-1 correct decimals
            if (converged_phi_today && converged_E_today) {
                if (FML::ThisTask == 0) {
                    std::cout << "JBD::init Guessing converged in " << iter << " iterations\n";
                }
                return; // hit, so stop; the cosmology is now initialized and ready-to-use
            }

            // Refine guess for phi_ini from bisection limits
            if (phi_today < phi_today_target) {
                phi_ini_lo = phi_ini; // underhit, so increase next guess
            } else {
                phi_ini_hi = phi_ini; //  overhit, so decrease next guess
            }

            // Refine guess for OmegaLambda from closure condition E0 == 1,
            // equivalent to phi == OmegaR + Omegab + OmegaCDM + OmegaNu + phi*(OmegaK + OmegaPhi) today (and OmegaPhi is defined below)
            // (reduces to familiar 1 == sum(Omega_i) only in the specific case with phi == 1 today!)
            double OmegaPhi = -dlogphi_dloga_of_a(1.0) + wBD/6 * dlogphi_dloga_of_a(1.0) * dlogphi_dloga_of_a(1.0);
            OmegaLambda = phi_today - OmegaR - this->get_rhoNu_exact(1.0) - Omegab - OmegaCDM - phi_today*(OmegaK + OmegaPhi); // equivalent to E0 == 1 // TODO: Hans says neutrino treatment should be corrected/improved at some point
        }

        throw std::runtime_error("JBD::init Guessing did not converge");
    };

    //========================================================================
    // Initialize cosmology (with current values of phi_ini and OmegaLambda)
    //========================================================================
    void init_current() {
        Cosmology::init();

        // Convenience functions for calculating E = sqrt(E2_frac_top / E2_frac_bot)
        auto E2_frac_top = [&](double loga, double logphi) {
            double a = std::exp(loga);
            double phi = std::exp(logphi);
            return OmegaR / (a*a*a*a) + this->get_rhoNu_exact(a) + (Omegab+OmegaCDM) / (a*a*a) + phi * OmegaK / (a*a) + OmegaLambda; // TODO: Hans says neutrino treatment should be corrected/improved at some point
        };
        auto E2_frac_bot = [&](double logphi, double dlogphi_dloga) {
            double phi = std::exp(logphi);
            return phi * (1.0 + dlogphi_dloga - wBD/6 * dlogphi_dloga * dlogphi_dloga);
        };
        auto E_func = [&](double loga, double logphi, double dlogphi_dloga) {
            return std::sqrt(E2_frac_top(loga, logphi) / E2_frac_bot(logphi, dlogphi_dloga));
        };

        // ODE system for scalar field phi (here "phi" means phi/phi0 TODO: verify!):
        // y0' = d(logphi)                  / dloga
        // y1' = d(a^3*E*phi*dlogphi/dloga) / dloga = 3/(3+2*wBD) * (OmegaM + 4*OmegaLambda*a^3) / E
        FML::SOLVERS::ODESOLVER::ODEFunction deriv = [&](double loga, const double * y, double * dy_dloga) {
            double a = std::exp(loga);

            // logphi is y0
            double logphi = y[0];
            double phi = std::exp(logphi);

            // dlogphi_dloga is solution of y1 == a^3 * E(dlogphi_dloga) * phi * dlogphi_dloga
            // Squaring it and expanding E2(dlogphi_dloga) = E2_frac_top / E2_frac_bot(dlogphi_dloga),
            // it becomes a quadratic equation that can be solved exactly
            double A = phi * a*a*a*a*a*a * E2_frac_top(loga, logphi) + wBD/6 * y[1]*y[1];
            double B = -y[1]*y[1];
            double C = -y[1]*y[1];
            double dlogphi_dloga = (-B + std::sqrt(B*B - 4*A*C)) / (2*A); // + to take positive solution

            dy_dloga[0] = dlogphi_dloga;
            dy_dloga[1] = 3.0 / (3.0+2.0*wBD) * (OmegaM + 4.0 * OmegaLambda * a*a*a) / E_func(loga, logphi, dlogphi_dloga);
            return GSL_SUCCESS;
        };

        // Integrate scalar field phi, assuming dlogphi_dloga == 0 at early times in radiation era
        // (i.e. neglecting the unphysical diverging mode in the approximate analytical solution phi = A + B/a)
        FML::SOLVERS::ODESOLVER::ODESolver ode;
        DVector loga_arr = FML::MATH::linspace(std::log(alow), std::log(ahigh), npts_loga); // scale factor logarithms to integrate over
        DVector y_ini{std::log(phi_ini), dlogphi_dloga_ini}; // assume dlogphi_dloga == 0
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
    double phi_of_a(double a) const { return std::exp(logphi_of_loga_spline(std::log(a))); } // normalized to 1 / phi_today = (4+2*wBD) / (3+2*wBD) / GeffG_today
    double dlogphi_dloga_of_a(double a) const { return logphi_of_loga_spline.deriv_x(std::log(a)); }

    // Override and extend parent info and output functions with additional parameters and scalar field
    void info() const override {
        Cosmology::info();
        if (FML::ThisTask == 0) {
            std::cout << "# wBD               : " << wBD << "\n";
            std::cout << "# GeffG_today       : " << GeffG_today << "\n";
            std::cout << "# phi_ini           : " << phi_ini << "\n";
            std::cout << "# dlogphi_dloga_ini : " << dlogphi_dloga_ini << "\n";
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
    double wBD; // independent
    double GeffG_today; // independent
    double phi_ini; // dependent on GeffG_today
    double dlogphi_dloga_ini = 0.0;

    //========================================================================
    // Splines for the Hubble function (E = H/H0) and JBD scalar field phi
    //========================================================================
    Spline logE_of_loga_spline;
    Spline logphi_of_loga_spline;
};
#endif
