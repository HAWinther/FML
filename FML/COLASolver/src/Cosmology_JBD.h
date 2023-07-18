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
    CosmologyJBD() { name = "JBD"; }

    //========================================================================
    // Read the parameters we need
    //========================================================================
    void read_parameters(ParameterMap & param) override {
        Cosmology::read_parameters(param);
        wBD = param.get<double>("cosmology_JBD_wBD");
        GeffG_today = param.get<double>("cosmology_JBD_GeffG_today");
    }

    //========================================================================
    // Initialize cosmology (bisect/shoot for phi_ini and OmegaLambda)
    //========================================================================
    void init() override {
        // Bisect and shoot for correct (phi_ini, OmegaLambda)
        // that gives desired Geff/G today and satisfies closure condition (E0 = 1)
        double phi_today_target = 1.0 / GeffG_today; // TODO: w-factors
        double phi_ini_lo = 0.0;
        double phi_ini_hi = phi_today_target;
        OmegaLambda = 1.0 - OmegaR - Omegab - OmegaCDM - OmegaK; // initial guess for OmegaLambda (neglecting neutrinos and scalar field)

        std::cout << "JBD::init Searching for phi_ini and OmegaLambda that gives "
                  << "phi_today = " << phi_today_target << " and "
                  << "E_today = (H/H0)_today = 1.0:" << "\n";

        for (int iter = 1; iter < 100; iter++) {
            phi_ini = (phi_ini_lo + phi_ini_hi) / 2.0; // try new phi_ini between bisection limits

            // Solve cosmology for current (phi_ini, OmegaLambda) and record phi and its derivative today
            init_current();

            std::cout << "#" << std::setiosflags(std::ios::right) << std::setw(2) << iter << std::setiosflags(std::ios::left) << ": "
                      << std::fixed << std::setprecision(8) // 8 decimals in all following numbers
                      << "phi_ini = " << std::setw(10) << phi_ini << ", OmegaLambda = " << std::setw(10) << OmegaLambda << " gives "
                      << "phi_today = " << std::setw(10) << phi_of_a(1.0) << ", E_today = " << std::setw(10) << HoverH0_of_a(1.0) << "\n";

            // Check for convergence (phi_today == phi_today_target and E0 == 1) TODO: generalize to G/G != 1
            bool converged_phi_today = std::fabs(phi_of_a(1.0) / phi_today_target - 1.0) < 1e-8;
            bool converged_E_today   = std::fabs(HoverH0_of_a(1.0)                - 1.0) < 1e-8;
            if (converged_phi_today && converged_E_today) {
                std::cout << "JBD::init Search for phi_ini and OmegaLambda converged in " << iter << " iterations\n";
                return; // hit, so stop; the cosmology is initialized and ready-to-use
            }

            // Refine guesses for phi_ini (from bisection limits) and OmegaLambda (from closure condition E0 == 1)
            if (phi_of_a(1.0) < phi_today_target) {
                phi_ini_lo = phi_ini; // underhit, so increase next guess
            } else {
                phi_ini_hi = phi_ini; //  overhit, so decrease next guess
            }
            double OmegaPhi = -dlogphi_dloga_of_a(1.0) + wBD/6 * dlogphi_dloga_of_a(1.0) * dlogphi_dloga_of_a(1.0); // defined from E0 == 1, so sum_i Omega_i == 1
            OmegaLambda = 1.0 - OmegaR - this->get_rhoNu_exact(1.0) - Omegab - OmegaCDM - OmegaK - OmegaPhi; // equivalent to E0 == 1 // TODO: correct/improve neutrino treatment?
        }

        throw std::runtime_error("JBD::init Search for phi_ini and OmegaLambda did not converge");
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
            return OmegaR / (a*a*a*a) + this->get_rhoNu_exact(a) + (Omegab+OmegaCDM) / (a*a*a) + phi * OmegaK / (a*a) + OmegaLambda; // TODO: correct/improve neutrino treatment?
        };
        auto E2_frac_bot = [&](double logphi, double dlogphi_dloga) {
            double phi = std::exp(logphi);
            return phi * (1.0 + dlogphi_dloga - wBD/6 * dlogphi_dloga * dlogphi_dloga);
        };
        auto E_func = [&](double loga, double logphi, double dlogphi_dloga) {
            return std::sqrt(E2_frac_top(loga, logphi) / E2_frac_bot(logphi, dlogphi_dloga));
        };

        // ODE system for scalar field phi (here "phi" means phi/phi0 TODO: does it, really?):
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

        // Integrate scalar field phi
        FML::SOLVERS::ODESOLVER::ODESolver ode;
        DVector loga_arr = FML::MATH::linspace(loga_min, loga_max, nloga); // scale factor logarithms to integrate over
        DVector yini{std::log(phi_ini), 0.0}; // assume dlogphi_dloga = 0 at early times // TODO: explain why
        ode.solve(deriv, loga_arr, yini);
        auto logphi_arr = ode.get_data_by_component(0);

        // Spline logphi(loga)
        logphi_of_loga_spline.create(loga_arr, logphi_arr, "JBD logphi(loga)");

        // Spline logE(loga)
        auto logE_of_loga_arr = loga_arr; // copy to create vector of same size (preserving loga)
        for (auto & logE : logE_of_loga_arr) {
            double loga = logE; // because we copied the vector above
            logE = std::log(E_func(loga, logphi_of_loga_spline(loga), logphi_of_loga_spline.deriv_x(loga)));
        }
        logE_of_loga_spline.create(loga_arr, logE_of_loga_arr, "JBD logE(loga)");
    }

    //========================================================================
    // Print some info
    //========================================================================
    void info() const override {
        Cosmology::info();
        if (FML::ThisTask == 0) {
            std::cout << "# wBD           : " << wBD << "\n";
            std::cout << "# GeffG_today   : " << GeffG_today << "\n";
            std::cout << "# phi_ini       : " << phi_ini << "\n";
            std::cout << "#=====================================================\n";
            std::cout << "\n";
        }
    }

    //========================================================================
    // Spline evaluation wrappers (for Hubble function and scalar field)
    //========================================================================
    double HoverH0_of_a(double a) const override { return std::exp(logE_of_loga_spline(std::log(a))); }
    double dlogHdloga_of_a(double a) const override { return logE_of_loga_spline.deriv_x(std::log(a)); }
    double phi_of_a(double a) { return std::exp(logphi_of_loga_spline(std::log(a))); } // normalized to 1/phi_today = GeffG_today
    double dlogphi_dloga_of_a(double a) { return logphi_of_loga_spline.deriv_x(std::log(a)); }

  protected:
    //========================================================================
    // Parameters specific to the JBD model
    //========================================================================
    double wBD; // independent
    double GeffG_today; // independent
    double phi_ini; // dependent on GeffG_today

    //========================================================================
    // Splines for the Hubble function (E = H/H0) and JBD scalar field phi
    //========================================================================
    Spline logE_of_loga_spline;
    Spline logphi_of_loga_spline;

    // Scale factor logarithm range to integrate/spline over
    const double loga_min = -10.0; // from deep in radiation era (when phi is close to constant)
    const double loga_max = 0.0;   // till today
    const int nloga = 1000;
};
#endif
