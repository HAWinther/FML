#ifndef COSMOLOGY_JBD_HEADER
#define COSMOLOGY_JBD_HEADER

#include <FML/FFTWGrid/FFTWGrid.h>
#include <FML/Global/Global.h>
#include <FML/LPT/DisplacementFields.h>
#include <FML/ODESolver/ODESolver.h>
#include <FML/ParameterMap/ParameterMap.h>
#include <FML/Spline/Spline.h>

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
        double OmegaPhi = 0.0; // bootstrap the first guess for OmegaLambda in the loop

        std::cout << "JBD::init Searching for phi_ini and OmegaLambda that gives "
                  << "phi_today = " << phi_today_target << " and "
                  << "E_today = (H/H0)_today = 1.0:" << "\n";

        for (int iter = 1; iter < 100; iter++) {
            // Refine guesses for phi_ini (from bisection limits) and OmegaLambda (from closure condition E0 == 1)
            phi_ini = (phi_ini_lo + phi_ini_hi) / 2.0;
            OmegaLambda = 1.0 - OmegaM - OmegaRtot - OmegaK - OmegaPhi; // equivalent to E0 == 1

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

            // Improve next guesses for phi_ini and OmegaLambda
            if (phi_of_a(1.0) < phi_today_target) {
                phi_ini_lo = phi_ini; // underhit, so increase next guess
            } else {
                phi_ini_hi = phi_ini; //  overhit, so decrease next guess
            }
            OmegaPhi = -dlogphi_dloga_of_a(1.0) + wBD/6 * dlogphi_dloga_of_a(1.0) * dlogphi_dloga_of_a(1.0); // defined from E0 == 1, so sum_i Omega_i == 1
        }

        throw std::runtime_error("JBD::init Search for phi_ini and OmegaLambda did not converge");
    };

    //========================================================================
    // Initialize cosmology (with current values of phi_ini and OmegaLambda)
    //========================================================================
    void init_current() {
        Cosmology::init();

        // Linearly spaced scale factor logarithms to integrate over
        const double dloga = (loga_max - loga_min) / double(nloga - 1);
        DVector loga_arr(nloga);
        for (int i = 0; i < nloga; i++) {
            loga_arr[i] = loga_min + i * dloga;
        }

        // E = H/H0 as function of loga, y = logphi and z = E a^3 e^y dy/dx. phi is here phi/phi0
        auto JBD_E_of_z = [&](double loga, double logphi, double z) {
            const double a = std::exp(loga);
            return std::sqrt(exp(-logphi) *
                                 (OmegaK / (a * a) + OmegaR / (a * a * a * a) + this->get_rhoNu_exact(a) + // TODO: phi * Omegak
                                  (Omegab + OmegaCDM) / (a * a * a) + OmegaLambda) + // TODO: OmegaM?
                             z * z / 12.0 * (2.0 * wBD + 3.0) * exp(-2 * logphi - 6 * loga)) -
                   z / 2.0 * exp(-logphi - 3 * loga);
        };

        // H as function of loga, logphi and dlogphidloga. phi is here phi/phi0
        auto JBD_E_of_dy = [&](double loga, double logphi, double dlogphidloga) {
            const double a = std::exp(loga);
            return std::sqrt(std::exp(-logphi) *
                             (OmegaR / (a * a * a * a) + this->get_rhoNu_exact(a) + // TODO: phi * Omegak
                              (Omegab + OmegaCDM) / (a * a * a) + OmegaLambda) / // TODO: OmegaM?
                             (1.0 + dlogphidloga - wBD / 6 * dlogphidloga * dlogphidloga));
        };

        // Solve the ODE for a given initial value
        auto solve_ode = [&](double _logphi_ini, double _dlogphidloga_ini) {
            // The neutrino treatment here is not 100%, but good enough
            // Should correct this at some point though
            FML::SOLVERS::ODESOLVER::ODEFunction deriv = [&](double loga, const double * y, double * dydx) {
                double E = JBD_E_of_z(loga, y[0], y[1]);
                dydx[0] = y[1] * std::exp(-y[0] - 3 * loga) / E;
                dydx[1] = 3.0 * std::exp(3.0 * loga) / (2 * wBD + 3) / E *
                          ((Omegab + OmegaCDM + OmegaMNu) * std::exp(-3 * loga) + 4.0 * OmegaLambda); // TODO: OmegaM?
                return GSL_SUCCESS;
            };
            FML::SOLVERS::ODESOLVER::ODESolver ode;
            const double loga_ini = loga_min;
            const double z_ini = JBD_E_of_dy(loga_ini, _logphi_ini, _dlogphidloga_ini) *
                                 std::exp(3.0 * loga_ini + _logphi_ini) * _dlogphidloga_ini;
            DVector ini{_logphi_ini, z_ini};
            ode.solve(deriv, loga_arr, ini);
            return ode.get_data_by_component(0);
        };

        // Solve the ODE for this initial value and spline logphi(loga) and logE(loga)
        auto logphi_arr = solve_ode(std::log(phi_ini), 0.0);
        logphi_of_loga_spline.create(loga_arr, logphi_arr, "JBD logphi(loga)");

        auto logE_of_loga_arr = loga_arr; // copy to create vector of same size (preserving loga)
        for (auto & logE : logE_of_loga_arr) {
            double loga = logE; // because we copied the vector above
            logE = std::log(JBD_E_of_dy(loga, logphi_of_loga_spline(loga), logphi_of_loga_spline.deriv_x(loga)));
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

