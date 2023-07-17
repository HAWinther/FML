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

        for (int iter = 0; iter < 1000; iter++) {
            // Refine guesses for phi_ini (from bisection limits) and OmegaLambda (from closure condition E0 == 1)
            double phi_ini = (phi_ini_lo + phi_ini_hi) / 2.0;
            double logphi_ini = std::log(phi_ini);
            OmegaLambda = 1.0 - OmegaM - OmegaRtot - OmegaK - OmegaPhi; // equivalent to E0 == 1

            // Solve cosmology for current (phi_ini, OmegaLambda) and record phi and its derivative today
            init(logphi_ini, OmegaLambda);
            double phi_today = std::exp(logphi_of_loga_spline(0.0)); // TODO: use get_phi
            double dlogphi_dloga_today = logphi_of_loga_spline.deriv_x(0.0); // TODO: create and use get_phi_derivative or something
            OmegaPhi = -dlogphi_dloga_today + wBD/6 * dlogphi_dloga_today * dlogphi_dloga_today; // defined so sum_i Omega_i == 1

            // Check for convergence (phi_today == phi_today_target and E0 == 1) TODO: generalize to G/G != 1
            bool converged_phi_today = std::fabs(phi_today / phi_today_target - 1.0) < 1e-8;
            bool converged_E_today   = std::fabs(HoverH0_of_a(1.0)            - 1.0) < 1e-8;
            if (converged_phi_today && converged_E_today) {
                std::cout << "JBD::init Convergence of solution found after " << iter << " iterations:\n";
                std::cout << "          Found phi_ini = " << std::exp(logphi_ini) << "\n";
                std::cout << "          Found OmegaLambda = " << OmegaLambda << "\n";
                return; // the cosmology is now initialized and ready-to-use
            }

            // Bisection step
            if (phi_today < phi_today_target) {
                phi_ini_lo = phi_ini; // underhit, so increase next guess
            } else {
                phi_ini_hi = phi_ini; //  overhit, so decrease next guess
            }
        }

        throw std::runtime_error("JBD::init Failed to converge");
    };

    //========================================================================
    // Initialize cosmology (with particular phi_ini and OmegaLambda)
    //========================================================================
    void init(double logphi_ini, double OmegaLambda) {
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
        auto logphi_arr = solve_ode(logphi_ini, 0.0);
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
            std::cout << "#=====================================================\n";
            std::cout << "\n";
        }
    }

    //========================================================================
    // Hubble function
    //========================================================================
    double HoverH0_of_a(double a) const override { return std::exp(logE_of_loga_spline(std::log(a))); }
    double dlogHdloga_of_a(double a) const override { return logE_of_loga_spline.deriv_x(std::log(a)); }

    // The JBD scalar normalized such that 1/phi_today = GeffG_today
    double get_phi(double a) { return std::exp(logphi_of_loga_spline(std::log(a))); }

  protected:
    //========================================================================
    // Parameters specific to the JBD model
    //========================================================================
    double wBD;
    double GeffG_today;

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

