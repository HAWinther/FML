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
        Omegabh2 = param.get<double>("cosmology_JBD_Omegabh2");
        OmegaMNuh2 = param.get<double>("cosmology_JBD_OmegaMNuh2");
        OmegaKh2 = param.get<double>("cosmology_JBD_OmegaKh2");
        OmegaCDMh2 = param.get<double>("cosmology_JBD_OmegaCDMh2");
        OmegaLambdah2 = param.get<double>("cosmology_JBD_OmegaLambdah2");
        wBD = param.get<double>("cosmology_JBD_wBD");
        GeffG_today = param.get<double>("cosmology_JBD_GeffG_today");

        // We have computed OmegaR/Nu in the base class so convert to physical parameters
        // (and it does not matter what value of h we used for these as long as we use the same here)
        // Compute photon density parameter
        OmegaRh2 = OmegaR * h * h;
        // Neutrino density parameter
        OmegaNuh2 = OmegaNu * h * h;
        // Convert computed neutrino mass to the value we want
        Mnu_eV = Mnu_eV * (OmegaNu / OmegaMNu) * (OmegaMNuh2 / OmegaNuh2);
    }

    //========================================================================
    // Initialize the cosmology
    //========================================================================
    void init() override {
        Cosmology::init();

        // Start deep in the radiation era
        // Set up array such thay a=1.0 is at i = npts_loga-1
        const double loga_ini = std::log(alow);
        const double loga_end = std::log(ahigh);
        const double dloga = (std::log(1.0) - loga_ini) / double(npts_loga - 1);
        DVector loga_arr;
        for (int i = 0;; i++) {
            loga_arr.push_back(loga_ini + i * dloga);
            if (loga_arr.back() > loga_end)
                break;
        }

        // H as function of loga, y = logphi and z = H a^3 e^y dy/dx. phi is here phi/phi0
        auto JBD_HubbleFunction_of_z = [&](double loga, double logphi, double z) {
            const double a = std::exp(loga);
            return std::sqrt(exp(-logphi) *
                                 (OmegaKh2 / (a * a) + OmegaRh2 / (a * a * a * a) + this->get_rhoNu_exact(a) * h * h +
                                  (Omegabh2 + OmegaCDMh2) / (a * a * a) + OmegaLambdah2) +
                             z * z / 12.0 * (2.0 * wBD + 3.0) * exp(-2 * logphi - 6 * loga)) -
                   z / 2.0 * exp(-logphi - 3 * loga);
        };

        // H as function of loga, logphi and dlogphidloga. phi is here phi/phi0
        auto JBD_HubbleFunction_of_dy = [&](double loga, double logphi, double dlogphidloga) {
            const double a = std::exp(loga);
            return std::sqrt(std::exp(-logphi) *
                             (OmegaRh2 / (a * a * a * a) + this->get_rhoNu_exact(a) * h * h +
                              (Omegabh2 + OmegaCDMh2) / (a * a * a) + OmegaLambdah2) /
                             (1.0 + dlogphidloga - wBD / 6 * dlogphidloga * dlogphidloga));
        };

        // Solve the ODE for a given initial value
        auto solve_ode = [&](double _logphi_ini, double _dlogphidloga_ini) {
            // The neutrino treatment here is not 100%, but good enough
            // Should correct this at some point though
            FML::SOLVERS::ODESOLVER::ODEFunction deriv = [&](double loga, const double * y, double * dydx) {
                double H = JBD_HubbleFunction_of_z(loga, y[0], y[1]);
                dydx[0] = y[1] * std::exp(-y[0] - 3 * loga) / H;
                dydx[1] = 3.0 * std::exp(3.0 * loga) / (2 * wBD + 3) / H *
                          ((Omegabh2 + OmegaCDMh2 + OmegaMNu) * std::exp(-3 * loga) + 4.0 * OmegaLambdah2);
                return GSL_SUCCESS;
            };
            FML::SOLVERS::ODESOLVER::ODESolver ode;
            const double z_ini = JBD_HubbleFunction_of_dy(loga_ini, _logphi_ini, _dlogphidloga_ini) *
                                 std::exp(3.0 * loga_ini + _logphi_ini) * _dlogphidloga_ini;
            DVector ini{_logphi_ini, z_ini};
            ode.solve(deriv, loga_arr, ini);
            return ode.get_data_by_component(0);
        };

        auto find_correct_IC_using_bisection = [&](double _desired_phi_today_over_phi0) {
            // Find the correct initial condition
            const double epsilon{1e-8};
            double philow = 0.0;
            double phihigh = _desired_phi_today_over_phi0;
            double _logphi_ini{};
            int istep = 0;
            while (istep < 1000) {
                ++istep;

                // Current value for phi
                const double phinow = (philow + phihigh) / 2.0;
                _logphi_ini = std::log(phinow);

                // Solve ODE
                const auto logphi_arr = solve_ode(_logphi_ini, 0.0);
                const double logphi_today = logphi_arr[npts_loga - 1];
                FML::assert_mpi(std::fabs(loga_arr[npts_loga - 1] - 0.0) < 1e-5,
                                "We assume below that the element loga_arr[npts_loga-1] is 0.0... its not!");

                // Check for convergence
                const double phi_today_over_phi0 = std::exp(logphi_today);
                if (std::fabs(phi_today_over_phi0 / _desired_phi_today_over_phi0 - 1.0) < epsilon) {
                    std::cout << "JBD::init Convergence of solution found after " << istep << " iterations\n";
                    return _logphi_ini;
                }

                // Bisection step
                if (phi_today_over_phi0 < _desired_phi_today_over_phi0) {
                    philow = phinow;
                } else {
                    phihigh = phinow;
                }
            }
            throw std::runtime_error("JBD::init Failed to converge");
            return _logphi_ini;
        };

        // The value we shoot for
        const double desired_phi_today_over_phi0 = 1.0 / GeffG_today;

        // Find the initial value that gives this value today
        const double logphi_ini = find_correct_IC_using_bisection(desired_phi_today_over_phi0);

        // Solve the ODE for this initial value and make splines
        auto logphi_arr = solve_ode(logphi_ini, 0.0);
        auto logh_of_loga_arr = loga_arr;
        logphi_of_loga_spline.create(loga_arr, logphi_arr, "JBD logphi(loga)");
        for (auto & loga : logh_of_loga_arr) {
            loga = std::log(
                JBD_HubbleFunction_of_dy(loga, logphi_of_loga_spline(loga), logphi_of_loga_spline.deriv_x(loga)));
        }
        loghubble_of_loga_spline.create(loga_arr, logh_of_loga_arr, "JBD logH(loga)");

        // Compute the value of 'h'
        h = std::exp(loghubble_of_loga_spline(0.0));

        // Make sure the spline has H(0) == 1
        for (auto & logH : logh_of_loga_arr) {
            logH -= std::log(h);
        }
        loghubble_of_loga_spline.create(loga_arr, logh_of_loga_arr, "JBD logH(loga)");

        // Set cosmological parameters
        OmegaR = OmegaRh2 / (h * h);
        OmegaNu = OmegaNuh2 / (h * h);
        OmegaK = OmegaKh2 / (h * h);
        OmegaMNu = OmegaMNuh2 / (h * h);
        OmegaCDM = OmegaCDMh2 / (h * h);
        Omegab = Omegabh2 / (h * h);
        OmegaLambda = OmegaLambda / (h * h);
        OmegaRtot = OmegaR + OmegaNu;
        OmegaM = Omegab + OmegaCDM + OmegaMNu;

        if (FML::ThisTask == 0) {
            std::cout << "JBD::init Found phi_ini = " << std::exp(logphi_ini) << "\n";
            std::cout << "          We have h = " << h << "\n";
            std::cout << "          Testing spline H(a=1)/H0 = " << HoverH0_of_a(1.0) << "\n";
            FML::assert_mpi(std::fabs(HoverH0_of_a(1.0) - 1.0) < 1e-5,
                            "JBD H(a=1)/H0 is not unity. Something went wrong");
        }
    }

    //========================================================================
    // Print some info
    //========================================================================
    void info() const override {
        Cosmology::info();
        if (FML::ThisTask == 0) {
            std::cout << "# wBD           : " << wBD << "\n";
            std::cout << "# Omegabh2      : " << Omegabh2 << "\n";
            std::cout << "# OmegaCDMh2    : " << OmegaCDMh2 << "\n";
            std::cout << "# OmegaMNuh2    : " << OmegaMNuh2 << "\n";
            std::cout << "# OmegaKh2      : " << OmegaKh2 << "\n";
            std::cout << "# OmegaLambdah2 : " << OmegaLambdah2 << "\n";
            std::cout << "# GeffG_today   : " << GeffG_today << "\n";
            std::cout << "#=====================================================\n";
            std::cout << "\n";
        }
    }

    //========================================================================
    // Hubble function
    //========================================================================
    double HoverH0_of_a(double a) const override { return std::exp(loghubble_of_loga_spline(std::log(a))); }
    double dlogHdloga_of_a(double a) const override { return loghubble_of_loga_spline.deriv_x(std::log(a)); }

    // The JBD scalar normalized such that 1/phi_today = GeffG_today
    double get_phi(double a) { return std::exp(logphi_of_loga_spline(std::log(a))); }

  protected:
    //========================================================================
    // Parameters specific to the JBD model
    //========================================================================
    double wBD;
    double GeffG_today;

    //========================================================================
    // We don't know 'h' a priori so we must take in physical parameters
    // and derive the usual parameters
    //========================================================================
    double OmegaRh2;
    double OmegaNuh2;
    double Omegabh2;
    double OmegaMNuh2;
    double OmegaCDMh2;
    double OmegaKh2;
    double OmegaLambdah2;

    //========================================================================
    // Splines for the hubble function and for the JBD scalar phi
    //========================================================================
    Spline loghubble_of_loga_spline;
    Spline logphi_of_loga_spline;

    // For the solving and splines of phi
    // We integrate from deep inside the radiation era
    const int npts_loga = 500;
    const double alow = 1e-6;
    const double ahigh = 2.0;
};
#endif

