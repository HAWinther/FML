#include "RecombinationHistory.h"

namespace FML {
    namespace COSMOLOGY {

        //====================================================
        // Constructors
        //====================================================

        RecombinationHistory::RecombinationHistory(std::shared_ptr<BackgroundCosmology> cosmo, ParameterMap & p)
            : cosmo(cosmo) {
            Yp = p.get<double>("Yp");
            reionization = p.get<bool>("reionization");
            userecfast = p.get<bool>("userecfast");
            rec_fudge_factor = p.get<double>("RecFudgeFactor");
            x_start_rec_array = std::min(p.get<double>("pert_x_initial"), -20.0);

            if (reionization) {
                z_reion = p.get<double>("z_reion");
                delta_z_reion = p.get<double>("delta_z_reion");
                helium_reionization = p.get<bool>("helium_reionization");
                if (helium_reionization) {
                    z_helium_reion = p.get<double>("z_helium_reion");
                    delta_z_helium_reion = p.get<double>("delta_z_helium_reion");
                }
            }
        }

        //====================================================
        // Class methods
        //====================================================

        void RecombinationHistory::info() const {
            std::cout << "\n";
            std::cout << "============================================\n";
            std::cout << "Info about recombination/reionization history class:\n";
            std::cout << "============================================\n";
            std::cout << "Using the cosmology  [" << cosmo->get_name() << "]\n";
            std::cout << "Yp:                   " << Yp << "\n";
            std::cout << "Reionization:         " << (reionization ? "true" : "false") << "\n";
            std::cout << "z_reion:              " << z_reion << "\n";
            std::cout << "delta_z_reion:        " << delta_z_reion << "\n";
            std::cout << "Helium doubly reion:  " << (helium_reionization ? "true" : "false") << "\n";
            std::cout << "z_helium_reion:       " << z_helium_reion << "\n";
            std::cout << "delta_z_helium_reion: " << delta_z_helium_reion << "\n";

            if (tau_of_x_spline) {
                double r_star = get_sound_horizon(x_star);
                double r_drag = get_sound_horizon(x_drag);
                double DA = cosmo->dA_of_x(x_star) * std::exp(-x_star);
                double theta_star = r_star / DA;
                double tau_reion = reionization ? tau_of_x_spline(std::log(1.0 / (1.0 + z_reion))) : 0.0;
                double kd_star = kd_of_x_spline(x_star);
                std::cout << "Recombination (Xe = 0.5)           z       = " << std::exp(-x_recombination) - 1 << "\n";
                std::cout << "Recombination (Xe = 0.5) with Saha z       = " << std::exp(-x_recombination_saha) - 1 << "\n";
                std::cout << "Last scattering (tau = 1.0) with Saha      = " << std::exp(-x_star_saha) - 1
                          << " Xe: " << Xe_of_x_saha(x_star_saha) << "\n";
                std::cout << "Last scattering (dgdx = 0.0)       z       = " << std::exp(-x_star2) - 1 << "\n";
                std::cout << "Last scattering (tau = 1.0)        zstar   = " << std::exp(-x_star) - 1
                          << " Xe: " << Xe_of_x(x_star) << "\n";
                std::cout << "                                   xstar   = " << x_star << "\n";
                std::cout << "Sound horizon at decoupling        r_star  = " << r_star / Constants.Mpc << " Mpc\n";
                std::cout << "                           100 theta_star  = " << 100.0 * theta_star << "\n";
                std::cout << "                                  DA_star  = " << DA / Constants.Gpc << " Gpc\n";
                std::cout << "Drag epoch (tau_baryon = 1.0)      zdrag   = " << std::exp(-x_drag) - 1 << "\n";
                std::cout << "Sound horizon at drag-epoch        r_zdrag = " << r_drag / Constants.Mpc << " Mpc\n";
                std::cout << "Optical depth reionization             tau = " << tau_reion << "\n";
                std::cout << "Damping scale at LSS:                  kD  = " << kd_star * Constants.Mpc << " 1/Mpc\n";
            }
            std::cout << "============================================\n";
            std::cout << "\n";
        }

        // Solve the Saha equation for H+ to get electon fraction
        std::pair<double, double>
        RecombinationHistory::electron_fraction_from_saha_equation_without_helium(double x) const {
            const double a = std::exp(x);

            // Physical constants
            const double k_b = Constants.k_b;
            const double G = Constants.G;
            const double m_e = Constants.m_e;
            const double hbar = Constants.hbar;
            const double m_H = Constants.m_H;
            const double epsilon_0 = Constants.epsilon_0;
            const double H0_over_h = Constants.H0_over_h;

            // Cosmological parameters
            const double OmegaB = cosmo->get_OmegaB();
            const double h = cosmo->get_h();
            const double TCMB_of_x = cosmo->get_TCMB(x);
            const double H0 = h * H0_over_h;

            // Baryon temperatur and number density
            const double kT_b = k_b * TCMB_of_x;
            const double n_b = (OmegaB * 3.0 * H0 * H0) / (8 * M_PI * G * m_H * a * a * a);

            // The rhs of Saha equation
            const double saha_factor = std::pow(m_e * kT_b / (hbar * hbar * 2.0 * M_PI), 1.5) / n_b * std::exp(-epsilon_0 / kT_b);

            // The equation reads X_e = saha_fac/2.0 * ( -1 + sqrt(1.0 + 4.0/saha_fac) )
            double f_e = 1.0;
            if (4.0 / saha_factor < 1e-5) {
                f_e = 1.0 - 1.0 / saha_factor;
            } else {
                f_e = saha_factor / 2.0 * (-1.0 + sqrt(1.0 + 4.0 / saha_factor));
            }

            // Just in case we want to compute this with an extremely early starting x
            if (a < 1e-10)
                f_e = 1.0;

            // Return electron fraction and number density
            const double Xe = f_e;
            const double ne = f_e * n_b;

            return std::pair<double, double>(Xe, ne);
        }

        // Solve the Saha equations for He+, He++ and H+ to get electron fraction
        std::pair<double, double>
        RecombinationHistory::electron_fraction_from_saha_equation_with_helium(double x) const {
            const double a = std::exp(x);

            // Physical constants
            const double k_b = Constants.k_b;
            const double G = Constants.G;
            const double m_e = Constants.m_e;
            const double hbar = Constants.hbar;
            const double m_H = Constants.m_H;
            const double epsilon_0 = Constants.epsilon_0;
            const double H0_over_h = Constants.H0_over_h;
            const double xhi0 = Constants.xhi0;
            const double xhi1 = Constants.xhi1;

            // Cosmological parameters
            const double OmegaB = cosmo->get_OmegaB();
            const double h = cosmo->get_h();
            const double TCMB_of_x = cosmo->get_TCMB(x);
            const double H0 = h * H0_over_h;

            // To make the code run with OmegaB = 0
            if (OmegaB == 0.0) {
                return std::pair<double, double>(1.0, 1e-100);
            }

            // Baryon temperatur and number density
            const double kT_b = k_b * TCMB_of_x;
            const double n_b = (OmegaB * 3.0 * H0 * H0) / (8 * M_PI * G * m_H * a * a * a);

            // Prefactor for rhs of Saha equations
            const double saha_factor = std::pow(m_e * kT_b / (hbar * hbar * 2.0 * M_PI), 1.5) / n_b;

            // No point of solving if the result is basically 0
            double factor = sqrt(saha_factor * std::exp(-epsilon_0 / kT_b) / (1.0 - Yp));
            if (factor < 1e-5)
                return {factor, (1.0 - Yp) * factor * n_b};

            // Iterative method for finding x_He_plus, x_He_plusplus, x_H_plus
            double f_e = 1.0;
            double f_e_old = 0.0;
            while (abs(f_e - f_e_old) > 1e-10) {
                // R.h.s of the Saha equations for He++, He+ and H+ respectivily
                const double rhs_x_He_plus = 2.0 * saha_factor * std::exp(-xhi0 / kT_b) / f_e;
                const double rhs_x_He_plusplus = 4.0 * saha_factor * std::exp(-xhi1 / kT_b) / f_e;
                const double rhs_x_H_plus = saha_factor * std::exp(-epsilon_0 / kT_b) / f_e;

                // Abundances of He++, He+ and H+ respectivily
                const double x_He_plus = rhs_x_He_plus / (1.0 + rhs_x_He_plus + rhs_x_He_plus * rhs_x_He_plusplus);
                const double x_He_plusplus = rhs_x_He_plusplus * x_He_plus;
                const double x_H_plus = rhs_x_H_plus / (1.0 + rhs_x_H_plus);

                // Calculate new value of f_e
                f_e_old = f_e;
                f_e = (2.0 * x_He_plusplus + x_He_plus) * Yp / 4.0 + (1.0 - Yp) * x_H_plus;
            }

            // Just in case we want to compute this with an extremely early starting x
            if (a < 1e-10)
                f_e = 1.0;

            // Return electron fraction and number density
            const double Xe = f_e / (1.0 - Yp);
            const double ne = f_e * n_b;

            return {Xe, ne};
        }

        void RecombinationHistory::solve() {
            solve_number_density_electrons();
            solve_for_optical_depth_tau();
            solve_extra();
        }

#ifdef USE_RECFAST
        // Here we run RECFAST and return Xe and Tb
        std::pair<DVector, DVector> RecombinationHistory::run_recfast(DVector & x_array) {
            CosmoPars RecfastCosmo;
            RecfastCosmo.Yp = Yp;
            RecfastCosmo.T0 = cosmo->get_TCMB();
            RecfastCosmo.Omega_m = cosmo->get_OmegaM();
            RecfastCosmo.Omega_b = cosmo->get_OmegaB();
            RecfastCosmo.Omega_k = cosmo->get_OmegaK();
            RecfastCosmo.Omega_L = cosmo->get_OmegaLambda();
            RecfastCosmo.h100 = cosmo->get_h();
            RecfastCosmo.Neff = cosmo->get_Neff();

            RecPars RecfastRec;
            RecfastRec.verbosity = false;
            RecfastRec.F = rec_fudge_factor;
            RecfastRec.npts = 50000;
            RecfastRec.zstart = 1e4;

            std::cout << "\n============================================\n";
            std::cout << "Calling Recfast                               \n";
            std::cout << "============================================\n\n";
            RecfastCosmo.show();
            RecfastRec.show();

            // What is returned from recfast...
            std::vector<double> zarr_RF, Xe_H_RF, Xe_He_RF, Xe_RF, TM_RF;

            // Make the call and fill the arrays
            Xe_frac(RecfastCosmo, RecfastRec, zarr_RF, Xe_H_RF, Xe_He_RF, Xe_RF, TM_RF);

            // Spline the results
            DVector xarr_RF(zarr_RF.size());
            for (size_t i = 0; i < zarr_RF.size(); i++)
                xarr_RF[i] = std::log(1.0 / (1.0 + zarr_RF[i]));
            Spline Xe_tmpspline(xarr_RF, Xe_RF, "Xe - Recfast");
            Spline Tb_tmpspline(xarr_RF, TM_RF, "Tb - Recfast");

            // Use the splines to fill the x_array we want the data on
            DVector Tb_array(x_array.size());
            DVector Xe_array(x_array.size());
            for (size_t i = 0; i < Xe_array.size(); i++) {
                const double x = x_array[i];
                double Tb = Tb_tmpspline(x);
                double Xe = Xe_tmpspline(x);
                if (x < xarr_RF[0]) {
                    Tb = Tb_tmpspline(xarr_RF[0]) * std::exp(xarr_RF[0] - x);
                    Xe = Xe_tmpspline(xarr_RF[0]);
                }
                Xe_array[i] = Xe;
                Tb_array[i] = Tb;
            }
            return {Xe_array, Tb_array};
        }
#endif

        // Solve for X_e and n_e using the Saha and Peebles equation and spline the result
        void RecombinationHistory::solve_number_density_electrons() {

            // Settings for the arrays we use below
            const int npts = npts_Xe_array;
            const double x_start = x_start_rec_array;
            const double x_end = x_end_rec_array;
            DVector x_array = FML::MATH::linspace(x_start, x_end, npts);

            // Set up arrays to compute X_e and n_e on
            DVector Xe_array(npts);
            DVector ne_array(npts);
            DVector Xe_saha_arr(npts);
            DVector Tb_array(npts);
            DVector cs2_baryon_array(npts);

            // Saha calculation
            for (int i = 0; i < npts; i++) {
                //==============================================================
                // Electron fraction and number density from Saha equation
                //==============================================================
                auto Xe_ne_data = electron_fraction_from_saha_equation_with_helium(x_array[i]);
                Xe_saha_arr[i] = Xe_ne_data.first;
                ne_array[i] = Xe_ne_data.second;
                Tb_array[i] = cosmo->get_TCMB(x_array[i]);
            }

            // Calculate recombination history
            for (int i = 0; i < npts; i++) {
                auto Xe_current = Xe_saha_arr[i];

                // Two regimes: Saha and Peebles regime
                if (Xe_saha_arr[i] >= Xe_saha_limit) {
                    Xe_array[i] = Xe_current;
                } else {

                    //==============================================================
                    // We need to solve the Peebles equation for the rest of the time
                    //==============================================================

                    // Constants and physical parameters needed below
                    const double m_H = Constants.m_H;
                    const double G = Constants.G;
                    const double OmegaB = cosmo->get_OmegaB();
                    const double H0 = cosmo->get_H0();
                    const double n_b0 = (OmegaB * 3.0 * H0 * H0) / (8 * M_PI * G * m_H);

                    // Make x-array for Peebles system from current time till the end
                    const auto first = x_array.begin() + i;
                    const auto last = x_array.end();
                    DVector x_array_peebles(first, last);

                    // The Peebles ODE equation
                    ODEFunction deriv = [&](double x, const double * y, double * dydx) {
                        return rhs_peebles_ode(x, y, dydx);
                    };

                    // Set initial conditions for { X_e, (T_b/T_gamma-1) }
                    DVector Xe_initial{Xe_current, 0.0};

                    // Solve the Peebles ODE
                    ODESolver peebles_Xe_ode(
                        FIDUCIAL_HSTART_ODE_PEEBLES, FIDUCIAL_ABSERR_ODE_PEEBLES, FIDUCIAL_RELERR_ODE_PEEBLES);
                    peebles_Xe_ode.solve(deriv, x_array_peebles, Xe_initial);

                    // Fetch the Xe solution
                    const auto data = peebles_Xe_ode.get_data_by_component(0);

                    // Fill up array with the result
                    for (int j = i; j < npts; j++) {
                        const double X_e = data[j - i];
                        Xe_array[j] = X_e;
                        ne_array[j] = (1.0 - Yp) * X_e * (n_b0 * std::exp(-3 * x_array[j]));
                    }

                    // Fetch the computed baryon temperature
                    const auto baryon_temp = peebles_Xe_ode.get_data_by_component(1);
                    for (int j = 0; j < npts; j++) {
                        double Tb_of_x = cosmo->get_TCMB(x_array[j]);
                        if (j >= i) {
                            Tb_of_x *= (1.0 + baryon_temp[j - i]);
                        }
                        Tb_array[j] = Tb_of_x;
                    }

                    // We are done so exit for loop
                    break;
                }
            }

#ifdef USE_RECFAST
            // Run RECFAST to get Xe and Tb and overwrite the resulrs we already have
            if (userecfast) {
                auto recfastdata = run_recfast(x_array);
                Xe_array = recfastdata.first;
                Tb_array = recfastdata.second;
            }
#endif

            // Spline baryon temperature
            Tb_spline.create(x_array, Tb_array, "Temp_baryon_of_x");

            // Make the baryon sound speed (cs/c)^2
            // The mean moleculary weight - we just use m_H here for simplicity
            // Add in the effects of reionization?
            for (int j = 0; j < npts; j++) {
                cs2_baryon_array[j] = (Constants.k_b * Tb_spline(x_array[j])) /
                                      (Constants.m_H * Constants.c * Constants.c) *
                                      (1.0 - Tb_spline.deriv_x(x_array[j]) / Tb_spline(x_array[j]) / 3.0);
            }
            cs2_baryon_spline.create(x_array, cs2_baryon_array, "cs2_baryon");

            // Spline up electron fractions
            Xe_of_x_spline.create(x_array, Xe_array, "Xe_of_x");
            Xe_of_x_saha_spline.create(x_array, Xe_saha_arr, "Xe_of_x_saha");

            // Find recombination redshift (NB: this has to be the Xe without reionization for the binary search to
            // work)
            try {
                x_recombination = FML::MATH::find_root_bisection(Xe_of_x_spline, 0.5);
                x_recombination_saha = FML::MATH::find_root_bisection(Xe_of_x_saha_spline, 0.5);
            } catch (...) {
                std::cout << "Error in computing x_recombination and/or x_recombination_saha\n";
                x_recombination = 0.0;
                x_recombination_saha = 0.0;
            }
        }

        int RecombinationHistory::rhs_peebles_ode(double x, const double * y, double * dydx) {
            const double X_e = y[0];
            const double a = std::exp(x);

            // Physical constants in SI units
            const double k_b = Constants.k_b;
            const double G = Constants.G;
            const double c = Constants.c;
            const double m_e = Constants.m_e;
            const double hbar = Constants.hbar;
            const double m_H = Constants.m_H;
            const double sigma_T = Constants.sigma_T;
            const double lambda_2s1s = Constants.lambda_2s1s;
            const double epsilon_0 = Constants.epsilon_0;

            // Cosmological and recombination parameters
            const double OmegaB = cosmo->get_OmegaB();
            const double TCMB_of_x = cosmo->get_TCMB(x);
            const double H0 = cosmo->get_H0();
            const double H = cosmo->H_of_x(x);
            const double Yp = get_Yp();

            // To be able to run the whole code without baryons
            if (OmegaB == 0.0) {
                dydx[0] = 0.0;
                dydx[1] = 0.0;
                return GSL_SUCCESS;
            }

            // Baryon temperature and number density
            const double n_b = (3.0 * OmegaB * H0 * H0) / (8 * M_PI * G * m_H * a * a * a);
            const double n_H = (1.0 - Yp) * n_b;

            // Here we use the evolved baryon temperature. If we don't evolve this then
            // an approx we can use is to set y = 0 below
            const double kT_b = k_b * TCMB_of_x * (1.0 + y[1]);

            // Factors in the Peebles equation
            const double eps_over_kT = epsilon_0 / kT_b;
            const double exp_eps_over_kT = eps_over_kT < 200.0 ? std::exp(eps_over_kT) : std::exp(200.0);
            const double phi_2 = 0.448 * std::log(eps_over_kT);
            const double alpha_2 = phi_2 * (8.0 / sqrt(3.0 * M_PI)) * sigma_T * sqrt(eps_over_kT);
            const double beta = alpha_2 * c * std::pow(m_e * kT_b / (2.0 * M_PI * hbar * hbar), 1.5) / exp_eps_over_kT;
            const double beta_2 = beta * std::pow(exp_eps_over_kT, 0.75);
            const double n1s = (1.0 - X_e) * n_H;
            const double lambda_a = H * std::pow(3.0 * epsilon_0 / (hbar * c), 3) / (64.0 * M_PI * M_PI * n1s);
            const double C_r = (lambda_2s1s + lambda_a) / (lambda_2s1s + lambda_a + beta_2);

            // Two different regimes to avoid numerical problems
            dydx[0] = C_r / H * (beta * (1.0 - X_e) - c * n_H * alpha_2 * X_e * X_e);

            // Evolve the baryon temperature ODE (y is Tb/Tgamma-1 so 0 when tightly coupled)
            const double R = 4.0 / 3.0 * cosmo->get_OmegaR() / OmegaB / a;
            const double dtaudx = c * sigma_T / H * n_b * X_e;
            dydx[1] = -y[1] - 1.0 - 2.0 * (Constants.m_H / Constants.m_e) * R * dtaudx * y[1];

            return GSL_SUCCESS;
        }

        void RecombinationHistory::output(const std::string filename) const {
            std::ofstream fp(filename.c_str());
            const int npts = 1000;
            const double x_start = x_start_rec_array;
            const double x_end = x_end_rec_array;
            DVector x_array = FML::MATH::linspace(x_start, x_end, npts);
            auto print_data = [&](const double x) {
                // 1
                fp << x << " ";

                // 2
                fp << Xe_of_x(x) << " ";
                fp << Xe_of_x_saha(x) << " ";

                // 4
                fp << ne_of_x(x) << " ";

                // 5
                fp << tau_of_x(x) << " ";
                fp << dtaudx_of_x(x) << " ";
                fp << ddtauddx_of_x(x) << " ";

                // 8
                fp << g_tilde_of_x(x) << " ";
                fp << dgdx_tilde_of_x(x) << " ";
                fp << ddgddx_tilde_of_x(x) << " ";

                // 10 in total
                fp << "\n";
            };
            std::for_each(x_array.begin(), x_array.end(), print_data);
        }

        // Solve for the optical depth tau, compute the visibility function and spline
        // the result
        void RecombinationHistory::solve_for_optical_depth_tau() {

            // Settings for the arrays we use below
            const int npts_before_reion = npts_tau_before_reion;
            const int npts_during_reion = npts_tau_during_reion;
            const int npts_after_reion = npts_tau_after_reion;
            const int npts = npts_before_reion + npts_during_reion + npts_after_reion - 2;
            const double x_start = x_start_rec_array;
            const double x_start_reion = std::log(1.0 / (1.0 + z_reion + 2 * delta_z_reion));
            const double x_end_reion = std::log(1.0 / (1.0 + z_reion - 2 * delta_z_reion));
            const double x_end = x_end_rec_array;

            // Set up x-arrays to integrate over. We split into three regions as we need
            // extra points in reionisation
            DVector x_array(npts);
            if (reionization) {
                DVector x_array_before_reion = FML::MATH::linspace(x_start, x_start_reion, npts_before_reion);
                DVector x_array_during_reion = FML::MATH::linspace(x_start_reion, x_end_reion, npts_during_reion);
                DVector x_array_after_reion = FML::MATH::linspace(x_end_reion, x_end, npts_after_reion);

                // Combine to one array. Last point in previous array equals first point in
                // new array so avoid duplications
                double * arr1 = &x_array[0];
                double * arr2 = &x_array[npts_before_reion - 1];
                double * arr3 = &x_array[npts_before_reion - 1 + npts_during_reion - 1];
                for (int i = 0; i < npts_before_reion; i++) {
                    arr1[i] = x_array_before_reion[i];
                }
                for (int i = 0; i < npts_during_reion; i++) {
                    arr2[i] = x_array_during_reion[i];
                }
                for (int i = 0; i < npts_after_reion; i++) {
                    arr3[i] = x_array_after_reion[i];
                }

                // We integrate backwards from x=0 so reverse the array
                std::reverse(x_array.begin(), x_array.end());
            } else {
                x_array = FML::MATH::linspace(x_end, x_start, npts);
            }

            // The ODE system dtau/dx, dtau_noreion/dx and dtau_baryon/dx
            ODEFunction deriv_tau = [&](double x, [[maybe_unused]] const double * y, double * dydx) {
                const double OmegaB = cosmo->get_OmegaB();

                // If we don't have baryons then tau = 0
                if (OmegaB == 0.0) {
                    dydx[0] = dydx[1] = dydx[2] = dydx[3] = dydx[4] = 0.0;
                    return GSL_SUCCESS;
                }

                const double c = Constants.c;
                const double sigma_T = Constants.sigma_T;
                const double a = std::exp(x);
                const double H = cosmo->H_of_x(x);
                const double n_e = ne_of_x(x);
                const double n_e_noreion = ne_of_x_noreion(x);
                const double R = 4.0 / 3.0 * cosmo->get_OmegaR() / OmegaB / a;

                // Set the derivative for photon optical depth with and without reionization
                dydx[0] = (-c * sigma_T * n_e / H);
                dydx[1] = (-c * sigma_T * n_e_noreion / H);

                // The baryon optical depth
                dydx[2] = -c * R * sigma_T * n_e_noreion / H;

                // Optical depth with Saha and with and without reionization
                const double n_e_saha = ne_of_x_saha(x);
                const double n_e_saha_noreion = ne_of_x_saha_noreion(x);
                dydx[3] = (-c * sigma_T * n_e_saha / H);
                dydx[4] = (-c * sigma_T * n_e_saha_noreion / H);

                return GSL_SUCCESS;
            };

            // Set initial conditions (3 components: with and without reion + baryons)
            DVector tau_initial(5, 0.0);

            // Solve the ODE
            ODESolver tau_ode(FIDUCIAL_HSTART_ODE_TAU, FIDUCIAL_ABSERR_ODE_TAU, FIDUCIAL_RELERR_ODE_TAU);
            tau_ode.solve(deriv_tau, x_array, tau_initial, gsl_odeiv2_step_rk4);

            // Fetch the solution
            auto tau_array = tau_ode.get_data_by_component(0);
            auto tau_noreion_array = tau_ode.get_data_by_component(1);
            auto tau_baryon_noreion_array = tau_ode.get_data_by_component(2);
            auto tau_saha_array = tau_ode.get_data_by_component(3);
            auto tau_saha_noreion_array = tau_ode.get_data_by_component(4);
            auto dtaudx_array = tau_ode.get_derivative_data_by_component(0);
            auto dtaudx_noreion_array = tau_ode.get_derivative_data_by_component(1);

            // Compute the visibility function
            DVector g_tilde_array(npts);
            DVector g_tilde_noreion_array(npts);
            for (int i = 0; i < npts; i++) {
                g_tilde_array[i] = -dtaudx_array[i] * std::exp(-tau_array[i]);
                g_tilde_noreion_array[i] = -dtaudx_noreion_array[i] * std::exp(-tau_noreion_array[i]);
            }

            // Make tau and gsplines
            tau_of_x_spline.create(x_array, tau_array, "tau_of_x");
            dtaudx_of_x_spline.create(x_array, dtaudx_array, "dtaudx_of_x");
            g_tilde_of_x_spline.create(x_array, g_tilde_array, "g_tilde_of_x");
            tau_of_x_noreion_spline.create(x_array, tau_noreion_array, "tau_of_x_noreion");
            g_tilde_of_x_noreion_spline.create(x_array, g_tilde_noreion_array, "g_tilde_of_x_noreion");
            tau_baryon_noreion_of_x_spline.create(x_array, tau_baryon_noreion_array, "tau_baryon_noreion_of_x");
            tau_of_x_saha_spline.create(x_array, tau_saha_array, "tau_of_x_saha");
            tau_of_x_saha_noreion_spline.create(x_array, tau_saha_noreion_array, "tau_of_x_saha_noreion");

            // Spline dgdx
            DVector dgdx_tilde_array(g_tilde_array);
            for (int i = 0; i < npts; i++) {
                double x = x_array[i];
                dgdx_tilde_array[i] = g_tilde_of_x_spline.deriv_x(x);
            }
            dgdx_tilde_of_x_spline.create(x_array, dgdx_tilde_array, "dgdx_tilde_of_x");

            // Spline ddgddx
            DVector ddgddx_tilde_array(g_tilde_array);
            for (int i = 0; i < npts; i++) {
                double x = x_array[i];
                ddgddx_tilde_array[i] = dgdx_tilde_of_x_spline.deriv_x(x);
            }
            ddgddx_tilde_of_x_spline.create(x_array, ddgddx_tilde_array, "ddgddx_tilde_of_x");
        }

        void RecombinationHistory::solve_extra() {
            const int npts = 1000;
            const double x_start = x_start_rec_array;
            const double x_end = x_end_rec_array;
            DVector x_array = FML::MATH::linspace(x_start, x_end, npts);

            // The sound horizon ODE dr_s/dx = cs(x) / Hp(x)
            ODEFunction deriv_sound = [&](double x, [[maybe_unused]] const double * y, double * dydx) {
                if (cosmo->get_OmegaB() == 0.0) {
                    dydx[0] = 0.0;
                } else {
                    dydx[0] = Constants.c * sqrt(get_sound_speed_squared(x)) / cosmo->Hp_of_x(x);
                }
                return GSL_SUCCESS;
            };

            // The initial condition for the sound horizon ODE
            DVector r_initial(1, 0.0);

            // Solve the sound horizon ODE
            ODESolver r_ode(FIDUCIAL_HSTART_ODE_TAU, FIDUCIAL_ABSERR_ODE_TAU, FIDUCIAL_RELERR_ODE_TAU);
            r_ode.solve(deriv_sound, x_array, r_initial);

            // Fetch the data and make a spline
            auto r_array = r_ode.get_data_by_component(0);
            sound_horizon_of_x_spline.create(x_array, r_array, "sound_horizon_of_x");

            // Compute the diffusion scale
            ODEFunction deriv_kd = [&](double x, [[maybe_unused]] const double * y, double * dydx) {
                double a = std::exp(x);
                double R = 4.0 / 3.0 * cosmo->get_OmegaR() / cosmo->get_OmegaB() / a;
                double Hp = cosmo->Hp_of_x(x);
                double rhs = -1.0 / 6.0 * (R * R + 16.0 * (1.0 + R) / 15.0) / ((1.0 + R) * (1.0 + R));
                rhs /= dtaudx_of_x(x) * Hp * Hp / Constants.c / Constants.c;
                dydx[0] = rhs;
                return GSL_SUCCESS;
            };
            DVector kd_initial{0.0};
            ODESolver kd_ode(FIDUCIAL_HSTART_ODE_TAU, FIDUCIAL_ABSERR_ODE_TAU, FIDUCIAL_RELERR_ODE_TAU);
            kd_ode.solve(deriv_kd, x_array, kd_initial);

            // Fetch the data and make a spline
            auto kd_array = kd_ode.get_data_by_component(0);
            for (auto & val : kd_array) {
                if (val > 0.0)
                    val = 1.0 / sqrt(val);
            }
            kd_of_x_spline.create(x_array, kd_array, "damping scale");

            try {
                // Find zstar (tau_noreion == 1)
                x_star = FML::MATH::find_root_bisection(tau_of_x_noreion_spline, 1.0);

                // Find zstar (tau_noreion == 1)
                x_star_saha = FML::MATH::find_root_bisection(tau_of_x_saha_noreion_spline, 1.0);

                // Find peak of visibility function
                std::function<double(double)> func2 = [&](double x) {
                    return dtaudx_of_x(x) * dtaudx_of_x(x) - ddtauddx_of_x(x);
                };
                x_star2 = FML::MATH::find_root_bisection(func2, {-9.0, -6.0});

                // Find zdrag (tau_baryon_noreion == 1)
                x_drag = FML::MATH::find_root_bisection(tau_baryon_noreion_of_x_spline, 1.0);
            } catch (...) {
                std::cout << "Error computing decoupling time\n";
                x_star = 0.0;
                x_star_saha = 0.0;
                x_star2 = 0.0;
                x_star2 = 0.0;
            }
        }

        //====================================================
        // Get methods
        //====================================================

        double RecombinationHistory::tau_of_x(double x) const { return tau_of_x_spline(x); }

        double RecombinationHistory::dtaudx_of_x(double x) const { return dtaudx_of_x_spline(x); }

        double RecombinationHistory::ddtauddx_of_x(double x) const { return dtaudx_of_x_spline.deriv_x(x); }

        double RecombinationHistory::tau_of_x_saha(double x) const { return tau_of_x_saha_spline(x); }

        double RecombinationHistory::tau_of_x_saha_noreion(double x) const { return tau_of_x_saha_noreion_spline(x); }

        double RecombinationHistory::g_tilde_of_x(double x) const { return g_tilde_of_x_spline(x); }

        double RecombinationHistory::dgdx_tilde_of_x(double x) const { return dgdx_tilde_of_x_spline(x); }

        double RecombinationHistory::ddgddx_tilde_of_x(double x) const { return ddgddx_tilde_of_x_spline(x); }

        double RecombinationHistory::Xe_of_x(double x) const {
            double Xe = Xe_of_x_spline(x);
            double f = Xe_reionization_factor_of_x(x);
            return Xe + f;
        }

        double RecombinationHistory::Xe_of_x_noreion(double x) const { return Xe_of_x_spline(x); }

        double RecombinationHistory::Xe_of_x_saha(double x) const {
            double Xe = Xe_of_x_saha_spline(x);
            double f = Xe_reionization_factor_of_x(x);
            return Xe + f;
        }

        double RecombinationHistory::Xe_of_x_saha_noreion(double x) const { return Xe_of_x_saha_spline(x); }

        double RecombinationHistory::ne_of_x(double x) const {
            static const double factor = 3.0 * std::pow(Constants.H0_over_h, 2) / (8 * M_PI * Constants.G * Constants.m_H);
            double n_b0 = factor * cosmo->get_OmegaB() * std::pow(cosmo->get_h(), 2);
            return Xe_of_x(x) * (1.0 - Yp) * n_b0 * std::exp(-3.0 * x);
        }

        double RecombinationHistory::ne_of_x_saha(double x) const {
            static const double factor = 3.0 * std::pow(Constants.H0_over_h, 2) / (8 * M_PI * Constants.G * Constants.m_H);
            double n_b0 = factor * cosmo->get_OmegaB() * std::pow(cosmo->get_h(), 2);
            return Xe_of_x_saha(x) * (1.0 - Yp) * n_b0 * std::exp(-3.0 * x);
        }

        double RecombinationHistory::ne_of_x_noreion(double x) const {
            static double factor = 3.0 * std::pow(Constants.H0_over_h, 2) / (8 * M_PI * Constants.G * Constants.m_H);
            double n_b0 = factor * cosmo->get_OmegaB() * std::pow(cosmo->get_h(), 2);
            return Xe_of_x_noreion(x) * (1.0 - Yp) * n_b0 * std::exp(-3.0 * x);
        }

        double RecombinationHistory::ne_of_x_saha_noreion(double x) const {
            static double factor = 3.0 * std::pow(Constants.H0_over_h, 2) / (8 * M_PI * Constants.G * Constants.m_H);
            double n_b0 = factor * cosmo->get_OmegaB() * std::pow(cosmo->get_h(), 2);
            return Xe_of_x_saha_noreion(x) * (1.0 - Yp) * n_b0 * std::exp(-3.0 * x);
        }

        // Implemention of reionization on Xe. This is the factor f in Xe = Xe^peebles + f
        double RecombinationHistory::Xe_reionization_factor_of_x(double x) const {
            if (!reionization) {
                return 0.0;
            }

            const double y = std::exp(-1.5 * x);
            const double yre = std::pow(1.0 + z_reion, 1.5);
            const double delta_y_reion = 1.5 * sqrt(1.0 + z_reion) * delta_z_reion;
            const double helium_fraction = 0.25 * Yp / (1.0 - Yp);

            // CAMB tanh reionization function
            double f = (1.0 + helium_fraction) / 2.0 * (1.0 + tanh((yre - y) / delta_y_reion));

            // Take into account that Helium probably gets doubly reionized at low
            // redshift
            if (helium_reionization) {
                const double z = std::exp(-x) - 1.0;
                f += helium_fraction / 2.0 * (1.0 + tanh((z_helium_reion - z) / delta_z_helium_reion));
            }
            return f;
        }

        // Baryon temperature
        double RecombinationHistory::get_Tbaryon(double x) const { return Tb_spline(x); }

        // Baryon speed of sound (cs/c)^2
        double RecombinationHistory::get_baryon_sound_speed_squared(double x) const {
            // Simpler approx:
            // 4.0 / 3.0 * Constants.k_b * cosmo->get_TCMB() * (1.0 - rec->get_Yp()) / (a * Constants.m_H * Constants.c
            // * Constants.c);
            return cs2_baryon_spline(x);
        }

        // (cs/c)^2 Photon-Baryon sound of speed (cs/c)^2
        double RecombinationHistory::get_sound_speed_squared(double x) const {
            const double R = 4.0 / 3.0 * cosmo->get_OmegaR() / cosmo->get_OmegaB() * std::exp(-x);
            return R / (1.0 + R) / 3.0;
        }

        double RecombinationHistory::get_sound_horizon(double x) const { return sound_horizon_of_x_spline(x); }

        double RecombinationHistory::get_x_start_rec_array() const { return x_start_rec_array; }

        double RecombinationHistory::get_Yp() const { return Yp; }

        double RecombinationHistory::get_xstar() const { return x_star; }

        double RecombinationHistory::get_xdrag() const { return x_drag; }

    } // namespace COSMOLOGY
} // namespace FML
