#include "Perturbations.h"

namespace FML {
    namespace COSMOLOGY {

        //====================================================
        // Constructors
        //====================================================

        Perturbations::Perturbations(std::shared_ptr<BackgroundCosmology> cosmo,
                                     std::shared_ptr<RecombinationHistory> rec,
                                     ParameterMap & p)
            : cosmo(cosmo), rec(rec) {
            bool polarization = p.get<bool>("polarization");
            bool neutrinos = p.get<bool>("neutrinos");
            int n_ell_theta = p.get<int>("n_ell_theta");
            int n_ell_nu = p.get<int>("n_ell_nu");
            double keta_min = p.get<double>("keta_min");
            double keta_max = p.get<double>("keta_max");
            int n_per_logint = p.get<int>("pert_integration_nk_per_logint");
            double deltax = p.get<double>("pert_delta_x");

            // Where to start integration from
            x_start = p.get<double>("pert_x_initial");
            x_end = 0.0;

            // Store all the ell's when we integrate or just 0,1,2
            pert_spline_all_ells = p.get<bool>("pert_spline_all_ells");

            // Set up k-range and sample frequency
            const double delta_log_k = std::log(10.0) / n_per_logint;
            k_min = keta_min / cosmo->eta_of_x(0.0);
            k_max = keta_max / cosmo->eta_of_x(0.0);

            // Check that k_min is not too small when we have curvature
            double OmegaK = cosmo->get_OmegaK();
            if (OmegaK > 0.0) {
                double K = cosmo->get_K();
                double k_min_curvature = sqrt(-K);
                if (k_min < k_min_curvature) {
                    k_min = 1.01 * k_min_curvature;
                    std::cout << "Open Universe: Adjusting kmin to " << k_min * Constants.Mpc
                              << " (1/Mpc) keta = " << k_min * cosmo->eta_of_x(0.0) << "\n";
                }
            }
            n_k_total = int(std::log(k_max / k_min) / delta_log_k);

            // Set up bookeeping system for perturbations
            psinfo = PerturbationSystemInfo(n_ell_theta, n_ell_theta * int(polarization), n_ell_nu * int(neutrinos));

            psinfo_tight_coupling = PerturbationSystemInfo(2, 0, n_ell_nu * int(neutrinos));

            // Make x-array to integrate over
            const double xstar = rec->get_xstar();
            double x = x_start;
            for (;;) {
                double delta_x_current = deltax;
                if (x > xstar - 0.3 and x < xstar + 1.0)
                    delta_x_current /= 5.0;
                if (x >= -1.1 * delta_x_current)
                    break;
                x += delta_x_current;
                x_array_integration.push_back(x);
            }
            x_array_integration.push_back(0.0);
            n_x_total = x_array_integration.size();

            // Show what every index is
            // psinfo_tight_coupling.debug_print();
            // psinfo.debug_print();
        }

        //====================================================
        // Class methods
        //====================================================

        void Perturbations::info() const {
            std::cout << "\n";
            std::cout << "============================================\n";
            std::cout << "Info about perturbations class:\n";
            std::cout << "============================================\n";
            std::cout << "x_start:       " << x_start << "\n";
            std::cout << "k_min (1/Mpc): " << k_min * Constants.Mpc << "\n";
            std::cout << "k_max (1/Mpc): " << k_max * Constants.Mpc << "\n";
            std::cout << "ckH0_min:      " << Constants.c * k_min / cosmo->get_H0() << "\n";
            std::cout << "ckH0_max:      " << Constants.c * k_max / cosmo->get_H0() << "\n";
            std::cout << "n_k:           " << n_k_total << "\n";
            std::cout << "Polarization:  " << (psinfo.n_ell_theta_p > 0 ? "true" : "false") << "\n";
            std::cout << "Neutrinos:     " << (psinfo.n_ell_nu > 0 ? "true" : "false") << "\n";
            std::cout << "For tight coupling:\n";
            std::cout << "n_tot:         " << psinfo_tight_coupling.n_tot << "\n";
            std::cout << "n_ell_theta:   " << psinfo_tight_coupling.n_ell_theta << "\n";
            std::cout << "start_theta:   " << psinfo_tight_coupling.index_theta_start << "\n";
            std::cout << "n_ell_theta_p: " << psinfo_tight_coupling.n_ell_theta_p << "\n";
            std::cout << "start_theta_p: " << psinfo_tight_coupling.index_theta_p_start << "\n";
            std::cout << "n_ell_nu:      " << psinfo_tight_coupling.n_ell_nu << "\n";
            std::cout << "start_nu:      " << psinfo_tight_coupling.index_nu_start << "\n";
            std::cout << "For full equations:\n";
            std::cout << "n_tot:         " << psinfo.n_tot << "\n";
            std::cout << "n_ell_theta:   " << psinfo.n_ell_theta << "\n";
            std::cout << "start_theta:   " << psinfo.index_theta_start << "\n";
            std::cout << "n_ell_theta_p: " << psinfo.n_ell_theta_p << "\n";
            std::cout << "start_theta_p: " << psinfo.index_theta_p_start << "\n";
            std::cout << "n_ell_nu:      " << psinfo.n_ell_nu << "\n";
            std::cout << "start_nu:      " << psinfo.index_nu_start << "\n";
            std::cout << "============================================\n\n";
        }

        DVector Perturbations::set_ic_after_tight_coupling(const DVector & y_tight_coupling,
                                                           const double x,
                                                           const double k) const {
            //=============================================================================

            // Fetch info about the full perturbation system
            const int n_ell_theta = psinfo.n_ell_theta;
            const int n_ell_theta_p = psinfo.n_ell_theta_p;
            const int n_ell_nu = psinfo.n_ell_nu;
            const int n_tot = psinfo.n_tot;
            const bool polarization = n_ell_theta_p > 0;

            // Fetch info about the tight coupling perturbation system
            const int n_ell_theta_tc = psinfo_tight_coupling.n_ell_theta;
            const int n_ell_nu_tc = psinfo_tight_coupling.n_ell_nu;
            assert(n_ell_theta_tc == 2);

            // Make the vector we are going to fill
            DVector y(n_tot);

            //=============================================================================
            // Reference and pointers to the perturbations quantities
            //=============================================================================
            double & delta_cdm = y[psinfo.index_delta_cdm];
            double & delta_b = y[psinfo.index_delta_b];
            double & v_cdm = y[psinfo.index_v_cdm];
            double & v_b = y[psinfo.index_v_b];
            double & Phi = y[psinfo.index_Phi];
            double * Theta = &y[psinfo.index_theta_start];
            double * Theta_p = &y[psinfo.index_theta_p_start];
            double * Nu = &y[psinfo.index_nu_start];
            // tcb::span<double> Theta  (y.data() + psinfo.index_theta_start,   n_ell_theta);
            // tcb::span<double> Theta_p(y.data() + psinfo.index_theta_p_start, n_ell_theta_p);
            // tcb::span<double> Nu     (y.data() + psinfo.index_nu_start,      n_ell_nu);

            //=============================================================================
            // Reference and pointers to the perturbations quantities in the tight
            // coupling regime
            //=============================================================================
            const double & delta_cdm_tc = y_tight_coupling[psinfo_tight_coupling.index_delta_cdm];
            const double & delta_b_tc = y_tight_coupling[psinfo_tight_coupling.index_delta_b];
            const double & v_cdm_tc = y_tight_coupling[psinfo_tight_coupling.index_v_cdm];
            const double & v_b_tc = y_tight_coupling[psinfo_tight_coupling.index_v_b];
            const double & Phi_tc = y_tight_coupling[psinfo_tight_coupling.index_Phi];
            const double * Theta_tc = &y_tight_coupling[psinfo_tight_coupling.index_theta_start];
            const double * Nu_tc = &y_tight_coupling[psinfo_tight_coupling.index_nu_start];
            // tcb::span<const double> Theta_tc  (y_tight_coupling.data() + psinfo_tight_coupling.index_theta_start,
            // n_ell_theta_tc); tcb::span<const double> Nu_tc     (y_tight_coupling.data() +
            // psinfo_tight_coupling.index_nu_start,      n_ell_nu_tc);
            //=============================================================================

            // Cosmological parameters and variables
            const double Hp = cosmo->Hp_of_x(x);

            // Recombination variables
            const double dtaudx = rec->dtaudx_of_x(x);
            const double ckoverHp = Constants.c * k / Hp;
            const double ckoverHpdtaudx = cosmo->get_OmegaB() > 0.0 ? ckoverHp / dtaudx : 0.0;

            // Quantities that might not exist
            const double theta2fac = polarization ? 8.0 / 15.0 : 20.0 / 45.0;
            const double Theta2 = -theta2fac * ckoverHpdtaudx * Theta_tc[1];

            //=============================================================================

            //=============================================================================
            // SET: Scalar quantities (Gravitational potental, baryons and CDM)
            //=============================================================================
            Phi = Phi_tc;
            delta_cdm = delta_cdm_tc;
            delta_b = delta_b_tc;
            v_cdm = v_cdm_tc;
            v_b = v_b_tc;

            //=============================================================================
            // SET: Photon temperature perturbations (Theta_ell)
            //=============================================================================
            Theta[0] = Theta_tc[0];
            Theta[1] = Theta_tc[1];
            for (int ell = 2; ell < n_ell_theta; ell++) {
                if (ell == 2) {
                    Theta[ell] = Theta2;
                } else {
                    Theta[ell] = -ell / (2.0 * ell + 1) * ckoverHpdtaudx * Theta[ell - 1];
                }
            }

            //=============================================================================
            // SET: Photon polarization perturbations (Theta_p_ell)
            //=============================================================================
            for (int ell = 0; ell < n_ell_theta_p; ell++) {
                if (ell == 0) {
                    Theta_p[ell] = 5.0 / 4.0 * Theta2;
                } else if (ell == 1) {
                    Theta_p[ell] = -ckoverHpdtaudx / 4.0 * Theta2;
                } else if (ell == 2) {
                    Theta_p[ell] = Theta2 / 4.0;
                } else {
                    Theta_p[ell] = -ell / (2.0 * ell + 1) * ckoverHpdtaudx * Theta_p[ell - 1];
                }
            }

            //=============================================================================
            // SET: Neutrino perturbations (N_ell)
            //=============================================================================
            for (int ell = 0; ell < n_ell_nu; ell++) {
                if (ell < n_ell_nu_tc) {
                    Nu[ell] = Nu_tc[ell];
                } else {
                    Nu[ell] = ckoverHp / (2.0 * ell + 1) * Nu[ell - 1];
                }
            }

            return y;
        }

        DVector Perturbations::set_ic(const double x, const double k) const {
            //=============================================================================

            // Fetch info about the tight coupling perturbation system
            const int n_ell_theta = psinfo_tight_coupling.n_ell_theta;
            const int n_ell_theta_p = psinfo_tight_coupling.n_ell_theta_p;
            const int n_ell_nu = psinfo_tight_coupling.n_ell_nu;
            const int n_tot = psinfo_tight_coupling.n_tot;
            const bool polarization = psinfo.n_ell_theta_p > 0;

            // Make the vector we are going to fill
            DVector y(n_tot);

            //=============================================================================
            // Reference and pointers to the perturbations quantities
            //=============================================================================
            double & delta_cdm = y[psinfo_tight_coupling.index_delta_cdm];
            double & delta_b = y[psinfo_tight_coupling.index_delta_b];
            double & v_cdm = y[psinfo_tight_coupling.index_v_cdm];
            double & v_b = y[psinfo_tight_coupling.index_v_b];
            double & Phi = y[psinfo_tight_coupling.index_Phi];
            double * Theta = &y[psinfo_tight_coupling.index_theta_start];
            double * Theta_p = &y[psinfo_tight_coupling.index_theta_p_start];
            double * Nu = &y[psinfo_tight_coupling.index_nu_start];
            // tcb::span<double> Theta  (y.data() + psinfo_tight_coupling.index_theta_start,   n_ell_theta);
            // tcb::span<double> Theta_p(y.data() + psinfo_tight_coupling.index_theta_p_start, n_ell_theta_p);
            // tcb::span<double> Nu     (y.data() + psinfo_tight_coupling.index_nu_start,      n_ell_nu);
            //=============================================================================

            // Cosmological parameters and variables
            const double OmegaNu = cosmo->get_OmegaNu();
            const double OmegaRtot = cosmo->get_OmegaRtot();
            const double f_nu = OmegaNu / OmegaRtot;
            const double H0 = cosmo->get_H0();
            const double Hp = cosmo->Hp_of_x(x);
            const double a = std::exp(x);

            // Recombination variables
            const double dtaudx = rec->dtaudx_of_x(x);
            const double ckoverHp = Constants.c * k / Hp;
            const double ckoverHpdtaudx = cosmo->get_OmegaB() > 0.0 ? ckoverHp / dtaudx : 0.0;

            //=============================================================================
            // SET: Scalar quantities (Gravitational potential, baryons and CDM)
            //=============================================================================
            const double Psi = -1.0 / (1.5 + 2.0 * f_nu / 5.0);
            Phi = -(1.0 + 2.0 / 5.0 * f_nu) * Psi;
            delta_cdm = delta_b = -3.0 / 2.0 * Psi;
            v_cdm = v_b = -ckoverHp / 2.0 * Psi;

            //=============================================================================
            // SET: Photon temperature perturbations (Theta_ell)
            //=============================================================================
            const double theta2fac = polarization ? 8.0 / 15.0 : 20.0 / 45.0;
            const double Theta2 = -theta2fac * ckoverHpdtaudx * (ckoverHp / 6.0 * Psi);
            for (int ell = 0; ell < n_ell_theta; ell++) {
                if (ell == 0) {
                    Theta[ell] = -Psi / 2.0;
                } else if (ell == 1) {
                    Theta[ell] = ckoverHp / 6.0 * Psi;
                } else if (ell == 2) {
                    Theta[ell] = Theta2;
                } else {
                    Theta[ell] = -ell / (2.0 * ell + 1) * ckoverHpdtaudx * Theta[ell - 1];
                };
            }

            //=============================================================================
            // SET: Photon polarization perturbations (Theta_p_ell)
            //=============================================================================
            for (int ell = 0; ell < n_ell_theta_p; ell++) {
                if (ell == 0) {
                    Theta_p[ell] = 5.0 / 4.0 * Theta2;
                } else if (ell == 1) {
                    Theta_p[ell] = -ckoverHpdtaudx / 4.0 * Theta2;
                } else if (ell == 2) {
                    Theta_p[ell] = Theta2 / 4.0;
                } else {
                    Theta_p[ell] = -ell / (2.0 * ell + 1) * ckoverHpdtaudx * Theta_p[ell - 1];
                }
            }

            //=============================================================================
            // SET: Neutrino perturbations (N_ell)
            //=============================================================================
            const double Nu2 = -std::pow(ckoverHp * Hp / H0 * a, 2) / 12.0 *
                               (OmegaNu == 0.0 ? -2.0 / 5.0 * Psi / OmegaRtot : (Phi + Psi) / OmegaNu);
            for (int ell = 0; ell < n_ell_nu; ell++) {
                if (ell == 0) {
                    Nu[ell] = -Psi / 2.0;
                } else if (ell == 1) {
                    Nu[ell] = ckoverHp / 6.0 * Psi;
                } else if (ell == 2) {
                    Nu[ell] = Nu2;
                } else {
                    Nu[ell] = ckoverHp / (2.0 * ell + 1) * Nu[ell - 1];
                }
            }

            return y;
        }

        double Perturbations::get_tight_coupling_time(const double k) const {
            bool verbose = false;
            const int n = 100;
            const double x_start_search = -15.0;
            const double x_end = -std::log(4000.0);

            // The larger the earlier we exit tight coupling
            // 10 is enough for ~% accuracy. 100 is enough for 0.1%
            const double dtaudx_factor = 100.0;

            // Tight coupling is deemed valid as long as dtau < 10 and ck/(H*dtau) > 0.1
            // We also use Xe(x) < Xe_limit as the condition of last time to switch
            // This last condition only helps to improve the accuracy of k close to the horizon
            // and the difference between a z=1000 and z=2000 switch for these modes are like 0.1% in Theta0
            double x = x_start_search;

            // If no baryons there is no tight coupling
            if (cosmo->get_OmegaB() == 0.0) {
                if (verbose)
                    std::cout << "k: " << k * Constants.Mpc << " z: " << std::exp(-x) - 1 << "\n";
                return x;
            }

            // Inefficient brute force search, but fast enough
            while (x < x_end) {
                x += (x_end - x_start_search) / double(n);
                const double ckoverHp = Constants.c * k / cosmo->Hp_of_x(x);
                const double dtaudx = std::fabs(rec->dtaudx_of_x(x));
                if (dtaudx < dtaudx_factor || dtaudx < dtaudx_factor * ckoverHp) {
                    if (verbose)
                        std::cout << "k: " << k * Constants.Mpc << " z: " << std::exp(-x) - 1 << "\n";
                    return x;
                }
            }
            x = x_end;

            if (verbose)
                std::cout << "k: " << k * Constants.Mpc << " z: " << std::exp(-x) - 1 << "\n";
            return x;
        }

        // The perturbations we don't integrate can be found from the initial conditions
        DVector Perturbations::set_all_perturbations_in_tight_coupling(const DVector & y_tight_coupling,
                                                                       const double x,
                                                                       const double k) const {
            //=============================================================================

            // Fetch info about the full perturbation system
            const int n_ell_theta = psinfo.n_ell_theta;
            const int n_ell_theta_p = psinfo.n_ell_theta_p;
            const int n_ell_nu = psinfo.n_ell_nu;
            bool polarization = n_ell_theta_p > 0;

            const int index_delta_cdm = psinfo.index_delta_cdm;
            const int index_delta_b = psinfo.index_delta_b;
            const int index_v_cdm = psinfo.index_v_cdm;
            const int index_v_b = psinfo.index_v_b;
            const int index_Phi = psinfo.index_Phi;

            const int index_theta_start = psinfo.index_theta_start;
            const int index_theta_p_start = psinfo.index_theta_p_start;
            const int index_nu_start = psinfo.index_nu_start;

            // Fetch info about the tight coupling full perturbation system
            const int n_ell_theta_tc = psinfo_tight_coupling.n_ell_theta;
            const int n_ell_nu_tc = psinfo_tight_coupling.n_ell_nu;

            const int index_delta_cdm_tc = psinfo_tight_coupling.index_delta_cdm;
            const int index_delta_b_tc = psinfo_tight_coupling.index_delta_b;
            const int index_v_cdm_tc = psinfo_tight_coupling.index_v_cdm;
            const int index_v_b_tc = psinfo_tight_coupling.index_v_b;
            const int index_Phi_tc = psinfo_tight_coupling.index_Phi;

            const int index_theta_start_tc = psinfo_tight_coupling.index_theta_start;
            const int index_nu_start_tc = psinfo_tight_coupling.index_nu_start;

            //=============================================================================

            // Cosmology and recombination varables
            const double ckoverHp = Constants.c * k / cosmo->Hp_of_x(x);
            const double dtaudx = rec->dtaudx_of_x(x);
            const double ckoverHpdtaudx = ckoverHp / dtaudx;

            // Make the vector we are going to fill
            DVector y(psinfo.n_tot);

            //=============================================================================

            //=============================================================================
            // SET: scalar quantities
            //=============================================================================
            y[index_delta_cdm] = y_tight_coupling[index_delta_cdm_tc];
            y[index_delta_b] = y_tight_coupling[index_delta_b_tc];
            y[index_v_cdm] = y_tight_coupling[index_v_cdm_tc];
            y[index_v_b] = y_tight_coupling[index_v_b_tc];
            y[index_Phi] = y_tight_coupling[index_Phi_tc];

            //=============================================================================
            // SET: Theta multipoles
            //=============================================================================
            const double theta2factor = polarization ? 8.0 / 15.0 : 20.0 / 45.0;
            const double Theta2 = -theta2factor * ckoverHpdtaudx * y_tight_coupling[index_theta_start_tc + 1];
            for (int ell = 0; ell < n_ell_theta; ell++) {
                if (ell < n_ell_theta_tc) {
                    y[index_theta_start + ell] = y_tight_coupling[index_theta_start_tc + ell];
                } else {
                    if (ell == 2) {
                        y[index_theta_start + ell] = Theta2;
                    } else {
                        y[index_theta_start + ell] =
                            -ell / (2.0 * ell + 1) * ckoverHpdtaudx * y[index_theta_start + ell - 1];
                    }
                }
            }

            //=============================================================================
            // SET: Theta_p multipoles
            //=============================================================================
            for (int ell = 0; ell < n_ell_theta_p; ell++) {
                if (ell == 0) {
                    y[index_theta_p_start + ell] = 5.0 / 4.0 * Theta2;
                } else if (ell == 1) {
                    y[index_theta_p_start + ell] = -ckoverHpdtaudx / 4.0 * Theta2;
                } else if (ell == 2) {
                    y[index_theta_p_start + ell] = Theta2 / 4.0;
                } else {
                    y[index_theta_p_start + ell] =
                        -ell / (2.0 * ell + 1) * ckoverHpdtaudx * y[index_theta_p_start + ell - 1];
                }
            }

            //=============================================================================
            // SET: Nu multipoles
            //=============================================================================
            for (int ell = 0; ell < n_ell_nu; ell++) {
                if (ell < n_ell_nu_tc) {
                    y[index_nu_start + ell] = y_tight_coupling[index_nu_start_tc + ell];
                } else {
                    y[index_nu_start + ell] = ckoverHp / (2.0 * ell + 1) * y[index_nu_start + ell - 1];
                }
            }

            return y;
        }

        void Perturbations::solve() {
            // Integrate all the perturbation equation and spline the result
            integrate_perturbations();

            // Compute source functions and spline the result
            compute_source_functions();
        }

        void Perturbations::integrate_perturbations() {
            // Scalar-DVector-Tensor
            int m_type = 0;

            //================================================================
            // Set up k-array
            //================================================================
            auto k_array = FML::MATH::linspace(std::log(k_min), std::log(k_max), n_k_total);
            for (auto & k : k_array)
                k = std::exp(k);

            // Set up q-array q = sqrt(k^2 + (1+|m|)K)
            // For convature we need: nu = q / sqrt|K| and chi = sqrt|K| (eta0-eta)
            auto q_array = k_array;
            const double K = cosmo->get_K();
            for (auto & q : q_array) {
                q = sqrt(q * q + K * (1.0 + m_type));
            }

            //================================================================
            // Set up x-points for which we will store the solution at
            //================================================================
            auto x_array = x_array_integration;

            // Make storage for all the (x,k,y_i) data (this can require O(Mb) storage)
            DVector2D results = DVector2D(psinfo.n_tot, DVector(n_k_total * n_x_total));

            std::cout << "Integrate " << n_k_total << " wavenumbers in the range [" << k_min * Constants.Mpc << " , "
                      << k_max * Constants.Mpc << "]\n";

            // Loop over all wavenumbers
            // Utils::StartTiming("PERT::integrating perturbations");
#ifdef USE_OMP
#pragma omp parallel for schedule(dynamic, 1)
#endif
            for (int ik = 0; ik < n_k_total; ik++) {
                // Progress bar (each thread has unique value of ik so no race)
                if ((10 * ik) / n_k_total != (10 * ik + 10) / n_k_total) {
                    std::cout << (100 * ik + 100) / n_k_total << "% " << std::flush;
                    if (ik == n_k_total - 1) {
                        std::cout << std::endl;
                    }
                }

                // Current value of k
                const double k = k_array[ik];

                // Find value to integrate to (check that x_end_tight is not before x_start)
                const double x_end_tight = get_tight_coupling_time(k);

                DVector x_array_tight, x_array_full;
                int lastindex = 0;
                for (size_t i = 0; i < x_array.size(); i++) {
                    if (x_array[i] < x_end_tight) {
                        x_array_tight.push_back(x_array[i]);
                        lastindex = i;
                    }
                }
                for (size_t i = lastindex; i < x_array.size(); i++) {
                    x_array_full.push_back(x_array[i]);
                }
                const int n_x_tight = x_array_tight.size();

                //===================================================================
                // Tight coupling integration
                //===================================================================

                // Set up initial conditions for the tight coupling regime
                auto y_pert_tight_coupling = set_ic(x_start, k);

                // The tight coupling ODE system
                ODEFunction deriv_tight_coupling = [&](double x, const double * y, double * dydx) {
                    return rhs_tight_coupling_ode(x, k, y, dydx);
                };

                // Integrate to the end of tight coupling
                ODESolver tight_coupling_ode(
                    FIDUCIAL_HSTART_ODE_TIGHT, FIDUCIAL_ABSERR_ODE_TIGHT, FIDUCIAL_RELERR_ODE_TIGHT);

                // Utils::StartTiming("PERT::integrate_tight (all threads)");
                tight_coupling_ode.solve(deriv_tight_coupling, x_array_tight, y_pert_tight_coupling);
                // Utils::EndTiming("PERT::integrate_tight (all threads)");

                //===================================================================
                // Full equation integration
                //===================================================================

                // Set up initial conditions
                y_pert_tight_coupling = tight_coupling_ode.get_final_data();
                auto y_pert_full = set_ic_after_tight_coupling(y_pert_tight_coupling, x_end_tight, k);

                // The full ODE system
                ODEFunction deriv_full = [&](double x, const double * y, double * dydx) {
                    return rhs_full_ode(x, k, y, dydx);
                };

                ODESolver full_ode(FIDUCIAL_HSTART_ODE_FULL, FIDUCIAL_ABSERR_ODE_FULL, FIDUCIAL_RELERR_ODE_FULL);

                // Integrate till the present time. If a Jacobian is availiable use that for
                // the largest k-modes as this is much faster
                // Utils::StartTiming("PERT::integrate_full (all threads)");
#define USE_JACOBIAN
#ifndef USE_JACOBIAN
                full_ode.solve(deriv_full, x_array_full, y_pert_full);
#else
                ODEFunctionJacobian jacobian_full = [&](double x, const double * y, double * dfdy, double * dfdt) {
                    return rhs_jacobian_full(x, k, y, dfdy, dfdt);
                };

                if (k * Constants.Mpc > 0.15) {
                    full_ode.solve(deriv_full, x_array_full, y_pert_full, gsl_odeiv2_step_msbdf, jacobian_full);
                } else {
                    full_ode.solve(deriv_full, x_array_full, y_pert_full);
                }
#endif
                // Utils::EndTiming("PERT::integrate_full (all threads)");

                //===================================================================
                // Store the data
                //===================================================================

                // Utils::StartTiming("PERT::store data");

                auto data_tight = tight_coupling_ode.get_data();
                auto data_full = full_ode.get_data();

                // Process the data from the tight regime into the same form as the full
                // regime and fill inn missing values
                DVector2D data_tight_full;
                const int n_eq_tight = psinfo_tight_coupling.n_tot;
                const int n_eq_full = psinfo.n_tot;
                for (int ix = 0; ix < n_x_tight; ix++) {
                    auto y_current = DVector(n_eq_tight);
                    for (int iq = 0; iq < n_eq_tight; iq++) {
                        y_current[iq] = data_tight[ix][iq];
                    }
                    auto tmp = set_all_perturbations_in_tight_coupling(y_current, x_array_tight[ix], k);
                    data_tight_full.push_back(tmp);
                }
                data_tight_full.insert(data_tight_full.end(), data_full.begin() + 1, data_full.end());

                // Store the data (this works with OpenMP without atomic as each thread
                // writes to different places in the array)
                for (int ix = 0; ix < n_x_total; ix++) {
                    for (int iq = 0; iq < n_eq_full; iq++) {
                        results[iq][ix + n_x_total * ik] = data_tight_full[ix][iq];
                    }
                }

                // Utils::EndTiming("PERT::store data");
            }
            // Utils::EndTiming("PERT::integrating perturbations");
            std::cout << "Done! Making splines and source functions\n\n";

            //=============================================================================

            // Utils::StartTiming("PERT::making splines");

            //=============================================================================
            // Splines of scalar quantities
            //=============================================================================
            delta_cdm_spline.create(x_array, k_array, results[psinfo.index_delta_cdm], "delta_cdm_x_k");
            delta_b_spline.create(x_array, k_array, results[psinfo.index_delta_b], "delta_b_x_k");
            v_cdm_spline.create(x_array, k_array, results[psinfo.index_v_cdm], "v_cdm_x_k");
            v_b_spline.create(x_array, k_array, results[psinfo.index_v_b], "v_b_x_k");
            Phi_spline.create(x_array, k_array, results[psinfo.index_Phi], "Phi_x_k");

            //=============================================================================
            // Splines of Theta_ell
            //=============================================================================
            Theta_spline = std::vector<Spline2D>(psinfo.n_ell_theta);
            for (int ell = 0; ell < (pert_spline_all_ells ? psinfo.n_ell_theta : 3); ell++) {
                std::string name = "theta" + std::to_string(ell) + "_x_k";
                Theta_spline[ell].create(x_array, k_array, results[psinfo.index_theta_start + ell], name);
            }

            //=============================================================================
            // Splines of Theta_ep_ll
            //=============================================================================
            Theta_p_spline = std::vector<Spline2D>(psinfo.n_ell_theta_p);
            if (psinfo.n_ell_theta_p > 0)
                for (int ell = 0; ell < (pert_spline_all_ells ? psinfo.n_ell_theta_p : 3); ell++) {
                    std::string name = "theta_p" + std::to_string(ell) + "_x_k";
                    Theta_p_spline[ell].create(x_array, k_array, results[psinfo.index_theta_p_start + ell], name);
                }

            //=============================================================================
            // Splines of Nu_ell
            //=============================================================================
            Nu_spline = std::vector<Spline2D>(psinfo.n_ell_nu);
            if (psinfo.n_ell_nu > 0)
                for (int ell = 0; ell < (pert_spline_all_ells ? psinfo.n_ell_nu : 3); ell++) {
                    std::string name = "nu" + std::to_string(ell) + "_x_k";
                    Nu_spline[ell].create(x_array, k_array, results[psinfo.index_nu_start + ell], name);
                }

            //=============================================================================
            // Spline of other quantities
            //=============================================================================

            // Compute Psi and Pi and zeta
            DVector Pi_array(n_k_total * n_x_total);
            DVector Psi_array(n_k_total * n_x_total);
            for (int ik = 0; ik < n_k_total; ik++) {
                const double k = k_array[ik];
                for (int ix = 0; ix < n_x_total; ix++) {
                    const int index = ix + n_x_total * ik;
                    const double x = x_array[ix];
                    const double Theta2 = get_Theta(x, k, 2);
                    const double Nu2 = get_Nu(x, k, 2);
                    const double Theta_p0 = get_Theta_p(x, k, 0);
                    const double Theta_p2 = get_Theta_p(x, k, 2);
                    const double ckoverHp = Constants.c * k / cosmo->Hp_of_x(x);
                    Psi_array[index] =
                        -get_Phi(x, k) -
                        12.0 / (ckoverHp * ckoverHp) * (cosmo->get_OmegaR(x) * Theta2 + cosmo->get_OmegaNu(x) * Nu2);
                    Pi_array[index] = (Theta2 + Theta_p0 + Theta_p2);
                }
            }
            Psi_spline.create(x_array, k_array, Psi_array, "Psi_x_k");
            Pi_spline.create(x_array, k_array, Pi_array, "Pi_x_k");

            // Derivative of Pi
            DVector dPidx_array(n_k_total * n_x_total);
            for (int ik = 0; ik < n_k_total; ik++) {
                const double k = k_array[ik];
                for (int ix = 0; ix < n_x_total; ix++) {
                    const int index = ix + n_x_total * ik;
                    const double x = x_array[ix];
                    dPidx_array[index] = Pi_spline.deriv_x(x, k);
                }
            }
            dPidx_spline.create(x_array, k_array, dPidx_array, "dPidx_x_k");

            // Curvature perturbation
            DVector zeta_array(n_k_total * n_x_total);
            for (int ik = 0; ik < n_k_total; ik++) {
                const double k = k_array[ik];
                for (int ix = 0; ix < n_x_total; ix++) {
                    const int index = ix + n_x_total * ik;
                    const double x = x_array[ix];
                    const double Phi = Phi_spline(x, k);
                    const double weff = cosmo->get_weff(x);
                    const double ckoverHp = Constants.c * k / cosmo->Hp_of_x(x);
                    double drho_source = 0.0;
                    drho_source += get_delta_cdm(x, k) * cosmo->get_OmegaCDM(x);
                    drho_source += get_delta_b(x, k) * cosmo->get_OmegaB(x);
                    drho_source += 4 * get_Theta(x, k, 0) * cosmo->get_OmegaR(x);
                    drho_source += 4 * get_Nu(x, k, 0) * cosmo->get_OmegaNu(x);
                    zeta_array[index] =
                        Phi + (-ckoverHp * ckoverHp / 3.0 * Phi + 0.5 * drho_source) / (1.5 * (1 + weff));
                }
            }
            zeta_spline.create(x_array, k_array, zeta_array, "zetaCurvPert_x_k");

            /*
               DVector dzetadx_array(n_k_total * n_x_total);
               for (int ik = 0; ik < n_k_total; ik++) {
               const double k = k_array[ik];
               for (int ix = 0; ix < n_x_total; ix++) {
               const int index = ix + n_x_total * ik;
               const double x = x_array[ix];

               const double weff = cosmo->get_weff(x);
               const double dweffdx = cosmo->get_dweffdx(x);
               const double Hp = cosmo->Hp_of_x(x);
               const double Phi = Phi_spline(x,k);
               const double PprimeoverRhoprime = weff - dweffdx/(3.0*(1+weff));

            // Non-adiabatic pressure contribution (P'/rho' - w_i) Omega_i delta_i
            double PnadTerm = 0.0;
            PnadTerm += (PprimeoverRhoprime-0.0 ) * get_delta_cdm(x,k)   * cosmo->get_OmegaCDM(x);
            PnadTerm += (PprimeoverRhoprime-0.0 ) * get_delta_b(x,k)     * cosmo->get_OmegaB(x);
            PnadTerm += (PprimeoverRhoprime-1/3.) * 4.0*get_Theta(x,k,0) * cosmo->get_OmegaR(x);
            PnadTerm += (PprimeoverRhoprime-1/3.) * 4.0*get_Nu(x,k,0)    * cosmo->get_OmegaNu(x);
            dzetadx_array[index] = -2.0/3.0/(1+weff) * PprimeoverRhoprime * Phi * std::pow(Constants.c * k / Hp,2) +
            PnadTerm/(1+weff);
            }
            }
            dzetadx_spline.create (x_array, k_array, dzetadx_array,  "dzetaCurvPertdx_x_k");
             */

            // Utils::EndTiming("PERT::making splines");
        }

        void Perturbations::compute_source_functions() {
            // Utils::StartTiming("PERT::making source");

            // The x and k arrays to use to make the spline
            // For simplicity we jusy use the same array as used when integrating the perturbations
            auto k_array = FML::MATH::linspace(std::log(k_min), std::log(k_max), n_k_total);
            for (auto & k : k_array)
                k = std::exp(k);
            auto x_array = FML::MATH::linspace(x_start, x_end, n_x_total);

            // Compute source functions
            DVector ST_array(n_k_total * n_x_total);
            DVector SE_array(n_k_total * n_x_total);
            DVector SN_array(n_k_total * n_x_total);

            DVector SW_array(n_k_total * n_x_total);
            DVector ISW_array(n_k_total * n_x_total);
            DVector VEL_array(n_k_total * n_x_total);
            DVector POL_array(n_k_total * n_x_total);

            // x dependent quantities needed below
            DVector exp_tau_array(n_x_total), chi_array(n_x_total);
            DVector Hp_array(n_x_total), dHpdxHp_array(n_x_total), ddHpddxHp_array(n_x_total);
            DVector g_array(n_x_total), dgdx_array(n_x_total), ddgddx_array(n_x_total);
            for (int ix = 0; ix < n_x_total; ix++) {
                const double x = x_array[ix];
                Hp_array[ix] = cosmo->Hp_of_x(x);
                dHpdxHp_array[ix] = cosmo->dHpdx_of_x(x) / Hp_array[ix];
                ddHpddxHp_array[ix] = cosmo->ddHpddx_of_x(x) / Hp_array[ix];
                exp_tau_array[ix] = std::exp(-rec->tau_of_x(x));
                chi_array[ix] = cosmo->eta_of_x(x_end) - cosmo->eta_of_x(x);
                g_array[ix] = rec->g_tilde_of_x(x);
                dgdx_array[ix] = rec->dgdx_tilde_of_x(x);
                ddgddx_array[ix] = rec->ddgddx_tilde_of_x(x);
            }

            for (int ix = 0; ix < n_x_total; ix++) {
                const double x = x_array[ix];
                for (int ik = 0; ik < n_k_total; ik++) {
                    const double k = k_array[ik];
                    const int index = ix + n_x_total * ik;

                    const double Hp = Hp_array[ix];
                    const double dHpdxHp = dHpdxHp_array[ix];
                    const double ddHpddxHp = ddHpddxHp_array[ix];
                    const double chi = chi_array[ix];
                    const double exp_tau = exp_tau_array[ix];
                    const double g = g_array[ix];
                    const double dgdx = dgdx_array[ix];
                    const double ddgddx = ddgddx_array[ix];
                    const double ckoverHp = Constants.c * k / Hp;

                    const double Theta0 = get_Theta(x, k, 0);
                    const double Pi = get_Pi(x, k);
                    // const double dPidx   = dPidx_spline(x, k);
                    // const double ddPiddx = dPidx_spline.deriv_x(x, k);
                    const double dPidx = Pi_spline.deriv_x(x, k);
                    const double ddPiddx = Pi_spline.deriv_xx(x, k);
                    const double dPhidx = Phi_spline.deriv_x(x, k);
                    const double dPsidx = Psi_spline.deriv_x(x, k);
                    const double Psi = get_Psi(x, k);
                    const double v_b = get_v_b(x, k);
                    const double dv_bdx = v_b_spline.deriv_x(x, k);

                    // The temperature source
                    const double SW_term = g * (Theta0 + Psi + Pi / 4.0);
                    const double ISW_term = exp_tau * (dPsidx - dPhidx);
                    const double velocity_term = -(1.0 / ckoverHp) * (dHpdxHp * g * v_b + dgdx * v_b + g * dv_bdx);
                    const double polarization_term =
                        3.0 / (4.0 * ckoverHp * ckoverHp) *
                        ((ddgddx * Pi + 2.0 * dgdx * dPidx + g * ddPiddx) + 3.0 * dHpdxHp * (dgdx * Pi + g * dPidx) +
                         g * Pi * (ddHpddxHp + dHpdxHp * dHpdxHp));

                    ST_array[index] = SW_term + ISW_term + velocity_term + polarization_term;
                    SW_array[index] = SW_term;
                    ISW_array[index] = ISW_term;
                    VEL_array[index] = velocity_term;
                    POL_array[index] = polarization_term;

                    // Polarization source
                    SE_array[index] = x > -0.001 ? 0.0 : 3.0 * g * Pi / 4.0 / std::pow(ckoverHp * Hp * chi / Constants.c, 2);

                    // Neutrino source
                    SN_array[index] = (dPsidx - dPhidx);
                }
            }

            // Spline up source functions
            ST_spline.create(x_array, k_array, ST_array, "Source_Temp_x_k");
            SE_spline.create(x_array, k_array, SE_array, "Source_Epol_x_k");
            SN_spline.create(x_array, k_array, SN_array, "Source_Nu_x_k");

            // Individual contributions to the temperature source function
            SW_spline.create(x_array, k_array, SW_array, "Source_SW_x_k");
            ISW_spline.create(x_array, k_array, ISW_array, "Source_ISW_x_k");
            VEL_spline.create(x_array, k_array, VEL_array, "Source_VEL_x_k");
            POL_spline.create(x_array, k_array, POL_array, "Source_POL_x_k");

            // Utils::EndTiming("PERT::making source");
        }

        void Perturbations::output_perturbations(const double k, const std::string filename) const {

            std::ofstream fp(filename.c_str());
            const int npts = 2000;
            auto x_array = FML::MATH::linspace(x_start, 0.0, npts);
            fp << "# x = log(a)    Perturbation quantities\n";
            auto print_data = [&](const double x) {
                double arg = k * cosmo->chi_of_x(x);

                // 1
                fp << x << " ";

                // 2
                fp << get_Theta(x, k, 0) + get_Psi(x, k) << " ";

                // 3
                fp << get_Theta(x, k, 0) << " ";
                fp << get_Theta(x, k, 1) << " ";
                fp << get_Theta(x, k, 2) << " ";

                // 6
                fp << get_Theta_p(x, k, 0) << " ";
                fp << get_Theta_p(x, k, 1) << " ";
                fp << get_Theta_p(x, k, 2) << " ";

                // 9
                fp << get_Nu(x, k, 0) << " ";
                fp << get_Nu(x, k, 1) << " ";
                fp << get_Nu(x, k, 2) << " ";

                // 12
                fp << get_delta_cdm(x, k) - 3.0 * cosmo->Hp_of_x(x) / (Constants.c * k) * get_v_cdm(x, k) << " ";
                fp << get_delta_cdm(x, k) << " ";
                fp << get_v_cdm(x, k) << " ";

                // 15
                fp << get_delta_b(x, k) - 3.0 * cosmo->Hp_of_x(x) / (Constants.c * k) * get_v_b(x, k) << " ";
                fp << get_delta_b(x, k) << " ";
                fp << get_v_b(x, k) << " ";

                // 18
                fp << get_Phi(x, k) << " ";
                fp << get_Psi(x, k) << " ";

                // 20
                fp << get_Pi(x, k) << " ";

                // 21
                fp << get_Source_T(x, k) << " ";
                fp << get_Source_E(x, k) << " ";

                // 23
                fp << FML::MATH::j_ell(5, arg) * get_Source_T(x, k) << " ";
                fp << FML::MATH::j_ell(50, arg) * get_Source_T(x, k) << " ";
                fp << FML::MATH::j_ell(500, arg) * get_Source_T(x, k) << " ";

                // fp << SW_spline(x,k) << " ";
                // fp << ISW_spline(x,k) << " ";
                // fp << VEL_spline(x,k) << " ";
                // fp << POL_spline(x,k) << " ";

                // 26
                fp << get_Source_E(x, k) * FML::MATH::j_ell(5, arg) * sqrt((5. + 2) * (5. + 1) * (5. + 0) * (5. - 1))
                   << " ";
                fp << get_Source_E(x, k) * FML::MATH::j_ell(50, arg) *
                          sqrt((50. + 2) * (50. + 1) * (50. + 0) * (50. - 1))
                   << " ";
                fp << get_Source_E(x, k) * FML::MATH::j_ell(500, arg) *
                          sqrt((500. + 2) * (500. + 1) * (500. + 0) * (500. - 1))
                   << " ";

                // 29
                fp << get_Source_N(x, k) * FML::MATH::j_ell(5, arg) << " ";
                fp << get_Source_N(x, k) * FML::MATH::j_ell(50, arg) << " ";
                fp << get_Source_N(x, k) * FML::MATH::j_ell(500, arg) << " ";

                // 31 in total
                fp << "\n";
            };
            std::for_each(x_array.begin(), x_array.end(), print_data);
        }

        void Perturbations::output_transfer(const double x, const std::string filename) const {
            std::ofstream fp(filename.c_str());
            const int npts = 1000;
            auto k_array = FML::MATH::linspace(std::log(k_min), std::log(k_max), npts);
            for (auto & k : k_array)
                k = std::exp(k);

            const double norm_k = Constants.Mpc / cosmo->get_h();
            const double norm = std::pow(cosmo->get_h() / Constants.Mpc, 2);
            fp << "# k (h/Mpc)   Matter   Baryon   CDM  CB  R  Nu   Rtot    (Units: (Mpc/h)^2)\n";
            auto print_data = [&](const double k) {
                fp << k * norm_k << " ";
                fp << get_transfer_gammaNbody(x, k) * norm << " ";
                fp << get_transfer_Phi(x, k) * norm << " ";
                fp << get_transfer_Delta_M(x, k) * norm << " ";
                fp << get_transfer_Delta_b(x, k) * norm << " ";
                fp << get_transfer_Delta_cdm(x, k) * norm << " ";
                fp << get_transfer_Delta_cb(x, k) * norm << " ";
                fp << get_transfer_Delta_R(x, k) * norm << " ";
                fp << get_transfer_Delta_Nu(x, k) * norm << " ";
                fp << get_transfer_Delta_Rtot(x, k) * norm << " ";
                fp << "\n";
            };
            std::for_each(k_array.begin(), k_array.end(), print_data);
        }

        //====================================================
        // Methods to solve the perturbations ODEs
        //====================================================

        // Derivatives in the tight coupling regime
        int Perturbations::rhs_tight_coupling_ode(double x, double k, const double * y, double * dydx) {
            //=============================================================================
            // Number of quantities we have
            //=============================================================================
            const int n_ell_nu = psinfo_tight_coupling.n_ell_nu;
            const int n_ell_theta = psinfo_tight_coupling.n_ell_theta;
            bool neutrinos = n_ell_nu > 0;
            bool polarization = psinfo.n_ell_theta_p > 0;
            assert(n_ell_theta == 2);

            //=============================================================================
            // Reference and pointers to the perturbations quantities in the tight
            // coupling regime. Bound checks in debug mode with span
            //=============================================================================
            const double & delta_cdm = y[psinfo_tight_coupling.index_delta_cdm];
            const double & delta_b = y[psinfo_tight_coupling.index_delta_b];
            const double & v_cdm = y[psinfo_tight_coupling.index_v_cdm];
            const double & v_b = y[psinfo_tight_coupling.index_v_b];
            const double & Phi = y[psinfo_tight_coupling.index_Phi];
            const double * Theta = &y[psinfo_tight_coupling.index_theta_start];
            const double * Nu = &y[psinfo_tight_coupling.index_nu_start];
            // tcb::span<const double> Theta  (y + psinfo_tight_coupling.index_theta_start,   n_ell_theta);
            // tcb::span<const double> Nu     (y + psinfo_tight_coupling.index_nu_start,      n_ell_nu);

            double & ddelta_cdmdx = dydx[psinfo_tight_coupling.index_delta_cdm];
            double & ddelta_bdx = dydx[psinfo_tight_coupling.index_delta_b];
            double & dv_cdmdx = dydx[psinfo_tight_coupling.index_v_cdm];
            double & dv_bdx = dydx[psinfo_tight_coupling.index_v_b];
            double & dPhidx = dydx[psinfo_tight_coupling.index_Phi];
            double * dThetadx = &dydx[psinfo_tight_coupling.index_theta_start];
            double * dNudx = &dydx[psinfo_tight_coupling.index_nu_start];
            // tcb::span<double> dThetadx  (dydx + psinfo_tight_coupling.index_theta_start,   n_ell_theta);
            // tcb::span<double> dNudx     (dydx + psinfo_tight_coupling.index_nu_start,      n_ell_nu);
            //=============================================================================

            //=============================================================================
            // Cosmological parameters and variables
            //=============================================================================
            const double OmegaB = cosmo->get_OmegaB();
            const double OmegaCDM = cosmo->get_OmegaCDM();
            const double OmegaR = cosmo->get_OmegaR();
            const double OmegaNu = cosmo->get_OmegaNu();
            const double H0 = cosmo->get_H0();
            const double Hp = cosmo->Hp_of_x(x);
            const double dHpdx = cosmo->dHpdx_of_x(x);
            const double eta = cosmo->eta_of_x(x);
            const double etaHp = eta * Hp / Constants.c;
            const double a = std::exp(x);

            //=============================================================================
            // Recombination variables
            //=============================================================================
            const double dtaudx = rec->dtaudx_of_x(x);
            const double ddtauddx = rec->ddtauddx_of_x(x);
            const double ckoverHp = Constants.c * k / Hp;
            const double ckoverH0 = Constants.c * k / H0;
            const double ckoverHpdtaudx = OmegaB > 0.0 ? ckoverHp / dtaudx : 0.0;
            const double theta2fac = polarization ? 8.0 / 15.0 : 20.0 / 45.0;

            //=============================================================================
            // The lowest multipoles
            // We might or might not have all of them (and it differs with and without polarization)
            //=============================================================================
            const double Nu0 = neutrinos ? Nu[0] : 0.0;
            const double Nu2 = neutrinos ? Nu[2] : 0.0;
            const double Theta2 = -theta2fac * ckoverHpdtaudx * Theta[1];
            const double Theta_p0 = 5.0 / 4.0 * Theta2;
            const double Theta_p2 = Theta2 / 4.0;
            const double Pi = Theta_p0 + Theta_p2 + Theta2;

            //=============================================================================

            // The second gravitational potential
            const double Psi = -Phi - 12.0 * std::pow(1.0 / (ckoverH0 * a), 2) * (OmegaR * Theta2 + OmegaNu * Nu2);

            //=============================================================================
            // SET: Derivative of Phi
            //=============================================================================
            dPhidx = Psi - ckoverHp * ckoverHp / 3.0 * Phi +
                     0.5 * (H0 * H0) / (Hp * Hp * a * a) *
                         (OmegaCDM * delta_cdm * a + OmegaB * delta_b * a + 4.0 * (OmegaR * Theta[0] + OmegaNu * Nu0));

            // The tight coupling factor q_tc
            const double R = OmegaB > 0.0 ? 4.0 / 3.0 * OmegaR / OmegaB / a : 0.0;
            const double sound_speed_squared = rec->get_baryon_sound_speed_squared(x);
            const double dTheta0dx = -ckoverHp * Theta[1] - dPhidx;
            const double dTheta2dx = ckoverHp / 5.0 * (2.0 * Theta[1]) + dtaudx * (Theta2 - Pi / 10.0);
            const double q_tc =
                (-((1.0 - R) * dtaudx + (1.0 + R) * ddtauddx) * (3.0 * Theta[1] + v_b) - ckoverHp * Psi +
                 (1.0 - dHpdx / Hp) * ckoverHp * (-Theta[0] + 2.0 * Theta2) + ckoverHp * (-dTheta0dx + dTheta2dx)) /
                ((1.0 + R) * dtaudx + dHpdx / Hp - 1.0);
            const double q_vb_tc =
                (-v_b - ckoverHp * sound_speed_squared * delta_b + R * (q_tc + ckoverHp * (-Theta[0] + 2.0 * Theta2))) /
                (1.0 + R);

            //=============================================================================
            // SET: Baryons and CDM
            //=============================================================================
            ddelta_cdmdx = ckoverHp * v_cdm - 3.0 * dPhidx;
            ddelta_bdx = ckoverHp * v_b - 3.0 * dPhidx;
            dv_cdmdx = -v_cdm - ckoverHp * Psi;
            dv_bdx = q_vb_tc - ckoverHp * Psi;

            //=============================================================================
            // SET: Photons multipoles (Theta_ell)
            //=============================================================================
            dThetadx[0] = -ckoverHp * Theta[1];
            dThetadx[1] = (q_tc - q_vb_tc) / 3.0;

            // Add sources
            dThetadx[0] += -dPhidx;
            dThetadx[1] += ckoverHp * Psi / 3.0;

            //=============================================================================
            // SET: Neutrinos multipoles (Nu_ell)
            //=============================================================================
            for (int ell = 0; ell < n_ell_nu; ell++) {
                const double prefac = ckoverHp / (2.0 * ell + 1);
                if (ell == 0) {
                    dNudx[ell] = prefac * (-(ell + 1) * Nu[ell + 1]);
                } else if (ell < n_ell_nu - 1) {
                    dNudx[ell] = prefac * (ell * Nu[ell - 1] - (ell + 1) * Nu[ell + 1]);
                } else {
                    dNudx[ell] = ckoverHp * Nu[ell - 1] - (ell + 1) / (etaHp)*Nu[ell];
                }

                // Add sources
                if (ell == 0)
                    dNudx[ell] += -dPhidx;
                if (ell == 1)
                    dNudx[ell] += ckoverHp / 3.0 * Psi;
            }

            return GSL_SUCCESS;
        }

        int Perturbations::rhs_jacobian_full(double x, double k, const double * y, double * dfdy, double * dfdt) {
            // Utils::StartTiming("PERT::jacobian");

            // This computes dfdt - explicit x-derivative of the rhs of the ODE system
            // and dfdy the Jacobian matrix of the system
            const double & delta_cdm = y[psinfo.index_delta_cdm];
            const double & delta_b = y[psinfo.index_delta_b];
            const double & v_cdm = y[psinfo.index_v_cdm];
            const double & v_b = y[psinfo.index_v_b];
            const double & Phi = y[psinfo.index_Phi];
            const double * Theta = &y[psinfo.index_theta_start];
            const double * Theta_p = &y[psinfo.index_theta_p_start];
            const double * Nu = &y[psinfo.index_nu_start];

            double & dfdt_delta_cdm = dfdt[psinfo.index_delta_cdm];
            double & dfdt_delta_b = dfdt[psinfo.index_delta_b];
            double & dfdt_v_cdm = dfdt[psinfo.index_v_cdm];
            double & dfdt_v_b = dfdt[psinfo.index_v_b];
            double & dfdt_Phi = dfdt[psinfo.index_Phi];
            double * dfdt_Theta = &dfdt[psinfo.index_theta_start];
            double * dfdt_Theta_p = &dfdt[psinfo.index_theta_p_start];
            double * dfdt_Nu = &dfdt[psinfo.index_nu_start];

            // Cosmology variables
            const double OmegaB = cosmo->get_OmegaB();
            const double OmegaCDM = cosmo->get_OmegaCDM();
            const double OmegaR = cosmo->get_OmegaR();
            const double OmegaNu = cosmo->get_OmegaNu();
            const double H0 = cosmo->get_H0();
            const double Hp = cosmo->Hp_of_x(x);
            const double dlogHpdx = cosmo->dHpdx_of_x(x) / Hp;
            const double eta = cosmo->eta_of_x(x);
            const double etaHp = eta * Hp / Constants.c;
            const double a = std::exp(x);
            const double doneoveretaHpdx = -1.0 / (etaHp) * (1.0 / etaHp + dlogHpdx);

            // Recombination variables
            const double dtaudx = rec->dtaudx_of_x(x);
            const double ddtauddx = rec->ddtauddx_of_x(x);
            const double ckoverHp = Constants.c * k / Hp;
            const double ckoverH0 = Constants.c * k / H0;

            // Some quantitiess needed below
            const double Nu0 = psinfo.n_ell_nu > 0 ? Nu[0] : 0.0;
            const double Nu2 = psinfo.n_ell_nu > 2 ? Nu[2] : 0.0;
            const double Theta_p0 = psinfo.n_ell_theta_p > 0 ? Theta_p[0] : 0.0;
            const double Theta_p2 = psinfo.n_ell_theta_p > 2 ? Theta_p[2] : 0.0;
            const double PhiH0H0term = 0.5 * (H0 * H0) / (Hp * Hp * a * a);
            const double Psiterm = 12.0 / (ckoverH0 * ckoverH0 * a * a);
            const double R = OmegaB > 0.0 ? 4.0 / 3.0 * OmegaR / OmegaB / a : 0.0;
            const double Pi = Theta[2] + Theta_p0 + Theta_p2;
            const double Psi = -Phi - Psiterm * (OmegaR * Theta[2] + OmegaNu * Nu2);
            const double dPsidt = 2.0 * Psiterm * (OmegaR * Theta[2] + OmegaNu * Nu2);
            ;

            // Derivatives of scalar equations
            dfdt_Phi = dPsidt + 2.0 * dlogHpdx * ckoverHp * ckoverHp * Phi / 3.0 +
                       PhiH0H0term * (+(-1.0 - 2 * dlogHpdx) * (OmegaCDM * delta_cdm * a + OmegaB * delta_b * a) +
                                      (-2.0 - 2 * dlogHpdx) * 4.0 * (OmegaR * Theta[0] + OmegaNu * Nu0));
            dfdt_delta_cdm = -dlogHpdx * ckoverHp * v_cdm - 3.0 * dfdt_Phi;
            dfdt_delta_b = -dlogHpdx * ckoverHp * v_b - 3.0 * dfdt_Phi;
            dfdt_v_cdm = dlogHpdx * ckoverHp * Psi - ckoverHp * dPsidt;
            dfdt_v_b = dlogHpdx * ckoverHp * Psi - ckoverHp * dPsidt + (ddtauddx - dtaudx) * R * (Theta[1] + v_b);

            // Photons
            for (int ell = 0; ell < psinfo.n_ell_theta; ell++) {
                if (ell == 0) {
                    dfdt_Theta[0] = dlogHpdx * ckoverHp * Theta[1] - dfdt_Phi;
                } else if (ell == 1) {
                    dfdt_Theta[1] =
                        -dlogHpdx * ckoverHp / (2.0 * ell + 1) * (ell * Theta[0] - (ell + 1) * Theta[2] + Psi) +
                        ckoverHp / 3.0 * dPsidt + ddtauddx * (Theta[ell] + v_b / 3.0);
                } else if (ell < psinfo.n_ell_theta - 1) {
                    dfdt_Theta[ell] =
                        -dlogHpdx * ckoverHp / (2.0 * ell + 1) * (ell * Theta[ell - 1] - (ell + 1) * Theta[ell + 1]) +
                        ddtauddx * (Theta[ell] - (ell == 2 ? Pi / 10.0 : 0.0));
                } else {
                    dfdt_Theta[ell] = -dlogHpdx * ckoverHp * Theta[ell - 1] - (ell + 1) * doneoveretaHpdx * Theta[ell] +
                                      ddtauddx * Theta[ell];
                }
            }

            // Polarization (Assuming nellp > 1)
            for (int ell = 0; ell < psinfo.n_ell_theta_p; ell++) {
                if (ell == 0) {
                    dfdt_Theta_p[ell] = dlogHpdx * ckoverHp * Theta_p[1] + ddtauddx * (Theta_p[ell] - Pi / 2.0);
                } else if (ell < psinfo.n_ell_theta_p - 1) {
                    dfdt_Theta_p[ell] = -dlogHpdx * ckoverHp / (2.0 * ell + 1) *
                                            (ell * Theta_p[ell - 1] - (ell + 1) * Theta_p[ell + 1]) +
                                        ddtauddx * (Theta_p[ell] - (ell == 2 ? Pi / 10.0 : 0.0));
                } else {
                    dfdt_Theta_p[ell] = -dlogHpdx * ckoverHp * Theta_p[ell - 1] -
                                        (ell + 1) * doneoveretaHpdx * Theta_p[ell] + ddtauddx * Theta_p[ell];
                }
            }

            // Neutrinos (Assuming nellnu > 2)
            for (int ell = 0; ell < psinfo.n_ell_nu; ell++) {
                if (ell == 0) {
                    dfdt_Nu[ell] = dlogHpdx * ckoverHp * Nu[1] - dfdt_Phi;
                } else if (ell == 1) {
                    dfdt_Nu[ell] = -dlogHpdx / 3.0 * ckoverHp * (Nu[0] - 2.0 * Nu[2] + Psi) + ckoverHp / 3.0 * dPsidt;
                } else if (ell < psinfo.n_ell_nu - 1) {
                    dfdt_Nu[ell] =
                        -dlogHpdx * ckoverHp / (2.0 * ell + 1) * (ell * Nu[ell - 1] - (ell + 1) * Nu[ell + 1]);
                } else {
                    dfdt_Nu[ell] = -dlogHpdx * ckoverHp * Nu[ell - 1] - (ell + 1) * doneoveretaHpdx * Nu[ell];
                }
            }

            // Now for the long part, set the derivatives...
            DVector2D jac(psinfo.n_tot, DVector(psinfo.n_tot, 0.0));

            // delta_cdm
            double * cur = &jac[psinfo.index_delta_cdm][0];
            cur[psinfo.index_delta_cdm] = -3.0 * PhiH0H0term * a * OmegaCDM;
            cur[psinfo.index_delta_b] = -3.0 * PhiH0H0term * a * OmegaB;
            cur[psinfo.index_v_cdm] = ckoverHp;
            cur[psinfo.index_Phi] = 3.0 + ckoverHp * ckoverHp;
            cur[psinfo.index_theta_start + 0] = -3.0 * PhiH0H0term * 4.0 * OmegaR;
            if (psinfo.n_ell_nu > 0) {
                cur[psinfo.index_nu_start + 0] = -3.0 * PhiH0H0term * 4.0 * OmegaNu;
            }
            cur[psinfo.index_theta_start + 2] = 3.0 * Psiterm * OmegaR;
            if (psinfo.n_ell_nu > 2) {
                cur[psinfo.index_nu_start + 2] = 3.0 * Psiterm * OmegaNu;
            }

            // delta_b
            cur = &jac[psinfo.index_delta_b][0];
            cur[psinfo.index_delta_b] = -3.0 * PhiH0H0term * a * OmegaB;
            cur[psinfo.index_delta_cdm] = -3.0 * PhiH0H0term * a * OmegaCDM;
            cur[psinfo.index_v_b] = ckoverHp;
            cur[psinfo.index_Phi] = 3.0 * (1.0 + ckoverHp * ckoverHp / 3.0);
            cur[psinfo.index_theta_start + 0] = -3.0 * PhiH0H0term * 4.0 * OmegaR;
            cur[psinfo.index_theta_start + 2] = 3.0 * Psiterm * OmegaR;
            if (psinfo.n_ell_nu > 2) {
                cur[psinfo.index_nu_start + 2] = 3.0 * Psiterm * OmegaNu;
            }
            if (psinfo.n_ell_nu > 0) {
                cur[psinfo.index_nu_start + 0] = -3.0 * PhiH0H0term * 4.0 * OmegaNu;
            }

            // v_cdm
            cur = &jac[psinfo.index_v_cdm][0];
            cur[psinfo.index_v_cdm] = -1.0;
            cur[psinfo.index_Phi] = ckoverHp;
            cur[psinfo.index_theta_start + 2] = ckoverHp * Psiterm * OmegaR;
            if (psinfo.n_ell_nu > 2) {
                cur[psinfo.index_nu_start + 2] = ckoverHp * Psiterm * OmegaNu;
            }

            // v_b
            cur = &jac[psinfo.index_v_b][0];
            cur[psinfo.index_v_b] = -1.0 + dtaudx * R;
            cur[psinfo.index_Phi] = ckoverHp;
            cur[psinfo.index_theta_start + 2] = ckoverHp * Psiterm * OmegaR;
            if (psinfo.n_ell_nu > 2) {
                cur[psinfo.index_nu_start + 2] = ckoverHp * Psiterm * OmegaNu;
            }
            cur[psinfo.index_theta_start + 1] = 3.0 * dtaudx * R;

            // Phi
            cur = &jac[psinfo.index_Phi][0];
            cur[psinfo.index_delta_cdm] = PhiH0H0term * a * OmegaCDM;
            cur[psinfo.index_delta_b] = PhiH0H0term * a * OmegaB;
            cur[psinfo.index_Phi] = -1.0 - ckoverHp * ckoverHp / 3.0;
            cur[psinfo.index_theta_start + 0] = PhiH0H0term * 4.0 * OmegaR;
            cur[psinfo.index_theta_start + 2] = -Psiterm * OmegaR;
            if (psinfo.n_ell_nu > 0) {
                cur[psinfo.index_nu_start + 0] = PhiH0H0term * 4.0 * OmegaNu;
            }
            if (psinfo.n_ell_nu > 2) {
                cur[psinfo.index_nu_start + 2] = -Psiterm * OmegaNu;
            }

            // Theta_ell
            for (int ell = 0; ell < psinfo.n_ell_theta; ell++) {
                cur = &jac[psinfo.index_theta_start + ell][0];
                if (ell == 0) {
                    cur[psinfo.index_delta_cdm] = -PhiH0H0term * a * OmegaCDM;
                    cur[psinfo.index_delta_b] = -PhiH0H0term * a * OmegaB;
                    cur[psinfo.index_Phi] = 1.0 + ckoverHp * ckoverHp / 3.0;
                    cur[psinfo.index_theta_start + 0] = -PhiH0H0term * 4.0 * OmegaR;
                    cur[psinfo.index_theta_start + 1] = -ckoverHp;
                    cur[psinfo.index_theta_start + 2] = Psiterm * OmegaR;
                    if (psinfo.n_ell_nu > 0) {
                        cur[psinfo.index_nu_start + 0] = -PhiH0H0term * 4.0 * OmegaNu;
                    }
                    if (psinfo.n_ell_nu > 2) {
                        cur[psinfo.index_nu_start + 2] = Psiterm * OmegaNu;
                    }
                } else if (ell == 1) {
                    cur[psinfo.index_v_b] = dtaudx / 3.0;
                    cur[psinfo.index_Phi] = -ckoverHp;
                    cur[psinfo.index_theta_start + 0] = ckoverHp;
                    cur[psinfo.index_theta_start + 1] = dtaudx;
                    cur[psinfo.index_theta_start + 2] = -2.0 * ckoverHp / 3.0 - ckoverHp * Psiterm * OmegaR;
                    if (psinfo.n_ell_nu > 2) {
                        cur[psinfo.index_nu_start + 2] = -ckoverHp * Psiterm * OmegaNu;
                    }
                } else if (ell < psinfo.n_ell_theta - 1) {
                    cur[psinfo.index_theta_start + ell - 1] = ell / (2.0 * ell + 1) * ckoverHp;
                    cur[psinfo.index_theta_start + ell] = dtaudx * (1.0 - static_cast<double>(ell == 2) / 10.0);
                    cur[psinfo.index_theta_start + ell + 1] = -(ell + 1) / (2.0 * ell + 1) * ckoverHp;
                    if (psinfo.n_ell_theta_p > 0) {
                        cur[psinfo.index_theta_p_start + 0] = -dtaudx / 10.0 * static_cast<double>(ell == 2);
                    }
                    if (psinfo.n_ell_theta_p > 2) {
                        cur[psinfo.index_theta_p_start + 2] = -dtaudx / 10.0 * static_cast<double>(ell == 2);
                    }
                } else {
                    cur[psinfo.index_theta_start + ell - 1] = ckoverHp;
                    cur[psinfo.index_theta_start + ell] = -(ell + 1) / (etaHp) + dtaudx;
                }
            }

            // Theta_p_ell
            for (int ell = 0; ell < psinfo.n_ell_theta_p; ell++) {
                cur = &jac[psinfo.index_theta_p_start + ell][0];
                if (ell == 0) {
                    cur[psinfo.index_theta_start + 2] = -dtaudx / 2.0;
                    cur[psinfo.index_theta_p_start + 0] = dtaudx / 2.0;
                    cur[psinfo.index_theta_p_start + 1] = -ckoverHp;
                    cur[psinfo.index_theta_p_start + 2] = -dtaudx / 2.0;
                } else if (ell < psinfo.n_ell_theta_p - 1) {
                    cur[psinfo.index_theta_start + 2] = -dtaudx / 10.0 * static_cast<double>(ell == 2);
                    cur[psinfo.index_theta_p_start + ell - 1] = ell * ckoverHp / (2.0 * ell + 1);
                    cur[psinfo.index_theta_p_start + ell] = dtaudx * (1.0 - static_cast<double>(ell == 2) / 10.0);
                    cur[psinfo.index_theta_p_start + ell + 1] = -(ell + 1) * ckoverHp / (2.0 * ell + 1);
                    cur[psinfo.index_theta_p_start + 0] = -dtaudx / 10.0 * static_cast<double>(ell == 2);
                } else {
                    cur[psinfo.index_theta_p_start + ell - 1] = ckoverHp;
                    cur[psinfo.index_theta_p_start + ell] = -(ell + 1) / (etaHp) + dtaudx;
                }
            }

            // Nu
            for (int ell = 0; ell < psinfo.n_ell_nu; ell++) {
                cur = &jac[psinfo.index_nu_start + ell][0];
                if (ell == 0) {
                    cur[psinfo.index_delta_cdm] = -PhiH0H0term * a * OmegaCDM;
                    cur[psinfo.index_delta_b] = -PhiH0H0term * a * OmegaB;
                    cur[psinfo.index_Phi] = 1.0 + ckoverHp * ckoverHp / 3.0;
                    cur[psinfo.index_nu_start + 0] = -PhiH0H0term * 4.0 * OmegaNu;
                    cur[psinfo.index_nu_start + 1] = -ckoverHp;
                    cur[psinfo.index_nu_start + 2] = Psiterm * OmegaNu;
                    cur[psinfo.index_theta_start + 0] = -PhiH0H0term * 4.0 * OmegaR;
                    cur[psinfo.index_theta_start + 2] = Psiterm * OmegaR;
                } else if (ell == 1) {
                    cur[psinfo.index_Phi] = -ckoverHp / 3.0;
                    cur[psinfo.index_nu_start + 0] = ckoverHp / 3.0;
                    cur[psinfo.index_nu_start + 2] =
                        -(ell + 1) / (2.0 * ell + 1) * ckoverHp - ckoverHp * Psiterm * OmegaNu;
                    cur[psinfo.index_theta_start + 2] = -ckoverHp * Psiterm * OmegaR;
                } else if (ell < psinfo.n_ell_nu - 1) {
                    cur[psinfo.index_nu_start + ell - 1] = ell / (2.0 * ell + 1) * ckoverHp;
                    cur[psinfo.index_nu_start + ell + 1] = -(ell + 1) / (2.0 * ell + 1) * ckoverHp;
                } else {
                    cur[psinfo.index_nu_start + ell - 1] = ckoverHp;
                    cur[psinfo.index_nu_start + ell] = -(ell + 1) / (etaHp);
                }
            }

            // Assign jacobian
            for (int i = 0; i < psinfo.n_tot; i++) {
                for (int j = 0; j < psinfo.n_tot; j++) {
                    dfdy[i * psinfo.n_tot + j] = jac[i][j];
                }
            }

            // Utils::EndTiming("PERT::jacobian");

            return GSL_SUCCESS;
        }

        // Derivatives in the full regime
        int Perturbations::rhs_full_ode(double x, double k, const double * y, double * dydx) {
            const int n_ell_nu = psinfo.n_ell_nu;
            const int n_ell_theta = psinfo.n_ell_theta;
            const int n_ell_theta_p = psinfo.n_ell_theta_p;
            assert(n_ell_theta >= 2);

            //=============================================================================
            // Reference and spans to the perturbations quantities in the full regime
            // Using span to enable bound checks
            //=============================================================================
            const double & delta_cdm = y[psinfo.index_delta_cdm];
            const double & delta_b = y[psinfo.index_delta_b];
            const double & v_cdm = y[psinfo.index_v_cdm];
            const double & v_b = y[psinfo.index_v_b];
            const double & Phi = y[psinfo.index_Phi];
            const double * Theta = &y[psinfo.index_theta_start];
            const double * Theta_p = &y[psinfo.index_theta_p_start];
            const double * Nu = &y[psinfo.index_nu_start];
            // tcb::span<const double> Theta  (y + psinfo.index_theta_start,   n_ell_theta);
            // tcb::span<const double> Theta_p(y + psinfo.index_theta_p_start, n_ell_theta_p);
            // tcb::span<const double> Nu     (y + psinfo.index_nu_start,      n_ell_nu);

            double & ddelta_cdmdx = dydx[psinfo.index_delta_cdm];
            double & ddelta_bdx = dydx[psinfo.index_delta_b];
            double & dv_cdmdx = dydx[psinfo.index_v_cdm];
            double & dv_bdx = dydx[psinfo.index_v_b];
            double & dPhidx = dydx[psinfo.index_Phi];
            double * dThetadx = &dydx[psinfo.index_theta_start];
            double * dTheta_pdx = &dydx[psinfo.index_theta_p_start];
            double * dNudx = &dydx[psinfo.index_nu_start];
            // tcb::span<double> dThetadx  (dydx + psinfo.index_theta_start,   n_ell_theta);
            // tcb::span<double> dTheta_pdx(dydx + psinfo.index_theta_p_start, n_ell_theta_p);
            // tcb::span<double> dNudx     (dydx + psinfo.index_nu_start,      n_ell_nu);
            //=============================================================================

            //=============================================================================
            // Cosmological parameters and variables
            //=============================================================================
            const double OmegaB = cosmo->get_OmegaB();
            const double OmegaCDM = cosmo->get_OmegaCDM();
            const double OmegaR = cosmo->get_OmegaR();
            const double OmegaNu = cosmo->get_OmegaNu();
            const double H0 = cosmo->get_H0();
            const double Hp = cosmo->Hp_of_x(x);
            const double eta = cosmo->eta_of_x(x);
            const double etaHp = eta * Hp / Constants.c;
            const double a = std::exp(x);

            //=============================================================================
            // Recombination variables
            //=============================================================================
            const double dtaudx = rec->dtaudx_of_x(x);
            const double ckoverHp = Constants.c * k / Hp;
            const double ckoverH0 = Constants.c * k / H0;

            //=============================================================================
            // The lowest multipoles (some of them might not be present)
            //=============================================================================
            const double Nu0 = n_ell_nu > 0 ? Nu[0] : 0.0;
            const double Nu2 = n_ell_nu > 2 ? Nu[2] : 0.0;
            const double Theta2 = n_ell_theta > 2 ? Theta[2] : 0.0;
            const double Theta_p0 = n_ell_theta_p > 0 ? Theta_p[0] : 0.0;
            const double Theta_p2 = n_ell_theta_p > 2 ? Theta_p[2] : 0.0;

            //=============================================================================

            // The R and Pi factors
            const double R = OmegaB > 0.0 ? 4.0 / 3.0 * OmegaR / OmegaB / a : 0.0;
            const double Pi = Theta2 + Theta_p0 + Theta_p2;
            const double sound_speed_squared = rec->get_baryon_sound_speed_squared(x);

            // The second gravitational potential
            const double Psi = -Phi - 12.0 / (ckoverH0 * ckoverH0 * a * a) * (OmegaR * Theta2 + OmegaNu * Nu2);

            //=============================================================================
            // SET: Gravitational potential
            //=============================================================================
            dPhidx = Psi - ckoverHp * ckoverHp / 3.0 * Phi +
                     0.5 * (H0 * H0) / (Hp * Hp * a * a) *
                         (OmegaCDM * delta_cdm * a + OmegaB * delta_b * a + 4.0 * (OmegaR * Theta[0] + OmegaNu * Nu0));

            //=============================================================================
            // SET: Baryons and CDM
            //=============================================================================
            ddelta_cdmdx = ckoverHp * v_cdm - 3.0 * dPhidx;
            ddelta_bdx = ckoverHp * v_b - 3.0 * dPhidx;
            dv_cdmdx = -v_cdm - ckoverHp * Psi;
            dv_bdx =
                -v_b - ckoverHp * Psi + ckoverHp * sound_speed_squared * delta_b + dtaudx * R * (3.0 * Theta[1] + v_b);

            //=============================================================================
            // SET: Photon multipoles (Theta_ell)
            //=============================================================================
            for (int ell = 0; ell < n_ell_theta; ell++) {
                const double prefac = ckoverHp / (2.0 * ell + 1);
                if (ell == 0) {
                    dThetadx[ell] = prefac * (-(ell + 1) * Theta[ell + 1]);
                } else if (ell < n_ell_theta - 1) {
                    dThetadx[ell] = prefac * (ell * Theta[ell - 1] - (ell + 1) * Theta[ell + 1]);
                } else {
                    dThetadx[ell] = ckoverHp * Theta[ell - 1] - (ell + 1) / (etaHp)*Theta[ell];
                }

                // Add general sources
                dThetadx[ell] += dtaudx * Theta[ell];

                // Add sources
                if (ell == 0)
                    dThetadx[ell] += -dtaudx * Theta[ell] - dPhidx;
                if (ell == 1)
                    dThetadx[ell] += dtaudx * v_b / 3.0 + ckoverHp / 3.0 * Psi;
                if (ell == 2)
                    dThetadx[ell] += -dtaudx * Pi / 10.0;
            }

            //=============================================================================
            // SET: Photon polarization multipoles (Theta_p_ell)
            //=============================================================================
            for (int ell = 0; ell < n_ell_theta_p; ell++) {
                const double prefac = ckoverHp / (2.0 * ell + 1);
                if (ell == 0) {
                    dTheta_pdx[ell] = prefac * (-(ell + 1) * Theta_p[ell + 1]);
                } else if (ell < n_ell_theta_p - 1) {
                    dTheta_pdx[ell] = prefac * (ell * Theta_p[ell - 1] - (ell + 1) * Theta_p[ell + 1]);
                } else {
                    dTheta_pdx[ell] = ckoverHp * Theta_p[ell - 1] - (ell + 1) / (etaHp)*Theta_p[ell];
                }

                // Add general sources
                dTheta_pdx[ell] += dtaudx * Theta_p[ell];

                // Add sources
                if (ell == 0)
                    dTheta_pdx[ell] += -dtaudx * Pi / 2.0;
                if (ell == 1)
                    dTheta_pdx[ell] += 0.0;
                if (ell == 2)
                    dTheta_pdx[ell] += -dtaudx * Pi / 10.0;
            }

            //=============================================================================
            // SET: Neutrino mutlipoles (Nu_ell)
            //=============================================================================
            for (int ell = 0; ell < n_ell_nu; ell++) {
                const double prefac = ckoverHp / (2.0 * ell + 1);
                if (ell == 0) {
                    dNudx[ell] = prefac * (-(ell + 1) * Nu[ell + 1]);
                } else if (ell < n_ell_nu - 1) {
                    dNudx[ell] = prefac * (ell * Nu[ell - 1] - (ell + 1) * Nu[ell + 1]);
                } else {
                    dNudx[ell] = ckoverHp * Nu[ell - 1] - (ell + 1) / (etaHp)*Nu[ell];
                }

                // Add sources
                if (ell == 0)
                    dNudx[ell] += -dPhidx;
                if (ell == 1)
                    dNudx[ell] += ckoverHp / 3.0 * Psi;
                if (ell == 2)
                    dNudx[ell] += 0.0;
            }

            return GSL_SUCCESS;
        }

        //====================================================
        // Scalar perturbation quantities
        //====================================================
        double Perturbations::get_delta_cdm(const double x, const double k) const { return delta_cdm_spline(x, k); }
        double Perturbations::get_delta_b(const double x, const double k) const { return delta_b_spline(x, k); }
        double Perturbations::get_Delta_M(const double x, const double k) const {
            // Gauge invariant total matter density contrast defined via D^2 Phi = 4piG a^2 rho DeltaM
            // where rho is total clustering energy density (i.e. no Lambda or K)
            const double ckoverHp = Constants.c * k / cosmo->Hp_of_x(x);
            return ckoverHp * ckoverHp * get_Phi(x, k) / (1.5 * (cosmo->get_OmegaM(x) + cosmo->get_OmegaR(x)));
        }
        double Perturbations::get_v_cdm(const double x, const double k) const { return v_cdm_spline(x, k); }
        double Perturbations::get_v_b(const double x, const double k) const { return v_b_spline(x, k); }
        double Perturbations::get_Phi(const double x, const double k) const { return Phi_spline(x, k); }
        double Perturbations::get_Psi(const double x, const double k) const { return Psi_spline(x, k); }
        double Perturbations::get_Pi(const double x, const double k) const { return Pi_spline(x, k); }

        //====================================================
        // Source functions needed for LOS integration
        //====================================================
        double Perturbations::get_Source_T(const double x, const double k) const { return ST_spline(x, k); }
        double Perturbations::get_Source_E(const double x, const double k) const { return SE_spline(x, k); }
        double Perturbations::get_Source_N(const double x, const double k) const { return SN_spline(x, k); }

        //====================================================
        // Individual contributions to the temperature source
        //====================================================
        double Perturbations::get_Source_SW_T(const double x, const double k) const { return SW_spline(x, k); }
        double Perturbations::get_Source_ISW_T(const double x, const double k) const { return ISW_spline(x, k); }
        double Perturbations::get_Source_VEL_T(const double x, const double k) const { return VEL_spline(x, k); }
        double Perturbations::get_Source_POL_T(const double x, const double k) const { return POL_spline(x, k); }

        //====================================================
        // Perturbation multipoles
        //====================================================

        double Perturbations::get_Theta(const double x, const double k, const int ell) const {
            if (ell >= 3 and not pert_spline_all_ells) {
                return 0.0;
            }
            return Theta_spline[ell](x, k);
        }

        double Perturbations::get_Theta_p(const double x, const double k, const int ell) const {
            if (psinfo.n_ell_theta_p == 0)
                return 0.0;
            if (ell >= 3 and not pert_spline_all_ells) {
                return 0.0;
            }
            return Theta_p_spline[ell](x, k);
        }

        double Perturbations::get_Nu(const double x, const double k, const int ell) const {
            if (psinfo.n_ell_nu == 0)
                return 0.0;
            if (ell >= 3 and not pert_spline_all_ells) {
                return 0.0;
            }
            return Nu_spline[ell](x, k);
        }

        //====================================================
        // Transfer functions
        //====================================================

        // The gamma-factor in the N-body gauge
        double Perturbations::get_transfer_gammaNbody(const double x, const double k) const {
            const double dRdx = zeta_spline.deriv_x(x, k);
            const double ddRddx = zeta_spline.deriv_xx(x, k);
            // const double ddRddx   = dzetadx_spline.deriv_x(x,k);
            const double Hp = cosmo->Hp_of_x(x);
            const double dHpdx = cosmo->dHpdx_of_x(x);
            const double ckoverHp = Constants.c * k / Hp;
            const double aniso = get_Psi(x, k) + get_Phi(x, k);
            const double gamma = -3.0 * (ddRddx + (dHpdx / Hp + 1.0) * dRdx) / (ckoverHp * ckoverHp) - aniso;
            return gamma;
        }

        // Gauge invariant curvature perturbation
        double Perturbations::get_transfer_zeta(const double x, const double k) const { return zeta_spline(x, k); }

        // veldiv_cdm = - ck get_v_cdm(x, k)
        double Perturbations::get_transfer_Delta_cdm(const double x, const double k) const {
            return (get_delta_cdm(x, k) - 3.0 * cosmo->Hp_of_x(x) * get_v_cdm(x, k) / (Constants.c * k)) / (k * k);
        }

        double Perturbations::get_transfer_Delta_b(const double x, const double k) const {
            return (get_delta_b(x, k) - 3.0 * cosmo->Hp_of_x(x) * get_v_b(x, k) / (Constants.c * k)) / (k * k);
        }

        double Perturbations::get_transfer_Delta_cb(const double x, const double k) const {
            const double OmegaB = cosmo->get_OmegaB();
            const double OmegaCDM = cosmo->get_OmegaCDM();
            return (get_transfer_Delta_b(x, k) * OmegaB + get_transfer_Delta_cdm(x, k) * OmegaCDM) /
                   (OmegaB + OmegaCDM);
        }

        double Perturbations::get_transfer_Delta_M(const double x, const double k) const {
            return get_Delta_M(x, k) / (k * k);
        }

        double Perturbations::get_transfer_Delta_R(const double x, const double k) const {
            return 4.0 * (get_Theta(x, k, 0) + 3.0 * cosmo->Hp_of_x(x) * get_Theta(x, k, 1) / (Constants.c * k)) /
                   (k * k);
        }

        double Perturbations::get_transfer_Delta_Rtot(const double x, const double k) const {
            const double OmegaR = cosmo->get_OmegaR();
            const double OmegaNu = cosmo->get_OmegaNu();
            return (get_transfer_Delta_R(x, k) * OmegaR + get_transfer_Delta_Nu(x, k) * OmegaNu) / (OmegaR + OmegaNu);
        }

        double Perturbations::get_transfer_Delta_Nu(const double x, const double k) const {
            return 4.0 * (get_Nu(x, k, 0) + 3.0 * cosmo->Hp_of_x(x) * get_Nu(x, k, 1) / (Constants.c * k)) / (k * k);
        }

        double Perturbations::get_transfer_v_R(const double x, const double k) const {
            // Minus sign for consistency with CAMB (our v is -v in CAMB)
            const double vR = -(-3.0 * get_Theta(x, k, 1));
            return -(Constants.c * k) / cosmo->Hp_of_x(x) * vR / (k * k);
        }

        double Perturbations::get_transfer_v_Nu(const double x, const double k) const {
            // Minus sign for consistency with CAMB
            const double vNu = -(-3.0 * get_Nu(x, k, 1));
            return -(Constants.c * k) / cosmo->Hp_of_x(x) * vNu / (k * k);
        }

        double Perturbations::get_transfer_v_cdm(const double x, const double k) const {
            // Minus sign for consistency with CAMB
            const double vCDM = -get_v_cdm(x, k);
            return -(Constants.c * k) / cosmo->Hp_of_x(x) * vCDM / (k * k);
        }

        double Perturbations::get_transfer_v_b(const double x, const double k) const {
            // Minus sign for consistency with CAMB
            const double vB = -get_v_b(x, k);
            return -(Constants.c * k) / cosmo->Hp_of_x(x) * vB / (k * k);
        }

        double Perturbations::get_transfer_v_b_v_c(const double x, const double k) const {
            // Minus sign for consistency with CAMB
            return -(get_v_b(x, k) - get_v_cdm(x, k)) / (k * k);
        }

        double Perturbations::get_transfer_Phi(const double x, const double k) const { return Phi_spline(x, k); }

        double Perturbations::get_transfer_Psi(const double x, const double k) const { return Psi_spline(x, k); }

        double Perturbations::get_transfer_Weyl(const double x, const double k) const {
            return (get_transfer_Psi(x, k) - get_transfer_Phi(x, k)) / 2.0;
        }

        //====================================================
        // Other stuff
        //====================================================

        double Perturbations::get_kmin() const { return k_min; }
        double Perturbations::get_kmax() const { return k_max; }

        double Perturbations::get_x_start() const { return x_start; }

        double Perturbations::get_x_end() const { return x_end; }
    } // namespace COSMOLOGY
} // namespace FML
