#include "PowerSpectrum.h"
#include <numeric>

namespace FML {
    namespace COSMOLOGY {

        //====================================================
        // Constructors
        //====================================================

        PowerSpectrum::PowerSpectrum(std::shared_ptr<BackgroundCosmology> cosmo,
                                     std::shared_ptr<RecombinationHistory> rec,
                                     std::shared_ptr<Perturbations> pert,
                                     ParameterMap & p)
            : cosmo(cosmo), rec(rec), pert(pert) {

            // Read parameters
            A_s = p.get<double>("A_s");
            n_s = p.get<double>("n_s");
            kpivot = p.get<double>("kpivot");
            ell_max = p.get<int>("ell_max");
            x_cell = std::log(1.0 / (1.0 + p.get<double>("CellOutputRedshift")));
            compute_temperature_cells = p.get<bool>("compute_temperature_cells");
            compute_polarization_cells = p.get<bool>("compute_polarization_cells");
            compute_neutrino_cells = p.get<bool>("compute_neutrino_cells");
            compute_lensing_cells = p.get<bool>("compute_lensing_cells");
            compute_corr_function = p.get<bool>("compute_corr_function");
            bessel_nsamples_per_osc = p.get<int>("bessel_nsamples_per_osc");
            los_integration_loga_nsamples = p.get<int>("los_integration_loga_nsamples");
            los_integration_nsamples_per_osc = p.get<int>("los_integration_nsamples_per_osc");
            cell_nsamples_per_osc = p.get<int>("cell_nsamples_per_osc");
            kmax = p.get<double>("keta_max") / cosmo->eta_of_x(0.0);

            // eta and tau at the output redshift
            // This is today, but in case we want to output how Cell looks like at a different redshift
            eta0 = cosmo->eta_of_x(x_cell);
            exptau0 = std::exp(rec->tau_of_x(x_cell));

            // Set min and max k to compute F_ell's on
            kmin = pert->get_kmin();
            kmax = std::min(pert->get_kmax(), kmax);

            // Create ell-array to compute Cells on
            bool sample_all_ells = false;
            ells = DVector();
            if (sample_all_ells) {
                ells = FML::MATH::linspace(2, ell_max, ell_max - 1);
            } else {
                for (int ell = 2; ell < ell_max;) {
                    ells.push_back(ell);

                    // Low ell sampling
                    if (ell < 10)
                        ell += 1;
                    else if (ell < 20)
                        ell += 2;
                    else if (ell < 50)
                        ell += 5;

                    // Acoustic peak sampling
                    else if (ell < 100)
                        ell += 10;
                    else if (ell < 300)
                        ell += 20;
                    else if (ell < 1000)
                        ell += 25;

                    // Damping tail sampling
                    else
                        ell += 50;
                }
                ells.push_back(ell_max);
            }
        }

        void PowerSpectrum::info() const {
            if (FML::ThisTask > 0)
                return;
            std::cout << "\n";
            std::cout << "============================================\n";
            std::cout << "Info about PowerSpectrum class:\n";
            std::cout << "============================================\n";
            std::cout << "A_s:         " << A_s << "\n";
            std::cout << "n_s:         " << n_s << "\n";
            std::cout << "kpivot:      " << kpivot * Constants.Mpc << " 1/Mpc\n";
            std::cout << "kmin:        " << kmin * Constants.Mpc << " 1/Mpc\n";
            std::cout << "kmax:        " << kmax * Constants.Mpc << " 1/Mpc\n";
            std::cout << "ell_max:     " << ell_max << "\n";
            std::cout << "Nells:       " << ells.size() << "\n";
            std::cout << "============================================\n";
            std::cout << "\n";

            timer.PrintAllTimings();
        }

        //=====================================================================
        // x = log(a) and k is the wavenumber in user units
        // The type can be:
        // B = baryons       CDM = cold dark matter    CB   = baryons + CDM
        // R = photons       Nu  = massless neutrinos  Rtot = total radiation
        // M = total matter
        //=====================================================================
        double PowerSpectrum::get_power_spectrum(double x, double k, std::string type) const {
            // BBKS fit used to interpolate outside of the k's we have
            static double aeq = cosmo->get_OmegaRtot() / cosmo->get_OmegaM();
            static double keq = cosmo->Hp_of_x(std::log(aeq)) / Constants.c;
            static auto bbks_fit = [](double k) {
                const double arg = k / keq;
                return std::log(1.0 + 0.171 * arg) / (0.171 * arg) *
                       std::pow(1 + 0.284 * arg + std::pow(1.18 * arg, 2) + std::pow(0.399 * arg, 3) +
                                    std::pow(0.490 * arg, 4),
                                -0.25);
            };

            const double kmax_pert = pert->get_kmax();
            const double k_true = k;
            k = std::max(kmin, std::min(k, kmax_pert));

            // Different cases: The gauge invariant density perturbation is
            // Delta = delta - 3(1+w)Hv/k given by the transfer functions
            // We fetch here the in-bounds value and then extrapolate it later
            // if needed
            double Delta = 0.0;
            if (type == "B") {
                // Baryon P(k)
                Delta = pert->get_transfer_Delta_b(x, k) * k * k;
            } else if (type == "CDM") {
                // Cold Dark Matter P(k)
                Delta = pert->get_transfer_Delta_cdm(x, k) * k * k;
            } else if (type == "CB") {
                // Baryon-Cold Dark Matter P(k)
                Delta = pert->get_transfer_Delta_cb(x, k) * k * k;
            } else if (type == "R") {
                // Photon density P(k): vR = -3Theta1 and delta = 4Theta0
                Delta = pert->get_transfer_Delta_R(x, k) * k * k;
            } else if (type == "Nu") {
                // Massless neutrino P(k)
                Delta = pert->get_transfer_Delta_Nu(x, k) * k * k;
            } else if (type == "Rtot") {
                // Total radiation P(k)
                Delta = pert->get_transfer_Delta_Rtot(x, k) * k * k;
            } else if (type == "M") {
                // Total matter P(k)
                Delta = pert->get_transfer_Delta_M(x, k) * k * k;
            } else {
                throw std::runtime_error("Unknown type in get_power_spectrum [" + type + "]");
            }

            // Asymptotic behavior
            if (k_true < kmin or k_true > kmax_pert) {
                if (k_true < kmin) {
                    Delta *= (k_true / kmin) * (k_true / kmin);
                }
                if (k_true > kmax_pert) {
                    // As derived in BBKS 1986
                    bool radiation = (type == "Nu" or type == "R" or type == "Rtot");
                    if (not radiation)
                        Delta *= bbks_fit(k_true) / bbks_fit(kmax_pert);
                }
            }

            return Delta * Delta * primordial_power_spectrum(k_true);
        }

        // Evaluate P(k) for all k's we integrate perturbations on
        std::pair<DVector, DVector>
        PowerSpectrum::get_power_spectrum_array(double x, int npts, std::string type) const {
            const double kmax_pert = pert->get_kmax();
            DVector k_array = FML::MATH::linspace(std::log(kmin), std::log(kmax_pert), npts);
            for (auto & k : k_array)
                k = std::exp(k);
            DVector pofk_array;
            for (auto & k : k_array) {
                pofk_array.push_back(get_power_spectrum(x, k, type));
            }
            return {k_array, pofk_array};
        }

        // The full matter power-spectrum
        double PowerSpectrum::get_matter_power_spectrum(double x, double k) const {
            return get_power_spectrum(x, k, "M");
        }

        double PowerSpectrum::primordial_power_spectrum_dimless(double k) const {
            return A_s * std::pow(k / kpivot, n_s - 1.0);
        }

        double PowerSpectrum::primordial_power_spectrum(double k) const {
            return 2.0 * M_PI * M_PI / (k * k * k) * primordial_power_spectrum_dimless(k);
        }

        void PowerSpectrum::generate_bessel_function_splines(double xmax, int nsamples_per_osc) {
            timer.StartTiming("POW::making bessel splines");
            if (FML::ThisTask == 0)
                std::cout << "Bessel splines\n";

            // The x-array we will compute it over
            const double deltax = 2.0 * M_PI / nsamples_per_osc;
            const int npts = int(xmax / deltax);
            DVector x_array = FML::MATH::linspace(0.0, xmax, npts);

            // Compute j_ell(x) for all x and ell
            const int ellmax = ells[ells.size() - 1];
            DVector2D results(ells.size(), DVector(npts));
#ifdef USE_OMP
#pragma omp parallel for schedule(dynamic, 1)
#endif
            for (int i = 0; i < npts; i++) {
                auto data = FML::MATH::j_ell_array(ellmax, x_array[i]);
                // Store the data we need
                for (size_t j = 0; j < ells.size(); j++)
                    results[j][i] = data[int(ells[j])];
            }

            // Make splines
            j_ell_splines = std::vector<Spline>(ells.size());
            for (size_t j = 1; j < ells.size(); j++)
                j_ell_splines[j].create(x_array, results[j]);

            // Higher resolution for ell=2
            int npts2 = int(1000 * 32 / 2.0 / M_PI);
            DVector x_array2 = FML::MATH::linspace(0, 1000.0, npts2);
            DVector jell2(npts2, 0.0);
#ifdef USE_OMP
#pragma omp parallel for schedule(dynamic, 1)
#endif
            for (int i = 1; i < npts2; i++) {
                double x = x_array2[i];
                double s = std::sin(x);
                double c = std::cos(x);
                jell2[i] = ((3.0 - x * x) * s - 3.0 * x * c) / (x * x * x);
            }
            j_ell_splines[0].create(x_array2, jell2);

            // For using the splines in general: allows for direct index
            // lookup for a given ell
            DVector index(ells);
            std::iota(index.begin(), index.end(), 0);
            index_of_ells_spline.create(ells, index, "index_of_ell");

            timer.EndTiming("POW::making bessel splines");
        }

        DVector2D PowerSpectrum::line_of_sight_integration_single(
            DVector & x_array,
            DVector & k_array,
            std::function<double(double, double)> & source_function,
            [[maybe_unused]] std::function<double(double, double)> & aux_norm) {
            timer.StartTiming("POW::LOS integration");

            const int nells = ells.size();

            // Make result struct
            DVector2D result = DVector2D(k_array.size(), DVector(ells.size(), 0.0));

            // When the bessel-function is irrelevant
            DVector kcut(ells.size());
            for (size_t i = 0; i < ells.size(); i++) {
                const double arg_min = ells[i] <= 10 ? 0.0 : (1.0 - 2.6 / std::sqrt(ells[i])) * ells[i];
                kcut[i] = arg_min / eta0;
            }

            // When the source function terms with g can be ignored
            // and adding in the region where ISW is relevant
            std::vector<int> g_relevant(x_array.size());
            for (size_t i = 0; i < x_array.size(); i++) {
                g_relevant[i] = int((rec->g_tilde_of_x(x_array[i]) > 1e-4));
                if (x_array[i] > -3.0)
                    g_relevant[i] = 1;
            }

            DVector chi_values(x_array.size());
            for (size_t ix = 0; ix < x_array.size(); ix++)
                chi_values[ix] = eta0 - cosmo->eta_of_x(x_array[ix]);

            // The k-values we loop over (doing it like this to ease the
            // parallelization)
            std::vector<int> ik_list(k_array.size());
            std::iota(ik_list.begin(), ik_list.end(), 0);
#ifdef USE_MPI
            // Compute what k's to deal with on the local task
            if (FML::NTasks > 1) {
                ik_list.clear();
                for (int i = 0; i < int(k_array.size()); i++) {
                    if (i % FML::NTasks == FML::ThisTask) {
                        ik_list.push_back(i);
                    }
                }
            }
#endif

#ifdef USE_OMP
#pragma omp parallel for schedule(dynamic, 1)
#endif
            for (size_t ii = 0; ii < ik_list.size(); ii++) {
                const int ik = ik_list[ii];
                const double k = k_array[ik];

#ifdef USE_ODESOLVER_LOS

                ODEFunction deriv = [&](double x, const double * y, double * dydx) {
                    const double eta = cosmo->eta_of_x(x);
                    const double arg = k * (eta0 - eta);
                    const double source = source_function(x, k);
                    for (int i = 0; i < nells; i++) {
                        const Spline & jell_spline = j_ell_splines[i];
                        const auto xrange = jell_spline.get_xrange();
                        if (arg > xrange.first and arg < xrange.second) {
                            const double jell = jell_spline(arg);
                            dydx[i] = source * jell * aux_norm(k, ells[i]);
                        } else {
                            dydx[i] = 0.0;
                        }
                    }
                    return GSL_SUCCESS;
                };

                // Solve the general line of sight integral F_ell(k) = Int dx jell(k(eta-eta0)) * S(x,k)
                DVector los_ini(nells, 0.0);
                ODESolver los_ode(FIDUCIAL_HSTART_ODE_LOS, FIDUCIAL_ABSERR_ODE_LOS, FIDUCIAL_RELERR_ODE_LOS);
                los_ode.solve(deriv, x_array, los_ini);

                auto data = los_ode.get_final_data();
                for (size_t i = 0; i < ells.size(); i++) {
                    data[i] /= aux_norm(k_array[ik], ells[i]);
                }

#else

                DVector data(ells.size(), 0.0);
                for (size_t ix = 1; ix < x_array.size(); ix++) {
                    if (g_relevant[ix] == 0 and x_array[ix] < -4.0)
                        continue;
                    const double dx = x_array[ix] - x_array[ix - 1];
                    const double arg = k * chi_values[ix];
                    const double sourcedx = source_function(x_array[ix], k) * dx;

                    for (int i = 0; i < nells; i++) {
                        if (k < kcut[i])
                            continue;
                        if (ells[i] > 30.0 and g_relevant[ix] == 0)
                            continue;
                        const Spline & jell_spline = j_ell_splines[i];
                        const double jell = jell_spline(arg);
                        data[i] += jell * sourcedx;
                    }
                }

#endif

                result[ik] = DVector(data.begin(), data.end());
            }

#ifdef USE_MPI
            // Its not that much data so we simply send all the data from all to all tasks and add up
            for (size_t ik = 0; ik < result.size(); ik++) {
                MPI_Allreduce(MPI_IN_PLACE, result[ik].data(), result[ik].size(), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
            }
#endif

            timer.EndTiming("POW::LOS integration");
            return result;
        }

        void PowerSpectrum::line_of_sight_integration(DVector & k_array) {
            const int nells = ells.size();
            const int n_k_total = k_array.size();

            // The x-array. For photon multipoles we only need to start right
            // before LSS as g = 0 otherwise
            const double x_start = rec->get_x_start_rec_array();
            const double x_start_photons = rec->get_xstar() - 0.5;
            const double x_end = x_cell;
            const int n = los_integration_loga_nsamples;
            auto x_array = FML::MATH::linspace(x_start, x_end, n);
            auto x_array_photons = FML::MATH::linspace(x_start_photons, x_end, n);

            //============================================================================
            // Solve for Theta_ell(k,x=xcell)
            //============================================================================
            if (compute_temperature_cells) {
                if (FML::ThisTask == 0)
                    std::cout << "Temperature ells\n";
                timer.StartTiming("POW::LOS integration thetaT");
                std::function<double(double, double)> source_function_T = [&](double x, double k) {
                    return exptau0 * pert->get_Source_T(x, k);
                };
                // Normalize while solving to make sure all ells have similar order of magnitude
                std::function<double(double, double)> solve_T_norm = []([[maybe_unused]] double k, double ell) {
                    return ell / 200.0;
                };
                DVector2D thetaT_ell_of_k =
                    line_of_sight_integration_single(x_array_photons, k_array, source_function_T, solve_T_norm);
                thetaT_ell_of_k_spline.create(k_array, ells, thetaT_ell_of_k, "thetaT_ell_of_k_spline");
                timer.EndTiming("POW::LOS integration thetaT");
            }

            //============================================================================
            // Solve for ThetaE_ell(k,x=xcell)
            //============================================================================
            if (compute_polarization_cells) {
                if (FML::ThisTask == 0)
                    std::cout << "Polarization ells\n";
                timer.StartTiming("POW::LOS integration thetaE");
                std::function<double(double, double)> source_function_E = [&](double x, double k) {
                    return exptau0 * pert->get_Source_E(x, k);
                };
                // Normalize while solving to make sure all ells have similar order of magnitude
                std::function<double(double, double)> solve_E_norm = []([[maybe_unused]] double k, double ell) {
                    return 1e2 * (ell / 200.0) * (ell / 200.0);
                };
                DVector2D thetaE_ell_of_k =
                    line_of_sight_integration_single(x_array_photons, k_array, source_function_E, solve_E_norm);
                // Add in the prefactor Sqrt((l+2)!/(l-2)!) for ThetaE
                for (int ik = 0; ik < n_k_total; ik++) {
                    for (int i = 0; i < nells; i++) {
                        thetaE_ell_of_k[ik][i] *=
                            std::sqrt((ells[i] + 2) * (ells[i] + 1) * (ells[i] + 0) * (ells[i] - 1));
                    }
                }
                thetaE_ell_of_k_spline.create(k_array, ells, thetaE_ell_of_k, "thetaE_ell_of_k_spline");
                timer.EndTiming("POW::LOS integration thetaE");
            }

            //============================================================================
            // Solve for CMB lensing potential Psi_ell(k,x=xcell)
            //============================================================================
            if (compute_lensing_cells) {
                if (FML::ThisTask == 0)
                    std::cout << "Lensingpotential ells\n";
                timer.StartTiming("POW::LOS integration Phi_lens");
                std::function<double(double, double)> source_function_L = [&](double x, double k) {
                    return 2.0 * pert->get_Psi(x, k) * lensing_source(x, x_cell);
                };
                // Normalize while solving to make sure all ells have similar order of magnitude
                std::function<double(double, double)> solve_lens_norm = []([[maybe_unused]] double k, double ell) {
                    return (ell / 100.0) * (ell / 100.0);
                };
                DVector2D lens_ell_of_k =
                    line_of_sight_integration_single(x_array, k_array, source_function_L, solve_lens_norm);
                lens_ell_of_k_spline.create(k_array, ells, lens_ell_of_k, "lens_ell_of_k_spline");
                timer.EndTiming("POW::LOS integration Phi_lens");
            }

            //============================================================================
            // Solve for Nu_ell(k,x=xcell)
            //============================================================================
            if (compute_neutrino_cells) {
                if (FML::ThisTask == 0)
                    std::cout << "Neutrino ells\n";
                timer.StartTiming("POW::LOS integration Nu");
                std::function<double(double, double)> source_function_N = [&](double x, double k) {
                    return pert->get_Source_N(x, k);
                };
                // Normalize while solving to make sure all ells have similar order of magnitude
                std::function<double(double, double)> solve_nu_norm = []([[maybe_unused]] double k, double ell) {
                    return (ell / 200.0);
                };
                DVector2D Nu_ell_of_k =
                    line_of_sight_integration_single(x_array, k_array, source_function_N, solve_nu_norm);
                Nu_ell_of_k_spline.create(k_array, ells, Nu_ell_of_k, "Nu_ell_of_k_spline");
                timer.EndTiming("POW::LOS integration Nu");
            }
            if (FML::ThisTask == 0)
                std::cout << "Done line of sight integration!\n";
        }

        DVector PowerSpectrum::solve_for_cell_single(DVector & log_k_array,
                                                     std::function<double(double, int)> & integrand,
                                                     [[maybe_unused]] double accuracy_limit) {
            const int nells = ells.size();

#ifdef USE_ODESOLVER_CELL

            ODEFunction deriv = [&](double logk, const double * y, double * dydx) {
                const double k = std::exp(logk);
                for (int i = 0; i < nells; i++) {
                    const double ell = ells[i];
                    dydx[i] = integrand(k, ell);
                }
                return GSL_SUCCESS;
            };

            // The initial condition
            DVector cell_ini(nells, 0.0);

            // Solve the system
            ODESolver cell_ode(1e-3, 0.0, accuracy_limit);
            cell_ode.solve(deriv, log_k_array, cell_ini);

            auto data = cell_ode.get_final_data();

#else

            // Direct summation
            DVector data(ells.size(), 0.0);
#ifdef USE_OMP
#pragma omp parallel for schedule(dynamic, 1)
#endif
            for (int i = 0; i < nells; i++) {
                for (size_t ik = 1; ik < log_k_array.size(); ik++) {
                    const double k = std::exp(log_k_array[ik]);
                    const double dlogk = log_k_array[ik] - log_k_array[ik - 1];
                    data[i] += integrand(k, ells[i]) * dlogk;
                }
            }
#endif

            return data;
        }

        void PowerSpectrum::compute_all_correlation_functions([[maybe_unused]] double xmin,
                                                              [[maybe_unused]] double xmax,
                                                              [[maybe_unused]] int nx) {
#ifdef USE_FFTW
            timer.StartTiming("POW::Correlation function wth FFTW");
            const double rmin = 1.0 * Constants.Mpc;
            const double rmax = 1000.0 * Constants.Mpc;

            //==================================================================
            // Computes the correlation function for all x's and r's
            // We can also use this to compute density profiles of an
            // initial delta-function overdensity
            //==================================================================

            // Types of spectra to compute
            std::vector<std::string> types = {"CDM", "B", "R", "Nu", "M"};

            // Make x_array to compute it on
            const double x_start = xmin;
            const double x_end = xmax;
            auto x_array = FML::MATH::linspace(x_start, x_end, nx);

            // Arrays to store the data in
            std::vector<DVector2D> results(types.size());
            DVector r;

            if (FML::ThisTask == 0)
                std::cout << "Compute all correlation functions at all times\n";

            for (size_t i = 0; i < x_array.size(); i++) {
                const double x = x_array[i];

                // Loop over the different spectra to compute
                for (size_t t = 0; t < types.size(); t++) {

                    // The power-spectrum k^2/2pi^2 P(k)
                    // This is slow... can do it much faster
                    std::function<double(double)> Delta_P = [&](double k) {
                        double pofk = get_power_spectrum(x, k, types[t]) * k * k * k / (2.0 * M_PI * M_PI);
                        // For computing density profiles
                        // pofk = pert->get_transfer_Delta_cdm(x, k) * k * k;
                        return pofk;
                    };

                    // Compute correlation function
                    auto corr = correlation_function_single_fftw(Delta_P, rmin, rmax);

                    // Store the data
                    if (i == 0 and t == 0)
                        r = corr.first;
                    results[t].push_back(corr.second);
                }
            }

            // Make splines (NB: order must agree with that in types above)
            xi_CDM_spline.create(x_array, r, results[0], "xi_CDM_spline");
            xi_B_spline.create(x_array, r, results[1], "xi_B_spline");
            xi_R_spline.create(x_array, r, results[2], "xi_R_spline");
            xi_Nu_spline.create(x_array, r, results[3], "xi_Nu_spline");
            xi_M_spline.create(x_array, r, results[4], "xi_M_spline");

            timer.EndTiming("POW::Correlation function wth FFTW");
#endif
        }

        void PowerSpectrum::output_correlation_function(double x, std::string filename) const {
            if (not compute_corr_function)
                return;

            std::ofstream fp(filename.c_str());
            if (not fp.is_open())
                return;

            auto yrange = xi_CDM_spline.get_yrange();
            double rmin = yrange.first;
            double rmax = yrange.second;
            int npts = 1000;
            auto r = FML::MATH::linspace(std::log(rmin), std::log(rmax), npts);
            for (auto & rr : r)
                rr = std::exp(rr);

            fp << "# r (Mpc/h)   CDM   Baryon   R    Nu    Matter\n";
            const double norm_r = (cosmo->get_h() / Constants.Mpc);
            for (size_t i = 0; i < r.size(); i++) {
                fp << r[i] * norm_r << " ";
                fp << xi_CDM_spline(x, r[i]) << " ";
                fp << xi_B_spline(x, r[i]) << " ";
                fp << xi_R_spline(x, r[i]) << " ";
                fp << xi_Nu_spline(x, r[i]) << " ";
                fp << xi_M_spline(x, r[i]) << " ";
                fp << "\n";
            }
        }

        void PowerSpectrum::solve() {

            //=========================================================================
            // The sampling of the line of sight integration
            //=========================================================================
            const double delta_k = 2.0 * M_PI / los_integration_nsamples_per_osc / eta0;
            const int n_k_total = int((kmax - kmin) / delta_k);
            auto k_array = FML::MATH::linspace(kmin, kmax, n_k_total);

            //=========================================================================
            // The sampling of the Cell integration
            // (Only relevant if we don't solve it as an ODE)
            //=========================================================================
            const double delta_logk = 2.0 * M_PI / cell_nsamples_per_osc / eta0;
            const int n_logk_total = int((kmax - kmin) / delta_logk);
            auto log_k_array = FML::MATH::linspace(std::log(kmin), std::log(kmax), n_logk_total);

            //=========================================================================
            // Compute the real space correlation functions
            //=========================================================================
            if (compute_corr_function)
                compute_all_correlation_functions();

            //=========================================================================
            // Make splines for j_ell
            //=========================================================================
            const double max_arg_bessel = kmax * eta0;
            generate_bessel_function_splines(max_arg_bessel, bessel_nsamples_per_osc);

            //=========================================================================
            // Line of sight integration to get Theta_ell(k)
            //=========================================================================
            line_of_sight_integration(k_array);

            //=========================================================================
            // Integration to get Cell by solving dCell^f/dlogk = Delta(k) * f_ell(k)^2
            //=========================================================================
            timer.StartTiming("POW::integrating Cells");

            if (thetaT_ell_of_k_spline) {
                timer.StartTiming("POW::integrating Cells - TT");
                const double units = std::pow(1e6 * cosmo->get_TCMB(x_cell) / Constants.K, 2);
                std::function<double(double, int)> D_ell_TT_integrand = [&](double k, int ell) {
                    const double normalization = ell * (ell + 1) / (2.0 * M_PI) * units;
                    const double thetaT_ell = thetaT_ell_of_k_spline(k, ell);
                    const double pofk = primordial_power_spectrum_dimless(k);
                    return 4.0 * M_PI * pofk * thetaT_ell * thetaT_ell * normalization;
                };
                auto cell_TT = solve_for_cell_single(log_k_array, D_ell_TT_integrand, 1.0);
                cell_TT_spline.create(ells, cell_TT, "Cell_TT_of_ell");
                timer.EndTiming("POW::integrating Cells - TT");
            }

            if (thetaE_ell_of_k_spline) {
                timer.StartTiming("POW::integrating Cells - EE");
                const double units = std::pow(1e6 * cosmo->get_TCMB(x_cell) / Constants.K, 2);
                std::function<double(double, int)> D_ell_EE_integrand = [&](double k, int ell) {
                    const double normalization = ell * (ell + 1) / (2.0 * M_PI) * units;
                    const double thetaE_ell = thetaE_ell_of_k_spline(k, ell);
                    const double pofk = primordial_power_spectrum_dimless(k);
                    return 4.0 * M_PI * pofk * thetaE_ell * thetaE_ell * normalization;
                };
                auto cell_EE = solve_for_cell_single(log_k_array, D_ell_EE_integrand, 1.0);
                cell_EE_spline.create(ells, cell_EE, "Cell_EE_of_ell");
                timer.EndTiming("POW::integrating Cells - EE");
            }

            if (thetaT_ell_of_k_spline and thetaE_ell_of_k_spline) {
                timer.StartTiming("POW::integrating Cells - TE");
                const double units = std::pow(1e6 * cosmo->get_TCMB(x_cell) / Constants.K, 2);
                std::function<double(double, int)> D_ell_TE_integrand = [&](double k, int ell) {
                    const double normalization = ell * (ell + 1) / (2.0 * M_PI) * units;
                    const double thetaT_ell = thetaT_ell_of_k_spline(k, ell);
                    const double thetaE_ell = thetaE_ell_of_k_spline(k, ell);
                    const double pofk = primordial_power_spectrum_dimless(k);
                    return 4.0 * M_PI * pofk * thetaT_ell * thetaE_ell * normalization;
                };
                auto cell_TE = solve_for_cell_single(log_k_array, D_ell_TE_integrand, 1.0);
                cell_TE_spline.create(ells, cell_TE, "Cell_TE_of_ell");
                timer.EndTiming("POW::integrating Cells - TE");
            }

            if (Nu_ell_of_k_spline) {
                timer.StartTiming("POW::integrating Cells - NN");
                const double xini = -15.0;
                const double units = std::pow(1e6 * cosmo->get_Tnu(x_cell) / Constants.K, 2);
                std::function<double(double, int)> D_ell_NN_integrand = [&](double k, int ell) {
                    const double normalization = ell * (ell + 1) / (2.0 * M_PI) * units;
                    const double arg = k * eta0;
                    const double Neff_ini =
                        pert->get_Nu(xini, k, 0) + pert->get_Psi(xini, k) + pert->get_Nu(xini, k, 2) / 4.0;
                    const double Nu_ell = Nu_ell_of_k_spline(k, ell) + get_j_ell(ell, arg) * Neff_ini;
                    const double pofk = primordial_power_spectrum_dimless(k);
                    return 4.0 * M_PI * pofk * Nu_ell * Nu_ell * normalization;
                };
                auto cell_NN = solve_for_cell_single(log_k_array, D_ell_NN_integrand, 1.0);
                cell_NN_spline.create(ells, cell_NN, "Cell_NN_of_ell");
                timer.EndTiming("POW::integrating Cells - NN");
            }

            if (lens_ell_of_k_spline) {
                timer.StartTiming("POW::integrating Cells - LL");
                const double units = 1.0;
                std::function<double(double, int)> D_ell_LL_integrand = [&](double k, int ell) {
                    const double normalization = units;
                    const double lens_pot_ell = lens_ell_of_k_spline(k, ell);
                    const double pofk = primordial_power_spectrum_dimless(k);
                    return 4.0 * M_PI * pofk * lens_pot_ell * lens_pot_ell * normalization;
                };
                auto cell_LL = solve_for_cell_single(log_k_array, D_ell_LL_integrand, 1.0);
                for (size_t i = 0; i < ells.size(); i++)
                    cell_LL[i] *= ells[i] * (ells[i] + 1) * ells[i] * (ells[i] + 1);
                cell_LL_spline.create(ells, cell_LL, "Cell_LL_of_ell");
                timer.EndTiming("POW::integrating Cells - LL");
            }

            if (lens_ell_of_k_spline and thetaT_ell_of_k_spline) {
                timer.StartTiming("POW::integrating Cells - TL");
                const double units = 1.0;
                std::function<double(double, int)> D_ell_TL_integrand = [&](double k, int ell) {
                    const double normalization = units;
                    const double lens_pot_ell = lens_ell_of_k_spline(k, ell);
                    const double theta = thetaT_ell_of_k_spline(k, ell);
                    const double pofk = primordial_power_spectrum_dimless(k);
                    return 4.0 * M_PI * pofk * lens_pot_ell * theta * normalization;
                };
                auto cell_TL = solve_for_cell_single(log_k_array, D_ell_TL_integrand, 1.0);
                for (size_t i = 0; i < ells.size(); i++)
                    cell_TL[i] *= 1.0;
                cell_TL_spline.create(ells, cell_TL, "Cell_TL_of_ell");
                timer.EndTiming("POW::integrating Cells - TL");
            }

            timer.EndTiming("POW::integrating Cells");
        }

        void PowerSpectrum::output_theta_ell(std::string filename) const {
            std::ofstream fp(filename.c_str());
            if (not fp.is_open())
                return;

            const int npts = 1000;
            auto k_array = FML::MATH::linspace(std::log(kmin), std::log(kmax), npts);
            for (auto & k : k_array)
                k = std::exp(k);

            fp << "# k (1/Mpc)  Theta_ell_T(l=6,500,1000,1500)   Theta_ell_E(l=...) Theta_ell_lens(l=...) \n";
            auto print_data = [&](const double k) {
                // 1
                fp << k * Constants.Mpc << " ";

                // 2: Theta_ell
                if (thetaT_ell_of_k_spline) {
                    fp << thetaT_ell_of_k_spline(k, 6) << " ";
                    fp << thetaT_ell_of_k_spline(k, 500) << " ";
                    fp << thetaT_ell_of_k_spline(k, 1000) << " ";
                    fp << thetaT_ell_of_k_spline(k, 1500) << " ";
                } else {
                    fp << 0.0 << " " << 0.0 << " " << 0.0 << " " << 0.0 << " ";
                }

                // 6: ThetaE_ell
                if (thetaE_ell_of_k_spline) {
                    fp << thetaE_ell_of_k_spline(k, 5) << " ";
                    fp << thetaE_ell_of_k_spline(k, 50) << " ";
                    fp << thetaE_ell_of_k_spline(k, 500) << " ";
                    fp << thetaE_ell_of_k_spline(k, 1000) << " ";
                } else {
                    fp << 0.0 << " " << 0.0 << " " << 0.0 << " " << 0.0 << " ";
                }

                // 10: lens-ell
                if (lens_ell_of_k_spline) {
                    fp << lens_ell_of_k_spline(k, 5) << " ";
                    fp << lens_ell_of_k_spline(k, 50) << " ";
                    fp << lens_ell_of_k_spline(k, 500) << " ";
                    fp << lens_ell_of_k_spline(k, 1000) << " ";
                } else {
                    fp << 0.0 << " " << 0.0 << " " << 0.0 << " " << 0.0 << " ";
                }
                fp << "\n";
            };
            std::for_each(k_array.begin(), k_array.end(), print_data);
        }

        void PowerSpectrum::output_angular_power_spectra(std::string filename) const {
            std::ofstream fp(filename.c_str());
            if (not fp.is_open())
                return;
            
            // Output in standard units of muK^2
            const int ellmax = int(ells[ells.size() - 1]);
            auto ellvalues = FML::MATH::linspace(2, ellmax, ellmax - 1);
            fp << "# ell  C_ell:  TT  EE  TE  LL  NN  TL\n";
            auto print_data = [&](const double ell) {
                fp << ell << " ";

                if (cell_TT_spline) {
                    fp << cell_TT_spline(ell) << " ";
                }
                if (cell_EE_spline) {
                    fp << cell_EE_spline(ell) << " ";
                }
                if (cell_TE_spline) {
                    fp << cell_TE_spline(ell) << " ";
                }
                if (cell_LL_spline) {
                    fp << cell_LL_spline(ell) << " ";
                }
                if (cell_NN_spline) {
                    fp << cell_NN_spline(ell) << " ";
                }
                if (cell_TL_spline) {
                    fp << cell_TL_spline(ell) / std::sqrt(cell_TT_spline(ell) * cell_LL_spline(ell)) << " ";
                }
                fp << "\n";
            };
            std::for_each(ellvalues.begin(), ellvalues.end(), print_data);
        }

        void PowerSpectrum::output_matter_power_spectrum(double x, std::string filename) const {
            std::ofstream fp(filename.c_str());
            if (not fp.is_open())
                return;
            
            const int npts = 100;

            // Output in units of k = h/Mpc and P(k) = (Mpc/h)^3
            auto pofk_M = get_power_spectrum_array(x, npts, "M");
            auto k_M = pofk_M.first;
            auto p_M = pofk_M.second;
            auto pofk_B = get_power_spectrum_array(x, npts, "B");
            auto p_B = pofk_B.second;
            auto pofk_CDM = get_power_spectrum_array(x, npts, "CDM");
            auto p_CDM = pofk_CDM.second;
            auto pofk_CB = get_power_spectrum_array(x, npts, "CB");
            auto p_CB = pofk_CB.second;
            auto pofk_R = get_power_spectrum_array(x, npts, "R");
            auto p_R = pofk_R.second;
            auto pofk_Nu = get_power_spectrum_array(x, npts, "Nu");
            auto p_Nu = pofk_Nu.second;
            auto pofk_Rtot = get_power_spectrum_array(x, npts, "Rtot");
            auto p_Rtot = pofk_Rtot.second;

            fp << "# k (h/Mpc)   Matter   Baryon   CDM  CB  R  Nu   Rtot   (Units: (Mpc/h)^3)\n";
            const double norm_k = Constants.Mpc / cosmo->get_h();
            const double norm = std::pow(cosmo->get_h() / Constants.Mpc, 3);
            for (size_t i = 0; i < k_M.size(); i++) {
                fp << k_M[i] * norm_k << " ";
                fp << p_M[i] * norm << " ";
                fp << p_B[i] * norm << " ";
                fp << p_CDM[i] * norm << " ";
                fp << p_CB[i] * norm << " ";
                fp << p_R[i] * norm << " ";
                fp << p_Nu[i] * norm << " ";
                fp << p_Rtot[i] * norm << " ";
                fp << "\n";
            }
        }

        double PowerSpectrum::get_cell_TT(double ell) const { return cell_TT_spline(ell); }
        double PowerSpectrum::get_cell_TE(double ell) const { return cell_TE_spline(ell); }
        double PowerSpectrum::get_cell_EE(double ell) const { return cell_EE_spline(ell); }
        double PowerSpectrum::get_cell_LL(double ell) const { return cell_LL_spline(ell); }
        double PowerSpectrum::get_cell_NN(double ell) const { return cell_NN_spline(ell); }
        double PowerSpectrum::get_corr_func_CDM(double x, double r) const { return xi_CDM_spline(x, r); }
        double PowerSpectrum::get_corr_func_B(double x, double r) const { return xi_B_spline(x, r); }
        double PowerSpectrum::get_corr_func_R(double x, double r) const { return xi_R_spline(x, r); }
        double PowerSpectrum::get_corr_func_Nu(double x, double r) const { return xi_Nu_spline(x, r); }
        double PowerSpectrum::get_corr_func_M(double x, double r) const { return xi_M_spline(x, r); }
        double PowerSpectrum::get_corr_func_CB(double x, double r) const {
            const double OmegaB = cosmo->get_OmegaB();
            const double OmegaCDM = cosmo->get_OmegaCDM();
            return (OmegaB * xi_B_spline(x, r) + OmegaCDM * xi_CDM_spline(x, r)) / (OmegaB + OmegaCDM);
        }
        double PowerSpectrum::get_corr_func_Rtot(double x, double r) const {
            const double OmegaR = cosmo->get_OmegaR();
            const double OmegaNu = cosmo->get_OmegaNu();
            return (OmegaR * xi_R_spline(x, r) + OmegaNu * xi_Nu_spline(x, r)) / (OmegaR + OmegaNu);
        }
        double PowerSpectrum::get_corr_func(double x, double r, std::string type) const {
            if (type == "M")
                return get_corr_func_M(x, r);
            if (type == "CDM")
                return get_corr_func_CDM(x, r);
            if (type == "B")
                return get_corr_func_B(x, r);
            if (type == "CB")
                return get_corr_func_CB(x, r);
            if (type == "R")
                return get_corr_func_R(x, r);
            if (type == "Nu")
                return get_corr_func_Nu(x, r);
            if (type == "Rtot")
                return get_corr_func_Rtot(x, r);
            throw std::runtime_error("Unknown type in get_corr_func[" + type + "]");
        }

        double PowerSpectrum::lensing_source(double x, double x_observer) const {
            if (x == 0.0) {
                return 0.0;
            }
            const double x_star = rec->get_xstar();
            // if( x > x_star ) return 0.0;
            const double chi0 = cosmo->chi_of_x(x_observer);
            const double chi = cosmo->chi_of_x(x) - chi0;
            const double chi_star = cosmo->chi_of_x(x_star) - chi0;
            const double Hp = cosmo->Hp_of_x(x);
            return (chi - chi_star) / (chi_star * chi) * (Constants.c / Hp);
        }

        // Computes the correlation function from a power-spectrum DeltaP = k^3 P(k)/2pi^2 by integrating
        // over [kmin, kmax]
        double
        correlation_function_single(double r, std::function<double(double)> & Delta_P, double kmin, double kmax) {

            ODEFunction dxidlogk_func = [&](double logkr, [[maybe_unused]] const double * y, double * dxidlogkr) {
                double kr = std::exp(logkr);
                dxidlogkr[0] = std::sin(kr) / (kr)*Delta_P(kr / r);
                return GSL_SUCCESS;
            };

            DVector xi_ini{0.0};
            DVector range{std::log(kmin * r), std::log(kmax * r)};
            ODESolver xi_ode(1e-5, 1e-9, 1e-9);
            xi_ode.solve(dxidlogk_func, range, xi_ini);

            return xi_ode.get_final_data_by_component(0);
        }

#ifdef USE_FFTW

        std::pair<DVector, DVector> correlation_function_single_fftlog(std::function<double(double)> & Delta_P,
                                                                       double rmin,
                                                                       double rmax,
                                                                       int ngrid) {
            // We want xi(r) for rmin ... rmax. We solve for it in rmin/padding ... rmax * padding
            // to avoid ringing/aliasing close to the edges. A factor of 15 seems to be enough
            // The number of points sets the accuracy. The fiducial choice below is to ensure
            // 1% accuracy around the BAO peak
            const double r0 = std::sqrt(rmin * rmax);
            const double paddingfactor = 15.0;
            const double k0 = 1.0 / r0;

            // Set ranges and make a log-spaced k array
            const double kmin_fft = k0 * r0 / rmax / paddingfactor;
            const double kmax_fft = k0 * r0 / rmin * paddingfactor;
            DVector k_array = FML::MATH::linspace(std::log(kmin_fft), std::log(kmax_fft), ngrid);
            for (auto & k : k_array)
                k = std::exp(k);

            // Fill P(k) array
            DVector Pk_array(ngrid);
            for (int i = 0; i < ngrid; i++) {
                Pk_array[i] = Delta_P(k_array[i]) / std::pow(k_array[i], 3) * (2.0 * M_PI * M_PI);
            }

            // FFTLog algorithm
            auto res = FML::SOLVERS::FFTLog::ComputeCorrelationFunction(k_array, Pk_array);

            return res;
        }

        // Computes the correlation function from a power-spectrum using FFTW
        std::pair<DVector, DVector> correlation_function_single_fftw(std::function<double(double)> & Delta_P,
                                                                     double rmin,
                                                                     double rmax,
                                                                     int ngrid_min) {

            // Set the boxsize and the grid resolution
            double Box = 4.0 * rmax;
            int ngrid = std::max(int(4.0 * (Box / rmin)), ngrid_min);

            // Find closest power of two for optimal FFTW
            for (int N = 2;; N *= 2) {
                if (N >= ngrid or N > 1e9) {
                    ngrid = N;
                    break;
                }
            }

            // Set up grid in k-space and fill it with the source
            fftw_complex * source = (fftw_complex *)fftw_malloc(sizeof(fftw_complex) * ngrid);
            for (int i = 0; i < ngrid; i++) {
                double im = 0.0;
                if (i == 0 or i == ngrid / 2) {
                    // Zero mode and largest mode
                    im = 0.0;
                } else if (i < ngrid / 2) {
                    // Positive modes
                    double k = 2.0 * M_PI / Box * i;
                    im = -Delta_P(k) / (k * k) / 2.0;
                } else {
                    // Negative modes
                    double k = 2.0 * M_PI / Box * (i - ngrid);
                    im = +Delta_P(-k) / (k * k) / 2.0;
                }
                source[i][0] = 0.0;
                source[i][1] = im;
            }

            // Transform to real space to get r*xi(r)
            fftw_plan plan = fftw_plan_dft_1d(ngrid, source, source, FFTW_BACKWARD, FFTW_MEASURE);
            fftw_execute(plan);
            fftw_destroy_plan(plan);

            // Normalize to get xi (i = 0 omitted to avoid division by 0)
            for (int i = 0; i < ngrid; i++) {
                double r = i * Box / double(ngrid);
                source[i][0] *= 2.0 * M_PI / Box;
                if (i > 0)
                    source[i][0] /= r;
            }

            // Make a spline (we only keep the
            DVector r_arr;
            DVector xi_arr;
            r_arr.reserve(ngrid);
            xi_arr.reserve(ngrid);
            for (int i = 0; i < ngrid; i++) {
                double r = i * Box / double(ngrid);
                if (r >= rmin and r <= rmax) {
                    r_arr.push_back(r);
                    xi_arr.push_back(source[i][0]);
                }
            }

            // Clean up FFTW
            fftw_free(source);

            return {r_arr, xi_arr};
        }
#endif

    } // namespace COSMOLOGY
} // namespace FML
