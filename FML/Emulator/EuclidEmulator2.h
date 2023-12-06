#ifndef EUCLIDEMULATOR2_HEADER
#define EUCLIDEMULATOR2_HEADER
#include <array>
#include <assert.h>
#include <fcntl.h>
#include <fstream>
#include <gsl/gsl_errno.h>
#include <gsl/gsl_integration.h>
#include <gsl/gsl_sf_legendre.h>
#include <gsl/gsl_spline.h>
#include <gsl/gsl_spline2d.h>
#include <iomanip>
#include <iostream>
#include <limits.h>
#include <math.h>
#include <regex>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <string>
#include <sys/mman.h>
#include <sys/stat.h>
#include <typeinfo>
#include <unistd.h>
#include <vector>

namespace FML {
    namespace EMULATOR {

        //=====================================================================
        /// This namespace contains the minimal code needed to run the EuclidEmulator2
        /// so that we easily can use it within the FML library
        ///
        /// The original code was written by Mischa Knabenhans and was
        /// taken from https://github.com/miknab/EuclidEmulator2
        /// For use of the emulator cit the EuclidEmulator2 paper: https://arxiv.org/abs/2010.11288
        ///
        /// The licence file from the original code-files is given below:
        ///  Copyright (c) 2020 Mischa Knabenhans
        ///
        ///  EuclidEmulator2 is free software: you can redistribute it and/or modify
        ///  it under the terms of the GNU General Public License as published by
        ///  the Free Software Foundation, either version 3 of the License, or
        ///  (at your option) any later version.
        ///
        ///  EuclidEmulator2 is distributed in the hope that it will be useful,
        ///  but WITHOUT ANY WARRANTY; without even the implied warranty of
        ///  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
        ///  GNU General Public License for more details.
        ///
        ///  You should have received a copy of the GNU General Public License
        ///  along with this program.  If not, see <http://www.gnu.org/licenses/>.
        //=====================================================================
        namespace EUCLIDEMULATOR2 {

#ifdef USE_GSL

            // We assume the EE2 datafile is in the same folder as this header with a .dat ending instead of .h
            // If this does not work with your compiler simply define path_to_ee2_datafile below
#ifdef __FILE__  
            std::string _temp_this_filepath = __FILE__;
            std::string _temp_this_filepath_no_ending = _temp_this_filepath.substr(0, _temp_this_filepath.size()-1);
            std::string path_to_ee2_datafile = _temp_this_filepath_no_ending + "dat";
#else 
            std::string path_to_ee2_datafile = "EuclidEmulator2.dat";
#endif

            // Base units
            const double second = 1.0;
            const double meter = 1.0;
            const double kilogram = 1.0;
            const double Kelvin = 1.0;
            // derived units
            const double kilometer = 1000 * meter;
            const double Joule = kilogram * std::pow(meter, 2) / std::pow(second, 2);
            const double Mpc = 3.085677581282e22 * meter;
            const double Msol = 1.98841e30 * kilogram;
            // units constant
            const double c = 2.99792458e8 * meter / second;
            const double G = 6.67428e-11 * std::pow(meter, 3) / (kilogram * std::pow(second, 2));
            const double kB = 1.3806504e-23 * Joule / Kelvin;
            const double eV = 1.602176487e-19 * Joule;
            const double hbar = 6.62606896e-34 / (2 * M_PI) * Joule * second;
            const double rho_crit_over_h2 = 2.77536627e11 * Msol / std::pow(Mpc, 3);
            // Constants and specifications of the training simulations
            const double Lbox = 1000 * Mpc;
            const double Tgamma = 2.7255 * Kelvin;

            bool does_ee2_datafile_exist() {
                std::ifstream f(path_to_ee2_datafile.c_str());
                if (not f.good()) {
                    return false;
                }
                return true;
            }

            void set_path_to_ee2_data(std::string filename) { path_to_ee2_datafile = filename; }

            class Cosmology {
              public:
                double cosmo[8], cosmo_tf[8];
                double Omega_gamma_0, Omega_nu_0, Omega_DE_0, rho_crit, T_gamma_0, T_nu_0;
                double minima[8] = {0.04, 0.24, 0.00, 0.92, 0.61, -1.3, -0.7, 1.7e-9};
                double maxima[8] = {0.06, 0.40, 0.15, 1.00, 0.73, -0.7, 0.7, 2.5e-9};
                std::string names[8] = {"OmegaB", "OmegaM", "SumMnu", "n_s", "h", "w_0", "w_a", "A_s"};
                Cosmology() = default;
                Cosmology(double Omega_b,
                          double Omega_m,
                          double Sum_m_nu,
                          double n_s,
                          double h,
                          double w_0,
                          double w_a,
                          double A_s);
                double compute_step_number(double z);
                static double rho_nu_i_integrand(double p, void * params);
                static double a2t_integrand(double a, void * params);
                bool good_parameter_ranges();
                bool is_good_to_use();
                void print_errors();

              private:
                double EPSCOSMO = 1e-6;
                int nSteps = 101, nTable = 101;
                double t0, t10, Delta_t, Neff = 3.046, H0;
                typedef struct {
                    double mnu_i;
                    double a;
                    Cosmology * csm_instance;
                } rho_nu_parameters;
                typedef struct {
                    Cosmology * csm_instance;
                } a2t_parameters;
                gsl_integration_workspace * gsl_wsp;
                gsl_interp_accel * acc;
                gsl_spline * z2nStep_spline;
                void isoprob_tf();
                void compute_z2nStep_spline();
                double Hubble(double a);
                double Omega_matter(double a);
                double Omega_gamma(double a);
                double Omega_nu(double a);
                double Omega_DE(double a);
                double T_gamma(double a);
                double T_nu(double a);
                double a2t(double a);
                double a2Hubble(double a);
            };

            Cosmology::Cosmology(double Omega_b,
                                 double Omega_m,
                                 double Sum_m_nu,
                                 double n_s,
                                 double h,
                                 double w_0,
                                 double w_a,
                                 double A_s) {
                gsl_wsp = gsl_integration_workspace_alloc(1000);
                acc = gsl_interp_accel_alloc();
                cosmo[0] = Omega_b;
                cosmo[1] = Omega_m;
                cosmo[2] = Sum_m_nu;
                cosmo[3] = n_s;
                cosmo[4] = h;
                cosmo[5] = w_0;
                cosmo[6] = w_a;
                cosmo[7] = A_s;
                T_gamma_0 = T_gamma(1.0);
                T_nu_0 = T_nu(1.0);
                rho_crit = rho_crit_over_h2 * std::pow(cosmo[4], 2);
                Omega_nu_0 = Omega_nu(1.0);
                Omega_gamma_0 = Omega_gamma(1.0);
                Omega_DE_0 = 1 - (cosmo[1] + Omega_gamma_0 + Omega_nu_0);
                t0 = a2t(1.0);
                t10 = a2t(1.0 / (10 + 1));
                Delta_t = (t0 - t10) / (nSteps - 1);
                z2nStep_spline = gsl_spline_alloc(gsl_interp_cspline, nTable);
                compute_z2nStep_spline();
                isoprob_tf();
            }

            bool Cosmology::is_good_to_use() { return good_parameter_ranges() and does_ee2_datafile_exist(); }

            bool Cosmology::good_parameter_ranges() {
                bool good = true;
                for (int i = 0; i < 8; i++) {
                    if (cosmo[i] < minima[i] || cosmo[i] > maxima[i]) {
                        good = false;
                    }
                    if (i == 5)
                        i++;
                }
                if (cosmo[6] < -0.7 || cosmo[6] > 0.5) {
                    good = false;
                }
                return good;
            }

            void Cosmology::print_errors() {
                for (int i = 0; i < 8; i++) {
                    if (cosmo[i] < minima[i] || cosmo[i] > maxima[i]) {
                        std::cout << "EuclidEmulator2::Error Cosmology parameter " << names[i] << " has range ["
                                  << minima[i] << " , " << maxima[i] << "]";
                        std::cout << ", but we got " << names[i] << " = " << cosmo[i] << std::endl;
                    }
                    if (i == 5)
                        i++;
                }
                if (cosmo[6] < -0.7 || cosmo[6] > 0.5) {
                    std::cout << "EuclidEmulator2::Error Cosmology parameter " << names[6] << " has range ["
                              << minima[6] << " , " << maxima[6] << "]";
                    std::cout << ", but we got " << names[6] << " = " << cosmo[6] << std::endl;
                }
                if (not does_ee2_datafile_exist()) {
                    std::cout << "EuclidEmulator2::Error cannot find EE2 datafile. Provided path: "
                              << path_to_ee2_datafile << std::endl;
                }
            }

            void Cosmology::isoprob_tf() {
                for (int i = 0; i < 8; i++) {
                    cosmo_tf[i] = 2 * (cosmo[i] - minima[i]) / (maxima[i] - minima[i]) - 1.0;
                }
            }

            double Cosmology::Omega_matter(double a) { return cosmo[1] / (a * a * a); }

            double Cosmology::Omega_gamma(double a) {
                double rho_gamma_0 =
                    M_PI * M_PI / 15.0 * std::pow(kB, 4) / (std::pow(hbar, 3) * std::pow(c, 5)) * std::pow(Tgamma, 4);
                Omega_gamma_0 = rho_gamma_0 / rho_crit;
                return Omega_gamma_0 / (a * a * a * a);
            }

            double Cosmology::T_gamma(double a) { return Tgamma / a; }

            double Cosmology::T_nu(double a) {
                return std::pow(Neff / 3.0, 0.25) * std::pow(4.0 / 11.0, 1. / 3.) * Tgamma / a;
            }

            double Cosmology::rho_nu_i_integrand(double p, void * params) {
                rho_nu_parameters * rho_nu_pars = reinterpret_cast<rho_nu_parameters *>(params);
                double T_nu = rho_nu_pars->csm_instance->T_nu_0;
                double mnui = rho_nu_pars->mnu_i;
                double p2 = p * p;
                double y = p2 / (std::exp(c / kB / T_nu * rho_nu_pars->a * p) + 1);
                return y * std::sqrt(mnui * mnui * c * c * c * c + p2 * c * c);
            }

            double Cosmology::Omega_nu(double a) {
                rho_nu_parameters rho_nu_pars;
                gsl_function F;
                double rho_nu_i, error;
                double pmax = 0.004 / a * eV / c;
                double prefactor = 1.0 / (std::pow(M_PI, 2) * std::pow(hbar, 3) * std::pow(c, 2));
                rho_nu_pars.mnu_i = cosmo[2] / 3.0 * eV / std::pow(c, 2);
                rho_nu_pars.a = a;
                rho_nu_pars.csm_instance = this;
                F.function = &rho_nu_i_integrand;
                F.params = &rho_nu_pars;
                gsl_integration_qag(&F, 0.0, pmax, 0.0, EPSCOSMO, 1000, GSL_INTEG_GAUSS61, gsl_wsp, &rho_nu_i, &error);
                rho_nu_i *= prefactor;
                return 3 * rho_nu_i / rho_crit;
            }

            double Cosmology::Omega_DE(double a) {
                double w_0 = cosmo[5];
                double w_a = cosmo[6];
                return Omega_DE_0 * std::pow(a, -3.0 * (1 + w_0 + w_a)) * std::exp(-3 * (1 - a) * w_a);
            }

            double Cosmology::a2Hubble(double a) {
                H0 = 100 * cosmo[4] * kilometer / second / Mpc;
                return H0 * std::sqrt(Cosmology::Omega_matter(a) + Cosmology::Omega_gamma(a) + Cosmology::Omega_nu(a) +
                                      Cosmology::Omega_DE(a));
            }

            double Cosmology::a2t_integrand(double lna, void * params) {
                a2t_parameters * a2t_pars = reinterpret_cast<a2t_parameters *>(params);
                return 1. / (a2t_pars->csm_instance->a2Hubble(std::exp(lna)));
            }

            double Cosmology::a2t(double a) {
                a2t_parameters a2t_params;
                gsl_function F;
                double result, error;
                a2t_params.csm_instance = this;
                F.function = &a2t_integrand;
                F.params = &a2t_params;
                gsl_integration_qag(
                    &F, -15, std::log(a), 0.0, EPSCOSMO, 1000, GSL_INTEG_GAUSS61, gsl_wsp, &result, &error);
                return result;
            }

            void Cosmology::compute_z2nStep_spline() {
                double t_current, z;
                double avec[101] = {0};
                double frac_nStep[101] = {0};
                double z10 = 10.0;
                for (int idx = 0; idx < nTable; idx++) {
                    z = z10 - idx * 0.1;
                    avec[idx] = 1.0 / (z + 1.0);
                    t_current = Cosmology::a2t(avec[idx]);
                    frac_nStep[idx] = (t_current - t10) / Delta_t;
                }
                assert(std::abs(frac_nStep[0]) < EPSCOSMO);
                assert(std::abs(frac_nStep[nTable - 1] - (nSteps - 1)) < EPSCOSMO);
                gsl_spline_init(z2nStep_spline, avec, frac_nStep, nTable);
            }

            double Cosmology::compute_step_number(double z) {
                if (std::abs(z) < EPSCOSMO) {
                    return 100.0;
                } else {
                    return gsl_spline_eval(z2nStep_spline, 1. / (z + 1.), acc);
                }
            }

            class EuclidEmulator {
              private:
                const int nz = 101;
                const int nk = 613;
                const int n_coeffs[14] = {53, 53, 117, 117, 53, 117, 117, 117, 117, 521, 117, 1539, 173, 457};
                const int lmax = 16;
                gsl_interp_accel * logk2pc_acc[15];
                gsl_interp_accel * logz2pc_acc[15];
                gsl_spline2d * logklogz2pc_spline[15];
                double * pc[15];
                double * pce_coeffs[14];
                double * pce_multiindex[14];
                std::array<std::vector<double>, 8> univ_legendre;
                void read_in_ee2_data_file();
                void pc_2d_interp();
                Cosmology csm{};

              public:
                double kvec[613];
                double Bvec[101][613];
                EuclidEmulator(Cosmology csm);
                EuclidEmulator(double OmegaB,
                               double OmegaM,
                               double Sum_m_nu,
                               double n_s,
                               double h,
                               double w_0,
                               double w_a,
                               double A_s);
                ~EuclidEmulator();
                void compute_boost(std::vector<double> redshift, int n_redshift);
                std::pair<std::vector<double>, std::vector<double>> compute_boost(double redshift);
                std::pair<std::vector<double>, std::vector<double>> get_boost(int iz);
            };

            EuclidEmulator::EuclidEmulator(Cosmology cosmo) : csm(cosmo) {
                read_in_ee2_data_file();
                pc_2d_interp();
                if (not csm.good_parameter_ranges()) {
                    throw std::runtime_error("Cosmological parameter(s) is out of bounds for EuclidEmulator2");
                }
            }

            EuclidEmulator::EuclidEmulator(double OmegaB,
                                           double OmegaM,
                                           double Sum_m_nu,
                                           double n_s,
                                           double h,
                                           double w_0,
                                           double w_a,
                                           double A_s)
                : EuclidEmulator(Cosmology(OmegaB, OmegaM, Sum_m_nu, n_s, h, w_0, w_a, A_s)) {}

            EuclidEmulator::~EuclidEmulator() {
                for (int i = 0; i < 15; i++) {
                    gsl_spline2d_free(logklogz2pc_spline[i]);
                }
            }

            void EuclidEmulator::read_in_ee2_data_file() {
                off_t size;
                struct stat s;
                double * data;
                double * kptr;
                int idx = 0;

                for (int iz = 0; iz < nz; iz++) {
                    for (int ik = 0; ik < nk; ik++) {
                        Bvec[iz][ik] = 0.0;
                    }
                }

                int fp = open(path_to_ee2_datafile.c_str(), O_RDONLY);
                if (!fp) {
                    std::cerr << "Unable to open " << path_to_ee2_datafile << "\n";
                    throw std::runtime_error("Error: the EuclidEmulator2 file (EuclidEmulator2.dat) can not be found. "
                                             "Use set_path_to_ee2_data to set this path\n");
                }
                [[maybe_unused]] auto status = fstat(fp, &s);
                size = s.st_size;
                data = (double *)mmap(0, size, PROT_READ, MAP_PRIVATE, fp, 0);
                for (int i = 0; i < 15; i++) {
                    pc[i] = &data[idx];
                    idx += nk * nz;
                }
                for (int i = 0; i < 14; i++) {
                    pce_coeffs[i] = &data[idx];
                    idx += n_coeffs[i];
                }
                for (int i = 0; i < 14; i++) {
                    pce_multiindex[i] = &data[idx];
                    idx += 8 * n_coeffs[i];
                }
                kptr = &data[idx];
                for (int i = 0; i < nk; i++) {
                    kvec[i] = kptr[i];
                }
                idx += nk;
                assert(idx == int(size / sizeof(double)));
            }

            void EuclidEmulator::pc_2d_interp() {
                double logk[nk];
                double stp[nz];
                for (int i = 0; i < nk; i++)
                    logk[i] = std::log(kvec[i]);
                for (int i = nz - 1; i >= 0; i--)
                    stp[i] = i;
                for (int i = 0; i < 15; i++) {
                    logk2pc_acc[i] = gsl_interp_accel_alloc();
                    logz2pc_acc[i] = gsl_interp_accel_alloc();
                    logklogz2pc_spline[i] = gsl_spline2d_alloc(gsl_interp2d_bicubic, nk, nz);
                    gsl_spline2d_init(logklogz2pc_spline[i], logk, stp, pc[i], nk, nz);
                }
            }

            std::pair<std::vector<double>, std::vector<double>> EuclidEmulator::compute_boost(double redshift) {
                std::vector<double> redshifts(1, redshift);
                compute_boost(redshifts, 1);
                return get_boost(0);
            }
            void EuclidEmulator::compute_boost(std::vector<double> redshift, int n_redshift) {
                double pc_weight;
                double basisfunc;
                double stp_no[n_redshift];
                for (int iz = 0; iz < n_redshift; iz++) {
                    if (redshift.at(iz) > 10.0 || redshift.at(iz) < 0.0) {
                        std::cout << "ERROR: EuclidEmulator2 accepts only redshifts in the interval [0.0, 10.0]\n"
                                  << "The current redshift z = " << redshift.at(iz) << " is therefore ignored."
                                  << std::endl;
                        continue;
                    }
                    stp_no[iz] = csm.compute_step_number(redshift.at(iz));
                }

                for (int ipar = 0; ipar < 8; ipar++) {
                    univ_legendre[ipar] = std::vector<double>(lmax + 1, 0.0);
                    gsl_sf_legendre_Pl_array(lmax, csm.cosmo_tf[ipar], univ_legendre[ipar].data());
                    for (int l = 0; l <= lmax; l++) {
                        univ_legendre[ipar][l] *= std::sqrt(2.0 * l + 1.0);
                    }
                }
                for (int iz = 0; iz < n_redshift; iz++) {
                    for (int ik = 0; ik < nk; ik++) {
                        Bvec[iz][ik] = gsl_spline2d_eval(
                            logklogz2pc_spline[0], std::log(kvec[ik]), stp_no[iz], logk2pc_acc[0], logz2pc_acc[0]);
                    }
                }
                for (int ipc = 1; ipc < 15; ipc++) {
                    pc_weight = 0.0;
                    for (int ic = 0; ic < n_coeffs[ipc - 1]; ic++) {
                        basisfunc = 1.0;
                        for (int ipar = 0; ipar < 8; ipar++) {
                            basisfunc *= univ_legendre[ipar][int(pce_multiindex[ipc - 1][ic * 8 + ipar])];
                        }
                        pc_weight += pce_coeffs[ipc - 1][ic] * basisfunc;
                    }
                    for (int iz = 0; iz < n_redshift; iz++) {
                        for (int ik = 0; ik < nk; ik++) {
                            Bvec[iz][ik] += (pc_weight * gsl_spline2d_eval(logklogz2pc_spline[ipc],
                                                                           std::log(kvec[ik]),
                                                                           stp_no[iz],
                                                                           logk2pc_acc[ipc],
                                                                           logz2pc_acc[ipc]));
                        }
                    }
                }
            }

            std::pair<std::vector<double>, std::vector<double>> EuclidEmulator::get_boost(int iz) {
                std::vector<double> boost(nk), k(nk);
                for (int ik = 0; ik < nk; ik++) {
                    boost[ik] = std::pow(10.0, Bvec[iz][ik]);
                    k[ik] = kvec[ik];
                }
                return {k, boost};
            }
#endif
        } // namespace EUCLIDEMULATOR2
    }     // namespace EMULATOR
} // namespace FML
#endif
