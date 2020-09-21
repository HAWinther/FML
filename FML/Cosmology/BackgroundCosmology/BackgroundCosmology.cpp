#include "BackgroundCosmology.h"

namespace FML {
    namespace COSMOLOGY {

        FML::UTILS::ConstantsAndUnits Constants;

        //====================================================
        // Constructors
        //====================================================

        BackgroundCosmology::BackgroundCosmology(const ParameterMap & p) {
            TCMB = p.get<double>("TCMB");
            Neff = p.get<double>("Neff");
            bool PhysicalParameters = p.get<bool>("PhysicalParameters");

            const double rhoc0_over_h2 = 3.0 * Constants.H0_over_h * Constants.H0_over_h / (8.0 * M_PI * Constants.G);
            const double Tnu_over_TCMB = std::pow(4.0 / 11.0, 1.0 / 3.0);

            double OmegaRh2 = 2.0 * (M_PI * M_PI / 30.0) * std::pow(Constants.k_b * TCMB / Constants.hbar, 4) *
                              Constants.hbar / std::pow(Constants.c, 5) / rhoc0_over_h2;
            double OmegaNuh2 = (7.0 / 8.0) * Neff * std::pow(Tnu_over_TCMB, 4) * OmegaRh2;
            double OmegaBh2, OmegaCDMh2, OmegaKh2, OmegaLambdah2, h2;

            if (PhysicalParameters) {
                OmegaBh2 = p.get<double>("OmegaBh2");
                OmegaCDMh2 = p.get<double>("OmegaCDMh2");
                OmegaKh2 = p.get<double>("OmegaKh2");
                OmegaLambdah2 = p.get<double>("OmegaLambdah2");
                h2 = OmegaBh2 + OmegaCDMh2 + OmegaKh2 + OmegaRh2 + OmegaNuh2 + OmegaLambdah2;
                assert(h2 > 0.0);
            } else {
                h2 = std::pow(p.get<double>("h"), 2);
                OmegaBh2 = p.get<double>("OmegaB") * h2;
                OmegaCDMh2 = p.get<double>("OmegaCDM") * h2;
                OmegaKh2 = p.get<double>("OmegaK") * h2;
                OmegaLambdah2 = (h2 - OmegaBh2 - OmegaCDMh2 - OmegaKh2 - OmegaRh2 - OmegaNuh2);
            }

            // If OmegaKh2 is too negative then H = 0 at some point we need OmegaK > -(OmegaLambda OmegaM^2)^1/3
            const double min_OmegaKh2 =
                -std::pow(OmegaLambdah2 * 27 / 4. * (OmegaCDMh2 + OmegaBh2) * (OmegaCDMh2 + OmegaBh2), 0.33333);
            if (OmegaKh2 < min_OmegaKh2) {
                throw std::runtime_error("OmegaKh2 = " + std::to_string(OmegaKh2) + " is too negative (must be > " +
                                         std::to_string(min_OmegaKh2) +
                                         " ). The Hubble factor will be zero at some point\n");
            }

            // Tiny unobservable values of the curvature is simply set to zero
            if (std::fabs(OmegaKh2 / h2) < 1e-6) {
                OmegaLambdah2 += OmegaKh2;
                OmegaKh2 = 0.0;
            }

            // Tiny unobservable values of dark energy is simply set to zero
            if (std::fabs(OmegaLambdah2 / h2) < 1e-6) {
                OmegaCDMh2 += OmegaLambdah2;
                OmegaLambdah2 = 0.0;
            }

            OmegaB = OmegaBh2 / (h2);
            OmegaCDM = OmegaCDMh2 / (h2);
            OmegaK = OmegaKh2 / (h2);
            OmegaLambda = OmegaLambdah2 / (h2);
            OmegaR = OmegaRh2 / (h2);
            OmegaNu = OmegaNuh2 / (h2);
            OmegaRtot = OmegaR + OmegaNu;
            OmegaM = OmegaB + OmegaCDM;
            h = std::sqrt(h2);
            H0 = h * Constants.H0_over_h;
            K = -OmegaK * H0 * H0 / (Constants.c * Constants.c);
        }

        //====================================================
        // Class methods
        //====================================================

        void BackgroundCosmology::info() const {
            double aeq = OmegaRtot / OmegaM;
            double K = -OmegaK * H0 * H0;

            std::cout << "\n";
            std::cout << "============================================\n";
            std::cout << "Info about cosmology class [" << name << "]:\n";
            std::cout << "============================================\n";
            std::cout << "OmegaB:      " << OmegaB << "\n";
            std::cout << "OmegaCDM:    " << OmegaCDM << "\n";
            std::cout << "OmegaLambda: " << OmegaLambda << "\n";
            std::cout << "OmegaK:      " << OmegaK << "\n";
            if (std::fabs(OmegaK) < 1e-6)
                std::cout << "A flat Universe\n";
            else if (1.0 - OmegaK < 1.0)
                std::cout << "An open Universe R = " << Constants.c / std::sqrt(-K) / Constants.Gpc << " Gpc\n";
            else
                std::cout << "A closed Universe R = " << Constants.c / std::sqrt(K) / Constants.Gpc << " Gpc\n";
            std::cout << "OmegaNu:     " << OmegaNu << "\n";
            std::cout << "OmegaM:      " << OmegaM << "\n";
            std::cout << "OmegaR:      " << OmegaR << "\n";
            std::cout << "OmegaRtot:   " << OmegaRtot << "\n";
            std::cout << "OmegaBh2:    " << OmegaB * h * h << "\n";
            std::cout << "OmegaCDMh2:  " << OmegaCDM * h * h << "\n";
            std::cout << "OmegaLh2:    " << OmegaLambda * h * h << "\n";
            std::cout << "Neff:        " << Neff << "\n";
            std::cout << "h:           " << h << "\n";
            std::cout << "TCMB:        " << TCMB / Constants.K << " K\n";
            std::cout << "Mat-rad equality redshift " << 1.0 / aeq - 1.0 << "\n";
            std::cout << "Equality scale: " << aeq * H_of_x(std::log(aeq)) / Constants.c * Constants.Mpc << " 1/Mpc\n";
            if (eta_of_x_spline) {
                std::cout << "The age of the universe    " << get_cosmic_time(0.0) / Constants.Gyr << " Gyr\n";
                std::cout << "Conformal time today       " << eta_of_x(0.0) / (Constants.Gyr * Constants.c) << " Gyr\n";
                std::cout << "Conformal distance today   " << eta_of_x(0.0) / Constants.Gpc << " Gpc\n";
            }
            std::cout << "============================================\n";
            std::cout << "\n";
        }

        void BackgroundCosmology::solve() {

            // Solve the background and make splines for Hubble functions
            compute_background();

            // Compute conformal time + cosmic time + ...
            compute_conformal_time();

            // Compute growth factors
            compute_growth_factors();
        }

        void BackgroundCosmology::compute_background() {

            // Make a array of log(a) from the very early Univers till a bit into the
            // future
            DVector x_array = FML::MATH::linspace(x_min_background, x_max_background, n_pts_splines);

            // Define the Hubble functions we are to spline
            auto H_function = [&](double x) {
                return H0 * std::sqrt(OmegaLambda + OmegaK * std::exp(-2 * x) + OmegaM * std::exp(-3 * x) + OmegaRtot * std::exp(-4 * x));
            };
            auto Hp_function = [&](double x) { return std::exp(x) * H_function(x); };
            auto dHdx_function = [&](double x) {
                return 1.0 / (2.0 * H_function(x)) * H0 * H0 *
                       (-2 * OmegaK * std::exp(-2 * x) - 3 * OmegaM * std::exp(-3 * x) - 4 * OmegaRtot * std::exp(-4 * x));
            };
            auto dHpdx_function = [&](double x) {
                return 1.0 / (2.0 * Hp_function(x)) * H0 * H0 *
                       (2 * OmegaLambda * std::exp(2 * x) - OmegaM * std::exp(-x) - 2 * OmegaRtot * std::exp(-2 * x));
            };
            auto w_function = [&](double x) {
                return (OmegaRtot * std::exp(-4 * x) / 3.0 - OmegaLambda - OmegaK * std::exp(-2 * x) / 3.0) /
                       std::pow(H_function(x) / H0, 2);
            };

            // Spline the Hubble function and derivatives
            DVector H_array(x_array);
            DVector Hp_array(x_array);
            DVector dHdx_array(x_array);
            DVector dHpdx_array(x_array);
            DVector w_array(x_array);
            for (auto & x : H_array) {
                double xs = x;
                x = H_function(x);
                if (x != x) {
                    throw std::runtime_error("Compute background. H(x) crossing 0.0 at x = " + std::to_string(xs) +
                                             "\n");
                }
            }
            for (auto & x : Hp_array) {
                x = Hp_function(x);
            }
            for (auto & x : dHdx_array) {
                x = dHdx_function(x);
            }
            for (auto & x : dHpdx_array) {
                x = dHpdx_function(x);
            }
            for (auto & x : w_array) {
                x = w_function(x);
            }
            H_spline.create(x_array, H_array, "H_of_x");
            Hp_spline.create(x_array, Hp_array, "Hp_of_x");
            dHdx_spline.create(x_array, dHdx_array, "dHdx_of_x");
            dHpdx_spline.create(x_array, dHpdx_array, "dHpdx_of_x");
            w_spline.create(x_array, w_array, "w_of_x");
        }

        void BackgroundCosmology::compute_conformal_time() {

            // Make a array of log(a) from the very early Univers till a bit into the
            // future
            DVector x_array = FML::MATH::linspace(x_min_background, x_max_background, n_pts_splines);

            // The ODE system deta/dx = c/Hp and dt/dx = c/H
            ODEFunction deriv = [&](double x, [[maybe_unused]] const double * y, double * dydx) {
                double HpoverH0 = Hp_of_x(x) / H0;

                // Conformal time deta/dx in units of c/H0
                dydx[0] = 1.0 / HpoverH0;

                // dt/dx (Age of the Universe) in units of 1/H0
                dydx[1] = std::exp(x) / HpoverH0;
                return GSL_SUCCESS;
            };

            // The initial conditions
            DVector eta_initial{0.0, 0.0};
            if (OmegaRtot > 0.0)
                eta_initial = {std::exp(x_array[0]) / std::sqrt(OmegaRtot), std::exp(2.0 * x_array[0]) / 2.0 / std::sqrt(OmegaRtot)};

            // Solve the ODE
            ODESolver eta_ode(
                FIDUCIAL_COSMO_HSTART_ODE_ETA, FIDUCIAL_COSMO_HSTART_ODE_ETA, FIDUCIAL_COSMO_RELERR_ODE_ETA);
            eta_ode.solve(deriv, x_array, eta_initial);

            // Fetch the result and get eta and t. Divide by H0 as we solved in units of
            // 1/H0
            auto eta = eta_ode.get_data_by_component(0);
            for (auto & y : eta) {
                y *= Constants.c / H0;
            }
            auto age = eta_ode.get_data_by_component(1);
            for (auto & y : age) {
                y *= 1.0 / H0;
            }

            // Spline up eta and the age of the Universe
            eta_of_x_spline.create(x_array, eta, "eta_of_x_spline");
            cosmic_time_of_x_spline.create(x_array, age, "cosmic_time_of_x_spline");

            // Remember to set eta0
            eta0 = eta_of_x_spline(0.0);

            // Comoving distance (only to the present time and not too far back)
            DVector chi_array(eta);
            DVector x_array_chi = FML::MATH::linspace(std::max(x_min_background, -15.0), 0.0, n_pts_splines);
            for (int i = 0; i < n_pts_splines; i++) {
                chi_array[i] = (eta0 - eta_of_x_spline(x_array_chi[i]));
            }
            chi_of_x_spline.create(x_array_chi, chi_array);
            x_of_chi_spline.create(chi_array, x_array_chi);
        }

        void BackgroundCosmology::compute_growth_factors() {
            // This is the growth factor for DeltaM, the total comoving matter perturbationvwhich includes radiation
            // If the flag below is true its the usual one
            const bool cdm_growth_fac = true;

            DVector x_array;
            if (cdm_growth_fac)
                x_array = FML::MATH::linspace(-std::log(1000.0), x_max_background, n_pts_splines);
            else
                x_array = FML::MATH::linspace(x_min_background, x_max_background, n_pts_splines);

            // Growth factors (1LPT and 2LPT)
            ODEFunction deriv_growth = [&](double x, const double * y, double * dydx) {
                const double D1 = y[0];
                const double V1 = y[1];
                const double D2 = y[2];
                const double V2 = y[3];
                const double H = H_of_x(x) / H0;
                const double a = std::exp(x), a2 = a * a;
                const double Omega = get_OmegaM(x) + (cdm_growth_fac ? 0.0 : get_OmegaRtot(x));
                dydx[0] = V1 / (H * a2);
                dydx[1] = 1.5 * Omega * (H * a2) * D1;
                dydx[2] = V2 / (H * a2);
                dydx[3] = 1.5 * Omega * (H * a2) * (D2 - D1 * D1);
                return GSL_SUCCESS;
            };

            // Initial conditions based on analytical solutions
            // D1 = a^n, D2 = prefac * D1^2
            const double xeq = std::log(OmegaRtot / OmegaM);
            const double xini = x_array[0];
            const double nindex = xini < xeq ? std::sqrt(1.5) : 1.0;
            const double prefac = xini < xeq ? -1.0 / 3.0 : -3.0 / 7.0;
            const double D1ini = 1.0;
            const double dD1dxini = nindex * D1ini;
            const double D2ini = prefac * D1ini * D1ini;
            const double dD2dxini = 2.0 * prefac * D1ini * dD1dxini;
            const double V1ini = dD1dxini * (H_of_x(xini) / H0 * std::exp(2.0 * xini));
            const double V2ini = dD2dxini * (H_of_x(xini) / H0 * std::exp(2.0 * xini));

            // Solve the ODE
            DVector D_ini{D1ini, V1ini, D2ini, V2ini};
            ODESolver growth_ode(1e-3, 1e-12, 1e-12);
            growth_ode.solve(deriv_growth, x_array, D_ini);

            // Fetch solution
            auto D1 = growth_ode.get_data_by_component(0);
            auto V1 = growth_ode.get_data_by_component(1);
            auto D2 = growth_ode.get_data_by_component(2);
            auto V2 = growth_ode.get_data_by_component(3);

            // Normalize to unity at present time and take the log
            DVector dD1dx(x_array.size());
            DVector dD2dx(x_array.size());
            for (size_t i = 0; i < x_array.size(); i++) {
                D1[i] /= D1[x_array.size() - 1];
                V1[i] /= D1[x_array.size() - 1];
                D2[i] /= D2[x_array.size() - 1];
                V2[i] /= D2[x_array.size() - 1];
                dD1dx[i] = V1[i] / (H_of_x(x_array[i]) / H_of_x(0.0) * std::exp(2.0 * x_array[i]));
                dD2dx[i] = V2[i] / (H_of_x(x_array[i]) / H_of_x(0.0) * std::exp(2.0 * x_array[i]));
            }

            // Spline up
            D1_spline.create(x_array, D1, "D1_spline");
            D2_spline.create(x_array, D2, "D2_spline");
        }

        void BackgroundCosmology::output(const std::string filename) const {
            std::ofstream fp(filename.c_str());

            const int npts = 100;
            DVector x_array = FML::MATH::linspace(x_min_background, x_max_background, npts);

            fp << "# x = log(a) Cosmology Quantities [ aH  (aH)'  (aH)'' eta/Mpc  Omegai's  LPT growthfactors ]\n";
            auto print_data = [&](const double x) {
                // 1
                fp << x << " ";

                // 2
                fp << Hp_of_x(x) << " ";
                fp << dHpdx_of_x(x) << " ";
                fp << ddHpddx_of_x(x) << " ";

                // 5
                fp << eta_of_x(x) / Constants.Mpc << " ";

                // 6
                fp << get_OmegaB(x) << " ";
                fp << get_OmegaCDM(x) << " ";
                fp << get_OmegaM(x) << " ";
                fp << get_OmegaLambda(x) << " ";
                fp << get_OmegaR(x) << " ";
                fp << get_OmegaNu(x) << " ";
                fp << get_OmegaRtot(x) << " ";
                fp << get_OmegaK(x) << " ";

                // 14
                fp << get_D1_LPT(x) << " ";
                fp << get_dD1dx_LPT(x) << " ";
                fp << get_D2_LPT(x) << " ";
                fp << get_dD2dx_LPT(x) << " ";

                // 17 lines in total
                fp << "\n";
            };
            std::for_each(x_array.begin(), x_array.end(), print_data);
        }

        //====================================================
        // Get methods
        //====================================================
        double BackgroundCosmology::H_of_x(double x) const { return H_spline(x); }

        double BackgroundCosmology::Hp_of_x(double x) const { return Hp_spline(x); }

        double BackgroundCosmology::dHdx_of_x(double x) const { return H_spline.deriv_x(x); }

        double BackgroundCosmology::ddHddx_of_x(double x) const { return H_spline.deriv_xx(x); }

        double BackgroundCosmology::dHpdx_of_x(double x) const { return Hp_spline.deriv_x(x); }

        double BackgroundCosmology::ddHpddx_of_x(double x) const { return Hp_spline.deriv_xx(x); }

        double BackgroundCosmology::eta_of_x(double x) const {
            if (x == 0.0)
                return eta0;
            return eta_of_x_spline(x);
        }

        double BackgroundCosmology::detadx_of_x(double x) const { return eta_of_x_spline.deriv_x(x); }

        double BackgroundCosmology::get_H0() const { return H0; }

        double BackgroundCosmology::get_h() const { return h; }

        double BackgroundCosmology::get_K() const { return K; }

        double BackgroundCosmology::get_Neff() const { return Neff; }

        double BackgroundCosmology::get_TCMB(double x) const {
            if (x == 0.0)
                return TCMB;
            return TCMB * std::exp(-x);
        }

        double BackgroundCosmology::get_Tnu(double x) const { return std::pow(4.0 / 11.0, 1.0 / 3.0) * get_TCMB(x); }

        double BackgroundCosmology::get_weff(double x) const {
            // return -( 1.0/3.0 + 2.0*dHpdx_of_x(x)/Hp_of_x(x)/3.0 );
            return w_spline(x);
        }

        double BackgroundCosmology::get_dweffdx(double x) const {
            // return -2.0*ddHpddx_of_x(x)/Hp_of_x(x)/3.0 + 2.0/3.0*std::pow(dHpdx_of_x(x)/Hp_of_x(x) ,2);
            return w_spline.deriv_x(x);
        }

        double BackgroundCosmology::get_OmegaB(double x) const {
            if (x == 0.0) {
                return OmegaB;
            }
            return OmegaB * std::exp(-3 * x) / std::pow(H_of_x(x) / H0, 2);
        }

        double BackgroundCosmology::get_OmegaM(double x) const {
            if (x == 0.0) {
                return OmegaM;
            }
            return OmegaM * std::exp(-3 * x) / std::pow(H_of_x(x) / H0, 2);
        }

        double BackgroundCosmology::get_OmegaR(double x) const {
            if (x == 0.0) {
                return OmegaR;
            }
            return OmegaR * std::exp(-4 * x) / std::pow(H_of_x(x) / H0, 2);
        }

        double BackgroundCosmology::get_OmegaRtot(double x) const {
            if (x == 0.0) {
                return OmegaRtot;
            }
            return OmegaRtot * std::exp(-4 * x) / std::pow(H_of_x(x) / H0, 2);
        }

        double BackgroundCosmology::get_OmegaNu(double x) const {
            if (x == 0.0) {
                return OmegaNu;
            }
            return OmegaNu * std::exp(-4 * x) / std::pow(H_of_x(x) / H0, 2);
        }

        double BackgroundCosmology::get_OmegaCDM(double x) const {
            if (x == 0.0) {
                return OmegaCDM;
            }
            return OmegaCDM * std::exp(-3 * x) / std::pow(H_of_x(x) / H0, 2);
        }

        double BackgroundCosmology::get_OmegaLambda(double x) const {
            if (x == 0.0) {
                return OmegaLambda;
            }
            return OmegaLambda / std::pow(H_of_x(x) / H0, 2);
        }

        double BackgroundCosmology::get_OmegaK(double x) const {
            if (x == 0.0) {
                return OmegaK;
            }
            return OmegaK * std::exp(-2 * x) / std::pow(H_of_x(x) / H0, 2);
        }

        std::string BackgroundCosmology::get_name() const { return name; }

        double BackgroundCosmology::get_cosmic_time(double x) const { return cosmic_time_of_x_spline(x); }

        //================================================================
        // Growth factors
        //================================================================

        double BackgroundCosmology::get_D1_LPT(double x) const {
            // 1LPT growth-factor normalized to unity at the present time
            return D1_spline(x);
        }
        double BackgroundCosmology::get_D2_LPT(double x) const {
            // 2LPT growth-factor normalized to unity at the present time
            return D2_spline(x);
        }
        double BackgroundCosmology::get_dD1dx_LPT(double x) const {
            // 1LPT growth-factor normalized to unity at the present time
            return D1_spline.deriv_x(x);
        }
        double BackgroundCosmology::get_dD2dx_LPT(double x) const {
            // 2LPT growth-factor normalized to unity at the present time
            return D2_spline.deriv_x(x);
        }

        //================================================================
        // Distance measures
        //================================================================

        // Comoving distance
        double BackgroundCosmology::chi_of_x(double x) const {
            return chi_of_x_spline(x);
            // return eta0 - eta_of_x(x);
        }

        double BackgroundCosmology::x_of_chi(double x) const { return x_of_chi_spline(x); }

        // Radial comoving coordinate as function of comoving distance
        double BackgroundCosmology::r_of_chi(double chi) const {
            if (std::fabs(OmegaK) < 1e-10)
                return chi;
            double fac = std::sqrt(std::fabs(OmegaK)) * H0 / Constants.c;
            return OmegaK < 0.0 ? std::sin(fac * chi) / fac : std::sinh(fac * chi) / fac;
        }

        // Radial comoving coordinate
        double BackgroundCosmology::r_of_x(double x) const { return r_of_chi(chi_of_x(x)); }

        // Angular diameter distance
        double BackgroundCosmology::dA_of_x(double x) const { return r_of_x(x) * std::exp(x); }

        // Luminosity distance
        double BackgroundCosmology::dL_of_x(double x) const { return r_of_x(x) * std::exp(-2.0 * x); }

    } // namespace COSMOLOGY
} // namespace FML
