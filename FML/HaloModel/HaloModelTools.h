#ifndef FML_HALOMODEL_HALOMODELTOOLS_H
#define FML_HALOMODEL_HALOMODELTOOLS_H

//=====================================================================================================================
// Building blocks shared by the halo models in FML:
//   HaloModel   (Halomodel.h):  the general halo model and HMcode-2016 (Mead et al. 2015, 2016)
//   HMcode2020  (HMcode2020.h): HMcode-2020 (Mead et al. 2021)
//   AxionHMcode (FML/AxionHMcode): the mixed cold/ultralight-axion halo model
//
// Units (unless stated otherwise): k in h/Mpc, R in Mpc/h, M in Msun/h, P(k) in (Mpc/h)^3 and the dimensionless
// power Delta^2(k) = k^3 P(k) / (2 pi^2). Peak height nu = delta_c / sigma(R).
//
// Only depends on GSL (Si and Ci for the NFW transform).
//=====================================================================================================================

#include <FML/HaloModel/MassFunctionFits.h>

#include <algorithm>
#include <cmath>
#include <functional>
#include <vector>

namespace FML {
    namespace COSMOLOGY {
        namespace HALOMODEL {

            //=========================================================================================================
            // Densities and mass-radius relations
            //=========================================================================================================

            /// The critical density today 3 H0^2 / (8 pi G) in (Msun/h) / (Mpc/h)^3 (from the constants in FML/Units).
            /// HMcode-2020 and AxionHMcode keep the rounded constants of their reference codes instead (see there).
            double critical_density();

            /// Mass in a comoving sphere of radius R with mean comoving density rho_mean
            inline double mass_of_radius(double R, double rho_mean) { return 4.0 * M_PI / 3.0 * rho_mean * R * R * R; }
            /// Comoving Lagrangian radius of a mass M with mean comoving density rho_mean
            inline double radius_of_mass(double M, double rho_mean) {
                return std::cbrt(3.0 * M / (4.0 * M_PI * rho_mean));
            }

            //=========================================================================================================
            // Filters and profiles
            //=========================================================================================================

            /// The Fourier transform of a spherical top-hat, W(y) = 3 (sin y - y cos y) / y^3, y = kR
            /// (series for |y| < 0.05, where the closed form loses precision)
            inline double tophat_window(double y) {
                if (std::abs(y) < 0.05) {
                    const double y2 = y * y;
                    return 1.0 - y2 / 10.0 + y2 * y2 / 280.0 - y2 * y2 * y2 / 15120.0;
                }
                return 3.0 * (std::sin(y) - y * std::cos(y)) / (y * y * y);
            }
            /// dW/dy of the top-hat window
            inline double tophat_window_derivative(double y) {
                if (std::abs(y) < 0.05) {
                    const double y2 = y * y;
                    return -y / 5.0 + y * y2 / 70.0 - y * y2 * y2 / 2520.0;
                }
                return 3.0 * ((y * y - 3.0) * std::sin(y) + 3.0 * y * std::cos(y)) / (y * y * y * y);
            }

            /// m(c) = ln(1 + c) - c/(1 + c): the mass of an NFW halo inside c scale radii in units of 4 pi rho_s r_s^3
            inline double nfw_mass_normalisation(double c) { return std::log1p(c) - c / (1.0 + c); }

            /// The Fourier transform u(k) of an NFW profile truncated at the virial radius r_v, normalised to u -> 1
            /// for k -> 0 (Cooray & Sheth 2002, eq. 81). k_rv = k r_v and c = r_v / r_s. A series in k r_v is used
            /// for k r_v < 0.01, where the closed form cancels.
            double nfw_fourier(double k_rv, double c);

            /// Real-space NFW profile as the dimensionless 4 pi r^3 rho(r) / M for r < r_v (zero outside)
            double nfw_profile_r3(double r, double rv, double c);

            //=========================================================================================================
            /// The variance of the linear density field smoothed on a scale R,
            ///   sigma^2(R) = int dln k Delta^2(k) W^2(kR),
            /// and the velocity dispersion sigma_v^2(R) = 1/3 int dln k Delta^2(k) / k^2 W^2(kR)
            /// as sums over quadrature nodes in ln k. The window is the top-hat unless another is given.
            //=========================================================================================================
            class LinearVariance {
              public:
                using Window = std::function<double(double)>;

                LinearVariance() = default;
                /// Trapezoid rule in ln k on n points from kmin to kmax; Delta2_of_k(k) = k^3 P(k) / (2 pi^2)
                LinearVariance(const std::function<double(double)> & Delta2_of_k,
                               double kmin,
                               double kmax,
                               int n,
                               Window window = nullptr);
                /// General quadrature: nodes k_i, weights w_i (in ln k) and Delta^2(k_i)
                LinearVariance(std::vector<double> k,
                               const std::vector<double> & weights_lnk,
                               const std::vector<double> & Delta2,
                               Window window = nullptr);

                double sigma2(double R) const;
                double sigma(double R) const { return std::sqrt(sigma2(R)); }
                /// dln sigma / dln R (analytic derivative of the window; top-hat only)
                double dlnsigma_dlnR(double R) const;
                /// sigma_v^2(R); R = 0 gives the unsmoothed 1D displacement variance
                double sigma_v2(double R = 0.0) const;
                double sigma_v(double R = 0.0) const { return std::sqrt(sigma_v2(R)); }
                /// The radius with sigma(R) = target in [Rmin, Rmax] (bisection in ln R). Returns Rmin if sigma <
                /// target already at Rmin and Rmax if sigma > target at Rmax.
                double radius_of_sigma(double target, double Rmin, double Rmax, double tolerance = 1e-10) const;

                const std::vector<double> & k_nodes() const { return k; }

              private:
                std::vector<double> k;
                std::vector<double> weight; // w_i Delta^2(k_i)
                Window window{};
                double W(double y) const { return window ? window(y) : tophat_window(y); }
            };

            //=========================================================================================================
            // Mass functions and halo bias in the peak-height form: f(nu) dnu is the fraction of the mass in haloes
            // with peak height in [nu, nu + dnu], so dn/dln M = rho_mean / M f(nu) |dnu / dln M|.
            //=========================================================================================================

            /// Sheth & Tormen (1999) multiplicity f(nu) = A (1 + (a nu^2)^-p) exp(-a nu^2 / 2) and the corresponding
            /// peak-background-split bias (Sheth, Mo & Tormen 2001)
            struct ShethTormen {
                double a{0.707};
                double p{0.3};
                double A{normalisation(0.707, 0.3)};

                /// The A that gives int f(nu) dnu = 1: A = sqrt(2a) / (sqrt(pi) + Gamma(1/2 - p) / 2^p)
                static double normalisation(double a, double p);
                double f(double nu) const { return A * (1.0 + std::pow(a * nu * nu, -p)) * std::exp(-a * nu * nu / 2.0); }
                double bias(double nu, double delta_c) const {
                    return 1.0 + (a * nu * nu - 1.0) / delta_c + 2.0 * p / delta_c / (1.0 + std::pow(a * nu * nu, p));
                }
            };

            /// A multiplicity f(sigma) (dn/dln M = rho/M f(sigma) |dln sigma/dln M|, as Tinker et al. 2008 and Watson
            /// et al. 2013 in MassFunctionFits.h) in the peak-height form f(nu) = f(sigma) / nu with sigma = delta_c / nu
            inline double multiplicity_from_fsigma(double fsigma, double nu) { return fsigma / nu; }

            //=========================================================================================================
            // Halo concentration
            //=========================================================================================================

            /// Bullock et al. (2001): c = A (1 + z_f) / (1 + z) = A a / a_f with the formation time x_f = ln a_f of
            /// the halo (never below A). x = ln a.
            inline double bullock2001_concentration(double A, double x, double x_formation) {
                return A * std::max(1.0, std::exp(x - x_formation));
            }

            //=========================================================================================================
            // HMcode collapse fits
            //=========================================================================================================

            /// The growth-history fits of Mead (2017, arXiv:1606.05345) used by HMcode-2020 (and AxionHMcode).
            /// g = D(a)/a and G = (1/a) int_0^a D(a')/a' da' for a growth D = a at early times; Omega_m at that time.
            /// delta_c = delta_c^EdS * delta_c_factor and Delta_v = Delta_v^EdS * Delta_v_factor (EdS values and any
            /// neutrino factors are set by the caller).
            struct Mead2017 {
                static double delta_c_factor(double g, double G, double Omega_m);
                static double Delta_v_factor(double g, double G, double Omega_m);
            };

            /// The HMcode-2016 fits (Mead et al. 2015, arXiv:1505.07833, with the neutrino factors of Mead et al. 2016,
            /// arXiv:1602.02154), as implemented in CAMB (halofit_version = mead2016). sigma8 and sigma_v are of the total
            /// matter at the redshift, n_eff of the cold matter at sigma(R) = delta_c, Omega_m of the total matter.
            struct HMcode2016Fits {
                static double delta_c(double sigma8, double Omega_m, double f_nu) {
                    return (1.59 + 0.0314 * std::log(sigma8)) * (1.0 + 0.0123 * std::log10(Omega_m)) *
                           (1.0 + 0.262 * f_nu);
                }
                static double Delta_v(double Omega_m, double f_nu) {
                    return 418.0 * std::pow(Omega_m, -0.352) * (1.0 + 0.916 * f_nu);
                }
                static double eta(double sigma8) { return 0.603 - 0.3 * sigma8; }
                static double f_damp(double sigma_v100) {
                    return std::max(std::min(0.0095 * std::pow(sigma_v100, 1.37), 0.99), 1e-3);
                }
                static double kstar(double sigma_v) { return 0.584 / sigma_v; }
                static double alpha(double n_eff) { return std::clamp(3.24 * std::pow(1.85, n_eff), 0.5, 2.0); }
                static constexpr double concentration_A = 3.13;
                /// The Dolag et al. (2004) concentration correction for dark energy, with the power 1.5 of Mead et al.
                /// (2016): (g(z=10) / g_LCDM(z=10))^1.5 with the growth g of the model and of LCDM (same Omega_m)
                static double dolag_factor(double growth_ratio_z10) { return std::pow(growth_ratio_z10, 1.5); }
            };

            /// The smoothed transition between the one- and two-halo terms of HMcode, (P1^alpha + P2^alpha)^(1/alpha)
            inline double hmcode_transition(double P1, double P2, double alpha) {
                return std::pow(std::pow(P1, alpha) + std::pow(P2, alpha), 1.0 / alpha);
            }

        } // namespace HALOMODEL
    } // namespace COSMOLOGY
} // namespace FML

#endif
