#include <FML/HaloModel/HaloModelTools.h>
#include <FML/Units/Units.h>

#include <gsl/gsl_sf_expint.h>

#include <stdexcept>
#include <utility>

namespace FML {
    namespace COSMOLOGY {
        namespace HALOMODEL {

            double critical_density() { return FML::UTILS::SI::rho_crit_over_h2_msun_mpc3; }

            double nfw_fourier(double k_rv, double c) {
                const double m = nfw_mass_normalisation(c);
                if (std::abs(k_rv) < 0.01) {
                    // u = 1 - k^2 <r^2> / 6 + k^4 <r^4> / 120 with the moments of the truncated profile
                    // (<r^n> / r_s^n = int_0^c x^(n+1) / (1 + x)^2 dx / m(c)); the next term is < 2e-16
                    const double u = 1.0 + c, lu = std::log1p(c);
                    const double I2 = (u * u - 1.0) / 2.0 - 3.0 * c + 3.0 * lu + 1.0 / u - 1.0;
                    const double I4 = (u * u * u * u - 1.0) / 4.0 - 5.0 * (u * u * u - 1.0) / 3.0 +
                                      5.0 * (u * u - 1.0) - 10.0 * c + 5.0 * lu + 1.0 / u - 1.0;
                    const double q2 = k_rv * k_rv / (c * c); // (k r_s)^2
                    return 1.0 - q2 * I2 / (6.0 * m) + q2 * q2 * I4 / (120.0 * m);
                }
                const double q = k_rv / c, qc = q + k_rv; // k r_s and k r_s (1 + c)
                return (std::sin(q) * (gsl_sf_Si(qc) - gsl_sf_Si(q)) + std::cos(q) * (gsl_sf_Ci(qc) - gsl_sf_Ci(q)) -
                        std::sin(k_rv) / qc) /
                       m;
            }

            double nfw_profile_r3(double r, double rv, double c) {
                // (a relative margin as rv from M might differ in the last digit from the rv the caller has)
                if (r > rv * (1.0 + 1e-10))
                    return 0.0;
                const double x = r * c / rv;
                return x * x / ((1.0 + x) * (1.0 + x)) / nfw_mass_normalisation(c);
            }

            //=========================================================================================================
            // LinearVariance
            //=========================================================================================================

            LinearVariance::LinearVariance(const std::function<double(double)> & Delta2_of_k,
                                           double kmin,
                                           double kmax,
                                           int n,
                                           Window window_)
                : window(std::move(window_)) {
                if (not(kmin > 0.0 and kmax > kmin and n >= 2))
                    throw std::invalid_argument("[LinearVariance] need 0 < kmin < kmax and n >= 2");
                const double dl = std::log(kmax / kmin) / (n - 1);
                k.resize(n);
                weight.resize(n);
                for (int i = 0; i < n; i++) {
                    k[i] = std::exp(std::log(kmin) + i * dl);
                    weight[i] = (i == 0 or i == n - 1 ? 0.5 : 1.0) * dl * Delta2_of_k(k[i]);
                }
            }

            LinearVariance::LinearVariance(std::vector<double> k_,
                                           const std::vector<double> & weights_lnk,
                                           const std::vector<double> & Delta2,
                                           Window window_)
                : k(std::move(k_)), weight(k.size()), window(std::move(window_)) {
                if (k.empty() or weights_lnk.size() != k.size() or Delta2.size() != k.size())
                    throw std::invalid_argument("[LinearVariance] k, weights and Delta^2 must have the same size > 0");
                for (size_t i = 0; i < k.size(); i++)
                    weight[i] = weights_lnk[i] * Delta2[i];
            }

            double LinearVariance::sigma2(double R) const {
                double s = 0.0;
                for (size_t i = 0; i < k.size(); i++) {
                    const double w = W(k[i] * R);
                    s += weight[i] * w * w;
                }
                return s;
            }

            double LinearVariance::dlnsigma_dlnR(double R) const {
                if (window)
                    throw std::runtime_error("[LinearVariance::dlnsigma_dlnR] only for the top-hat window");
                double s = 0.0, ds = 0.0;
                for (size_t i = 0; i < k.size(); i++) {
                    const double y = k[i] * R, w = tophat_window(y);
                    s += weight[i] * w * w;
                    ds += weight[i] * w * tophat_window_derivative(y) * y;
                }
                return ds / s;
            }

            double LinearVariance::sigma_v2(double R) const {
                double s = 0.0;
                for (size_t i = 0; i < k.size(); i++) {
                    const double w = R == 0.0 ? 1.0 : W(k[i] * R);
                    s += weight[i] * w * w / (k[i] * k[i]);
                }
                return s / 3.0;
            }

            double LinearVariance::radius_of_sigma(double target, double Rmin, double Rmax, double tolerance) const {
                double a = std::log(Rmin), b = std::log(Rmax);
                const double fa = sigma(Rmin) - target, fb = sigma(Rmax) - target;
                if (fa <= 0.0)
                    return Rmin;
                if (fb >= 0.0)
                    return Rmax;
                while (b - a > tolerance) {
                    const double m = 0.5 * (a + b);
                    if (sigma(std::exp(m)) - target > 0.0)
                        a = m;
                    else
                        b = m;
                }
                return std::exp(0.5 * (a + b));
            }

            //=========================================================================================================
            // Mass functions and collapse fits
            //=========================================================================================================

            double ShethTormen::normalisation(double a, double p) {
                return std::sqrt(2.0 * a) / (std::sqrt(M_PI) + std::tgamma(0.5 - p) / std::pow(2.0, p));
            }

            double Mead2017::delta_c_factor(double g, double G, double Omega_m) {
                auto fit = [&](double p0, double p1, double p2, double p3) {
                    return p0 + p1 * (1 - g) + p2 * std::pow(1 - g, 2) + p3 * (1 - G);
                };
                return 1 + fit(-0.0069, -0.0208, 0.0312, 0.0021) * std::log10(Omega_m) +
                       fit(0.0001, -0.0647, -0.0417, 0.0646);
            }

            double Mead2017::Delta_v_factor(double g, double G, double Omega_m) {
                auto fit = [&](double p0, double p1, double p2, double p3) {
                    return p0 + p1 * (1 - g) + p2 * std::pow(1 - g, 2) + p3 * (1 - G);
                };
                const double l = std::log10(Omega_m);
                return 1 + fit(-0.79, -10.17, 2.51, 6.51) * l + fit(-1.89, 0.38, 18.8, -15.87) * l * l;
            }

        } // namespace HALOMODEL
    } // namespace COSMOLOGY
} // namespace FML
