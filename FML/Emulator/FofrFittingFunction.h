#ifndef FML_EMULATOR_FOFRFITTINGFUNCTION_H
#define FML_EMULATOR_FOFRFITTINGFUNCTION_H

//=====================================================================================================================
// Fitting functions for the enhancement of the matter power spectrum in Hu-Sawicky f(R) gravity (n = 1) with respect
// to LCDM with the same initial conditions, B(k, z) = P_f(R)(k, z) / P_LCDM(k, z):
//
//   nonlinear_boost(z, fR0, k): the nonlinear P(k), Winther et al. 2019 (arXiv:1903.08798, PRD 100, 123540), eq. 7
//   linear_boost(z, fR0, k):    linear theory, the same paper, eq. 5
//
// From https://github.com/HAWinther/FofrFittingFunction (C/fit.c and Python/fit_lin.py). |fR0| > 0, k in h/Mpc.
//
// Nonlinear: calibrated on the ELEPHANT simulations (ECOSMOG; WMAP9: Omega_m = 0.281, Omega_b = 0.046, h = 0.6974,
// n_s = 0.971, sigma8 = 0.82) for 1e-7 <= |fR0| <= 1e-4, 0 <= z <= 3 and 0.01 <= k <= 10 h/Mpc. The paper quotes
// <~1% for most scales and redshifts (worse at the smallest scales for the largest |fR0|), 1-2% for k < 1 and 2-3% for
// k < 10 h/Mpc in an out-of-sample test (|fR0| = 2e-5). A 10% change in Omega_m (sigma8) changes the enhancement by
// ~1% (2-3%); n_s and H0 have a negligible effect. Three fits around |fR0| = 5e-6, 1e-5 and 5e-5 are interpolated
// linearly in |fR0|. Outside the fitted ranges the value at the closest point is returned (k > 10 h/Mpc: the value
// at k = 10; small k tends to 1 by design), and the result is never below 1.
//
// Linear: fitted for |fR0| = 1e-5 and Omega_m = 0.281, other |fR0| by the scaling k -> k sqrt(|fR0| / 1e-5);
// 1e-7 <= |fR0| <= 1e-3, z <= 5 and an effective k <= 100 h/Mpc (clamped). (C/fit_lin.c in the repository tests
// |k - 1| with the integer abs, which sets k = 1 for all effective k < 2: the Python version, used here, does not.)
//=====================================================================================================================

#include <algorithm>
#include <cmath>

namespace FML {
    namespace EMULATOR {
        namespace FOFRFITTINGFUNCTION {

            namespace detail {
                // The three nonlinear fits, around |fR0| = 5e-6, 1e-5 and 5e-5 (51 parameters each)
                constexpr double fR0_low = 5e-6;
                constexpr double param_low[51] = {
                    0.768779,   -0.405375,  0.0075176,  0.0288574,   -0.0638224, -0.401206,  0.369507,   0.109392,
                    -0.342089,  0.226376,   -0.107105,  0.0484649,   -0.024377,  -0.051962,  -0.0351849, 0.147194,
                    0.061761,   -0.131382,  0.00759035, -0.00101884, 0.0118011,  -0.0296267, 0.025968,   0.076885,
                    0.0312734,  0.0293253,  -0.0141899, 0.109011,    0.0818948,  -0.0568241, 0.120272,   0.0249235,
                    -0.0298492, 0.0354401,  -0.262769,  0.230278,    -0.139116,  -0.132313,  0.13132,    -0.0565551,
                    -0.0338864, 0.0712653,  0.20246,    -0.116113,   0.102453,   0.0632254,  0.0694305,  0.00296431,
                    0.0522931,  0.0780708,  -0.0977045};
                constexpr double fR0_mid = 1e-5;
                constexpr double param_mid[51] = {
                    0.936496,    -0.545832,  0.634804,  -0.0290649,   -0.0954373,  -0.342491, 0.491066,  0.297816,
                    -0.287142,   -0.0399919, 0.3037,    0.360959,     0.000615209, -0.00941931, -0.0181341, 0.376297,
                    0.486358,    0.0349385,  0.240066,  0.188202,     0.665834,    0.0122249, -0.0343399, -0.0520361,
                    0.261006,    0.525633,   0.266255,  0.393546,     0.29088,     -0.411491, 0.776609,  0.470777,
                    -0.681923,   -0.079589,  -0.282388, 0.53954,      -0.0930797,  0.0783781, 0.194957,  0.270378,
                    0.370288,    0.194857,   0.318637,  0.0457011,    0.139237,    0.033403,  0.0762982, -0.000104704,
                    -0.00275824, 0.0461644,  0.189897};
                constexpr double fR0_high = 5e-5;
                constexpr double param_high[51] = {
                    0.572477,    0.254686,  1.21637,   0.00046274, -0.0901242, -0.355849,   2.31154,    2.29822,
                    -0.483186,   0.4988,    0.36089,   0.0703424,  0.0257389,  0.0168936,   -0.030697,  -0.206992,
                    0.266084,    0.603357,  0.574264,  -0.30799,   0.831644,   -0.00936448, 0.00221153, 0.0076829,
                    -0.650381,   0.0179215, 0.927038,  0.77903,    0.919643,   -0.936328,   1.26756,    1.44477,
                    -1.44129,    0.219594,  0.353883,  1.02533,    -0.251705,  0.124875,    0.345995,   -0.146438,
                    0.0200251,   0.0892343, 0.284755,  -0.158286,  0.541178,   -0.0471913,  0.139772,   -0.134888,
                    0.0959162,   0.368186,  -0.157828};

                // The fitting formula, r = ln(|fR0| / |fR0|_fid)
                inline double ratio_by_param(double r, double aminusone, double k, const double * param) {
                    const double am2 = aminusone * aminusone, r2 = r * r;
                    auto coeff = [&](int i0, int i1, int i2) { return param[i0] + param[i1] * r + param[i2] * r2; };
                    auto in_a = [&](double x0, double x1, double x2) { return x0 + x1 * aminusone + x2 * am2; };
                    const double b = in_a(coeff(0, 9, 18), coeff(1, 10, 19), coeff(2, 11, 20));
                    const double c = in_a(coeff(3, 12, 21), coeff(4, 13, 22), coeff(5, 14, 23));
                    const double d = in_a(coeff(6, 15, 24), coeff(7, 16, 25), coeff(8, 17, 26));
                    const double e = in_a(1.0 + param[27] * r + param[30] * r2,
                                          param[28] * r + param[31] * r2,
                                          param[29] * r + param[32] * r2);
                    const double f = in_a(coeff(33, 36, 39), coeff(34, 37, 40), coeff(35, 38, 41));
                    const double g = in_a(coeff(42, 45, 48), coeff(43, 46, 49), coeff(44, 47, 50));
                    return 1.0 + b * (1.0 + c * k) / (1.0 + e * k) * std::pow(std::atan(d * k), 1.0 + f + g * k);
                }
            } // namespace detail

            /// The ranges of the nonlinear fit (outside them the closest value is used)
            constexpr double nonlinear_kmax = 10.0;
            constexpr double nonlinear_zmax = 3.0;
            constexpr double nonlinear_fR0min = 1e-7;
            constexpr double nonlinear_fR0max = 1e-4;

            /// P_f(R) / P_LCDM of the nonlinear matter power spectrum for Hu-Sawicky f(R) with n = 1 (Winther et al.
            /// 2019). fR0 is |f_R0| (the sign is ignored), k in h/Mpc.
            inline double nonlinear_boost(double z, double fR0, double k) {
                k = std::min(k, nonlinear_kmax);
                z = std::clamp(z, 0.0, nonlinear_zmax);
                fR0 = std::clamp(std::abs(fR0), nonlinear_fR0min, nonlinear_fR0max);
                const double am1 = 1.0 / (1.0 + z) - 1.0;
                using namespace detail;
                double ratio;
                if (fR0 > fR0_high) {
                    ratio = ratio_by_param(std::log(fR0 / fR0_high), am1, k, param_high);
                } else if (fR0 < fR0_low) {
                    ratio = ratio_by_param(std::log(fR0 / fR0_low), am1, k, param_low);
                } else if (fR0 > fR0_mid) {
                    const double mid = ratio_by_param(std::log(fR0 / fR0_mid), am1, k, param_mid);
                    const double high = ratio_by_param(std::log(fR0 / fR0_high), am1, k, param_high);
                    ratio = mid + (high - mid) * (fR0 - fR0_mid) / (fR0_high - fR0_mid);
                } else {
                    const double low = ratio_by_param(std::log(fR0 / fR0_low), am1, k, param_low);
                    const double mid = ratio_by_param(std::log(fR0 / fR0_mid), am1, k, param_mid);
                    ratio = low + (mid - low) * (fR0 - fR0_low) / (fR0_mid - fR0_low);
                }
                return std::max(ratio, 1.0);
            }

            /// P_f(R) / P_LCDM in linear theory for Hu-Sawicky f(R) with n = 1 (Winther et al. 2019, eq. 5). fR0 is
            /// |f_R0|, k in h/Mpc.
            inline double linear_boost(double z, double fR0, double k) {
                constexpr double fR0_fid = 1e-5;
                constexpr double param[12] = {3.1,     2.34466, -1.86362,  34.4951, 28.8637, -13.1302,
                                              0.146542, -0.0103024, -0.149446, 1.62807, 0.71291, -1.41003};
                fR0 = std::clamp(std::abs(fR0), 1e-7, 1e-3);
                // The effective wavenumber
                k = std::min(k * std::sqrt(fR0 / fR0_fid), 100.0);
                z = std::clamp(z, 0.0, 5.0);
                if (std::abs(k - 1.0) < 1e-10)
                    k = 1.0 + 1e-10;
                if (k < 1e-10)
                    return 1.0;
                const double am1 = 1.0 / (1.0 + z) - 1.0;
                auto in_a = [&](int i) { return param[i] + param[i + 1] * am1 + param[i + 2] * am1 * am1; };
                const double b = in_a(0), c = in_a(3), d = in_a(6), e = in_a(9);
                return 1.0 + (b * k) * (b * k) / (1.0 + c * k * k) +
                       std::abs(std::log(k) * k / (k - 1.0)) * d * std::atan(e * k);
            }

        } // namespace FOFRFITTINGFUNCTION
    } // namespace EMULATOR
} // namespace FML

#endif
