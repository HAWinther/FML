#ifndef FML_HYBRIDBIASMODIFIEDGRAVITY_HEADER
#define FML_HYBRIDBIASMODIFIEDGRAVITY_HEADER

#include <cmath>
#include <functional>
#include <stdexcept>
#include <vector>

namespace FML {
    namespace CORRELATIONFUNCTIONS {
        namespace HYBRIDBIAS {

            //=====================================================================================================
            ///
            /// Hu-Sawicki f(R) gravity for the LPT predictions of the hybrid bias spectra in the simplest
            /// approximation: the linear growth is scale dependent, D(k,a), and the loop kernels are those of
            /// Einstein-de Sitter (as in GR). The f(R) prediction is then zeldovich_hybrid_bias_spectra /
            /// oneloop_hybrid_bias_spectra with the f(R) linear power spectrum P_fR(k,a) = P_GR(k,a) [D_fR(k,a) /
            /// D_GR(a)]^2 (and the Lagrangian operators built from the f(R) linear field, as in a simulation that
            /// uses D_fR(k,a) for the hybrid bias fields). This misses the scale dependence of the second and third
            /// order growth (and screening) in the kernels (see Aviles & Cervantes-Cota 2017; Brando, Li & Koyama
            /// 2026 for the full treatment).
            ///
            /// The background is flat LCDM (f(R) with |f_R0| << 1) with matter and a cosmological constant
            /// (radiation is neglected). The linear growth equation is
            ///   D'' + (2 + dlogH/dloga) D' = 1.5 Omega_M(a) mu(k,a) D,  mu = 1 + (1/3) k^2 / (k^2 + a^2 m^2(a))
            /// with ' = d/dloga and m^2(a) = H0^2 (Omega_M a^-3 + 4 Omega_L) [(Omega_M a^-3 + 4 Omega_L) / (Omega_M +
            /// 4 Omega_L)]^(n+1) / ((n+1) |f_R0|), both D's starting in the growing mode D = a at aini.
            ///
            //=====================================================================================================

            /// Hu-Sawicki f(R) model on a flat LCDM background
            struct HuSawickiFofR {
                double OmegaM{0.3};   ///< Matter density parameter today
                double fofr0{1e-5};   ///< |f_R0|
                double nfofr{1.0};    ///< The Hu-Sawicki n
            };

            namespace detail {
                /// m^2(a) / H0^2 (the mass of the scalaron)
                inline double fofr_mass2_over_H02(const HuSawickiFofR & model, double a) {
                    const double OmegaL = 1.0 - model.OmegaM;
                    const double fac = model.OmegaM / (a * a * a) + 4.0 * OmegaL;
                    const double fac0 = model.OmegaM + 4.0 * OmegaL;
                    return fac * std::pow(fac / fac0, model.nfofr + 1.0) / ((model.nfofr + 1.0) * model.fofr0);
                }

                /// The linear growth factor (growing mode, D = a at aini) for mu(a) (RK4 in log a)
                inline double linear_growth(const HuSawickiFofR & model,
                                            double a,
                                            const std::function<double(double)> & mu,
                                            double aini,
                                            int nsteps) {
                    const double OmegaM = model.OmegaM, OmegaL = 1.0 - OmegaM;
                    auto deriv = [&](double x, const double * y, double * dydx) {
                        const double aa = std::exp(x);
                        const double E2 = OmegaM / (aa * aa * aa) + OmegaL;
                        const double OmegaMa = OmegaM / (aa * aa * aa) / E2;
                        const double dlogHdx = -1.5 * OmegaMa;
                        dydx[0] = y[1];
                        dydx[1] = 1.5 * OmegaMa * mu(aa) * y[0] - (2.0 + dlogHdx) * y[1];
                    };
                    const double x0 = std::log(aini), x1 = std::log(a), h = (x1 - x0) / nsteps;
                    double y[2] = {aini, aini};
                    double k1[2], k2[2], k3[2], k4[2], yt[2];
                    for (int i = 0; i < nsteps; i++) {
                        const double x = x0 + i * h;
                        deriv(x, y, k1);
                        for (int j = 0; j < 2; j++)
                            yt[j] = y[j] + 0.5 * h * k1[j];
                        deriv(x + 0.5 * h, yt, k2);
                        for (int j = 0; j < 2; j++)
                            yt[j] = y[j] + 0.5 * h * k2[j];
                        deriv(x + 0.5 * h, yt, k3);
                        for (int j = 0; j < 2; j++)
                            yt[j] = y[j] + h * k3[j];
                        deriv(x + h, yt, k4);
                        for (int j = 0; j < 2; j++)
                            y[j] += h / 6.0 * (k1[j] + 2.0 * k2[j] + 2.0 * k3[j] + k4[j]);
                    }
                    return y[0];
                }
            } // namespace detail

            //=====================================================================================================
            /// @brief The ratio D_fR(k,a) / D_GR(a) of the linear growth factors in f(R) and in GR (same
            /// background, both normalized to the same growing mode at early times)
            ///
            /// @param[in] model The f(R) model
            /// @param[in] a The scale factor
            /// @param[in] k_hmpc The wavenumber in h/Mpc
            /// @param[in] aini Where both start in the growing mode D = a (the fifth force is negligible there for
            /// |f_R0| <~ 1e-4)
            /// @param[in] nsteps The number of RK4 steps in log a
            //=====================================================================================================
            inline double fofr_linear_growth_ratio(const HuSawickiFofR & model,
                                                   double a,
                                                   double k_hmpc,
                                                   double aini = 1e-3,
                                                   int nsteps = 2000) {
                if (model.fofr0 <= 0.0)
                    return 1.0;
                // k in units of H0
                const double koverH0 = k_hmpc * 2997.92458;
                auto mu_fofr = [&](double aa) {
                    const double m2a2 = aa * aa * detail::fofr_mass2_over_H02(model, aa);
                    return 1.0 + (1.0 / 3.0) * koverH0 * koverH0 / (koverH0 * koverH0 + m2a2);
                };
                auto mu_gr = [](double) { return 1.0; };
                return detail::linear_growth(model, a, mu_fofr, aini, nsteps) /
                       detail::linear_growth(model, a, mu_gr, aini, nsteps);
            }

            //=====================================================================================================
            /// @brief The f(R) linear power spectrum P_fR(k,a) = P_GR(k,a) [D_fR(k,a) / D_GR(a)]^2 (the growth ratio
            /// is tabulated in log k and interpolated)
            ///
            /// @param[in] pofk_gr The GR (LCDM) linear power spectrum at a (k in h/Mpc)
            /// @param[in] model The f(R) model
            /// @param[in] a The scale factor
            /// @param[in] kmin, kmax The range of the table in h/Mpc (outside it the ratio at the end is used)
            /// @param[in] ntable The number of points in the table
            ///
            /// @return P_fR(k) as a function (it keeps a copy of pofk_gr and the table)
            //=====================================================================================================
            inline std::function<double(double)> fofr_linear_power_spectrum(const std::function<double(double)> & pofk_gr,
                                                                            const HuSawickiFofR & model,
                                                                            double a,
                                                                            double kmin = 1e-5,
                                                                            double kmax = 100.0,
                                                                            int ntable = 400) {
                if (not(kmin > 0.0 and kmax > kmin and ntable >= 2))
                    throw std::runtime_error("[fofr_linear_power_spectrum] Bad table range");
                std::vector<double> ratio2(ntable);
                const double lkmin = std::log(kmin), dlk = std::log(kmax / kmin) / (ntable - 1);
#ifdef USE_OMP
#pragma omp parallel for schedule(dynamic, 4)
#endif
                for (int i = 0; i < ntable; i++) {
                    const double r = fofr_linear_growth_ratio(model, a, std::exp(lkmin + dlk * i));
                    ratio2[i] = r * r;
                }
                return [pofk_gr, ratio2, lkmin, dlk, ntable](double k) {
                    if (k <= 0.0)
                        return 0.0;
                    const double t = std::min(std::max((std::log(k) - lkmin) / dlk, 0.0), double(ntable - 1));
                    const int i = std::min(int(t), ntable - 2);
                    const double f = t - i;
                    return pofk_gr(k) * (ratio2[i] * (1.0 - f) + ratio2[i + 1] * f);
                };
            }

        } // namespace HYBRIDBIAS
    } // namespace CORRELATIONFUNCTIONS
} // namespace FML
#endif
