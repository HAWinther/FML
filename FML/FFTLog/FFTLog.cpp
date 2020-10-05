#include "FFTLog.h"
#include <array>
#include <iostream>
#include <tgmath.h>

namespace FML {
    namespace SOLVERS {
        namespace FFTLog {

            /// @brief Computes the Gamma function using the Lanczos approximation
            /// See https://en.wikipedia.org/wiki/Lanczos_approximation
            static CDouble gamma(CDouble z) {

                if (z.real() < 0.5) {
                    return M_PI / (std::sin(M_PI * z) * gamma(1. - z));
                }

                // Formula below is for gamma(1+z) so remove 1 to get z
                z -= 1;

                // Lanczos coefficients for g = 7 with way too many digits
                const double g = 7;
                static std::array<double, 9> p = {0.999999999999809932276847,
                                                  676.5203681218850985670091,
                                                  -1259.13921672240287047156,
                                                  771.3234287776530788486528,
                                                  -176.615029162140599065845,
                                                  12.50734327868690481445893,
                                                  -0.13857109526572011689554,
                                                  9.984369578019570859563e-6,
                                                  1.505632735149311558340e-7};
                CDouble Ag = p[0];
                for (int n = 1; n < int(p.size()); n++) {
                    Ag += p[n] / (z + double(n));
                }

                return std::sqrt(2 * M_PI) * std::pow(z + g + 0.5, z + 0.5) * std::exp(-z - g - 0.5) * Ag;
            }

            static void lngamma_4(double x, double y, double * lnr, double * arg) {
                const CDouble w = std::log(gamma(CDouble(x, y)));
                if (lnr)
                    *lnr = w.real();
                if (arg)
                    *arg = w.imag();
            }

            /// @brief Internal method. Compute a "good" value of k*r
            static double goodkr(int N, double mu, double q, double L, double kr) {
                const double xp = (mu + 1 + q) / 2;
                const double xm = (mu + 1 - q) / 2;
                const double y = M_PI * N / (2 * L);

                double lnr, argm, argp;
                lngamma_4(xp, y, &lnr, &argp);
                lngamma_4(xm, y, &lnr, &argm);

                const double arg = std::log(2 / kr) * N / L + (argp + argm) / M_PI;
                const double iarg = round(arg);
                if (arg != iarg) {
                    kr *= std::exp((arg - iarg) * L / N);
                }
                return kr;
            }

            /// @brief Internal method. Compute the u coefficients
            CVector ComputeCoefficients(int N, double mu, double q, double L, double kcrc) {
                const double y = M_PI / L;
                const double k0r0 = kcrc * std::exp(-L);
                const double t = -2 * y * std::log(k0r0 / 2);
                CVector u(N);

                if (q == 0) {
                    const double x = (mu + 1) / 2;
                    double lnr, phi;
                    for (int m = 0; m <= N / 2; m++) {
                        lngamma_4(x, m * y, &lnr, &phi);
                        u[m] = std::polar(1.0, m * t + 2 * phi);
                    }
                } else {
                    const double xp = (mu + 1 + q) / 2;
                    const double xm = (mu + 1 - q) / 2;
                    double lnrp, phip, lnrm, phim;
                    for (int m = 0; m <= N / 2; m++) {
                        lngamma_4(xp, m * y, &lnrp, &phip);
                        lngamma_4(xm, m * y, &lnrm, &phim);
                        u[m] = std::polar(std::exp(q * std::log(2) + lnrp - lnrm), m * t + phip - phim);
                    }
                }

                for (int m = N / 2 + 1; m < N; m++) {
                    u[m] = std::conj(u[N - m]);
                }
                if ((N % 2) == 0) {
                    u[N / 2] = u[N / 2].real();
                }
                return u;
            }

            std::pair<DVector, CVector> DiscreteHankelTransform(const DVector & r,
                                                                const CVector & a,
                                                                double mu,
                                                                double q,
                                                                double kcrc,
                                                                int noring,
                                                                CDouble * u) {
                const int N = int(r.size());
                const double L = std::log(r[N - 1] / r[0]) * N / (N - 1.);
                CVector b(N);
                CVector ulocal;
                if (u == nullptr) {
                    if (noring) {
                        kcrc = goodkr(N, mu, q, L, kcrc);
                    }
                    ulocal = ComputeCoefficients(N, mu, q, L, kcrc);
                    u = ulocal.data();
                }

                // Compute the convolution b = a*u using FFTs
                // NB: don't use FFTW_MEASURE as it will overwrite the a-array. To use this we must
                // take a copy and copy back after the plans have been made
                fftw_complex * grid_in = reinterpret_cast<fftw_complex *>(const_cast<CDouble *>(a.data()));
                fftw_complex * grid_out = reinterpret_cast<fftw_complex *>(b.data());
                fftw_plan forward_plan = fftw_plan_dft_1d(N, grid_in, grid_out, FFTW_FORWARD, FFTW_ESTIMATE);
                fftw_plan reverse_plan = fftw_plan_dft_1d(N, grid_out, grid_out, FFTW_BACKWARD, FFTW_ESTIMATE);

                // Transform
                fftw_execute(forward_plan);

                // Multiply by u
                const double fftw_norm = 1.0 / double(N);
                for (int m = 0; m < N; m++) {
                    b[m] *= u[m] * fftw_norm;
                }

                // Transform back
                fftw_execute(reverse_plan);
                fftw_destroy_plan(forward_plan);
                fftw_destroy_plan(reverse_plan);

                // Reverse b array
                for (int n = 0; n < N / 2; n++) {
                    const auto tmp = b[n];
                    b[n] = b[N - n - 1];
                    b[N - n - 1] = tmp;
                }

                // Compute k's corresponding to input r's (or vice versa)
                const double k0r0 = kcrc * std::exp(-L);
                DVector k(N);
                k[0] = k0r0 / r[0];
                for (int n = 1; n < N; n++) {
                    k[n] = k[0] * std::exp(n * L / N);
                }
                return {k, b};
            }

            std::pair<DVector, DVector> ComputeXiLM(int ell, int m, const DVector & k, const DVector & pk) {
                assert(k.size() == pk.size());

                // Set the integrand
                CVector a(k.size());
                for (size_t i = 0; i < k.size(); i++) {
                    a[i] = std::pow(k[i], m - 0.5) * pk[i];
                }

                // Transform
                auto result = DiscreteHankelTransform(k, a, ell + 0.5, 0, 1, 1, nullptr);
                auto & r = result.first;
                auto & b = result.second;

                // Set output and normalize
                DVector xi(k.size());
                for (size_t i = 0; i < xi.size(); i++) {
                    xi[i] = std::pow(2 * M_PI * r[i], -1.5) * b[i].real();
                }

                return {r, xi};
            }

            std::pair<DVector, DVector> ComputeCorrelationFunction(const DVector & k, const DVector & pk) {
                return ComputeXiLM(0, 2, k, pk);
            }

            std::pair<DVector, DVector> ComputePowerSpectrum(const DVector & r, const DVector & xi) {
                auto result = ComputeXiLM(0, 2, r, xi);

                // Normalize. There is a factor (2pi)^3 difference relative to xi(r)
                for (auto & pk : result.second) {
                    pk *= 8 * M_PI * M_PI * M_PI;
                }

                return result;
            }
        } // namespace FFTLog
    }     // namespace SOLVERS
} // namespace FML
