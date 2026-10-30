#include "FFTLog.h"
#include <array>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <utility>
#include <tgmath.h>

namespace FML {
    namespace SOLVERS {
        namespace FFTLog {

            /// @brief log Gamma(z) from the log of the Lanczos formula (g = 7, see
            /// https://en.wikipedia.org/wiki/Lanczos_approximation). Gamma itself under/overflows for |Im z| >~ 470
            /// (log(gamma(z)) gave -inf with a random phase or NaN). For Re z < 1/2 the reflection formula with a
            /// stable log(sin(pi z))
            static CDouble lngamma(CDouble z) {
                if (z.real() < 0.5) {
                    // log(pi / sin(pi z)) - log Gamma(1 - z) with log(sin(pi z)) computed without overflow:
                    // sin(pi z) = (e^{i pi z} - e^{-i pi z}) / 2i, factor out the larger exponential
                    const CDouble I(0, 1);
                    const CDouble ipz = I * M_PI * z;
                    const double s = ipz.real() >= 0 ? 1.0 : -1.0;
                    const CDouble logsin = s * ipz + std::log((1.0 - std::exp(-2.0 * s * ipz)) / (2.0 * I * s));
                    return std::log(M_PI) - logsin - lngamma(1. - z);
                }
                z -= 1;
                const double g = 7;
                static const std::array<double, 9> p = {0.999999999999809932276847,
                                                        676.5203681218850985670091,
                                                        -1259.13921672240287047156,
                                                        771.3234287776530788486528,
                                                        -176.615029162140599065845,
                                                        12.50734327868690481445893,
                                                        -0.13857109526572011689554,
                                                        9.984369578019570859563e-6,
                                                        1.505632735149311558340e-7};
                CDouble Ag = p[0];
                for (int n = 1; n < int(p.size()); n++)
                    Ag += p[n] / (z + double(n));
                const CDouble t = z + g + 0.5;
                return 0.5 * std::log(2 * M_PI) + (z + 0.5) * std::log(t) - t + std::log(Ag);
            }

            static void lngamma_4(double x, double y, double * lnr, double * arg) {
                const CDouble w = lngamma(CDouble(x, y));
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
                        u[m] = std::polar(std::exp(q * std::log(2) + lnrp - lnrm), m * t + phip + phim);
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

            struct Workspace::Impl {
                size_t n;
                fftw_complex * input = nullptr;
                fftw_complex * output = nullptr;
                fftw_plan forward = nullptr;
                fftw_plan reverse = nullptr;
                Statistics stats;
                CVector coefficients;
                double mu = 0, q = 0, L = 0, requested_kr = 0, effective_kr = 0;
                bool noring = false;

                explicit Impl(size_t n) : n(n) {}
                ~Impl() {
                    if (forward)
                        fftw_destroy_plan(forward);
                    if (reverse)
                        fftw_destroy_plan(reverse);
                    fftw_free(input);
                    fftw_free(output);
                }
            };

            Workspace::Workspace(unsigned flags) : planner_flags(flags) {}
            Workspace::~Workspace() = default;
            Workspace::Workspace(Workspace &&) noexcept = default;
            Workspace & Workspace::operator=(Workspace &&) noexcept = default;
            void Workspace::clear() noexcept { impl.reset(); }
            size_t Workspace::size() const noexcept { return impl ? impl->n : 0; }
            size_t Workspace::buffer_bytes() const noexcept { return 2 * size() * sizeof(fftw_complex); }
            Workspace::Statistics Workspace::statistics() const noexcept { return impl ? impl->stats : Statistics{}; }

            void Workspace::prepare(size_t n) {
                if (n < 2 || n > size_t(std::numeric_limits<int>::max()) ||
                    n > std::numeric_limits<size_t>::max() / (2 * sizeof(fftw_complex)))
                    throw std::invalid_argument("FFTLog: transform length must be in [2, INT_MAX] and fit in memory");
                if (size() == n)
                    return;
                // Build separately so allocation/planning failure preserves the previous workspace.
                auto next = std::make_unique<Impl>(n);
                next->input = fftw_alloc_complex(n);
                next->output = fftw_alloc_complex(n);
                if (!next->input || !next->output)
                    throw std::bad_alloc();
                std::memset(next->input, 0, n * sizeof(fftw_complex));
                std::memset(next->output, 0, n * sizeof(fftw_complex));
                next->forward = fftw_plan_dft_1d(int(n), next->input, next->output, FFTW_FORWARD, planner_flags);
                next->reverse = fftw_plan_dft_1d(int(n), next->output, next->output, FFTW_BACKWARD, planner_flags);
                if (!next->forward || !next->reverse)
                    throw std::runtime_error("FFTLog: FFTW could not create transform plans");
                next->stats = statistics();
                next->stats.plan_builds += 2;
                impl.swap(next);
            }

            static double logarithmic_span(const DVector & r) {
                if (r.size() < 2 || r.size() > size_t(std::numeric_limits<int>::max()))
                    throw std::invalid_argument("FFTLog: transform length must be in [2, INT_MAX]");
                const double L = std::log(r.back() / r.front()) * r.size() / (r.size() - 1.0);
                if (!(r.front() > 0) || !std::isfinite(r.front()) || !std::isfinite(r.back()) || !(L > 0) ||
                    !std::isfinite(L))
                    throw std::invalid_argument("FFTLog: expected a positive, increasing logarithmic grid");
                return L;
            }

            DVector Workspace::execute(const DVector & r,
                                       double mu,
                                       double q,
                                       double kr,
                                       bool noring,
                                       const CVector * supplied) {
                const double L = logarithmic_span(r);
                if (!(kr > 0) || !std::isfinite(kr))
                    throw std::invalid_argument("FFTLog: kcrc must be positive and finite");
                auto & state = *impl;
                const size_t n = state.n;
                const CVector * u = supplied;
                if (!supplied) {
                    if (state.coefficients.empty() || state.mu != mu || state.q != q || state.L != L ||
                        state.requested_kr != kr || state.noring != noring) {
                        const double effective = noring ? goodkr(int(n), mu, q, L, kr) : kr;
                        auto coefficients = ComputeCoefficients(int(n), mu, q, L, effective);
                        state.coefficients = std::move(coefficients);
                        state.mu = mu;
                        state.q = q;
                        state.L = L;
                        state.requested_kr = kr;
                        state.effective_kr = effective;
                        state.noring = noring;
                        ++state.stats.coefficient_builds;
                    }
                    kr = state.effective_kr;
                    u = &state.coefficients;
                }
                // Supplied coefficients retain the historical convention: no goodkr adjustment,
                // even with noring=true. They are read on every call and never cached.
                fftw_execute(state.forward);
                const double norm = 1.0 / double(n);
                for (size_t i = 0; i < n; ++i) {
                    const auto value = CDouble(state.output[i][0], state.output[i][1]) * ((*u)[i] * norm);
                    state.output[i][0] = value.real();
                    state.output[i][1] = value.imag();
                }
                fftw_execute(state.reverse);
                DVector k(n);
                // The reversed output satisfies k[0] * r[n-1] = kr. Include one
                // log step relative to exp(-L), since L is n steps rather than n-1.
                k[0] = kr * std::exp(-L + L / n) / r[0];
                for (size_t i = 1; i < n; ++i)
                    k[i] = k[0] * std::exp(i * L / n);
                return k;
            }

            std::pair<DVector, CVector> DiscreteHankelTransform(const DVector & r,
                                                                const CVector & a,
                                                                Workspace & workspace,
                                                                double mu,
                                                                double q,
                                                                double kcrc,
                                                                bool noring,
                                                                CVector * u) {
                logarithmic_span(r);
                if (a.size() != r.size() || (u && u->size() != r.size()))
                    throw std::invalid_argument("FFTLog: input and coefficient lengths must match the grid");
                workspace.prepare(r.size());
                auto & state = *workspace.impl;
                for (size_t i = 0; i < r.size(); ++i) {
                    state.input[i][0] = a[i].real();
                    state.input[i][1] = a[i].imag();
                }
                auto k = workspace.execute(r, mu, q, kcrc, noring, u);
                CVector b(r.size());
                for (size_t i = 0; i < r.size(); ++i) {
                    const auto & value = state.output[r.size() - 1 - i];
                    b[i] = CDouble(value[0], value[1]);
                }
                return {std::move(k), std::move(b)};
            }

            std::pair<DVector, CVector> DiscreteHankelTransform(const DVector & r,
                                                                const CVector & a,
                                                                double mu,
                                                                double q,
                                                                double kcrc,
                                                                bool noring,
                                                                CVector * u) {
                Workspace workspace;
                return DiscreteHankelTransform(r, a, workspace, mu, q, kcrc, noring, u);
            }

            std::pair<DVector, DVector>
            ComputeXiLM(int ell, int m, const DVector & k, const DVector & pk, Workspace & workspace, double q) {
                logarithmic_span(k);
                if (k.size() != pk.size())
                    throw std::invalid_argument("FFTLog: spectrum and grid lengths must match");
                workspace.prepare(k.size());
                auto & state = *workspace.impl;
                // Fill aligned input directly and normalize output directly, avoiding two temporary complex vectors.
                for (size_t i = 0; i < k.size(); ++i) {
                    state.input[i][0] = std::pow(k[i], m - 0.5 - q) * pk[i];
                    state.input[i][1] = 0;
                }
                auto r = workspace.execute(k, ell + 0.5, q, 1, true, nullptr);
                DVector xi(k.size());
                for (size_t i = 0; i < xi.size(); ++i)
                    xi[i] = std::pow(2 * M_PI * r[i], -1.5) * std::pow(r[i], -q) * state.output[k.size() - 1 - i][0];
                return {std::move(r), std::move(xi)};
            }

            std::pair<DVector, DVector> ComputeXiLM(int ell, int m, const DVector & k, const DVector & pk, double q) {
                Workspace workspace;
                return ComputeXiLM(ell, m, k, pk, workspace, q);
            }

            std::pair<DVector, DVector>
            ComputeCorrelationFunction(const DVector & k, const DVector & pk, Workspace & workspace) {
                return ComputeXiLM(0, 2, k, pk, workspace);
            }
            std::pair<DVector, DVector> ComputeCorrelationFunction(const DVector & k, const DVector & pk) {
                Workspace workspace;
                return ComputeCorrelationFunction(k, pk, workspace);
            }

            std::pair<DVector, DVector>
            ComputePowerSpectrum(const DVector & r, const DVector & xi, Workspace & workspace) {
                auto result = ComputeXiLM(0, 2, r, xi, workspace);
                for (auto & pk : result.second)
                    pk *= 8 * M_PI * M_PI * M_PI;
                return result;
            }
            std::pair<DVector, DVector> ComputePowerSpectrum(const DVector & r, const DVector & xi) {
                Workspace workspace;
                return ComputePowerSpectrum(r, xi, workspace);
            }
        } // namespace FFTLog
    } // namespace SOLVERS
} // namespace FML
