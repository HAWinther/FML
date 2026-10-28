#ifndef FML_HYBRIDBIASONELOOP_HEADER
#define FML_HYBRIDBIASONELOOP_HEADER

#include <FML/ComputePowerSpectra/HybridBiasZeldovich.h>

namespace FML {
    namespace CORRELATIONFUNCTIONS {
        namespace HYBRIDBIAS {

            //=====================================================================================================
            ///
            /// The one-loop Lagrangian perturbation theory (CLEFT: Carlson, Reid & White 2013; Vlah, Castorina & White
            /// 2016; Chen, Vlah & White 2020) prediction for the hybrid bias basis spectra P_ab(k) of the operators 1,
            /// delta_L, delta_L^2, s^2 (see HybridBiasSpectra.h).
            ///
            /// The bias operators are built from the linear field (as in the simulation measurement) and the
            /// displacement is Psi = Psi^(1) + Psi^(2) + Psi^(3). The linear A_ij(q) is kept in the exponent (resummed)
            /// and everything else is expanded to one loop. The q-functions and the integrands follow velocileptors
            /// (LPT/cleft_fftw.py, Utils/qfuncfft.py) term by term, with the loop integrals Q_n(k), R_n(k) (Matsubara
            /// 2008) computed directly in k-space so that a sharp cut of P(k) is handled exactly. The relation to the
            /// velocileptors components (bias monomials 1, b1, b1^2, b2, b1 b2, b2^2, bs, b1 bs, b2 bs, bs^2 with
            /// delta_t = 1 + b1 delta + b2/2 delta^2 + bs s^2) is P_11 = [1], P_1d = [b1]/2, P_dd = [b1^2], P_1d2 =
            /// [b2], P_dd2 = [b1 b2], P_d2d2 = 4 [b2^2], P_1s2 = [bs]/2, P_ds2 = [b1 bs]/2, P_d2s2 = [b2 bs], P_s2s2 =
            /// [bs^2].
            ///
            /// The spectra with nabla^2 delta (not part of velocileptors) are computed at the same order: nabla^2
            /// delta_L = F(k) delta_L(k) with F = -k^2 is a linear operator like delta_L, so the one-loop terms are
            /// those of delta_L with F on the leg of the operator. The Zel'dovich part (zeldovich_hybrid_bias_spectra)
            /// plus, for P_1n: U3 (F(k) R1) and A10 = <delta_1 Delta_i Delta_j> (R_n with F on the external or on the
            /// internal leg and Q5 with F on a loop leg); for P_dn and P_nn: U11 = <a_1 b_2 Delta_i>. P_d2n and P_s2n
            /// are complete at Zel'dovich order (as P_dd2 and P_ds2). For F = 1 the expressions reduce to those of
            /// velocileptors (checked to 1e-6), and the corrections agree with the difference between particles
            /// moved with 2LPT and with the Zel'dovich approximation (small amplitude) to a few percent, as for P_1d
            /// and P_dd. No counterterms are included (the usual alpha k^2 P counterterm is degenerate with the
            /// nabla^2 delta operator).
            ///
            /// The filter W(k) (e.g. a sharp cut at the Nyquist frequency of the particle lattice) is applied to the
            /// linear field, so the same filter is used for the operators and the displacement. Units: those of P(k).
            ///
            //=====================================================================================================

            namespace detail {

                /// Matsubara's R1, R2 kernels (series for small and large r)
                inline void R_kernels(double r, double & R1, double & R2) {
                    if (r < 0.05) {
                        const double r2 = r * r;
                        R1 = 16.0 * r2 / 15.0 - 16.0 * r2 * r2 / 35.0 + 16.0 * r2 * r2 * r2 / 315.0;
                        R2 = 4.0 * r2 / 15.0 - 12.0 * r2 * r2 / 35.0 + 4.0 * r2 * r2 * r2 / 63.0;
                        return;
                    }
                    if (r > 20.0) {
                        const double x2 = 1.0 / (r * r);
                        R1 = 16.0 / 15.0 - 16.0 * x2 / 35.0 + 16.0 * x2 * x2 / 315.0;
                        R2 = -4.0 / 15.0 + 12.0 * x2 / 35.0 - 4.0 * x2 * x2 / 63.0;
                        return;
                    }
                    if (std::abs(1.0 - r) < 1e-12) {
                        // the limit r -> 1 (the log terms vanish)
                        R1 = 2.0 / 3.0;
                        R2 = 0.0;
                        return;
                    }
                    const long double rr = r, r2 = rr * rr;
                    const long double L = std::log(std::abs((1.0L + rr) / (1.0L - rr)));
                    R1 = double(-(1 + r2) / (24 * r2) * (3 - 14 * r2 + 3 * r2 * r2) +
                                std::pow(r2 - 1, 4) / (16 * r2 * rr) * L);
                    R2 = double((1 - r2) / (24 * r2) * (3 - 2 * r2 + 3 * r2 * r2) +
                                std::pow(r2 - 1, 3) * (1 + r2) / (16 * r2 * rr) * L);
                }

                /// Composite Gauss-Legendre on [a,b] with nseg segments (log spaced if log_spacing and a > 0)
                inline void composite_gl(double a,
                                         double b,
                                         int nseg,
                                         int order,
                                         bool log_spacing,
                                         std::vector<double> & x,
                                         std::vector<double> & w) {
                    static thread_local std::map<int, std::pair<std::vector<double>, std::vector<double>>> cache;
                    if (cache.find(order) == cache.end()) {
                        std::vector<double> gx, gw;
                        gauss_legendre(order, gx, gw);
                        cache[order] = {gx, gw};
                    }
                    const auto & gx = cache[order].first;
                    const auto & gw = cache[order].second;
                    x.clear();
                    w.clear();
                    if (b <= a)
                        return;
                    for (int s = 0; s < nseg; s++) {
                        double lo, hi;
                        if (log_spacing and a > 0.0) {
                            lo = a * std::pow(b / a, double(s) / nseg);
                            hi = a * std::pow(b / a, double(s + 1) / nseg);
                        } else {
                            lo = a + (b - a) * s / nseg;
                            hi = a + (b - a) * (s + 1) / nseg;
                        }
                        for (int i = 0; i < order; i++) {
                            x.push_back(0.5 * (lo + hi) + 0.5 * (hi - lo) * gx[i]);
                            w.push_back(0.5 * (hi - lo) * gw[i]);
                        }
                    }
                }

                /// The one-loop spectra Q1, Q2, Q5, Q8, Qs2, R1, R2 (Matsubara 2008; Qs2 from the shear terms of
                /// CLEFT) at k for a power spectrum that is zero above kmax
                /// With a weight F(p) on one leg (for an operator F(k) delta_L(k), e.g. F = -k^2 for nabla^2 delta):
                /// Q5F = Q5 with F on the leg of the operator, R1F, R2F = R1, R2 with F on the internal leg
                struct LoopSpectra {
                    double Q1, Q2, Q5, Q8, Qs2, R1, R2, Q5F, R1F, R2F;
                };
                inline LoopSpectra loop_spectra(double k,
                                                const std::function<double(double)> & P,
                                                double kmin,
                                                double kmax,
                                                const std::function<double(double)> & F) {
                    LoopSpectra res{0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
                    std::vector<double> px, pw, qx, qw;
                    // Q_n = k/(4 pi^2) int dp int dp' (p'/p) P(p) P(p') Q~_n(r = p/k, x) with p' = |k - p|
                    // The p-integral is split at k (where the lower limit of p' is zero)
                    std::vector<double> breaks{kmin};
                    if (k > kmin and k < kmax)
                        breaks.push_back(k);
                    if (std::abs(kmax - k) > kmin and std::abs(kmax - k) < kmax and std::abs(kmax - k) != k)
                        breaks.push_back(std::abs(kmax - k));
                    breaks.push_back(kmax);
                    std::sort(breaks.begin(), breaks.end());
                    for (size_t b = 0; b + 1 < breaks.size(); b++) {
                        const double lo = breaks[b], hi = breaks[b + 1];
                        if (hi <= lo)
                            continue;
                        const int nseg = std::max(4, int(8 * std::log(hi / lo)));
                        composite_gl(lo, hi, nseg, 8, true, px, pw);
                        for (size_t i = 0; i < px.size(); i++) {
                            const double p = px[i];
                            const double Pp = P(p);
                            if (Pp == 0.0)
                                continue;
                            const double a = std::max(std::abs(k - p), 1e-12 * k), c = std::min(k + p, kmax);
                            if (c <= a)
                                continue;
                            const int nseg2 = std::max(2, int(4 * std::log(c / a)));
                            composite_gl(a, c, std::min(nseg2, 40), 8, true, qx, qw);
                            const double r = p / k;
                            for (size_t j = 0; j < qx.size(); j++) {
                                const double pp = qx[j];
                                const double y = pp / k, y2 = y * y;
                                const double x = (k * k + p * p - pp * pp) / (2.0 * k * p);
                                const double omx2 = 1.0 - x * x;
                                const double mu = (x - r) / y;
                                const double mu2 = mu * mu;
                                const double fac = pw[i] * qw[j] * (pp / p) * Pp * P(pp);
                                res.Q1 += fac * r * r * omx2 * omx2 / (y2 * y2);
                                res.Q2 += fac * omx2 * r * x * (1.0 - r * x) / (y2 * y2);
                                res.Q5 += fac * r * x * omx2 / y2;
                                res.Q8 += fac * r * r * omx2 / y2;
                                res.Qs2 += fac * r * r * (-3.0 * mu2 * mu2 + 4.0 * mu2 - 1.0);
                                // A10 with p the leg of the operator and |k - p| that of Psi^(1): equals Q5 for F = 1
                                res.Q5F += fac * F(p) * r * r * (1.0 - mu2) * (1.0 - r * x) / y2;
                            }
                        }
                    }
                    const double norm = k / (4.0 * M_PI * M_PI);
                    res.Q1 *= norm;
                    res.Q2 *= norm;
                    res.Q5 *= norm;
                    res.Q8 *= norm;
                    res.Qs2 *= norm;
                    res.Q5F *= norm;

                    // R_n = k^2 / (4 pi^2) P(k) int dp P(p) R~_n(p / k) (log singularity at p = k: refine towards it)
                    const double Pk = P(k);
                    if (Pk != 0.0) {
                        double s1 = 0.0, s2 = 0.0, s1F = 0.0, s2F = 0.0;
                        auto add = [&](double lo, double hi, int nseg) {
                            composite_gl(lo, hi, nseg, 8, true, px, pw);
                            for (size_t i = 0; i < px.size(); i++) {
                                double R1, R2;
                                R_kernels(px[i] / k, R1, R2);
                                const double Pp = P(px[i]);
                                const double Fp = F(px[i]);
                                s1 += pw[i] * Pp * R1;
                                s2 += pw[i] * Pp * R2;
                                s1F += pw[i] * Pp * Fp * R1;
                                s2F += pw[i] * Pp * Fp * R2;
                            }
                        };
                        auto add_refined = [&](double lo, double hi, bool towards_hi) {
                            // geometric refinement towards the singular end (segments halving in size, the last one
                            // reaching the end)
                            const double len = hi - lo;
                            double frac = 1.0;
                            for (int level = 0; level < 60; level++) {
                                const bool last = (0.5 * frac * len <= 1e-14 * k) or level == 59;
                                const double f0 = last ? 0.0 : 0.5 * frac;
                                if (towards_hi)
                                    add(hi - frac * len, hi - f0 * len, 2);
                                else
                                    add(lo + f0 * len, lo + frac * len, 2);
                                if (last)
                                    break;
                                frac = f0;
                            }
                        };
                        if (k > kmin) {
                            const double hi = std::min(k, kmax);
                            if (k < kmax) {
                                add(kmin,
                                    0.5 * k > kmin ? 0.5 * k : kmin,
                                    std::max(4, int(8 * std::log(0.5 * k / kmin))));
                                if (0.5 * k > kmin)
                                    add_refined(0.5 * k, hi, true);
                            } else {
                                add(kmin, kmax, std::max(4, int(8 * std::log(kmax / kmin))));
                            }
                        }
                        if (k < kmax) {
                            const double hi = std::min(2.0 * k, kmax);
                            add_refined(std::max(k, kmin), hi, false);
                            if (kmax > 2.0 * k)
                                add(2.0 * k, kmax, std::max(4, int(8 * std::log(kmax / (2.0 * k)))));
                        }
                        res.R1 = k * k / (4.0 * M_PI * M_PI) * Pk * s1;
                        res.R2 = k * k / (4.0 * M_PI * M_PI) * Pk * s2;
                        res.R1F = k * k / (4.0 * M_PI * M_PI) * Pk * s1F;
                        res.R2F = k * k / (4.0 * M_PI * M_PI) * Pk * s2F;
                    }
                    return res;
                }
                /// The spectra G(k) the q-functions are generalized correlation functions of (velocileptors
                /// notation; GVS = 2 GV1 + GV3; G22A is the delta_ij part of A22, zero in GR). The ones ending in N are
                /// for the nabla^2 delta spectra: A10 = <delta_1 Delta_i Delta_j> and U3 = <delta_1 Psi^(3)> with the
                /// operator F delta (P_1n), U11 = <a_1 b_2 Delta_i> for (a,b) = (delta, F delta) and (F delta, F delta)
                enum LoopSpec {
                    LIN,
                    G13,
                    G22,
                    GT,
                    GVS,
                    G10A,
                    G10B,
                    GU3,
                    GU11,
                    GU20,
                    GUS2,
                    GU3N,
                    G10AN,
                    G10BN,
                    GU11DN,
                    GU11NN,
                    G22A,
                    NSPEC
                };
                /// Fills G[s] (s != LIN) at k and the integrand Z(k) of zerolag_10 = int dk Z(k) / (2 pi^2)
                using LoopFill = std::function<void(double k, double * G, double & zerolag_integrand)>;

                inline std::vector<std::vector<double>>
                oneloop_from_loop_functions(const std::function<double(double)> & pofk,
                                            const std::function<double(double)> & W,
                                            const std::vector<double> & kvalues,
                                            const ZeldovichOptions & options,
                                            const LoopFill & fill,
                                            std::vector<std::string> * names,
                                            std::vector<std::array<int, 2>> * pairs);
            } // namespace detail

            //=====================================================================================================
            /// @brief The one-loop (CLEFT) prediction for the hybrid bias spectra
            ///
            /// @param[in] pofk The linear power spectrum P(k)
            /// @param[in] W The filter of the linear field (operators and displacement)
            /// @param[in] kvalues The k-values to compute the spectra at
            /// @param[in] options The options (kmax_integration must be set; W must be zero above it)
            /// @param[out] names, pairs As for zeldovich_hybrid_bias_spectra
            ///
            /// @return result[pair][ik]
            //=====================================================================================================
            inline std::vector<std::vector<double>>
            oneloop_hybrid_bias_spectra(const std::function<double(double)> & pofk,
                                        const std::function<double(double)> & W,
                                        const std::vector<double> & kvalues,
                                        const ZeldovichOptions & options,
                                        std::vector<std::string> * names = nullptr,
                                        std::vector<std::array<int, 2>> * pairs = nullptr) {
                using namespace detail;
                if (options.kmax_integration <= options.kmin_integration)
                    throw std::runtime_error("[oneloop_hybrid_bias_spectra] Set kmax_integration (the cutoff)");
                const int nop = options.noperators;
                if (nop < 1 or nop > 5)
                    throw std::runtime_error("[oneloop_hybrid_bias_spectra] noperators must be 1, ..., 5");
                if (kvalues.empty())
                    return {};

                // The operator nabla^2 delta_L is F(k) delta_L(k) with F = -k^2
                const std::function<double(double)> Fop = [](double k) { return -k * k; };

                const double kmin_int = options.kmin_integration;
                const double kcut = options.kmax_integration;
                auto PW = [&](double k) {
                    if (k < kmin_int or k > kcut)
                        return 0.0;
                    const double w = W(k);
                    return pofk(k) * w * w;
                };

                // The loop spectra on a grid in k up to 2 kcut: log spaced below the cut and above it (the R-type
                // spectra contain P(k) and stop at the cut, so we do not interpolate across it)
                const int nloop1 = 500, nloop = 600;
                const double kl_min = kmin_int, kl_max = 2.0 * kcut;
                std::vector<double> kloop(nloop);
                std::vector<LoopSpectra> loops(nloop);
                for (int i = 0; i < nloop1; i++)
                    kloop[i] = kl_min * std::pow(kcut * (1.0 - 1e-9) / kl_min, i / double(nloop1 - 1));
                for (int i = nloop1; i < nloop; i++)
                    kloop[i] = kcut * (1.0 + 1e-9) *
                               std::pow(kl_max / (kcut * (1.0 + 1e-9)), (i - nloop1) / double(nloop - nloop1 - 1));
#ifdef USE_OMP
#pragma omp parallel for schedule(dynamic, 4)
#endif
                for (int i = 0; i < nloop; i++)
                    loops[i] = loop_spectra(kloop[i], PW, kmin_int, kcut, Fop);
                auto loop_at = [&](double k) {
                    LoopSpectra res{0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
                    if (k <= kl_min or k >= kl_max)
                        return res;
                    const int i =
                        std::min(nloop - 2, int(std::upper_bound(kloop.begin(), kloop.end(), k) - kloop.begin()) - 1);
                    if (i == nloop1 - 1)
                        return res; // between the two nodes at the cut
                    const double f = std::log(k / kloop[i]) / std::log(kloop[i + 1] / kloop[i]);
                    auto interp = [&](double LoopSpectra::* m) { return loops[i].*m * (1 - f) + loops[i + 1].*m * f; };
                    for (auto m : {&LoopSpectra::Q1,
                                   &LoopSpectra::Q2,
                                   &LoopSpectra::Q5,
                                   &LoopSpectra::Q8,
                                   &LoopSpectra::Qs2,
                                   &LoopSpectra::R1,
                                   &LoopSpectra::R2,
                                   &LoopSpectra::Q5F,
                                   &LoopSpectra::R1F,
                                   &LoopSpectra::R2F})
                        res.*m = interp(m);
                    return res;
                };

                // The spectra the q-functions are made from
                detail::LoopFill fill = [&](double k, double * G, double & zerolag_integrand) {
                    const auto L = loop_at(k);
                    G[G13] = 5.0 / 21.0 * L.R1;
                    G[G22] = 9.0 / 98.0 * L.Q1;
                    G[GT] = -3.0 / 7.0 * (2 * L.R1 + 4 * L.R2 + L.Q1 + 2 * L.Q2);
                    // 2 GV1 + GV3 with GV1 = 3/35 (-3 R1 + 4 R2 + Q1 + 2 Q2), GV3 = 3/35 (2 R1 + 4 R2 - 4 Q1 + 2 Q2)
                    G[GVS] = 3.0 / 35.0 * (-4 * L.R1 + 12 * L.R2 - 2 * L.Q1 + 6 * L.Q2);
                    G[G10A] = (4 * L.R2 + 2 * L.Q5) / 14.0;
                    G[G10B] = (3 * L.R1 + 4 * L.R2 + 2 * L.Q5) / 14.0;
                    G[GU3] = -5.0 / 21.0 * L.R1;
                    G[GU11] = -6.0 / 7.0 * (L.R1 + L.R2);
                    G[GU20] = -3.0 / 7.0 * L.Q8;
                    G[GUS2] = -1.0 / 7.0 * L.Qs2;
                    const double Fk = Fop(k);
                    // A10 = int d^3k e^(ikq) (a delta_ij + b k_i k_j / k^2) (in units of velocileptors' A10, which
                    // is half of <delta_1 Delta_i Delta_j>); for F = 1 these are G10A and G10B
                    const double k2a = 3.0 / 7.0 * Fk * L.R1;
                    const double k2b =
                        -3.0 / 7.0 * Fk * (L.R1 + 2.0 * L.R2) - 6.0 / 7.0 * L.Q5F - 6.0 / 7.0 * (L.R1F + L.R2F);
                    G[G10AN] = -0.5 * (k2a + k2b / 3.0);
                    G[G10BN] = -k2b / 6.0;
                    G[GU3N] = -5.0 / 21.0 * Fk * L.R1;
                    G[GU11DN] = -3.0 / 7.0 * ((L.R1F + L.R2F) + Fk * (L.R1 + L.R2));
                    G[GU11NN] = -6.0 / 7.0 * Fk * (L.R1F + L.R2F);
                    // velocileptors: zerolag_10 = int dk (R1 - R2) / 7 / (2 pi^2)
                    zerolag_integrand = (L.R1 - L.R2) / 7.0;
                };
                return detail::oneloop_from_loop_functions(pofk, W, kvalues, options, fill, names, pairs);
            }

            //=====================================================================================================
            // The q-functions and the mu/q integration (shared by the GR and the modified gravity predictions)
            //=====================================================================================================
            inline std::vector<std::vector<double>>
            detail::oneloop_from_loop_functions(const std::function<double(double)> & pofk,
                                                const std::function<double(double)> & W,
                                                const std::vector<double> & kvalues,
                                                const ZeldovichOptions & options,
                                                const LoopFill & fill,
                                                std::vector<std::string> * names,
                                                std::vector<std::array<int, 2>> * pairs) {
                using namespace detail;
                const int nop = options.noperators;
                if (kvalues.empty())
                    return {};

                // The Zel'dovich part of the spectra with nabla^2 delta (the loop corrections are added below)
                std::vector<std::vector<double>> zeldovich;
                if (nop == 5)
                    zeldovich = zeldovich_hybrid_bias_spectra(pofk, W, W, kvalues, options);

                const double kmin_int = options.kmin_integration;
                const double kcut = options.kmax_integration;
                auto PW = [&](double k) {
                    if (k < kmin_int or k > kcut)
                        return 0.0;
                    const double w = W(k);
                    return pofk(k) * w * w;
                };
                double kout_min = 1e100, kout_max = 0.0;
                for (auto k : kvalues) {
                    if (k > 0.0)
                        kout_min = std::min(kout_min, k);
                    kout_max = std::max(kout_max, k);
                }

                // The q-grid (as for the Zel'dovich prediction; the loop terms extend to 2 kcut)
                double dq = options.dq;
                if (dq <= 0.0)
                    dq = std::min(M_PI / (4.0 * kout_max), M_PI / (8.0 * kcut));
                double qmax = options.qmax;
                if (qmax <= 0.0)
                    qmax = std::max(16.0 * M_PI / kout_min, 40.0 * 2.0 * M_PI / kcut);
                const int nq = 2 * int(std::ceil(qmax / dq / 2.0));
                dq = qmax / nq;

                // The k-grid for the q-functions (to 2 kcut)
                int nk = options.nk_integration;
                const double kint_max = 2.0 * kcut;
                const double k1 = std::min(0.01 * kcut, 2.0 * M_PI / qmax);
                if (nk <= 0)
                    nk = std::max(4000, int(16.0 * (kint_max - k1) * qmax / (2.0 * M_PI)));
                std::vector<double> kint, wint;
                {
                    const int nlog = 400;
                    const double lk0 = std::log(kmin_int), lk1 = std::log(k1), dl = (lk1 - lk0) / nlog;
                    for (int i = 0; i < nlog; i++) {
                        kint.push_back(std::exp(lk0 + dl * i));
                        wint.push_back(std::exp(lk0 + dl * i) * dl * (i == 0 ? 0.5 : 1.0));
                    }
                    const double dk = (kint_max - k1) / nk;
                    for (int i = 0; i <= nk; i++) {
                        kint.push_back(k1 + dk * i);
                        double w = dk;
                        if (i == 0)
                            w = 0.5 * dk + 0.5 * k1 * dl;
                        if (i == nk)
                            w = 0.5 * dk;
                        wint.push_back(w);
                    }
                }
                const size_t nkint = kint.size();

                // The spectra G(k) the q-functions are generalized correlation functions of
                std::vector<std::vector<double>> G(NSPEC, std::vector<double>(nkint, 0.0));
                std::vector<double> zl(nkint, 0.0);
#ifdef USE_OMP
#pragma omp parallel for schedule(static)
#endif
                for (size_t i = 0; i < nkint; i++) {
                    double g[NSPEC] = {};
                    fill(kint[i], g, zl[i]);
                    for (int sp = 0; sp < NSPEC; sp++)
                        G[sp][i] = g[sp];
                    G[LIN][i] = PW(kint[i]);
                }
                // zerolag_10 (and for the nabla^2 version A10(q = 0) = 0 gives zerolag = xi_{0,-2}[G10AN](0))
                double zerolag10 = 0.0, zerolag10N = 0.0;
                for (size_t i = 0; i < nkint; i++) {
                    zerolag10 += wint[i] * zl[i] / (2.0 * M_PI * M_PI);
                    zerolag10N += wint[i] * G[G10AN][i] / (2.0 * M_PI * M_PI);
                }

                // The generalized correlation functions xi_{l,n}[G](q) = int dk k^(2+n) G(k) j_l(kq) / (2 pi^2)
                struct XiSpec {
                    int spec, l, n;
                };
                const std::vector<XiSpec> xis{
                    {LIN, 0, -2},   {LIN, 2, -2},   {LIN, 1, -1},    {LIN, 0, 0},     {LIN, 2, 0},
                    {LIN, 4, 0},    {LIN, 3, -1},   {G13, 0, -2},    {G13, 2, -2},    {G22, 0, -2},
                    {G22, 2, -2},   {GT, 3, -3},    {GVS, 1, -3},    {G10A, 0, -2},   {G10B, 2, -2},
                    {GU3, 1, -1},   {GU11, 1, -1},  {GU20, 1, -1},   {GUS2, 1, -1},   {GU3N, 1, -1},
                    {G10AN, 0, -2}, {G10BN, 2, -2}, {GU11DN, 1, -1}, {GU11NN, 1, -1}, {G22A, 0, -2}};
                const int nxi = int(xis.size());
                // The integration weights times k^(2+n) G(k)
                std::vector<std::vector<double>> wG(nkint, std::vector<double>(nxi, 0.0));
                for (size_t i = 0; i < nkint; i++)
                    for (int s = 0; s < nxi; s++)
                        wG[i][s] = wint[i] * std::pow(kint[i], 2 + xis[s].n) * G[xis[s].spec][i];
                std::vector<std::vector<double>> xi(nxi, std::vector<double>(nq + 1, 0.0));
#ifdef USE_OMP
#pragma omp parallel for schedule(dynamic, 16)
#endif
                for (int iq = 0; iq <= nq; iq++) {
                    const double q = iq * dq;
                    std::vector<double> sums(nxi, 0.0);
                    double jl[5];
                    for (size_t i = 0; i < nkint; i++) {
                        spherical_bessel_0_to_4(kint[i] * q, jl);
                        const auto & wg = wG[i];
                        for (int s = 0; s < nxi; s++)
                            sums[s] += wg[s] * jl[xis[s].l];
                    }
                    for (int s = 0; s < nxi; s++)
                        xi[s][iq] = sums[s] / (2.0 * M_PI * M_PI);
                }
                enum XiIdx {
                    L0M2,
                    L2M2,
                    L1M1,
                    L00,
                    L20,
                    L40,
                    L3M1,
                    A13_0,
                    A13_2,
                    A22_0,
                    A22_2,
                    T3M3,
                    VSM3,
                    X10A,
                    X10B,
                    U3_,
                    U11_,
                    U20_,
                    US2_,
                    U3N_,
                    X10AN,
                    X10BN,
                    U11DN_,
                    U11NN_,
                    A22A
                };

                // The q-functions (velocileptors notation)
                struct QF {
                    double X, Y, U, xi, V, Xs2, Ys2, zeta, chi;
                    double Xloop, Yloop, Vloop, Tloop, X10, Y10, U3, U11, U20, Us2;
                    double X10N, Y10N, U3N, U11DN, U11NN;
                };
                std::vector<QF> qf(nq + 1);
                for (int iq = 0; iq <= nq; iq++) {
                    QF f;
                    auto x = [&](int s) { return xi[s][iq]; };
                    f.X = 2.0 / 3.0 * (xi[L0M2][0] - x(L0M2) - x(L2M2));
                    f.Y = 2.0 * x(L2M2);
                    f.U = -x(L1M1);
                    f.xi = x(L00);
                    const double J2 = 2.0 * x(L1M1) / 15.0 - 0.2 * x(L3M1);
                    const double J3 = -0.2 * x(L1M1) - 0.2 * x(L3M1);
                    const double J4 = x(L3M1);
                    f.V = 4.0 * J2 * x(L20);
                    f.Xs2 = 4.0 * J3 * J3;
                    f.Ys2 = 6 * J2 * J2 + 8 * J2 * J3 + 4 * J2 * J4 + 4 * J3 * J3 + 8 * J3 * J4 + 2 * J4 * J4;
                    f.zeta = 2.0 * (4.0 * x(L00) * x(L00) / 45.0 + 8.0 * x(L20) * x(L20) / 63.0 +
                                    8.0 * x(L40) * x(L40) / 35.0);
                    f.chi = 4.0 * x(L20) * x(L20) / 3.0;
                    const double Xloop13 = 2.0 / 3.0 * (xi[A13_0][0] - x(A13_0) - x(A13_2));
                    const double Yloop13 = 2.0 * x(A13_2);
                    // A22 = a delta_ij + b khat khat: G22 = k^2 b, G22A = k^2 a
                    const double Xloop22 =
                        2.0 / 3.0 * (xi[A22_0][0] - x(A22_0) - x(A22_2)) + 2.0 * (xi[A22A][0] - x(A22A));
                    const double Yloop22 = 2.0 * x(A22_2);
                    f.Xloop = 2.0 * Xloop13 + Xloop22;
                    f.Yloop = 2.0 * Yloop13 + Yloop22;
                    const double T112 = x(T3M3);
                    // 3 (2 V1 + V3) with V1 = xi_{1,-3}[GV1] - T/5, V3 = xi_{1,-3}[GV3] - T/5
                    f.Vloop = 3.0 * (x(VSM3) - 0.6 * T112);
                    f.Tloop = 3.0 * T112;
                    f.X10 = 2.0 * (zerolag10 - x(X10A) - x(X10B));
                    f.Y10 = 2.0 * 3.0 * x(X10B);
                    f.U3 = x(U3_);
                    f.U11 = x(U11_);
                    f.U20 = x(U20_);
                    f.Us2 = x(US2_);
                    f.X10N = 2.0 * (zerolag10N - x(X10AN) - x(X10BN));
                    f.Y10N = 2.0 * 3.0 * x(X10BN);
                    f.U3N = x(U3N_);
                    f.U11DN = x(U11DN_);
                    f.U11NN = x(U11NN_);
                    qf[iq] = f;
                }
                const double Xinf = 2.0 / 3.0 * xi[L0M2][0];
                const double sigmaloop = 2.0 * 2.0 / 3.0 * xi[A13_0][0] + 2.0 / 3.0 * xi[A22_0][0] + 2.0 * xi[A22A][0];
                const double X10inf = 2.0 * zerolag10;
                const double X10Ninf = 2.0 * zerolag10N;

                // The mu-integration rules
                std::map<int, std::pair<std::vector<double>, std::vector<double>>> rules;
                auto nodes_needed = [&](double kq) { return 32 * (1 + int((0.75 * kq + 32) / 32)); };
                for (auto k : kvalues)
                    for (int n = nodes_needed(0.0); n <= nodes_needed(k * qmax); n += 32)
                        if (rules.find(n) == rules.end()) {
                            std::vector<double> x, w;
                            gauss_legendre(n, x, w);
                            rules[n] = {x, w};
                        }

                // The CLEFT components (velocileptors order): 1, b1, b1^2, b2, b1b2, b2^2, bs, b1bs, b2bs, bs^2,
                // then the one-loop corrections to P_1n, P_dn, P_nn (n = nabla^2 delta) beyond Zel'dovich
                constexpr int NC = 13;
                std::vector<std::array<double, NC>> comp(kvalues.size());
#ifdef USE_OMP
#pragma omp parallel for schedule(dynamic, 1)
#endif
                for (int ik = 0; ik < int(kvalues.size()); ik++) {
                    const double k = kvalues[ik];
                    std::array<double, NC> sum{};
                    if (k <= 0.0) {
                        comp[ik] = sum;
                        continue;
                    }
                    const double k2 = k * k, k3 = k2 * k;
                    const double expinf = std::exp(-0.5 * k2 * Xinf);
                    // The large-q limits of the components (their Fourier transform is a delta function, but
                    // times the linear part of the exponent it is not)
                    std::array<double, NC> Finf{};
                    Finf[0] = 1.0 - 0.5 * k2 * sigmaloop;
                    Finf[1] = -k2 * X10inf;
                    Finf[10] = -0.5 * k2 * X10Ninf;
                    std::array<Complex, NC> F;
                    for (int iq = 1; iq <= nq; iq++) {
                        const double q = iq * dq;
                        const auto & f = qf[iq];
                        const auto & rule = rules.at(nodes_needed(k * q));
                        const auto & mu = rule.first;
                        const auto & w = rule.second;
                        std::array<Complex, NC> musum{};
                        for (size_t im = 0; im < mu.size(); im++) {
                            const double m = mu[im], m2 = m * m, m3 = m2 * m;
                            const double eps = -0.5 * k2 * (f.X - Xinf + f.Y * m2);
                            const double E = expinf * std::exp(eps);
                            const Complex I(0.0, 1.0);
                            F[0] =
                                1.0 - 0.5 * k2 * (f.Xloop + m2 * f.Yloop) - I * k3 / 6.0 * (f.Vloop * m + f.Tloop * m3);
                            F[1] = 2.0 * I * k * (f.U + f.U3) * m - k2 * (f.X10 + f.Y10 * m2);
                            F[2] = f.xi - k2 * m2 * f.U * f.U + I * k * f.U11 * m;
                            F[3] = -k2 * m2 * f.U * f.U + I * k * f.U20 * m;
                            F[4] = 2.0 * I * k * f.U * f.xi * m;
                            F[5] = 0.5 * f.xi * f.xi;
                            F[6] = -k2 * (f.Xs2 + m2 * f.Ys2) + 2.0 * I * k * f.Us2 * m;
                            F[7] = 2.0 * I * k * f.V * m;
                            F[8] = f.chi;
                            F[9] = f.zeta;
                            F[10] = I * k * f.U3N * m - 0.5 * k2 * (f.X10N + f.Y10N * m2);
                            F[11] = I * k * f.U11DN * m;
                            F[12] = I * k * f.U11NN * m;
                            // The linear (in the linear correlators) parts we transform analytically
                            std::array<Complex, NC> lin{};
                            lin[1] = 2.0 * I * k * f.U * m;
                            lin[2] = f.xi;
                            const Complex phase = std::polar(w[im] * 0.5, k * q * m);
                            for (int c = 0; c < NC; c++)
                                musum[c] += phase * (E * F[c] - expinf * (Finf[c] * (1.0 + eps) + lin[c]));
                        }
                        const double ws = (iq == nq) ? 1.0 : (iq % 2 == 1 ? 4.0 : 2.0);
                        for (int c = 0; c < NC; c++)
                            sum[c] += ws * q * q * musum[c].real();
                    }
                    for (int c = 0; c < NC; c++)
                        sum[c] *= 4.0 * M_PI * dq / 3.0;
                    // The analytic part: FT[eps] = P(k), FT[2 i k U mu] = 2 P(k), FT[xi] = P(k)
                    const double P = PW(k);
                    for (int c = 0; c < NC; c++)
                        sum[c] += expinf * Finf[c] * P;
                    sum[1] += expinf * 2.0 * P;
                    sum[2] += expinf * P;
                    comp[ik] = sum;
                }

                // To the basis spectra
                std::vector<std::array<int, 2>> pairlist;
                for (int a = 0; a < nop; a++)
                    for (int b = a; b < nop; b++)
                        pairlist.push_back({a, b});
                if (names) {
                    const auto & all = operator_names();
                    *names = std::vector<std::string>(all.begin(), all.begin() + nop);
                }
                if (pairs)
                    *pairs = pairlist;
                std::vector<std::vector<double>> result(pairlist.size(), std::vector<double>(kvalues.size(), 0.0));
                for (size_t p = 0; p < pairlist.size(); p++) {
                    const int a = pairlist[p][0], b = pairlist[p][1];
                    for (size_t ik = 0; ik < kvalues.size(); ik++) {
                        const auto & c = comp[ik];
                        double v = 0.0;
                        if (a == ONE and b == NABLA2DELTA)
                            v = zeldovich[p][ik] + c[10];
                        else if (a == DELTA and b == NABLA2DELTA)
                            v = zeldovich[p][ik] + c[11];
                        else if (a == NABLA2DELTA and b == NABLA2DELTA)
                            v = zeldovich[p][ik] + c[12];
                        else if (a == NABLA2DELTA or b == NABLA2DELTA)
                            v = zeldovich[p][ik];
                        else if (a == ONE and b == ONE)
                            v = c[0];
                        else if (a == ONE and b == DELTA)
                            v = 0.5 * c[1];
                        else if (a == ONE and b == DELTA2)
                            v = c[3];
                        else if (a == ONE and b == S2)
                            v = 0.5 * c[6];
                        else if (a == DELTA and b == DELTA)
                            v = c[2];
                        else if (a == DELTA and b == DELTA2)
                            v = c[4];
                        else if (a == DELTA and b == S2)
                            v = 0.5 * c[7];
                        else if (a == DELTA2 and b == DELTA2)
                            v = 4.0 * c[5];
                        else if (a == DELTA2 and b == S2)
                            v = c[8];
                        else if (a == S2 and b == S2)
                            v = c[9];
                        result[p][ik] = v;
                    }
                }
                if (options.verbose)
                    std::cout << "[oneloop_hybrid_bias_spectra] nq = " << nq << " dq = " << dq << " qmax = " << qmax
                              << " nk = " << nkint << " sigma^2 = " << 0.5 * Xinf << " sigma_loop = " << sigmaloop
                              << "\n";
                return result;
            }

        } // namespace HYBRIDBIAS
    } // namespace CORRELATIONFUNCTIONS
} // namespace FML
#endif
