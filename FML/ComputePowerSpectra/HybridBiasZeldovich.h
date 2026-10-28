#ifndef FML_HYBRIDBIASZELDOVICH_HEADER
#define FML_HYBRIDBIASZELDOVICH_HEADER

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <iostream>
#include <functional>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef USE_OMP
#include <omp.h>
#endif

#include <FML/ComputePowerSpectra/HybridBiasOperators.h>

namespace FML {
    namespace CORRELATIONFUNCTIONS {
        namespace HYBRIDBIAS {

            //=====================================================================================================
            ///
            /// The Zel'dovich (resummed 1LPT) prediction for the hybrid bias basis spectra P_ab(k) of the operators
            /// 1, delta_L, delta_L^2 - <delta_L^2>, s^2 - <s^2>, nabla^2 delta_L (see HybridBiasSpectra.h).
            ///
            /// With the Zel'dovich displacement Psi(k) = i k / k^2 delta_L(k) everything is Gaussian and
            ///
            ///   P_ab(k) = int d^3q e^{ik.q} < O_a(q1) O_b(q2) e^{ik.Delta} >,  Delta = Psi(q1) - Psi(q2), q = q1 - q2
            ///           = int d^3q e^{ik.q} e^{-k_i k_j A_ij(q) / 2} F_ab(k, q)
            ///
            /// where A_ij = <Delta_i Delta_j> and F_ab is a polynomial in the correlators of the fields and the
            /// displacement. Shifting the Gaussian fields by their covariance with Delta, O -> O + i k.<O Delta>,
            /// gives (D = i k.U, U = <delta(q1) Delta>, S_ij = i k_l <s_ij(q1) Delta_l>, D_L = i k.<nabla^2 delta(q1)
            /// Delta>, xi = <delta delta>, C_ij = <delta s_ij>, T_ijkl = <s_ij s_kl>, xi_{dL} = <delta nabla^2 delta>,
            /// C^L_ij = <s_ij nabla^2 delta>, xi_LL = <nabla^2 delta nabla^2 delta>):
            ///
            ///   (1,1) = 1            (1,d) = D             (1,d2) = D^2              (1,s2) = S:S     (1,L) = D_L
            ///   (d,d) = xi + D^2     (d,d2) = 2 D xi + D^3 (d,s2) = 2 C:S + D S:S    (d,L) = xi_dL + D D_L
            ///   (d2,d2) = 2 xi^2 + 4 D^2 xi + D^4          (d2,s2) = 2 C:C + 4 D C:S + D^2 S:S
            ///   (d2,L) = 2 D xi_dL + D^2 D_L               (s2,s2) = 2 T:T + 4 S.T.S + (S:S)^2
            ///   (s2,L) = 2 C^L:S + S:S D_L                 (L,L) = xi_LL + D_L^2
            ///
            /// All the correlators are tensors built from q-hat with radial functions int dk k^n P(k) j_l(kq); the
            /// angular structure is found once by projecting the products of the kernels (1, k_i k_j / k^2 -
            /// delta_ij/3, i k_i / k, -k^2) onto spherical harmonics, so no tensor algebra is done by hand. The part
            /// linear in the correlators is Fourier transformed analytically (it gives exp(-k^2 sigma^2) times P_L,
            /// P_L, -k^2 P_L,
            /// ... for (1,1), (1,d), (1,L), ...) and the rest, which falls off quickly with q, is integrated
            /// numerically over q and mu = k.q / kq.
            ///
            /// The operators can be filtered by W_op(k) (e.g. a sharp cut at the Nyquist frequency of the particle
            /// lattice, which is what the simulation measurement uses) independently of the displacement field
            /// (W_disp(k)). Units: whatever P(k) is given in (e.g. k in h/Mpc and P in (Mpc/h)^3; q is then in Mpc/h).
            /// nabla^2 delta is in units of 1/k^2 (as in HybridBiasSpectra::scale).
            ///
            /// The displacement variance sigma^2 = int dk P W_disp^2 / (6 pi^2) must be finite (it is for any physical
            /// P(k), which goes as k^ns at small k; for e.g. a power-law with a steep slope set kmin_integration to the
            /// fundamental mode of the box you compare with).
            ///
            //=====================================================================================================

            struct ZeldovichOptions {
                /// The range of the k-integrals over P(k) (kmax: where W_op and W_disp are zero, e.g. the cutoff)
                double kmin_integration{1e-5};
                double kmax_integration{0.0};
                /// Number of points in the k-integrals (0: automatic)
                int nk_integration{0};
                /// The largest q and the spacing of the q-integral (0: automatic from the output k values)
                double qmax{0.0};
                double dq{0.0};
                /// How many of the operators (1, delta, delta2, s2, nabla2delta) to use
                int noperators{5};
                bool verbose{false};
            };

            namespace detail {

                /// Spherical Bessel functions j_0, ..., j_4 (series for small x, closed forms otherwise)
                inline void spherical_bessel_0_to_4(double x, double * j) {
                    if (x < 5.0) {
                        const double x2 = x * x;
                        double xl = 1.0, dfact = 1.0;
                        for (int l = 0; l <= 4; l++) {
                            if (l > 0) {
                                xl *= x;
                                dfact *= (2 * l + 1);
                            }
                            double term = 1.0, sum = 1.0;
                            for (int n = 1; n < 30; n++) {
                                term *= -0.5 * x2 / (n * (2.0 * l + 2.0 * n + 1.0));
                                sum += term;
                            }
                            j[l] = xl / dfact * sum;
                        }
                        return;
                    }
                    const double s = std::sin(x), c = std::cos(x);
                    j[0] = s / x;
                    j[1] = (s / x - c) / x;
                    for (int l = 1; l < 4; l++)
                        j[l + 1] = (2 * l + 1) / x * j[l] - j[l - 1];
                }

                /// Gauss-Legendre nodes and weights on [-1,1]
                inline void gauss_legendre(int n, std::vector<double> & x, std::vector<double> & w) {
                    x.resize(n);
                    w.resize(n);
                    for (int i = 0; i < (n + 1) / 2; i++) {
                        double z = std::cos(M_PI * (i + 0.75) / (n + 0.5)), pp = 0.0;
                        for (int it = 0; it < 100; it++) {
                            double p1 = 1.0, p2 = 0.0;
                            for (int j = 0; j < n; j++) {
                                const double p3 = p2;
                                p2 = p1;
                                p1 = ((2.0 * j + 1.0) * z * p2 - j * p3) / (j + 1.0);
                            }
                            pp = n * (z * p1 - p2) / (z * z - 1.0);
                            const double z1 = z;
                            z = z1 - p1 / pp;
                            if (std::abs(z - z1) < 1e-15)
                                break;
                        }
                        x[i] = -z;
                        x[n - 1 - i] = z;
                        w[i] = w[n - 1 - i] = 2.0 / ((1.0 - z * z) * pp * pp);
                    }
                }

                inline double legendre(int l, double x) {
                    double p0 = 1.0, p1 = x;
                    if (l == 0)
                        return p0;
                    for (int j = 1; j < l; j++) {
                        const double p2 = ((2.0 * j + 1.0) * x * p1 - j * p0) / (j + 1.0);
                        p0 = p1;
                        p1 = p2;
                    }
                    return p1;
                }

                using Complex = std::complex<double>;

                /// A field: a radial factor k^power (times sign) and an angular kernel K(k-hat) (a tensor of rank 0,
                /// 1 or 2 stored with 3^rank components)
                struct FieldKernel {
                    int rank;
                    int power;
                    std::function<Complex(const double * khat, int component)> K;
                };

                inline FieldKernel kernel_delta() {
                    return {0, 0, [](const double *, int) { return Complex(1.0, 0.0); }};
                }
                inline FieldKernel kernel_nabla2() {
                    return {0, 2, [](const double *, int) { return Complex(-1.0, 0.0); }};
                }
                inline FieldKernel kernel_shear() {
                    return {2, 0, [](const double * n, int c) {
                                const int i = c / 3, j = c % 3;
                                return Complex(n[i] * n[j] - (i == j ? 1.0 / 3.0 : 0.0), 0.0);
                            }};
                }
                inline FieldKernel kernel_displacement() {
                    return {1, -1, [](const double * n, int c) { return Complex(0.0, n[c]); }};
                }

                /// The harmonic decomposition of K_a(n) K_b(n)^* at n = z-hat: for l = 0..lmax the tensor
                /// (2l+1) < P_l(n.z) K_a K_b^* >_sphere (components a-index major). The correlation is then
                /// <a(q1) b(q2)> = sum_l i^l R_l(q) proj[l] with R_l = int dk k^2 / (2 pi^2) k^(pa+pb) P W_a W_b
                /// j_l(kq)
                inline std::vector<std::vector<Complex>>
                harmonic_projection(const FieldKernel & a, const FieldKernel & b, int lmax) {
                    const int na = int(std::pow(3, a.rank)), nb = int(std::pow(3, b.rank));
                    std::vector<std::vector<Complex>> proj(lmax + 1, std::vector<Complex>(na * nb, 0.0));
                    std::vector<double> mu, w;
                    gauss_legendre(16, mu, w);
                    const int nphi = 32;
                    for (size_t im = 0; im < mu.size(); im++) {
                        const double st = std::sqrt(1.0 - mu[im] * mu[im]);
                        for (int ip = 0; ip < nphi; ip++) {
                            const double phi = 2.0 * M_PI * (ip + 0.5) / nphi;
                            const double n[3] = {st * std::cos(phi), st * std::sin(phi), mu[im]};
                            const double weight = w[im] / 2.0 / nphi;
                            for (int ca = 0; ca < na; ca++) {
                                const Complex Ka = a.K(n, ca);
                                for (int cb = 0; cb < nb; cb++) {
                                    const Complex v = Ka * std::conj(b.K(n, cb)) * weight;
                                    for (int l = 0; l <= lmax; l++)
                                        proj[l][ca * nb + cb] += (2.0 * l + 1.0) * legendre(l, mu[im]) * v;
                                }
                            }
                        }
                    }
                    return proj;
                }
            } // namespace detail

            //=====================================================================================================
            /// @brief The Zel'dovich prediction for the hybrid bias spectra
            ///
            /// @param[in] pofk The linear power spectrum P(k)
            /// @param[in] W_operators The filter of the operators (e.g. 1 for k < kcut and 0 otherwise)
            /// @param[in] W_displacement The filter of the displacement field
            /// @param[in] kvalues The k-values to compute the spectra at
            /// @param[in] options The options (kmax_integration must be set)
            /// @param[out] names The names of the operators
            /// @param[out] pairs The pairs (a,b) in the same order as HybridBiasSpectra
            ///
            /// @return result[pair][ik] = P_ab(k[ik])
            //=====================================================================================================
            inline std::vector<std::vector<double>>
            zeldovich_hybrid_bias_spectra(const std::function<double(double)> & pofk,
                                          const std::function<double(double)> & W_operators,
                                          const std::function<double(double)> & W_displacement,
                                          const std::vector<double> & kvalues,
                                          const ZeldovichOptions & options,
                                          std::vector<std::string> * names = nullptr,
                                          std::vector<std::array<int, 2>> * pairs = nullptr) {
                using namespace detail;
                if (options.kmax_integration <= options.kmin_integration)
                    throw std::runtime_error("[zeldovich_hybrid_bias_spectra] Set kmax_integration (the cutoff)");
                if (options.noperators < 1 or options.noperators > 5)
                    throw std::runtime_error("[zeldovich_hybrid_bias_spectra] noperators must be 1, ..., 5");
                const int nop = options.noperators;
                if (kvalues.empty())
                    return {};
                double kout_min = 1e100, kout_max = 0.0;
                for (auto k : kvalues) {
                    if (k > 0.0)
                        kout_min = std::min(kout_min, k);
                    kout_max = std::max(kout_max, k);
                }
                const double kmax_int = options.kmax_integration;
                const double kmin_int = options.kmin_integration;

                // The q-grid: resolve the oscillations of e^{ikq mu} and the correlators, and go far enough out
                double dq = options.dq;
                if (dq <= 0.0)
                    dq = std::min(M_PI / (4.0 * kout_max), M_PI / (4.0 * kmax_int));
                double qmax = options.qmax;
                if (qmax <= 0.0)
                    qmax = std::max(16.0 * M_PI / kout_min, 40.0 * 2.0 * M_PI / kmax_int);
                const int nq = 2 * int(std::ceil(qmax / dq / 2.0));
                dq = qmax / nq;

                // The k-grid for the integrals over P(k): log spaced below k1 and linear above (to resolve j_l(kq))
                int nk = options.nk_integration;
                const double k1 = std::min(0.01 * kmax_int, 2.0 * M_PI / qmax);
                if (nk <= 0)
                    nk = std::max(4000, int(16.0 * (kmax_int - k1) * qmax / (2.0 * M_PI)));
                std::vector<double> kint, wint;
                {
                    const int nlog = 400;
                    for (int i = 0; i < nlog; i++) {
                        // Trapezoid in log k
                        const double lk0 = std::log(kmin_int), lk1 = std::log(k1);
                        const double lk = lk0 + (lk1 - lk0) * i / double(nlog);
                        const double dl = (lk1 - lk0) / nlog;
                        kint.push_back(std::exp(lk));
                        wint.push_back(std::exp(lk) * dl * (i == 0 ? 0.5 : 1.0));
                    }
                    const double dk = (kmax_int - k1) / nk;
                    for (int i = 0; i <= nk; i++) {
                        kint.push_back(k1 + dk * i);
                        double w = dk;
                        if (i == 0)
                            w = 0.5 * dk + 0.5 * k1 * (std::log(k1) - std::log(kmin_int)) / nlog;
                        if (i == nk)
                            w = 0.5 * dk;
                        wint.push_back(w);
                    }
                }
                const size_t nkint = kint.size();
                std::vector<double> P(nkint), Wo(nkint), Wd(nkint);
                for (size_t i = 0; i < nkint; i++) {
                    P[i] = pofk(kint[i]);
                    Wo[i] = W_operators(kint[i]);
                    Wd[i] = W_displacement(kint[i]);
                }

                // The fields: 0 = delta, 1 = s, 2 = nabla^2 delta, 3 = Psi
                const std::vector<FieldKernel> kernels{kernel_delta(),
                                                       kernel_shear(),
                                                       kernel_nabla2(),
                                                       kernel_displacement()};
                const int PSI = 3;
                auto filter = [&](int field, size_t i) { return field == PSI ? Wd[i] : Wo[i]; };

                // The correlators we need: <a(q1) b(q2)> for these pairs
                struct Correlator {
                    int a, b;
                    std::vector<std::vector<Complex>> proj;
                    std::vector<std::vector<double>> R; // R[l][iq]
                };
                std::vector<std::array<int, 2>>
                    needed{{PSI, PSI}, {0, PSI}, {1, PSI}, {2, PSI}, {0, 0}, {0, 1}, {1, 1}, {0, 2}, {1, 2}, {2, 2}};
                std::vector<Correlator> corr;
                for (auto & ab : needed) {
                    Correlator c;
                    c.a = ab[0];
                    c.b = ab[1];
                    c.proj = harmonic_projection(kernels[c.a], kernels[c.b], 4);
                    c.R.assign(5, std::vector<double>(nq + 1, 0.0));
                    corr.push_back(c);
                }

                // Which l are non-zero for a correlator
                auto used_l = [&](const Correlator & c, int l) {
                    for (auto v : c.proj[l])
                        if (std::abs(v) > 1e-10)
                            return true;
                    return false;
                };

                // The radial integrals R_l(q) = int dk k^2 / (2 pi^2) k^(pa + pb) P W_a W_b j_l(kq)
#ifdef USE_OMP
#pragma omp parallel for schedule(dynamic, 16)
#endif
                for (int iq = 0; iq <= nq; iq++) {
                    const double q = iq * dq;
                    std::vector<std::array<double, 5>> jl(nkint);
                    for (size_t i = 0; i < nkint; i++)
                        spherical_bessel_0_to_4(kint[i] * q, jl[i].data());
                    for (auto & c : corr) {
                        const int power = kernels[c.a].power + kernels[c.b].power;
                        for (int l = 0; l <= 4; l++) {
                            if (not used_l(c, l))
                                continue;
                            double sum = 0.0;
                            for (size_t i = 0; i < nkint; i++)
                                sum += wint[i] * kint[i] * kint[i] * std::pow(kint[i], power) * P[i] * filter(c.a, i) *
                                       filter(c.b, i) * jl[i][l];
                            c.R[l][iq] = sum / (2.0 * M_PI * M_PI);
                        }
                    }
                }

                // The correlation tensor of correlator c at q (index iq), with q-hat = z
                auto tensor = [&](const Correlator & c, int iq) {
                    std::vector<double> t(c.proj[0].size(), 0.0);
                    const Complex il[5] = {1.0, Complex(0, 1), -1.0, Complex(0, -1), 1.0};
                    for (int l = 0; l <= 4; l++)
                        for (size_t m = 0; m < t.size(); m++) {
                            const Complex v = il[l] * c.R[l][iq] * c.proj[l][m];
                            t[m] += v.real();
                        }
                    return t;
                };

                // The scalar q-functions (with q-hat = z):
                //   A = X delta + Y zz, U = Uq z, W_L = WLq z,  V_ijl V_ijm = alpha delta_lm + beta z_l z_m,
                //   V_ijl T_ijab V_abm = alphaT delta_lm + betaT z_l z_m, C_ij V_ijl = gamma z_l,
                //   C^L_ij V_ijl = gammaL z_l, C:C, T:T, xi, xi_dL, xi_LL
                struct QFunctions {
                    double X, Y, U, WL, alpha, beta, alphaT, betaT, gamma, gammaL, CC, TT, xi, xidL, xiLL;
                };
                std::vector<QFunctions> qf(nq + 1);
                double Xinf = 0.0;
                {
                    const auto t0 = tensor(corr[0], 0); // <Psi_i Psi_j>(0)
                    Xinf = 2.0 * t0[0];
                }
                for (int iq = 0; iq <= nq; iq++) {
                    auto PP = tensor(corr[0], iq);
                    auto dP = tensor(corr[1], iq);
                    auto sP = tensor(corr[2], iq);
                    auto LP = tensor(corr[3], iq);
                    auto dd = tensor(corr[4], iq);
                    auto ds = tensor(corr[5], iq);
                    auto ss = tensor(corr[6], iq);
                    auto dL = tensor(corr[7], iq);
                    auto sL = tensor(corr[8], iq);
                    auto LL = tensor(corr[9], iq);
                    QFunctions f;
                    // A_ij = 2 (<Psi Psi>(0) - <Psi_i(q1) Psi_j(q2)>)
                    f.X = Xinf - 2.0 * PP[0 * 3 + 0];
                    f.Y = (Xinf - 2.0 * PP[2 * 3 + 2]) - f.X;
                    // The shifts <O(q1) Delta> = -<O(q1) Psi(q2)> (<O Psi> at the same point is zero)
                    f.U = -dP[2];
                    f.WL = -LP[2];
                    std::array<double, 27> V;
                    for (int m = 0; m < 27; m++)
                        V[m] = -sP[m];
                    // V_ijl V_ijm
                    auto VV = [&](int l, int m) {
                        double s = 0.0;
                        for (int ij = 0; ij < 9; ij++)
                            s += V[ij * 3 + l] * V[ij * 3 + m];
                        return s;
                    };
                    f.alpha = VV(0, 0);
                    f.beta = VV(2, 2) - f.alpha;
                    auto VTV = [&](int l, int m) {
                        double s = 0.0;
                        for (int ij = 0; ij < 9; ij++)
                            for (int ab = 0; ab < 9; ab++)
                                s += V[ij * 3 + l] * ss[ij * 9 + ab] * V[ab * 3 + m];
                        return s;
                    };
                    f.alphaT = VTV(0, 0);
                    f.betaT = VTV(2, 2) - f.alphaT;
                    f.gamma = 0.0;
                    f.gammaL = 0.0;
                    for (int ij = 0; ij < 9; ij++) {
                        f.gamma += ds[ij] * V[ij * 3 + 2];
                        f.gammaL += sL[ij] * V[ij * 3 + 2];
                    }
                    f.CC = 0.0;
                    for (int ij = 0; ij < 9; ij++)
                        f.CC += ds[ij] * ds[ij];
                    f.TT = 0.0;
                    for (int m = 0; m < 81; m++)
                        f.TT += ss[m] * ss[m];
                    f.xi = dd[0];
                    f.xidL = dL[0];
                    f.xiLL = LL[0];
                    qf[iq] = f;
                }

                // The pairs
                std::vector<std::array<int, 2>> pairlist;
                for (int a = 0; a < nop; a++)
                    for (int b = a; b < nop; b++)
                        pairlist.push_back({a, b});
                if (names) {
                    const std::vector<std::string> all{"1", "delta", "delta2", "s2", "nabla2delta"};
                    *names = std::vector<std::string>(all.begin(), all.begin() + nop);
                }
                if (pairs)
                    *pairs = pairlist;
                const int npairs = int(pairlist.size());
                auto pair_index = [&](int a, int b) {
                    for (int i = 0; i < npairs; i++)
                        if (pairlist[i][0] == a and pairlist[i][1] == b)
                            return i;
                    return -1;
                };

                // Gauss-Legendre rules for the mu-integral (cached by size)
                std::map<int, std::pair<std::vector<double>, std::vector<double>>> rules;
                auto nodes_needed = [&](double kq) { return 32 * (1 + int((0.75 * kq + 32) / 32)); };
                for (auto k : kvalues)
                    for (int n = nodes_needed(0.0); n <= nodes_needed(k * qmax); n += 32)
                        if (rules.find(n) == rules.end()) {
                            std::vector<double> x, w;
                            gauss_legendre(n, x, w);
                            rules[n] = {x, w};
                        }

                // The linear (analytic) part and the numerical part
                std::vector<std::vector<double>> result(npairs, std::vector<double>(kvalues.size(), 0.0));
#ifdef USE_OMP
#pragma omp parallel for schedule(dynamic, 1)
#endif
                for (int ik = 0; ik < int(kvalues.size()); ik++) {
                    const double k = kvalues[ik];
                    if (k <= 0.0)
                        continue;
                    const double k2 = k * k;
                    const double expinf = std::exp(-0.5 * k2 * Xinf);
                    std::vector<double> sum(npairs, 0.0);
                    std::vector<Complex> F(npairs);
                    for (int iq = 1; iq <= nq; iq++) {
                        const double q = iq * dq;
                        const auto & f = qf[iq];
                        const auto & rule = rules.at(nodes_needed(k * q));
                        const auto & mu = rule.first;
                        const auto & w = rule.second;
                        std::vector<Complex> musum(npairs, 0.0);
                        for (size_t im = 0; im < mu.size(); im++) {
                            const double m = mu[im], m2 = m * m;
                            const double E = std::exp(-0.5 * k2 * (f.X + f.Y * m2));
                            const Complex D(0.0, k * f.U * m);
                            const Complex DL(0.0, k * f.WL * m);
                            const double SS = -k2 * (f.alpha + f.beta * m2);
                            const double STS = -k2 * (f.alphaT + f.betaT * m2);
                            const Complex CS(0.0, k * f.gamma * m);
                            const Complex CLS(0.0, k * f.gammaL * m);
                            const Complex D2 = D * D;
                            // The full integrand E F minus the large-q limit and the part linear in the correlators
                            // (whose Fourier transform we add analytically)
                            const double dA = k2 * (f.X - Xinf + f.Y * m2);
                            auto set = [&](int a, int b, Complex full, Complex linear) {
                                const int i = pair_index(a, b);
                                if (i >= 0)
                                    F[i] = E * full - expinf * linear;
                            };
                            set(ONE, ONE, 1.0, 1.0 - 0.5 * dA);
                            set(ONE, DELTA, D, D);
                            set(ONE, DELTA2, D2, 0.0);
                            set(ONE, S2, SS, 0.0);
                            set(ONE, NABLA2DELTA, DL, DL);
                            set(DELTA, DELTA, f.xi + D2, f.xi);
                            set(DELTA, DELTA2, 2.0 * D * f.xi + D2 * D, 0.0);
                            set(DELTA, S2, 2.0 * CS + D * SS, 0.0);
                            set(DELTA, NABLA2DELTA, f.xidL + D * DL, f.xidL);
                            set(DELTA2, DELTA2, 2.0 * f.xi * f.xi + 4.0 * D2 * f.xi + D2 * D2, 0.0);
                            set(DELTA2, S2, 2.0 * f.CC + 4.0 * D * CS + D2 * SS, 0.0);
                            set(DELTA2, NABLA2DELTA, 2.0 * D * f.xidL + D2 * DL, 0.0);
                            set(S2, S2, 2.0 * f.TT + 4.0 * STS + SS * SS, 0.0);
                            set(S2, NABLA2DELTA, 2.0 * CLS + SS * DL, 0.0);
                            set(NABLA2DELTA, NABLA2DELTA, f.xiLL + DL * DL, f.xiLL);
                            const Complex phase = std::polar(w[im] * 0.5, k * q * m);
                            for (int i = 0; i < npairs; i++)
                                musum[i] += phase * F[i];
                        }
                        // Simpson in q (the integrand is zero at q = 0)
                        const double ws = (iq == nq) ? 1.0 : (iq % 2 == 1 ? 4.0 : 2.0);
                        for (int i = 0; i < npairs; i++)
                            sum[i] += ws * q * q * musum[i].real();
                    }
                    // int d^3q = 4 pi int q^2 dq <...>_mu (we averaged over mu above)
                    for (int i = 0; i < npairs; i++)
                        result[i][ik] = 4.0 * M_PI * sum[i] * dq / 3.0;

                    // The linear part: exp(-k^2 X_inf / 2) times the Fourier transform of the linear terms
                    const double PL = pofk(k);
                    const double wo = W_operators(k), wd = W_displacement(k);
                    auto add = [&](int a, int b, double value) {
                        const int i = pair_index(a, b);
                        if (i >= 0)
                            result[i][ik] += expinf * value;
                    };
                    add(ONE, ONE, PL * wd * wd);
                    add(ONE, DELTA, PL * wo * wd);
                    add(ONE, NABLA2DELTA, -k2 * PL * wo * wd);
                    add(DELTA, DELTA, PL * wo * wo);
                    add(DELTA, NABLA2DELTA, -k2 * PL * wo * wo);
                    add(NABLA2DELTA, NABLA2DELTA, k2 * k2 * PL * wo * wo);
                }
                if (options.verbose)
                    std::cout << "[zeldovich_hybrid_bias_spectra] nq = " << nq << " dq = " << dq << " qmax = " << qmax
                              << " nk = " << nkint << " sigma^2 = X_inf / 2 = " << 0.5 * Xinf << "\n";
                return result;
            }

        } // namespace HYBRIDBIAS
    } // namespace CORRELATIONFUNCTIONS
} // namespace FML
#endif
