#ifndef FML_PERTURBATIONTHEORY_EPTREDSHIFTSPACE_H
#define FML_PERTURBATIONTHEORY_EPTREDSHIFTSPACE_H

#include <FML/FFTLog/SphericalBesselTransform.h>
#include <FML/PerturbationTheory/LPTRedshiftSpace.h>
#include <FML/PerturbationTheory/LagrangianQFunctions.h>
#include <FML/PerturbationTheory/NoWiggle.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

namespace FML::PERTURBATION {

    //=================================================================================================================
    ///
    /// One-loop Eulerian perturbation theory (EPT) for the velocity moments of biased tracers, with the one-loop
    /// spectra computed with FFTLog from the "k-expanded" Lagrangian expansion (all displacements expanded) and
    /// mapped to the Eulerian bias basis. A port of velocileptors' EPT/cleft_kexpanded_fftw.py,
    /// EPT/velocity_moments_kexpanded_fftw.py and EPT/ept_fftw.py (Chen, Vlah & White 2020, arXiv:2005.00523,
    /// https://github.com/sfschen/velocileptors, MIT license). Not IR resummed: the building block of
    /// EPTRedshiftSpace.
    ///
    /// The tables (on the output grid k, with bias monomials of the Eulerian bias b1, b2, bs, b3):
    ///   pk: b1^2, b1 b2, b1 bs, b2^2, b2 bs, bs^2, b1 b3                        (the real-space P(k))
    ///   vk: b1, b1^2, b2, b1 b2, bs, b1 bs, b3                                   (pairwise velocity, f = 1)
    ///   s0k, s2k: 1, b1, b1^2, b2, bs                                            (velocity dispersion, Legendre)
    ///   g1k, g3k: 1, b1                                                          (third moment, polynomial basis)
    ///   k0, k2, k4                                                               (fourth moment, matter)
    /// The linear P is multiplied by exp(-(k/cutoff)^2) on the internal grid (velocileptors: cutoff = 20). The
    /// options are the internal grid, see EPTRedshiftSpace for its accuracy.
    ///
    //=================================================================================================================
    class EPTVelocityMoments {
      public:
        using DVector = std::vector<double>;
        struct Options {
            double cutoff{20.0};     // P(k) is multiplied by exp(-(k/cutoff)^2) (h/Mpc)
            double log10_kmin{-5.0}; // the internal k-grid (h/Mpc); the q-grid is its inverse (velocileptors: -4)
            double log10_kmax{3.0};
            int N{6000};             // points in the internal grids (velocileptors: 4000)
        };

        DVector k;                        // the output grid
        std::array<DVector, 7> pk, vk;    // [column][k]
        std::array<DVector, 5> s0k, s2k;  // [column][k]
        std::array<DVector, 2> g1k, g3k;  // [column][k]
        DVector k0, k2, k4;

        EPTVelocityMoments() = default;

        /// kint, pint: the internal grid and the linear P on it (with the cutoff); kout: the output grid
        EPTVelocityMoments(const DVector & kint_, const DVector & pint_, const DVector & qint_, const DVector & kout)
            : k(kout), kint(kint_), pint(pint_), qint(qint_) {
            N = int(kint.size());
            qf = LagrangianQFunctions(kint, pint, qint, 0.0, true);
            sph = SOLVERS::FFTLog::SphericalBesselTransform(qint, 5, false, true, true);
            compute();
        }

        const LagrangianQFunctions & qfunctions() const { return qf; }

      private:
        DVector kint, pint, qint;
        int N{0};
        LagrangianQFunctions qf;
        SOLVERS::FFTLog::SphericalBesselTransform sph;

        // The Legendre projections of mu^n onto j_l (the l-th term of the angular integral)
        static double mu0(int l) { return l == 0 ? 1.0 : 0.0; }
        static double mu1(int l) { return l == 1 ? 1.0 : 0.0; }
        static double mu2(int l) { return (l == 0 ? 1. / 3 : 0.0) - (l == 2 ? 2. / 3 : 0.0); }
        static double mu3(int l) { return (l == 1 ? 0.6 : 0.0) - (l == 3 ? 0.4 : 0.0); }
        static double mu4(int l) { return (l == 0 ? 0.2 : 0.0) - (l == 2 ? 4. / 7 : 0.0) + (l == 4 ? 8. / 35 : 0.0); }

        // tot(k) += k^kpow 4 pi sum_l int dq q^2 j_l(kq) [F_l(q) - F_l(qmax)] on kint (integrand(l, i) = F_l(q_i))
        template <class F>
        void add_hankel(DVector & tot, int kpow, F && integrand) {
            DVector Fq(N), g(N);
            for (int l = 0; l < 5; l++) {
                bool nonzero = false;
                for (int i = 0; i < N; i++) {
                    Fq[i] = integrand(l, i);
                    nonzero = nonzero or Fq[i] != 0.0;
                }
                if (not nonzero)
                    continue;
                const double last = Fq[N - 1];
                for (int i = 0; i < N; i++)
                    Fq[i] -= last;
                sph.transform(l, Fq.data(), g.data());
                const auto & y = sph.y(l);
                for (int i = 0; i < N; i++) {
                    const double kk = kint[i];
                    size_t idx = size_t(std::lower_bound(y.begin(), y.end(), kk) - y.begin());
                    idx = std::clamp<size_t>(idx, 1, y.size() - 1);
                    const double w = (kk - y[idx - 1]) / (y[idx] - y[idx - 1]);
                    tot[i] += 4 * M_PI * std::pow(kk, kpow) * (g[idx - 1] + w * (g[idx] - g[idx - 1]));
                }
            }
        }

        // The 12 Lagrangian bias components (1, b1, b1^2, b2, b1 b2, b2^2, bs, b1 bs, b2 bs, bs^2, b3, b1 b3) of
        // the k-expanded LPT moments on kint, interpolated (linearly) onto k. Index 1-based below (velocileptors).
        using Table = std::array<DVector, 12>;

        Table to_output(const Table & t) const {
            Table out;
            for (int c = 0; c < 12; c++) {
                out[c].assign(k.size(), 0.0);
                bool nonzero = false;
                for (double v : t[c])
                    nonzero = nonzero or v != 0.0;
                if (nonzero)
                    out[c] = detail::interp_clamped(k, kint, t[c]);
            }
            return out;
        }

        void compute() {
            const auto & q = qf;
            const auto & X = q.Xlin;
            const auto & Y = q.Ylin;
            const auto & U = q.Ulin;
            const auto & xi = q.corlin;
            auto zero = [&]() {
                Table t;
                for (auto & c : t)
                    c.assign(N, 0.0);
                return t;
            };

            // Power spectrum (KECLEFT)
            Table P = zero();
            for (int i = 0; i < N; i++) {
                P[0][i] = pint[i] + 9. / 98 * q.Q1[i] + 10. / 21 * q.R1[i] + 3. / 7 * (2 * q.R2[i] + q.Q2[i]);
                P[1][i] = 2 * pint[i] + 10. / 21 * q.R1[i] + 1. / 7 * (6 * q.R1[i] + 12 * q.R2[i] + 6 * q.Q5[i]);
                P[2][i] = pint[i] + 6. / 7 * (q.R1[i] + q.R2[i]);
                P[3][i] = 3. / 7 * q.Q8[i];
                P[6][i] = 2. / 7 * q.Qs2[i];
                P[10][i] = 2 * q.Rb3[i] * pint[i];
                P[11][i] = 2 * q.Rb3[i] * pint[i];
            }
            add_hankel(P[5], 0, [&](int l, int i) { return mu0(l) * 0.5 * xi[i] * xi[i]; });
            add_hankel(P[8], 0, [&](int l, int i) { return mu0(l) * q.chi[i]; });
            add_hankel(P[9], 0, [&](int l, int i) { return mu0(l) * q.zeta[i]; });
            add_hankel(P[4], 1, [&](int l, int i) { return mu1(l) * (-2 * U[i] * xi[i]); });
            add_hankel(P[7], 1, [&](int l, int i) { return mu1(l) * (-2 * q.V[i]); });
            add_hankel(P[2], 2, [&](int l, int i) {
                return mu0(l) * (-0.5 * X[i] * xi[i]) + mu2(l) * (-0.5 * Y[i] * xi[i] - U[i] * U[i]);
            });
            add_hankel(P[3], 2, [&](int l, int i) { return mu2(l) * (-U[i] * U[i]); });
            add_hankel(P[6], 2, [&](int l, int i) { return mu0(l) * (-q.Xs2[i]) + mu2(l) * (-q.Ys2[i]); });
            add_hankel(P[1], 3, [&](int l, int i) { return mu1(l) * (U[i] * X[i]) + mu3(l) * (U[i] * Y[i]); });
            add_hankel(P[0], 4, [&](int l, int i) {
                return mu0(l) * (X[i] * X[i] / 8) + mu2(l) * (X[i] * Y[i] / 4) + mu4(l) * (Y[i] * Y[i] / 8);
            });

            // Pairwise velocity (with f = 1)
            Table V = zero();
            for (int i = 0; i < N; i++) {
                const double kk = kint[i];
                V[0][i] = (-2 * pint[i] - 2 * (2 * 9. / 98 * q.Q1[i] + 4 * 5. / 21 * q.R1[i]) -
                           12. / 7 * (q.Q2[i] + 2 * q.R2[i])) /
                          kk;
                V[1][i] = (-2 * pint[i] - 3 * (12 * q.R2[i] + 6 * q.Q5[i] + 6 * q.R1[i]) / 7 - 3 * (10. / 21 * q.R1[i])) / kk;
                V[2][i] = -12. / 7 * (q.R1[i] + q.R2[i]) / kk;
                V[3][i] = -6. / 7 * q.Q8[i] / kk;
                V[6][i] = -2 * 2 * 1. / 7 * q.Qs2[i] / kk;
                V[10][i] = -2 * q.Rb3[i] * pint[i] / kk;
            }
            add_hankel(V[4], 0, [&](int l, int i) { return mu1(l) * (2 * xi[i] * U[i]); });
            add_hankel(V[7], 0, [&](int l, int i) { return mu1(l) * (2 * q.V[i]); });
            add_hankel(V[2], 1, [&](int l, int i) {
                return mu0(l) * (xi[i] * X[i]) + mu2(l) * (2 * U[i] * U[i] + xi[i] * Y[i]);
            });
            add_hankel(V[3], 1, [&](int l, int i) { return mu2(l) * (2 * U[i] * U[i]); });
            add_hankel(V[6], 1, [&](int l, int i) { return mu0(l) * (2 * q.Xs2[i]) + mu2(l) * (2 * q.Ys2[i]); });
            add_hankel(V[1], 2, [&](int l, int i) {
                return mu1(l) * (-2 * U[i] * X[i] - U[i] * X[i]) + mu3(l) * (-2 * U[i] * Y[i] - U[i] * Y[i]);
            });
            add_hankel(V[0], 3, [&](int l, int i) {
                return mu0(l) * (-0.5 * X[i] * X[i]) + mu2(l) * (-0.5 * (X[i] * Y[i] + Y[i] * X[i])) +
                       mu4(l) * (-0.5 * Y[i] * Y[i]);
            });

            // sigma_par
            Table Sp = zero();
            for (int i = 0; i < N; i++) {
                const double ksq = kint[i] * kint[i];
                Sp[0][i] = (-2 * pint[i] - 2 * (4 * 9. / 98 * q.Q1[i] + 6 * 5. / 21 * q.R1[i]) -
                            6 * (5. / 3 * 3. / 7 * (q.Q2[i] + 2 * q.R2[i]))) /
                           ksq;
                Sp[1][i] = (-2 * 2 * (12. / 7 * q.R2[i] + 6. / 7 * q.Q5[i] + 6. / 7 * q.R1[i])) / ksq;
            }
            add_hankel(Sp[2], 0, [&](int l, int i) {
                return mu0(l) * (xi[i] * X[i]) + mu2(l) * (xi[i] * Y[i] + 2 * U[i] * U[i]);
            });
            add_hankel(Sp[3], 0, [&](int l, int i) { return mu2(l) * (2 * U[i] * U[i]); });
            add_hankel(Sp[6], 0, [&](int l, int i) { return mu0(l) * (2 * q.Xs2[i]) + mu2(l) * (2 * q.Ys2[i]); });
            add_hankel(Sp[1], 1, [&](int l, int i) {
                return mu1(l) * (-2 * (U[i] * X[i] + 2 * U[i] * X[i])) + mu3(l) * (-2 * (U[i] * Y[i] + 2 * U[i] * Y[i]));
            });
            add_hankel(Sp[0], 2, [&](int l, int i) {
                return mu0(l) * (-X[i] * X[i] - 0.5 * X[i] * X[i]) +
                       mu2(l) * (-2 * X[i] * Y[i] - 0.5 * (X[i] * Y[i] + Y[i] * X[i])) +
                       mu4(l) * (-Y[i] * Y[i] - 0.5 * Y[i] * Y[i]);
            });

            // Tr sigma
            Table St = zero();
            for (int i = 0; i < N; i++) {
                const double ksq = kint[i] * kint[i];
                St[0][i] = (-2 * pint[i] - 2 * (4 * 9. / 98 * q.Q1[i] + 6 * 5. / 21 * q.R1[i]) +
                            6. / 7 * (q.Q1[i] - 5 * q.Q2[i] + 4 * q.R1[i] - 10 * q.R2[i])) /
                           ksq;
                St[1][i] = -4 / ksq * 3. / 7 * (4 * q.R2[i] + 2 * q.Q5[i]);
            }
            add_hankel(St[2], 0, [&](int l, int i) {
                return mu0(l) * (xi[i] * (3 * X[i] + Y[i]) + 2 * U[i] * U[i]);
            });
            add_hankel(St[3], 0, [&](int l, int i) { return mu0(l) * (2 * U[i] * U[i]); });
            add_hankel(St[6], 0, [&](int l, int i) { return mu0(l) * (2 * (3 * q.Xs2[i] + q.Ys2[i])); });
            add_hankel(St[1], 1, [&](int l, int i) {
                return mu1(l) * (-2 * U[i] * (3 * X[i] + Y[i]) - 4 * U[i] * (X[i] + Y[i]));
            });
            add_hankel(St[0], 2, [&](int l, int i) {
                return mu0(l) * ((3 * X[i] + Y[i]) * (-0.5 * X[i]) - X[i] * X[i]) +
                       mu2(l) * ((3 * X[i] + Y[i]) * (-0.5 * Y[i]) - (Y[i] * Y[i] + 2 * X[i] * Y[i]));
            });

            // gamma1, gamma2 (columns 1, b1)
            Table G1 = zero(), G2 = zero();
            for (int i = 0; i < N; i++) {
                const double k3 = std::pow(kint[i], 3);
                G1[0][i] = 36. / 7 * (q.Q2[i] + 2 * q.R2[i]) / k3;
                G2[0][i] = -12. / 7 * (2 * q.R1[i] - 6 * q.R2[i] + q.Q1[i] - 3 * q.Q2[i]) / k3;
            }
            add_hankel(G1[1], 0, [&](int l, int i) { return mu1(l) * (6 * U[i] * X[i]) + mu3(l) * (6 * U[i] * Y[i]); });
            add_hankel(G1[0], 1, [&](int l, int i) {
                return mu0(l) * (3 * X[i] * X[i]) + mu2(l) * (3 * (X[i] * Y[i] + Y[i] * X[i])) + mu4(l) * (3 * Y[i] * Y[i]);
            });
            add_hankel(G2[1], 0, [&](int l, int i) { return mu1(l) * (2 * U[i] * (5 * X[i] + 3 * Y[i])); });
            add_hankel(G2[0], 1, [&](int l, int i) {
                return mu0(l) * (5 * X[i] * X[i] + X[i] * Y[i]) +
                       mu2(l) * (2 * X[i] * Y[i] + Y[i] * (5 * X[i] + 3 * Y[i]));
            });

            // kappa (3 components)
            Table Kp = zero();
            add_hankel(Kp[0], 0, [&](int l, int i) {
                return mu0(l) * (15 * X[i] * X[i] + 10 * X[i] * Y[i] + 3 * Y[i] * Y[i]);
            });
            add_hankel(Kp[1], 0, [&](int l, int i) {
                return mu0(l) * (5 * X[i] * X[i] + X[i] * Y[i]) + mu2(l) * (7 * X[i] * Y[i] + 3 * Y[i] * Y[i]);
            });
            add_hankel(Kp[2], 0, [&](int l, int i) {
                return mu0(l) * (3 * X[i] * X[i]) + mu2(l) * (6 * X[i] * Y[i]) + mu4(l) * (3 * Y[i] * Y[i]);
            });

            // Onto the output grid and to the EPT basis (1-based column indices as in velocileptors)
            const Table Po = to_output(P), Vo = to_output(V), Spo = to_output(Sp), Sto = to_output(St);
            const Table G1o = to_output(G1), G2o = to_output(G2), Ko = to_output(Kp);
            const size_t nk = k.size();
            for (auto & c : pk)
                c.assign(nk, 0.0);
            for (auto & c : vk)
                c.assign(nk, 0.0);
            for (auto & c : s0k)
                c.assign(nk, 0.0);
            for (auto & c : s2k)
                c.assign(nk, 0.0);
            for (auto & c : g1k)
                c.assign(nk, 0.0);
            for (auto & c : g3k)
                c.assign(nk, 0.0);
            k0.assign(nk, 0.0);
            k2.assign(nk, 0.0);
            k4.assign(nk, 0.0);
            for (size_t j = 0; j < nk; j++) {
                auto p = [&](int c) { return Po[c - 1][j]; };
                auto v = [&](int c) { return Vo[c - 1][j]; };
                auto s0 = [&](int c) { return Sto[c - 1][j] / 3.0; };
                auto s2 = [&](int c) { return Spo[c - 1][j] - Sto[c - 1][j] / 3.0; };
                pk[0][j] = p(1);
                pk[1][j] = p(5) - 16. / 21 * p(6) + 2. / 7 * p(9);
                pk[2][j] = p(8) + 4. / 7 * p(10) - 8. / 21 * p(9);
                pk[3][j] = p(6);
                pk[4][j] = p(9);
                pk[5][j] = p(10);
                pk[6][j] = 2 * p(3) - p(2) - 8. / 21 * p(5) + 2. / 7 * p(8);

                vk[0][j] = v(1) - v(3) + 8. / 21 * v(5) - 2. / 7 * v(8);
                vk[1][j] = v(3) - 8. / 21 * v(5) + 2. / 7 * v(8);
                vk[2][j] = v(4) - v(5);
                vk[3][j] = v(5);
                vk[4][j] = v(7) - v(8);
                vk[5][j] = v(8);
                vk[6][j] = v(2) - v(1) - v(3) - 8. / 21 * (v(4) - v(5)) + 2. / 7 * (v(7) - v(8));

                for (int t = 0; t < 2; t++) {
                    auto s = [&](int c) { return t == 0 ? s0(c) : s2(c); };
                    auto & out = t == 0 ? s0k : s2k;
                    out[0][j] = s(1) - s(2) + s(3) + 8. / 21 * s(4) - 2. / 7 * s(7);
                    out[1][j] = s(2) - 2 * s(3) - 8. / 21 * s(4) + 2. / 7 * s(7);
                    out[2][j] = s(3);
                    out[3][j] = s(4);
                    out[4][j] = s(7);
                }

                // Polynomial basis: g1 = 3/2 (gamma2 - gamma1), g3 = 5/2 gamma1 - 3/2 gamma2
                auto g1 = [&](int c) { return 1.5 * G2o[c - 1][j] - 1.5 * G1o[c - 1][j]; };
                auto g3 = [&](int c) { return 2.5 * G1o[c - 1][j] - 1.5 * G2o[c - 1][j]; };
                g1k[0][j] = g1(1) - g1(2);
                g1k[1][j] = g1(2);
                g3k[0][j] = g3(1) - g3(2);
                g3k[1][j] = g3(2);

                const double K1 = Ko[0][j], K2 = Ko[1][j], K3 = Ko[2][j];
                k0[j] = 3. / 8 * (K1 - 2 * K2 + K3);
                k2[j] = 3. / 4 * (-K1 + 6 * K2 - 5 * K3);
                k4[j] = 1. / 8 * (3 * K1 - 30 * K2 + 35 * K3);
            }
        }
    };

    //=================================================================================================================
    ///
    /// The IR-resummed redshift-space power spectrum P(k, mu) and its multipoles P_0, P_2, P_4 of biased tracers in
    /// one-loop Eulerian perturbation theory (the "full resummation" of the moment expansion: the one-loop P(k, mu)
    /// of the velocity moments up to the fourth, with the wiggle part damped by exp(-k^2 Sigma^2 (1 + f (2 + f)
    /// mu^2) / 2), Sigma^2 = X(r_bao) + Y(r_bao)/3 of the no-wiggle spectrum). A port of velocileptors'
    /// EPT/ept_fullresum_fftw.py (REPT; Chen, Vlah & White 2020, arXiv:2005.00523; Chen, Vlah, Castorina & White
    /// 2021, arXiv:2012.04636; https://github.com/sfschen/velocileptors, MIT license).
    ///
    /// Bias: Eulerian, delta_t = b1 delta + b2/2 (delta^2 - <delta^2>) + bs (s^2 - <s^2>) + b3 O_3 (so P -> (b1 +
    /// f mu^2)^2 P_lin on large scales; note the Lagrangian b1 of LPTRedshiftSpace is this b1 - 1); counterterms
    /// (alpha0 + alpha2 mu^2 + alpha4 mu^4 + alpha6 mu^6) k^2 (P_nw + e^(-k^2 Sigma^2(mu)/2) P_w); stochastic terms
    /// sn + sn2 k^2 mu^2 + sn4 k^4 mu^4.
    ///
    /// Input: the linear P(k) at the redshift (k in h/Mpc, P in (Mpc/h)^3) on any increasing grid, and optionally
    /// the no-wiggle spectrum on the same grid. Without it the no-wiggle spectrum is computed on the internal grid
    /// with the discrete sine transform method of NoWiggle.h (default) or, as velocileptors' REPT, a Savitzky-Golay
    /// filter (order 4, width ln 7 in ln k; its ~0.5% broadband bias at k >~ 0.3 h/Mpc goes straight into P, where
    /// the wiggles are damped away; velocileptors also applies its cutoff exp(-(k/cutoff)^2) to it twice, here
    /// once). The output grid is log-spaced, kmin - kmax with nk points.
    /// Alcock-Paczynski parameters apar, aperp (k and mu observed; velocileptors' convention).
    ///
    /// Accuracy (tests/test_rept.cpp). With velocileptors' internal grid (log10_kmin = -4, N = 4000) it reproduces
    /// velocileptors to ~1e-8 (the same no-wiggle spectrum). That grid is not converged: the k-expanded loop terms
    /// are large cancelling powers of k, and the truncation of the q-range at 1e4 Mpc/h leaves errors of 0.2% (k =
    /// 0.3 h/Mpc) - 0.5% (k = 0.5) in the real-space P (and ~0.7% in P(k, mu = 0.7) at k = 0.5). The default grid
    /// (1e-5 - 1e3 h/Mpc, N = 6000) agrees with the direct one-loop SPT of OneLoop.h to 6e-5 for k <= 0.5 and is
    /// converged to ~5e-5 of P_0. The angular integrals are exact (l <= 4); ngauss = 4 (velocileptors' default) is
    /// converged to < 1e-6. Cost ~0.3 s to construct (two sets of q-functions); the tables are cheap. Not thread safe
    /// (one object per thread; the FFTW plans are made in the constructor).
    ///
    //=================================================================================================================
    class EPTRedshiftSpace {
      public:
        using DVector = std::vector<double>;

        struct Options {
            double kmin{1e-2}, kmax{0.5}; // the output grid (h/Mpc), log-spaced
            int nk{100};
            double rbao{110.0};           // Sigma^2 = X + Y/3 at this separation (Mpc/h)
            double sigma2_bao{-1.0};      // >= 0: use this Sigma^2 ((Mpc/h)^2) instead
            EPTVelocityMoments::Options internal{};
            /// The no-wiggle spectrum when none is given (NoWiggle.h), computed on the internal grid: the discrete
            /// sine transform method (default) or velocileptors' Savitzky-Golay filter (biased by ~0.5% for
            /// k >~ 0.3 h/Mpc, where the result is the no-wiggle part)
            enum class NoWiggle { DST, SavitzkyGolay };
            NoWiggle nowiggle{NoWiggle::DST};
            NoWiggleDSTOptions dst{};
        };

        /// The columns of the tables: the bias monomials 1, b1, b1^2, b2, b1 b2, b2^2, bs, b1 bs, b2 bs, bs^2, b3,
        /// b1 b3, the counterterms alpha0, alpha2, alpha4, alpha6 and the stochastic terms sn, sn2, sn4
        static constexpr int ncolumns = 19;
        using Columns = std::array<double, ncolumns>;
        using Bias = LPTRedshiftSpace::Bias;
        using Multipoles = LPTRedshiftSpace::Multipoles;
        struct MultipoleTable {
            DVector k;
            std::vector<Columns> p0, p2, p4;
        };

        EPTRedshiftSpace() = default;
        EPTRedshiftSpace(const DVector & k, const DVector & P) : EPTRedshiftSpace(k, P, {}, Options()) {}
        EPTRedshiftSpace(const DVector & k, const DVector & P, const DVector & Pnw)
            : EPTRedshiftSpace(k, P, Pnw, Options()) {}
        EPTRedshiftSpace(const DVector & k, const DVector & P, const DVector & Pnw, Options options) : opt(options) {
            const auto & io = opt.internal;
            if (k.size() != P.size() or k.size() < 4 or (not Pnw.empty() and Pnw.size() != k.size()))
                throw std::invalid_argument("[EPTRedshiftSpace] k, P (and Pnw) must have the same size (>= 4)");
            if (io.N < 16 or not(io.cutoff > 0) or not(io.log10_kmax > io.log10_kmin) or opt.nk < 4 or
                not(opt.kmin > 0) or not(opt.kmax > opt.kmin))
                throw std::invalid_argument("[EPTRedshiftSpace] invalid options");
            if (std::log10(opt.kmin) < io.log10_kmin or std::log10(opt.kmax) > io.log10_kmax)
                throw std::invalid_argument("[EPTRedshiftSpace] the output k-range must be inside the internal grid");
            kv = detail::logspace(std::log10(opt.kmin), std::log10(opt.kmax), opt.nk);
            const DVector kint = detail::logspace(io.log10_kmin, io.log10_kmax, io.N);
            const DVector qint = detail::logspace(-io.log10_kmax, -io.log10_kmin, io.N);
            DVector cut(io.N);
            for (int i = 0; i < io.N; i++)
                cut[i] = std::exp(-std::pow(kint[i] / io.cutoff, 2));

            const detail::LogInterp Pk(k, P);
            DVector p_raw(io.N), pint(io.N), pnw_int(io.N);
            for (int i = 0; i < io.N; i++)
                p_raw[i] = Pk(kint[i]);
            plin.resize(opt.nk);
            for (int j = 0; j < opt.nk; j++)
                plin[j] = Pk(kv[j]);
            plin_nw.resize(opt.nk);
            if (Pnw.empty()) {
                if (opt.nowiggle == Options::NoWiggle::DST)
                    pnw_int = nowiggle_dst(kint, p_raw, opt.dst);
                else
                    pnw_int = detail::savgol_filter(p_raw, detail::savgol_window(kint, std::log(7.0)), 4);
                const detail::LogInterp Pnwk(kint, pnw_int);
                for (int j = 0; j < opt.nk; j++)
                    plin_nw[j] = Pnwk(kv[j]);
            } else {
                const detail::LogInterp Pnwk(k, Pnw);
                for (int i = 0; i < io.N; i++)
                    pnw_int[i] = Pnwk(kint[i]);
                for (int j = 0; j < opt.nk; j++)
                    plin_nw[j] = Pnwk(kv[j]);
            }
            for (int i = 0; i < io.N; i++) {
                pint[i] = p_raw[i] * cut[i];
                pnw_int[i] *= cut[i];
            }
            plin_w.resize(opt.nk);
            for (int j = 0; j < opt.nk; j++)
                plin_w[j] = plin[j] - plin_nw[j];

            ept = EPTVelocityMoments(kint, pint, qint, kv);
            ept_nw = EPTVelocityMoments(kint, pnw_int, qint, kv);
            if (opt.sigma2_bao >= 0.0) {
                sigma2_bao = opt.sigma2_bao;
            } else {
                const auto & qn = ept_nw.qfunctions();
                DVector XY(io.N);
                for (int i = 0; i < io.N; i++)
                    XY[i] = qn.Xlin[i] + qn.Ylin[i] / 3.0;
                sigma2_bao = detail::interp_clamped(opt.rbao, qint, XY);
            }
        }

        static const std::array<std::string, ncolumns> & column_names() {
            static const std::array<std::string, ncolumns> names{"1",      "b1",     "b1^2",   "b2",     "b1 b2",
                                                                 "b2^2",   "bs",     "b1 bs",  "b2 bs",  "bs^2",
                                                                 "b3",     "b1 b3",  "alpha0", "alpha2", "alpha4",
                                                                 "alpha6", "sn",     "sn2",    "sn4"};
            return names;
        }

        /// The output grid and the linear spectra on it
        const DVector & k() const { return kv; }
        const DVector & plinear() const { return plin; }
        const DVector & plinear_nowiggle() const { return plin_nw; }
        double sigma2_BAO() const { return sigma2_bao; }
        /// The unresummed EPT tables of P and of its no-wiggle version
        const EPTVelocityMoments & moments() const { return ept; }
        const EPTVelocityMoments & moments_nowiggle() const { return ept_nw; }
        const Options & options() const { return opt; }

        /// The columns of P(k_obs, mu_obs) on the output grid (k_obs = k())
        std::vector<Columns> table_at_mu(double f, double mu_obs, double apar = 1.0, double aperp = 1.0) const {
            const int nk = opt.nk;
            const auto ap = ap_transform(mu_obs, apar, aperp);
            const double mu = ap.mu, mu2 = mu * mu;
            DVector damp_exp(nk), damp(nk);
            for (int j = 0; j < nk; j++) {
                damp_exp[j] = -0.5 * kv[j] * kv[j] * sigma2_bao * (1 + f * (2 + f) * mu2);
                damp[j] = std::exp(damp_exp[j]);
            }
            static const int idx[12][5] = {{0, 0, 1, 1, 1}, {0, 1, 2, 2, 0}, {1, 2, 3, 0, 0}, {0, 3, 4, 0, 0},
                                           {2, 4, 0, 0, 0}, {4, 0, 0, 0, 0}, {0, 5, 5, 0, 0}, {3, 6, 0, 0, 0},
                                           {5, 0, 0, 0, 0}, {6, 0, 0, 0, 0}, {0, 7, 0, 0, 0}, {7, 0, 0, 0, 0}};
            std::vector<DVector> cols(ncolumns, DVector(nk, 0.0));
            for (int c = 0; c < 12; c++) {
                const int * ii = idx[c];
                for (int j = 0; j < nk; j++) {
                    const double kk = kv[j], km = kk * mu;
                    auto term = [&](const EPTVelocityMoments & e) {
                        double r = 0.0;
                        if (ii[0])
                            r += e.pk[ii[0] - 1][j];
                        if (ii[1])
                            r -= f * kk * mu2 * e.vk[ii[1] - 1][j];
                        if (ii[2])
                            r -= 0.5 * f * f * kk * kk * mu2 * (e.s0k[ii[2] - 1][j] + 0.5 * e.s2k[ii[2] - 1][j] * (3 * mu2 - 1));
                        if (ii[3])
                            r += f * f * f * km * km * km / 6. * (e.g1k[ii[3] - 1][j] * mu + e.g3k[ii[3] - 1][j] * mu * mu2);
                        if (ii[4])
                            r += f * f * f * f * km * km * km * km / 24. * (e.k0[j] + e.k2[j] * mu2 + e.k4[j] * mu2 * mu2);
                        return r;
                    };
                    const double nw = term(ept_nw), all = term(ept);
                    double r = nw + damp[j] * (all - nw);
                    // Linear theory compensation of (b1 + f mu^2)^2 P_w
                    if (ii[0] == 1)
                        r -= damp_exp[j] * damp[j] * plin_w[j];
                    if (ii[1] == 1)
                        r -= damp_exp[j] * damp[j] * 2 * f * mu2 * plin_w[j];
                    if (ii[2] == 1)
                        r -= damp_exp[j] * damp[j] * f * f * mu2 * mu2 * plin_w[j];
                    cols[c][j] = r;
                }
            }
            for (int j = 0; j < nk; j++) {
                const double k2 = kv[j] * kv[j], pres = plin_nw[j] + damp[j] * plin_w[j];
                cols[12][j] = k2 * pres;
                cols[13][j] = k2 * mu2 * pres;
                cols[14][j] = k2 * mu2 * mu2 * pres;
                cols[15][j] = k2 * mu2 * mu2 * mu2 * pres;
                cols[16][j] = 1.0;
                cols[17][j] = k2 * mu2;
                cols[18][j] = k2 * k2 * mu2 * mu2;
            }
            std::vector<Columns> out(nk);
            const bool no_ap = apar == 1.0 and aperp == 1.0;
            const detail::NotAKnotSpline spline(ap.kobs);
            for (int c = 0; c < ncolumns; c++) {
                const DVector v = no_ap ? cols[c] : spline(cols[c], kv);
                for (int j = 0; j < nk; j++)
                    out[j][c] = v[j] / ap.vol;
            }
            return out;
        }

        /// P(k_obs, mu_obs) for the bias parameters (bFoG: the extra -bFoG k^4 mu^4 (b1 + f mu^2)^2 P_res term)
        DVector pkmu(double f, double mu_obs, const Bias & b, double apar = 1.0, double aperp = 1.0, double bFoG = 0.0) const {
            const auto t = table_at_mu(f, mu_obs, apar, aperp);
            const Columns m = monomials(b);
            DVector out(opt.nk, 0.0);
            for (int j = 0; j < opt.nk; j++)
                for (int c = 0; c < ncolumns; c++)
                    out[j] += m[c] * t[j][c];
            if (bFoG != 0.0) {
                const auto ap = ap_transform(mu_obs, apar, aperp);
                const double mu2 = ap.mu * ap.mu;
                DVector fog(opt.nk);
                for (int j = 0; j < opt.nk; j++) {
                    const double damp = std::exp(-0.5 * kv[j] * kv[j] * sigma2_bao * (1 + f * (2 + f) * mu2));
                    fog[j] = -bFoG * mu2 * mu2 * std::pow(kv[j], 4) * std::pow(b.b1 + f * mu2, 2) *
                             (plin_nw[j] + damp * plin_w[j]);
                }
                if (not(apar == 1.0 and aperp == 1.0))
                    fog = detail::NotAKnotSpline(ap.kobs)(fog, kv);
                for (int j = 0; j < opt.nk; j++)
                    out[j] += fog[j] / ap.vol;
            }
            return out;
        }

        /// The multipoles of the columns (Gauss-Legendre in mu with 2 ngauss points)
        MultipoleTable multipole_components(double f, double apar = 1.0, double aperp = 1.0, int ngauss = 4) const {
            DVector nus, ws;
            LPTRedshiftSpace::gauss_legendre(2 * ngauss, nus, ws);
            const int nk = opt.nk, nnu = 2 * ngauss;
            std::vector<std::vector<Columns>> table(nnu);
            for (int ii = 0; ii < ngauss; ii++)
                table[ii] = table_at_mu(f, nus[ii], apar, aperp);
            for (int ii = 0; ii < ngauss; ii++)
                table[nnu - 1 - ii] = table[ii];
            MultipoleTable out;
            out.k = kv;
            out.p0.assign(nk, Columns{});
            out.p2.assign(nk, Columns{});
            out.p4.assign(nk, Columns{});
            for (int ii = 0; ii < nnu; ii++) {
                const double x2 = nus[ii] * nus[ii];
                const double L2 = 0.5 * (3 * x2 - 1), L4 = (35 * x2 * x2 - 30 * x2 + 3) / 8.0;
                for (int j = 0; j < nk; j++)
                    for (int c = 0; c < ncolumns; c++) {
                        out.p0[j][c] += 0.5 * ws[ii] * table[ii][j][c];
                        out.p2[j][c] += 2.5 * ws[ii] * L2 * table[ii][j][c];
                        out.p4[j][c] += 4.5 * ws[ii] * L4 * table[ii][j][c];
                    }
            }
            return out;
        }

        /// The multipoles P_0, P_2, P_4 for the bias parameters
        static Multipoles combine(const MultipoleTable & t, const Bias & b) {
            const Columns m = monomials(b);
            Multipoles out;
            out.k = t.k;
            for (size_t j = 0; j < t.k.size(); j++) {
                double p0 = 0, p2 = 0, p4 = 0;
                for (int c = 0; c < ncolumns; c++) {
                    p0 += t.p0[j][c] * m[c];
                    p2 += t.p2[j][c] * m[c];
                    p4 += t.p4[j][c] * m[c];
                }
                out.p0.push_back(p0);
                out.p2.push_back(p2);
                out.p4.push_back(p4);
            }
            return out;
        }
        Multipoles multipoles(double f, const Bias & b, double apar = 1.0, double aperp = 1.0, int ngauss = 4) const {
            return combine(multipole_components(f, apar, aperp, ngauss), b);
        }

        static Columns monomials(const Bias & b) {
            return Columns{1,        b.b1,     b.b1 * b.b1, b.b2,      b.b1 * b.b2, b.b2 * b.b2, b.bs,
                           b.b1 * b.bs, b.b2 * b.bs, b.bs * b.bs, b.b3, b.b1 * b.b3, b.alpha0, b.alpha2,
                           b.alpha4, b.alpha6, b.sn,      b.sn2,       b.sn4};
        }

      private:
        Options opt;
        DVector kv, plin, plin_nw, plin_w;
        double sigma2_bao{0.0};
        EPTVelocityMoments ept, ept_nw;

        // mu_true, the k_obs at which the true k = kv are seen, and the volume factor apar aperp^2
        // (https://arxiv.org/abs/1312.4611 eqs. 58-60, as velocileptors)
        struct AP {
            double mu, vol;
            DVector kobs;
        };
        AP ap_transform(double mu_obs, double apar, double aperp) const {
            AP r;
            const double F = apar / aperp;
            const double fac = std::sqrt(1 + mu_obs * mu_obs * (1. / (F * F) - 1));
            r.mu = mu_obs / F / fac;
            r.vol = aperp * aperp * apar;
            r.kobs.resize(kv.size());
            for (size_t j = 0; j < kv.size(); j++)
                r.kobs[j] = kv[j] * aperp / fac;
            return r;
        }
    };

} // namespace FML::PERTURBATION

#endif
