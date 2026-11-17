#ifndef FML_PERTURBATIONTHEORY_LPTREDSHIFTSPACE_H
#define FML_PERTURBATIONTHEORY_LPTREDSHIFTSPACE_H

#include <FML/FFTLog/SphericalBesselTransform.h>
#include <FML/PerturbationTheory/LagrangianQFunctions.h>

#include <array>
#include <cmath>
#include <functional>
#include <stdexcept>
#include <string>
#include <vector>

namespace FML::PERTURBATION {

    //=================================================================================================================
    ///
    /// The redshift-space power spectrum P(k, mu) and its multipoles P_0, P_2, P_4 of biased tracers in one-loop
    /// Lagrangian perturbation theory, with the long-wavelength displacements resummed: the direct Lagrangian
    /// expansion of Chen, Vlah, Castorina & White 2021 (arXiv:2012.04636), a port of velocileptors' LPT/lpt_rsd_fftw.py
    /// (Chen, Vlah & White, https://github.com/sfschen/velocileptors, MIT license).
    ///
    /// Bias (Lagrangian, delta_t = 1 + b1 delta + b2/2 (delta^2 - <delta^2>) + bs (s^2 - <s^2>) + b3 O_3), EFT
    /// counterterms (alpha0 + alpha2 mu^2 + alpha4 mu^4 + alpha6 mu^6) k^2 P_ct with P_ct the linear power with the
    /// linear exponent resummed (the Zel'dovich P(k) with use_Pzel), and stochastic terms sn + sn2 k^2 mu^2 +
    /// sn4 k^4 mu^4. The Eulerian linear bias is 1 + b1, and P(k, mu) -> (1 + b1 + f mu^2)^2 P_lin on large scales.
    ///
    /// Input: the linear P(k) at the redshift (k in h/Mpc, P in (Mpc/h)^3) on any increasing grid, extrapolated as a
    /// power law at the ends (velocileptors' loginterp) and multiplied by exp(-(k/cutoff)^2) on the internal log
    /// grid 10^log10_kmin - 10^log10_kmax with N points. f is the growth rate at the redshift. The angular integrals
    /// are series in hypergeometric functions (nmax terms) times spherical Bessel functions j_l, l < jn, done with
    /// FFTLog. Alcock-Paczynski: apar, aperp in the multipoles.
    ///
    /// Accuracy (tests/test_lptrsd.cpp). With velocileptors' settings (jn = 5, nmax = 5, N = 256) it reproduces
    /// velocileptors to ~1e-10. Those settings are not converged: at k = 0.2 (0.3) h/Mpc P_0 is off by 0.8% (1.5%)
    /// and P_2 by 4% (1%) in the angular series, so the defaults are jn = nmax = 10 (converged to < 1e-4 for
    /// k <= 0.4 h/Mpc). Against a direct integration over q of the redshift-space Zel'dovich P(k, mu) (f = 0.8):
    /// 0.2-0.4% with N = 256, ~0.1% with N = 1024 (the default). The f = 0 limit agrees with the real-space CLEFT of
    /// ComputePowerSpectra/HybridBiasOneLoop.h (direct integration) to 2e-4. ngauss = 4 (the default; velocileptors:
    /// 3, which gets P_4 only to ~1%) is converged to ~1e-4 of P_0. Valid for k >~ 0.005 h/Mpc with the default grid (log10_kmin = -5 for lower k: the
    /// Kaiser limit to ~5e-4 at k = 0.001). N must be even and log10_kmax <= 1 (smaller q loses all precision in the
    /// series; the same in velocileptors). Cost: ~0.8 s for the multipole tables at 50 k (defaults; 0.1 s with
    /// velocileptors' settings). Not thread safe (one object per thread; the FFTW plans are made in the constructor).
    ///
    //=================================================================================================================
    class LPTRedshiftSpace {
      public:
        using DVector = std::vector<double>;

        struct Options {
            double cutoff{10.0};       // P(k) is multiplied by exp(-(k/cutoff)^2) (h/Mpc)
            double log10_kmin{-4.0};   // the internal k-grid (h/Mpc)
            double log10_kmax{1.0};   // at most 1 (the q-grid starts at 10^-log10_kmax)
            int N{1024};               // points in the internal k- and q-grids (even; velocileptors: 256)
            int jn{10};                // spherical Bessel functions j_l, l < jn (velocileptors: 5)
            int nmax{10};              // terms in the hypergeometric series (velocileptors: 5)
            double kIR{0.0};           // > 0: resum only the k < kIR part of the linear displacements (Gaussian split)
            bool one_loop{true};       // false: no loop terms (resummed linear Lagrangian theory)
            bool use_Pzel{false};      // the counterterms multiply the Zel'dovich P(k) instead of the linear one
        };

        /// The components of P(k, mu): the coefficients of the bias monomials 1, b1, b1^2, b2, b1 b2, b2^2, bs, b1 bs,
        /// b2 bs, bs^2, b3, b1 b3 and P_ct (the counterterms are (alpha0 + alpha2 mu^2 + ...) k^2 P_ct)
        static constexpr int ncomponents = 13;
        using Components = std::array<double, ncomponents>;
        /// The columns of the multipole tables: the 12 bias monomials, the counterterms k^2 mu^(0,2,4,6) P_ct and the
        /// stochastic terms 1, k^2 mu^2, k^4 mu^4 (Legendre projected, with AP)
        static constexpr int ncolumns = 19;
        using Columns = std::array<double, ncolumns>;

        struct Bias {
            double b1{0}, b2{0}, bs{0}, b3{0};
            double alpha0{0}, alpha2{0}, alpha4{0}, alpha6{0};
            double sn{0}, sn2{0}, sn4{0};
        };
        struct MultipoleTable {
            DVector k;
            std::vector<Columns> p0, p2, p4;
            /// The multipoles of the resummed linear P_ct(k, mu) (without the k^2 of the counterterms)
            DVector pres0, pres2, pres4;
        };
        struct Multipoles {
            DVector k, p0, p2, p4;
        };

        LPTRedshiftSpace() = default;
        LPTRedshiftSpace(const DVector & k, const DVector & P) : LPTRedshiftSpace(k, P, Options()) {}
        LPTRedshiftSpace(const DVector & k, const DVector & P, Options options) : opt(options) {
            if (k.size() != P.size() or k.size() < 4)
                throw std::invalid_argument("[LPTRedshiftSpace] k and P must have the same size (>= 4)");
            if (opt.N < 8 or opt.N % 2 != 0 or opt.jn < 1 or opt.nmax < 1 or not(opt.cutoff > 0) or
                not(opt.log10_kmax > opt.log10_kmin))
                throw std::invalid_argument("[LPTRedshiftSpace] invalid options (N must be even)");
            if (opt.log10_kmax > 1.0 + 1e-12)
                throw std::invalid_argument("[LPTRedshiftSpace] log10_kmax > 1 (q < 0.1 Mpc/h): the angular series "
                                            "loses all precision at small k q");
            const detail::LogInterp Pk(k, P);
            kint = detail::logspace(opt.log10_kmin, opt.log10_kmax, opt.N);
            qint = detail::logspace(-opt.log10_kmax, -opt.log10_kmin, opt.N);
            pint.resize(opt.N);
            for (int i = 0; i < opt.N; i++)
                pint[i] = Pk(kint[i]) * std::exp(-std::pow(kint[i] / opt.cutoff, 2));
            qf = LagrangianQFunctions(kint, pint, qint, opt.kIR, opt.one_loop);
            const int N = opt.N;
            XYlin.resize(N);
            for (int i = 0; i < N; i++)
                XYlin[i] = qf.Xlin_lt[i] + qf.Ylin_lt[i];
            sigma = XYlin.back();
            X10.resize(N);
            Y10.resize(N);
            for (int i = 0; i < N; i++) {
                X10[i] = 2 * qf.X10loop12[i];
                Y10[i] = 2 * qf.Y10loop12[i];
            }
            sph = SOLVERS::FFTLog::SphericalBesselTransform(qint, opt.jn, false, true, true);
        }

        static const std::array<std::string, ncomponents> & component_names() {
            static const std::array<std::string, ncomponents> names{
                "1", "b1", "b1^2", "b2", "b1 b2", "b2^2", "bs", "b1 bs", "b2 bs", "bs^2", "b3", "b1 b3", "P_ct"};
            return names;
        }

        /// The components of P(k, mu) at the growth rate f and mu = nu (no AP) for the k-values
        std::vector<Components> pk_components(double f, double nu, const DVector & k) {
            const auto r = rsd_factors(f, nu);
            std::vector<Components> out;
            for (double kk : k)
                out.push_back(p_integrals(r, kk));
            return out;
        }

        /// P(k, mu) from its components
        static double combine(const Components & c, double k, double nu, const Bias & b) {
            const std::array<double, 12> m{1, b.b1, b.b1 * b.b1, b.b2, b.b1 * b.b2, b.b2 * b.b2,
                                           b.bs, b.b1 * b.bs, b.b2 * b.bs, b.bs * b.bs, b.b3, b.b1 * b.b3};
            double res = 0.0;
            for (int i = 0; i < 12; i++)
                res += c[i] * m[i];
            const double nu2 = nu * nu;
            res += (b.alpha0 + b.alpha2 * nu2 + b.alpha4 * nu2 * nu2 + b.alpha6 * nu2 * nu2 * nu2) * k * k * c[12];
            return res + b.sn + b.sn2 * k * k * nu2 + b.sn4 * k * k * k * k * nu2 * nu2;
        }

        /// P(k, mu) for the bias parameters
        DVector pkmu(double f, double nu, const DVector & k, const Bias & b) {
            const auto c = pk_components(f, nu, k);
            DVector out(k.size());
            for (size_t i = 0; i < k.size(); i++)
                out[i] = combine(c[i], k[i], nu, b);
            return out;
        }

        /// The multipoles of the columns (Gauss-Legendre in mu with 2 ngauss points), with the Alcock-Paczynski
        /// parameters apar, aperp (k_true = k sqrt(1 + mu^2 ((aperp/apar)^2 - 1)) / aperp, ...)
        MultipoleTable multipole_components(double f, const DVector & k, double apar = 1.0, double aperp = 1.0, int ngauss = 4) {
            DVector nus, ws;
            gauss_legendre(2 * ngauss, nus, ws);
            const int nk = int(k.size()), nnu = 2 * ngauss;
            std::vector<std::vector<Columns>> table(nnu, std::vector<Columns>(nk));
            std::vector<DVector> pres(nnu, DVector(nk));
            const double vol_fac = apar * aperp * aperp;
            for (int ii = 0; ii < ngauss; ii++) {
                const double nu = nus[ii];
                const double fac = std::sqrt(1 + nu * nu * (std::pow(aperp / apar, 2) - 1));
                const double k_apfac = fac / aperp;
                const double nu_true = nu * aperp / apar / fac;
                const auto r = rsd_factors(f, nu_true);
                for (int jj = 0; jj < nk; jj++) {
                    const double ktrue = k_apfac * k[jj];
                    const auto pt = p_integrals(r, ktrue);
                    auto & row = table[ii][jj];
                    for (int c = 0; c < 12; c++)
                        row[c] = pt[c];
                    const double k2 = ktrue * ktrue, n2 = nu_true * nu_true;
                    row[12] = k2 * pt[12];
                    row[13] = k2 * n2 * pt[12];
                    row[14] = k2 * n2 * n2 * pt[12];
                    row[15] = k2 * n2 * n2 * n2 * pt[12];
                    row[16] = 1;
                    row[17] = k2 * n2;
                    row[18] = k2 * k2 * n2 * n2;
                    pres[ii][jj] = pt[12];
                }
            }
            for (int ii = 0; ii < ngauss; ii++) {
                table[nnu - 1 - ii] = table[ii];
                pres[nnu - 1 - ii] = pres[ii];
            }
            MultipoleTable out;
            out.k = k;
            out.p0.assign(nk, Columns{});
            out.p2.assign(nk, Columns{});
            out.p4.assign(nk, Columns{});
            out.pres0.assign(nk, 0.0);
            out.pres2.assign(nk, 0.0);
            out.pres4.assign(nk, 0.0);
            for (int ii = 0; ii < nnu; ii++) {
                const double x = nus[ii], x2 = x * x;
                const double L0 = 1.0, L2 = 0.5 * (3 * x2 - 1), L4 = (35 * x2 * x2 - 30 * x2 + 3) / 8.0;
                for (int jj = 0; jj < nk; jj++) {
                    for (int c = 0; c < ncolumns; c++) {
                        out.p0[jj][c] += 0.5 * ws[ii] * L0 * table[ii][jj][c] / vol_fac;
                        out.p2[jj][c] += 2.5 * ws[ii] * L2 * table[ii][jj][c] / vol_fac;
                        out.p4[jj][c] += 4.5 * ws[ii] * L4 * table[ii][jj][c] / vol_fac;
                    }
                    out.pres0[jj] += 0.5 * ws[ii] * L0 * pres[ii][jj] / vol_fac;
                    out.pres2[jj] += 2.5 * ws[ii] * L2 * pres[ii][jj] / vol_fac;
                    out.pres4[jj] += 4.5 * ws[ii] * L4 * pres[ii][jj] / vol_fac;
                }
            }
            return out;
        }

        /// The multipoles P_0, P_2, P_4 for the bias parameters
        static Multipoles combine(const MultipoleTable & t, const Bias & b) {
            const Columns m{1,
                            b.b1,
                            b.b1 * b.b1,
                            b.b2,
                            b.b1 * b.b2,
                            b.b2 * b.b2,
                            b.bs,
                            b.b1 * b.bs,
                            b.b2 * b.bs,
                            b.bs * b.bs,
                            b.b3,
                            b.b1 * b.b3,
                            b.alpha0,
                            b.alpha2,
                            b.alpha4,
                            b.alpha6,
                            b.sn,
                            b.sn2,
                            b.sn4};
            Multipoles out;
            out.k = t.k;
            for (size_t i = 0; i < t.k.size(); i++) {
                double p0 = 0, p2 = 0, p4 = 0;
                for (int c = 0; c < ncolumns; c++) {
                    p0 += t.p0[i][c] * m[c];
                    p2 += t.p2[i][c] * m[c];
                    p4 += t.p4[i][c] * m[c];
                }
                out.p0.push_back(p0);
                out.p2.push_back(p2);
                out.p4.push_back(p4);
            }
            return out;
        }
        Multipoles multipoles(double f, const DVector & k, const Bias & b, double apar = 1.0, double aperp = 1.0, int ngauss = 4) {
            return combine(multipole_components(f, k, apar, aperp, ngauss), b);
        }

        //=============================================================================================================
        // Spectra with a linear operator O(k) = F(k) delta_L(k) (e.g. nabla^2 delta_L: F = -k^2) as a bias operator,
        // at the same order as the delta_L terms (the hybrid bias spectra in redshift space,
        // ComputePowerSpectra/HybridBiasOneLoopRSD.h). They are the delta_L terms with F on the leg of the operator:
        // the q-functions of delta_L (Ulin, U3, X10, Y10, corlin, U11, V) are replaced by those with F on that leg.
        //=============================================================================================================

        /// The q-functions with the operator O on one leg (on the internal q-grid): U = <O_1 Delta>, U3 = <O_1
        /// Psi^(3)>, X10, Y10 (A10 = <O_1 Delta_i Delta_j>, as 2 X10loop12, 2 Y10loop12 for delta), xi = <delta O>,
        /// xiOO = <O O>, U11 = <delta_1 O_2 Delta>, U11OO = <O_1 O_2 Delta>, V (the shear term <O_1 s^2_2 Delta>)
        struct OperatorQFunctions {
            DVector U, U3, X10, Y10, xi, xiOO, U11, U11OO, V;
            double F_kref{0.0}; // F(kref_operator)
        };

        /// The operator spectra (1,O), (delta,O), (delta^2,O), (s^2,O), (O,O) in the normalization of the hybrid
        /// bias basis spectra (delta_t = 1 + b1 delta + b2/2 delta^2 + bs s^2 + bO O: P_tt = ... + 2 bO P_1O +
        /// 2 b1 bO P_dO + b2 bO P_d2O + 2 bs bO P_s2O + bO^2 P_OO; for F = 1 they are [b1]/2, [b1^2], [b1 b2],
        /// [b1 bs]/2 and [b1^2] of pk_components)
        static constexpr int noperator_components = 5;
        /// The k where the constant from the missing small-q part of (O, O) is measured (h/Mpc; the components are
        /// valid for k >~ 0.005 h/Mpc, as the others)
        static constexpr double kref_operator = 5e-3;
        using OperatorComponents = std::array<double, noperator_components>;

        /// The q-functions of the operator F(k) delta_L from the loop spectra with F on the operator leg, given on
        /// k_internal(): Q5F (Q5 with F on the leg p of the operator), R1F, R2F (R1, R2 with F on the internal
        /// leg) (ComputePowerSpectra/HybridBiasOneLoop.h detail::loop_spectra). For F = 1 (and Q5F = Q5, R1F =
        /// R1, R2F = R2 of qfunctions()) these are the delta_L q-functions.
        OperatorQFunctions operator_qfunctions(const std::function<double(double)> & F,
                                               const DVector & Q5F,
                                               const DVector & R1F,
                                               const DVector & R2F) {
            const size_t n = kint.size();
            if (Q5F.size() != n or R1F.size() != n or R2F.size() != n)
                throw std::invalid_argument("[LPTRedshiftSpace::operator_qfunctions] loop spectra must be on k_internal()");
            DVector Fk(n), FP(n), F2P(n), FR1(n), GA(n), GB(n), G11(n), G11OO(n);
            // Without the loop terms (one_loop = false) R1, R2 are zero (the operator spectra are then the Zel'dovich
            // ones with the linear exponent resummed)
            const DVector zero(n, 0.0);
            const DVector & R1 = qf.R1.size() == n ? qf.R1 : zero;
            const DVector & R2 = qf.R2.size() == n ? qf.R2 : zero;
            for (size_t i = 0; i < n; i++) {
                Fk[i] = F(kint[i]);
                FP[i] = Fk[i] * pint[i];
                F2P[i] = Fk[i] * Fk[i] * pint[i];
                FR1[i] = -5.0 / 21.0 * Fk[i] * R1[i];
                // A10 for the operator on the external leg (F R1), the internal leg (R1F, R2F) or the leg of Q5
                const double k2a = 3.0 / 7.0 * Fk[i] * R1[i];
                const double k2b =
                    -3.0 / 7.0 * Fk[i] * (R1[i] + 2.0 * R2[i]) - 6.0 / 7.0 * Q5F[i] - 6.0 / 7.0 * (R1F[i] + R2F[i]);
                GA[i] = -0.5 * (k2a + k2b / 3.0);
                GB[i] = -k2b / 6.0;
                G11[i] = -3.0 / 7.0 * ((R1F[i] + R2F[i]) + Fk[i] * (R1[i] + R2[i]));
                G11OO[i] = -6.0 / 7.0 * Fk[i] * (R1F[i] + R2F[i]);
            }
            OperatorQFunctions o;
            o.F_kref = F(kref_operator);
            o.U = qf.xi_l_n(1, -1, FP);
            for (auto & v : o.U)
                v = -v;
            o.xi = qf.xi_l_n(0, 0, FP);
            o.xiOO = qf.xi_l_n(0, 0, F2P);
            o.U3 = qf.xi_l_n(1, -1, FR1);
            // X10(q = 0) = 0: the zero-lag term is int dk GA / (2 pi^2) (trapezoid in k)
            double zerolag = 0.0;
            for (size_t i = 0; i + 1 < n; i++)
                zerolag += 0.5 * (GA[i] + GA[i + 1]) * (kint[i + 1] - kint[i]);
            zerolag /= 2 * M_PI * M_PI;
            const auto xa = qf.xi_l_n(0, -2, GA);
            const auto xb = qf.xi_l_n(2, -2, GB);
            o.X10.resize(qint.size());
            o.Y10.resize(qint.size());
            for (size_t i = 0; i < qint.size(); i++) {
                o.X10[i] = 2.0 * (zerolag - xa[i] - xb[i]);
                o.Y10[i] = 6.0 * xb[i];
            }
            o.U11 = qf.xi_l_n(1, -1, G11);
            o.U11OO = qf.xi_l_n(1, -1, G11OO);
            // V = 4 J2 <O s> with J2 = 2/15 xi_1^-1 - xi_3^-1 / 5 (the shear-displacement correlator)
            const auto x1m1 = qf.xi_l_n(1, -1);
            const auto x3m1 = qf.xi_l_n(3, -1);
            const auto x20F = qf.xi_l_n(2, 0, FP);
            o.V.resize(qint.size());
            for (size_t i = 0; i < qint.size(); i++)
                o.V[i] = 4.0 * (2.0 * x1m1[i] / 15.0 - 0.2 * x3m1[i]) * x20F[i];
            return o;
        }

        /// The operator spectra at the growth rate f and mu = nu (no AP) for the k-values
        std::vector<OperatorComponents>
        operator_components(const OperatorQFunctions & o, double f, double nu, const DVector & k) {
            const size_t N = qint.size();
            for (const DVector * v : {&o.U, &o.U3, &o.X10, &o.Y10, &o.xi, &o.xiOO, &o.U11, &o.U11OO, &o.V})
                if (v->size() != N)
                    throw std::invalid_argument("[LPTRedshiftSpace::operator_components] q-functions not on q_internal()");
            const auto r = rsd_factors(f, nu);
            std::vector<OperatorComponents> out;
            // The q-grid starts at 10^-log10_kmax: the part of the integrals from smaller q is missing. For (O, O),
            // whose correlator <O O> = int k^2 dk F^2 P j0(kq) is concentrated at small q, this gives a constant (k <<
            // 1 / q_min) that does not go to zero as k -> 0 (it must: O has no k = 0 mode). We remove it with the
            // value at kref_operator, where (O, O) is its linear value F(k)^2 P(k)
            DVector kall{kref_operator};
            kall.insert(kall.end(), k.begin(), k.end());
            for (double kk : kall) {
                const double Kfac = r.Kfac, ksq = kk * kk, K = kk * Kfac, Ksq = K * K, Knfac = nu * (1 + f);
                const auto & Xg = qf.Xlin_gt;
                const auto & Yg = qf.Ylin_gt;
                out.push_back(p_integrals_generic<noperator_components>(
                    r, kk, [&](int, int i, const AngularFactors & a, OperatorComponents & c) {
                        const double xg = Xg[i], yg = Yg[i], Ul = qf.Ulin[i];
                        const double zel0 = a.mu0 - 0.5 * Ksq * (xg * a.mu0 + yg * a.mu2);
                        const double kq1 = K * a.mu1 + f * kk * nu * a.nq1;
                        // (1, O): half the b1 integrand with the operator q-functions
                        c[0] = 0.5 * (-2 * K * (o.U[i] + o.U3[i]) * a.mu1 - Ksq * (o.X10[i] * a.mu0 + o.Y10[i] * a.mu2) -
                                      4 * f * kk * nu * o.U3[i] * a.nq1 -
                                      f * ksq * nu * (o.X10[i] * Knfac * a.mu0 + Kfac * o.Y10[i] * a.mu_nq1) -
                                      2 * K * o.U[i] * (-0.5 * Ksq * (xg * a.mu1 + yg * a.mu3)));
                        // (delta, O): the b1^2 integrand with one leg on the operator
                        c[1] = o.xi[i] * zel0 - Ksq * Ul * o.U[i] * a.mu2 - kq1 * o.U11[i];
                        // (delta^2, O): the b1 b2 integrand (the shift of delta^2 times <delta O>)
                        c[2] = -2 * K * Ul * o.xi[i] * a.mu1;
                        // (s^2, O): half the b1 bs integrand
                        c[3] = -K * o.V[i] * a.mu1;
                        // (O, O): the b1^2 integrand with both legs on the operator
                        c[4] = o.xiOO[i] * zel0 - Ksq * o.U[i] * o.U[i] * a.mu2 - kq1 * o.U11OO[i];
                    }));
            }
            {
                const double Fr = o.F_kref;
                const double lin = Fr * Fr * detail::interp_clamped(std::log(kref_operator), logkint(), pint);
                const double constant = out[0][4] - lin;
                for (size_t i = 1; i < out.size(); i++)
                    out[i][4] -= constant;
            }
            out.erase(out.begin());
            return out;
        }

        const LagrangianQFunctions & qfunctions() const { return qf; }
        DVector logkint() const {
            DVector l(kint.size());
            for (size_t i = 0; i < kint.size(); i++)
                l[i] = std::log(kint[i]);
            return l;
        }
        const DVector & k_internal() const { return kint; }
        const DVector & p_internal() const { return pint; }
        const DVector & q_internal() const { return qint; }
        const Options & options() const { return opt; }

        /// Gauss-Legendre nodes (increasing) and weights on [-1, 1]
        static void gauss_legendre(int n, DVector & x, DVector & w) {
            x.assign(n, 0.0);
            w.assign(n, 0.0);
            for (int i = 0; i < (n + 1) / 2; i++) {
                double z = std::cos(M_PI * (i + 0.75) / (n + 0.5)), dp = 0.0;
                for (int it = 0; it < 100; it++) {
                    double p0 = 1.0, p1 = 0.0;
                    for (int j = 1; j <= n; j++) {
                        const double p2 = p1;
                        p1 = p0;
                        p0 = ((2.0 * j - 1.0) * z * p1 - (j - 1.0) * p2) / j;
                    }
                    dp = n * (z * p0 - p1) / (z * z - 1.0);
                    const double dz = p0 / dp;
                    z -= dz;
                    if (std::abs(dz) < 1e-16)
                        break;
                }
                x[i] = -z;
                x[n - 1 - i] = z;
                w[i] = w[n - 1 - i] = 2.0 / ((1.0 - z * z) * dp * dp);
            }
        }

      private:
        Options opt;
        DVector kint, qint, pint, XYlin, X10, Y10;
        double sigma{0.0};
        LagrangianQFunctions qf;
        SOLVERS::FFTLog::SphericalBesselTransform sph;

        // The angular structure for one (f, nu): the hypergeometric series coefficients of the derivatives of G (in
        // the order G0, dG/dA, dG/dC, d2G/dA2, d2G/dCdA, d2G/dC2, d3G/dA3, d3G/dCdA2, d4G/dA4) and (B A^2/rho^2)^j
        struct RSD {
            double f, nu, Kfac, s, c, ic2, Anu, Bnu;
            std::array<std::vector<DVector>, 9> coeff; // [type][l][n]
            std::vector<DVector> powerfacs;            // [j][q]
        };

        static double hyp2f1_terminating(double a, int n, double c, double x) {
            // 2F1(a, -n; c; x) (a polynomial of degree n)
            double term = 1.0, sum = 1.0;
            for (int j = 0; j < n; j++) {
                term *= (a + j) * (-n + j) / ((c + j) * (j + 1.0)) * x;
                sum += term;
            }
            return sum;
        }

        RSD rsd_factors(double f, double nu) const {
            RSD r;
            const int jn = opt.jn, nmax = opt.nmax;
            r.f = f;
            r.nu = nu;
            r.Kfac = std::sqrt(1 + f * (2 + f) * nu * nu);
            r.s = f * nu * std::sqrt(1 - nu * nu) / r.Kfac;
            r.c = std::sqrt(1 - r.s * r.s);
            r.ic2 = 1 / (r.c * r.c);
            r.Anu = nu * (1 + f) / r.Kfac;
            r.Bnu = std::sqrt(1 - nu * nu) / r.Kfac;
            const double s = r.s, c = r.c, s2 = s * s, c2 = c * c, x = r.ic2;
            for (auto & t : r.coeff)
                t.assign(jn, DVector(nmax, 0.0));
            for (int ll = 0; ll < jn; ll++)
                for (int nn = 0; nn < nmax; nn++) {
                    const double n = ll + nn, m = ll;
                    const int ni = ll + nn;
                    const double h1 = hyp2f1_terminating(0.5 - n, ni, 0.5 - m - n, x);
                    const double h2 = hyp2f1_terminating(1.5 - n, ni, 0.5 - m - n, x);
                    const double fnm =
                        std::tgamma(m + n + 0.5) / std::tgamma(m + 1) / std::tgamma(n + 0.5) / std::tgamma(1 - m + n);
                    // G0
                    r.coeff[0][ll][nn] = fnm * h1;
                    // dG/dA (without 1/(kq))
                    r.coeff[1][ll][nn] = fnm * (s * (-h1 + (1 - 2 * n) * h2)) * (-s / c);
                    // dG/dC
                    r.coeff[2][ll][nn] = fnm * s * (-h1 + (1 - 2 * n) * h2);
                    // d2G/dA2
                    r.coeff[3][ll][nn] =
                        fnm * (1 - 1. / x) * ((2 * m - 1 - 4 * n * (m + 1)) * h1 + (1 - 4 * n * n + m * (4 * n - 2)) * h2);
                    // d2G/dCdA
                    {
                        double ret = -(2 * (m - 2 * n * (1 + m)) * c2 + s2) * h1;
                        ret += (1 - 2 * n) * (2 * (m - n) * c2 + s2) * h2;
                        r.coeff[4][ll][nn] = fnm * ret * s / c;
                    }
                    // d2G/dC2
                    {
                        double ret = ((1 + 2 * m - 4 * n * (1 + m)) * c2 + 2 * s2) * h1;
                        ret += -(1 - 2 * n) * ((1 + 2 * m - 2 * n) * c2 + 2 * s2) * h2;
                        r.coeff[5][ll][nn] = fnm * ret;
                    }
                    // d3G/dA3
                    {
                        const double c1A = 2 * (1 - m) * (1 - 2 * m) + 8 * (2 - m) * (1 + m) * n + 8 * n * n * (1 + m);
                        const double c1C = -(1 - 2 * m + 4 * n * (1 + m));
                        const double c2A = -(1 - 2 * n) * (2 * (1 - 2 * m + 2 * n) * (1 - m + n));
                        const double c2C = (1 - 2 * n) * (1 - 2 * m + 4 * n * (1 + m));
                        const double ret = (c1A * c2 + c1C * s2) * h1 + (c2A * c2 + c2C * s2) * h2;
                        r.coeff[6][ll][nn] = fnm * ret * (s2 / c);
                    }
                    // d3G/dCdA2
                    {
                        double c1 = 2 * (m - 2 * m * m - 4 * n * (1 - m * m) - 4 * n * n * (1 + m)) * c2;
                        c1 += 3 * (1 - 2 * m + 4 * n * (1 + m)) * s2;
                        double cc2 = 2 * (1 - 2 * m + 2 * n) * (m - n) * c2;
                        cc2 += (3 - 6 * m + 8 * n + 4 * m * n) * s2;
                        cc2 *= -(1 - 2 * n);
                        r.coeff[7][ll][nn] = fnm * (c1 * h1 + cc2 * h2) * s;
                    }
                    // d4G/dA4
                    {
                        const double m2 = m * m, m3 = m2 * m, n2 = n * n, n3 = n2 * n;
                        const double c1A = -6 + 22 * m - 24 * m2 + 8 * m3 + n * (-76 - 28 * m + 32 * m2 - 16 * m3) +
                                           n2 * (-56 - 24 * m + 32 * m2) + n3 * (-16 - 16 * m);
                        const double c1C = 9 - 24 * m + 12 * m2 + n * (56 + 24 * m - 32 * m2) + n2 * (32 + 48 * m + 16 * m2);
                        const double c2A = 2 * (-3 + 2 * m - 2 * n) * (1 - 2 * m + 2 * n) * (1 - m + n);
                        const double c2C = 9 - 24 * m + 12 * m2 + n * (44 + 8 * m - 16 * m2) + n2 * (20 + 16 * m);
                        double ret = (c1A * c2 + c1C * s2) * h1;
                        ret += -(1 - 2 * n) * (c2A * c2 + c2C * s2) * h2;
                        r.coeff[8][ll][nn] = ret * fnm * s2;
                    }
                }
            // (B / rho^2)^j with Bfac = -K^2 Y / 2 (times k^(2j) later)
            const int N = opt.N;
            r.powerfacs.assign(jn + nmax, DVector(N, 1.0));
            for (int i = 0; i < N; i++) {
                const double b = -0.5 * r.Kfac * r.Kfac * qf.Ylin_lt[i] / r.ic2;
                double v = 1.0;
                for (int j = 0; j < jn + nmax; j++) {
                    r.powerfacs[j][i] = v;
                    v *= b;
                }
            }
            return r;
        }

        Components p_integrals(const RSD & r, double k) {
            const double f = r.f, nu = r.nu, Kfac = r.Kfac;
            const double ksq = k * k, K = k * Kfac, Ksq = K * K, Knfac = nu * (1 + f);
            const auto & Xg = qf.Xlin_gt;
            const auto & Yg = qf.Ylin_gt;
            const auto & q = qf;
            return p_integrals_generic<ncomponents>(r, k, [&](int, int i, const AngularFactors & a, Components & out) {
                const double xg = Xg[i], yg = Yg[i];
                    double c0 = a.mu0 - 0.5 * Ksq * (xg * a.mu0 + yg * a.mu2);
                    c0 += (-0.5 * ksq *
                               (2 * (Kfac * Kfac + 2 * f * (1 + f) * nu * nu) * a.mu0 * q.Xloop13[i] +
                                2 * (Kfac * Kfac * a.mu2 + 2 * f * Kfac * nu * a.mu_nq1) * q.Yloop13[i] +
                                (Kfac * Kfac + 2 * f * (1 + f) * nu * nu + f * f * nu * nu) * a.mu0 * q.Xloop22[i] +
                                (Kfac * Kfac * a.mu2 + 2 * f * Kfac * nu * a.mu_nq1 + f * f * nu * nu * a.nq2) * q.Yloop22[i]) +
                           Ksq * Ksq / 8. * (xg * xg * a.mu0 + 2 * xg * yg * a.mu2 + yg * yg * a.mu4));
                    c0 += 0.5 * k * k * k *
                          (2 * Kfac * (Kfac * Kfac + f * (1 + f) * nu * nu) * a.mu1 * q.V1loop112[i] +
                           Kfac * Kfac * (Kfac * a.mu1 + f * nu * a.nq1) * q.V3loop112[i] +
                           Kfac * Kfac * (Kfac * a.mu3 + f * nu * a.mu2_nq1) * q.Tloop112[i]);
                    out[0] = c0;
                    const double Ul = q.Ulin[i], xi = q.corlin[i];
                    out[1] = -2 * K * (Ul + q.U3[i]) * a.mu1 - Ksq * (X10[i] * a.mu0 + Y10[i] * a.mu2) -
                                       4 * f * k * nu * q.U3[i] * a.nq1 -
                                       f * ksq * nu * (X10[i] * Knfac * a.mu0 + Kfac * Y10[i] * a.mu_nq1) -
                                       2 * K * Ul * (-0.5 * Ksq * (xg * a.mu1 + yg * a.mu3));
                    out[2] = xi * (a.mu0 - 0.5 * Ksq * (xg * a.mu0 + yg * a.mu2)) - Ksq * Ul * Ul * a.mu2 -
                                       (K * a.mu1 + f * k * nu * a.nq1) * q.U11[i];
                    out[3] = -Ksq * Ul * Ul * a.mu2 - k * (Kfac * a.mu1 + f * nu * a.nq1) * q.U20[i];
                    out[4] = -2 * K * Ul * xi * a.mu1;
                    out[5] = 0.5 * xi * xi * a.mu0;
                    out[6] = -Ksq * (q.Xs2[i] * a.mu0 + q.Ys2[i] * a.mu2) - 2 * k * (Kfac * a.mu1 + f * nu * a.nq1) * q.Us2[i];
                    out[7] = -2 * K * q.V[i] * a.mu1;
                    out[8] = q.chi[i] * a.mu0;
                    out[9] = q.zeta[i] * a.mu0;
                    out[10] = -2 * K * q.Ub3[i] * a.mu1;
                    out[11] = 2 * q.theta[i] * a.mu0;
                    out[12] = opt.use_Pzel ? a.mu0 - 0.5 * Ksq * (xg * a.mu0 + yg * a.mu2) : xi * a.mu0;
            });
        }

        /// The angular factors of the integrands at (l, q_i) (velocileptors' notation)
        struct AngularFactors {
            double mu0, mu1, mu2, mu3, mu4, nq1, mu_nq1, nq2, mu2_nq1;
        };

        /// The Fourier-Bessel integrals of NC integrands: build(l, i, angular factors, out) sets out[c] for the
        /// q-point i (the resummed exponent and the transform are done here)
        template <int NC, class Builder>
        std::array<double, NC> p_integrals_generic(const RSD & r, double k, Builder && build) {
            const int jn = opt.jn, nmax = opt.nmax, N = opt.N;
            const double Kfac = r.Kfac, Anu = r.Anu, Bnu = r.Bnu;
            const double K = k * Kfac, Ksq = K * K;

            // The 9 G-functions [type][l][q]
            const int npow[9] = {0, 1, 1, 2, 2, 2, 3, 3, 4};
            std::array<std::vector<DVector>, 9> G;
            DVector kpow(jn + nmax);
            for (int j = 0; j < jn + nmax; j++)
                kpow[j] = std::pow(k, 2 * j);
            for (int t = 0; t < 9; t++) {
                G[t].assign(jn, DVector(N, 0.0));
                for (int l = 0; l < jn; l++)
                    for (int nn = 0; nn < nmax; nn++) {
                        const double a = r.coeff[t][l][nn] * kpow[l + nn];
                        const auto & pf = r.powerfacs[l + nn];
                        for (int i = 0; i < N; i++)
                            G[t][l][i] += a * pf[i];
                    }
                if (npow[t] > 0)
                    for (int l = 0; l < jn; l++)
                        for (int i = 0; i < N; i++)
                            G[t][l][i] /= std::pow(k * qint[i], npow[t]);
            }
            const auto & G0 = G[0];
            const auto & dGdA = G[1];
            const auto & dGdC = G[2];
            const auto & d2GdA2 = G[3];
            const auto & d2GdCdA = G[4];
            const auto & d2GdC2 = G[5];
            const auto & d3GdA3 = G[6];
            const auto & d3GdCdA2 = G[7];
            const auto & d4GdA4 = G[8];
            // X[l - m] (zero for l < m)
            auto sh = [](const std::vector<DVector> & X, int l, int m, int i) { return l >= m ? X[l - m][i] : 0.0; };

            std::array<double, NC> ret{};
            std::vector<DVector> integrands(NC, DVector(N, 0.0));
            for (int l = 0; l < jn; l++) {
                for (int i = 0; i < N; i++) {
                    const double A = k * qint[i] * r.c, C = k * qint[i] * r.s;
                    const double A2 = A * A, A3 = A2 * A, A4 = A2 * A2;
                    const double G01 = -(dGdA[l][i] + 0.5 * A * sh(G0, l, 1, i));
                    const double G02 =
                        -(d2GdA2[l][i] + A * sh(dGdA, l, 1, i) + 0.5 * sh(G0, l, 1, i) + 0.25 * A2 * sh(G0, l, 2, i));
                    const double G03 = d3GdA3[l][i] + 1.5 * A * sh(d2GdA2, l, 1, i) + 1.5 * sh(dGdA, l, 1, i) +
                                       0.75 * A2 * sh(dGdA, l, 2, i) + 0.75 * A * sh(G0, l, 2, i) +
                                       (A3 / 8.) * sh(G0, l, 3, i);
                    const double G04 = d4GdA4[l][i] + 2 * A * sh(d3GdA3, l, 1, i) + 3 * sh(d2GdA2, l, 1, i) +
                                       1.5 * A2 * sh(d2GdA2, l, 2, i) + 3 * A * sh(dGdA, l, 2, i) +
                                       0.75 * sh(G0, l, 2, i) + 0.5 * A3 * sh(dGdA, l, 3, i) +
                                       0.75 * A2 * sh(G0, l, 3, i) + (A4 / 16.) * sh(G0, l, 4, i);
                    const double G10 = dGdC[l][i] + 0.5 * C * sh(G0, l, 1, i);
                    const double G11 = d2GdCdA[l][i] + 0.5 * C * sh(dGdA, l, 1, i) + 0.5 * A * sh(dGdC, l, 1, i) +
                                       0.25 * A * C * sh(G0, l, 2, i);
                    const double G20 =
                        -(d2GdC2[l][i] + C * sh(dGdC, l, 1, i) + 0.5 * sh(G0, l, 1, i) + 0.25 * C * C * sh(G0, l, 2, i));
                    const double G12 = -(d3GdCdA2[l][i] + 0.5 * C * sh(d2GdA2, l, 1, i) + A * sh(d2GdCdA, l, 1, i) +
                                         0.5 * sh(dGdC, l, 1, i) + 0.5 * A * C * sh(dGdA, l, 2, i) +
                                         0.25 * A2 * sh(dGdC, l, 2, i) + 0.25 * C * sh(G0, l, 2, i) +
                                         (A2 * C / 8.) * sh(G0, l, 3, i));

                    AngularFactors a;
                    a.mu0 = G0[l][i];
                    a.mu1 = G01;
                    a.mu2 = G02;
                    a.mu3 = G03;
                    a.mu4 = G04;
                    a.nq1 = Anu * G01 + Bnu * G10;
                    a.mu_nq1 = Anu * G02 + Bnu * G11;
                    a.nq2 = Anu * Anu * G02 + 2 * Anu * Bnu * G11 + Bnu * Bnu * G20;
                    a.mu2_nq1 = Anu * G03 + Bnu * G12;

                    std::array<double, NC> out{};
                    build(l, i, a, out);
                    for (int c = 0; c < NC; c++)
                        integrands[c][i] = out[c];
                }
                // The resummed exponent times (-2/(kq))^l; for l = 0 the large-q value is subtracted
                for (int i = 0; i < N; i++) {
                    const double fac = std::exp(-0.5 * Ksq * (XYlin[i] - sigma)) * std::pow(-2.0 / (k * qint[i]), l);
                    for (int c = 0; c < NC; c++)
                        integrands[c][i] *= fac;
                }
                if (l == 0)
                    for (int c = 0; c < NC; c++) {
                        const double last = integrands[c][N - 1];
                        for (int i = 0; i < N; i++)
                            integrands[c][i] -= last;
                    }
                // Transform and interpolate (linearly, extrapolating at the ends) at k
                const auto & y = sph.y(l);
                size_t idx = size_t(std::lower_bound(y.begin(), y.end(), k) - y.begin());
                idx = std::clamp<size_t>(idx, 1, y.size() - 1);
                const double w = (k - y[idx - 1]) / (y[idx] - y[idx - 1]);
                DVector g(N);
                for (int c = 0; c < NC; c++) {
                    sph.transform(l, integrands[c].data(), g.data());
                    ret[c] += g[idx - 1] + w * (g[idx] - g[idx - 1]);
                }
            }
            const double suppress = std::exp(-0.5 * Ksq * sigma);
            for (auto & v : ret)
                v *= 4 * suppress * M_PI;
            return ret;
        }
    };

} // namespace FML::PERTURBATION

#endif
