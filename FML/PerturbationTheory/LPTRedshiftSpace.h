#ifndef FML_PERTURBATIONTHEORY_LPTREDSHIFTSPACE_H
#define FML_PERTURBATIONTHEORY_LPTREDSHIFTSPACE_H

#include <FML/FFTLog/SphericalBesselTransform.h>
#include <FML/PerturbationTheory/LagrangianQFunctions.h>

#include <array>
#include <cmath>
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

        const LagrangianQFunctions & qfunctions() const { return qf; }
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
            const int jn = opt.jn, nmax = opt.nmax, N = opt.N;
            const double f = r.f, nu = r.nu, Kfac = r.Kfac, Anu = r.Anu, Bnu = r.Bnu;
            const double ksq = k * k, K = k * Kfac, Ksq = K * K, Knfac = nu * (1 + f);

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

            const auto & Xg = qf.Xlin_gt;
            const auto & Yg = qf.Ylin_gt;
            const auto & q = qf;
            Components ret{};
            std::vector<DVector> integrands(ncomponents, DVector(N, 0.0));
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

                    const double mu0 = G0[l][i], mu1 = G01, mu2 = G02, mu3 = G03, mu4 = G04;
                    const double nq1 = Anu * G01 + Bnu * G10;
                    const double mu_nq1 = Anu * G02 + Bnu * G11;
                    const double nq2 = Anu * Anu * G02 + 2 * Anu * Bnu * G11 + Bnu * Bnu * G20;
                    const double mu2_nq1 = Anu * G03 + Bnu * G12;
                    const double xg = Xg[i], yg = Yg[i];

                    double c0 = mu0 - 0.5 * Ksq * (xg * mu0 + yg * mu2);
                    c0 += (-0.5 * ksq *
                               (2 * (Kfac * Kfac + 2 * f * (1 + f) * nu * nu) * mu0 * q.Xloop13[i] +
                                2 * (Kfac * Kfac * mu2 + 2 * f * Kfac * nu * mu_nq1) * q.Yloop13[i] +
                                (Kfac * Kfac + 2 * f * (1 + f) * nu * nu + f * f * nu * nu) * mu0 * q.Xloop22[i] +
                                (Kfac * Kfac * mu2 + 2 * f * Kfac * nu * mu_nq1 + f * f * nu * nu * nq2) * q.Yloop22[i]) +
                           Ksq * Ksq / 8. * (xg * xg * mu0 + 2 * xg * yg * mu2 + yg * yg * mu4));
                    c0 += 0.5 * k * k * k *
                          (2 * Kfac * (Kfac * Kfac + f * (1 + f) * nu * nu) * mu1 * q.V1loop112[i] +
                           Kfac * Kfac * (Kfac * mu1 + f * nu * nq1) * q.V3loop112[i] +
                           Kfac * Kfac * (Kfac * mu3 + f * nu * mu2_nq1) * q.Tloop112[i]);
                    integrands[0][i] = c0;
                    const double Ul = q.Ulin[i], xi = q.corlin[i];
                    integrands[1][i] = -2 * K * (Ul + q.U3[i]) * mu1 - Ksq * (X10[i] * mu0 + Y10[i] * mu2) -
                                       4 * f * k * nu * q.U3[i] * nq1 -
                                       f * ksq * nu * (X10[i] * Knfac * mu0 + Kfac * Y10[i] * mu_nq1) -
                                       2 * K * Ul * (-0.5 * Ksq * (xg * mu1 + yg * mu3));
                    integrands[2][i] = xi * (mu0 - 0.5 * Ksq * (xg * mu0 + yg * mu2)) - Ksq * Ul * Ul * mu2 -
                                       (K * mu1 + f * k * nu * nq1) * q.U11[i];
                    integrands[3][i] = -Ksq * Ul * Ul * mu2 - k * (Kfac * mu1 + f * nu * nq1) * q.U20[i];
                    integrands[4][i] = -2 * K * Ul * xi * mu1;
                    integrands[5][i] = 0.5 * xi * xi * mu0;
                    integrands[6][i] = -Ksq * (q.Xs2[i] * mu0 + q.Ys2[i] * mu2) - 2 * k * (Kfac * mu1 + f * nu * nq1) * q.Us2[i];
                    integrands[7][i] = -2 * K * q.V[i] * mu1;
                    integrands[8][i] = q.chi[i] * mu0;
                    integrands[9][i] = q.zeta[i] * mu0;
                    integrands[10][i] = -2 * K * q.Ub3[i] * mu1;
                    integrands[11][i] = 2 * q.theta[i] * mu0;
                    integrands[12][i] = opt.use_Pzel ? mu0 - 0.5 * Ksq * (xg * mu0 + yg * mu2) : xi * mu0;
                }
                // The resummed exponent times (-2/(kq))^l; for l = 0 the large-q value is subtracted
                for (int i = 0; i < N; i++) {
                    const double fac = std::exp(-0.5 * Ksq * (XYlin[i] - sigma)) * std::pow(-2.0 / (k * qint[i]), l);
                    for (int c = 0; c < ncomponents; c++)
                        integrands[c][i] *= fac;
                }
                if (l == 0)
                    for (int c = 0; c < ncomponents; c++) {
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
                for (int c = 0; c < ncomponents; c++) {
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
