#ifndef FML_PERTURBATION_LOOPINTEGRALS_H
#define FML_PERTURBATION_LOOPINTEGRALS_H

#include <FML/PerturbationTheory/SPTKernels.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <functional>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef USE_OMP
#include <omp.h>
#endif

//=====================================================================================================================
///
/// Direct quadrature of one-loop integrals in redshift space (the engine of TNS.h and EFTofLSS.h). The integrands are
/// functions of the wave vectors, so any kernel (SPTKernels.h) and any dependence on the line of sight can be
/// integrated; the dependence on mu = k.los/k is resolved on a set of mu nodes.
///
///  - integrate_22: int d^3q/(2pi)^3 g(k, q, p = k - q, los) over the (q, p) triangle domain (as OneLoop.h: composite
///    Gauss-Legendre in log q and log p with breaks where the domain changes, which resolves p -> 0) and the azimuth
///    phi of q around k (Gauss-Chebyshev in cos phi, exact for integrands that are polynomials in cos phi of degree <
///    2 nphi; the kernels here are). The integrand gets the linear spectra at q, p and k.
///  - P13 type: int d^3q/(2pi)^3 g(k, q, -q, los) P(q). For kernels of degree 0 in the momenta (EdS) the angular
///    integral is a function of r = q/k only: Angular13 tabulates it once on fixed r nodes (log Gauss-Legendre with a
///    break at r = 1) and P13(k) is a sum over the nodes with P(k r). The angular integral uses the q -> -q symmetry
///    of the kernels (half the sphere, doubled) and p = |k - q| as the variable on it (resolves r -> 1).
///
/// Units: whatever P and k are in (k [h/Mpc], P [(Mpc/h)^3] in FML). Accuracy is set by QuadratureOptions (tests:
/// TNS.h, EFTofLSS.h).
///
//=====================================================================================================================

namespace FML::PERTURBATION::LOOPS {

    using SPT::Vec3;
    using DVector = std::vector<double>;
    constexpr double pi = 3.141592653589793238462643383279502884;

    struct QuadratureOptions {
        int order = 8;          // Gauss-Legendre points per segment (q, p and r)
        double log_step = 0.25; // largest segment width in ln q, ln p, ln r
        int nphi = 6;           // Gauss-Chebyshev points in cos phi on [0, pi] (exact to degree 2 nphi - 1)
    };

    namespace detail {
        inline void require(bool ok, const std::string & msg) {
            if (not ok)
                throw std::invalid_argument(msg);
        }
        inline std::pair<DVector, DVector> gauss_legendre(int n) {
            require(n >= 2 and n <= 64, "Gauss-Legendre order must be in [2, 64]");
            DVector x(n), w(n);
            for (int i = 0; i < (n + 1) / 2; ++i) {
                double z = std::cos(pi * (i + .75) / (n + .5)), dp = 0;
                for (int it = 0; it < 100; ++it) {
                    double p = 1, prev = 0;
                    for (int j = 1; j <= n; ++j) {
                        double old = p;
                        p = ((2 * j - 1) * z * p - (j - 1) * prev) / j;
                        prev = old;
                    }
                    dp = n * (z * p - prev) / (z * z - 1);
                    double next = z - p / dp;
                    if (std::abs(next - z) < 4e-16) {
                        z = next;
                        break;
                    }
                    z = next;
                }
                x[i] = -z;
                x[n - 1 - i] = z;
                w[i] = w[n - 1 - i] = 2 / ((1 - z * z) * dp * dp);
            }
            return {x, w};
        }
        /// Nodes and weights of int_lo^hi dx g(x) as composite Gauss-Legendre in ln x (weights include the x of dx =
        /// x d ln x), with the segment ends at the given breaks
        inline void log_nodes(DVector breaks,
                              const QuadratureOptions & opt,
                              const std::pair<DVector, DVector> & gl,
                              DVector & x,
                              DVector & w) {
            x.clear();
            w.clear();
            std::sort(breaks.begin(), breaks.end());
            breaks.erase(std::unique(breaks.begin(), breaks.end()), breaks.end());
            for (size_t b = 1; b < breaks.size(); b++) {
                const double a = std::log(breaks[b - 1]), c = std::log(breaks[b]);
                if (not(c > a))
                    continue;
                const int nseg = std::max(1, int(std::ceil((c - a) / opt.log_step)));
                for (int s = 0; s < nseg; s++) {
                    const double lo = a + (c - a) * s / nseg, hi = a + (c - a) * (s + 1) / nseg;
                    for (size_t j = 0; j < gl.first.size(); j++) {
                        const double v = std::exp(0.5 * (lo + hi) + 0.5 * (hi - lo) * gl.first[j]);
                        x.push_back(v);
                        w.push_back(gl.second[j] * 0.5 * (hi - lo) * v);
                    }
                }
            }
        }
    } // namespace detail

    //=================================================================================================================
    /// A linear power spectrum P(k) from a table: cubic (natural) spline in ln P(ln k), zero outside the table
    //=================================================================================================================
    class Spectrum {
      public:
        Spectrum() = default;
        Spectrum(const DVector & k, const DVector & P) {
            detail::require(k.size() >= 4 and k.size() == P.size(), "Spectrum: need >= 4 points");
            for (size_t i = 0; i < k.size(); i++) {
                detail::require(k[i] > 0 and P[i] > 0 and std::isfinite(P[i]), "Spectrum: need k > 0 and P > 0");
                if (i > 0)
                    detail::require(k[i] > k[i - 1], "Spectrum: k must increase");
                x.push_back(std::log(k[i]));
                y.push_back(std::log(P[i]));
            }
            spline(y, y2);
        }
        /// A spectrum that may change sign (e.g. the wiggles P - P_nw): cubic spline in P(ln k)
        static Spectrum signed_spectrum(const DVector & k, const DVector & P) {
            Spectrum s;
            s.logarithmic = false;
            for (size_t i = 0; i < k.size(); i++) {
                detail::require(k[i] > 0 and std::isfinite(P[i]), "Spectrum: need k > 0, finite P");
                s.x.push_back(std::log(k[i]));
                s.y.push_back(P[i]);
            }
            s.spline(s.y, s.y2);
            return s;
        }
        double operator()(double k) const {
            if (x.empty() or not(k > 0))
                return 0.0;
            const double lk = std::log(k);
            if (lk < x.front() or lk > x.back())
                return 0.0;
            size_t i = size_t(std::upper_bound(x.begin(), x.end(), lk) - x.begin());
            i = std::min(std::max(i, size_t(1)), x.size() - 1);
            const double h = x[i] - x[i - 1], a = (x[i] - lk) / h, b = 1.0 - a;
            const double v = a * y[i - 1] + b * y[i] + ((a * a * a - a) * y2[i - 1] + (b * b * b - b) * y2[i]) * h * h / 6.0;
            return logarithmic ? std::exp(v) : v;
        }
        double kmin() const { return std::exp(x.front()); }
        double kmax() const { return std::exp(x.back()); }

      private:
        DVector x, y, y2;
        bool logarithmic{true};
        void spline(const DVector & yy, DVector & d2) {
            const size_t n = x.size();
            d2.assign(n, 0.0);
            DVector u(n, 0.0);
            for (size_t i = 1; i + 1 < n; i++) {
                const double sig = (x[i] - x[i - 1]) / (x[i + 1] - x[i - 1]);
                const double p = sig * d2[i - 1] + 2.0;
                d2[i] = (sig - 1.0) / p;
                u[i] = (yy[i + 1] - yy[i]) / (x[i + 1] - x[i]) - (yy[i] - yy[i - 1]) / (x[i] - x[i - 1]);
                u[i] = (6.0 * u[i] / (x[i + 1] - x[i - 1]) - sig * u[i - 1]) / p;
            }
            for (size_t i = n - 1; i-- > 0;)
                d2[i] = d2[i] * d2[i + 1] + u[i];
        }
    };

    /// What the integrands get at a quadrature point: the vectors (k along z, the line of sight in the x-z plane) and
    /// the spectra (one value per spectrum given) at |q|, |p| and |k|
    struct Point {
        Vec3 k, q, p, los;
        double kmag, qmag, pmag, mu;
        const double * Pq;
        const double * Pp;
        const double * Pk;
    };

    //=================================================================================================================
    /// out[i][imu][c] = int d^3q/(2pi)^3 g_c(point) at k[i] and mu[imu], with g(point, values) writing ncomp values.
    /// The spectra are zero outside [qmin, qmax] (the integration domain of q and p). Parallel over k with OpenMP;
    /// the integrand must be thread safe.
    //=================================================================================================================
    template <class Integrand>
    std::vector<std::vector<DVector>> integrate_22(const DVector & kout,
                                                   const DVector & mu,
                                                   const std::vector<const Spectrum *> & spectra,
                                                   double qmin,
                                                   double qmax,
                                                   int ncomp,
                                                   const QuadratureOptions & opt,
                                                   Integrand && g) {
        detail::require(qmin > 0 and qmax > qmin and ncomp > 0 and opt.nphi >= 1 and opt.log_step > 0,
                        "integrate_22: invalid range or options");
        const auto gl = detail::gauss_legendre(opt.order);
        const size_t ns = spectra.size();
        // Gauss-Chebyshev in cos phi on [0, pi]; int_0^2pi dphi = 2 int_0^pi (integrands even in sin phi)
        DVector cphi(opt.nphi), sphi(opt.nphi);
        for (int j = 0; j < opt.nphi; j++) {
            const double phi = pi * (j + 0.5) / opt.nphi;
            cphi[j] = std::cos(phi);
            sphi[j] = std::sin(phi);
        }
        const double wphi = 2.0 * pi / opt.nphi / std::pow(2.0 * pi, 3);
        std::vector<std::vector<DVector>> out(kout.size(), std::vector<DVector>(mu.size(), DVector(ncomp, 0.0)));
        std::vector<Vec3> los(mu.size());
        for (size_t m = 0; m < mu.size(); m++)
            los[m] = Vec3{std::sqrt(std::max(0.0, 1.0 - mu[m] * mu[m])), 0.0, mu[m]};
        std::string error;
#ifdef USE_OMP
#pragma omp parallel for schedule(dynamic, 1)
#endif
        for (size_t ik = 0; ik < kout.size(); ik++) {
            try {
                const double k = kout[ik];
                DVector Pk(ns), Pq(ns), Pp(ns), vals(ncomp);
                for (size_t s = 0; s < ns; s++)
                    Pk[s] = (*spectra[s])(k);
                DVector qs, qw, ps, pw;
                DVector qbreaks{qmin, qmax};
                for (double b : {k, k - qmin, k + qmin, qmax - k, k - qmax})
                    if (b > qmin and b < qmax)
                        qbreaks.push_back(b);
                detail::log_nodes(qbreaks, opt, gl, qs, qw);
                Point pt;
                pt.k = Vec3{0, 0, k};
                pt.kmag = k;
                for (size_t iq = 0; iq < qs.size(); iq++) {
                    const double q = qs[iq];
                    const double plo = std::max(qmin, std::abs(k - q)), phi_ = std::min(qmax, k + q);
                    if (not(phi_ > plo))
                        continue;
                    for (size_t s = 0; s < ns; s++)
                        Pq[s] = (*spectra[s])(q);
                    detail::log_nodes({plo, phi_}, opt, gl, ps, pw);
                    for (size_t ip = 0; ip < ps.size(); ip++) {
                        const double p = ps[ip];
                        const double x = std::clamp((k * k + q * q - p * p) / (2.0 * k * q), -1.0, 1.0);
                        const double sx = std::sqrt(std::max(0.0, 1.0 - x * x));
                        for (size_t s = 0; s < ns; s++)
                            Pp[s] = (*spectra[s])(p);
                        const double w = qw[iq] * pw[ip] * q * p / k * wphi;
                        for (int j = 0; j < opt.nphi; j++) {
                            pt.q = Vec3{q * sx * cphi[j], q * sx * sphi[j], q * x};
                            pt.p = pt.k - pt.q;
                            pt.qmag = q;
                            pt.pmag = p;
                            pt.Pq = Pq.data();
                            pt.Pp = Pp.data();
                            pt.Pk = Pk.data();
                            for (size_t m = 0; m < mu.size(); m++) {
                                pt.los = los[m];
                                pt.mu = mu[m];
                                std::fill(vals.begin(), vals.end(), 0.0);
                                g(pt, vals.data());
                                auto & o = out[ik][m];
                                for (int c = 0; c < ncomp; c++)
                                    o[c] += w * vals[c];
                            }
                        }
                    }
                }
            } catch (const std::exception & e) {
#ifdef USE_OMP
#pragma omp critical
#endif
                error = e.what();
            }
        }
        if (not error.empty())
            throw std::runtime_error(error);
        return out;
    }

    //=================================================================================================================
    /// The angular integrals J_c(r, mu) = int dOmega_q g_c(k = 1 along z, q = r qhat, -q, los(mu)) of a kernel of
    /// degree 0 in the momenta that is symmetric under q -> -q, tabulated on fixed r nodes. A P13-type integral is then
    /// int d^3q/(2pi)^3 g P(q) = k^3/(2pi)^3 sum_r w_r r^2 P(k r) J(r, mu) (sum_with).
    //=================================================================================================================
    class Angular13 {
      public:
        Angular13() = default;
        template <class Kernel>
        Angular13(double rmin, double rmax, const DVector & mu_, int ncomp_, const QuadratureOptions & opt, Kernel && g)
            : mu(mu_), ncomp(ncomp_) {
            detail::require(rmin > 0 and rmax > 1.0 and rmin < 1.0 and ncomp > 0, "Angular13: invalid r range");
            const auto gl = detail::gauss_legendre(opt.order);
            detail::log_nodes({rmin, 1.0, rmax}, opt, gl, r, w);
            J.assign(r.size(), std::vector<DVector>(mu.size(), DVector(ncomp, 0.0)));
            std::string error;
#ifdef USE_OMP
#pragma omp parallel for schedule(dynamic, 4)
#endif
            for (size_t ir = 0; ir < r.size(); ir++) {
                try {
                    J[ir] = angular(r[ir], mu, ncomp, opt, gl, g);
                } catch (const std::exception & e) {
#ifdef USE_OMP
#pragma omp critical
#endif
                    error = e.what();
                }
            }
            if (not error.empty())
                throw std::runtime_error(error);
        }

        /// The angular integral J_c(r, mu) at one r
        template <class Kernel>
        static std::vector<DVector> angular(double rr,
                                            const DVector & mu,
                                            int ncomp,
                                            const QuadratureOptions & opt,
                                            const std::pair<DVector, DVector> & gl,
                                            Kernel && g) {
            std::vector<DVector> out(mu.size(), DVector(ncomp, 0.0));
            DVector cphi(opt.nphi), sphi(opt.nphi);
            for (int j = 0; j < opt.nphi; j++) {
                const double phi = pi * (j + 0.5) / opt.nphi;
                cphi[j] = std::cos(phi);
                sphi[j] = std::sin(phi);
            }
            std::vector<Vec3> los(mu.size());
            for (size_t m = 0; m < mu.size(); m++)
                los[m] = Vec3{std::sqrt(std::max(0.0, 1.0 - mu[m] * mu[m])), 0.0, mu[m]};
            DVector ps, pw, vals(ncomp);
            // Half the sphere, x = cos(k, q) in [0, 1], as p = |k - q| in [|1 - r|, sqrt(1 + r^2)]: dx = p dp / r
            detail::log_nodes({std::abs(1.0 - rr), std::sqrt(1.0 + rr * rr)}, opt, gl, ps, pw);
            const Vec3 k{0, 0, 1};
            for (size_t ip = 0; ip < ps.size(); ip++) {
                const double p = ps[ip];
                const double x = std::clamp((1.0 + rr * rr - p * p) / (2.0 * rr), -1.0, 1.0);
                const double sx = std::sqrt(std::max(0.0, 1.0 - x * x));
                // 2 (both hemispheres) x 2 (phi in [0, 2pi] from [0, pi]) x pi/nphi (Gauss-Chebyshev)
                const double wt = pw[ip] * p / rr * 4.0 * pi / opt.nphi;
                for (int j = 0; j < opt.nphi; j++) {
                    const Vec3 q{rr * sx * cphi[j], rr * sx * sphi[j], rr * x};
                    for (size_t m = 0; m < mu.size(); m++) {
                        std::fill(vals.begin(), vals.end(), 0.0);
                        g(k, q, los[m], vals.data());
                        for (int c = 0; c < ncomp; c++)
                            out[m][c] += wt * vals[c];
                    }
                }
            }
            return out;
        }

        /// out[imu][c] = int d^3q/(2pi)^3 g_c P(q) at k (P zero outside its table and the r range)
        std::vector<DVector> integrate(double k, const Spectrum & P) const {
            std::vector<DVector> out(mu.size(), DVector(ncomp, 0.0));
            const double pref = k * k * k / std::pow(2.0 * pi, 3);
            for (size_t ir = 0; ir < r.size(); ir++) {
                const double Pq = P(k * r[ir]);
                if (Pq == 0.0)
                    continue;
                const double wt = pref * w[ir] * r[ir] * r[ir] * Pq;
                for (size_t m = 0; m < mu.size(); m++)
                    for (int c = 0; c < ncomp; c++)
                        out[m][c] += wt * J[ir][m][c];
            }
            return out;
        }
        const DVector & rnodes() const { return r; }

      private:
        DVector mu, r, w;
        int ncomp{0};
        std::vector<std::vector<DVector>> J;
    };

    /// Gauss-Legendre nodes of mu^2 in (0, 1) (n of them, from mu in (0, 1)) and the solution of the Vandermonde
    /// system: the coefficients c_n of sum_n c_n mu^(2n) (n < number of nodes) through values at the nodes
    struct MuPolynomial {
        DVector mu, t;
        explicit MuPolynomial(int n = 5) {
            const auto gl = detail::gauss_legendre(2 * n);
            for (int i = n; i < 2 * n; i++) {
                mu.push_back(gl.first[i]);
                t.push_back(gl.first[i] * gl.first[i]);
            }
        }
        /// Coefficients of t^0..t^(n-1) from values at the nodes (Newton divided differences)
        DVector coefficients(const DVector & values) const {
            const size_t n = t.size();
            DVector a = values;
            for (size_t j = 1; j < n; j++)
                for (size_t i = n - 1; i >= j; i--)
                    a[i] = (a[i] - a[i - 1]) / (t[i] - t[i - j]);
            // Newton form -> monomial coefficients
            DVector c(n, 0.0);
            for (size_t j = n; j-- > 0;) {
                // c = c * (t - t_j) + a_j
                DVector nc(n, 0.0);
                for (size_t i = 0; i < n; i++) {
                    if (i + 1 < n)
                        nc[i + 1] += c[i];
                    nc[i] -= c[i] * t[j];
                }
                nc[0] += a[j];
                c = nc;
            }
            return c;
        }
    };

} // namespace FML::PERTURBATION::LOOPS
#endif
