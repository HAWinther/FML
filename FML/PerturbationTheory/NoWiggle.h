#ifndef FML_PERTURBATIONTHEORY_NOWIGGLE_H
#define FML_PERTURBATIONTHEORY_NOWIGGLE_H

#include <FML/Cosmology/Spectra/EisensteinHu.h>
#include <FML/PerturbationTheory/LagrangianQFunctions.h>

#include <algorithm>
#include <cmath>
#include <fftw3.h>
#include <functional>
#include <stdexcept>
#include <utility>
#include <vector>

//=====================================================================================================================
///
/// Wiggle/no-wiggle splits of a linear power spectrum: the smooth (no-wiggle) P_nw(k) with the baryon acoustic
/// oscillations removed, P_w = P - P_nw. For IR resummation (EPTRedshiftSpace.h), BAO templates and damping models.
///
///  - nowiggle_dst: the discrete sine transform method of Hamann et al. 2010 (arXiv:1003.3999) in the form of
///    Wallisch 2018 (arXiv:1810.02800; velocileptors' Utils/pnw_dst.py, reproduced to 1e-13): the BAO bump of the DST
///    of ln(k P(k)) on a linear k-grid is cut out and interpolated over. Needs no cosmology. Smooth spectra are
///    returned to < 1e-4, and it agrees with the Gaussian-EH method to < 1% for k >= 0.15 h/Mpc, but the first
///    wiggle (k ~ 0.005 - 0.05) is uncertain at the few % level (it depends on the cut; ~5% from Gaussian-EH there).
///    The bump is found from the extrema of the smoothed second derivative of the DST; unlike velocileptors, shallow
///    extrema (< 5% of the largest swing) are ignored, as velocileptors' choice depends on the sampling of P
///    (3% changes when the same P is resampled; here < 0.5%).
///  - nowiggle_savgol: a Savitzky-Golay filter (polynomial order 4, a window of ln 7 in ln k) of P on a log grid
///    (velocileptors' REPT default). Leaves ~2% wiggles at k ~ 0.1 - 0.3 h/Mpc and biases a smooth spectrum by up
///    to 0.5% (k >~ 0.5 h/Mpc, the window spans curvature).
///  - nowiggle_gaussian: a Gaussian smoothing (sigma = 0.25 in ln k) of P / P_ref with a smooth reference shape
///    P_ref, by default the Eisenstein & Hu no-wiggle fit (Vlah, Seljak, Chu & Feng 2016, arXiv:1509.02120; as in
///    HMcode-2020). The cleanest of the three when P_ref is close to the broadband (the residual wiggles of a CAMB
///    P are ~5e-4), biased by up to ~1% for a poor reference shape (Omega_m off by 0.06).
///  - The Eisenstein & Hu (1998) transfer functions with and without the wiggles are those of
///    Cosmology/Spectra/EisensteinHu.h (also available here).
///
/// Units: k in h/Mpc, P in (Mpc/h)^3. The accuracy of the methods is in tests/test_nowiggle.cpp.
///
//=====================================================================================================================
namespace FML::PERTURBATION {

    namespace detail {

        /// Savitzky-Golay smoothing of uniformly sampled y (scipy.signal.savgol_filter(y, window, order) with the
        /// default mode 'interp': the ends are the polynomial fit to the first/last window points)
        inline DVector savgol_filter(const DVector & y, int window, int order) {
            const int n = int(y.size());
            if (window % 2 != 1 or window <= order or window > n)
                throw std::invalid_argument("[savgol_filter] the window must be odd, > order and <= the data size");
            const int h = window / 2, np = order + 1;
            // Weights w_j such that sum_j w_j y_j is the least-squares polynomial (in t_j = (j - h) / h) at t
            auto weights = [&](double t) {
                std::vector<DVector> M(np, DVector(np, 0.0));
                std::vector<DVector> At(np, DVector(window));
                for (int j = 0; j < window; j++) {
                    const double tj = (j - h) / double(h);
                    double v = 1.0;
                    for (int a = 0; a < np; a++, v *= tj)
                        At[a][j] = v;
                }
                for (int a = 0; a < np; a++)
                    for (int b = 0; b < np; b++)
                        for (int j = 0; j < window; j++)
                            M[a][b] += At[a][j] * At[b][j];
                // Solve M c = (1, t, t^2, ...) (symmetric positive definite: Gaussian elimination)
                DVector rhs(np);
                double v = 1.0;
                for (int a = 0; a < np; a++, v *= t)
                    rhs[a] = v;
                for (int a = 0; a < np; a++) {
                    for (int b = a + 1; b < np; b++) {
                        const double r = M[b][a] / M[a][a];
                        for (int c = a; c < np; c++)
                            M[b][c] -= r * M[a][c];
                        rhs[b] -= r * rhs[a];
                    }
                }
                DVector c(np);
                for (int a = np - 1; a >= 0; a--) {
                    double s = rhs[a];
                    for (int b = a + 1; b < np; b++)
                        s -= M[a][b] * c[b];
                    c[a] = s / M[a][a];
                }
                DVector w(window, 0.0);
                for (int j = 0; j < window; j++)
                    for (int a = 0; a < np; a++)
                        w[j] += c[a] * At[a][j];
                return w;
            };
            DVector out(n, 0.0);
            const DVector wc = weights(0.0);
            for (int i = h; i < n - h; i++) {
                double s = 0.0;
                for (int j = 0; j < window; j++)
                    s += wc[j] * y[i - h + j];
                out[i] = s;
            }
            for (int i = 0; i < h; i++) {
                const DVector wl = weights((i - h) / double(h));
                const DVector wr = weights((window - h + i - h) / double(h));
                double sl = 0.0, sr = 0.0;
                for (int j = 0; j < window; j++) {
                    sl += wl[j] * y[j];
                    sr += wr[j] * y[n - window + j];
                }
                out[i] = sl;
                out[n - h + i] = sr;
            }
            return out;
        }

        /// A cubic spline with not-a-knot end conditions, extrapolated with the end cubics
        /// (scipy.interpolate.interp1d(kind='cubic', fill_value='extrapolate')). The knots are fixed; any number of
        /// data sets can be interpolated with the same factorization.
        class NotAKnotSpline {
          public:
            NotAKnotSpline() = default;
            explicit NotAKnotSpline(const DVector & x_) : x(x_) {
                n = int(x.size());
                if (n < 4)
                    throw std::invalid_argument("[NotAKnotSpline] need >= 4 knots");
                hs.resize(n - 1);
                for (int i = 0; i < n - 1; i++) {
                    hs[i] = x[i + 1] - x[i];
                    if (not(hs[i] > 0))
                        throw std::invalid_argument("[NotAKnotSpline] the knots must increase");
                }
                // Tridiagonal system for M_1 ... M_(n-2) after eliminating M_0 and M_(n-1) with the not-a-knot
                // conditions h1 (M1 - M0) = h0 (M2 - M1) and the same at the end
                const int m = n - 2;
                a.assign(m, 0.0);
                b.assign(m, 0.0);
                c.assign(m, 0.0);
                for (int r = 0; r < m; r++) {
                    const int i = r + 1;
                    a[r] = hs[i - 1];
                    b[r] = 2 * (hs[i - 1] + hs[i]);
                    c[r] = hs[i];
                }
                const double h0 = hs[0], h1 = hs[1];
                b[0] += h0 * (h0 + h1) / h1;
                c[0] -= h0 * h0 / h1;
                const double hm = hs[n - 2], hm1 = hs[n - 3];
                b[m - 1] += hm * (hm + hm1) / hm1;
                a[m - 1] -= hm * hm / hm1;
                // Thomas factorization
                cp.assign(m, 0.0);
                den.assign(m, 0.0);
                for (int r = 0; r < m; r++) {
                    den[r] = b[r] - (r > 0 ? a[r] * cp[r - 1] : 0.0);
                    cp[r] = c[r] / den[r];
                }
            }

            /// The spline of y at the points xe
            DVector operator()(const DVector & y, const DVector & xe) const {
                if (int(y.size()) != n)
                    throw std::invalid_argument("[NotAKnotSpline] y has the wrong size");
                const int m = n - 2;
                DVector d(m), M(n);
                for (int r = 0; r < m; r++) {
                    const int i = r + 1;
                    d[r] = 6 * ((y[i + 1] - y[i]) / hs[i] - (y[i] - y[i - 1]) / hs[i - 1]);
                }
                DVector dp(m);
                for (int r = 0; r < m; r++)
                    dp[r] = (d[r] - (r > 0 ? a[r] * dp[r - 1] : 0.0)) / den[r];
                for (int r = m - 1; r >= 0; r--)
                    M[r + 1] = dp[r] - (r < m - 1 ? cp[r] * M[r + 2] : 0.0);
                M[0] = ((hs[0] + hs[1]) * M[1] - hs[0] * M[2]) / hs[1];
                M[n - 1] = ((hs[n - 2] + hs[n - 3]) * M[n - 2] - hs[n - 2] * M[n - 3]) / hs[n - 3];
                DVector out(xe.size());
                for (size_t e = 0; e < xe.size(); e++) {
                    const double xx = xe[e];
                    int i = int(std::upper_bound(x.begin(), x.end(), xx) - x.begin()) - 1;
                    i = std::clamp(i, 0, n - 2);
                    const double h = hs[i], A = x[i + 1] - xx, B = xx - x[i];
                    out[e] = M[i] * A * A * A / (6 * h) + M[i + 1] * B * B * B / (6 * h) + (y[i] / h - M[i] * h / 6) * A +
                             (y[i + 1] / h - M[i + 1] * h / 6) * B;
                }
                return out;
            }

          private:
            DVector x, hs, a, b, c, cp, den;
            int n{0};
        };

        /// The odd Savitzky-Golay window spanning about width in ln k on the log grid kg (velocileptors' choice)
        inline int savgol_window(const DVector & kg, double width) {
            const size_t n = kg.size();
            return int(std::ceil(width / std::log(kg[n - 1] / kg[n - 2]))) / 2 * 2 + 1;
        }
    } // namespace detail

    // The Eisenstein & Hu (1998) fits (Cosmology/Spectra/EisensteinHu.h)
    using FML::COSMOLOGY::eisenstein_hu_power_shape;
    using FML::COSMOLOGY::eisenstein_hu_power_shape_nowiggle;
    using FML::COSMOLOGY::eisenstein_hu_transfer;
    using FML::COSMOLOGY::eisenstein_hu_transfer_nowiggle;
    using FML::COSMOLOGY::EisensteinHuParameters;

    //=================================================================================================================
    // No-wiggle spectra. Input: P on any increasing grid k (extrapolated as power laws where needed), output on k.
    //=================================================================================================================

    /// Options of the discrete sine transform method (velocileptors' pnw_dst defaults)
    struct NoWiggleDSTOptions {
        double kmin{1e-3};  // the linear k-grid of the transform (h/Mpc); outside it P_nw = P
        double kmax{10.0};
        int log2N{16};      // 2^log2N points
        int ii_l{-1};       // > 0 (with ii_r): the BAO bump is the DST indices ii_l < i < ii_r (even and odd);
        int ii_r{-1};       // otherwise found from the extrema of the smoothed second derivative
    };

    /// The no-wiggle P(k) by the discrete sine transform method (Hamann et al. 2010, Wallisch 2018). A port of
    /// velocileptors' Utils/pnw_dst.py (reproduced; it sets P_nw = 0 outside [kmin, kmax], here P_nw = P)
    inline std::vector<double> nowiggle_dst(const std::vector<double> & k,
                                            const std::vector<double> & P,
                                            const NoWiggleDSTOptions & opt = {}) {
        using DVector = std::vector<double>;
        if (k.size() != P.size() or k.size() < 4)
            throw std::invalid_argument("[nowiggle_dst] k and P must have the same size (>= 4)");
        if (not(opt.kmin > 0) or not(opt.kmax > opt.kmin) or opt.log2N < 8 or opt.log2N > 24)
            throw std::invalid_argument("[nowiggle_dst] invalid options");
        if (opt.kmin < k.front() or opt.kmax > k.back())
            throw std::invalid_argument("[nowiggle_dst] [kmin, kmax] must be inside the input k-range");
        const int n = 1 << opt.log2N, nh = n / 2;
        DVector ks(n), lnkp(k.size());
        for (int i = 0; i < n; i++)
            ks[i] = opt.kmin + (opt.kmax - opt.kmin) * i / double(n - 1);
        for (size_t i = 0; i < k.size(); i++)
            lnkp[i] = std::log(k[i] * P[i]);
        DVector lnps = detail::NotAKnotSpline(k)(lnkp, ks);

        // DST-II (scipy.fftpack.dst type 2 = FFTW RODFT10)
        DVector dst(n);
        {
            fftw_plan plan = fftw_plan_r2r_1d(n, lnps.data(), dst.data(), FFTW_RODFT10, FFTW_ESTIMATE);
            fftw_execute(plan);
            fftw_destroy_plan(plan);
        }
        DVector odd(nh), even(nh);
        for (int i = 0; i < nh; i++) {
            even[i] = dst[2 * i];
            odd[i] = dst[2 * i + 1];
        }

        // The BAO bump: from the first minimum and second maximum of the smoothed second derivative
        auto gradient = [](const DVector & f) {
            const size_t m = f.size();
            DVector g(m);
            g[0] = f[1] - f[0];
            g[m - 1] = f[m - 1] - f[m - 2];
            for (size_t i = 1; i + 1 < m; i++)
                g[i] = 0.5 * (f[i + 1] - f[i - 1]);
            return g;
        };
        auto gaussian_filter = [](const DVector & f, double sigma) {
            // scipy.ndimage.gaussian_filter1d (truncate = 4, mode = 'reflect')
            const int m = int(f.size()), r = int(4 * sigma + 0.5);
            DVector w(2 * r + 1);
            double norm = 0;
            for (int j = -r; j <= r; j++)
                norm += (w[j + r] = std::exp(-0.5 * j * j / (sigma * sigma)));
            DVector g(m, 0.0);
            for (int i = 0; i < m; i++)
                for (int j = -r; j <= r; j++) {
                    int idx = i + j;
                    while (idx < 0 or idx >= m)
                        idx = idx < 0 ? -idx - 1 : 2 * m - idx - 1;
                    g[i] += w[j + r] / norm * f[idx];
                }
            return g;
        };
        // The local extrema of the smoothed second derivative (scipy.signal.argrelmin/argrelmax), with adjacent
        // min-max pairs shallower than 5% of the largest swing merged away (velocileptors takes them all, so a
        // tiny dip on a plateau before the bump, which depends on the sampling of P, can be taken for its edge)
        auto bump = [&](const DVector & f, int & il, int & ir) {
            const DVector d2 = gaussian_filter(gradient(gradient(f)), 4.0);
            std::vector<std::pair<int, bool>> ext; // (index, is a maximum)
            for (size_t i = 1; i + 1 < d2.size(); i++) {
                if (d2[i] < d2[i - 1] and d2[i] < d2[i + 1])
                    ext.push_back({int(i), false});
                if (d2[i] > d2[i - 1] and d2[i] > d2[i + 1])
                    ext.push_back({int(i), true});
            }
            double swing = 0;
            for (size_t e = 1; e < ext.size(); e++)
                swing = std::max(swing, std::abs(d2[ext[e].first] - d2[ext[e - 1].first]));
            bool merged = true;
            while (merged) {
                merged = false;
                for (size_t e = 1; e < ext.size(); e++)
                    if (std::abs(d2[ext[e].first] - d2[ext[e - 1].first]) < 0.05 * swing) {
                        ext.erase(ext.begin() + long(e) - 1, ext.begin() + long(e) + 1);
                        merged = true;
                        break;
                    }
            }
            il = ir = -1;
            int nmax = 0;
            for (const auto & e : ext) {
                if (not e.second and il < 0)
                    il = e.first;
                if (e.second and ++nmax == 2)
                    ir = e.first;
            }
            if (il < 0 or ir < 0)
                throw std::runtime_error("[nowiggle_dst] could not find the BAO bump in the DST (set ii_l, ii_r)");
        };
        int lo_e, hi_e, lo_o, hi_o;
        if (opt.ii_l > 0 and opt.ii_r > opt.ii_l) {
            lo_e = lo_o = opt.ii_l;
            hi_e = hi_o = opt.ii_r;
        } else {
            // velocileptors: (ii_l - 3, ii_r + 10) for the even and (ii_l - 3, ii_r + 20) for the odd coefficients
            int il, ir;
            bump(even, il, ir);
            lo_e = il - 3;
            hi_e = ir + 10;
            bump(odd, il, ir);
            lo_o = il - 3;
            hi_o = ir + 20;
        }
        // Interpolate i^2 DST_i over the bump (not-a-knot cubic, scipy interp1d(kind='cubic'))
        auto fill = [&](const DVector & f, int lo, int hi) {
            DVector x, y, iis(nh);
            for (int i = 0; i < nh; i++) {
                iis[i] = i;
                if (not(i > lo and i < hi)) {
                    x.push_back(i);
                    y.push_back(double(i) * i * f[i]);
                }
            }
            DVector g = detail::NotAKnotSpline(x)(y, iis);
            for (int i = 1; i < nh; i++)
                g[i] /= double(i) * i;
            g[0] = f[0];
            return g;
        };
        const DVector ie = fill(even, lo_e, hi_e), io = fill(odd, lo_o, hi_o);
        DVector in(n), lnps_nw(n);
        for (int i = 0; i < nh; i++) {
            in[2 * i] = ie[i];
            in[2 * i + 1] = io[i];
        }
        // DST-III (scipy.fftpack.idst type 2 = FFTW RODFT01), normalized by 2n
        {
            fftw_plan plan = fftw_plan_r2r_1d(n, in.data(), lnps_nw.data(), FFTW_RODFT01, FFTW_ESTIMATE);
            fftw_execute(plan);
            fftw_destroy_plan(plan);
        }
        DVector pnw_lin(n);
        for (int i = 0; i < n; i++)
            pnw_lin[i] = std::exp(lnps_nw[i] / (2.0 * n)) / ks[i];
        const DVector inside = detail::NotAKnotSpline(ks)(pnw_lin, k);
        DVector out(k.size());
        for (size_t i = 0; i < k.size(); i++)
            out[i] = (k[i] >= opt.kmin and k[i] <= opt.kmax) ? inside[i] : P[i];
        return out;
    }

    /// Options of the Savitzky-Golay method (velocileptors' REPT default with EPTRedshiftSpace's grid)
    struct NoWiggleSavGolOptions {
        double log10_kmin{-5.0}; // the log grid of the filter (h/Mpc)
        double log10_kmax{3.0};
        int N{6000};
        double width{1.9459101090932196}; // the window in ln k (ln 7)
        int order{4};
    };

    /// The no-wiggle P(k) by a Savitzky-Golay filter in ln k
    inline std::vector<double> nowiggle_savgol(const std::vector<double> & k,
                                               const std::vector<double> & P,
                                               const NoWiggleSavGolOptions & opt = {}) {
        using DVector = std::vector<double>;
        if (k.size() != P.size() or k.size() < 4)
            throw std::invalid_argument("[nowiggle_savgol] k and P must have the same size (>= 4)");
        const DVector kg = detail::logspace(opt.log10_kmin, opt.log10_kmax, opt.N);
        const detail::LogInterp Pk(k, P);
        DVector pg(opt.N);
        for (int i = 0; i < opt.N; i++)
            pg[i] = Pk(kg[i]);
        const int window = detail::savgol_window(kg, opt.width);
        const detail::LogInterp Pnw(kg, detail::savgol_filter(pg, window, opt.order));
        DVector out(k.size());
        for (size_t i = 0; i < k.size(); i++)
            out[i] = Pnw(k[i]);
        return out;
    }

    /// The no-wiggle P(k) by a Gaussian smoothing (standard deviation sigma_lnk in ln k) of P / P_ref, P_ref a
    /// smooth shape (its normalization is irrelevant). Evaluated on a log grid with dlnk = sigma_lnk / 20, the
    /// kernel truncated at 4 sigma and renormalized near the ends of the input k-range
    inline std::vector<double> nowiggle_gaussian(const std::vector<double> & k,
                                                 const std::vector<double> & P,
                                                 const std::function<double(double)> & Pref,
                                                 double sigma_lnk = 0.25) {
        using DVector = std::vector<double>;
        if (k.size() != P.size() or k.size() < 4 or not(sigma_lnk > 0))
            throw std::invalid_argument("[nowiggle_gaussian] k and P must have the same size (>= 4), sigma > 0");
        const double dlnk = sigma_lnk / 20, lnk0 = std::log(k.front()), lnk1 = std::log(k.back());
        const int n = int(std::ceil((lnk1 - lnk0) / dlnk)) + 1;
        const detail::LogInterp Pk(k, P);
        DVector lnkg(n), r(n);
        for (int i = 0; i < n; i++) {
            lnkg[i] = std::min(lnk0 + i * dlnk, lnk1);
            const double kk = std::exp(lnkg[i]);
            r[i] = Pk(kk) / Pref(kk);
        }
        const int rad = int(std::ceil(4 * sigma_lnk / dlnk));
        DVector rs(n);
        for (int i = 0; i < n; i++) {
            double sum = 0, norm = 0;
            for (int j = std::max(0, i - rad); j <= std::min(n - 1, i + rad); j++) {
                const double x = (lnkg[j] - lnkg[i]) / sigma_lnk, w = std::exp(-0.5 * x * x);
                sum += w * r[j];
                norm += w;
            }
            rs[i] = sum / norm;
        }
        const detail::NaturalCubicSpline spline(lnkg, rs);
        DVector out(k.size());
        for (size_t i = 0; i < k.size(); i++)
            out[i] = Pref(k[i]) * spline(std::log(k[i]));
        return out;
    }

    /// As above with the Eisenstein & Hu no-wiggle shape k^n_s T_nw(k)^2 as the reference
    inline std::vector<double> nowiggle_gaussian(const std::vector<double> & k,
                                                 const std::vector<double> & P,
                                                 const EisensteinHuParameters & eh,
                                                 double sigma_lnk = 0.25) {
        return nowiggle_gaussian(
            k, P, [&](double kk) { return eisenstein_hu_power_shape_nowiggle(kk, eh); },
            sigma_lnk);
    }

} // namespace FML::PERTURBATION

#endif
