#include <FML/CosmologyFitting/PlanckLikelihoods.h>

#include <gsl/gsl_linalg.h>

#include <cmath>
#include <fstream>
#include <limits>
#include <map>
#include <sstream>
#include <stdexcept>

#ifdef USE_OMP
#include <omp.h>
#endif

namespace FML::COSMOLOGY::FITTING {

    namespace {
        constexpr double NEG_INF = -std::numeric_limits<double>::infinity();

        std::vector<std::vector<double>> read_matrix(const std::string & file) {
            std::ifstream f(file);
            if (!f)
                throw std::runtime_error("cannot open " + file + " (run python3 python/fetch_planck_data.py)");
            std::vector<std::vector<double>> rows;
            std::string line;
            while (std::getline(f, line)) {
                if (line.empty() || line[0] == '#')
                    continue;
                std::istringstream is(line);
                std::vector<double> r;
                double v;
                while (is >> v)
                    r.push_back(v);
                if (!r.empty())
                    rows.push_back(std::move(r));
            }
            return rows;
        }

        // In-place Cholesky factorization A = L L^T of a symmetric positive-definite n x n matrix (row-major, the
        // lower triangle is used and overwritten by L; the upper triangle is zeroed). Right-looking, by blocks,
        // with the trailing update spread over the OpenMP threads.
        void cholesky_lower(std::vector<double> & A, size_t n) {
            const size_t nb = 128;
            for (size_t k0 = 0; k0 < n; k0 += nb) {
                const size_t k1 = std::min(n, k0 + nb);
                // factor the diagonal block and the panel below it, column by column
                for (size_t k = k0; k < k1; k++) {
                    double d = A[k * n + k];
                    for (size_t p = k0; p < k; p++)
                        d -= A[k * n + p] * A[k * n + p];
                    if (!(d > 0.0))
                        throw std::invalid_argument("Cholesky: the matrix is not positive definite");
                    d = std::sqrt(d);
                    A[k * n + k] = d;
#ifdef USE_OMP
#pragma omp parallel for schedule(static)
#endif
                    for (size_t i = k + 1; i < n; i++) {
                        double s = A[i * n + k];
                        for (size_t p = k0; p < k; p++)
                            s -= A[i * n + p] * A[k * n + p];
                        A[i * n + k] = s / d;
                    }
                }
                // trailing update A22 -= L21 L21^T (lower triangle)
#ifdef USE_OMP
#pragma omp parallel for schedule(dynamic, 16)
#endif
                for (size_t i = k1; i < n; i++) {
                    const double * li = &A[i * n + k0];
                    for (size_t j = k1; j <= i; j++) {
                        const double * lj = &A[j * n + k0];
                        double s = 0.0;
                        for (size_t p = 0; p < k1 - k0; p++)
                            s += li[p] * lj[p];
                        A[i * n + j] -= s;
                    }
                }
            }
            for (size_t i = 0; i < n; i++)
                for (size_t j = i + 1; j < n; j++)
                    A[i * n + j] = 0.0;
        }
    } // namespace

    // =============================================================================
    // Not-a-knot cubic spline
    // =============================================================================

    NotAKnotSpline::NotAKnotSpline(std::vector<double> x, std::vector<double> y) : xs(std::move(x)), ys(std::move(y)) {
        const size_t n = xs.size();
        if (n < 4 || ys.size() != n)
            throw std::invalid_argument("NotAKnotSpline: need at least 4 points");
        std::vector<double> h(n - 1);
        for (size_t i = 0; i + 1 < n; i++) {
            h[i] = xs[i + 1] - xs[i];
            if (!(h[i] > 0.0))
                throw std::invalid_argument("NotAKnotSpline: x must be strictly increasing");
        }
        // Second derivatives M: h_{i-1} M_{i-1} + 2 (h_{i-1} + h_i) M_i + h_i M_{i+1} = r_i (i = 1..n-2) with the
        // third derivative continuous at x_1 and x_{n-2}: M_0 = ((h0 + h1) M_1 - h0 M_2)/h1 and the mirror image,
        // substituted into the first and last rows -> tridiagonal in M_1..M_{n-2}
        const size_t m = n - 2;
        std::vector<double> a(m, 0.0), b(m, 0.0), c(m, 0.0), r(m, 0.0);
        for (size_t i = 1; i <= m; i++) {
            const size_t k = i - 1;
            a[k] = h[i - 1];
            b[k] = 2.0 * (h[i - 1] + h[i]);
            c[k] = h[i];
            r[k] = 6.0 * ((ys[i + 1] - ys[i]) / h[i] - (ys[i] - ys[i - 1]) / h[i - 1]);
        }
        const double h0 = h[0], h1 = h[1], hl = h[n - 2], hm = h[n - 3];
        if (m == 2) {
            // n = 4: a single cubic through the four points (both conditions on the same interior)
            // rows: [b0 + h0 (h0+h1)/h1, c0 - h0^2/h1] [M1]   [r0]
            //       [a1 - hl^2/hm, b1 + hl (hl+hm)/hm] [M2] = [r1]
            const double A00 = b[0] + h0 * (h0 + h1) / h1, A01 = c[0] - h0 * h0 / h1;
            const double A10 = a[1] - hl * hl / hm, A11 = b[1] + hl * (hl + hm) / hm;
            const double det = A00 * A11 - A01 * A10;
            M.assign(n, 0.0);
            M[1] = (r[0] * A11 - A01 * r[1]) / det;
            M[2] = (A00 * r[1] - A10 * r[0]) / det;
        } else {
            b[0] += h0 * (h0 + h1) / h1;
            c[0] -= h0 * h0 / h1;
            b[m - 1] += hl * (hl + hm) / hm;
            a[m - 1] -= hl * hl / hm;
            // Thomas algorithm
            for (size_t k = 1; k < m; k++) {
                const double w = a[k] / b[k - 1];
                b[k] -= w * c[k - 1];
                r[k] -= w * r[k - 1];
            }
            M.assign(n, 0.0);
            M[m] = r[m - 1] / b[m - 1];
            for (size_t k = m - 1; k-- > 0;)
                M[k + 1] = (r[k] - c[k] * M[k + 2]) / b[k];
        }
        M[0] = ((h0 + h1) * M[1] - h0 * M[2]) / h1;
        M[n - 1] = ((hl + hm) * M[n - 2] - hl * M[n - 3]) / hm;
    }

    size_t NotAKnotSpline::interval(double x) const {
        if (x <= xs.front())
            return 0;
        if (x >= xs.back())
            return xs.size() - 2;
        size_t lo = 0, hi = xs.size() - 1;
        while (hi - lo > 1) {
            const size_t mid = (lo + hi) / 2;
            (xs[mid] > x ? hi : lo) = mid;
        }
        return lo;
    }

    double NotAKnotSpline::operator()(double x) const {
        const size_t i = interval(x);
        const double h = xs[i + 1] - xs[i], A = (xs[i + 1] - x) / h, B = (x - xs[i]) / h;
        return A * ys[i] + B * ys[i + 1] + ((A * A * A - A) * M[i] + (B * B * B - B) * M[i + 1]) * h * h / 6.0;
    }

    double NotAKnotSpline::derivative(double x) const {
        const size_t i = interval(x);
        const double h = xs[i + 1] - xs[i], A = (xs[i + 1] - x) / h, B = (x - xs[i]) / h;
        return (ys[i + 1] - ys[i]) / h - (3.0 * A * A - 1.0) * h / 6.0 * M[i] + (3.0 * B * B - 1.0) * h / 6.0 * M[i + 1];
    }

    // =============================================================================
    // Low-ell TT (Commander)
    // =============================================================================

    PlanckLowlTTLikelihood::PlanckLowlTTLikelihood(const std::string & dir, int lmin_in, int lmax_in)
        : lmin(lmin_in), lmax(lmax_in) {
        if (lmin < 2 || lmax > 200 || lmin >= lmax)
            throw std::invalid_argument("planck_lowl_TT: need 2 <= lmin < lmax <= 200");
        const size_t n = size_t(lmax - lmin + 1), o = size_t(lmin - 2); // the files start at ell = 2
        const auto cov = read_matrix(dir + "/cov.txt");
        const auto mu_all = read_matrix(dir + "/mu.txt");
        const auto mus_all = read_matrix(dir + "/mu_sigma.txt");
        const auto cl = read_matrix(dir + "/cl2x_1.txt");
        const auto val = read_matrix(dir + "/cl2x_2.txt");
        if (cov.size() < o + n || mu_all.size() < o + n || mus_all.size() < o + n || cl.empty() ||
            cl.size() != val.size() || cl[0].size() < o + n)
            throw std::runtime_error("planck_lowl_TT: unexpected data in " + dir);
        gsl_matrix * C = gsl_matrix_alloc(n, n);
        for (size_t i = 0; i < n; i++)
            for (size_t j = 0; j < n; j++)
                gsl_matrix_set(C, i, j, cov[o + i][o + j]);
        gsl_set_error_handler_off();
        if (gsl_linalg_cholesky_decomp1(C) != 0)
            throw std::runtime_error("planck_lowl_TT: covariance not positive definite");
        gsl_linalg_cholesky_invert(C);
        icov.assign(n, std::vector<double>(n));
        for (size_t i = 0; i < n; i++)
            for (size_t j = 0; j < n; j++)
                icov[i][j] = gsl_matrix_get(C, i, j);
        gsl_matrix_free(C);
        const size_t nbins = cl.size();
        for (size_t i = 0; i < n; i++) {
            mu.push_back(mu_all[o + i][0]);
            std::vector<double> x(nbins), y(nbins);
            for (size_t j = 0; j < nbins; j++) {
                x[j] = cl[j][o + i];
                y[j] = val[j][o + i];
            }
            // the prior range: inside the flat ends (x = -5 and x = 5) of the transformation, two nodes in
            size_t j = 0;
            while (j < nbins && std::abs(y[j] + 5.0) < 1e-4)
                j++;
            lo.push_back(x[std::min(j + 2, nbins - 1)]);
            j = nbins - 1;
            while (j > 0 && std::abs(y[j] - 5.0) < 1e-4)
                j--;
            hi.push_back(x[j >= 2 ? j - 2 : 0]);
            spline.emplace_back(x, y);
        }
        std::vector<double> D(size_t(lmax) + 1, 0.0);
        for (size_t i = 0; i < n; i++)
            D[size_t(lmin) + i] = mus_all[o + i][0];
        offset = 0.0;
        offset = loglike(D, 1.0);
    }

    double PlanckLowlTTLikelihood::loglike(const std::vector<double> & Dtt, double A) const {
        if (int(Dtt.size()) <= lmax)
            throw std::invalid_argument("planck_lowl_TT: spectrum needed up to ell = " + std::to_string(lmax));
        const size_t n = size_t(lmax - lmin + 1);
        const double cal = A * A;
        std::vector<double> d(n);
        double lnL = 0.0;
        for (size_t i = 0; i < n; i++) {
            const double c = Dtt[size_t(lmin) + i] / cal;
            if (c < lo[i] || c > hi[i])
                return NEG_INF;
            const double dx = spline[i].derivative(c);
            if (dx < 0.0)
                return NEG_INF;
            lnL += std::log(dx);
            d[i] = spline[i](c) - mu[i];
        }
        double q = 0.0;
        for (size_t i = 0; i < n; i++)
            for (size_t j = 0; j < n; j++)
                q += d[i] * icov[i][j] * d[j];
        return lnL - 0.5 * q - offset;
    }

    double PlanckLowlTTLikelihood::loglike(const Theory & t) const {
        return loglike(t.cmb().tt, t.nuisance("A_planck", 1.0));
    }

    // =============================================================================
    // Low-ell EE (SimAll)
    // =============================================================================

    PlanckLowlEELikelihood::PlanckLowlEELikelihood(const std::string & dir) {
        table = read_matrix(dir + "/prob_table.txt");
        if (table.empty() || table[0].size() != size_t(lmax - lmin + 1))
            throw std::runtime_error("planck_lowl_EE: unexpected table in " + dir);
    }

    double PlanckLowlEELikelihood::loglike(const std::vector<double> & Dee, double A) const {
        if (int(Dee.size()) <= lmax)
            throw std::invalid_argument("planck_lowl_EE: spectrum needed up to ell = 29");
        double lnL = 0.0;
        for (int l = lmin; l <= lmax; l++) {
            const double x = Dee[size_t(l)] / (A * A * step);
            if (!(x > -1.0) || x >= double(table.size())) // int() truncates towards zero
                return NEG_INF;
            lnL += table[size_t(x)][size_t(l - lmin)];
        }
        return lnL;
    }

    double PlanckLowlEELikelihood::loglike(const Theory & t) const {
        return loglike(t.cmb().ee, t.nuisance("A_planck", 1.0));
    }

    // =============================================================================
    // CamSpec PR4 (NPIPE)
    // =============================================================================

    CamSpecNPIPELikelihood::CamSpecNPIPELikelihood(const std::string & dir,
                                                   const std::vector<std::string> & use,
                                                   std::string name)
        : label(std::move(name)) {
        for (const auto & s : use)
            if (s != "143x143" && s != "217x217" && s != "143x217" && s != "TE" && s != "EE")
                throw std::invalid_argument("planck_camspec_npipe: unknown spectrum " + s);
        // the data ranges (all spectra, in the order of the covariance) and the spectra (row = ell, column = spectrum)
        std::vector<std::string> names;
        std::vector<int> lmn, lmx;
        {
            std::ifstream f(dir + "/like_NPIPE_12.6_unified_data_ranges.txt");
            if (!f)
                throw std::runtime_error("cannot open the CamSpec data in " + dir +
                                         " (run python3 python/fetch_planck_data.py camspec)");
            std::string nm;
            int a, b;
            while (f >> nm >> a >> b) {
                names.push_back(nm);
                lmn.push_back(a);
                lmx.push_back(b);
            }
        }
        const auto spectra = read_matrix(dir + "/like_NPIPE_12.6_unified_spectra.txt");
        size_t nX = 0;
        std::vector<size_t> used_idx;
        for (size_t i = 0; i < names.size(); i++) {
            const size_t n = size_t(lmx[i] - lmn[i] + 1);
            bool used = false;
            for (const auto & s : use)
                used = used || s == names[i];
            if (used) {
                blocks.push_back({names[i], lmn[i], lmx[i]});
                for (int l = lmn[i]; l <= lmx[i]; l++) {
                    used_idx.push_back(nX + size_t(l - lmn[i]));
                    data.push_back(spectra.at(size_t(l)).at(i));
                }
                has_tt = has_tt || names[i].find('x') != std::string::npos;
                has_te = has_te || names[i] == "TE";
                has_ee = has_ee || names[i] == "EE";
            }
            nX += n;
        }
        if (blocks.size() != use.size())
            throw std::runtime_error("planck_camspec_npipe: a requested spectrum is not in the data");
        // covariance: nX x nX float32
        const std::string cf = dir + "/like_NPIPE_12.6_unified_cov.bin";
        std::ifstream f(cf, std::ios::binary);
        if (!f)
            throw std::runtime_error("cannot open " + cf);
        f.seekg(0, std::ios::end);
        if (size_t(f.tellg()) != nX * nX * sizeof(float))
            throw std::runtime_error(cf + ": unexpected size");
        f.seekg(0);
        const size_t n = used_idx.size();
        L.assign(n * n, 0.0);
        std::vector<float> row(nX);
        size_t next = 0; // rows are read in order; keep the used ones
        for (size_t r = 0; r < nX && next < n; r++) {
            f.read(reinterpret_cast<char *>(row.data()), std::streamsize(nX * sizeof(float)));
            if (used_idx[next] != r)
                continue;
            for (size_t j = 0; j <= next; j++)
                L[next * n + j] = double(row[used_idx[j]]);
            next++;
        }
        cholesky_lower(L, n);
    }

    std::vector<std::string> CamSpecNPIPELikelihood::nuisance_parameters() const {
        std::vector<std::string> p{"A_planck"};
        for (const auto & b : blocks) {
            if (b.name == "143x143")
                p.insert(p.end(), {"amp_143", "n_143"});
            if (b.name == "217x217")
                p.insert(p.end(), {"amp_217", "n_217"});
            if (b.name == "143x217")
                p.insert(p.end(), {"amp_143x217", "n_143x217"});
            if (b.name == "TE")
                p.push_back("calTE");
            if (b.name == "EE")
                p.push_back("calEE");
        }
        return p;
    }

    double CamSpecNPIPELikelihood::chi2(const std::vector<double> & Dtt,
                                        const std::vector<double> & Dte,
                                        const std::vector<double> & Dee,
                                        const Nuisance & p) const {
        const double cP = p.A_planck * p.A_planck;
        std::vector<double> d = data;
        size_t ix = 0;
        for (const auto & b : blocks) {
            const std::vector<double> * D = nullptr;
            double cal = cP, amp = 0.0, tilt = 1.0;
            if (b.name == "143x143") {
                D = &Dtt, amp = p.amp_143, tilt = p.n_143;
            } else if (b.name == "217x217") {
                D = &Dtt, amp = p.amp_217, tilt = p.n_217;
            } else if (b.name == "143x217") {
                D = &Dtt, amp = p.amp_143x217, tilt = p.n_143x217;
            } else if (b.name == "TE") {
                D = &Dte, cal *= p.calTE;
            } else {
                D = &Dee, cal *= p.calEE;
            }
            if (int(D->size()) <= b.lmax)
                throw std::invalid_argument(label + ": spectra needed up to ell = 2500");
            const bool tt = b.name.find('x') != std::string::npos;
            for (int l = b.lmin; l <= b.lmax; l++, ix++) {
                const double fg = tt ? amp * std::pow(l / 1500.0, tilt) : 0.0;
                d[ix] -= ((*D)[size_t(l)] + fg) / cal;
            }
        }
        // chi^2 = |L^-1 d|^2
        const size_t n = d.size();
        double c2 = 0.0;
        for (size_t i = 0; i < n; i++) {
            const double * li = &L[i * n];
            double s = d[i];
            for (size_t j = 0; j < i; j++)
                s -= li[j] * d[j];
            d[i] = s / li[i];
            c2 += d[i] * d[i];
        }
        return c2;
    }

    double CamSpecNPIPELikelihood::loglike(const Theory & t) const {
        const auto & c = t.cmb();
        Nuisance p;
        p.A_planck = t.nuisance("A_planck", p.A_planck);
        p.calTE = t.nuisance("calTE", p.calTE);
        p.calEE = t.nuisance("calEE", p.calEE);
        p.amp_143 = t.nuisance("amp_143", p.amp_143);
        p.amp_217 = t.nuisance("amp_217", p.amp_217);
        p.amp_143x217 = t.nuisance("amp_143x217", p.amp_143x217);
        p.n_143 = t.nuisance("n_143", p.n_143);
        p.n_217 = t.nuisance("n_217", p.n_217);
        p.n_143x217 = t.nuisance("n_143x217", p.n_143x217);
        return -0.5 * chi2(c.tt, c.te, c.ee, p);
    }

    // =============================================================================
    // ACT DR6 lensing
    // =============================================================================

    ACTDR6LensingLikelihood::ACTDR6LensingLikelihood(const std::string & dir,
                                                     const std::string & variant,
                                                     std::string name)
        : label(std::move(name)) {
        const std::string stem = dir + "/" + variant;
        auto vec = [&](const std::string & suffix) {
            std::vector<double> v;
            for (const auto & r : read_matrix(stem + suffix))
                v.insert(v.end(), r.begin(), r.end());
            return v;
        };
        {
            std::ifstream f(stem + "_data.txt");
            if (!f)
                throw std::runtime_error("cannot open " + stem + "_data.txt (python3 python/convert_act_dr6_lensing.py)");
        }
        data = vec("_data.txt");
        offset = vec("_offset.txt");
        cinv = read_matrix(stem + "_cinv.txt");
        m_kk = read_matrix(stem + "_kk.txt");
        m_tt = read_matrix(stem + "_tt.txt");
        m_ee = read_matrix(stem + "_ee.txt");
        m_bb = read_matrix(stem + "_bb.txt");
        m_te = read_matrix(stem + "_te.txt");
        const size_t nb = data.size();
        bool ok = offset.size() == nb && cinv.size() == nb;
        for (const auto * m : {&cinv, &m_kk, &m_tt, &m_ee, &m_bb, &m_te})
            ok = ok && m->size() == nb;
        const size_t nL = nb ? m_kk[0].size() : 0;
        for (const auto * m : {&m_kk, &m_tt, &m_ee, &m_bb, &m_te})
            for (const auto & r : *m)
                ok = ok && r.size() == nL;
        for (const auto & r : cinv)
            ok = ok && r.size() == nb;
        if (!ok || nb == 0 || nL < 3)
            throw std::runtime_error("act_dr6_lensing: inconsistent data in " + stem + "_*.txt");
        lmax = int(nL) - 1;
    }

    double ACTDR6LensingLikelihood::loglike_cl(const std::vector<double> & tt,
                                               const std::vector<double> & ee,
                                               const std::vector<double> & bb,
                                               const std::vector<double> & te,
                                               const std::vector<double> & kk) const {
        for (const auto * v : {&tt, &ee, &bb, &te, &kk})
            if (int(v->size()) <= lmax)
                throw std::invalid_argument(label + ": spectra needed up to L = " + std::to_string(lmax));
        const size_t nb = data.size(), nL = size_t(lmax) + 1;
        std::vector<double> d(nb);
        for (size_t b = 0; b < nb; b++) {
            double m = offset[b];
            for (size_t L = 0; L < nL; L++)
                m += m_kk[b][L] * kk[L] + m_tt[b][L] * tt[L] + m_ee[b][L] * ee[L] + m_bb[b][L] * bb[L] +
                     m_te[b][L] * te[L];
            d[b] = data[b] - m;
        }
        double c2 = 0.0;
        for (size_t i = 0; i < nb; i++)
            for (size_t j = 0; j < nb; j++)
                c2 += d[i] * cinv[i][j] * d[j];
        return -0.5 * c2;
    }

    double ACTDR6LensingLikelihood::loglike(const Theory & t) const {
        // Theory: D_ell = l(l+1) C_l / 2pi [muK^2] and pp = [L(L+1)]^2 C_L^phiphi / 2pi
        const auto & c = t.cmb();
        const size_t nL = size_t(lmax) + 1;
        std::vector<double> tt(nL, 0.0), ee(nL, 0.0), bb(nL, 0.0), te(nL, 0.0), kk(nL, 0.0);
        for (const auto * v : {&c.tt, &c.ee, &c.bb, &c.te, &c.pp})
            if (v->size() < nL)
                throw std::invalid_argument(label + ": spectra needed up to L = " + std::to_string(lmax));
        for (size_t L = 2; L < nL; L++) {
            const double f = 2.0 * M_PI / (L * (L + 1.0));
            tt[L] = c.tt[L] * f;
            ee[L] = c.ee[L] * f;
            bb[L] = c.bb[L] * f;
            te[L] = c.te[L] * f;
            kk[L] = c.pp[L] * 2.0 * M_PI / 4.0;
        }
        return loglike_cl(tt, ee, bb, te, kk);
    }

} // namespace FML::COSMOLOGY::FITTING
