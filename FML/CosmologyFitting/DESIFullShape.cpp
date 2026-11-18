#include <FML/CosmologyFitting/DESIFullShape.h>
#include <FML/PerturbationTheory/EPTRedshiftSpace.h>

#include <gsl/gsl_linalg.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <mutex>
#include <sstream>
#include <stdexcept>

namespace FML::COSMOLOGY::FITTING {

    namespace {
        const std::vector<std::string> all_samples{"BGS_z0", "LRG_z0", "LRG_z1", "LRG_z2", "ELG_z1", "QSO_z0"};

        // The marginalized linear parameters (alpha0p, alpha2p, sn0p, sn2p) and their Gaussian prior widths
        constexpr int nlin = 4;
        constexpr double lin_sigma[nlin] = {12.5, 12.5, 2.0, 5.0};

        // FFTW planning (in the REPT constructor) is not thread safe
        std::mutex fftw_plan_mutex;

        std::vector<std::vector<double>> read_matrix(const std::string & file) {
            std::ifstream f(file);
            if (!f)
                throw std::runtime_error("cannot open " + file + " (python3 python/convert_desi_dr1_fullshape.py)");
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

        // DESI's satellite fraction and velocity dispersion [Mpc/h] of the stochastic k^2 mu^2 term (Maus, Zhao)
        std::pair<double, double> fsat_sigv(const std::string & sample) {
            const std::string t = sample.substr(0, 3);
            if (t == "BGS")
                return {0.15, 150.0 * std::cbrt(10.0) * std::sqrt(1.2) / 70.0};
            if (t == "LRG")
                return {0.15, 150.0 * std::cbrt(10.0) * std::sqrt(1.8) / 70.0};
            if (t == "ELG")
                return {0.10, 150.0 * std::sqrt(2.1) / 70.0};
            return {0.03, 150.0 * std::pow(10.0, 0.7 / 3.0) * std::sqrt(2.4) / 70.0};
        }

        using DVector = std::vector<double>;
        // y(x) -> y(xn) by the not-a-knot cubic spline (scipy's interp1d kind='cubic'; extrapolated), optionally in
        // log10-log10
        DVector interp(const DVector & x, const DVector & y, const DVector & xn, bool loglog) {
            if (!loglog)
                return PERTURBATION::detail::NotAKnotSpline(x)(y, xn);
            DVector lx(x.size()), ly(y.size()), lxn(xn.size());
            for (size_t i = 0; i < x.size(); i++) {
                lx[i] = std::log10(x[i]);
                ly[i] = std::log10(y[i]);
            }
            for (size_t i = 0; i < xn.size(); i++)
                lxn[i] = std::log10(xn[i]);
            DVector out = PERTURBATION::detail::NotAKnotSpline(lx)(ly, lxn);
            for (auto & v : out)
                v = std::pow(10.0, v);
            return out;
        }
    } // namespace

    struct DESIDR1FullShapeLikelihood::Tables {
        // [sample][ell index][k_theory] -> the 19 REPT columns
        std::vector<std::array<std::vector<PERTURBATION::EPTRedshiftSpace::Columns>, 3>> poles;
    };

    DESIDR1FullShapeLikelihood::DESIDR1FullShapeLikelihood(const std::string & dir,
                                                           std::vector<std::string> samples,
                                                           std::string name,
                                                           bool marginalize)
        : label(std::move(name)), marg(marginalize) {
        for (const auto & s : samples)
            if (std::find(all_samples.begin(), all_samples.end(), s) == all_samples.end())
                throw std::invalid_argument("desi_dr1_fs: unknown sample " + s);
        for (const auto & s : all_samples) // DESI's order (the REPT is built at the first redshift)
            if (std::find(samples.begin(), samples.end(), s) != samples.end())
                names.push_back(s);
        if (names.empty())
            throw std::invalid_argument("desi_dr1_fs: no samples");
        for (const auto & s : names) {
            const std::string stem = dir + "/" + s;
            const auto meta = read_matrix(stem + "_meta.txt");
            if (meta.size() != 1 || meta[0].size() != 3)
                throw std::runtime_error("desi_dr1_fs: bad " + stem + "_meta.txt");
            zeff.push_back(meta[0][0]);
            shotnoise.push_back(meta[0][1]);
            const size_t n = size_t(meta[0][2]);
            std::vector<double> d;
            for (const auto & r : read_matrix(stem + "_data.txt"))
                d.push_back(r.at(2));
            const auto k = read_matrix(stem + "_ktheory.txt");
            std::vector<double> kk;
            for (const auto & r : k)
                kk.push_back(r.at(0));
            if (kth.empty())
                kth = kk;
            else if (kk.size() != kth.size() || !std::equal(kk.begin(), kk.end(), kth.begin(), [](double a, double b) {
                         return std::abs(a - b) < 1e-12;
                     }))
                throw std::runtime_error("desi_dr1_fs: the theory k differ between the samples");
            auto W = read_matrix(stem + "_window.txt");
            const auto C = read_matrix(stem + "_cov.txt");
            bool ok = d.size() == n && W.size() == n && C.size() == n;
            for (const auto & r : W)
                ok = ok && r.size() == 3 * kth.size();
            for (const auto & r : C)
                ok = ok && r.size() == n;
            if (!ok)
                throw std::runtime_error("desi_dr1_fs: inconsistent data for " + s);
            gsl_matrix * A = gsl_matrix_alloc(n, n);
            for (size_t i = 0; i < n; i++)
                for (size_t j = 0; j < n; j++)
                    gsl_matrix_set(A, i, j, 0.5 * (C[i][j] + C[j][i]));
            gsl_set_error_handler_off();
            if (gsl_linalg_cholesky_decomp1(A) != 0)
                throw std::runtime_error("desi_dr1_fs: the covariance of " + s + " is not positive definite");
            gsl_linalg_cholesky_invert(A);
            std::vector<std::vector<double>> P(n, std::vector<double>(n));
            for (size_t i = 0; i < n; i++)
                for (size_t j = 0; j < n; j++)
                    P[i][j] = gsl_matrix_get(A, i, j);
            gsl_matrix_free(A);
            data.push_back(d);
            window.push_back(std::move(W));
            icov.push_back(std::move(P));
        }
    }

    int DESIDR1FullShapeLikelihood::n_data() const {
        int n = 0;
        for (const auto & d : data)
            n += int(d.size());
        return n;
    }

    std::vector<std::string> DESIDR1FullShapeLikelihood::nuisance_parameters() const {
        std::vector<std::string> p;
        for (const auto & s : names)
            for (const char * b : {"_b1p", "_b2p", "_bsp"})
                p.push_back("desi_" + s + b);
        return p;
    }

    void DESIDR1FullShapeLikelihood::setup_fiducial(const Theory & t) const {
        TheoryOptions o;
        o.cmb = false;
        o.pk_z = {0.0};
        const RunSettings fs = DESIShapeFitLikelihood::fiducial_settings(t.settings());
        Theory tf(fs, o);
        fid.hrd = fs.cosmo.h * tf.r_drag();
        for (double z : zeff) {
            fid.dmh.push_back(tf.D_M(z) * fs.cosmo.h);
            fid.dhh.push_back(tf.D_H(z) * fs.cosmo.h);
        }
        fid.filter = PeakAverageBAOFilter(tf);
    }

    DESIDR1FullShapeLikelihood::LinearInputs DESIDR1FullShapeLikelihood::linear_inputs(const Theory & t) const {
        std::call_once(fid_once, [&] { setup_fiducial(t); });
        LinearInputs in;
        const double h = t.settings().cosmo.h;
        const double s = h * t.r_drag() / fid.hrd;
        const auto & kg = Theory::pk_k_grid();
        // DESI's input grid geomspace(min(5e-4, k_0/2), max(1, 2 k_max), 500)
        const double k1 = std::min(5e-4, kth.front() / 2), k2 = std::max(1.0, kth.back() * 2);
        in.k.resize(500);
        for (int i = 0; i < 500; i++)
            in.k[i] = k1 * std::pow(k2 / k1, i / 499.0);
        for (size_t i = 0; i < zeff.size(); i++) {
            const auto & P = t.pk_cb(zeff[i]);
            in.pdd.push_back(interp(kg, P, in.k, true));
            in.pnw.push_back(interp(kg, fid.filter.nowiggle(P, t.settings(), s), in.k, true));
            in.ptt.push_back(interp(kg, t.pk_theta(zeff[i]), in.k, true));
            in.qpar.push_back(t.D_H(zeff[i]) * h / fid.dhh[i]);
            in.qper.push_back(t.D_M(zeff[i]) * h / fid.dmh[i]);
        }
        return in;
    }

    const DESIDR1FullShapeLikelihood::Tables & DESIDR1FullShapeLikelihood::tables(const Theory & t) const {
        return t.cached<Tables>(label + "/tables", [&] { return make_tables(linear_inputs(t)); });
    }

    DESIDR1FullShapeLikelihood::Tables DESIDR1FullShapeLikelihood::make_tables(const LinearInputs & in) const {
        {
            using PERTURBATION::EPTRedshiftSpace;
            const DVector & kin = in.k;
            const auto & pdd = in.pdd;
            const auto & pnw = in.pnw;
            const auto & ptt = in.ptt;
            const size_t nz = zeff.size();
            EPTRedshiftSpace::Options opt;
            opt.kmin = kth.front();
            opt.kmax = kth.back();
            opt.nk = 200;
            opt.rbao = 110.0;
            opt.internal.cutoff = 20.0;
            opt.internal.log10_kmin = -4.0;
            opt.internal.log10_kmax = 3.0;
            opt.internal.N = 4000;
            std::unique_ptr<EPTRedshiftSpace> rept;
            {
                std::lock_guard<std::mutex> lock(fftw_plan_mutex);
                rept = std::make_unique<EPTRedshiftSpace>(kin, pdd[0], pnw[0], opt);
            }
            DVector kv = rept->k(), kv1 = kv;
            kv1.push_back(1.0);
            Tables out;
            double pcb0_at1 = 0.0;
            for (size_t i = 0; i < nz; i++) {
                const DVector pcb = interp(kin, pdd[i], kv1, true), pcbnw = interp(kin, pnw[i], kv1, true),
                              pt = interp(kin, ptt[i], kv1, true);
                if (i == 0)
                    pcb0_at1 = pcb.back();
                const double Dz = std::sqrt(pcb.back() / pcb0_at1);
                DVector fk(kv.size()), pc(pcb.begin(), pcb.end() - 1), pn(pcbnw.begin(), pcbnw.end() - 1);
                for (size_t j = 0; j < kv.size(); j++)
                    fk[j] = std::sqrt(pt[j] / pcb[j]);
                const auto mt = rept->multipole_components_scaled(fk, pc, pn, Dz, in.qpar[i], in.qper[i], 4);
                std::array<std::vector<EPTRedshiftSpace::Columns>, 3> poles;
                const std::vector<EPTRedshiftSpace::Columns> * src[3] = {&mt.p0, &mt.p2, &mt.p4};
                for (int l = 0; l < 3; l++) {
                    poles[l].assign(kth.size(), EPTRedshiftSpace::Columns{});
                    for (int c = 0; c < EPTRedshiftSpace::ncolumns; c++) {
                        DVector col(kv.size());
                        for (size_t j = 0; j < kv.size(); j++)
                            col[j] = (*src[l])[j][c];
                        const DVector v = interp(kv, col, kth, false);
                        for (size_t j = 0; j < kth.size(); j++)
                            poles[l][j][c] = v[j];
                    }
                }
                out.poles.push_back(std::move(poles));
            }
            return out;
        }
    }

    std::vector<double> DESIDR1FullShapeLikelihood::poles(const Theory & t,
                                                          size_t i,
                                                          const std::vector<double> & bias,
                                                          const std::vector<double> & linear) const {
        const double s8 = t.sigma8_cb(zeff[i]);
        return poles_impl(tables(t), i, s8, t.sigma8_theta(zeff[i]) / s8, bias, linear);
    }

    std::vector<double> DESIDR1FullShapeLikelihood::poles_impl(const Tables & tab,
                                                               size_t i,
                                                               double s8,
                                                               double f,
                                                               const std::vector<double> & bias,
                                                               const std::vector<double> & linear) const {
        const auto & T = tab.poles.at(i);
        const double b1L = bias.at(0) / s8 - 1.0, b2L = bias.at(1) / (s8 * s8), bsL = bias.at(2) / (s8 * s8);
        const double b1 = 1.0 + b1L, b2 = 8.0 / 21.0 * b1L + b2L, bs = bsL - 2.0 / 7.0 * b1L, b3 = b1L; // b3L = 0
        const double a0 = linear.size() > 0 ? linear[0] : 0.0, a2 = linear.size() > 1 ? linear[1] : 0.0;
        const double sn0 = linear.size() > 2 ? linear[2] : 0.0, sn2 = linear.size() > 3 ? linear[3] : 0.0;
        const auto [fsat, sigv] = fsat_sigv(names[i]);
        const double sn = shotnoise[i];
        // the coefficients of the 19 columns (DESI's physical basis; alpha4p = alpha6p = sn4p = 0)
        const double m[19] = {1.0,
                              b1,
                              b1 * b1,
                              b2,
                              b1 * b2,
                              b2 * b2,
                              bs,
                              b1 * bs,
                              b2 * bs,
                              bs * bs,
                              b3,
                              b1 * b3,
                              b1 * b1 * a0,
                              f * b1 * (a0 + a2),
                              f * f * a2,
                              0.0,
                              sn * sn0,
                              sn * fsat * sigv * sigv * sn2,
                              0.0};
        std::vector<double> out(3 * kth.size(), 0.0);
        for (int l = 0; l < 3; l++)
            for (size_t j = 0; j < kth.size(); j++) {
                double v = 0.0;
                for (int c = 0; c < 19; c++)
                    v += m[c] * T[l][j][c];
                out[l * kth.size() + j] = v;
            }
        return out;
    }

    DESIDR1FullShapeLikelihood::SampleResult DESIDR1FullShapeLikelihood::sample(const Theory & t,
                                                                                size_t i,
                                                                                const std::vector<double> & bias,
                                                                                const std::vector<double> & linear) const {
        const double s8 = t.sigma8_cb(zeff[i]);
        return sample_impl(tables(t), i, s8, t.sigma8_theta(zeff[i]) / s8, bias, linear);
    }

    std::vector<double> DESIDR1FullShapeLikelihood::loglike_samples(const LinearInputs & in,
                                                                    const std::vector<double> & sigma8,
                                                                    const std::vector<double> & f,
                                                                    const std::vector<std::vector<double>> & bias) const {
        const Tables T = make_tables(in);
        std::vector<double> out;
        for (size_t i = 0; i < names.size(); i++)
            out.push_back(sample_impl(T, i, sigma8.at(i), f.at(i), bias.at(i), {}).loglike);
        return out;
    }

    DESIDR1FullShapeLikelihood::SampleResult DESIDR1FullShapeLikelihood::sample_impl(const Tables & T,
                                                                                     size_t i,
                                                                                     double s8,
                                                                                     double f,
                                                                                     const std::vector<double> & bias,
                                                                                     const std::vector<double> & linear) const {
        const auto & W = window[i];
        const auto & P = icov[i];
        const size_t n = data[i].size();
        auto convolve = [&](const std::vector<double> & p) {
            std::vector<double> o(n, 0.0);
            for (size_t a = 0; a < n; a++)
                for (size_t b = 0; b < p.size(); b++)
                    o[a] += W[a][b] * p[b];
            return o;
        };
        SampleResult r;
        r.theory = convolve(poles_impl(T, i, s8, f, bias, linear));
        // the derivatives with respect to the linear parameters (the theory is linear in them)
        const std::vector<double> p0 = poles_impl(T, i, s8, f, bias, {});
        std::vector<std::vector<double>> G(nlin);
        for (int a = 0; a < nlin; a++) {
            std::vector<double> e(nlin, 0.0);
            e[a] = 1.0;
            auto pa = poles_impl(T, i, s8, f, bias, e);
            for (size_t j = 0; j < pa.size(); j++)
                pa[j] -= p0[j];
            G[a] = convolve(pa);
        }
        // As DESI's likelihood: at the linear parameters = 0, diff = theory - data, then the Gaussian integral over
        // them with the prior N(0, sigma^2)
        const std::vector<double> T0 = convolve(p0);
        std::vector<double> diff(n), Pd(n, 0.0);
        for (size_t a = 0; a < n; a++)
            diff[a] = T0[a] - data[i][a];
        for (size_t a = 0; a < n; a++)
            for (size_t b = 0; b < n; b++)
                Pd[a] += P[a][b] * diff[b];
        double chi2 = 0.0;
        for (size_t a = 0; a < n; a++)
            chi2 += diff[a] * Pd[a];
        double grad[nlin], LH[nlin][nlin], PH[nlin][nlin];
        for (int a = 0; a < nlin; a++) {
            grad[a] = 0.0;
            for (size_t j = 0; j < n; j++)
                grad[a] -= G[a][j] * Pd[j];
        }
        for (int a = 0; a < nlin; a++)
            for (int b = 0; b < nlin; b++) {
                double v = 0.0;
                for (size_t j = 0; j < n; j++)
                    for (size_t l = 0; l < n; l++)
                        v += G[a][j] * P[j][l] * G[b][l];
                LH[a][b] = -v;
                PH[a][b] = LH[a][b] - (a == b ? 1.0 / (lin_sigma[a] * lin_sigma[a]) : 0.0);
            }
        // dx = -PH^-1 grad
        gsl_matrix * M = gsl_matrix_alloc(nlin, nlin);
        gsl_vector * bvec = gsl_vector_alloc(nlin);
        gsl_vector * x = gsl_vector_alloc(nlin);
        gsl_permutation * perm = gsl_permutation_alloc(nlin);
        for (int a = 0; a < nlin; a++) {
            for (int b = 0; b < nlin; b++)
                gsl_matrix_set(M, a, b, -PH[a][b]);
            gsl_vector_set(bvec, a, grad[a]);
        }
        int sgn;
        gsl_linalg_LU_decomp(M, perm, &sgn);
        gsl_linalg_LU_solve(M, perm, bvec, x);
        const double logdet = gsl_linalg_LU_lndet(M); // ln det(-PH)
        double dx[nlin];
        for (int a = 0; a < nlin; a++)
            dx[a] = gsl_vector_get(x, a);
        gsl_matrix_free(M);
        gsl_vector_free(bvec);
        gsl_vector_free(x);
        gsl_permutation_free(perm);
        double ll = -0.5 * chi2, prior = 0.0;
        for (int a = 0; a < nlin; a++) {
            ll += grad[a] * dx[a];
            prior -= 0.5 * dx[a] * dx[a] / (lin_sigma[a] * lin_sigma[a]);
            for (int b = 0; b < nlin; b++)
                ll += 0.5 * dx[a] * LH[a][b] * dx[b];
        }
        if (marg)
            ll -= 0.5 * logdet;
        r.loglike = ll + prior;
        r.best_linear.assign(dx, dx + nlin);
        return r;
    }

    double DESIDR1FullShapeLikelihood::loglike(const Theory & t) const {
        double ll = 0.0;
        for (size_t i = 0; i < names.size(); i++) {
            const std::string p = "desi_" + names[i] + "_";
            const std::vector<double> bias{
                t.nuisance(p + "b1p", 1.0), t.nuisance(p + "b2p", 0.0), t.nuisance(p + "bsp", 0.0)};
            ll += sample(t, i, bias).loglike;
        }
        return ll;
    }

} // namespace FML::COSMOLOGY::FITTING
