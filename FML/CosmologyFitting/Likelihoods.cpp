#include <FML/CosmologyFitting/Likelihoods.h>
#include <FML/WeakLensing/CosmicShear.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>

#include <gsl/gsl_linalg.h>

namespace FML::COSMOLOGY::FITTING {

    namespace {
        std::vector<std::vector<double>> read_rows(const std::string & file) {
            std::ifstream in(file);
            if (!in)
                throw std::runtime_error("cannot open data file " + file + "");
            std::vector<std::vector<double>> rows;
            std::string line;
            while (std::getline(in, line)) {
                const auto p = line.find('#');
                if (p != std::string::npos)
                    line = line.substr(0, p);
                std::istringstream is(line);
                std::vector<double> r;
                double v;
                while (is >> v)
                    r.push_back(v);
                if (!r.empty())
                    rows.push_back(r);
            }
            return rows;
        }
    } // namespace

    // =============================================================================
    // Gaussian data
    // =============================================================================

    void GaussianData::set(const std::vector<double> & mean, const std::vector<std::vector<double>> & cov) {
        const size_t n = mean.size();
        if (cov.size() != n)
            throw std::invalid_argument("GaussianData: covariance has the wrong size");
        mu = mean;
        gsl_matrix * m = gsl_matrix_alloc(n, n);
        for (size_t i = 0; i < n; i++) {
            if (cov[i].size() != n)
                throw std::invalid_argument("GaussianData: covariance is not square");
            for (size_t j = 0; j < n; j++)
                gsl_matrix_set(m, i, j, cov[i][j]);
        }
        gsl_set_error_handler_off();
        if (gsl_linalg_cholesky_decomp1(m) != 0) {
            gsl_matrix_free(m);
            throw std::invalid_argument("GaussianData: covariance is not positive definite");
        }
        gsl_linalg_cholesky_invert(m);
        icov.assign(n, std::vector<double>(n));
        for (size_t i = 0; i < n; i++)
            for (size_t j = 0; j < n; j++)
                icov[i][j] = gsl_matrix_get(m, i, j);
        gsl_matrix_free(m);
    }

    double GaussianData::chi2(const std::vector<double> & model) const {
        double c2 = 0.0;
        const size_t n = mu.size();
        for (size_t i = 0; i < n; i++)
            for (size_t j = 0; j < n; j++)
                c2 += (model[i] - mu[i]) * icov[i][j] * (model[j] - mu[j]);
        return c2;
    }

    // =============================================================================
    // Supernovae
    // =============================================================================

    SupernovaLikelihood::SupernovaLikelihood(const std::string & file, bool free_amplitude) : free_amp(free_amplitude) {
        for (const auto & r : read_rows(file)) {
            if (r.size() < 3)
                throw std::runtime_error("supernova data: expected z, d_L, sigma in " + file);
            z.push_back(r[0]);
            dL.push_back(r[1]);
            sigma.push_back(r[2]);
        }
    }

    // chi^2(A) = sum (d_i - A t_i)^2/s_i^2 = a A^2 - 2 b A + c. Marginalized over A with a flat prior:
    // -2 ln L = c - b^2/a + ln a (+ const)
    double SupernovaLikelihood::loglike(const Theory & t) const {
        double a = 0.0, b = 0.0, cc = 0.0;
        for (size_t i = 0; i < z.size(); i++) {
            const double th = t.d_L(z[i]) / 1000.0, w = 1.0 / (sigma[i] * sigma[i]); // Gpc
            a += th * th * w;
            b += th * dL[i] * w;
            cc += dL[i] * dL[i] * w;
        }
        if (!free_amp)
            return -0.5 * (a - 2.0 * b + cc);
        return -0.5 * (cc - b * b / a + std::log(a));
    }

    // =============================================================================
    // Supernovae with covariance
    // =============================================================================

    namespace {
        // all numbers of a text file (fast; whitespace or comma separated, '#' lines skipped)
        std::vector<double> read_numbers(const std::string & file) {
            std::ifstream in(file);
            if (!in)
                throw std::runtime_error("cannot open data file " + file +
                                         " (large SN data: "
                                         "python3 CosmologyFitting/python/fetch_sn_data.py)");
            std::vector<double> v;
            std::string line;
            while (std::getline(in, line)) {
                if (line.empty() || line[0] == '#')
                    continue;
                const char * p = line.c_str();
                char * end = nullptr;
                while (*p) {
                    const double x = std::strtod(p, &end);
                    if (end == p) {
                        p++;
                        continue;
                    }
                    v.push_back(x);
                    p = end;
                }
            }
            return v;
        }
        // a whitespace- or comma-separated table with a header line of column names
        struct Table {
            std::vector<std::string> names;
            std::vector<std::vector<std::string>> rows;
            size_t col(const std::string & n, const std::string & file) const {
                for (size_t i = 0; i < names.size(); i++)
                    if (names[i] == n)
                        return i;
                throw std::runtime_error("column " + n + " not found in " + file);
            }
        };
        Table read_table(const std::string & file) {
            std::ifstream in(file);
            if (!in)
                throw std::runtime_error("cannot open data file " + file +
                                         " (large SN data: "
                                         "python3 CosmologyFitting/python/fetch_sn_data.py)");
            Table t;
            std::string line;
            auto split = [](std::string s) {
                for (char & ch : s)
                    if (ch == ',')
                        ch = ' ';
                std::istringstream is(s);
                std::vector<std::string> out;
                std::string w;
                while (is >> w)
                    out.push_back(w);
                return out;
            };
            while (std::getline(in, line)) {
                if (line.empty())
                    continue;
                if (t.names.empty()) {
                    if (line[0] == '#')
                        line = line.substr(1);
                    t.names = split(line);
                    continue;
                }
                if (line[0] == '#')
                    continue;
                auto r = split(line); // (Union3's lcparam header names one column more than its rows have)
                if (r.size() + 1 >= t.names.size()) {
                    r.resize(t.names.size());
                    t.rows.push_back(r);
                }
            }
            return t;
        }
        // square matrix from "N, then N^2 numbers" (CosmoMC format)
        std::vector<double> read_square(const std::string & file, size_t n) {
            auto v = read_numbers(file);
            if (v.empty() || size_t(v[0]) != n || v.size() != 1 + n * n)
                throw std::runtime_error("covariance " + file + ": expected " + std::to_string(n) + "^2 entries");
            return std::vector<double>(v.begin() + 1, v.end());
        }
    } // namespace

    SNIaLikelihood::SNIaLikelihood(std::string name,
                                   std::vector<double> z_cmb,
                                   std::vector<double> z_hel,
                                   std::vector<double> mag,
                                   const std::vector<double> & cov)
        : label(std::move(name)), zc(std::move(z_cmb)), zh(std::move(z_hel)), m(std::move(mag)) {
        const size_t n = zc.size();
        if (zh.size() != n || m.size() != n || cov.size() != n * n)
            throw std::invalid_argument(label + ": inconsistent data sizes");
        gsl_matrix * A = gsl_matrix_alloc(n, n);
        for (size_t i = 0; i < n; i++)
            for (size_t j = 0; j < n; j++)
                gsl_matrix_set(A, i, j, 0.5 * (cov[i * n + j] + cov[j * n + i]));
        gsl_set_error_handler_off();
        if (gsl_linalg_cholesky_decomp1(A) != 0) {
            gsl_matrix_free(A);
            throw std::invalid_argument(label + ": covariance is not positive definite");
        }
        gsl_linalg_cholesky_invert(A);
        icov.resize(n * n);
        rowsum.assign(n, 0.0);
        for (size_t i = 0; i < n; i++)
            for (size_t j = 0; j < n; j++) {
                icov[i * n + j] = gsl_matrix_get(A, i, j);
                rowsum[i] += icov[i * n + j];
            }
        gsl_matrix_free(A);
        for (double r : rowsum)
            e += r;
    }

    double SNIaLikelihood::loglike_mu(const std::vector<double> & mu) const {
        const size_t n = zc.size();
        std::vector<double> d(n);
        for (size_t i = 0; i < n; i++)
            d[i] = m[i] - mu[i];
        double a = 0.0, b = 0.0;
        for (size_t i = 0; i < n; i++) {
            const double * row = &icov[i * n];
            double s = 0.0;
            for (size_t j = 0; j < n; j++)
                s += row[j] * d[j];
            a += d[i] * s;
            b += rowsum[i] * d[i];
        }
        return -0.5 * (a - b * b / e + std::log(e / (2.0 * M_PI)));
    }

    double SNIaLikelihood::offset(const std::vector<double> & mu) const {
        double b = 0.0;
        for (size_t i = 0; i < zc.size(); i++)
            b += rowsum[i] * (m[i] - mu[i]);
        return b / e;
    }

    double SNIaLikelihood::loglike(const Theory & t) const {
        std::vector<double> mu(zc.size());
        for (size_t i = 0; i < zc.size(); i++)
            mu[i] = 5.0 * std::log10((1.0 + zh[i]) * t.D_M(zc[i])) + 25.0; // D_M in Mpc
        return loglike_mu(mu);
    }

    std::unique_ptr<SNIaLikelihood> SNIaLikelihood::pantheonplus(const std::string & dir) {
        const std::string f = dir + "/Pantheon+SH0ES.dat";
        const Table t = read_table(f);
        const size_t iz = t.col("zHD", f), ih = t.col("zHEL", f), im = t.col("m_b_corr", f);
        const size_t N = t.rows.size();
        const auto C = read_square(dir + "/Pantheon+SH0ES_STAT+SYS.cov", N);
        std::vector<size_t> keep;
        for (size_t i = 0; i < N; i++)
            if (std::stod(t.rows[i][iz]) > 0.01)
                keep.push_back(i); // no Cepheid hosts
        std::vector<double> z, zh, m, c(keep.size() * keep.size());
        for (size_t a = 0; a < keep.size(); a++) {
            z.push_back(std::stod(t.rows[keep[a]][iz]));
            zh.push_back(std::stod(t.rows[keep[a]][ih]));
            m.push_back(std::stod(t.rows[keep[a]][im]));
            for (size_t b = 0; b < keep.size(); b++)
                c[a * keep.size() + b] = C[keep[a] * N + keep[b]];
        }
        return std::make_unique<SNIaLikelihood>("pantheonplus", z, zh, m, c);
    }

    std::unique_ptr<SNIaLikelihood> SNIaLikelihood::desy5(const std::string & dir) {
        const std::string f = dir + "/DES-SN5YR_HD.csv";
        const Table t = read_table(f);
        const size_t iz = t.col("zHD", f), ih = t.col("zHEL", f), im = t.col("MU", f), ie = t.col("MUERR_FINAL", f);
        const size_t N = t.rows.size();
        auto c = read_square(dir + "/covsys_000.txt", N);
        std::vector<double> z, zh, m;
        for (size_t i = 0; i < N; i++) {
            z.push_back(std::stod(t.rows[i][iz]));
            zh.push_back(std::stod(t.rows[i][ih]));
            m.push_back(std::stod(t.rows[i][im]));
            c[i * N + i] += std::pow(std::stod(t.rows[i][ie]), 2);
        }
        return std::make_unique<SNIaLikelihood>("desy5", z, zh, m, c);
    }

    std::unique_ptr<SNIaLikelihood> SNIaLikelihood::union3(const std::string & dir) {
        const std::string f = dir + "/lcparam_full.txt";
        const Table t = read_table(f);
        const size_t iz = t.col("zcmb", f), ih = t.col("zhel", f), im = t.col("mb", f), ie = t.col("dmb", f);
        const size_t N = t.rows.size();
        auto c = read_square(dir + "/mag_covmat.txt", N);
        std::vector<double> z, zh, m;
        for (size_t i = 0; i < N; i++) {
            z.push_back(std::stod(t.rows[i][iz]));
            zh.push_back(std::stod(t.rows[i][ih]));
            m.push_back(std::stod(t.rows[i][im]));
            c[i * N + i] += std::pow(std::stod(t.rows[i][ie]), 2);
        }
        return std::make_unique<SNIaLikelihood>("union3", z, zh, m, c);
    }

    // =============================================================================
    // BAO
    // =============================================================================

    BAOLikelihood::BAOLikelihood(const std::string & label_in,
                                 const std::string & mean_file,
                                 const std::string & cov_file,
                                 double rs_fid_in)
        : label(label_in), rs_fid(rs_fid_in) {
        std::ifstream in(mean_file);
        if (!in)
            throw std::runtime_error("cannot open data file " + mean_file + "");
        std::vector<double> mean;
        std::string line;
        while (std::getline(in, line)) {
            if (line.empty() || line[0] == '#')
                continue;
            std::istringstream is(line);
            double zz, v;
            std::string name;
            if (!(is >> zz >> v >> name))
                continue;
            z.push_back(zz);
            mean.push_back(v);
            if (name == "DV_over_rs")
                q.push_back(Quantity::DV_over_rd);
            else if (name == "DM_over_rs")
                q.push_back(Quantity::DM_over_rd);
            else if (name == "DH_over_rs")
                q.push_back(Quantity::DH_over_rd);
            else if (name == "bao_Hz_rs" && rs_fid > 0.0)
                q.push_back(Quantity::H_rd);
            else if (name == "f_sigma8")
                q.push_back(Quantity::fsigma8);
            else
                throw std::runtime_error("BAO data: unknown quantity " + name + " in " + mean_file +
                                         (name == "bao_Hz_rs" ? " (bao_Hz_rs needs rs_fid)" : ""));
        }
        const auto cov = read_rows(cov_file);
        if (cov.size() != mean.size())
            throw std::runtime_error("BAO data: the covariance " + cov_file + " does not match " + mean_file);
        data.set(mean, cov);
    }

    std::vector<double> BAOLikelihood::growth_redshifts() const {
        std::vector<double> zz;
        for (size_t i = 0; i < z.size(); i++)
            if (q[i] == Quantity::fsigma8)
                zz.push_back(z[i]);
        return zz;
    }

    std::vector<double> BAOLikelihood::model(const Theory & t) const {
        const double rd = t.r_drag();
        const double scale = rs_fid > 0.0 ? rs_fid : 1.0; // distances in Mpc when scaled by rs_fid
        const double c_kms = Constants::c / 1e3;
        std::vector<double> m(z.size());
        for (size_t i = 0; i < z.size(); i++) {
            switch (q[i]) {
                case Quantity::DV_over_rd:
                    m[i] = t.D_V(z[i]) / rd * scale;
                    break;
                case Quantity::DM_over_rd:
                    m[i] = t.D_M(z[i]) / rd * scale;
                    break;
                case Quantity::DH_over_rd:
                    m[i] = t.D_H(z[i]) / rd * scale;
                    break;
                case Quantity::H_rd: // H(z) r_d / rs_fid [km/s/Mpc], H = c / D_H
                    m[i] = c_kms / t.D_H(z[i]) * rd / rs_fid;
                    break;
                case Quantity::fsigma8:
                    m[i] = t.fsigma8(z[i]);
                    break;
            }
        }
        return m;
    }

    double BAOLikelihood::loglike(const Theory & t) const { return -0.5 * data.chi2(model(t)); }

    // =============================================================================
    // Planck 2018 distance priors (Chen, Huang & Wang 2019, Table 1, LCDM, TT,TE,EE+lowE)
    // =============================================================================

    PlanckDistancePriors::PlanckDistancePriors() {
        const std::vector<double> mean = {1.7502, 301.471, 0.02236}, sig = {0.0046, 0.0895, 0.00015};
        const double corr[3][3] = {{1.0, 0.46, -0.66}, {0.46, 1.0, -0.33}, {-0.66, -0.33, 1.0}};
        std::vector<std::vector<double>> cov(3, std::vector<double>(3));
        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 3; j++)
                cov[i][j] = corr[i][j] * sig[i] * sig[j];
        data.set(mean, cov);
    }

    double PlanckDistancePriors::loglike(const Theory & t) const {
        return -0.5 * data.chi2({t.R_shift(), t.l_A(), t.omega_b()});
    }

    double BBNPrior::loglike(const Theory & t) const {
        const double d = (t.omega_b() - 0.02218) / 0.00055;
        return -0.5 * d * d;
    }

    // =============================================================================
    // Planck 2018 plik_lite (as planck-lite-py and Cobaya's native plik_lite)
    // =============================================================================

    namespace {
        constexpr int NTT = 215, NTE = 199, NEE = 199, PLMIN = 30;
        std::vector<double> read_column(const std::string & file, size_t col) {
            std::vector<double> v;
            for (const auto & r : read_rows(file)) {
                if (r.size() <= col)
                    throw std::runtime_error("too few columns in " + file);
                v.push_back(r[col]);
            }
            return v;
        }
    } // namespace

    PlanckLiteLikelihood::PlanckLiteLikelihood(const std::string & dir, bool tt_only_in, bool low_ell_in)
        : tt_only(tt_only_in), low_ell(low_ell_in) {
        const auto blmin = read_column(dir + "/blmin.dat", 0), blmax = read_column(dir + "/blmax.dat", 0);
        const auto bw = read_column(dir + "/bweight.dat", 0);
        const auto X = read_column(dir + "/cl_cmb_plik_v22.dat", 1);
        const int nhi = NTT + NTE + NEE;
        if (int(X.size()) != nhi || int(blmin.size()) < NTT || int(blmax.size()) < NTT)
            throw std::runtime_error("plik_lite: unexpected data size in " + dir);
        auto make_bin = [&](int i,
                            int lmin0,
                            const std::vector<double> & lo,
                            const std::vector<double> & hi,
                            const std::vector<double> & w) {
            Bin b;
            b.lmin = int(lo[i]) + lmin0;
            b.lmax = int(hi[i]) + lmin0;
            for (int k = int(lo[i]); k <= int(hi[i]); k++)
                b.w.push_back(w.at(k));
            return b;
        };

        // Covariance of the high-ell bins: Fortran unformatted record of nhi^2 doubles; the lower triangle is used
        std::vector<std::vector<double>> C(nhi, std::vector<double>(nhi));
        {
            std::ifstream in(dir + "/c_matrix_plik_v22.dat", std::ios::binary);
            if (!in)
                throw std::runtime_error("cannot open " + dir + "/c_matrix_plik_v22.dat");
            std::int32_t rec = 0;
            in.read(reinterpret_cast<char *>(&rec), 4);
            if (rec != std::int32_t(sizeof(double) * nhi * nhi))
                throw std::runtime_error("plik_lite: bad covariance record");
            std::vector<double> buf(size_t(nhi) * nhi);
            in.read(reinterpret_cast<char *>(buf.data()), std::streamsize(buf.size() * sizeof(double)));
            if (!in)
                throw std::runtime_error("plik_lite: covariance file too short");
            for (int i = 0; i < nhi; i++)
                for (int j = 0; j < nhi; j++)
                    C[i][j] = buf[size_t(i) * nhi + j];
            for (int i = 0; i < nhi; i++)
                for (int j = i; j < nhi; j++)
                    C[i][j] = C[j][i];
        }

        std::vector<double> mean;
        std::vector<double> lo_sig;
        if (low_ell) {
            const auto lmn = read_column(dir + "/blmin_low_ell.dat", 0),
                       lmx = read_column(dir + "/blmax_low_ell.dat", 0);
            const auto lw = read_column(dir + "/bweight_low_ell.dat", 0);
            const auto lX = read_column(dir + "/CTT_bin_low_ell_2018.dat", 1),
                       lS = read_column(dir + "/CTT_bin_low_ell_2018.dat", 2);
            for (size_t i = 0; i < lX.size(); i++) {
                bins_tt.push_back(make_bin(int(i), 2, lmn, lmx, lw));
                mean.push_back(lX[i]);
                lo_sig.push_back(lS[i]);
            }
        }
        for (int i = 0; i < NTT; i++)
            bins_tt.push_back(make_bin(i, PLMIN, blmin, blmax, bw));
        for (int i = 0; i < NTT; i++)
            mean.push_back(X[i]);
        int n_hi = NTT;
        if (!tt_only) {
            for (int i = 0; i < NTE; i++)
                bins_pol.push_back(make_bin(i, PLMIN, blmin, blmax, bw));
            for (int i = NTT; i < nhi; i++)
                mean.push_back(X[i]);
            n_hi = nhi;
        }
        const size_t nlo = lo_sig.size(), n = nlo + n_hi;
        std::vector<std::vector<double>> cov(n, std::vector<double>(n, 0.0));
        for (size_t i = 0; i < nlo; i++)
            cov[i][i] = lo_sig[i] * lo_sig[i];
        for (int i = 0; i < n_hi; i++)
            for (int j = 0; j < n_hi; j++)
                cov[nlo + i][nlo + j] = C[i][j];
        data.set(mean, cov);
    }

    std::string PlanckLiteLikelihood::name() const {
        return std::string("planck_lite_") + (tt_only ? "TT" : "TTTEEE") + (low_ell ? "_lowl" : "");
    }

    double PlanckLiteLikelihood::loglike(const std::vector<double> & Dtt,
                                         const std::vector<double> & Dte,
                                         const std::vector<double> & Dee,
                                         double A) const {
        if (int(Dtt.size()) <= 2508 || (!tt_only && (Dte.size() <= 1996 || Dee.size() <= 1996)))
            throw std::invalid_argument("planck_lite: spectra needed up to ell = 2508");
        auto binned = [](const Bin & b, const std::vector<double> & D) {
            double v = 0.0;
            for (int l = b.lmin; l <= b.lmax; l++)
                v += b.w[l - b.lmin] * D[l] * 2.0 * M_PI / (l * (l + 1.0));
            return v;
        };
        const double cal = 1.0 / (A * A);
        std::vector<double> m;
        m.reserve(data.size());
        for (const auto & b : bins_tt)
            m.push_back(binned(b, Dtt) * cal);
        for (const auto & b : bins_pol)
            m.push_back(binned(b, Dte) * cal);
        for (const auto & b : bins_pol)
            m.push_back(binned(b, Dee) * cal);
        return -0.5 * data.chi2(m);
    }

    double PlanckLiteLikelihood::loglike(const Theory & t) const {
        const auto & c = t.cmb();
        return loglike(c.tt, c.te, c.ee, t.nuisance("A_planck", 1.0));
    }

    // =============================================================================
    // Planck 2018 lensing (CMBlikes format)
    // =============================================================================

    PlanckLensingLikelihood::PlanckLensingLikelihood(const std::string & dir, bool cmb_marged) : marged(cmb_marged) {
        const std::string stem = dir + "/smicadx12_Dec5_ftl_mv2_ndclpp_p_teb_consext8";
        const std::string var = marged ? stem + "_CMBmarged" : stem;
        // bandpowers: bin, L_min, L_max, L_av, PP, error, Ahat
        std::vector<double> pp_hat;
        for (const auto & r : read_rows(stem + "_bandpowers.dat")) {
            if (r.size() < 5)
                throw std::runtime_error("planck_lensing: bad bandpower file");
            pp_hat.push_back(r[4]);
        }
        const size_t nb = pp_hat.size();
        const auto cov = read_rows(var + "_cov.dat");
        if (cov.size() != nb)
            throw std::runtime_error("planck_lensing: covariance does not match the bandpowers");
        data.set(pp_hat, cov);
        const int nL = lmax - lmin + 1;
        // window files: L, then one column per spectrum (in_order)
        auto windows = [&](const std::string & d, size_t ncol, std::vector<std::vector<std::vector<double>> *> out) {
            for (auto * o : out)
                o->assign(nb, std::vector<double>(nL, 0.0));
            for (size_t b = 0; b < nb; b++) {
                for (const auto & r : read_rows(d + "/window" + std::to_string(b + 1) + ".dat")) {
                    if (r.size() != ncol + 1)
                        throw std::runtime_error("planck_lensing: unexpected columns in " + d);
                    const int L = int(std::lround(r[0]));
                    if (L < lmin || L > lmax)
                        continue;
                    for (size_t k = 0; k < ncol; k++)
                        (*out[k])[b][size_t(L - lmin)] = r[k + 1];
                }
            }
        };
        windows(stem + "_window", 1, {&w_pp});
        if (marged)
            windows(var + "_lens_delta_window", 1, {&c_pp});
        else
            windows(var + "_lens_delta_window", 4, {&c_tt, &c_ee, &c_te, &c_pp}); // in_order TT EE TE PP
        for (const auto & r : read_rows(var + "_lensing_fiducial_correction.dat"))
            fid_corr.push_back(r.at(1));
        if (fid_corr.size() != nb)
            throw std::runtime_error("planck_lensing: fiducial correction does not match the bandpowers");
    }

    std::vector<double> PlanckLensingLikelihood::binned_theory(const std::vector<double> & Dtt,
                                                               const std::vector<double> & Dte,
                                                               const std::vector<double> & Dee,
                                                               const std::vector<double> & pp,
                                                               double A_planck) const {
        if (int(pp.size()) <= lmax ||
            (!marged && (int(Dtt.size()) <= lmax || int(Dee.size()) <= lmax || int(Dte.size()) <= lmax)))
            throw std::invalid_argument("planck_lensing: spectra needed up to ell = 2500");
        const double ic = 1.0 / (A_planck * A_planck); // calibration of the CMB spectra (T and E maps)
        std::vector<double> th(w_pp.size(), 0.0);
        for (size_t b = 0; b < th.size(); b++) {
            double s = -fid_corr[b];
            for (int L = lmin; L <= lmax; L++) {
                const size_t i = size_t(L - lmin);
                s += (w_pp[b][i] + c_pp[b][i]) * pp[L];
                if (!marged)
                    s += ic * (c_tt[b][i] * Dtt[L] + c_ee[b][i] * Dee[L] + c_te[b][i] * Dte[L]);
            }
            th[b] = s;
        }
        return th;
    }

    double PlanckLensingLikelihood::loglike(const std::vector<double> & Dtt,
                                            const std::vector<double> & Dte,
                                            const std::vector<double> & Dee,
                                            const std::vector<double> & pp,
                                            double A_planck) const {
        return -0.5 * data.chi2(binned_theory(Dtt, Dte, Dee, pp, A_planck));
    }

    double PlanckLensingLikelihood::loglike(const Theory & t) const {
        const auto & c = t.cmb();
        return loglike(c.tt, c.te, c.ee, c.pp, marged ? 1.0 : t.nuisance("A_planck", 1.0));
    }

    // =============================================================================
    // Mock CMB data
    // =============================================================================

    CMBMockLikelihood::CMBMockLikelihood(const std::string & file,
                                         int lmax_in,
                                         double fsky_in,
                                         double noise,
                                         double beam,
                                         bool use_bb)
        : lmax(lmax_in), fsky(fsky_in), bb(use_bb) {
        if (lmax < 2 || !(fsky > 0.0 && fsky <= 1.0) || noise < 0.0 || beam < 0.0)
            throw std::invalid_argument("cmb_mock: need lmax >= 2, 0 < fsky <= 1, noise >= 0, beam >= 0");
        for (auto * v : {&dTT, &dEE, &dTE, &dBB, &nT, &nP})
            v->assign(lmax + 1, 0.0);
        std::vector<bool> have(lmax + 1, false);
        for (const auto & r : read_rows(file)) { // ell, D_TT, D_EE, D_TE, ... (cells_lensed.txt or cells.txt)
            if (r.size() < (bb ? 5u : 4u))
                throw std::runtime_error("cmb_mock: expected ell, TT, EE, TE" + std::string(bb ? ", BB" : "") + " in " +
                                         file);
            const int l = int(std::lround(r[0]));
            if (l < 2 || l > lmax)
                continue;
            const double f = 2.0 * M_PI / (l * (l + 1.0));
            dTT[l] = r[1] * f;
            dEE[l] = r[2] * f;
            dTE[l] = r[3] * f;
            if (bb)
                dBB[l] = r[4] * f;
            have[l] = true;
        }
        for (int l = 2; l <= lmax; l++)
            if (!have[l])
                throw std::runtime_error("cmb_mock: " + file + " has no ell = " + std::to_string(l));
        const double arcmin = M_PI / (180.0 * 60.0), sb = beam * arcmin / std::sqrt(8.0 * std::log(2.0));
        for (int l = 2; l <= lmax; l++) {
            nT[l] = std::pow(noise * arcmin, 2) * std::exp(l * (l + 1.0) * sb * sb);
            nP[l] = 2.0 * nT[l];
            dTT[l] += nT[l];
            dEE[l] += nP[l];
            dBB[l] += nP[l];
        }
    }

    double CMBMockLikelihood::loglike(const std::vector<double> & Dtt,
                                      const std::vector<double> & Dte,
                                      const std::vector<double> & Dee,
                                      const std::vector<double> & Dbb) const {
        if (int(Dtt.size()) <= lmax || int(Dte.size()) <= lmax || int(Dee.size()) <= lmax ||
            (bb && int(Dbb.size()) <= lmax))
            throw std::invalid_argument("cmb_mock: spectra needed up to ell = " + std::to_string(lmax));
        double m2lnL = 0.0;
        for (int l = 2; l <= lmax; l++) {
            const double f = 2.0 * M_PI / (l * (l + 1.0));
            const double T = Dtt[l] * f + nT[l], E = Dee[l] * f + nP[l], X = Dte[l] * f;
            const double detC = T * E - X * X, detD = dTT[l] * dEE[l] - dTE[l] * dTE[l];
            if (!(detC > 0.0))
                return -std::numeric_limits<double>::infinity();
            const double tr = (dTT[l] * E + dEE[l] * T - 2.0 * dTE[l] * X) / detC;
            m2lnL += (2.0 * l + 1.0) * fsky * (tr + std::log(detC / detD) - 2.0);
            if (bb) {
                const double B = Dbb[l] * f + nP[l];
                if (!(B > 0.0))
                    return -std::numeric_limits<double>::infinity();
                m2lnL += (2.0 * l + 1.0) * fsky * (dBB[l] / B + std::log(B / dBB[l]) - 1.0);
            }
        }
        return -0.5 * m2lnL;
    }

    double CMBMockLikelihood::loglike(const Theory & t) const {
        const auto & c = t.cmb();
        return loglike(c.tt, c.te, c.ee, c.bb);
    }

    // =============================================================================

    // =============================================================================
    // DESI DR1 ShapeFit + BAO
    // =============================================================================

    namespace {
        // Natural cubic spline (as scipy's CubicSpline(bc_type='natural'), which cosmoprimo uses); outside the nodes
        // the end polynomials are extrapolated
        class NaturalSpline {
          public:
            NaturalSpline(std::vector<double> x_in, std::vector<double> y_in) : x(std::move(x_in)), y(std::move(y_in)) {
                const size_t n = x.size();
                if (n < 3 || y.size() != n)
                    throw std::runtime_error("NaturalSpline: need at least 3 points");
                y2.assign(n, 0.0);
                std::vector<double> u(n, 0.0);
                for (size_t i = 1; i + 1 < n; i++) {
                    const double sig = (x[i] - x[i - 1]) / (x[i + 1] - x[i - 1]);
                    const double p = sig * y2[i - 1] + 2.0;
                    y2[i] = (sig - 1.0) / p;
                    const double d = (y[i + 1] - y[i]) / (x[i + 1] - x[i]) - (y[i] - y[i - 1]) / (x[i] - x[i - 1]);
                    u[i] = (6.0 * d / (x[i + 1] - x[i - 1]) - sig * u[i - 1]) / p;
                }
                y2[n - 1] = 0.0;
                for (size_t k = n - 1; k-- > 0;)
                    y2[k] = y2[k] * y2[k + 1] + u[k];
            }
            double operator()(double xx) const {
                const size_t n = x.size();
                size_t hi = std::upper_bound(x.begin(), x.end(), xx) - x.begin();
                hi = std::clamp<size_t>(hi, 1, n - 1);
                const size_t lo = hi - 1;
                const double h = x[hi] - x[lo];
                const double a = (x[hi] - xx) / h, b = (xx - x[lo]) / h;
                return a * y[lo] + b * y[hi] + ((a * a * a - a) * y2[lo] + (b * b * b - b) * y2[hi]) * h * h / 6.0;
            }

          private:
            std::vector<double> x, y, y2;
        };

        // Eisenstein & Hu (1998) no-wiggle P(k) shape k^n_s T^2 as cosmoprimo's eisenstein_hu_nowiggle engine
        // (omega_m = omega_cdm + omega_b, z_drag of Hu & Sugiyama 1996 E1); k in h/Mpc
        struct EHNoWiggle {
            double omega_b, omega_m, h, theta, n_s, rs, alpha_gamma;
            EHNoWiggle(double ob, double oc, double h_, double TCMB, double ns)
                : omega_b(ob), omega_m(ob + oc), h(h_), theta(TCMB / 2.7), n_s(ns) {
                const double z_eq = 2.5e4 * omega_m * std::pow(theta, -4) - 1.0;
                const double k_eq = 0.0746 * omega_m * std::pow(theta, -2);
                const double b1 = 0.313 * std::pow(omega_m, -0.419) * (1.0 + 0.607 * std::pow(omega_m, 0.674));
                const double b2 = 0.238 * std::pow(omega_m, 0.223);
                const double z_drag = 1345.0 * std::pow(omega_m, 0.251) / (1.0 + 0.659 * std::pow(omega_m, 0.828)) *
                                      (1.0 + b1 * std::pow(omega_b, b2));
                const double R_drag = 31.5 * omega_b * std::pow(theta, -4) * (1000.0 / (1.0 + z_drag));
                const double R_eq = 31.5 * omega_b * std::pow(theta, -4) * (1000.0 / (1.0 + z_eq));
                rs = 2.0 / (3.0 * k_eq) * std::sqrt(6.0 / R_eq) *
                     std::log((std::sqrt(1.0 + R_drag) + std::sqrt(R_drag + R_eq)) / (1.0 + std::sqrt(R_eq)));
                const double fb = omega_b / omega_m;
                alpha_gamma = 1.0 - 0.328 * std::log(431.0 * omega_m) * fb + 0.38 * std::log(22.3 * omega_m) * fb * fb;
            }
            double operator()(double k_hMpc) const {
                const double k = k_hMpc * h, ks = k * rs;
                const double gamma = omega_m * (alpha_gamma + (1.0 - alpha_gamma) / (1.0 + std::pow(0.43 * ks, 4)));
                const double q = k * theta * theta / gamma;
                const double L0 = std::log(2.0 * M_E + 1.8 * q), C0 = 14.2 + 731.0 / (1.0 + 62.5 * q);
                const double T = L0 / (L0 + C0 * q * q);
                return std::pow(k_hMpc, n_s) * T * T;
            }
        };

        EHNoWiggle eh_of(const RunSettings & r) {
            const auto & c = r.cosmo;
            return EHNoWiggle(c.OmegaB * c.h * c.h, c.OmegaCDM * c.h * c.h, c.h, c.TCMB, c.n_s);
        }
    } // namespace

    DESIShapeFitLikelihood::DESIShapeFitLikelihood(const std::string & mean_file, const std::string & cov_file) {
        std::ifstream in(mean_file);
        if (!in)
            throw std::runtime_error("cannot open data file " + mean_file);
        std::vector<double> mean;
        std::string line;
        while (std::getline(in, line)) {
            if (line.empty() || line[0] == '#')
                continue;
            std::istringstream is(line);
            double zz, v;
            std::string name;
            if (!(is >> zz >> v >> name))
                continue;
            z.push_back(zz);
            mean.push_back(v);
            if (name == "qiso")
                q.push_back(Quantity::qiso);
            else if (name == "qap")
                q.push_back(Quantity::qap);
            else if (name == "qpar")
                q.push_back(Quantity::qpar);
            else if (name == "qper")
                q.push_back(Quantity::qper);
            else if (name == "df")
                q.push_back(Quantity::df);
            else if (name == "dm")
                q.push_back(Quantity::dm);
            else
                throw std::runtime_error("ShapeFit data: unknown quantity " + name + " in " + mean_file);
            if ((name == "df" || name == "dm") && std::find(zs.begin(), zs.end(), zz) == zs.end())
                zs.push_back(zz);
        }
        const auto cov = read_rows(cov_file);
        if (cov.size() != mean.size())
            throw std::runtime_error("ShapeFit data: the covariance " + cov_file + " does not match " + mean_file);
        data.set(mean, cov);
    }

    RunSettings DESIShapeFitLikelihood::fiducial_settings(const RunSettings & settings) {
        // cosmoprimo.fiducial.DESI(): Planck 2018 LCDM (h = 0.6736, omega_b = 0.02237, omega_cdm = 0.12, one massive
        // neutrino of 0.06 eV, N_eff = 3.046, A_s = 2.083e-9, n_s = 0.9649, T_cmb = 2.7255 K, BBN helium)
        RunSettings f = settings;
        const double h = 0.6736;
        auto fmt = [](double v) {
            std::ostringstream o;
            o << std::setprecision(17) << v;
            return o.str();
        };
        f.cosmo = ParameterFile::read("",
                                      {"h=0.6736",
                                       "OmegaB=" + fmt(0.02237 / (h * h)),
                                       "OmegaCDM=" + fmt(0.12 / (h * h)),
                                       "m_nu=0.06",
                                       "Neff=3.046",
                                       "A_s=2.083e-9",
                                       "n_s=0.9649",
                                       "TCMB=2.7255"})
                      .cosmo;
        return f;
    }

    void DESIShapeFitLikelihood::setup_fiducial(const Theory & t) const {
        TheoryOptions o;
        o.cmb = false;
        o.growth_z = zs;
        o.pk_z = zs;
        o.pk_z.push_back(0.0);
        const RunSettings fs = fiducial_settings(t.settings());
        Theory tf(fs, o);
        const double rd = tf.r_drag();
        fid.hrd = fs.cosmo.h * rd;
        for (double zz : z) {
            fid.dm_rd.push_back(tf.D_M(zz) / rd);
            fid.dh_rd.push_back(tf.D_H(zz) / rd);
        }

        // The nodes of cosmoprimo's PeakAveragePowerSpectrumBAOFilter: the extrema of P/P_EH of the fiducial at z = 0
        // for 1e-2 < k < 1 h/Mpc (after dividing by the smooth trend a/k + b + c k + d k^2 with the values and first
        // differences of P/P_EH at both ends of 1e-3 <= k <= 1), plus all grid points below 1e-3 h/Mpc and from the
        // last point below 1 h/Mpc on
        const auto & k = Theory::pk_k_grid();
        const auto & P0 = tf.pk_cb(0.0);
        const EHNoWiggle eh = eh_of(fs);
        std::vector<int> index;
        for (int i = 0; i < int(k.size()); i++)
            if (k[i] >= 1e-3 && k[i] <= 1.0)
                index.push_back(i);
        const int n = int(index.size());
        std::vector<double> kf(n), ratio(n);
        for (int j = 0; j < n; j++) {
            kf[j] = k[index[j]];
            ratio[j] = P0[index[j]] / eh(kf[j]);
        }
        auto basis = [](double kk, int i) { return std::pow(kk, i - 1); };
        double A[16], b[4] = {ratio[0], ratio[1] - ratio[0], ratio[n - 1], ratio[n - 2] - ratio[n - 1]};
        for (int i = 0; i < 4; i++) {
            A[0 * 4 + i] = basis(kf[0], i);
            A[1 * 4 + i] = basis(kf[1], i) - basis(kf[0], i);
            A[2 * 4 + i] = basis(kf[n - 1], i);
            A[3 * 4 + i] = basis(kf[n - 2], i) - basis(kf[n - 1], i);
        }
        gsl_matrix_view Am = gsl_matrix_view_array(A, 4, 4);
        gsl_vector_view bv = gsl_vector_view_array(b, 4);
        gsl_vector * c = gsl_vector_alloc(4);
        gsl_permutation * perm = gsl_permutation_alloc(4);
        int sgn;
        gsl_linalg_LU_decomp(&Am.matrix, perm, &sgn);
        gsl_linalg_LU_solve(&Am.matrix, perm, &bv.vector, c);
        std::vector<double> detrended(n);
        for (int j = 0; j < n; j++) {
            double corr = 0.0;
            for (int i = 0; i < 4; i++)
                corr += gsl_vector_get(c, i) * basis(kf[j], i);
            detrended[j] = ratio[j] / corr;
        }
        gsl_vector_free(c);
        gsl_permutation_free(perm);
        const int ik0 = int(std::upper_bound(kf.begin(), kf.end(), 1e-2) - kf.begin()) + 1;
        const int N = int(k.size());
        for (int set = 0; set < 2; set++) {
            const double sign = set == 0 ? 1.0 : -1.0; // maxima, minima (scipy.signal.find_peaks)
            std::vector<int> ik;
            for (int j = ik0 + 1; j + 1 < n; j++)
                if (sign * detrended[j] > sign * detrended[j - 1] && sign * detrended[j] > sign * detrended[j + 1])
                    ik.push_back(j + index[0]);
            if (ik.empty())
                throw std::runtime_error("desi_dr1_shapefit: no BAO extrema found in the fiducial P(k)");
            const int npadlow = index[0];
            const int ikmax = std::max(index[n - 1], ik.back() + 1);
            auto & nodes = fid.nodes[set];
            for (int i = 0; i < npadlow; i++)
                nodes.push_back(i);
            nodes.insert(nodes.end(), ik.begin(), ik.end());
            for (int i = ikmax; i < N; i++)
                nodes.push_back(i);
            fid.npad_low[set] = npadlow;
            fid.npeaks[set] = int(ik.size());
            fid.npad_high[set] = N - ikmax;
        }
        for (double zz : zs) {
            const auto [m, fA] = shapefit_impl(tf, zz, 1.0);
            fid.m.push_back(m);
            fid.fA.push_back(fA);
        }
    }

    std::pair<double, double> DESIShapeFitLikelihood::shapefit(const Theory & t, double zz, double s) const {
        std::call_once(fid_once, [&] { setup_fiducial(t); });
        return shapefit_impl(t, zz, s);
    }

    // The no-wiggle P(k) of the PeakAverage filter: P_EH times the mean of the two envelopes of P/P_EH through its
    // maxima and minima (and the padding), each a natural cubic spline in log10 k of ln(P/P_EH) at the fiducial node
    // positions divided by s (s rising linearly from 1 over the low-k padding and falling back to 1 over the high-k
    // padding); ln(P/P_EH) at those positions from a natural spline over the whole grid. Then m = dln P_nw/dln k and
    // A_p = P(k_p)/s^3 at k_p = 0.03/s h/Mpc (log-log natural splines of P_nw and P on the grid, as cosmoprimo's
    // PowerSpectrumInterpolator1D), and f = sigma_8,thetatheta/sigma_8,cb
    std::pair<double, double> DESIShapeFitLikelihood::shapefit_impl(const Theory & t, double zz, double s) const {
        const auto & k = Theory::pk_k_grid();
        const auto & P = t.pk_cb(zz);
        const EHNoWiggle eh = eh_of(t.settings());
        const size_t N = k.size();
        std::vector<double> lk(N), lny(N), ehk(N);
        for (size_t i = 0; i < N; i++) {
            lk[i] = std::log10(k[i]);
            ehk[i] = eh(k[i]);
            lny[i] = std::log(P[i] / ehk[i]);
        }
        const NaturalSpline lnratio(lk, lny);
        std::vector<double> env(N, 0.0);
        for (int set = 0; set < 2; set++) {
            const auto & nodes = fid.nodes[set];
            const int nl = fid.npad_low[set], np = fid.npeaks[set], nh = fid.npad_high[set];
            std::vector<double> xx, yy;
            for (int j = 0; j < int(nodes.size()); j++) {
                double r;
                if (j < nl)
                    r = nl > 1 ? 1.0 + (s - 1.0) * j / (nl - 1.0) : s;
                else if (j < nl + np)
                    r = s;
                else
                    r = nh > 1 ? s + (1.0 - s) * (j - nl - np) / (nh - 1.0) : s;
                const double lx = std::log10(k[nodes[j]] / r);
                xx.push_back(lx);
                yy.push_back(lnratio(lx));
            }
            const NaturalSpline envelope(xx, yy);
            for (size_t i = 0; i < N; i++)
                env[i] += 0.5 * std::exp(envelope(lk[i]));
        }
        std::vector<double> lpnw(N), lp(N);
        for (size_t i = 0; i < N; i++) {
            lpnw[i] = std::log10(env[i] * ehk[i]);
            lp[i] = std::log10(P[i]);
        }
        const NaturalSpline pnw(lk, lpnw), pk(lk, lp);
        const double kp = 0.03 / s, k1 = kp * 0.99, k2 = kp * 1.01;
        const double m = (pnw(std::log10(k2)) - pnw(std::log10(k1))) / (std::log10(k2) - std::log10(k1));
        const double Ap = std::pow(10.0, pk(std::log10(kp))) / (s * s * s);
        const double f = t.sigma8_theta(zz) / t.sigma8_cb(zz);
        return {m, f * std::sqrt(Ap)};
    }

    std::vector<double> DESIShapeFitLikelihood::model(const Theory & t) const {
        const double rd = t.r_drag();
        std::call_once(fid_once, [&] { setup_fiducial(t); });
        const double sr = t.settings().cosmo.h * rd / fid.hrd;
        std::vector<std::pair<double, double>> sf(zs.size());
        for (size_t j = 0; j < zs.size(); j++)
            sf[j] = shapefit_impl(t, zs[j], sr);
        std::vector<double> m(z.size());
        for (size_t i = 0; i < z.size(); i++) {
            const double apar = t.D_H(z[i]) / rd / fid.dh_rd[i], aper = t.D_M(z[i]) / rd / fid.dm_rd[i];
            const size_t j = std::find(zs.begin(), zs.end(), z[i]) - zs.begin();
            switch (q[i]) {
                case Quantity::qiso:
                    m[i] = std::cbrt(apar * aper * aper);
                    break;
                case Quantity::qap:
                    m[i] = apar / aper;
                    break;
                case Quantity::qpar:
                    m[i] = apar;
                    break;
                case Quantity::qper:
                    m[i] = aper;
                    break;
                case Quantity::df:
                    m[i] = sf[j].second / fid.fA[j];
                    break;
                case Quantity::dm:
                    m[i] = sf[j].first - fid.m[j];
                    break;
            }
        }
        return m;
    }

    double DESIShapeFitLikelihood::loglike(const Theory & t) const { return -0.5 * data.chi2(model(t)); }

    // =============================================================================
    // KiDS-1000 COSEBIs
    // =============================================================================

    struct KiDS1000COSEBIsLikelihood::Impl {
        std::unique_ptr<WEAKLENSING::LogCOSEBIs> cosebis;
        std::vector<double> ell; // of the C_ell
    };

    KiDS1000COSEBIsLikelihood::~KiDS1000COSEBIsLikelihood() = default;

    KiDS1000COSEBIsLikelihood::KiDS1000COSEBIsLikelihood(const std::string & dir) : impl(std::make_unique<Impl>()) {
        const auto rows = read_rows(dir + "/cosebis_En.txt");
        std::vector<double> mean;
        for (const auto & r : rows) {
            if (r.size() != 4)
                throw std::runtime_error("kids1000_cosebis: cosebis_En.txt needs bin1 bin2 n E_n");
            const int i = int(r[0]) - 1, j = int(r[1]) - 1, n = int(r[2]);
            if (i < 0 || j < i || j > 4 || n < 1 || n > 5)
                throw std::runtime_error("kids1000_cosebis: bad bin pair or n in cosebis_En.txt");
            pair_index.push_back(i * 5 - i * (i + 1) / 2 + j);
            n_index.push_back(n - 1);
            mean.push_back(r[3]);
        }
        const auto cov = read_rows(dir + "/cosebis_cov.txt");
        if (cov.size() != mean.size())
            throw std::runtime_error("kids1000_cosebis: covariance and data differ in size");
        data.set(mean, cov);
        // n(z): histograms at z_mid, with n = 0 at z = 0 (as CosmoSIS' load_nz_fits)
        const auto nzr = read_rows(dir + "/nz_source.txt");
        zmid = {0.0};
        nz.assign(5, std::vector<double>{0.0});
        for (const auto & r : nzr) {
            if (r.size() != 8)
                throw std::runtime_error("kids1000_cosebis: nz_source.txt needs z_low z_mid z_high n_1..n_5");
            zmid.push_back(r[1]);
            for (int b = 0; b < 5; b++)
                nz[b].push_back(r[3 + b]);
        }
        const auto dz = read_rows(dir + "/dz_prior.txt");
        if (dz.size() != 6)
            throw std::runtime_error("kids1000_cosebis: dz_prior.txt needs the mean and the 5 x 5 covariance");
        dz_mean = dz[0];
        {
            gsl_matrix * c = gsl_matrix_alloc(5, 5);
            gsl_matrix * ci = gsl_matrix_alloc(5, 5);
            gsl_permutation * perm = gsl_permutation_alloc(5);
            for (int a = 0; a < 5; a++)
                for (int b = 0; b < 5; b++)
                    gsl_matrix_set(c, a, b, dz[1 + a].at(b));
            int sign = 0;
            gsl_linalg_LU_decomp(c, perm, &sign);
            gsl_linalg_LU_invert(c, perm, ci);
            dz_icov.assign(5, std::vector<double>(5, 0.0));
            for (int a = 0; a < 5; a++)
                for (int b = 0; b < 5; b++)
                    dz_icov[a][b] = gsl_matrix_get(ci, a, b);
            gsl_permutation_free(perm);
            gsl_matrix_free(ci);
            gsl_matrix_free(c);
        }
        for (int i = 0; i <= 122; i++)
            zgrid.push_back(0.05 * i);
        for (int i = 0; i < 80; i++)
            impl->ell.push_back(2.0 * std::pow(5e4, i / 79.0)); // 2 - 1e5
        const double arcmin = M_PI / 180.0 / 60.0;
        impl->cosebis = std::make_unique<WEAKLENSING::LogCOSEBIs>(0.5 * arcmin, 300.0 * arcmin, 5, 2.0, 1e5);
    }

    double KiDS1000COSEBIsLikelihood::chi2_dz_prior(const Theory & t) const {
        std::vector<double> d(5);
        for (int i = 0; i < 5; i++)
            d[i] = t.nuisance("kids_dz" + std::to_string(i + 1), dz_mean[i]) - dz_mean[i];
        double c2 = 0;
        for (int a = 0; a < 5; a++)
            for (int b = 0; b < 5; b++)
                c2 += d[a] * dz_icov[a][b] * d[b];
        return c2;
    }

    std::vector<double> KiDS1000COSEBIsLikelihood::model(const Theory & t) const {
        namespace WL = WEAKLENSING;
        const double h = t.settings().cosmo.h;
        // Background tables
        std::vector<double> chi, D;
        for (double z : zgrid) {
            chi.push_back(t.chi_mpch(z));
            D.push_back(t.growth_D(z));
        }
        WL::detail::CubicSpline chi_of_z(zgrid, chi), z_of_chi(chi, zgrid), D_of_z(zgrid, D);
        // ln P(z, ln k)
        const auto & kg = Theory::pk_nl_k_grid();
        std::vector<double> lk;
        for (double k : kg)
            lk.push_back(std::log(k));
        std::vector<std::vector<double>> lnP;
        for (double z : zgrid) {
            std::vector<double> r;
            for (double P : t.pk_nl(z))
                r.push_back(std::log(P));
            lnP.push_back(r);
        }
        Spline2D lnP_spline;
        lnP_spline.create(zgrid, lk, lnP, "kids1000_cosebis ln P_NL(z, ln k)");

        WL::ShearModel m;
        m.geometry.omega_m = t.omega_m() / (h * h);
        m.geometry.z_of_chi = [&](double c) { return std::max(0.0, z_of_chi(c)); };
        m.chi_of_z = [&](double z) { return chi_of_z(z); };
        m.growth = [&](double z) { return D_of_z(z); };
        // Modified lensing (Brans-Dicke: G/phi): Sigma(z) on the kernel; the prefactor uses Omega_m of the model's
        // density parameters (G_*)
        std::vector<double> sig;
        bool gr = true;
        for (double z : zgrid) {
            sig.push_back(t.lensing_Sigma(z));
            gr = gr && sig.back() == 1.0;
        }
        WL::detail::CubicSpline Sigma_of_z(zgrid, sig);
        if (!gr)
            m.lensing_strength = [&](double z) { return Sigma_of_z(z); };
        m.power.value = [&](double k, double z) { return std::exp(lnP_spline(z, std::log(k))); };
        m.power.k_min = kg.front();
        m.power.k_max = kg.back();
        m.power.z_min = 0.0;
        m.power.z_max = zgrid.back();

        WL::ShearSystematics sys;
        sys.ia.A = t.nuisance("kids_A_IA", 0.0);
        sys.ia.eta = 0.0;
        // C1 rho_crit = 5e-14 h^-2 M_sun^-1 Mpc^3 rho_crit (CosmoSIS, the default of WL::NLA)
        std::vector<WL::RedshiftDistribution> dist;
        for (int b = 0; b < 5; b++) {
            const double dz = t.nuisance("kids_dz" + std::to_string(b + 1), dz_mean[b]);
            // As CosmoSIS (photoz_bias with interpolation = cubic, then a cubic spline of n(z) in the projection):
            // the shifted histogram n(z_k - dz) at the nodes z_k by cubic interpolation, then a cubic spline through
            // the nodes, sampled every 0.002 (negative overshoots set to 0). With linear interpolation (CosmicShear's
            // shifted_distribution) chi^2 changes by ~0.1
            sys.delta_z.push_back(0.0);
            WL::detail::CubicSpline S(zmid, nz[b]);
            std::vector<double> ns;
            for (double z : zmid) {
                const double zz = z - dz;
                ns.push_back(zz < zmid.front() || zz > zmid.back() ? 0.0 : S(zz));
            }
            WL::detail::CubicSpline S2(zmid, ns);
            WL::RedshiftDistribution d;
            const int nf = int(std::ceil(zmid.back() / 0.002));
            for (int i = 0; i <= nf; i++) {
                const double z = zmid.back() * i / nf;
                d.z.push_back(z);
                d.n.push_back(std::max(0.0, S2(z)));
            }
            dist.push_back(d);
        }
        const auto cl = WL::shear_spectra(dist, m, sys, impl->ell);
        const auto E = impl->cosebis->E(cl);
        std::vector<double> out;
        for (size_t i = 0; i < pair_index.size(); i++)
            out.push_back(E[pair_index[i]][n_index[i]]);
        return out;
    }

    double KiDS1000COSEBIsLikelihood::loglike(const Theory & t) const {
        if (t.settings().ps.nonlinear == PowerSpectrumOptions::Nonlinear::None)
            throw std::invalid_argument("kids1000_cosebis needs a nonlinear P(k) (e.g. nonlinear = hmcode2016)");
        return -0.5 * (data.chi2(model(t)) + chi2_dz_prior(t));
    }

    // =============================================================================

    namespace Likelihoods {
        std::vector<std::string> names() {
            return {"sn",
                    "sn_free_amplitude",
                    "pantheonplus",
                    "desy5",
                    "union3",
                    "desi_dr2",
                    "desi_dr1",
                    "boss_dr12",
                    "boss_dr12_bao",
                    "eboss_dr16_lrg",
                    "eboss_dr16_qso",
                    "desi_dr1_shapefit",
                    "kids1000_cosebis",
                    "planck_dp",
                    "bbn",
                    "planck_lite_TTTEEE",
                    "planck_lite_TT",
                    "planck_lite_TTTEEE_lowl",
                    "planck_lite_TT_lowl",
                    "planck_lensing",
                    "planck_lensing_marged",
                    "cmb_mock"};
        }
        std::vector<std::string> option_names() {
            return {"cmb_mock_file",
                    "cmb_mock_lmax",
                    "cmb_mock_fsky",
                    "cmb_mock_noise",
                    "cmb_mock_beam",
                    "cmb_mock_bb"};
        }

        std::unique_ptr<Likelihood> create(const std::string & n, const std::map<std::string, std::string> & o) {
            for (const auto & kv : o) {
                bool known = false;
                for (const auto & on : option_names())
                    known = known || on == kv.first;
                if (!known)
                    throw std::invalid_argument("unknown likelihood option '" + kv.first + "'");
            }
            auto opt = [&](const std::string & k, const std::string & def) { return o.count(k) ? o.at(k) : def; };
            if (n == "sn")
                return std::make_unique<SupernovaLikelihood>();
            if (n == "pantheonplus")
                return SNIaLikelihood::pantheonplus();
            if (n == "desy5")
                return SNIaLikelihood::desy5();
            if (n == "union3")
                return SNIaLikelihood::union3();
            if (n == "sn_free_amplitude")
                return std::make_unique<SupernovaLikelihood>(
                    Utils::fml_file("CosmologyFitting/external/supernovadata.txt"),
                    true);
            if (n == "desi_dr2")
                return std::make_unique<BAOLikelihood>(
                    "desi_dr2",
                    Utils::fml_file("CosmologyFitting/external/bao/desi_dr2_mean.txt"),
                    Utils::fml_file("CosmologyFitting/external/bao/desi_dr2_cov.txt"));
            if (n == "desi_dr1")
                return std::make_unique<BAOLikelihood>(
                    "desi_dr1",
                    Utils::fml_file("CosmologyFitting/external/bao/desi_dr1_mean.txt"),
                    Utils::fml_file("CosmologyFitting/external/bao/desi_dr1_cov.txt"));
            const std::string bao = Utils::fml_file("CosmologyFitting/external/bao/");
            if (n == "boss_dr12")
                return std::make_unique<BAOLikelihood>("boss_dr12",
                                                       bao + "sdss_DR12Consensus_final.dat",
                                                       bao + "final_consensus_covtot_dM_Hz_fsig.txt",
                                                       147.78);
            if (n == "boss_dr12_bao")
                return std::make_unique<BAOLikelihood>("boss_dr12_bao",
                                                       bao + "sdss_DR12Consensus_bao.dat",
                                                       bao + "BAO_consensus_covtot_dM_Hz.txt",
                                                       147.78);
            if (n == "eboss_dr16_lrg")
                return std::make_unique<BAOLikelihood>("eboss_dr16_lrg",
                                                       bao + "sdss_DR16_BAOplus_LRG_FSBAO_DMDHfs8.dat",
                                                       bao + "sdss_DR16_BAOplus_LRG_FSBAO_DMDHfs8_covtot.txt");
            if (n == "eboss_dr16_qso")
                return std::make_unique<BAOLikelihood>("eboss_dr16_qso",
                                                       bao + "sdss_DR16_BAOplus_QSO_FSBAO_DMDHfs8.dat",
                                                       bao + "sdss_DR16_BAOplus_QSO_FSBAO_DMDHfs8_covtot.txt");
            if (n == "desi_dr1_shapefit")
                return std::make_unique<DESIShapeFitLikelihood>(bao + "desi_dr1_shapefit_mean.txt",
                                                                bao + "desi_dr1_shapefit_cov.txt");
            if (n == "kids1000_cosebis")
                return std::make_unique<KiDS1000COSEBIsLikelihood>(Utils::fml_file("CosmologyFitting/external/kids1000"));
            if (n == "planck_dp")
                return std::make_unique<PlanckDistancePriors>();
            if (n == "bbn")
                return std::make_unique<BBNPrior>();
            const std::string pl = Utils::fml_file("CosmologyFitting/external/planck2018_lite");
            if (n == "planck_lite_TTTEEE")
                return std::make_unique<PlanckLiteLikelihood>(pl, false, false);
            if (n == "planck_lite_TT")
                return std::make_unique<PlanckLiteLikelihood>(pl, true, false);
            if (n == "planck_lite_TTTEEE_lowl")
                return std::make_unique<PlanckLiteLikelihood>(pl, false, true);
            if (n == "planck_lite_TT_lowl")
                return std::make_unique<PlanckLiteLikelihood>(pl, true, true);
            if (n == "planck_lensing")
                return std::make_unique<PlanckLensingLikelihood>(
                    Utils::fml_file("CosmologyFitting/external/planck2018_lensing"),
                    false);
            if (n == "planck_lensing_marged")
                return std::make_unique<PlanckLensingLikelihood>(
                    Utils::fml_file("CosmologyFitting/external/planck2018_lensing"),
                    true);
            if (n == "cmb_mock") {
                const std::string b = opt("cmb_mock_bb", "false");
                if (b != "true" && b != "false")
                    throw std::invalid_argument("cmb_mock_bb must be true or false");
                return std::make_unique<CMBMockLikelihood>(opt("cmb_mock_file", "data/cmb_mock/cells_lensed.txt"),
                                                           std::stoi(opt("cmb_mock_lmax", "2500")),
                                                           std::stod(opt("cmb_mock_fsky", "0.6")),
                                                           std::stod(opt("cmb_mock_noise", "33")),
                                                           std::stod(opt("cmb_mock_beam", "7")),
                                                           b == "true");
            }
            std::string all;
            for (const auto & s : names())
                all += (all.empty() ? "" : ", ") + s;
            throw std::invalid_argument("unknown likelihood '" + n + "' (available: " + all + ")");
        }
    } // namespace Likelihoods

} // namespace FML::COSMOLOGY::FITTING
