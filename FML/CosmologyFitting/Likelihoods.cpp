#include <FML/CosmologyFitting/Likelihoods.h>

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <limits>
#include <sstream>
#include <stdexcept>

#include <gsl/gsl_linalg.h>

namespace FML::COSMOLOGY::FITTING {

namespace {
  std::vector<std::vector<double>> read_rows(const std::string& file) {
    std::ifstream in(file);
    if (!in) throw std::runtime_error("cannot open data file " + file + "");
    std::vector<std::vector<double>> rows;
    std::string line;
    while (std::getline(in, line)) {
      const auto p = line.find('#');
      if (p != std::string::npos) line = line.substr(0, p);
      std::istringstream is(line);
      std::vector<double> r;
      double v;
      while (is >> v) r.push_back(v);
      if (!r.empty()) rows.push_back(r);
    }
    return rows;
  }
}

// =============================================================================
// Gaussian data
// =============================================================================

void GaussianData::set(const std::vector<double>& mean, const std::vector<std::vector<double>>& cov) {
  const size_t n = mean.size();
  if (cov.size() != n) throw std::invalid_argument("GaussianData: covariance has the wrong size");
  mu = mean;
  gsl_matrix* m = gsl_matrix_alloc(n, n);
  for (size_t i = 0; i < n; i++) {
    if (cov[i].size() != n) throw std::invalid_argument("GaussianData: covariance is not square");
    for (size_t j = 0; j < n; j++) gsl_matrix_set(m, i, j, cov[i][j]);
  }
  gsl_set_error_handler_off();
  if (gsl_linalg_cholesky_decomp1(m) != 0) {
    gsl_matrix_free(m);
    throw std::invalid_argument("GaussianData: covariance is not positive definite");
  }
  gsl_linalg_cholesky_invert(m);
  icov.assign(n, std::vector<double>(n));
  for (size_t i = 0; i < n; i++)
    for (size_t j = 0; j < n; j++) icov[i][j] = gsl_matrix_get(m, i, j);
  gsl_matrix_free(m);
}

double GaussianData::chi2(const std::vector<double>& model) const {
  double c2 = 0.0;
  const size_t n = mu.size();
  for (size_t i = 0; i < n; i++)
    for (size_t j = 0; j < n; j++) c2 += (model[i] - mu[i]) * icov[i][j] * (model[j] - mu[j]);
  return c2;
}

// =============================================================================
// Supernovae
// =============================================================================

SupernovaLikelihood::SupernovaLikelihood(const std::string& file, bool free_amplitude) : free_amp(free_amplitude) {
  for (const auto& r : read_rows(file)) {
    if (r.size() < 3) throw std::runtime_error("supernova data: expected z, d_L, sigma in " + file);
    z.push_back(r[0]);
    dL.push_back(r[1]);
    sigma.push_back(r[2]);
  }
}

// chi^2(A) = sum (d_i - A t_i)^2/s_i^2 = a A^2 - 2 b A + c. Marginalized over A with a flat prior:
// -2 ln L = c - b^2/a + ln a (+ const)
double SupernovaLikelihood::loglike(const Theory& t) const {
  double a = 0.0, b = 0.0, cc = 0.0;
  for (size_t i = 0; i < z.size(); i++) {
    const double th = t.d_L(z[i]) / 1000.0, w = 1.0 / (sigma[i] * sigma[i]);   // Gpc
    a += th * th * w;
    b += th * dL[i] * w;
    cc += dL[i] * dL[i] * w;
  }
  if (!free_amp) return -0.5 * (a - 2.0 * b + cc);
  return -0.5 * (cc - b * b / a + std::log(a));
}

// =============================================================================
// Supernovae with covariance
// =============================================================================

namespace {
  // all numbers of a text file (fast; whitespace or comma separated, '#' lines skipped)
  std::vector<double> read_numbers(const std::string& file) {
    std::ifstream in(file);
    if (!in) throw std::runtime_error("cannot open data file " + file + " (large SN data: "
                                      "python3 CosmologyFitting/python/fetch_sn_data.py)");
    std::vector<double> v;
    std::string line;
    while (std::getline(in, line)) {
      if (line.empty() || line[0] == '#') continue;
      const char* p = line.c_str();
      char* end = nullptr;
      while (*p) {
        const double x = std::strtod(p, &end);
        if (end == p) { p++; continue; }
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
    size_t col(const std::string& n, const std::string& file) const {
      for (size_t i = 0; i < names.size(); i++) if (names[i] == n) return i;
      throw std::runtime_error("column " + n + " not found in " + file);
    }
  };
  Table read_table(const std::string& file) {
    std::ifstream in(file);
    if (!in) throw std::runtime_error("cannot open data file " + file + " (large SN data: "
                                      "python3 CosmologyFitting/python/fetch_sn_data.py)");
    Table t;
    std::string line;
    auto split = [](std::string s) {
      for (char& ch : s) if (ch == ',') ch = ' ';
      std::istringstream is(s);
      std::vector<std::string> out;
      std::string w;
      while (is >> w) out.push_back(w);
      return out;
    };
    while (std::getline(in, line)) {
      if (line.empty()) continue;
      if (t.names.empty()) {
        if (line[0] == '#') line = line.substr(1);
        t.names = split(line);
        continue;
      }
      if (line[0] == '#') continue;
      auto r = split(line);   // (Union3's lcparam header names one column more than its rows have)
      if (r.size() + 1 >= t.names.size()) { r.resize(t.names.size()); t.rows.push_back(r); }
    }
    return t;
  }
  // square matrix from "N, then N^2 numbers" (CosmoMC format)
  std::vector<double> read_square(const std::string& file, size_t n) {
    auto v = read_numbers(file);
    if (v.empty() || size_t(v[0]) != n || v.size() != 1 + n * n)
      throw std::runtime_error("covariance " + file + ": expected " + std::to_string(n) + "^2 entries");
    return std::vector<double>(v.begin() + 1, v.end());
  }
}

SNIaLikelihood::SNIaLikelihood(std::string name, std::vector<double> z_cmb, std::vector<double> z_hel,
                               std::vector<double> mag, const std::vector<double>& cov)
  : label(std::move(name)), zc(std::move(z_cmb)), zh(std::move(z_hel)), m(std::move(mag)) {
  const size_t n = zc.size();
  if (zh.size() != n || m.size() != n || cov.size() != n * n) throw std::invalid_argument(label + ": inconsistent data sizes");
  gsl_matrix* A = gsl_matrix_alloc(n, n);
  for (size_t i = 0; i < n; i++)
    for (size_t j = 0; j < n; j++) gsl_matrix_set(A, i, j, 0.5 * (cov[i * n + j] + cov[j * n + i]));
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
  for (double r : rowsum) e += r;
}

double SNIaLikelihood::loglike_mu(const std::vector<double>& mu) const {
  const size_t n = zc.size();
  std::vector<double> d(n);
  for (size_t i = 0; i < n; i++) d[i] = m[i] - mu[i];
  double a = 0.0, b = 0.0;
  for (size_t i = 0; i < n; i++) {
    const double* row = &icov[i * n];
    double s = 0.0;
    for (size_t j = 0; j < n; j++) s += row[j] * d[j];
    a += d[i] * s;
    b += rowsum[i] * d[i];
  }
  return -0.5 * (a - b * b / e + std::log(e / (2.0 * M_PI)));
}

double SNIaLikelihood::offset(const std::vector<double>& mu) const {
  double b = 0.0;
  for (size_t i = 0; i < zc.size(); i++) b += rowsum[i] * (m[i] - mu[i]);
  return b / e;
}

double SNIaLikelihood::loglike(const Theory& t) const {
  std::vector<double> mu(zc.size());
  for (size_t i = 0; i < zc.size(); i++) mu[i] = 5.0 * std::log10((1.0 + zh[i]) * t.D_M(zc[i])) + 25.0;   // D_M in Mpc
  return loglike_mu(mu);
}

std::unique_ptr<SNIaLikelihood> SNIaLikelihood::pantheonplus(const std::string& dir) {
  const std::string f = dir + "/Pantheon+SH0ES.dat";
  const Table t = read_table(f);
  const size_t iz = t.col("zHD", f), ih = t.col("zHEL", f), im = t.col("m_b_corr", f);
  const size_t N = t.rows.size();
  const auto C = read_square(dir + "/Pantheon+SH0ES_STAT+SYS.cov", N);
  std::vector<size_t> keep;
  for (size_t i = 0; i < N; i++) if (std::stod(t.rows[i][iz]) > 0.01) keep.push_back(i);   // no Cepheid hosts
  std::vector<double> z, zh, m, c(keep.size() * keep.size());
  for (size_t a = 0; a < keep.size(); a++) {
    z.push_back(std::stod(t.rows[keep[a]][iz]));
    zh.push_back(std::stod(t.rows[keep[a]][ih]));
    m.push_back(std::stod(t.rows[keep[a]][im]));
    for (size_t b = 0; b < keep.size(); b++) c[a * keep.size() + b] = C[keep[a] * N + keep[b]];
  }
  return std::make_unique<SNIaLikelihood>("pantheonplus", z, zh, m, c);
}

std::unique_ptr<SNIaLikelihood> SNIaLikelihood::desy5(const std::string& dir) {
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

std::unique_ptr<SNIaLikelihood> SNIaLikelihood::union3(const std::string& dir) {
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

BAOLikelihood::BAOLikelihood(const std::string& label_in, const std::string& mean_file, const std::string& cov_file)
  : label(label_in) {
  std::ifstream in(mean_file);
  if (!in) throw std::runtime_error("cannot open data file " + mean_file + "");
  std::vector<double> mean;
  std::string line;
  while (std::getline(in, line)) {
    if (line.empty() || line[0] == '#') continue;
    std::istringstream is(line);
    double zz, v;
    std::string name;
    if (!(is >> zz >> v >> name)) continue;
    z.push_back(zz);
    mean.push_back(v);
    if (name == "DV_over_rs") q.push_back(Quantity::DV_over_rd);
    else if (name == "DM_over_rs") q.push_back(Quantity::DM_over_rd);
    else if (name == "DH_over_rs") q.push_back(Quantity::DH_over_rd);
    else throw std::runtime_error("BAO data: unknown quantity " + name + " in " + mean_file);
  }
  data.set(mean, read_rows(cov_file));
}

std::vector<double> BAOLikelihood::model(const Theory& t) const {
  const double rd = t.r_drag();
  std::vector<double> m(z.size());
  for (size_t i = 0; i < z.size(); i++) {
    switch (q[i]) {
      case Quantity::DV_over_rd: m[i] = t.D_V(z[i]) / rd; break;
      case Quantity::DM_over_rd: m[i] = t.D_M(z[i]) / rd; break;
      case Quantity::DH_over_rd: m[i] = t.D_H(z[i]) / rd; break;
    }
  }
  return m;
}

double BAOLikelihood::loglike(const Theory& t) const { return -0.5 * data.chi2(model(t)); }

// =============================================================================
// Planck 2018 distance priors (Chen, Huang & Wang 2019, Table 1, LCDM, TT,TE,EE+lowE)
// =============================================================================

PlanckDistancePriors::PlanckDistancePriors() {
  const std::vector<double> mean = {1.7502, 301.471, 0.02236}, sig = {0.0046, 0.0895, 0.00015};
  const double corr[3][3] = {{1.0, 0.46, -0.66}, {0.46, 1.0, -0.33}, {-0.66, -0.33, 1.0}};
  std::vector<std::vector<double>> cov(3, std::vector<double>(3));
  for (int i = 0; i < 3; i++)
    for (int j = 0; j < 3; j++) cov[i][j] = corr[i][j] * sig[i] * sig[j];
  data.set(mean, cov);
}

double PlanckDistancePriors::loglike(const Theory& t) const {
  return -0.5 * data.chi2({t.R_shift(), t.l_A(), t.omega_b()});
}

double BBNPrior::loglike(const Theory& t) const {
  const double d = (t.omega_b() - 0.02218) / 0.00055;
  return -0.5 * d * d;
}

// =============================================================================
// Planck 2018 plik_lite (as planck-lite-py and Cobaya's native plik_lite)
// =============================================================================

namespace {
  constexpr int NTT = 215, NTE = 199, NEE = 199, PLMIN = 30;
  std::vector<double> read_column(const std::string& file, size_t col) {
    std::vector<double> v;
    for (const auto& r : read_rows(file)) {
      if (r.size() <= col) throw std::runtime_error("too few columns in " + file);
      v.push_back(r[col]);
    }
    return v;
  }
}

PlanckLiteLikelihood::PlanckLiteLikelihood(const std::string& dir, bool tt_only_in, bool low_ell_in)
  : tt_only(tt_only_in), low_ell(low_ell_in) {
  const auto blmin = read_column(dir + "/blmin.dat", 0), blmax = read_column(dir + "/blmax.dat", 0);
  const auto bw = read_column(dir + "/bweight.dat", 0);
  const auto X = read_column(dir + "/cl_cmb_plik_v22.dat", 1);
  const int nhi = NTT + NTE + NEE;
  if (int(X.size()) != nhi || int(blmin.size()) < NTT || int(blmax.size()) < NTT)
    throw std::runtime_error("plik_lite: unexpected data size in " + dir);
  auto make_bin = [&](int i, int lmin0, const std::vector<double>& lo, const std::vector<double>& hi,
                      const std::vector<double>& w) {
    Bin b;
    b.lmin = int(lo[i]) + lmin0;
    b.lmax = int(hi[i]) + lmin0;
    for (int k = int(lo[i]); k <= int(hi[i]); k++) b.w.push_back(w.at(k));
    return b;
  };

  // Covariance of the high-ell bins: Fortran unformatted record of nhi^2 doubles; the lower triangle is used
  std::vector<std::vector<double>> C(nhi, std::vector<double>(nhi));
  {
    std::ifstream in(dir + "/c_matrix_plik_v22.dat", std::ios::binary);
    if (!in) throw std::runtime_error("cannot open " + dir + "/c_matrix_plik_v22.dat");
    std::int32_t rec = 0;
    in.read(reinterpret_cast<char*>(&rec), 4);
    if (rec != std::int32_t(sizeof(double) * nhi * nhi)) throw std::runtime_error("plik_lite: bad covariance record");
    std::vector<double> buf(size_t(nhi) * nhi);
    in.read(reinterpret_cast<char*>(buf.data()), std::streamsize(buf.size() * sizeof(double)));
    if (!in) throw std::runtime_error("plik_lite: covariance file too short");
    for (int i = 0; i < nhi; i++)
      for (int j = 0; j < nhi; j++) C[i][j] = buf[size_t(i) * nhi + j];
    for (int i = 0; i < nhi; i++)
      for (int j = i; j < nhi; j++) C[i][j] = C[j][i];
  }

  std::vector<double> mean;
  std::vector<double> lo_sig;
  if (low_ell) {
    const auto lmn = read_column(dir + "/blmin_low_ell.dat", 0), lmx = read_column(dir + "/blmax_low_ell.dat", 0);
    const auto lw = read_column(dir + "/bweight_low_ell.dat", 0);
    const auto lX = read_column(dir + "/CTT_bin_low_ell_2018.dat", 1), lS = read_column(dir + "/CTT_bin_low_ell_2018.dat", 2);
    for (size_t i = 0; i < lX.size(); i++) {
      bins_tt.push_back(make_bin(int(i), 2, lmn, lmx, lw));
      mean.push_back(lX[i]);
      lo_sig.push_back(lS[i]);
    }
  }
  for (int i = 0; i < NTT; i++) bins_tt.push_back(make_bin(i, PLMIN, blmin, blmax, bw));
  for (int i = 0; i < NTT; i++) mean.push_back(X[i]);
  int n_hi = NTT;
  if (!tt_only) {
    for (int i = 0; i < NTE; i++) bins_pol.push_back(make_bin(i, PLMIN, blmin, blmax, bw));
    for (int i = NTT; i < nhi; i++) mean.push_back(X[i]);
    n_hi = nhi;
  }
  const size_t nlo = lo_sig.size(), n = nlo + n_hi;
  std::vector<std::vector<double>> cov(n, std::vector<double>(n, 0.0));
  for (size_t i = 0; i < nlo; i++) cov[i][i] = lo_sig[i] * lo_sig[i];
  for (int i = 0; i < n_hi; i++)
    for (int j = 0; j < n_hi; j++) cov[nlo + i][nlo + j] = C[i][j];
  data.set(mean, cov);
}

std::string PlanckLiteLikelihood::name() const {
  return std::string("planck_lite_") + (tt_only ? "TT" : "TTTEEE") + (low_ell ? "_lowl" : "");
}

double PlanckLiteLikelihood::loglike(const std::vector<double>& Dtt, const std::vector<double>& Dte,
                                     const std::vector<double>& Dee, double A) const {
  if (int(Dtt.size()) <= 2508 || (!tt_only && (Dte.size() <= 1996 || Dee.size() <= 1996)))
    throw std::invalid_argument("planck_lite: spectra needed up to ell = 2508");
  auto binned = [](const Bin& b, const std::vector<double>& D) {
    double v = 0.0;
    for (int l = b.lmin; l <= b.lmax; l++) v += b.w[l - b.lmin] * D[l] * 2.0 * M_PI / (l * (l + 1.0));
    return v;
  };
  const double cal = 1.0 / (A * A);
  std::vector<double> m;
  m.reserve(data.size());
  for (const auto& b : bins_tt) m.push_back(binned(b, Dtt) * cal);
  for (const auto& b : bins_pol) m.push_back(binned(b, Dte) * cal);
  for (const auto& b : bins_pol) m.push_back(binned(b, Dee) * cal);
  return -0.5 * data.chi2(m);
}

double PlanckLiteLikelihood::loglike(const Theory& t) const {
  const auto& c = t.cmb();
  return loglike(c.tt, c.te, c.ee, t.nuisance("A_planck", 1.0));
}

// =============================================================================
// Planck 2018 lensing (CMBlikes format)
// =============================================================================

PlanckLensingLikelihood::PlanckLensingLikelihood(const std::string& dir, bool cmb_marged) : marged(cmb_marged) {
  const std::string stem = dir + "/smicadx12_Dec5_ftl_mv2_ndclpp_p_teb_consext8";
  const std::string var = marged ? stem + "_CMBmarged" : stem;
  // bandpowers: bin, L_min, L_max, L_av, PP, error, Ahat
  std::vector<double> pp_hat;
  for (const auto& r : read_rows(stem + "_bandpowers.dat")) {
    if (r.size() < 5) throw std::runtime_error("planck_lensing: bad bandpower file");
    pp_hat.push_back(r[4]);
  }
  const size_t nb = pp_hat.size();
  const auto cov = read_rows(var + "_cov.dat");
  if (cov.size() != nb) throw std::runtime_error("planck_lensing: covariance does not match the bandpowers");
  data.set(pp_hat, cov);
  const int nL = lmax - lmin + 1;
  // window files: L, then one column per spectrum (in_order)
  auto windows = [&](const std::string& d, size_t ncol, std::vector<std::vector<std::vector<double>>*> out) {
    for (auto* o : out) o->assign(nb, std::vector<double>(nL, 0.0));
    for (size_t b = 0; b < nb; b++) {
      for (const auto& r : read_rows(d + "/window" + std::to_string(b + 1) + ".dat")) {
        if (r.size() != ncol + 1) throw std::runtime_error("planck_lensing: unexpected columns in " + d);
        const int L = int(std::lround(r[0]));
        if (L < lmin || L > lmax) continue;
        for (size_t k = 0; k < ncol; k++) (*out[k])[b][size_t(L - lmin)] = r[k + 1];
      }
    }
  };
  windows(stem + "_window", 1, {&w_pp});
  if (marged) windows(var + "_lens_delta_window", 1, {&c_pp});
  else windows(var + "_lens_delta_window", 4, {&c_tt, &c_ee, &c_te, &c_pp});   // in_order TT EE TE PP
  for (const auto& r : read_rows(var + "_lensing_fiducial_correction.dat")) fid_corr.push_back(r.at(1));
  if (fid_corr.size() != nb) throw std::runtime_error("planck_lensing: fiducial correction does not match the bandpowers");
}

std::vector<double> PlanckLensingLikelihood::binned_theory(const std::vector<double>& Dtt, const std::vector<double>& Dte,
                                                           const std::vector<double>& Dee, const std::vector<double>& pp,
                                                           double A_planck) const {
  if (int(pp.size()) <= lmax || (!marged && (int(Dtt.size()) <= lmax || int(Dee.size()) <= lmax || int(Dte.size()) <= lmax)))
    throw std::invalid_argument("planck_lensing: spectra needed up to ell = 2500");
  const double ic = 1.0 / (A_planck * A_planck);   // calibration of the CMB spectra (T and E maps)
  std::vector<double> th(w_pp.size(), 0.0);
  for (size_t b = 0; b < th.size(); b++) {
    double s = -fid_corr[b];
    for (int L = lmin; L <= lmax; L++) {
      const size_t i = size_t(L - lmin);
      s += (w_pp[b][i] + c_pp[b][i]) * pp[L];
      if (!marged) s += ic * (c_tt[b][i] * Dtt[L] + c_ee[b][i] * Dee[L] + c_te[b][i] * Dte[L]);
    }
    th[b] = s;
  }
  return th;
}

double PlanckLensingLikelihood::loglike(const std::vector<double>& Dtt, const std::vector<double>& Dte,
                                        const std::vector<double>& Dee, const std::vector<double>& pp, double A_planck) const {
  return -0.5 * data.chi2(binned_theory(Dtt, Dte, Dee, pp, A_planck));
}

double PlanckLensingLikelihood::loglike(const Theory& t) const {
  const auto& c = t.cmb();
  return loglike(c.tt, c.te, c.ee, c.pp, marged ? 1.0 : t.nuisance("A_planck", 1.0));
}

// =============================================================================
// Mock CMB data
// =============================================================================

CMBMockLikelihood::CMBMockLikelihood(const std::string& file, int lmax_in, double fsky_in, double noise, double beam,
                                     bool use_bb)
  : lmax(lmax_in), fsky(fsky_in), bb(use_bb) {
  if (lmax < 2 || !(fsky > 0.0 && fsky <= 1.0) || noise < 0.0 || beam < 0.0)
    throw std::invalid_argument("cmb_mock: need lmax >= 2, 0 < fsky <= 1, noise >= 0, beam >= 0");
  for (auto* v : {&dTT, &dEE, &dTE, &dBB, &nT, &nP}) v->assign(lmax + 1, 0.0);
  std::vector<bool> have(lmax + 1, false);
  for (const auto& r : read_rows(file)) {   // ell, D_TT, D_EE, D_TE, ... (cells_lensed.txt or cells.txt)
    if (r.size() < (bb ? 5u : 4u)) throw std::runtime_error("cmb_mock: expected ell, TT, EE, TE" + std::string(bb ? ", BB" : "") + " in " + file);
    const int l = int(std::lround(r[0]));
    if (l < 2 || l > lmax) continue;
    const double f = 2.0 * M_PI / (l * (l + 1.0));
    dTT[l] = r[1] * f; dEE[l] = r[2] * f; dTE[l] = r[3] * f;
    if (bb) dBB[l] = r[4] * f;
    have[l] = true;
  }
  for (int l = 2; l <= lmax; l++)
    if (!have[l]) throw std::runtime_error("cmb_mock: " + file + " has no ell = " + std::to_string(l));
  const double arcmin = M_PI / (180.0 * 60.0), sb = beam * arcmin / std::sqrt(8.0 * std::log(2.0));
  for (int l = 2; l <= lmax; l++) {
    nT[l] = std::pow(noise * arcmin, 2) * std::exp(l * (l + 1.0) * sb * sb);
    nP[l] = 2.0 * nT[l];
    dTT[l] += nT[l];
    dEE[l] += nP[l];
    dBB[l] += nP[l];
  }
}

double CMBMockLikelihood::loglike(const std::vector<double>& Dtt, const std::vector<double>& Dte,
                                  const std::vector<double>& Dee, const std::vector<double>& Dbb) const {
  if (int(Dtt.size()) <= lmax || int(Dte.size()) <= lmax || int(Dee.size()) <= lmax || (bb && int(Dbb.size()) <= lmax))
    throw std::invalid_argument("cmb_mock: spectra needed up to ell = " + std::to_string(lmax));
  double m2lnL = 0.0;
  for (int l = 2; l <= lmax; l++) {
    const double f = 2.0 * M_PI / (l * (l + 1.0));
    const double T = Dtt[l] * f + nT[l], E = Dee[l] * f + nP[l], X = Dte[l] * f;
    const double detC = T * E - X * X, detD = dTT[l] * dEE[l] - dTE[l] * dTE[l];
    if (!(detC > 0.0)) return -std::numeric_limits<double>::infinity();
    const double tr = (dTT[l] * E + dEE[l] * T - 2.0 * dTE[l] * X) / detC;
    m2lnL += (2.0 * l + 1.0) * fsky * (tr + std::log(detC / detD) - 2.0);
    if (bb) {
      const double B = Dbb[l] * f + nP[l];
      if (!(B > 0.0)) return -std::numeric_limits<double>::infinity();
      m2lnL += (2.0 * l + 1.0) * fsky * (dBB[l] / B + std::log(B / dBB[l]) - 1.0);
    }
  }
  return -0.5 * m2lnL;
}

double CMBMockLikelihood::loglike(const Theory& t) const {
  const auto& c = t.cmb();
  return loglike(c.tt, c.te, c.ee, c.bb);
}

// =============================================================================

namespace Likelihoods {
  std::vector<std::string> names() {
    return {"sn", "sn_free_amplitude", "pantheonplus", "desy5", "union3", "desi_dr2", "desi_dr1", "planck_dp", "bbn", "planck_lite_TTTEEE",
            "planck_lite_TT", "planck_lite_TTTEEE_lowl", "planck_lite_TT_lowl", "planck_lensing", "planck_lensing_marged",
            "cmb_mock"};
  }
  std::vector<std::string> option_names() {
    return {"cmb_mock_file", "cmb_mock_lmax", "cmb_mock_fsky", "cmb_mock_noise", "cmb_mock_beam", "cmb_mock_bb"};
  }

  std::unique_ptr<Likelihood> create(const std::string& n, const std::map<std::string, std::string>& o) {
    for (const auto& kv : o) {
      bool known = false;
      for (const auto& on : option_names()) known = known || on == kv.first;
      if (!known) throw std::invalid_argument("unknown likelihood option '" + kv.first + "'");
    }
    auto opt = [&](const std::string& k, const std::string& def) { return o.count(k) ? o.at(k) : def; };
    if (n == "sn") return std::make_unique<SupernovaLikelihood>();
    if (n == "pantheonplus") return SNIaLikelihood::pantheonplus();
    if (n == "desy5") return SNIaLikelihood::desy5();
    if (n == "union3") return SNIaLikelihood::union3();
    if (n == "sn_free_amplitude") return std::make_unique<SupernovaLikelihood>(Utils::fml_file("CosmologyFitting/external/supernovadata.txt"), true);
    if (n == "desi_dr2")
      return std::make_unique<BAOLikelihood>("desi_dr2", Utils::fml_file("CosmologyFitting/external/bao/desi_dr2_mean.txt"), Utils::fml_file("CosmologyFitting/external/bao/desi_dr2_cov.txt"));
    if (n == "desi_dr1")
      return std::make_unique<BAOLikelihood>("desi_dr1", Utils::fml_file("CosmologyFitting/external/bao/desi_dr1_mean.txt"), Utils::fml_file("CosmologyFitting/external/bao/desi_dr1_cov.txt"));
    if (n == "planck_dp") return std::make_unique<PlanckDistancePriors>();
    if (n == "bbn") return std::make_unique<BBNPrior>();
    const std::string pl = Utils::fml_file("CosmologyFitting/external/planck2018_lite");
    if (n == "planck_lite_TTTEEE") return std::make_unique<PlanckLiteLikelihood>(pl, false, false);
    if (n == "planck_lite_TT") return std::make_unique<PlanckLiteLikelihood>(pl, true, false);
    if (n == "planck_lite_TTTEEE_lowl") return std::make_unique<PlanckLiteLikelihood>(pl, false, true);
    if (n == "planck_lite_TT_lowl") return std::make_unique<PlanckLiteLikelihood>(pl, true, true);
    if (n == "planck_lensing") return std::make_unique<PlanckLensingLikelihood>(Utils::fml_file("CosmologyFitting/external/planck2018_lensing"), false);
    if (n == "planck_lensing_marged") return std::make_unique<PlanckLensingLikelihood>(Utils::fml_file("CosmologyFitting/external/planck2018_lensing"), true);
    if (n == "cmb_mock") {
      const std::string b = opt("cmb_mock_bb", "false");
      if (b != "true" && b != "false") throw std::invalid_argument("cmb_mock_bb must be true or false");
      return std::make_unique<CMBMockLikelihood>(opt("cmb_mock_file", "data/cmb_mock/cells_lensed.txt"),
                                                 std::stoi(opt("cmb_mock_lmax", "2500")), std::stod(opt("cmb_mock_fsky", "0.6")),
                                                 std::stod(opt("cmb_mock_noise", "33")), std::stod(opt("cmb_mock_beam", "7")),
                                                 b == "true");
    }
    std::string all;
    for (const auto& s : names()) all += (all.empty() ? "" : ", ") + s;
    throw std::invalid_argument("unknown likelihood '" + n + "' (available: " + all + ")");
  }
}

} // namespace FML::COSMOLOGY::FITTING
