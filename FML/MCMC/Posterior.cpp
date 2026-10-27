#include <FML/MCMC/Posterior.h>

#include <cmath>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>

namespace FML::MCMC {

Posterior::Posterior(std::vector<SampledParameter> params, std::vector<std::string> derived_names,
                     std::vector<std::string> derived_labels, LikelihoodFunction like_in)
  : par(std::move(params)), dnames(std::move(derived_names)), dlabels(std::move(derived_labels)),
    like(std::move(like_in)) {
  if (par.empty()) throw std::invalid_argument("Posterior: no sampled parameters");
  for (const auto& p : par) {
    if (!(p.min < p.max) || p.start < p.min || p.start > p.max || !(p.proposal > 0.0))
      throw std::invalid_argument("Posterior: parameter " + p.name + " needs min < start < max and proposal > 0");
    if (p.gaussian_prior && !(p.prior_sigma > 0.0))
      throw std::invalid_argument("Posterior: Gaussian prior on " + p.name + " needs sigma > 0");
    all_names.push_back(p.name);
  }
  if (dlabels.size() != dnames.size()) dlabels = dnames;
  for (const auto& d : dnames) all_names.push_back(d);
  nslow = par.size();
  for (size_t i = 0; i < par.size(); i++)
    if (par[i].fast) { nslow = i; break; }
  for (size_t i = nslow; i < par.size(); i++)
    if (!par[i].fast) throw std::invalid_argument("Posterior: fast parameters must come after the slow ones");
  if (nslow == 0) nslow = par.size();   // all fast: no distinction
}

double Posterior::log_prior(const std::vector<double>& x) const {
  double lp = 0.0;
  for (size_t i = 0; i < par.size(); i++) {
    if (!(x[i] >= par[i].min && x[i] <= par[i].max)) return -std::numeric_limits<double>::infinity();
    if (par[i].gaussian_prior) {
      const double d = (x[i] - par[i].prior_mean) / par[i].prior_sigma;
      lp -= 0.5 * d * d;
    }
  }
  return lp;
}

Posterior::Point Posterior::evaluate(const std::vector<double>& x) const {
  Point p;
  const double lpr = log_prior(x);
  if (!std::isfinite(lpr)) return p;
  LikelihoodResult r;
  try { r = like(x); } catch (const std::exception&) { return p; }
  if (!std::isfinite(r.loglike)) return p;
  p.loglike = r.loglike;
  p.logpost = lpr + r.loglike;
  p.derived = std::move(r.derived);
  return p;
}

std::vector<double> Posterior::start() const {
  std::vector<double> x;
  for (const auto& p : par) x.push_back(p.start);
  return x;
}

std::vector<std::vector<double>> Posterior::diagonal_covariance() const {
  const size_t d = par.size();
  std::vector<std::vector<double>> C(d, std::vector<double>(d, 0.0));
  for (size_t i = 0; i < d; i++) C[i][i] = par[i].proposal * par[i].proposal;
  return C;
}

void Posterior::write_paramnames(const std::string& root) const {
  {
    std::ofstream f(root + ".paramnames");
    if (!f) throw std::runtime_error("cannot write " + root + ".paramnames");
    for (const auto& p : par) f << p.name << "\t" << (p.label.empty() ? p.name : p.label) << "\n";
    for (size_t i = 0; i < dnames.size(); i++) f << dnames[i] << "*\t" << dlabels[i] << "\n";
  }
  {
    std::ofstream f(root + ".ranges");
    f << std::setprecision(10);
    for (const auto& p : par) f << p.name << " " << p.min << " " << p.max << "\n";
    for (const auto& n : dnames) f << n << " N N\n";
  }
}

bool cholesky(const std::vector<std::vector<double>>& A, std::vector<std::vector<double>>& L) {
  const size_t n = A.size();
  L.assign(n, std::vector<double>(n, 0.0));
  for (size_t i = 0; i < n; i++)
    for (size_t j = 0; j <= i; j++) {
      double s = A[i][j];
      for (size_t k = 0; k < j; k++) s -= L[i][k] * L[j][k];
      if (i == j) {
        if (!(s > 0.0)) return false;
        L[i][i] = std::sqrt(s);
      } else {
        L[i][j] = s / L[j][j];
      }
    }
  return true;
}

bool spd_inverse(const std::vector<std::vector<double>>& A, std::vector<std::vector<double>>& inv) {
  std::vector<std::vector<double>> L;
  if (!cholesky(A, L)) return false;
  const size_t n = A.size();
  inv.assign(n, std::vector<double>(n, 0.0));
  for (size_t c = 0; c < n; c++) {   // solve L L^T x = e_c
    std::vector<double> y(n), x(n);
    for (size_t i = 0; i < n; i++) {
      double s = (i == c) ? 1.0 : 0.0;
      for (size_t k = 0; k < i; k++) s -= L[i][k] * y[k];
      y[i] = s / L[i][i];
    }
    for (size_t i = n; i-- > 0;) {
      double s = y[i];
      for (size_t k = i + 1; k < n; k++) s -= L[k][i] * x[k];
      x[i] = s / L[i][i];
    }
    for (size_t i = 0; i < n; i++) inv[i][c] = x[i];
  }
  return true;
}

std::vector<std::vector<double>> read_covmat(const std::string& file, const std::vector<SampledParameter>& par,
                                                       std::vector<std::string>& missing) {
  std::ifstream in(file);
  if (!in) throw std::runtime_error("cannot open covmat " + file);
  std::string line;
  std::getline(in, line);
  if (line.empty() || line[0] != '#') throw std::runtime_error(file + ": first line must be '# name1 name2 ...'");
  std::istringstream hs(line.substr(1));
  std::vector<std::string> names;
  std::string n;
  while (hs >> n) names.push_back(n);
  std::vector<std::vector<double>> M;
  while (std::getline(in, line)) {
    std::istringstream is(line);
    std::vector<double> r;
    double v;
    while (is >> v) r.push_back(v);
    if (!r.empty()) M.push_back(r);
  }
  if (M.size() != names.size()) throw std::runtime_error(file + ": matrix size does not match the names");
  for (const auto& r : M) if (r.size() != names.size()) throw std::runtime_error(file + ": matrix not square");
  const size_t d = par.size();
  std::vector<std::vector<double>> C(d, std::vector<double>(d, 0.0));
  std::vector<int> idx(d, -1);
  for (size_t i = 0; i < d; i++) {
    for (size_t k = 0; k < names.size(); k++) if (names[k] == par[i].name) idx[i] = int(k);
    if (idx[i] < 0) missing.push_back(par[i].name);
  }
  for (size_t i = 0; i < d; i++)
    for (size_t j = 0; j < d; j++) {
      if (idx[i] >= 0 && idx[j] >= 0) C[i][j] = M[idx[i]][idx[j]];
      else if (i == j) C[i][i] = par[i].proposal * par[i].proposal;
    }
  return C;
}

void write_covmat(const std::string& file, const std::vector<SampledParameter>& par,
                            const std::vector<std::vector<double>>& C) {
  std::ofstream f(file);
  if (!f) throw std::runtime_error("cannot write " + file);
  f << "#";
  for (const auto& p : par) f << " " << p.name;
  f << "\n" << std::setprecision(10);
  for (const auto& r : C) {
    for (size_t j = 0; j < r.size(); j++) f << (j ? " " : "") << r[j];
    f << "\n";
  }
}

} // namespace FML::MCMC
