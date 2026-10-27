#include <FML/MCMC/Sampler.h>
#include <FML/MCMC/Parallel.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace FML::MCMC {

namespace {
  constexpr double NEG_INF = -std::numeric_limits<double>::infinity();

  std::vector<std::vector<double>> sample_cov(const std::vector<std::vector<double>>& xs, size_t from, size_t d) {
    std::vector<double> m(d, 0.0);
    const size_t n = xs.size() - from;
    for (size_t k = from; k < xs.size(); k++)
      for (size_t i = 0; i < d; i++) m[i] += xs[k][i] / n;
    std::vector<std::vector<double>> c(d, std::vector<double>(d, 0.0));
    for (size_t k = from; k < xs.size(); k++)
      for (size_t i = 0; i < d; i++)
        for (size_t j = 0; j < d; j++) c[i][j] += (xs[k][i] - m[i]) * (xs[k][j] - m[j]) / (n - 1);
    return c;
  }
}

Sampler::Sampler(Posterior posterior, SamplerOptions options) : post(std::move(posterior)), par(post.params()), opt(options) {
  if (opt.chains < 1 || opt.burn_in < 0 || opt.check_every < 1 || opt.adapt_every < 1)
    throw std::invalid_argument("Sampler: invalid options");
  if (opt.chains < Parallel::size())
    throw std::invalid_argument("Sampler: fewer chains than MPI processes");
  n_slow = post.n_slow();
  if (opt.oversample_fast < 0) throw std::invalid_argument("Sampler: oversample_fast must be >= 0");
  if (opt.resume && opt.output_root.empty()) throw std::invalid_argument("Sampler: resume needs output_root");
  if (!opt.initial_covmat.empty()) {
    std::vector<std::vector<double>> L;
    bool ok = opt.initial_covmat.size() == par.size();
    for (const auto& r : opt.initial_covmat) ok = ok && r.size() == par.size();
    if (!ok || !cholesky(opt.initial_covmat, L))
      throw std::invalid_argument("Sampler: initial_covmat must be a positive-definite d x d matrix");
  }
}

bool Sampler::is_root() const { return Parallel::rank() == 0; }

void Sampler::set_proposal(Chain& c, const std::vector<std::vector<double>>& cov) const {
  std::vector<std::vector<double>> L;
  if (cholesky(cov, L)) c.L = L;   // otherwise keep the previous proposal
}

void Sampler::init_chain(Chain& c, int index) {
  const size_t d = par.size();
  std::seed_seq seq{opt.seed, std::uint64_t(index), std::uint64_t(0x9e3779b97f4a7c15ULL)};
  c.rng.seed(seq);
  std::normal_distribution<double> N(0.0, 1.0);
  c.index = index;
  std::vector<std::vector<double>> cov(d, std::vector<double>(d, 0.0));
  for (size_t i = 0; i < d; i++) cov[i][i] = par[i].proposal * par[i].proposal;
  if (!opt.initial_covmat.empty()) cov = opt.initial_covmat;
  set_proposal(c, cov);
  for (int tries = 0; tries < 1000; tries++) {
    std::vector<double> x(d);
    for (size_t i = 0; i < d; i++)
      x[i] = std::clamp(par[i].start + 0.5 * par[i].proposal * N(c.rng), par[i].min, par[i].max);
    const auto pt = post.evaluate(x);
    if (!pt.valid()) continue;
    c.x = x;
    c.derived = pt.derived;
    c.ll = pt.loglike;
    c.lp = pt.logpost;
    c.best_x = x; c.best_derived = pt.derived; c.best_lp = c.lp; c.best_ll = c.ll;
    return;
  }
  throw std::runtime_error("Sampler: no valid starting point found near the start values");
}

void Sampler::adapt(Chain& c) const {
  const size_t d = par.size();
  const double rate = double(c.accepted_window) / opt.adapt_every;
  c.accepted_window = 0;
  c.scale *= std::exp(2.0 * (rate - 0.25));
  c.scale = std::clamp(c.scale, 1e-3, 1e3);
  if (n_slow < par.size() && opt.oversample_fast > 0) {
    const double rf = double(c.accepted_window_fast) / (double(opt.adapt_every) * opt.oversample_fast);
    c.accepted_window_fast = 0;
    c.scale_fast = std::clamp(c.scale_fast * std::exp(2.0 * (rf - 0.25)), 1e-3, 1e3);
  }
  const size_t n = c.history.size();
  if (n < std::max<size_t>(10 * d, 200)) return;
  auto cov = sample_cov(c.history, n / 5, d);   // drop the first 20% (the approach to the typical set)
  const double f = 2.38 * 2.38 / d;
  for (size_t i = 0; i < d; i++) {
    for (size_t j = 0; j < d; j++) cov[i][j] *= f;
    cov[i][i] += 1e-12 * par[i].proposal * par[i].proposal;
  }
  set_proposal(c, cov);
}

bool Sampler::mh_update(Chain& c, bool fast_only) {
  const size_t d = par.size();
  std::normal_distribution<double> N(0.0, 1.0);
  std::uniform_real_distribution<double> U(0.0, 1.0);
  std::vector<double> z(d, 0.0), y(d);
  for (size_t i = fast_only ? n_slow : 0; i < d; i++) z[i] = N(c.rng);
  const double sq = std::sqrt(fast_only ? c.scale_fast : c.scale);
  for (size_t i = 0; i < d; i++) {
    double v = 0.0;
    for (size_t j = fast_only ? n_slow : 0; j <= i; j++) v += c.L[i][j] * z[j];
    y[i] = c.x[i] + sq * v;
  }
  auto pt = post.evaluate(y);
  if (!pt.valid()) return false;
  const double lp = pt.logpost;
  if (!(std::log(U(c.rng)) < lp - c.lp)) return false;
  c.x = y;
  c.derived = std::move(pt.derived);
  c.lp = lp;
  c.ll = pt.loglike;
  if (lp > c.best_lp) { c.best_lp = lp; c.best_ll = c.ll; c.best_x = y; c.best_derived = c.derived; }
  return true;
}

void Sampler::advance(Chain& c, int n_steps, bool burn) {
  const bool has_fast = n_slow < par.size() && opt.oversample_fast > 0;
  for (int s = 0; s < n_steps; s++) {
    const bool acc = mh_update(c, false);
    if (has_fast)
      for (int f = 0; f < opt.oversample_fast; f++) {
        const bool af = mh_update(c, true);
        c.steps_fast++;
        if (af) { c.accepted_fast++; c.accepted_window_fast++; }
      }
    c.steps++;
    if (acc) { c.accepted++; c.accepted_window++; }
    if (burn) {
      c.history.push_back(c.x);
      if (c.steps % opt.adapt_every == 0) adapt(c);
    } else {
      c.steps_sampling++;
      if (acc) c.accepted_sampling++;
      std::vector<double> row = c.x;
      row.insert(row.end(), c.derived.begin(), c.derived.end());
      c.rows.push_back(std::move(row));
      c.rows_lp.push_back(c.lp);
    }
  }
}

// Per chain n, means and variances of the sampled parameters, gathered over the MPI processes
double Sampler::gelman_rubin() const {
  const size_t d = par.size();
  std::vector<double> local;
  for (const auto& c : chains) {
    const size_t n = c.rows.size();
    local.push_back(double(n));
    std::vector<double> mean(d, 0.0), var(d, 0.0);
    for (size_t k = 0; k < n; k++)
      for (size_t p = 0; p < d; p++) mean[p] += c.rows[k][p] / n;
    for (size_t k = 0; k < n; k++)
      for (size_t p = 0; p < d; p++) var[p] += std::pow(c.rows[k][p] - mean[p], 2) / (n - 1.0);
    local.insert(local.end(), mean.begin(), mean.end());
    local.insert(local.end(), var.begin(), var.end());
  }
  const auto all = Parallel::allgather(local);
  const size_t stride = 1 + 2 * d, m = all.size() / stride;
  if (m < 2) return std::numeric_limits<double>::infinity();
  const double n = all[0];
  if (n < 10) return std::numeric_limits<double>::infinity();
  double worst = 0.0;
  for (size_t p = 0; p < d; p++) {
    double W = 0.0, gm = 0.0, B = 0.0;
    for (size_t c = 0; c < m; c++) { W += all[c * stride + 1 + d + p] / m; gm += all[c * stride + 1 + p] / m; }
    for (size_t c = 0; c < m; c++) B += std::pow(all[c * stride + 1 + p] - gm, 2) / (m - 1);   // B/n of Gelman-Rubin
    const double vplus = (n - 1.0) / n * W + B;
    worst = std::max(worst, vplus / W - 1.0);
  }
  return worst;
}

void Sampler::run() {
  const int rank = Parallel::rank(), nproc = Parallel::size();
  std::vector<int> mine;   // global indices of the chains of this process
  for (int i = rank; i < opt.chains; i += nproc) mine.push_back(i);
  chains.clear();
  chains.resize(mine.size());
  int nt = opt.threads > 0 ? opt.threads : int(mine.size());
#ifdef _OPENMP
  nt = std::max(1, std::min(nt, omp_get_max_threads()));
#else
  nt = 1;
#endif
  const int nc = int(chains.size());
  const bool talk = opt.verbose && is_root();
  // acceptance of the chains so far (all processes)
  auto mean_acceptance = [&](bool sampling) {
    std::vector<double> v(2, 0.0);
    for (const auto& c : chains) {
      v[0] += sampling ? c.accepted_sampling : c.accepted;
      v[1] += sampling ? c.steps_sampling : c.steps;
    }
    Parallel::allreduce_sum(v);
    return v[1] > 0 ? v[0] / v[1] : 0.0;
  };
  // Start, or resume from the checkpoints (all chains or none)
  bool resumed = false;
  if (opt.resume) {
    std::vector<double> got(1, 0.0);
    for (int i = 0; i < nc; i++) got[0] += load_checkpoint(chains[i], mine[i]) ? 1.0 : 0.0;
    Parallel::allreduce_sum(got);
    if (int(got[0]) == opt.chains) resumed = true;
    else if (got[0] > 0.0) throw std::runtime_error("Sampler: checkpoints found for only some of the chains");
  }
  if (!resumed) {
#pragma omp parallel for schedule(dynamic, 1) num_threads(nt)
    for (int i = 0; i < nc; i++) init_chain(chains[i], mine[i]);
  }
  // all chains advance in lockstep: the progress is the (common) step count
  std::vector<double> st(1, nc > 0 ? double(chains[0].steps) : 0.0);
  Parallel::allreduce_sum(st);
  const long steps0 = long(st[0] / nproc + 0.5);
  if (resumed && talk)
    std::cout << "  resumed from " << opt.output_root << "_*.checkpoint at step " << steps0 << " per chain\n" << std::flush;
  const bool ckpt = opt.checkpoint && !opt.output_root.empty();
  auto checkpoint = [&]() {
    if (!ckpt) return;
    for (const auto& c : chains) write_checkpoint(c);
  };

  // Burn-in with adaptation
  int done = int(std::min<long>(steps0, opt.burn_in));
  while (done < opt.burn_in) {
    const int n = std::min(opt.check_every, opt.burn_in - done);
#pragma omp parallel for schedule(dynamic, 1) num_threads(nt)
    for (int i = 0; i < nc; i++) advance(chains[i], n, true);
    done += n;
    const double acc = mean_acceptance(false);
    if (talk)
      std::cout << "  burn-in " << done << "/" << opt.burn_in << " steps per chain, acceptance " << std::setprecision(3)
                << acc << "\n" << std::flush;
    if (done < opt.burn_in) checkpoint();
  }
  // Freeze the proposal: covariance of the second half of the burn-in of all chains together
  const size_t d = par.size();
  if (opt.burn_in > 0 && !(nc > 0 && chains[0].frozen)) {
    // sums over all processes: [count, sum scale, sum scale_fast, sum x (d), sum x x^T (d^2)]
    std::vector<double> S(3 + d + d * d, 0.0);
    for (const auto& c : chains) {
      S[1] += c.scale;
      S[2] += c.scale_fast;
      for (size_t k = c.history.size() / 2; k < c.history.size(); k++) {
        const auto& x = c.history[k];
        S[0] += 1.0;
        for (size_t i = 0; i < d; i++) {
          S[3 + i] += x[i];
          for (size_t j = 0; j < d; j++) S[3 + d + i * d + j] += x[i] * x[j];
        }
      }
    }
    Parallel::allreduce_sum(S);
    const double N = S[0];
    const double scale = S[1] / opt.chains, scale_fast = S[2] / opt.chains;
    for (auto& c : chains) { c.scale = scale; c.scale_fast = scale_fast; }
    if (N > 2.0 * d + 2.0) {
      std::vector<std::vector<double>> cov(d, std::vector<double>(d));
      const double f = 2.38 * 2.38 / d;
      for (size_t i = 0; i < d; i++)
        for (size_t j = 0; j < d; j++)
          cov[i][j] = f * (S[3 + d + i * d + j] - S[3 + i] * S[3 + j] / N) / (N - 1.0);
      for (size_t i = 0; i < d; i++) cov[i][i] += 1e-12 * par[i].proposal * par[i].proposal;
      for (auto& c : chains) set_proposal(c, cov);
    }
  }
  for (auto& c : chains) { c.frozen = true; c.history.clear(); c.history.shrink_to_fit(); }
  checkpoint();

  // Sampling
  long steps = std::max<long>(steps0, opt.burn_in);
  if (resumed && steps > opt.burn_in) {
    R1 = gelman_rubin();
    if (steps - opt.burn_in >= opt.min_samples && R1 < opt.Rminus1_stop) steps = opt.max_steps;   // already done
  }
  while (true) {
    const int n = int(std::min<long>(opt.check_every, opt.max_steps - steps));
    if (n <= 0) break;
#pragma omp parallel for schedule(dynamic, 1) num_threads(nt)
    for (int i = 0; i < nc; i++) advance(chains[i], n, false);
    steps += n;
    R1 = gelman_rubin();
    const double acc = mean_acceptance(true);
    checkpoint();
    double acc_fast = -1.0;
    if (n_slow < par.size() && opt.oversample_fast > 0) {
      std::vector<double> v(2, 0.0);
      for (const auto& c : chains) { v[0] += c.accepted_fast; v[1] += c.steps_fast; }
      Parallel::allreduce_sum(v);
      acc_fast = v[1] > 0 ? v[0] / v[1] : 0.0;
    }
    if (talk) {
      std::cout << "  " << steps - opt.burn_in << " samples per chain, acceptance " << std::setprecision(3) << acc;
      if (acc_fast >= 0.0) std::cout << " (fast " << acc_fast << ")";
      std::cout << ", R-1 = " << R1 << "\n" << std::flush;
    }
    if (steps - opt.burn_in >= opt.min_samples && R1 < opt.Rminus1_stop) break;
  }
  gather_chains();
  // Best fit over all chains
  for (const auto& c : chains)
    if (c.best_lp > best_lp) {
      best_lp = c.best_lp;
      best_ll = c.best_ll;
      best = c.best_x;
      best.insert(best.end(), c.best_derived.begin(), c.best_derived.end());
    }
  if (!opt.output_root.empty() && is_root()) write_output();
  Parallel::barrier();
}

// Every process gets all chains: [index, steps, accepted, steps_sampling, accepted_sampling, best_lp, best_ll,
// n_rows, best_x (d), best_derived (nd), rows (n_rows (d + nd)), rows_lp (n_rows)] per chain
void Sampler::gather_chains() {
  if (Parallel::size() == 1) return;
  const size_t d = par.size(), nd = post.derived_names().size(), w = d + nd;
  std::vector<double> local;
  for (const auto& c : chains) {
    local.insert(local.end(), {double(c.index), double(c.steps), double(c.accepted), double(c.steps_sampling),
                               double(c.accepted_sampling), c.best_lp, c.best_ll, double(c.rows.size())});
    std::vector<double> bx = c.best_x, bd = c.best_derived;
    bx.resize(d, 0.0);
    bd.resize(nd, 0.0);
    local.insert(local.end(), bx.begin(), bx.end());
    local.insert(local.end(), bd.begin(), bd.end());
    for (const auto& r : c.rows) {
      if (r.size() != w) throw std::runtime_error("Sampler: inconsistent row length");
      local.insert(local.end(), r.begin(), r.end());
    }
    local.insert(local.end(), c.rows_lp.begin(), c.rows_lp.end());
  }
  const auto all = Parallel::allgather(local);
  std::vector<Chain> out(opt.chains);
  size_t k = 0;
  while (k < all.size()) {
    const int idx = int(all[k]);
    Chain& c = out.at(idx);
    c.index = idx;
    c.steps = long(all[k + 1]); c.accepted = long(all[k + 2]);
    c.steps_sampling = long(all[k + 3]); c.accepted_sampling = long(all[k + 4]);
    c.best_lp = all[k + 5]; c.best_ll = all[k + 6];
    const size_t nr = size_t(all[k + 7]);
    k += 8;
    c.best_x.assign(all.begin() + k, all.begin() + k + d); k += d;
    c.best_derived.assign(all.begin() + k, all.begin() + k + nd); k += nd;
    c.rows.resize(nr);
    for (size_t r = 0; r < nr; r++, k += w) c.rows[r].assign(all.begin() + k, all.begin() + k + w);
    c.rows_lp.assign(all.begin() + k, all.begin() + k + nr); k += nr;
  }
  chains = std::move(out);
}

double Sampler::acceptance() const {
  long a = 0, n = 0;
  for (const auto& c : chains) { a += c.accepted_sampling; n += c.steps_sampling; }
  return n > 0 ? double(a) / n : 0.0;
}

long Sampler::n_samples() const {
  long n = 0;
  for (const auto& c : chains) n += long(c.rows.size());
  return n;
}

std::vector<double> Sampler::mean() const {
  const size_t np = post.names().size();
  std::vector<double> m(np, 0.0);
  const double N = double(n_samples());
  for (const auto& c : chains)
    for (const auto& r : c.rows)
      for (size_t i = 0; i < np; i++) m[i] += r[i] / N;
  return m;
}

std::vector<double> Sampler::stddev() const {
  const auto m = mean();
  const size_t np = post.names().size();
  std::vector<double> v(np, 0.0);
  const double N = double(n_samples());
  for (const auto& c : chains)
    for (const auto& r : c.rows)
      for (size_t i = 0; i < np; i++) v[i] += (r[i] - m[i]) * (r[i] - m[i]) / (N - 1.0);
  for (auto& x : v) x = std::sqrt(x);
  return v;
}

std::vector<std::vector<double>> Sampler::covariance() const {
  std::vector<std::vector<double>> all;
  for (const auto& c : chains) all.insert(all.end(), c.rows.begin(), c.rows.end());
  return sample_cov(all, 0, par.size());
}

// GetDist chain: "weight -ln(posterior) params derived", repeated rows merged into weights
void Sampler::write_chain_file(const Chain& c, const std::string& file) const {
  const std::string tmp = file + ".tmp";
  {
    std::ofstream f(tmp);
    if (!f) throw std::runtime_error("Sampler: cannot write " + tmp);
    f << std::setprecision(12);
    const auto& rows = c.rows;
    for (size_t k = 0; k < rows.size();) {
      size_t w = 1;
      while (k + w < rows.size() && rows[k + w] == rows[k]) w++;
      f << w << " " << -c.rows_lp[k];
      for (double v : rows[k]) f << " " << v;
      f << "\n";
      k += w;
    }
  }
  std::rename(tmp.c_str(), file.c_str());
}

// Checkpoint: the state of one chain (text, full precision; the samples are in <root>_<i>.txt)
void Sampler::write_checkpoint(const Chain& c) const {
  const std::string base = opt.output_root + "_" + std::to_string(c.index + 1);
  write_chain_file(c, base + ".txt");
  const std::string tmp = base + ".checkpoint.tmp";
  {
    std::ofstream f(tmp);
    if (!f) throw std::runtime_error("Sampler: cannot write " + tmp);
    const size_t d = par.size();
    f << std::setprecision(17);
    f << "mcmc_checkpoint 1\n" << c.index << " " << d << " " << post.derived_names().size() << "\n";
    f << c.steps << " " << c.accepted << " " << c.steps_sampling << " " << c.accepted_sampling << " "
      << c.accepted_window << " " << c.steps_fast << " " << c.accepted_fast << " " << c.accepted_window_fast << " "
      << (c.frozen ? 1 : 0) << "\n";
    f << c.scale << " " << c.scale_fast << " " << c.lp << " " << c.ll << " " << c.best_lp << " " << c.best_ll << "\n";
    auto vec = [&](const std::vector<double>& v) {
      for (double x : v) f << x << " ";
      f << "\n";
    };
    vec(c.x);
    vec(c.derived);
    vec(c.best_x);
    vec(c.best_derived);
    for (const auto& r : c.L) vec(r);
    f << c.history.size() << "\n";
    for (const auto& r : c.history) vec(r);
    f << c.rows.size() << "\n";   // for a consistency check of <root>_<i>.txt
    f << c.rng << "\n";
  }
  std::rename(tmp.c_str(), (base + ".checkpoint").c_str());
}

bool Sampler::load_checkpoint(Chain& c, int index) const {
  const std::string base = opt.output_root + "_" + std::to_string(index + 1);
  std::ifstream f(base + ".checkpoint");
  if (!f) return false;
  const size_t d = par.size(), nd = post.derived_names().size();
  std::string tag;
  int version = 0, idx = -1;
  size_t dd = 0, ndd = 0;
  f >> tag >> version >> idx >> dd >> ndd;
  if (tag != "mcmc_checkpoint" || version != 1 || idx != index || dd != d || ndd != nd)
    throw std::runtime_error("Sampler: " + base + ".checkpoint does not match this run (parameters or chain)");
  int frozen = 0;
  f >> c.steps >> c.accepted >> c.steps_sampling >> c.accepted_sampling >> c.accepted_window >> c.steps_fast
    >> c.accepted_fast >> c.accepted_window_fast >> frozen;
  c.frozen = frozen != 0;
  f >> c.scale >> c.scale_fast >> c.lp >> c.ll >> c.best_lp >> c.best_ll;
  auto vec = [&](std::vector<double>& v, size_t n) {
    v.assign(n, 0.0);
    for (auto& x : v) f >> x;
  };
  vec(c.x, d);
  vec(c.derived, nd);
  vec(c.best_x, d);
  vec(c.best_derived, nd);
  c.L.assign(d, {});
  for (auto& r : c.L) vec(r, d);
  size_t nh = 0;
  f >> nh;
  c.history.assign(nh, {});
  for (auto& r : c.history) vec(r, d);
  size_t nrows = 0;
  f >> nrows >> c.rng;
  if (!f) throw std::runtime_error("Sampler: cannot read " + base + ".checkpoint");
  c.index = index;
  // samples from the chain file (weights expanded)
  c.rows.clear();
  c.rows_lp.clear();
  std::ifstream ch(base + ".txt");
  std::string line;
  while (std::getline(ch, line)) {
    std::istringstream is(line);
    double w, mlp;
    if (!(is >> w >> mlp)) continue;
    std::vector<double> r(d + nd);
    for (auto& v : r) is >> v;
    for (int k = 0; k < int(std::lround(w)); k++) { c.rows.push_back(r); c.rows_lp.push_back(-mlp); }
  }
  if (c.rows.size() != nrows)
    throw std::runtime_error("Sampler: " + base + ".txt has " + std::to_string(c.rows.size()) + " samples, the checkpoint " +
                             std::to_string(nrows));
  return true;
}

void Sampler::write_output() const {
  const std::string& root = opt.output_root;
  for (const auto& c : chains) write_chain_file(c, root + "_" + std::to_string(c.index + 1) + ".txt");
  post.write_paramnames(root);
  write_covmat(root + ".covmat", par, covariance());
  {
    std::ofstream f(root + ".summary.txt");
    const auto m = mean(), s = stddev();
    f << "# " << chains.size() << " chains, " << n_samples() << " samples after burn-in, acceptance "
      << acceptance() << ", R-1 = " << R1 << "\n";
    f << "# best fit: -2 ln L = " << -2.0 * best_ll << ", -2 ln posterior = " << -2.0 * best_lp << "\n";
    f << "# name  mean  std  lower68  upper68  best_fit\n";
    for (size_t i = 0; i < post.names().size(); i++) {
      std::vector<double> v;
      for (const auto& c : chains)
        for (const auto& r : c.rows) v.push_back(r[i]);
      std::sort(v.begin(), v.end());
      auto q = [&](double p) { return v.empty() ? 0.0 : v[std::min(v.size() - 1, size_t(p * v.size()))]; };
      f << std::setprecision(8) << post.names()[i] << " " << m[i] << " " << s[i] << " " << q(0.15865) << " "
        << q(0.84135) << " " << (best.empty() ? 0.0 : best[i]) << "\n";
    }
  }
}

} // namespace FML::MCMC
