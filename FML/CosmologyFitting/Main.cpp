// =============================================================================
// ./mcmc: Markov chain Monte Carlo fits, best fits and profile likelihoods of background data (supernovae, BAO,
// Planck distance priors, BBN) and CMB spectra (Planck 2018 plik_lite, mock data).
//
//   make && ./mcmc params/mcmc_lcdm.ini [name=value ...]          (from the CosmologyFitting folder)
//   make clean && make USE_MPI=true && mpirun -np 4 ./mcmc params/mcmc_planck_lite_lcdm.ini   (chains over MPI)
//
// The parameter file has the format of ./cmb (name = value) plus:
//   likelihoods      = pantheonplus, desy5, union3, desi_dr2, planck_dp, bbn, planck_lite_TTTEEE, cmb_mock, ...
//                      (Likelihoods.h)
//   sample_<name>    = start, min, max, proposal [, latex label]  a sampled parameter (flat prior on [min, max])
//   gauss_<name>     = mean, sigma                                an additional Gaussian prior on it
//   mcmc_chains, mcmc_threads, mcmc_burn_in, mcmc_adapt_every, mcmc_min_samples, mcmc_max_steps, mcmc_check_every, mcmc_Rminus1,
//   mcmc_seed, mcmc_output (output root, e.g. chains/lcdm), mcmc_sound_horizon (recombination | fit)
//   mcmc_covmat      = file    initial proposal covariance (GetDist .covmat: "# name1 name2 ..." + matrix);
//                              parameters not in the file get their diagonal proposal^2
//   mcmc_resume      = true    continue from <mcmc_output>_<i>.checkpoint (written after every round unless
//                              mcmc_checkpoint = false) if present for all chains
//   mcmc_oversample_fast = n   fast-only updates per step for the nuisance parameters (default 4; the theory is
//                              cached, so these cost only the likelihood)
//   sampler          = mcmc (default) | minimize | hessian
//   mcmc_hessian     = file    (sampler = hessian, implied by this key) instead of sampling: covariance from the
//                              numerical Hessian of -ln posterior at the start point (2d^2 + 1 evaluations, spread
//                              over MPI processes), written to file as a .covmat, then exit. The steps start at the
//                              proposal widths and are adapted to 1.5 conditional sigma (hessian_adapt_steps = true,
//                              2d evaluations per pass, <= 4 passes; false: the proposal widths)
//   sampler = minimize: maximum of the posterior (MCMC/Minimizer.h) written to <mcmc_output>.minimum.txt, with
//   minimize_starts (4), minimize_max_evals (3000 per start), minimize_tol (0.01 in -2 ln P),
//   minimize_scatter (1: the starts after the first are drawn from start + scatter * N(0, covmat)),
//   minimize_hessian = true    then the covariance from the Hessian at the best fit -> <mcmc_output>.covmat,
//   minimize_profile = name, min, max, n   then the profile -2 ln P(name) at n values -> <mcmc_output>.profile_<name>.txt
//   The starts and the profile points are spread over MPI processes (and OpenMP threads, as the chains).
//   cmb_mock_file, cmb_mock_lmax, cmb_mock_fsky, cmb_mock_noise, cmb_mock_beam   options of cmb_mock
// Entries on the command line replace those of the file (also sample_* and gauss_*).
// Every other name is a parameter of ./cmb (cosmology, model and its parameters, precision) and fixes it, or a
// nuisance parameter of a likelihood (A_planck) fixed to that value.
//
// Sampled names are those of ./cmb (h, OmegaB, OmegaCDM, OmegaK, m_nu, Neff, n_s, A_s, z_reion, model parameters
// such as w0, wa), nuisance parameters (A_planck), or the aliases H0, omega_b, omega_cdm, Omega_m, logA,
// tau_reio and theta_MC_100 (SampledParameterMap.h).
// For CMB likelihoods the chains of one process are advanced one after the other (mcmc_threads = 1, each
// evaluation uses all OpenMP threads) unless mcmc_threads is given.
// Output: GetDist chains <root>_<i>.txt, <root>.paramnames, .ranges, .covmat and <root>.summary.txt. With several
// likelihoods the chi^2 = -2 ln L of each is a derived parameter chi2_<likelihood>.
// =============================================================================

#include <FML/Cosmology/EinsteinBoltzmannSolver/ParameterFile.h>
#include <FML/Cosmology/Common.h>
#include <FML/CosmologyFitting/Likelihoods.h>
#include <FML/MCMC/Sampler.h>
#include <FML/MCMC/Minimizer.h>
#include <FML/MCMC/Parallel.h>
#include <FML/CosmologyFitting/SampledParameterMap.h>
#include <FML/CosmologyFitting/Theory.h>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <set>
#include <sstream>

using namespace FML::COSMOLOGY;
using namespace FML::MCMC;
using namespace FML::COSMOLOGY::FITTING;

using namespace Constants;

namespace {
  std::string trim(const std::string& s) {
    const auto a = s.find_first_not_of(" \t\r\n"), b = s.find_last_not_of(" \t\r\n");
    return a == std::string::npos ? "" : s.substr(a, b - a + 1);
  }
  std::vector<std::string> split(const std::string& s) {
    std::vector<std::string> out;
    std::stringstream ss(s);
    std::string t;
    while (std::getline(ss, t, ',')) if (!trim(t).empty()) out.push_back(trim(t));
    return out;
  }

  std::map<std::string, std::string> default_labels() {
    return {{"h", "h"}, {"H0", "H_0"}, {"omega_b", "\\omega_b"}, {"omega_cdm", "\\omega_c"}, {"Omega_m", "\\Omega_m"},
            {"OmegaB", "\\Omega_b"}, {"OmegaCDM", "\\Omega_c"}, {"OmegaK", "\\Omega_K"}, {"m_nu", "m_\\nu"},
            {"Neff", "N_{\\rm eff}"}, {"w0", "w_0"}, {"wa", "w_a"}, {"n_s", "n_s"}, {"logA", "\\ln(10^{10}A_s)"},
            {"A_s", "A_s"}, {"tau_reio", "\\tau"}, {"z_reion", "z_{\\rm re}"}, {"theta_MC_100", "100\\theta_{\\rm MC}"},
            {"A_planck", "A_{\\rm Planck}"}};
  }
}

int main(int argc, char** argv) {
  const bool root = Parallel::rank() == 0;
  auto fail = [&](const std::string& msg) {
    std::cerr << "[process " << Parallel::rank() << "] " << msg << "\n";
    Parallel::abort(1);
  };
  if (argc < 2 || std::string(argv[1]) == "--help") {
    if (root) std::cout << "Usage: ./mcmc <parameter file> [name=value ...]   (see CosmologyFitting/Main.cpp)\n";
    return argc < 2;
  }
  // --- read the file and the command line ------------------------------------------------------------------
  std::vector<std::pair<std::string, std::string>> kv;
  {
    std::ifstream in(argv[1]);
    if (!in) fail(std::string("Cannot open ") + argv[1]);
    std::string line;
    while (std::getline(in, line)) {
      const auto p = line.find('#');
      if (p != std::string::npos) line = line.substr(0, p);
      const auto e = line.find('=');
      if (e == std::string::npos) { if (!trim(line).empty()) fail("Bad line: " + line); continue; }
      kv.push_back({trim(line.substr(0, e)), trim(line.substr(e + 1))});
    }
    for (int i = 2; i < argc; i++) {
      const std::string a = argv[i];
      const auto e = a.find('=');
      if (e == std::string::npos) fail("Unexpected argument " + a);
      const std::string key = trim(a.substr(0, e));   // replaces the entry of the file
      kv.erase(std::remove_if(kv.begin(), kv.end(), [&](const auto& p) { return p.first == key; }), kv.end());
      kv.push_back({key, trim(a.substr(e + 1))});
    }
  }

  SamplerOptions opt;
  opt.output_root = "chains/run";
  MinimizerOptions mopt;
  std::string sampler = "mcmc";
  std::string profile_spec;
  bool min_hessian = false, hessian_adapt = true;
  TheoryOptions topt;
  std::vector<std::string> like_names;
  std::vector<std::pair<std::string, std::string>> base_kv;   // name, value for ParameterFile (or fixed nuisance)
  std::map<std::string, std::string> like_opts;
  std::vector<SampledParameter> params;
  std::map<std::string, std::pair<double, double>> gauss;
  std::string covmat_file, hessian_file;
  bool threads_set = false;
  const auto labels = default_labels();
  try {
    std::set<std::string> lopt;
    for (const auto& n : Likelihoods::option_names()) lopt.insert(n);
    for (const auto& [k, v] : kv) {
      if (k == "likelihoods") like_names = split(v);
      else if (k == "mcmc_chains") opt.chains = std::stoi(v);
      else if (k == "mcmc_threads") { opt.threads = std::stoi(v); threads_set = true; }
      else if (k == "mcmc_burn_in") opt.burn_in = std::stoi(v);
      else if (k == "mcmc_adapt_every") opt.adapt_every = std::stoi(v);
      else if (k == "mcmc_min_samples") opt.min_samples = std::stoi(v);
      else if (k == "mcmc_max_steps") opt.max_steps = std::stoi(v);
      else if (k == "mcmc_check_every") opt.check_every = std::stoi(v);
      else if (k == "mcmc_Rminus1") opt.Rminus1_stop = std::stod(v);
      else if (k == "mcmc_seed") opt.seed = std::stoull(v);
      else if (k == "mcmc_resume") opt.resume = (v == "true" || v == "1");
      else if (k == "mcmc_checkpoint") opt.checkpoint = (v == "true" || v == "1");
      else if (k == "mcmc_oversample_fast") opt.oversample_fast = std::stoi(v);
      else if (k == "mcmc_output") opt.output_root = v;
      else if (k == "mcmc_covmat") covmat_file = v;
      else if (k == "mcmc_hessian") hessian_file = v;
      else if (k == "sampler") sampler = v;
      else if (k == "minimize_starts") mopt.starts = std::stoi(v);
      else if (k == "minimize_max_evals") mopt.max_evals = std::stoi(v);
      else if (k == "minimize_tol") mopt.tol = std::stod(v);
      else if (k == "minimize_scatter") mopt.start_scatter = std::stod(v);
      else if (k == "minimize_profile") profile_spec = v;
      else if (k == "minimize_hessian") min_hessian = (v == "true" || v == "1");
      else if (k == "hessian_adapt_steps") hessian_adapt = (v == "true" || v == "1");
      else if (k == "mcmc_sound_horizon") {
        if (v == "fit") topt.sound_horizon = TheoryOptions::SoundHorizon::Fit;
        else if (v == "recombination") topt.sound_horizon = TheoryOptions::SoundHorizon::Recombination;
        else throw std::invalid_argument("mcmc_sound_horizon must be recombination or fit");
      } else if (lopt.count(k)) {
        like_opts[k] = v;
      } else if (k.rfind("sample_", 0) == 0) {
        const auto t = split(v);
        if (t.size() < 4) throw std::invalid_argument(k + ": expected start, min, max, proposal [, label]");
        SampledParameter p;
        p.name = k.substr(7);
        p.start = std::stod(t[0]); p.min = std::stod(t[1]); p.max = std::stod(t[2]); p.proposal = std::stod(t[3]);
        if (t.size() > 4) {   // the label may contain commas
          p.label = t[4];
          for (size_t i = 5; i < t.size(); i++) p.label += "," + t[i];
        } else {
          p.label = labels.count(p.name) ? labels.at(p.name) : p.name;
        }
        params.push_back(p);
      } else if (k.rfind("gauss_", 0) == 0) {
        const auto t = split(v);
        if (t.size() != 2) throw std::invalid_argument(k + ": expected mean, sigma");
        gauss[k.substr(6)] = {std::stod(t[0]), std::stod(t[1])};
      } else {
        base_kv.push_back({k, v});
      }
    }
    for (auto& p : params)
      if (gauss.count(p.name)) { p.gaussian_prior = true; p.prior_mean = gauss[p.name].first; p.prior_sigma = gauss[p.name].second; gauss.erase(p.name); }
    if (!gauss.empty()) throw std::invalid_argument("gauss_" + gauss.begin()->first + " for a parameter that is not sampled");
    if (like_names.empty()) throw std::invalid_argument("no likelihoods given");
    if (!hessian_file.empty()) {
      if (sampler != "mcmc" && sampler != "hessian") throw std::invalid_argument("mcmc_hessian needs sampler = hessian");
      sampler = "hessian";
    }
    if (sampler != "mcmc" && sampler != "minimize" && sampler != "hessian")
      throw std::invalid_argument("sampler must be mcmc, minimize or hessian");
    if (sampler == "hessian" && hessian_file.empty()) throw std::invalid_argument("sampler = hessian needs mcmc_hessian = file");
    if ((!profile_spec.empty() || min_hessian) && sampler != "minimize")
      throw std::invalid_argument("minimize_profile and minimize_hessian need sampler = minimize");
  } catch (const std::exception& e) {
    fail(std::string("Error in the MCMC settings: ") + e.what());
  }

  // --- likelihoods, nuisance parameters, fixed parameters -------------------------------------------------------
  std::vector<std::unique_ptr<Likelihood>> likes;
  bool need_rec = false, need_cmb = false;
  std::set<std::string> nuisance_names;
  std::map<std::string, double> fixed_nuisance;
  std::vector<std::string> base;   // "name=value" for ParameterFile
  try {
    for (const auto& n : like_names) {
      likes.push_back(Likelihoods::create(n, like_opts));
      need_rec = need_rec || likes.back()->needs_recombination();
      if (likes.back()->needs_cmb()) {
        topt.cmb_ell_max = need_cmb ? std::max(topt.cmb_ell_max, likes.back()->cmb_ell_max()) : likes.back()->cmb_ell_max();
        need_cmb = true;
      }
      for (const auto& nn : likes.back()->nuisance_parameters()) nuisance_names.insert(nn);
    }
    if (!like_opts.empty() && std::find(like_names.begin(), like_names.end(), "cmb_mock") == like_names.end())
      throw std::invalid_argument("cmb_mock_* options given without the cmb_mock likelihood");
    for (const auto& [k, v] : base_kv) {
      if (nuisance_names.count(k)) fixed_nuisance[k] = std::stod(v);
      else base.push_back(k + "=" + v);
    }
    ParameterFile::read("", base);   // checks the fixed parameters
  } catch (const std::exception& e) {
    fail(std::string("Error: ") + e.what());
  }
  // nuisance parameters are fast (the theory is cached) and go last
  for (auto& p : params) p.fast = nuisance_names.count(p.name) > 0;
  std::stable_partition(params.begin(), params.end(), [](const SampledParameter& p) { return !p.fast; });
  std::map<std::string, int> pidx;
  for (size_t i = 0; i < params.size(); i++) pidx[params[i].name] = int(i);
  size_t n_slow = 0;
  while (n_slow < params.size() && !params[n_slow].fast) n_slow++;
  if (need_cmb && !threads_set) opt.threads = 1;

  // --- the parameter point -> RunSettings and nuisance parameters ----------------------------------------------
  std::unique_ptr<SampledParameterMap> pmap;
  try {
    std::vector<std::string> pn;
    for (const auto& p : params) pn.push_back(p.name);
    pmap = std::make_unique<SampledParameterMap>(base, pn, nuisance_names);
  } catch (const std::exception& e) {
    fail(std::string("Error: ") + e.what());
  }

  // --- derived parameters -----------------------------------------------------------------------------------
  const bool yp_bbn = ParameterFile::read("", base).cosmo.Yp_bbn;   // Yp is derived only when it varies
  std::vector<std::string> dn = {"H0", "omega_b", "omega_cdm", "Omega_m", "Omega_DE", "age", "Yp"};
  std::vector<std::string> dl = {"H_0", "\\omega_b", "\\omega_c", "\\Omega_m", "\\Omega_{\\rm DE}", "t_0\\,[{\\rm Gyr}]", "Y_p"};
  const bool deriv_rec = need_rec || need_cmb;
  if (deriv_rec) {
    for (const auto& n : {"r_drag", "hr_drag", "z_star", "theta_star_100"}) dn.push_back(n);
    for (const auto& l : {"r_d\\,[{\\rm Mpc}]", "hr_d\\,[{\\rm Mpc}]", "z_*", "100\\theta_*"}) dl.push_back(l);
  }
  if (need_cmb) {
    for (const auto& n : {"theta_MC_100", "tau_reio", "z_reion", "logA", "As_exp_m2tau", "sigma8", "S8"}) dn.push_back(n);
    for (const auto& l : {"100\\theta_{\\rm MC}", "\\tau", "z_{\\rm re}", "\\ln(10^{10}A_s)", "10^9A_se^{-2\\tau}",
                          "\\sigma_8", "S_8"}) dl.push_back(l);
  }
  std::vector<int> keep;
  // chi^2 = -2 ln L of each likelihood (when there are several)
  const bool chi2_each = likes.size() > 1;
  {
    std::vector<std::string> n2, l2;
    for (size_t i = 0; i < dn.size(); i++)
      if (!pidx.count(dn[i]) && (dn[i] != "Yp" || yp_bbn)) { keep.push_back(int(i)); n2.push_back(dn[i]); l2.push_back(dl[i]); }
    dn = n2;
    dl = l2;
  }
  if (chi2_each)
    for (const auto& L : likes) {
      std::string tex;
      for (char ch : L->name()) tex += ch == '_' ? std::string("\\_") : std::string(1, ch);
      dn.push_back("chi2_" + L->name());
      dl.push_back("\\chi^2_{\\rm " + tex + "}");
    }

  // Theory and derived parameters for the slow parameters, cached (two entries per thread: the current point of
  // a chain and the last proposal), so that fast (nuisance) updates cost only the likelihoods
  struct Cached {
    std::vector<double> key;
    std::shared_ptr<Theory> theory;
    std::vector<double> derived;
  };
  auto evaluate = [&](const std::vector<double>& x) {
    const RunSettings s = pmap->settings(x);
    auto t = std::make_shared<Theory>(s, topt);
    const auto& c = s.cosmo;
    const double h2 = c.h * c.h;
    const auto& b = t->background();
    const double Om = c.OmegaM() + c.OmegaNcdm();
    std::vector<double> d = {100.0 * c.h, c.OmegaB * h2, c.OmegaCDM * h2, Om, b.OmegaDE_of_x(0.0), b.t0() / Gyr,
                             c.helium_fraction()};
    if (deriv_rec) {
      const double rd = t->r_drag();
      d.insert(d.end(), {rd, rd * c.h, t->z_star(), t->theta_star()});
    }
    if (need_cmb) {
      const double tau = t->tau_reio(), s8 = t->sigma8();
      d.insert(d.end(), {t->theta_MC(), tau, c.z_reion, std::log(1e10 * c.A_s), 1e9 * c.A_s * std::exp(-2.0 * tau), s8,
                         s8 * std::sqrt(Om / 0.3)});
    }
    return Cached{std::vector<double>(x.begin(), x.begin() + n_slow), t, d};
  };
  LikelihoodFunction like = [&](const std::vector<double>& x) {
    thread_local std::vector<Cached> cache;   // most recent first
    const std::vector<double> key(x.begin(), x.begin() + n_slow);
    size_t hit = cache.size();
    for (size_t i = 0; i < cache.size(); i++) if (cache[i].key == key) { hit = i; break; }
    if (hit == cache.size()) {
      cache.insert(cache.begin(), evaluate(x));
      if (cache.size() > 2) cache.pop_back();
      hit = 0;
    } else if (hit != 0) {
      std::swap(cache[0], cache[hit]);
    }
    Theory& t = *cache[0].theory;
    for (const auto& [n, val] : fixed_nuisance) t.set_nuisance(n, val);
    for (size_t i = n_slow; i < params.size(); i++) t.set_nuisance(params[i].name, x[i]);
    LikelihoodResult r;
    double ll = 0.0;
    std::vector<double> chi2;
    for (const auto& L : likes) {
      const double l = L->loglike(t);
      ll += l;
      chi2.push_back(-2.0 * l);
    }
    for (int i : keep) r.derived.push_back(cache[0].derived[i]);
    if (chi2_each) r.derived.insert(r.derived.end(), chi2.begin(), chi2.end());
    r.loglike = ll;
    return r;
  };

  Utils::verbose = false;
  if (root) {
    std::cout << "MCMC: " << params.size() << " parameters, likelihoods:";
    for (const auto& L : likes) std::cout << " " << L->name() << " (" << L->n_data() << ")";
    std::cout << ", " << opt.chains << " chains";
    if (Parallel::enabled()) std::cout << " on " << Parallel::size() << " MPI processes";
    std::cout << "\n";
    if (!nuisance_names.empty()) {
      std::cout << "  nuisance parameters:";
      for (const auto& n : nuisance_names)
        std::cout << " " << n << (pidx.count(n) ? " (sampled)" : fixed_nuisance.count(n) ? " (fixed)" : " (default)");
      std::cout << "\n";
    }
  }
  try {
    std::vector<double> x0;
    for (const auto& p : params) x0.push_back(p.start);
    if (!covmat_file.empty()) {
      std::vector<std::string> missing;
      opt.initial_covmat = FML::MCMC::read_covmat(covmat_file, params, missing);
      mopt.covmat = opt.initial_covmat;
      if (root) {
        std::cout << "  initial proposal from " << covmat_file;
        if (!missing.empty()) {
          std::cout << " (not in the file, diagonal:";
          for (const auto& m : missing) std::cout << " " << m;
          std::cout << ")";
        }
        std::cout << "\n";
      }
    }
    // timing of one evaluation
    Utils::Timer tm;
    const auto r0 = like(x0);
    if (root) std::cout << "  ln L at the start = " << std::setprecision(8) << r0.loglike << " (" << std::setprecision(3)
                        << tm.elapsed() << " s per evaluation)\n" << std::flush;

    const Posterior post(params, dn, dl, like);
    mopt.verbose = opt.verbose;
    mopt.seed = opt.seed;
    mopt.threads = opt.threads;
    if (need_cmb) mopt.progress_every = 10;
    if (root) {
      const auto dir = std::filesystem::path(opt.output_root).parent_path();
      if (!dir.empty()) std::filesystem::create_directories(dir);
    }
    Parallel::barrier();
    std::vector<double> steps;
    for (const auto& p : params) steps.push_back(p.proposal);

    if (sampler == "hessian") {   // covariance from the Hessian at the start point
      Minimizer mz(post, mopt);
      std::vector<std::vector<double>> C;
      const bool ok = Minimizer::covariance(mz.hessian(x0, steps, hessian_adapt), steps, C);
      if (root) {
        if (!ok) std::cerr << "Hessian is not positive definite (start point far from the peak, or steps too large); "
                              "writing the diagonal 1/H_ii instead\n";
        FML::MCMC::write_covmat(hessian_file, params, C);
        std::cout << "Covariance written to " << hessian_file << "; standard deviations:\n";
        for (size_t i = 0; i < params.size(); i++)
          std::cout << "  " << std::left << std::setw(14) << params[i].name << std::setprecision(4) << std::sqrt(C[i][i]) << "\n";
      }
      return 0;
    }

    if (sampler == "minimize") {
      Minimizer mz(post, mopt);
      const MinimizerResult best = mz.run();
      if (root) {
        mz.write(opt.output_root, best);
        std::cout << "\nBest fit: -2 ln L = " << std::setprecision(10) << best.chi2() << ", -2 ln posterior = "
                  << -2.0 * best.logpost << (best.converged ? "" : " (not converged)") << "\n";
        for (size_t i = 0; i < post.names().size(); i++)
          std::cout << "  " << std::left << std::setw(16) << post.names()[i] << std::setprecision(8) << " = "
                    << (i < params.size() ? best.x[i] : best.derived[i - params.size()]) << "\n";
        std::cout << "Written to " << opt.output_root << ".minimum.txt\n" << std::flush;
      }
      if (min_hessian) {   // covariance at the best fit, a proposal for the MCMC
        std::vector<std::vector<double>> C;
        const bool ok = Minimizer::covariance(mz.hessian(best.x, steps, hessian_adapt), steps, C);
        if (root) {
          FML::MCMC::write_covmat(opt.output_root + ".covmat", params, C);
          std::cout << "Covariance from the Hessian at the best fit written to " << opt.output_root << ".covmat"
                    << (ok ? "" : " (not positive definite: diagonal 1/H_ii)") << "\n";
        }
      }
      if (!profile_spec.empty()) {   // name, min, max, n
        const auto t = split(profile_spec);
        if (t.size() != 4) throw std::invalid_argument("minimize_profile: expected name, min, max, n");
        if (!pidx.count(t[0])) throw std::invalid_argument("minimize_profile: " + t[0] + " is not sampled");
        const size_t i = size_t(pidx.at(t[0]));
        const double lo = std::stod(t[1]), hi = std::stod(t[2]);
        const int n = std::stoi(t[3]);
        if (n < 1) throw std::invalid_argument("minimize_profile: n must be >= 1");
        std::vector<double> vals;
        for (int k = 0; k < n; k++) vals.push_back(n == 1 ? lo : lo + (hi - lo) * k / (n - 1.0));
        if (root) std::cout << "Profile in " << t[0] << " (" << n << " points)\n" << std::flush;
        const auto prof = mz.profile(i, vals, best.x);
        if (root) {
          const std::string file = opt.output_root + ".profile_" + t[0] + ".txt";
          mz.write_profile(file, i, prof);
          std::cout << "  " << t[0] << "  -2 ln P - min\n";
          double mn = -2.0 * best.logpost;
          for (const auto& r : prof) mn = std::min(mn, -2.0 * r.logpost);
          for (const auto& r : prof)
            std::cout << "  " << std::setw(12) << std::setprecision(6) << r.x[i] << "  " << std::setprecision(4)
                      << -2.0 * r.logpost - mn << "\n";
          std::cout << "Profile written to " << file << "\n";
        }
      }
      return 0;
    }

    Sampler mc(post, opt);
    mc.run();
    if (root) {
      const auto m = mc.mean(), sd = mc.stddev(), bf = mc.best_fit();
      std::cout << "\n" << mc.n_samples() << " samples, acceptance " << mc.acceptance() << ", R-1 = " << mc.Rminus1()
                << ", best fit -2 ln L = " << -2.0 * mc.best_loglike() << "\n";
      for (size_t i = 0; i < mc.names().size(); i++)
        std::cout << "  " << std::left << std::setw(14) << mc.names()[i] << std::setprecision(6) << " = " << std::setw(11) << m[i]
                  << " +- " << std::setw(11) << sd[i] << "  (best fit " << bf[i] << ")\n";
      std::cout << "Chains written to " << opt.output_root << "_*.txt (GetDist format)\n";
    }
  } catch (const std::exception& e) {
    fail(std::string("Error: ") + e.what());
  }
  return 0;
}
