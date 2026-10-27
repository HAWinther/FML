// =============================================================================
// Regression test: key outputs of several configurations and models at reduced resolution, compared with stored
// reference values (tests/reference/regression.txt). Guards refactorings that must not change
// the results.
//
//   ./test_regression              compare (relative tolerance 3e-7; 1e-4 when compiled with fused multiply-adds)
//   ./test_regression --update     rewrite the reference file (only after an intended change)
// The current values are also written to data/regression_last.txt.
//
// The tolerance is set by the numerical noise of the adaptive ODE solvers: changing the order of
// floating-point operations or of the state vector, or ode_relerr by one part in 1e9, moves the outputs
// by up to ~1e-7 (high-ell C_ell, lensed BB). Physics changes show up far above this. The reference values were made
// without fused multiply-adds (x86-64 without -march=native); with them (-march=native, FML's default, or ARM) the
// rounding differs and the outputs move by up to ~3e-5, so the tolerance is then 1e-4.
// =============================================================================

#include <FML/Cosmology/Background/BackgroundCosmology.h>
#include <FML/Cosmology/Background/CosmologyParameters.h>
#include <FML/Cosmology/Perturbations/Perturbations.h>
#include <FML/Cosmology/Spectra/PowerSpectrum.h>
#include <FML/Cosmology/Recombination/RecombinationHistory.h>
#include "TestHelpers.h"
#include <FML/Cosmology/Common.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <functional>
#include <iomanip>
#include <map>
#include <sstream>
#include <string>
#include <vector>

using namespace FML::COSMOLOGY;

using namespace Constants;

namespace {
  const char* REF = "tests/reference/regression.txt";
  std::vector<std::pair<std::string, double>> values;
  void put(const std::string& name, double v) { values.push_back({name, v}); }

  struct Run {
    std::string name;
    std::function<void(CosmologyParameters&, PerturbationOptions&, PowerSpectrumOptions&)> set;
  };

  void run(const Run& r) {
    CosmologyParameters p = CosmologyParameters::fiducial();
    PerturbationOptions po;
    po.n_k = 100;
    PowerSpectrumOptions so;
    so.ell_max = 800;
    r.set(p, po, so);
    BackgroundCosmology cosmo(p);
    cosmo.solve();
    RecombinationHistory rec(cosmo);
    rec.solve();
    Perturbations pert(cosmo, rec, po);
    pert.solve();
    PowerSpectrum ps(cosmo, rec, pert, so);
    ps.solve();
    const std::string n = r.name + ".";
    put(n + "eta0", cosmo.eta0() / Mpc);
    put(n + "H(z=1)", cosmo.H_of_x(-std::log(2.0)));
    put(n + "tau_reion", rec.tau_of_x(-7.0));
    for (double k : {0.001, 0.02, 0.15}) {
      const double kk = k / Mpc;
      std::ostringstream t;
      t << k;
      put(n + "Phi(x=-7,k=" + t.str() + ")", pert.get_Phi(-7.0, kk));
      put(n + "Phi(x=0,k=" + t.str() + ")", pert.get_Phi(0.0, kk));
      put(n + "Psi(x=-3,k=" + t.str() + ")", pert.get_Psi(-3.0, kk));
      put(n + "delta_cdm(x=0,k=" + t.str() + ")", pert.get_delta_cdm(0.0, kk));
      put(n + "P(k=" + t.str() + ")", ps.get_matter_power_spectrum(0.0, kk));
    }
    for (int l : {2, 10, 50, 220, 500, 800}) {
      put(n + "TT(" + std::to_string(l) + ")", ps.get_Dell_TT(l));
      if (po.polarization) {
        put(n + "EE(" + std::to_string(l) + ")", ps.get_Dell_EE(l));
        put(n + "TE(" + std::to_string(l) + ")", ps.get_Dell_TE(l));
      }
      if (so.lensing) {
        put(n + "TTlensed(" + std::to_string(l) + ")", ps.get_Dell_TT_lensed(l));
        put(n + "BBlensed(" + std::to_string(l) + ")", ps.get_Dell_BB_lensed(l));
        put(n + "Cpp(" + std::to_string(l) + ")", ps.get_Cell_phiphi(l));
      }
    }
    put(n + "sigma8", ps.get_sigma8());
  }
}

int main(int argc, char** argv) {
  const bool update = argc > 1 && std::strcmp(argv[1], "--update") == 0;
  const std::vector<Run> runs = {
    {"fiducial_lensed", [](auto&, auto&, PowerSpectrumOptions& s) { s.lensing = true; }},
    {"T_only_noneutrinos", [](CosmologyParameters& p, PerturbationOptions& o, auto&) {
       p.Neff = 0.0; o.neutrinos = false; o.polarization = false; }},
    {"massive_nu", [](CosmologyParameters& p, auto&, auto&) { p.m_nu = 0.1; }},
    {"closed", [](CosmologyParameters& p, auto&, auto&) { p.OmegaK = -0.05; }},
    {"open_massive_nu", [](CosmologyParameters& p, auto&, auto&) { p.OmegaK = 0.05; p.m_nu = 0.2; }},
    {"musigma", [](CosmologyParameters& p, auto&, auto&) {
       p.model = "musigma"; p.model_params = {{"mu0", 0.3}, {"Sigma0", 0.1}}; }},
    {"ndgp", [](CosmologyParameters& p, auto&, auto&) { p.model = "dgp"; p.model_params = {{"Omega_rc", 0.25}}; }},
    {"quintessence", [](CosmologyParameters& p, auto&, auto&) {
       p.model = "quintessence"; p.model_params = {{"quint_lambda", 1.0}}; }},
    {"jbd", [](CosmologyParameters& p, auto&, auto&) { p.model = "jbd"; p.model_params = {{"bd_omega", 100.0}}; }},
    {"fr", [](CosmologyParameters& p, auto&, auto&) { p.model = "fr"; p.model_params = {{"fR0", 1e-5}}; }},
    {"coupled_quintessence", [](CosmologyParameters& p, auto&, auto&) {
       p.model = "coupled_quintessence"; p.model_params = {{"cq_lambda", 0.5}, {"cq_beta", 0.05}}; }},
    {"dark_scattering", [](CosmologyParameters& p, auto&, auto&) {
       p.model = "dark_scattering"; p.model_params = {{"ds_w", -0.9}, {"ds_xi", 10.0}}; }},
    {"cubic_galileon", [](CosmologyParameters& p, auto&, auto&) { p.model = "cubic_galileon"; }},
    {"horndeski", [](CosmologyParameters& p, auto&, auto&) {
       p.model = "horndeski";
       p.model_params = {{"hd_cK", 1.0}, {"hd_cB", 0.6}, {"hd_cM", 0.3}, {"hd_w0", -0.95}, {"hd_wa", 0.1}}; }},
    {"ede", [](CosmologyParameters& p, auto&, auto&) {
       p.model = "ede"; p.model_params = {{"ede_fEDE", 0.1}, {"ede_log10zc", 3.5}}; }},
    {"ide", [](CosmologyParameters& p, auto&, auto&) {
       p.model = "ide"; p.model_params = {{"ide_xi", -0.1}, {"ide_w", -0.95}}; }},
  };
  for (const auto& r : runs) {
    std::cout << "Running " << r.name << "\n" << std::flush;
    run(r);
  }

  // The current values (also when comparing), for inspection
  {
    std::ofstream fp("data/regression_last.txt");
    fp << std::setprecision(17);
    for (const auto& [k, v] : values) fp << k << " " << v << "\n";
  }
  if (update) {
    std::ofstream fp(REF);
    if (!fp) { std::cerr << "cannot write " << REF << "\n"; return 1; }
    fp << "# Reference values for tests/test_regression.cpp (./test_regression --update)\n";
    fp << std::setprecision(17);
    for (const auto& [k, v] : values) fp << k << " " << v << "\n";
    std::cout << "Wrote " << values.size() << " values to " << REF << "\n";
    return 0;
  }

  std::ifstream fp(REF);
  if (!fp) { std::cerr << "missing " << REF << " (run from the project root)\n"; return 1; }
  std::map<std::string, double> ref;
  std::string line;
  while (std::getline(fp, line)) {
    if (line.empty() || line[0] == '#') continue;
    std::istringstream is(line);
    std::string k;
    double v;
    is >> k >> v;
    ref[k] = v;
  }
  std::cout << "Comparing " << values.size() << " values with " << REF << "\n";
  std::vector<std::pair<double, std::string>> changes;
  int missing = 0;
  for (const auto& [k, v] : values) {
    if (!ref.count(k)) { missing++; continue; }
    changes.push_back({std::fabs(v - ref[k]) / std::max(std::fabs(ref[k]), 1e-300), k});
  }
  std::sort(changes.rbegin(), changes.rend());
  Test::check("all values present in the reference", missing == 0 && ref.size() == values.size());
  std::cout << "  largest relative changes:\n";
  for (size_t i = 0; i < std::min<size_t>(5, changes.size()); i++)
    std::cout << "    " << changes[i].first << "  " << changes[i].second << "\n";
  const double worst = changes.empty() ? 0.0 : changes[0].first;
#if defined(__FMA__) || defined(__ARM_FEATURE_FMA) || defined(__aarch64__)
  Test::check_close("outputs unchanged (relative 1e-4 with FMA)", worst, 0.0, 1e-4);
#else
  Test::check_close("outputs unchanged (relative 3e-7)", worst, 0.0, 3e-7);
#endif
  return Test::summary();
}
