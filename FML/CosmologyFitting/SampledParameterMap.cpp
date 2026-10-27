#include <FML/CosmologyFitting/SampledParameterMap.h>
#include <FML/Cosmology/Recombination/RecombinationHistory.h>
#include <FML/Cosmology/Common.h>
#include <FML/CosmologyFitting/Theory.h>

#include <cmath>
#include <iomanip>
#include <map>
#include <sstream>
#include <stdexcept>

namespace FML::COSMOLOGY::FITTING {

const std::set<std::string>& SampledParameterMap::aliases() {
  static const std::set<std::string> a = {"H0", "omega_b", "omega_cdm", "Omega_m", "logA", "tau_reio", "theta_MC_100"};
  return a;
}

SampledParameterMap::SampledParameterMap(std::vector<std::string> fixed_in, std::vector<std::string> sampled,
                           std::set<std::string> nuisance)
  : fixed(std::move(fixed_in)), names(std::move(sampled)), nuis(std::move(nuisance)) {
  std::set<std::string> s(names.begin(), names.end());
  if (s.size() != names.size()) throw std::invalid_argument("a parameter is sampled twice");
  theta = s.count("theta_MC_100") > 0;
  for (const char* n : {"H0", "h", "Omega_m", "OmegaB", "OmegaCDM"})
    if (theta && s.count(n))
      throw std::invalid_argument(std::string("theta_MC_100 sets h at fixed omega_b, omega_cdm: do not also sample ") + n);
  if (s.count("H0") && s.count("h")) throw std::invalid_argument("sample H0 or h, not both");
  if (s.count("logA") && s.count("A_s")) throw std::invalid_argument("sample logA or A_s, not both");
  if (s.count("tau_reio") && s.count("z_reion")) throw std::invalid_argument("sample tau_reio or z_reion, not both");
  if (s.count("Omega_m") && (s.count("omega_cdm") || s.count("OmegaCDM")))
    throw std::invalid_argument("Omega_m sets OmegaCDM: do not also sample omega_cdm or OmegaCDM");
  ParameterFile::read("", fixed);   // checks the fixed parameters
}

RunSettings SampledParameterMap::settings(const std::vector<double>& x) const {
  if (x.size() != names.size()) throw std::invalid_argument("SampledParameterMap: wrong number of values");
  std::vector<std::string> ov = fixed;
  std::map<std::string, double> v;
  for (size_t i = 0; i < names.size(); i++) v[names[i]] = x[i];
  auto fmt = [](double d) { std::ostringstream o; o << std::setprecision(17) << d; return o.str(); };
  if (v.count("H0")) ov.push_back("h=" + fmt(v["H0"] / 100.0));
  if (v.count("logA")) ov.push_back("A_s=" + fmt(1e-10 * std::exp(v["logA"])));
  for (const auto& [n, val] : v)
    if (!aliases().count(n) && !nuis.count(n)) ov.push_back(n + "=" + fmt(val));
  RunSettings s = ParameterFile::read("", ov);
  // physical densities: from the aliases, otherwise those of the fixed parameters at the fixed h
  const double h2_0 = s.cosmo.h * s.cosmo.h;
  const double wb = v.count("omega_b") ? v["omega_b"] : s.cosmo.OmegaB * h2_0;
  const double wc = v.count("omega_cdm") ? v["omega_cdm"] : s.cosmo.OmegaCDM * h2_0;
  auto set_h = [&](double h) {
    s.cosmo.h = h;
    if (v.count("omega_b") || theta) s.cosmo.OmegaB = wb / (h * h);
    if (v.count("omega_cdm") || theta) s.cosmo.OmegaCDM = wc / (h * h);
  };
  set_h(s.cosmo.h);
  if (v.count("Omega_m")) s.cosmo.OmegaCDM = v["Omega_m"] - s.cosmo.OmegaB - s.cosmo.OmegaNcdm();
  if (theta) {
    TheoryOptions fit;
    fit.sound_horizon = TheoryOptions::SoundHorizon::Fit;
    auto f = [&](double h) {
      set_h(h);
      s.cosmo.validate();
      return Theory(s, fit).theta_MC() - v["theta_MC_100"];
    };
    set_h(Utils::find_root(f, 0.4, 1.0, 1e-9, 1e-9));
  }
  s.cosmo.validate();
  if (v.count("tau_reio")) {
    BackgroundCosmology bg(s.cosmo);
    bg.solve(s.bg_x_start, s.bg_x_end, s.bg_n_points);
    const RecombinationHistory r(bg, s.rec);
    const double tau = v["tau_reio"];
    auto g = [&](double z) { return r.tau_reionization_model(z) - tau; };
    if (g(1.0) > 0.0 || g(40.0) < 0.0) throw std::runtime_error("tau_reio outside the range of z_reion in [1, 40]");
    s.cosmo.z_reion = Utils::find_root(g, 1.0, 40.0, 1e-8, 1e-8);
  }
  return s;
}

} // namespace FML::COSMOLOGY::FITTING
