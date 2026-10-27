// =============================================================================
// Tests of the model framework (Model.h, Species.h, Gravity.h): the registry, model parameters in the
// parameter file, and a model assembled from custom parts. Run with `make test`.
//
// The custom model splits the CDM into the standard ColdDarkMatter and a second pressureless species
// ("dust") with its own SpeciesPerturbations written here. This must reproduce LCDM exactly (up to the
// ODE tolerance), which checks that an extra species enters the background, the initial conditions, the
// state vector, the Einstein equations (through its Moments) and the stored outputs correctly.
// =============================================================================

#include <FML/Cosmology/Background/BackgroundCosmology.h>
#include <FML/Cosmology/Background/Model.h>
#include <FML/Cosmology/EinsteinBoltzmannSolver/ParameterFile.h>
#include <FML/Cosmology/Perturbations/Perturbations.h>
#include <FML/Cosmology/Recombination/RecombinationHistory.h>
#include "TestHelpers.h"
#include <FML/Cosmology/Common.h>
#include <FML/Cosmology/Models/GeneralRelativity.h>
#include <FML/Cosmology/Models/StandardSpecies.h>

#include <cmath>
#include <functional>

using namespace FML::COSMOLOGY;

using namespace Constants;
using Test::check;
using Test::check_close;
using Test::check_rel;

namespace {
  bool throws(const std::function<void()>& f) {
    try { f(); } catch (const std::exception&) { return true; }
    return false;
  }

  // Pressureless matter, written independently of the CDM in StandardSpecies.cpp
  class DustPerturbations : public SpeciesPerturbations {
    public:
      explicit DustPerturbations(const Species& s) : species(s) {}
      std::string name() const override { return "dust"; }
      int n_vars(int) const override { return 2; }
      int cache_size() const override { return 1; }
      void update_cache(const ModeAtX& b, double* cache) const override { cache[0] = b.H0a2 * species.rho(b.x); }
      void initial_conditions(const ModeAtX& b, const InitialData& ic, double* y) const override {
        y[0] = -1.5 * ic.Psi;
        y[1] = -0.5 * b.ckH * ic.Psi;
      }
      Moments moments(const ModeAtX&, const double* cache, int, const double* y) const override {
        Moments m;
        m.drho = cache[0] * y[0];
        m.flux = -cache[0] * y[1];
        return m;
      }
      void rhs(const ModeAtX& b, const double*, int, const MetricAtX& m, const double* y, double* dydx) const override {
        dydx[0] = b.ckH * y[1] - 3.0 * m.dPhi;
        dydx[1] = -y[1] - b.ckH * m.Psi;
      }
      std::vector<std::string> output_names() const override { return {"delta_dust", "v_dust"}; }
      void outputs(const ModeAtX&, const double*, int, const double* y, double* out) const override {
        out[0] = y[0];
        out[1] = y[1];
      }

    private:
      const Species& species;
  };

  class Dust : public PowerLawSpecies {
    public:
      explicit Dust(double Omega0) : PowerLawSpecies("dust", Role::Other, Omega0, 3.0) {}
      std::unique_ptr<SpeciesPerturbations> perturbations(const Perturbations&) const override {
        return std::make_unique<DustPerturbations>(*this);
      }
  };

  // A second CDM species (duplicate output names)
  class SecondCDM : public ColdDarkMatter {
    public:
      using ColdDarkMatter::ColdDarkMatter;
      std::string name() const override { return "cdm2"; }
  };

  struct Run {
    BackgroundCosmology cosmo;
    RecombinationHistory rec;
    Perturbations pert;
    Run(const CosmologyParameters& p, const Model& m, const PerturbationOptions& o)
      : cosmo(p, m), rec((cosmo.solve(), cosmo)), pert(cosmo, rec, o) {
      rec.solve();
      pert.solve();
    }
  };
}

int main() {
  std::cout << "Registry\n";
  {
    check("lcdm is registered", !throws([] { Models::find("lcdm"); }));
    check("unknown model is an error", throws([] { Models::find("no_such_model"); }));
    CosmologyParameters p;
    check("default model is lcdm", p.model == "lcdm");
    const Model m = Models::build(p);
    check("lcdm: 5 species (no massive neutrinos), GR",
          m.species.size() == 5 && m.gravity->name() == "GR" && m.species.back()->role() == Species::Role::DarkEnergy);
    p.m_nu = 0.06;
    check("lcdm with m_nu > 0: 6 species", Models::build(p).species.size() == 6);
    CosmologyParameters q;
    q.model_params["mu0"] = 0.1;
    check("parameter of no model is an error (validate)", throws([&] { q.validate(); }));
    q.model_params.clear();
    q.model = "no_such_model";
    check("unknown model is an error (validate)", throws([&] { q.validate(); }));
  }

  std::cout << "Parameter file\n";
  {
    check("model = lcdm", ParameterFile::read("", {"model=lcdm"}).cosmo.model == "lcdm");
    check("unknown model is an error", throws([] { ParameterFile::read("", {"model=foo"}); }));
    check("model parameter of no model is an error", throws([] { ParameterFile::read("", {"mu0=0.1"}); }));
    check("lcdm has no model parameters", ParameterFile::model_parameter_names("lcdm").empty());
  }

  std::cout << "Background of a custom model (CDM split into cdm + dust)\n";
  CosmologyParameters p;
  Model split = Models::lcdm(p);
  split.name = "split";
  split.species[1] = std::make_shared<ColdDarkMatter>(0.5 * p.OmegaCDM);
  split.species.insert(split.species.begin() + 2, std::make_shared<Dust>(0.5 * p.OmegaCDM));
  {
    BackgroundCosmology lcdm(p), sp(p, split);
    double err = 0.0;
    for (double x = -15.0; x <= 2.0; x += 0.5) err = std::max(err, std::fabs(sp.Hp_of_x(x) / lcdm.Hp_of_x(x) - 1.0));
    check_close("Hp(x) = LCDM", err, 0.0, 1e-14);
    check_rel("rho_of(Other) = dust", sp.rho_of(Species::Role::Other, -1.0), 0.5 * lcdm.rhoCDM(-1.0), 1e-14);
    check_rel("rhoCDM = half of LCDM", sp.rhoCDM(-1.0), 0.5 * lcdm.rhoCDM(-1.0), 1e-14);
  }

  std::cout << "Perturbations of the custom model\n";
  {
    PerturbationOptions o;
    o.n_k = 6;
    o.k_min = 0.001 / Mpc;
    o.k_max = 0.2 / Mpc;
    Run l(p, Models::lcdm(p), o), s(p, split, o);
    check("species outputs: delta_cdm, delta_dust, Nu_0",
          s.pert.has("delta_cdm") && s.pert.has("delta_dust") && s.pert.has("Nu_0") && !l.pert.has("delta_dust"));
    check("unknown output is an error", throws([&] { s.pert.get("delta_foo", 0.0, 0.01 / Mpc); }));
    double ePhi = 0.0, eT = 0.0, eD = 0.0;
    for (double k : s.pert.get_k_array())
      for (double x : {-10.0, -7.0, -3.0, 0.0}) {
        ePhi = std::max(ePhi, std::fabs(s.pert.get_Phi(x, k) / l.pert.get_Phi(x, k) - 1.0));
        eT = std::max(eT, std::fabs(s.pert.get_Source_T(x, k) - l.pert.get_Source_T(x, k)) /
                              std::fabs(l.pert.get_Theta(0, x, k) + l.pert.get_Psi(x, k)));
        eD = std::max(eD, std::fabs(s.pert.get("delta_dust", x, k) / l.pert.get_delta_cdm(x, k) - 1.0));
        eD = std::max(eD, std::fabs(s.pert.get_delta_cdm(x, k) / l.pert.get_delta_cdm(x, k) - 1.0));
      }
    check_close("Phi = LCDM", ePhi, 0.0, 1e-6);
    check_close("S_T = LCDM (relative to Theta_0 + Psi)", eT, 0.0, 3e-5);
    check_close("delta_dust = delta_cdm = LCDM delta_cdm", eD, 0.0, 1e-6);

    Model dup = Models::lcdm(p);
    dup.species.push_back(std::make_shared<SecondCDM>(0.0));
    BackgroundCosmology cd(p, dup);
    cd.solve();
    RecombinationHistory rd(cd);
    check("two species with the same output names is an error", throws([&] { Perturbations pd(cd, rd, o); }));
  }

  return Test::summary();
}
