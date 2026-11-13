// =============================================================================
// Tests of the model framework: the model registry (Cosmology/Perturbations/Presets.h, built on the background registry
// of Cosmology/Background/Presets.h: names, parameters, descriptions), model parameters in the parameter file, and a
// model assembled from custom parts (a ComponentBackground and a PERTURBATIONS::PerturbationModel_Assembled). Run with
// `make test`.
//
// The custom model splits the CDM into the standard CDM and a second pressureless species ("dust") with its own
// SpeciesPerturbation written here. This must reproduce LCDM exactly (up to the ODE tolerance), which checks that an
// extra species enters the background, the initial conditions, the state vector, the Einstein equations (through its
// Moments) and the stored outputs correctly.
// =============================================================================

#include <FML/Cosmology/Background/BackgroundCosmology.h>
#include <FML/Cosmology/EinsteinBoltzmannSolver/ParameterFile.h>
#include <FML/Cosmology/Perturbations/Perturbations.h>
#include <FML/Cosmology/Recombination/RecombinationHistory.h>
#include <FML/tests/TestHelpers.h>
#include <FML/Cosmology/Common.h>
#include <FML/Cosmology/Background/Presets.h>
#include <FML/Cosmology/Perturbations/Presets.h>

#include <cmath>
#include <functional>
#include <set>
#include <string>
#include <vector>

using FML::TESTS::throws;

using namespace FML::COSMOLOGY;

using namespace Constants;
using FML::TESTS::check;
using FML::TESTS::check_close;
using FML::TESTS::check_rel;
namespace BG = FML::COSMOLOGY::BACKGROUND;
namespace PT = FML::COSMOLOGY::PERTURBATIONS;

namespace {

    // Pressureless matter, written independently of SpeciesPerturbation_CDM
    class DustPerturbations : public PT::SpeciesPerturbation {
      public:
        explicit DustPerturbations(const BG::Species & s) : species(s) {}
        std::string name() const override { return "dust"; }
        int n_vars(int) const override { return 2; }
        int cache_size() const override { return 1; }
        void update_cache(const ModeAtX & b, double * cache) const override { cache[0] = b.H0a2 * species.rho(b.x); }
        void initial_conditions(const ModeAtX & b, const InitialData & ic, double * y) const override {
            y[0] = -1.5 * ic.Psi;
            y[1] = -0.5 * b.ckH * ic.Psi;
        }
        Moments moments(const ModeAtX &, const double * cache, int, const double * y) const override {
            Moments m;
            m.drho = cache[0] * y[0];
            m.flux = -cache[0] * y[1];
            return m;
        }
        void rhs(const ModeAtX & b, const double *, int, const MetricAtX & m, const double * y, double * dydx)
            const override {
            dydx[0] = b.ckH * y[1] - 3.0 * m.dPhi;
            dydx[1] = -y[1] - b.ckH * m.Psi;
        }
        std::vector<std::string> output_names() const override { return {"delta_dust", "v_dust"}; }
        void outputs(const ModeAtX &, const double *, int, const double * y, double * out) const override {
            out[0] = y[0];
            out[1] = y[1];
        }

      private:
        const BG::Species & species;
    };

    // The parameters of the background of a run (as BackgroundCosmology)
    BG::BackgroundParameters background_parameters(const CosmologyParameters & p) {
        BG::BackgroundParameters b;
        b.h = p.h;
        b.Omega_b = p.OmegaB;
        b.Omega_cdm = p.OmegaCDM;
        b.Omega_k = p.OmegaK;
        b.T_CMB = p.TCMB;
        b.N_eff = p.Neff;
        if (p.has_massive_nu())
            b.m_nu_eV = {p.m_nu};
        b.T_ncdm_over_TCMB = p.T_ncdm_over_TCMB;
        return b;
    }

    // LCDM with the CDM split into cdm + dust (half each), on the range of BackgroundCosmology; with_dust_perturbations
    // = false: a second CDM perturbation instead of the dust's (duplicate output names)
    PT::PerturbationModelPtr split_model(const CosmologyParameters & p, bool with_dust_perturbations = true) {
        const auto bp = background_parameters(p);
        auto s = BG::standard_species(bp);
        s[1] = std::make_shared<BG::PowerLawSpecies>("cdm", BG::Component::CDM, 0.5 * p.OmegaCDM, 3.0);
        auto dust = std::make_shared<BG::PowerLawSpecies>("dust", BG::Component::Other, 0.5 * p.OmegaCDM, 3.0);
        s.insert(s.begin() + 2, dust);
        s.push_back(std::make_shared<BG::PowerLawSpecies>("lambda",
                                                          BG::Component::DarkEnergy,
                                                          1.0 - p.OmegaK - BG::density_today(s),
                                                          0.0));
        auto bg = std::make_shared<const BG::ComponentBackground>("split",
                                                                  bp,
                                                                  s,
                                                                  std::make_shared<BG::GeneralRelativity>(),
                                                                  BG::BackgroundRange{-25.0, 5.0, 200});
        const BG::Species * d = dust.get();
        auto standard = std::make_shared<PT::StandardSpeciesPerturbations>(*bg);
        return std::make_shared<const PT::PerturbationModel_Assembled>(
            "split",
            bg,
            [standard, d, with_dust_perturbations](const PT::PerturbationContext & ctx) {
                PT::SpeciesPerturbationList l;
                standard->add_to(ctx, l);
                if (with_dust_perturbations)
                    l.push_back(std::make_unique<DustPerturbations>(*d));
                else
                    l.push_back(std::make_unique<PT::SpeciesPerturbation_CDM>(*d, ctx));
                return l;
            },
            [](const PT::PerturbationContext & ctx) {
                return std::make_unique<PT::PerturbedFieldEquation_GeneralRelativity>(ctx);
            });
    }

    struct Run {
        BackgroundCosmology cosmo;
        RecombinationHistory rec;
        Perturbations pert;
        Run(const CosmologyParameters & p, const PerturbationOptions & o)
            : cosmo(p), rec((cosmo.solve(), cosmo)), pert(cosmo, rec, o) {
            finish();
        }
        Run(const CosmologyParameters & p, PT::PerturbationModelPtr m, const PerturbationOptions & o)
            : cosmo(p, std::move(m)), rec((cosmo.solve(), cosmo)), pert(cosmo, rec, o) {
            finish();
        }
        void finish() {
            rec.solve();
            pert.solve();
        }
    };
} // namespace

int main() {
    std::cout << "Registry\n";
    {
        check(!throws([] { PT::find_perturbation_model("lcdm"); }), "lcdm is registered");
        check(throws([] { PT::find_perturbation_model("no_such_model"); }), "unknown model is an error");
        // One registry entry per model in both registries, every parameter once with a description; the model's
        // parameters are those of its background followed by those of its perturbations
        {
            bool same = PT::perturbation_registry().size() == BG::registry().size();
            bool described = true;
            for (const auto & m : PT::perturbation_registry()) {
                same = same && !throws([&] { BG::find_model(m.name); }) && !m.description.empty();
                const auto all = PT::model_parameters(m.name);
                const auto & bp = BG::find_model(m.name).parameters;
                same = same && all.size() == bp.size() + m.parameters.size();
                std::set<std::string> names;
                for (size_t i = 0; i < all.size(); i++) {
                    described = described && !all[i].description.empty() && names.insert(all[i].name).second;
                    same = same && all[i].name == (i < bp.size() ? bp[i].name : m.parameters[i - bp.size()].name);
                }
            }
            check(same, "model registry: the background registry's models, parameters = background's + perturbations'");
            check(described, "model registry: every parameter once per model, with a description");
            const auto names = [](const std::string & model) { return ParameterFile::model_parameter_names(model); };
            check(names("jbd") ==
                          std::vector<std::string>{"bd_omega", "bd_GeffG_today", "bd_density_parameter_definition"} &&
                      names("dark_scattering") == std::vector<std::string>{"ds_w", "ds_xi", "ds_cs2"} &&
                      names("quintessence_pngb") ==
                          std::vector<std::string>{"quint_f", "quint_phi_ini", "quint_dphidlna_ini"} &&
                      names("ede_zc") == std::vector<std::string>{"ede_fEDE", "ede_zc", "ede_theta_i", "ede_n"},
                  "model parameters: jbd, dark_scattering, quintessence_pngb, ede_zc");
        }
        CosmologyParameters p;
        check(p.model == "lcdm", "default model is lcdm");
        {
            BackgroundCosmology b(p);
            check(b.background().species().size() == 5 && b.gravity_name() == "GR" &&
                      b.perturbation_model()->species_perturbations(b.perturbation_model()->context()).size() == 2,
                  "lcdm: 5 species (no massive neutrinos), GR, perturbations of cdm and massless neutrinos");
        }
        p.m_nu = 0.06;
        check(BackgroundCosmology(p).background().species().size() == 6, "lcdm with m_nu > 0: 6 species");
        // every model of the registry has a background and perturbations
        bool all = true;
        for (const auto & m : PT::perturbation_registry()) {
            CosmologyParameters q;
            q.model = m.name;
            if (m.name == "horndeski_table") { // a model given by a table: the cubic Galileon of SimulationCode
                q.model_file = "../../SimulationCode/input/horndeski/cubic_galileon_gravity.txt";
                q.h = 0.671;
                q.OmegaB = 0.049;
                q.OmegaCDM = 0.2685;
                q.Neff = 3.046;
                q.TCMB = 2.7255;
            }
            try {
                BackgroundCosmology b(q);
                all = all && b.perturbation_model() && b.perturbation_model()->background_ptr() == b.background_ptr();
            } catch (const std::exception & e) {
                std::cout << "  " << m.name << ": " << e.what() << "\n";
                all = false;
            }
        }
        check(all, "every model (default parameters) has its background and perturbation model");
        CosmologyParameters q;
        q.model_params["mu0"] = 0.1;
        check(throws([&] { q.validate(); }), "parameter of no model is an error (validate)");
        q.model_params.clear();
        q.model = "no_such_model";
        check(throws([&] { q.validate(); }), "unknown model is an error (validate)");
    }

    std::cout << "Parameter file\n";
    {
        check(ParameterFile::read("", {"model=lcdm"}).cosmo.model == "lcdm", "model = lcdm");
        check(throws([] { ParameterFile::read("", {"model=foo"}); }), "unknown model is an error");
        check(throws([] { ParameterFile::read("", {"mu0=0.1"}); }), "model parameter of no model is an error");
        check(ParameterFile::model_parameter_names("lcdm").empty(), "lcdm has no model parameters");
        const auto pngb = ParameterFile::read("", {"model=quintessence_pngb", "quint_f=0.8", "quint_phi_ini=0.5"});
        const auto jbd =
            ParameterFile::read("", {"model=jbd", "bd_GeffG_today=0.95", "bd_density_parameter_definition=1"});
        check(pngb.cosmo.model_param("quint_f") == 0.8 && pngb.cosmo.model_param("quint_phi_ini") == 0.5 &&
                  pngb.cosmo.model_param("quint_dphidlna_ini") == 0.0 &&
                  jbd.cosmo.model_param("bd_GeffG_today") == 0.95 &&
                  jbd.cosmo.model_param("bd_density_parameter_definition") == 1.0,
              "the models and parameters of the registry: quintessence_pngb (quint_f, quint_phi_ini), jbd "
              "(bd_GeffG_today, bd_density_parameter_definition)");
        check(throws([] { ParameterFile::read("", {"model=dark_scattering", "bd_omega=10"}); }) &&
                  throws([] { ParameterFile::read("", {"model=fr", "quint_phi_ini=0.1"}); }),
              "a parameter of another model is an error");
    }

    std::cout << "Background of a custom model (CDM split into cdm + dust)\n";
    CosmologyParameters p;
    {
        BackgroundCosmology lcdm(p), sp(p, split_model(p));
        double err = 0.0;
        for (double x = -15.0; x <= 2.0; x += 0.5)
            err = std::max(err, std::fabs(sp.Hp_of_x(x) / lcdm.Hp_of_x(x) - 1.0));
        check_close(err, 0.0, 1e-14, "Hp(x) = LCDM");
        check_rel(sp.rho_of(BG::Component::Other, -1.0), 0.5 * lcdm.rhoCDM(-1.0), 1e-14, "rho_of(Other) = dust");
        check_rel(sp.rhoCDM(-1.0), 0.5 * lcdm.rhoCDM(-1.0), 1e-14, "rhoCDM = half of LCDM");
        CosmologyParameters q = p;
        q.h = 0.7;
        check(throws([&] { BackgroundCosmology b(q, split_model(p)); }),
              "parameters that are not those of the model's background are an error");
        check(throws([&] { sp.solve(-30.0, 0.0); }), "solving beyond the range of the model's background is an error");
    }

    std::cout << "Perturbations of the custom model\n";
    {
        PerturbationOptions o;
        o.n_k = 6;
        o.k_min = 0.001 / Mpc;
        o.k_max = 0.2 / Mpc;
        Run l(p, o), s(p, split_model(p), o);
        check(s.pert.has("delta_cdm") && s.pert.has("delta_dust") && s.pert.has("Nu_0") && !l.pert.has("delta_dust"),
              "species outputs: delta_cdm, delta_dust, Nu_0");
        check(throws([&] { s.pert.get("delta_foo", 0.0, 0.01 / Mpc); }), "unknown output is an error");
        double ePhi = 0.0, eT = 0.0, eD = 0.0;
        for (double k : s.pert.get_k_array())
            for (double x : {-10.0, -7.0, -3.0, 0.0}) {
                ePhi = std::max(ePhi, std::fabs(s.pert.get_Phi(x, k) / l.pert.get_Phi(x, k) - 1.0));
                eT = std::max(eT,
                              std::fabs(s.pert.get_Source_T(x, k) - l.pert.get_Source_T(x, k)) /
                                  std::fabs(l.pert.get_Theta(0, x, k) + l.pert.get_Psi(x, k)));
                eD = std::max(eD, std::fabs(s.pert.get("delta_dust", x, k) / l.pert.get_delta_cdm(x, k) - 1.0));
                eD = std::max(eD, std::fabs(s.pert.get_delta_cdm(x, k) / l.pert.get_delta_cdm(x, k) - 1.0));
            }
        check_close(ePhi, 0.0, 1e-6, "Phi = LCDM");
        check_close(eT, 0.0, 3e-5, "S_T = LCDM (relative to Theta_0 + Psi)");
        check_close(eD, 0.0, 1e-6, "delta_dust = delta_cdm = LCDM delta_cdm");

        BackgroundCosmology cd(p, split_model(p, false));
        cd.solve();
        RecombinationHistory rd(cd);
        check(throws([&] { Perturbations pd(cd, rd, o); }), "two species with the same output names is an error");
    }

    return FML::TESTS::summary("EBS models");
}
