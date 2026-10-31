#ifndef FML_BACKGROUND_PRESETS_H
#define FML_BACKGROUND_PRESETS_H

// =====================================================================================================================
// The registry of the background models by name, and make_background(name, params, model params).
//
// Each model is a BackgroundModel_<name>.h with its make_ function (species + field equations, see
// ComponentBackground.h); the field equations other than GR are in FieldEquation_<name>.h:
//   lcdm                                   BackgroundModel_LCDM.h                  make_lcdm
//   w0wa                                   BackgroundModel_w0wa.h                  make_w0wa
//   dgp                                    BackgroundModel_DGP.h                   make_dgp
//   quintessence(_inverse_power_law, ...)  BackgroundModel_Quintessence.h          make_quintessence
//   ede, ede_zc                            BackgroundModel_EDE.h                   make_ede
//   jbd                                    BackgroundModel_BransDicke.h            make_jbd
//   coupled_quintessence                   BackgroundModel_CoupledQuintessence.h   make_coupled_quintessence
//   cubic_galileon                         BackgroundModel_CubicGalileon.h         make_cubic_galileon
//   dark_scattering                        BackgroundModel_DarkScattering.h        make_dark_scattering
//   ide                                    BackgroundModel_IDE.h                   make_ide
//   musigma, fr, horndeski                 the LCDM / w0wa background (their parameters only modify the
//                                          perturbations, which are not part of this module)
//
// Adding a model: write its Species and/or FieldEquation_<name>.h, a BackgroundModel_<name>.h with its make_ function
// and an entry in registry().
// =====================================================================================================================

#include <FML/Background/BackgroundModel_BransDicke.h>
#include <FML/Background/BackgroundModel_CoupledQuintessence.h>
#include <FML/Background/BackgroundModel_CubicGalileon.h>
#include <FML/Background/BackgroundModel_DGP.h>
#include <FML/Background/BackgroundModel_DarkScattering.h>
#include <FML/Background/BackgroundModel_EDE.h>
#include <FML/Background/BackgroundModel_IDE.h>
#include <FML/Background/BackgroundModel_LCDM.h>
#include <FML/Background/BackgroundModel_Quintessence.h>
#include <FML/Background/BackgroundModel_w0wa.h>
#include <FML/Background/BackgroundParameters.h>
#include <FML/Background/ComponentBackground.h>

#include <functional>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace FML::COSMOLOGY::BACKGROUND {

    // -----------------------------------------------------------------------------------------------------------------
    // Registry
    // -----------------------------------------------------------------------------------------------------------------

    struct ModelInfo {
        std::string name, description;
        /// The model's own parameters and their defaults
        std::map<std::string, double> parameters;
        std::function<ComponentBackgroundPtr(const BackgroundParameters &,
                                             const std::map<std::string, double> &,
                                             BackgroundRange)>
            build;
    };

    inline const std::vector<ModelInfo> & registry() {
        static const std::vector<ModelInfo> models{
            {"lcdm",
             "Cosmological constant, GR",
             {},
             [](const BackgroundParameters & p, const std::map<std::string, double> &, BackgroundRange r) {
                 return make_lcdm(p, r);
             }},
            {"w0wa",
             "Dark energy with w = w0 + wa (1 - a), GR",
             {{"w0", -1.0}, {"wa", 0.0}},
             [](const BackgroundParameters & p, const std::map<std::string, double> & m, BackgroundRange r) {
                 return make_w0wa(p, m.at("w0"), m.at("wa"), r);
             }},
            {"dgp",
             "DGP braneworld (dgp_branch = +1 self-accelerating, -1 normal) with Lambda",
             {{"Omega_rc", 0.25}, {"dgp_branch", -1.0}, {"dgp_lcdm_background", 0.0}},
             [](const BackgroundParameters & p, const std::map<std::string, double> & m, BackgroundRange r) {
                 return make_dgp(p, m.at("Omega_rc"), int(m.at("dgp_branch")), m.at("dgp_lcdm_background") != 0.0, r);
             }},
            {"quintessence",
             "Quintessence V = V0 exp(-quint_lambda phi) (phi in M_pl), V0 from E(0) = 1; GR",
             {{"quint_lambda", 1.0}, {"quint_phi_ini", 0.0}, {"quint_dphidlna_ini", 0.0}},
             [](const BackgroundParameters & p, const std::map<std::string, double> & m, BackgroundRange r) {
                 return make_quintessence(p,
                                          QuintessencePotential::exponential(m.at("quint_lambda")),
                                          m.at("quint_phi_ini"),
                                          m.at("quint_dphidlna_ini"),
                                          r);
             }},
            {"quintessence_inverse_power_law",
             "Quintessence V = V0 phi^-quint_alpha (Ratra-Peebles tracker, quint_phi_ini > 0), V0 from E(0) = 1; GR",
             {{"quint_alpha", 1.0}, {"quint_phi_ini", 0.1}, {"quint_dphidlna_ini", 0.0}},
             [](const BackgroundParameters & p, const std::map<std::string, double> & m, BackgroundRange r) {
                 return make_quintessence(p,
                                          QuintessencePotential::inverse_power_law(m.at("quint_alpha")),
                                          m.at("quint_phi_ini"),
                                          m.at("quint_dphidlna_ini"),
                                          r);
             }},
            {"quintessence_pngb",
             "Quintessence V = V0 (1 + cos(phi / quint_f)) (thawing PNGB), V0 from E(0) = 1; GR",
             {{"quint_f", 1.0}, {"quint_phi_ini", 1.0}, {"quint_dphidlna_ini", 0.0}},
             [](const BackgroundParameters & p, const std::map<std::string, double> & m, BackgroundRange r) {
                 return make_quintessence(p,
                                          QuintessencePotential::pngb(m.at("quint_f")),
                                          m.at("quint_phi_ini"),
                                          m.at("quint_dphidlna_ini"),
                                          r);
             }},
            {"quintessence_hilltop",
             "Quintessence V = V0 (1 - (|phi| / quint_mu)^quint_p)^2 (thawing hilltop), V0 from E(0) = 1; GR",
             {{"quint_mu", 1.0}, {"quint_p", 2.0}, {"quint_phi_ini", 0.1}, {"quint_dphidlna_ini", 0.0}},
             [](const BackgroundParameters & p, const std::map<std::string, double> & m, BackgroundRange r) {
                 return make_quintessence(p,
                                          QuintessencePotential::hilltop(m.at("quint_mu"), m.at("quint_p")),
                                          m.at("quint_phi_ini"),
                                          m.at("quint_dphidlna_ini"),
                                          r);
             }},
            {"ede",
             "Early dark energy V ~ (1 - cos(phi/f))^ede_n with f_EDE = max rho_EDE/rho_tot at z_c, + Lambda from E(0) "
             "= 1; GR",
             {{"ede_fEDE", 0.1}, {"ede_log10zc", 3.5}, {"ede_theta_i", 2.8}, {"ede_n", 3.0}},
             [](const BackgroundParameters & p, const std::map<std::string, double> & m, BackgroundRange r) {
                 return make_ede(p, m.at("ede_fEDE"), m.at("ede_log10zc"), m.at("ede_theta_i"), m.at("ede_n"), r);
             }},
            {"ede_zc",
             "Early dark energy as \"ede\" with the redshift ede_zc itself (SimulationCode's cosmology_ede_zc) instead "
             "of its log10; GR",
             {{"ede_fEDE", 0.1}, {"ede_zc", 3500.0}, {"ede_theta_i", 2.8}, {"ede_n", 3.0}},
             [](const BackgroundParameters & p, const std::map<std::string, double> & m, BackgroundRange r) {
                 if (not(m.at("ede_zc") > 0.0))
                     throw std::invalid_argument("[ede_zc] Need ede_zc > 0");
                 return make_ede(p, m.at("ede_fEDE"), std::log10(m.at("ede_zc")), m.at("ede_theta_i"), m.at("ede_n"), r);
             }},
            {"jbd",
             "Jordan-Brans-Dicke with Lambda; phi today from bd_GeffG_today, Omega_Lambda from E(0) = 1. "
             "bd_density_parameter_definition: 0 = hi-class (the Einstein-Boltzmann solver), 1 = Gtoday, 2 = Gnewton, "
             "3 = Gbare (SimulationCode's cosmology_JBD_density_parameter_definition, see FieldEquation_BransDicke.h)",
             {{"bd_omega", 1000.0}, {"bd_GeffG_today", 1.0}, {"bd_density_parameter_definition", 0.0}},
             [](const BackgroundParameters & p, const std::map<std::string, double> & m, BackgroundRange r) {
                 const double d = m.at("bd_density_parameter_definition");
                 if (d != 0.0 and d != 1.0 and d != 2.0 and d != 3.0)
                     throw std::invalid_argument("[jbd] bd_density_parameter_definition must be 0, 1, 2 or 3");
                 return make_jbd(p, m.at("bd_omega"), m.at("bd_GeffG_today"), BransDickeDensityDefinition(int(d)), r);
             }},
            {"coupled_quintessence",
             "Quintessence (V ~ exp(-lambda phi)) coupled to CDM (m ~ exp(-beta phi)), V0 from closure, GR",
             {{"cq_lambda", 0.5}, {"cq_beta", 0.05}},
             [](const BackgroundParameters & p, const std::map<std::string, double> & m, BackgroundRange r) {
                 return make_coupled_quintessence(p, m.at("cq_lambda"), m.at("cq_beta"), r);
             }},
            {"cubic_galileon",
             "Cubic Galileon on its tracker (flat, no Lambda): Omega_phi from closure, no free parameter",
             {},
             [](const BackgroundParameters & p, const std::map<std::string, double> &, BackgroundRange r) {
                 return make_cubic_galileon(p, r);
             }},
            {"dark_scattering",
             "Dark matter - dark energy scattering: the background is wCDM with w = ds_w (ds_xi, ds_cs2 only affect the "
             "perturbations and are ignored here)",
             {{"ds_w", -0.9}, {"ds_xi", 10.0}, {"ds_cs2", 1.0}},
             [](const BackgroundParameters & p, const std::map<std::string, double> & m, BackgroundRange r) {
                 return make_dark_scattering(p, m.at("ds_w"), r);
             }},
            {"ide",
             "Dark-energy fluid (constant w) exchanging energy with CDM, Q = xi H rho_DE (ide_cs2 and "
             "ide_stability_check only concern the perturbations and are ignored here)",
             {{"ide_xi", -0.1}, {"ide_w", -0.95}, {"ide_cs2", 1.0}, {"ide_stability_check", 1.0}},
             [](const BackgroundParameters & p, const std::map<std::string, double> & m, BackgroundRange r) {
                 return make_ide(p, m.at("ide_w"), m.at("ide_xi"), r);
             }},
            {"musigma",
             "LCDM background (mu0, Sigma0 only modify the perturbations and are ignored here)",
             {{"mu0", 0.0}, {"Sigma0", 0.0}},
             [](const BackgroundParameters & p, const std::map<std::string, double> &, BackgroundRange r) { return make_lcdm(p, r); }},
            {"fr",
             "Hu-Sawicky f(R): LCDM background (fR0, fR_n only modify the perturbations and are ignored here)",
             {{"fR0", 1e-5}, {"fR_n", 1.0}},
             [](const BackgroundParameters & p, const std::map<std::string, double> &, BackgroundRange r) { return make_lcdm(p, r); }},
            {"horndeski",
             "Horndeski (alpha parameterization): LCDM or w0wa background from hd_w0, hd_wa (the other parameters only "
             "modify the perturbations and are ignored here)",
             {{"hd_parametrization", 0.0},
              {"hd_cK", 1.0},
              {"hd_cB", 0.0},
              {"hd_cM", 0.0},
              {"hd_M2_ini", 1.0},
              {"hd_w0", -1.0},
              {"hd_wa", 0.0},
              {"hd_stability_check", 1.0}},
             [](const BackgroundParameters & p, const std::map<std::string, double> & m, BackgroundRange r) {
                 const double w0 = m.at("hd_w0"), wa = m.at("hd_wa");
                 return (w0 == -1.0 and wa == 0.0) ? make_lcdm(p, r) : make_w0wa(p, w0, wa, r);
             }},
        };
        return models;
    }

    inline const ModelInfo & find_model(const std::string & name) {
        for (const auto & m : registry())
            if (m.name == name)
                return m;
        std::string known;
        for (const auto & m : registry())
            known += " " + m.name;
        throw std::invalid_argument("[Background] Unknown model [" + name + "] (known:" + known + ")");
    }

    /// The model by name. model_parameters: the model's parameters (others are an error; missing ones take the
    /// defaults of the registry)
    inline ComponentBackgroundPtr make_background(const std::string & model,
                                                  const BackgroundParameters & p,
                                                  const std::map<std::string, double> & model_parameters = {},
                                                  BackgroundRange r = {}) {
        const auto & info = find_model(model);
        auto values = info.parameters;
        for (const auto & [key, value] : model_parameters) {
            if (not values.count(key))
                throw std::invalid_argument("[Background] Model [" + model + "] has no parameter [" + key + "]");
            values[key] = value;
        }
        return info.build(p, values, r);
    }

} // namespace FML::COSMOLOGY::BACKGROUND

#endif
