#ifndef FML_COSMOLOGY_BACKGROUND_PRESETS_H
#define FML_COSMOLOGY_BACKGROUND_PRESETS_H

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
//   musigma, fr, horndeski                 the LCDM / w0wa background of the models that modify the perturbations
//
// Every entry lists the parameters of the background only, with their defaults and descriptions (ModelParameter). A
// full model (background + perturbations) is an entry of the same name in the perturbation registry
// (Cosmology/Perturbations/Presets.h), which adds the parameters that only concern the perturbations (mu0, Sigma0,
// fR0, fR_n, hd_*, ds_xi, ds_cs2, ide_cs2, ide_stability_check); PERTURBATIONS::model_parameters(name) lists all of
// them and PERTURBATIONS::make_model_background(name, params, all model params) builds the background from them.
//
// Adding a model: write its Species and/or FieldEquation_<name>.h, a BackgroundModel_<name>.h with its make_ function
// and an entry in registry() (and its perturbations in Cosmology/Perturbations).
// =====================================================================================================================

#include <FML/Cosmology/Background/BackgroundModel_BransDicke.h>
#include <FML/Cosmology/Background/BackgroundModel_CoupledQuintessence.h>
#include <FML/Cosmology/Background/BackgroundModel_CubicGalileon.h>
#include <FML/Cosmology/Background/BackgroundModel_DGP.h>
#include <FML/Cosmology/Background/BackgroundModel_DarkScattering.h>
#include <FML/Cosmology/Background/BackgroundModel_EDE.h>
#include <FML/Cosmology/Background/BackgroundModel_IDE.h>
#include <FML/Cosmology/Background/BackgroundModel_LCDM.h>
#include <FML/Cosmology/Background/BackgroundModel_Quintessence.h>
#include <FML/Cosmology/Background/BackgroundModel_w0wa.h>
#include <FML/Cosmology/Background/BackgroundParameters.h>
#include <FML/Cosmology/Background/ComponentBackground.h>

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

    /// The values of a model's parameters by name
    using ModelParameters = std::map<std::string, double>;

    /// A parameter of a model: its name, default value and description (for parameter files)
    struct ModelParameter {
        std::string name;
        double default_value;
        std::string description;
    };
    using ModelParameterList = std::vector<ModelParameter>;

    struct ModelInfo {
        std::string name, description;
        /// The parameters of the background with their defaults
        ModelParameterList parameters;
        /// The background from the values of all its parameters (see parameter_values)
        std::function<ComponentBackgroundPtr(const BackgroundParameters &, const ModelParameters &, BackgroundRange)>
            build;
    };

    /// The values of the parameters of a list: the defaults, replaced by the given values. A given name that is not in
    /// the list is an error (what: the model, for the message)
    inline ModelParameters
    parameter_values(const ModelParameterList & list, const ModelParameters & given, const std::string & what) {
        ModelParameters values;
        for (const auto & q : list)
            values[q.name] = q.default_value;
        for (const auto & [key, value] : given) {
            if (not values.count(key))
                throw std::invalid_argument(what + " has no parameter [" + key + "]");
            values[key] = value;
        }
        return values;
    }

    inline const std::vector<ModelInfo> & registry() {
        static const std::string PHI_INI = "initial field phi (reduced Planck masses) at the start of the background",
                                 DPHI_INI = "initial dphi/dln a (0: at rest)",
                                 EDE_F = "f_EDE: maximum fraction rho_EDE/rho_tot",
                                 EDE_THETA = "initial field theta_i = phi_i/f (0 < theta_i < pi)",
                                 EDE_N = "exponent n of the potential (w -> (n-1)/(n+1) when oscillating)";
        static const std::vector<ModelInfo> models{
            {"lcdm",
             "Cosmological constant, GR",
             {},
             [](const BackgroundParameters & p, const ModelParameters &, BackgroundRange r) {
                 return make_lcdm(p, r);
             }},
            {"w0wa",
             "Dark energy with w = w0 + wa (1 - a), GR",
             {{"w0", -1.0, "w0 of the equation of state"}, {"wa", 0.0, "wa of the equation of state"}},
             [](const BackgroundParameters & p, const ModelParameters & m, BackgroundRange r) {
                 return make_w0wa(p, m.at("w0"), m.at("wa"), r);
             }},
            {"dgp",
             "DGP braneworld (flat): modified Friedmann equation (dgp_branch = +1 self-accelerating, -1 normal) with "
             "Lambda from E(0) = 1",
             {{"Omega_rc",
               0.25,
               "c^2/(4 r_c^2 H0^2) (0.25: H0 r_c/c = 1). 0 with dgp_branch = 1: sDGP without dark energy"},
              {"dgp_branch", -1.0, "-1: normal branch (nDGP, with Lambda), +1: self-accelerating branch (sDGP)"},
              {"dgp_lcdm_background", 0.0, "1: LCDM expansion, only the perturbations modified (nDGP of N-body work)"}},
             [](const BackgroundParameters & p, const ModelParameters & m, BackgroundRange r) {
                 const double branch = m.at("dgp_branch");
                 if (branch != 1.0 and branch != -1.0)
                     throw std::invalid_argument("[dgp] dgp_branch must be +1 or -1");
                 return make_dgp(p, m.at("Omega_rc"), int(branch), m.at("dgp_lcdm_background") != 0.0, r);
             }},
            {"quintessence",
             "Quintessence V = V0 exp(-quint_lambda phi) (phi in M_pl), V0 from E(0) = 1; GR",
             {{"quint_lambda", 1.0, "slope lambda of the potential (phi in reduced Planck masses); 0 = LCDM"},
              {"quint_phi_ini", 0.0, PHI_INI},
              {"quint_dphidlna_ini", 0.0, DPHI_INI}},
             [](const BackgroundParameters & p, const ModelParameters & m, BackgroundRange r) {
                 return make_quintessence(p,
                                          QuintessencePotential::exponential(m.at("quint_lambda")),
                                          m.at("quint_phi_ini"),
                                          m.at("quint_dphidlna_ini"),
                                          r);
             }},
            {"quintessence_inverse_power_law",
             "Quintessence V = V0 phi^-quint_alpha (Ratra-Peebles tracker, quint_phi_ini > 0), V0 from E(0) = 1; GR",
             {{"quint_alpha", 1.0, "exponent alpha of the potential V0 phi^-alpha (phi in reduced Planck masses)"},
              {"quint_phi_ini", 0.1, PHI_INI + std::string(" (> 0)")},
              {"quint_dphidlna_ini", 0.0, DPHI_INI}},
             [](const BackgroundParameters & p, const ModelParameters & m, BackgroundRange r) {
                 return make_quintessence(p,
                                          QuintessencePotential::inverse_power_law(m.at("quint_alpha")),
                                          m.at("quint_phi_ini"),
                                          m.at("quint_dphidlna_ini"),
                                          r);
             }},
            {"quintessence_pngb",
             "Quintessence V = V0 (1 + cos(phi / quint_f)) (thawing PNGB), V0 from E(0) = 1; GR",
             {{"quint_f", 1.0, "scale f of the potential V0 (1 + cos(phi/f)) (reduced Planck masses)"},
              {"quint_phi_ini", 1.0, PHI_INI},
              {"quint_dphidlna_ini", 0.0, DPHI_INI}},
             [](const BackgroundParameters & p, const ModelParameters & m, BackgroundRange r) {
                 return make_quintessence(p,
                                          QuintessencePotential::pngb(m.at("quint_f")),
                                          m.at("quint_phi_ini"),
                                          m.at("quint_dphidlna_ini"),
                                          r);
             }},
            {"quintessence_hilltop",
             "Quintessence V = V0 (1 - (|phi| / quint_mu)^quint_p)^2 (thawing hilltop), V0 from E(0) = 1; GR",
             {{"quint_mu", 1.0, "scale mu of the potential V0 (1 - (|phi|/mu)^p)^2 (reduced Planck masses)"},
              {"quint_p", 2.0, "exponent p of the hilltop potential"},
              {"quint_phi_ini", 0.1, PHI_INI},
              {"quint_dphidlna_ini", 0.0, DPHI_INI}},
             [](const BackgroundParameters & p, const ModelParameters & m, BackgroundRange r) {
                 return make_quintessence(p,
                                          QuintessencePotential::hilltop(m.at("quint_mu"), m.at("quint_p")),
                                          m.at("quint_phi_ini"),
                                          m.at("quint_dphidlna_ini"),
                                          r);
             }},
            {"ede",
             "Early dark energy V ~ (1 - cos(phi/f))^ede_n with f_EDE = max rho_EDE/rho_tot at z_c, + Lambda from E(0) "
             "= 1; GR",
             {{"ede_fEDE", 0.1, EDE_F},
              {"ede_log10zc", 3.5, "log10 of the redshift z_c of the maximum"},
              {"ede_theta_i", 2.8, EDE_THETA},
              {"ede_n", 3.0, EDE_N}},
             [](const BackgroundParameters & p, const ModelParameters & m, BackgroundRange r) {
                 return make_ede(p, m.at("ede_fEDE"), m.at("ede_log10zc"), m.at("ede_theta_i"), m.at("ede_n"), r);
             }},
            {"ede_zc",
             "Early dark energy as \"ede\" with the redshift ede_zc itself (SimulationCode's cosmology_ede_zc) instead "
             "of its log10; GR",
             {{"ede_fEDE", 0.1, EDE_F},
              {"ede_zc", 3500.0, "redshift z_c of the maximum"},
              {"ede_theta_i", 2.8, EDE_THETA},
              {"ede_n", 3.0, EDE_N}},
             [](const BackgroundParameters & p, const ModelParameters & m, BackgroundRange r) {
                 if (not(m.at("ede_zc") > 0.0))
                     throw std::invalid_argument("[ede_zc] Need ede_zc > 0");
                 return make_ede(p,
                                 m.at("ede_fEDE"),
                                 std::log10(m.at("ede_zc")),
                                 m.at("ede_theta_i"),
                                 m.at("ede_n"),
                                 r);
             }},
            {"jbd",
             "Jordan-Brans-Dicke with Lambda; phi today from bd_GeffG_today, Omega_Lambda from E(0) = 1",
             {{"bd_omega", 1000.0, "Brans-Dicke parameter omega (GR as omega -> infinity; Cassini: omega > 40000)"},
              {"bd_GeffG_today", 1.0, "G_eff/G today: phi(0) = phi_*/bd_GeffG_today (1: the Newtonian G today)"},
              {"bd_density_parameter_definition",
               0.0,
               "the G of the density parameters (SimulationCode's cosmology_JBD_density_parameter_definition, "
               "FieldEquation_BransDicke.h): 0 = hi-class (G of the action), 1 = Gtoday (G/phi(0)), 2 = Gnewton "
               "(G/phi_*), 3 = Gbare"}},
             [](const BackgroundParameters & p, const ModelParameters & m, BackgroundRange r) {
                 const double d = m.at("bd_density_parameter_definition");
                 if (d != 0.0 and d != 1.0 and d != 2.0 and d != 3.0)
                     throw std::invalid_argument("[jbd] bd_density_parameter_definition must be 0, 1, 2 or 3");
                 return make_jbd(p, m.at("bd_omega"), m.at("bd_GeffG_today"), BransDickeDensityDefinition(int(d)), r);
             }},
            {"coupled_quintessence",
             "Quintessence (V ~ exp(-lambda phi)) coupled to CDM (m ~ exp(-beta phi)), V0 from closure, GR",
             {{"cq_lambda", 0.5, "slope lambda of the potential (phi in reduced Planck masses)"},
              {"cq_beta", 0.05, "coupling beta (Amendola's sqrt(2/3) beta); CDM feels G (1 + 2 beta^2)"}},
             [](const BackgroundParameters & p, const ModelParameters & m, BackgroundRange r) {
                 return make_coupled_quintessence(p, m.at("cq_lambda"), m.at("cq_beta"), r);
             }},
            {"cubic_galileon",
             "Cubic Galileon on its tracker (flat, no Lambda): Omega_phi from closure, no free parameter",
             {},
             [](const BackgroundParameters & p, const ModelParameters &, BackgroundRange r) {
                 return make_cubic_galileon(p, r);
             }},
            {"dark_scattering",
             "Dark matter - dark energy scattering: the background is wCDM with w = ds_w (the scattering only affects "
             "the perturbations)",
             {{"ds_w", -0.9, "equation of state w of the dark energy (constant, != -1)"}},
             [](const BackgroundParameters & p, const ModelParameters & m, BackgroundRange r) {
                 return make_dark_scattering(p, m.at("ds_w"), r);
             }},
            {"ide",
             "Dark-energy fluid (constant w) exchanging energy with CDM, Q = xi H rho_DE",
             {{"ide_xi", -0.1, "coupling xi (> 0: energy flows to the dark matter)"},
              {"ide_w", -0.95, "equation of state w of the dark energy (constant, != -1)"}},
             [](const BackgroundParameters & p, const ModelParameters & m, BackgroundRange r) {
                 return make_ide(p, m.at("ide_w"), m.at("ide_xi"), r);
             }},
            {"musigma",
             "mu-Sigma: the LCDM background (mu0, Sigma0 only modify the perturbations)",
             {},
             [](const BackgroundParameters & p, const ModelParameters &, BackgroundRange r) {
                 return make_lcdm(p, r);
             }},
            {"fr",
             "Hu-Sawicky f(R): the LCDM background (to O(fR0); fR0, fR_n only modify the perturbations)",
             {},
             [](const BackgroundParameters & p, const ModelParameters &, BackgroundRange r) {
                 return make_lcdm(p, r);
             }},
            {"horndeski",
             "Horndeski (alpha parameterization): the LCDM or w0wa background from hd_w0, hd_wa (the alphas only "
             "modify the perturbations)",
             {{"hd_w0", -1.0, "w0 of the expansion (w0-wa dark energy, GR Friedmann equation)"},
              {"hd_wa", 0.0, "wa of the expansion"}},
             [](const BackgroundParameters & p, const ModelParameters & m, BackgroundRange r) {
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

    /// The model by name. model_parameters: parameters of its background (others are an error, also those of its
    /// perturbations: see PERTURBATIONS::make_model_background; missing ones take the defaults of the registry)
    inline ComponentBackgroundPtr make_background(const std::string & model,
                                                  const BackgroundParameters & p,
                                                  const ModelParameters & model_parameters = {},
                                                  BackgroundRange r = {}) {
        const auto & info = find_model(model);
        return info.build(p,
                          parameter_values(info.parameters, model_parameters, "[Background] Model [" + model + "]"),
                          r);
    }

} // namespace FML::COSMOLOGY::BACKGROUND

#endif
