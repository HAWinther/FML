#ifndef FML_COSMOLOGY_PERTURBATIONS_PRESETS_H
#define FML_COSMOLOGY_PERTURBATIONS_PRESETS_H

// =====================================================================================================================
// The registry of the full models (background + perturbations) by name, and make_perturbation_model(name, params, model
// params). It is built on the background registry (Cosmology/Background/Presets.h): every entry has the name of the
// background registry entry it is built on and adds the parameters that only concern the perturbations, so the
// parameters of a model are those of its background (BACKGROUND::registry()) followed by those of its perturbations
// (here), each listed once with its default and description (model_parameters(name); parameter files, --write-defaults
// of the Einstein-Boltzmann solver):
//   lcdm                                   PerturbationModel_LCDM.h
//   w0wa                                   PerturbationModel_w0wa.h                  (smooth dark energy, warns)
//   musigma                                PerturbationModel_MuSigma.h               mu0, Sigma0
//   dgp                                    PerturbationModel_DGP.h
//   fr                                     PerturbationModel_HuSawicky.h             fR0, fR_n
//   horndeski                              PerturbationModel_Horndeski.h             hd_parametrization, hd_cK, hd_cB,
//                                                                                    hd_cM, hd_M2_ini,
//                                                                                    hd_stability_check
//   quintessence(_inverse_power_law, ...)  PerturbationModel_Quintessence.h
//   ede, ede_zc                            PerturbationModel_EDE.h
//   axion                                  PerturbationModel_Axion.h
//   jbd                                    PerturbationModel_BransDicke.h
//   coupled_quintessence                   PerturbationModel_CoupledQuintessence.h
//   cubic_galileon                         PerturbationModel_CubicGalileon.h
//   dark_scattering                        PerturbationModel_DarkScattering.h        ds_xi, ds_cs2
//   ide                                    PerturbationModel_IDE.h                   ide_cs2, ide_stability_check
// make_perturbation_model(name, background, model params) builds the model on a background made by
// make_model_background with the same name and parameters (e.g. on another range).
// Adding a model: its PerturbationModel_<name>.h (and SpeciesPerturbation_ / PerturbedFieldEquation_ headers) and an
// entry in perturbation_registry() (its background and the background's parameters in BACKGROUND::registry()).
// =====================================================================================================================

#include <FML/Cosmology/Background/BackgroundParameters.h>
#include <FML/Cosmology/Background/ComponentBackground.h>
#include <FML/Cosmology/Background/Presets.h>
#include <FML/Cosmology/Perturbations/PerturbationModel.h>
#include <FML/Cosmology/Perturbations/PerturbationModel_Axion.h>
#include <FML/Cosmology/Perturbations/PerturbationModel_BransDicke.h>
#include <FML/Cosmology/Perturbations/PerturbationModel_CoupledQuintessence.h>
#include <FML/Cosmology/Perturbations/PerturbationModel_CubicGalileon.h>
#include <FML/Cosmology/Perturbations/PerturbationModel_DGP.h>
#include <FML/Cosmology/Perturbations/PerturbationModel_DarkScattering.h>
#include <FML/Cosmology/Perturbations/PerturbationModel_EDE.h>
#include <FML/Cosmology/Perturbations/PerturbationModel_Horndeski.h>
#include <FML/Cosmology/Perturbations/PerturbationModel_HuSawicky.h>
#include <FML/Cosmology/Perturbations/PerturbationModel_IDE.h>
#include <FML/Cosmology/Perturbations/PerturbationModel_LCDM.h>
#include <FML/Cosmology/Perturbations/PerturbationModel_MuSigma.h>
#include <FML/Cosmology/Perturbations/PerturbationModel_Quintessence.h>
#include <FML/Cosmology/Perturbations/PerturbationModel_w0wa.h>

#include <functional>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace FML::COSMOLOGY::PERTURBATIONS {

    using ModelParameters = BACKGROUND::ModelParameters;
    using ModelParameter = BACKGROUND::ModelParameter;
    using ModelParameterList = BACKGROUND::ModelParameterList;

    struct PerturbationModelInfo {
        /// The name, also that of the background in BACKGROUND::registry() the model is built on
        std::string name;
        /// The full model (background and perturbations)
        std::string description;
        /// The parameters that only concern the perturbations, with their defaults (those of the background are in
        /// BACKGROUND::registry())
        ModelParameterList parameters;
        /// The model on its background; m: the values of all the model's parameters (model_parameter_values)
        std::function<PerturbationModelPtr(BACKGROUND::ComponentBackgroundPtr, const ModelParameters & m)> build;
    };

    inline const std::vector<PerturbationModelInfo> & perturbation_registry() {
        using BG = BACKGROUND::ComponentBackgroundPtr;
        using M = const ModelParameters &;
        auto quintessence = [](BG bg, M) {
            return std::make_shared<const PerturbationModel_Quintessence>(std::move(bg));
        };
        auto ede = [](BG bg, M) { return std::make_shared<const PerturbationModel_EDE>(std::move(bg)); };
        static const std::string QUINTESSENCE = "thawing from phi_ini; V0 from closure; with its perturbations; GR",
                                 EDE = "with its perturbations, + Lambda (closure); GR";
        static const std::vector<PerturbationModelInfo> models{
            {"lcdm",
             "LCDM (+ curvature, massive neutrinos) in general relativity",
             {},
             [](BG bg, M) { return std::make_shared<const PerturbationModel_LCDM>(std::move(bg)); }},
            {"w0wa",
             "dark energy with w(a) = w0 + wa (1 - a), smooth (no perturbations: for background data); GR",
             {},
             [](BG bg, M) { return std::make_shared<const PerturbationModel_w0wa>(std::move(bg)); }},
            {"musigma",
             "LCDM background, perturbations with mu(a), Sigma(a) of Planck/DES (flat)",
             {{"mu0", 0.0, "mu = 1 + mu0 Omega_DE(a)/Omega_DE,0: modified growth (Psi)"},
              {"Sigma0", 0.0, "Sigma = 1 + Sigma0 Omega_DE(a)/Omega_DE,0: modified lensing (Psi - Phi)"}},
             [](BG bg, M m) {
                 return std::make_shared<const PerturbationModel_MuSigma>(std::move(bg), m.at("mu0"), m.at("Sigma0"));
             }},
            {"dgp",
             "DGP braneworld (flat): modified Friedmann equation, quasi-static mu(a), Sigma = 1; Omega_Lambda from "
             "E(0) "
             "= 1",
             {},
             [](BG bg, M) { return std::make_shared<const PerturbationModel_DGP>(std::move(bg)); }},
            {"fr",
             "Hu-Sawicky f(R) (flat): LCDM background, quasi-static mu(k, a) and Sigma(a)",
             {{"fR0", 1e-5, "|f_R0|, the scalar field today (GR for 0)"},
              {"fR_n", 1.0, "exponent n of the Hu-Sawicky model"}},
             [](BG bg, M m) {
                 return std::make_shared<const PerturbationModel_HuSawicky>(std::move(bg), m.at("fR0"), m.at("fR_n"));
             }},
            {"horndeski",
             "Horndeski in the alpha parameterization of hi_class (alpha_T = 0, flat): LCDM or w0-wa expansion, "
             "alpha_X = c_X f(a); stability checked",
             {{"hd_parametrization",
               0.0,
               "f(a): 0 = Omega_DE(a) (propto_omega), 1 = a (propto_scale), 2 = 1 (constant_alphas)"},
              {"hd_cK", 1.0, "c_K: kineticity alpha_K = c_K f(a)"},
              {"hd_cB", 0.0, "c_B: braiding alpha_B = c_B f(a) (Bellini-Sawicki convention, as hi_class)"},
              {"hd_cM", 0.0, "c_M: Planck-mass run rate alpha_M = dln M*^2/dln a = c_M f(a)"},
              {"hd_M2_ini", 1.0, "M*^2 at early times (a = 1e-14 for constant_alphas)"},
              {"hd_stability_check",
               1.0,
               "1: stop for a ghost (D < 0) or gradient (c_s^2 < 0) instability, M*^2 <= 0, or c_s^2 > 1e5 (too "
               "stiff)"}},
             [](BG bg, M m) {
                 return std::make_shared<const PerturbationModel_Horndeski>(
                     std::move(bg),
                     make_standard_alphas(m.at("hd_parametrization"),
                                          m.at("hd_cK"),
                                          m.at("hd_cB"),
                                          m.at("hd_cM"),
                                          m.at("hd_M2_ini")),
                     m.at("hd_stability_check") != 0.0);
             }},
            {"quintessence", "scalar field with V = V0 exp(-lambda phi), " + QUINTESSENCE, {}, quintessence},
            {"quintessence_inverse_power_law",
             "scalar field with V = V0 phi^-alpha (Ratra-Peebles tracker), " + QUINTESSENCE,
             {},
             quintessence},
            {"quintessence_pngb",
             "scalar field with V = V0 (1 + cos(phi/f)) (PNGB), " + QUINTESSENCE,
             {},
             quintessence},
            {"quintessence_hilltop",
             "scalar field with V = V0 (1 - (|phi|/mu)^p)^2 (hilltop), " + QUINTESSENCE,
             {},
             quintessence},
            {"ede", "axion-like early dark energy V ~ (1 - cos(phi/f))^n " + EDE, {}, ede},
            {"ede_zc", "as ede with the redshift ede_zc of the maximum instead of its log10, " + EDE, {}, ede},
            {"axion",
             "mixed dark matter: CDM + an ultralight axion field (Klein-Gordon, then the WKB fluid with c_s^2 = k^2 / "
             "(k^2 + 4 m^2 a^2)), + Lambda (closure); GR",
             {{"ax_log10_Hinf",
               0.0,
               "log10 of the Hubble rate of inflation [GeV] for the axion isocurvature (axionCAMB's Hinf; sets f_axi, "
               "n_axi from the initial field; <= 0: off)"}},
             [](BG bg, M) { return std::make_shared<const PerturbationModel_Axion>(std::move(bg)); }},
            {"jbd",
             "Jordan-Brans-Dicke (flat) with Lambda; phi today gives G_eff = G bd_GeffG_today, Omega_Lambda from E(0) "
             "= 1",
             {},
             [](BG bg, M) { return std::make_shared<const PerturbationModel_BransDicke>(std::move(bg)); }},
            {"coupled_quintessence",
             "quintessence (V ~ exp(-lambda phi)) coupled to CDM (m ~ exp(-beta phi)); GR",
             {},
             [](BG bg, M) { return std::make_shared<const PerturbationModel_CoupledQuintessence>(std::move(bg)); }},
            {"cubic_galileon",
             "cubic Galileon on its tracker (flat, no Lambda): Omega_phi from closure, no free parameter",
             {},
             [](BG bg, M) { return std::make_shared<const PerturbationModel_CubicGalileon>(std::move(bg)); }},
            {"dark_scattering",
             "wCDM (dark-energy fluid with perturbations) with elastic dark matter - dark energy scattering; GR",
             {{"ds_xi", 10.0, "xi = sigma_D / m_c [barn/GeV] (0: wCDM without scattering)"},
              {"ds_cs2", 1.0, "rest-frame sound speed squared of the dark energy"}},
             [](BG bg, M m) {
                 return std::make_shared<const PerturbationModel_DarkScattering>(std::move(bg),
                                                                                 m.at("ds_xi"),
                                                                                 m.at("ds_cs2"));
             }},
            {"ide",
             "dark-energy fluid (w, cs2) exchanging energy with CDM, Q = xi H rho_DE along u_c; GR",
             {{"ide_cs2", 1.0, "rest-frame sound speed squared of the dark energy"},
              {"ide_stability_check", 1.0, "1: stop if the doom factor xi/(3(1+w)) > 0 (large-scale instability)"}},
             [](BG bg, M m) {
                 return std::make_shared<const PerturbationModel_IDE>(std::move(bg),
                                                                      m.at("ide_cs2"),
                                                                      m.at("ide_stability_check") != 0.0);
             }},
        };
        return models;
    }

    inline const PerturbationModelInfo & find_perturbation_model(const std::string & model) {
        for (const auto & m : perturbation_registry())
            if (m.name == model)
                return m;
        std::string known;
        for (const auto & m : perturbation_registry())
            known += " " + m.name;
        throw std::invalid_argument("[Perturbations] Unknown model [" + model + "] (known:" + known + ")");
    }

    /// All parameters of a model: those of its background (BACKGROUND::registry()), then those of its perturbations
    inline ModelParameterList model_parameters(const std::string & model) {
        const auto & info = find_perturbation_model(model);
        auto list = BACKGROUND::find_model(model).parameters;
        list.insert(list.end(), info.parameters.begin(), info.parameters.end());
        return list;
    }

    /// The values of all parameters of a model: the defaults, replaced by the given values. A parameter that is not
    /// one of the model's is an error
    inline ModelParameters model_parameter_values(const std::string & model, const ModelParameters & given) {
        return BACKGROUND::parameter_values(model_parameters(model), given, "Model [" + model + "]");
    }

    /// Of the given parameters of a model (checked as by model_parameter_values), those of its background
    inline ModelParameters background_model_parameters(const std::string & model, const ModelParameters & given) {
        model_parameter_values(model, given);
        ModelParameters b;
        for (const auto & q : BACKGROUND::find_model(model).parameters)
            if (given.count(q.name))
                b[q.name] = given.at(q.name);
        return b;
    }

    /// The background of a model from all its parameters (those of the perturbations are checked, and not needed)
    inline BACKGROUND::ComponentBackgroundPtr make_model_background(const std::string & model,
                                                                    const BACKGROUND::BackgroundParameters & p,
                                                                    const ModelParameters & model_parameters = {},
                                                                    BACKGROUND::BackgroundRange r = {}) {
        return BACKGROUND::make_background(model, p, background_model_parameters(model, model_parameters), r);
    }

    /// The model by name on a background made by make_model_background(model, ..., model_parameters). Parameters that
    /// are not the model's are an error, missing ones take the defaults
    inline PerturbationModelPtr make_perturbation_model(const std::string & model,
                                                        BACKGROUND::ComponentBackgroundPtr background,
                                                        const ModelParameters & model_parameters = {}) {
        const auto & info = find_perturbation_model(model);
        return info.build(std::move(background), model_parameter_values(model, model_parameters));
    }

    /// The model by name with its background. model_parameters: all its parameters (background and perturbations)
    inline PerturbationModelPtr make_perturbation_model(const std::string & model,
                                                        const BACKGROUND::BackgroundParameters & p,
                                                        const ModelParameters & model_parameters = {},
                                                        BACKGROUND::BackgroundRange r = {}) {
        return make_perturbation_model(model, make_model_background(model, p, model_parameters, r), model_parameters);
    }

} // namespace FML::COSMOLOGY::PERTURBATIONS

#endif
