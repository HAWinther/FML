#ifndef FML_PERTURBATIONS_PRESETS_H
#define FML_PERTURBATIONS_PRESETS_H

// =====================================================================================================================
// The registry of the perturbation models by name, and make_perturbation_model(name, params, model params). The names
// and the model parameters (with their defaults and checks) are those of the background registry
// (Background/Presets.h make_background), which builds the background the model is built on; the parameters that only
// concern the perturbations (mu0, Sigma0, fR0, fR_n, hd_*, ds_xi, ds_cs2, ide_cs2, ide_stability_check) are read here:
//   lcdm                                   PerturbationModel_LCDM.h
//   w0wa                                   PerturbationModel_w0wa.h                  (smooth dark energy, warns)
//   musigma                                PerturbationModel_MuSigma.h               (mu0, Sigma0)
//   dgp                                    PerturbationModel_DGP.h
//   fr                                     PerturbationModel_HuSawicky.h             (fR0, fR_n)
//   horndeski                              PerturbationModel_Horndeski.h             (hd_parametrization, hd_cK, hd_cB,
//                                                                                     hd_cM, hd_M2_ini,
//                                                                                     hd_stability_check)
//   quintessence(_inverse_power_law, ...)  PerturbationModel_Quintessence.h
//   ede, ede_zc                            PerturbationModel_EDE.h
//   jbd                                    PerturbationModel_BransDicke.h
//   coupled_quintessence                   PerturbationModel_CoupledQuintessence.h
//   cubic_galileon                         PerturbationModel_CubicGalileon.h
//   dark_scattering                        PerturbationModel_DarkScattering.h        (ds_xi, ds_cs2)
//   ide                                    PerturbationModel_IDE.h                   (ide_cs2, ide_stability_check)
// make_perturbation_model(name, background, model params) builds the model on a background made by make_background
// with the same name and parameters (e.g. on another range).
// Adding a model: its PerturbationModel_<name>.h (and SpeciesPerturbation_ / PerturbedFieldEquation_ headers) and an
// entry in perturbation_registry() (its background and parameters in BACKGROUND::registry()).
// =====================================================================================================================

#include <FML/Background/BackgroundParameters.h>
#include <FML/Background/ComponentBackground.h>
#include <FML/Background/Presets.h>
#include <FML/Perturbations/PerturbationModel.h>
#include <FML/Perturbations/PerturbationModel_BransDicke.h>
#include <FML/Perturbations/PerturbationModel_CoupledQuintessence.h>
#include <FML/Perturbations/PerturbationModel_CubicGalileon.h>
#include <FML/Perturbations/PerturbationModel_DGP.h>
#include <FML/Perturbations/PerturbationModel_DarkScattering.h>
#include <FML/Perturbations/PerturbationModel_EDE.h>
#include <FML/Perturbations/PerturbationModel_Horndeski.h>
#include <FML/Perturbations/PerturbationModel_HuSawicky.h>
#include <FML/Perturbations/PerturbationModel_IDE.h>
#include <FML/Perturbations/PerturbationModel_LCDM.h>
#include <FML/Perturbations/PerturbationModel_MuSigma.h>
#include <FML/Perturbations/PerturbationModel_Quintessence.h>
#include <FML/Perturbations/PerturbationModel_w0wa.h>

#include <functional>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace FML::COSMOLOGY::PERTURBATIONS {

    using ModelParameters = std::map<std::string, double>;

    struct PerturbationModelInfo {
        std::string name;
        /// The model on its background (built by the background registry); m: all the model's parameters (the
        /// defaults of the background registry with the given values)
        std::function<PerturbationModelPtr(BACKGROUND::ComponentBackgroundPtr, const ModelParameters & m)> build;
    };

    inline const std::vector<PerturbationModelInfo> & perturbation_registry() {
        using BG = BACKGROUND::ComponentBackgroundPtr;
        using M = const ModelParameters &;
        auto quintessence = [](BG bg, M) { return std::make_shared<const PerturbationModel_Quintessence>(std::move(bg)); };
        auto ede = [](BG bg, M) { return std::make_shared<const PerturbationModel_EDE>(std::move(bg)); };
        static const std::vector<PerturbationModelInfo> models{
            {"lcdm", [](BG bg, M) { return std::make_shared<const PerturbationModel_LCDM>(std::move(bg)); }},
            {"w0wa", [](BG bg, M) { return std::make_shared<const PerturbationModel_w0wa>(std::move(bg)); }},
            {"musigma",
             [](BG bg, M m) {
                 return std::make_shared<const PerturbationModel_MuSigma>(std::move(bg), m.at("mu0"), m.at("Sigma0"));
             }},
            {"dgp", [](BG bg, M) { return std::make_shared<const PerturbationModel_DGP>(std::move(bg)); }},
            {"fr",
             [](BG bg, M m) {
                 return std::make_shared<const PerturbationModel_HuSawicky>(std::move(bg), m.at("fR0"), m.at("fR_n"));
             }},
            {"horndeski",
             [](BG bg, M m) {
                 return std::make_shared<const PerturbationModel_Horndeski>(
                     std::move(bg),
                     make_standard_alphas(
                         m.at("hd_parametrization"), m.at("hd_cK"), m.at("hd_cB"), m.at("hd_cM"), m.at("hd_M2_ini")),
                     m.at("hd_stability_check") != 0.0);
             }},
            {"quintessence", quintessence},
            {"quintessence_inverse_power_law", quintessence},
            {"quintessence_pngb", quintessence},
            {"quintessence_hilltop", quintessence},
            {"ede", ede},
            {"ede_zc", ede},
            {"jbd", [](BG bg, M) { return std::make_shared<const PerturbationModel_BransDicke>(std::move(bg)); }},
            {"coupled_quintessence",
             [](BG bg, M) { return std::make_shared<const PerturbationModel_CoupledQuintessence>(std::move(bg)); }},
            {"cubic_galileon",
             [](BG bg, M) { return std::make_shared<const PerturbationModel_CubicGalileon>(std::move(bg)); }},
            {"dark_scattering",
             [](BG bg, M m) {
                 return std::make_shared<const PerturbationModel_DarkScattering>(
                     std::move(bg), m.at("ds_xi"), m.at("ds_cs2"));
             }},
            {"ide",
             [](BG bg, M m) {
                 return std::make_shared<const PerturbationModel_IDE>(
                     std::move(bg), m.at("ide_cs2"), m.at("ide_stability_check") != 0.0);
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
        throw std::invalid_argument("[Perturbations] No perturbations for the model [" + model + "] (known:" + known +
                                    ")");
    }

    /// The model by name on a background made by BACKGROUND::make_background(model, ..., model_parameters). The
    /// parameters are checked against the background registry (unknown ones are an error, missing ones take its
    /// defaults)
    inline PerturbationModelPtr make_perturbation_model(const std::string & model,
                                                        BACKGROUND::ComponentBackgroundPtr background,
                                                        const ModelParameters & model_parameters = {}) {
        const auto & info = find_perturbation_model(model);
        auto values = BACKGROUND::find_model(model).parameters;
        for (const auto & [key, value] : model_parameters) {
            if (not values.count(key))
                throw std::invalid_argument("[Perturbations] Model [" + model + "] has no parameter [" + key + "]");
            values[key] = value;
        }
        return info.build(std::move(background), values);
    }

    /// The model by name with its background. model_parameters: as for BACKGROUND::make_background
    inline PerturbationModelPtr make_perturbation_model(const std::string & model,
                                                        const BACKGROUND::BackgroundParameters & p,
                                                        const ModelParameters & model_parameters = {},
                                                        BACKGROUND::BackgroundRange r = {}) {
        find_perturbation_model(model);
        return make_perturbation_model(model, BACKGROUND::make_background(model, p, model_parameters, r), model_parameters);
    }

} // namespace FML::COSMOLOGY::PERTURBATIONS

#endif
