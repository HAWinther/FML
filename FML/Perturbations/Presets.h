#ifndef FML_PERTURBATIONS_PRESETS_H
#define FML_PERTURBATIONS_PRESETS_H

// =====================================================================================================================
// The registry of the perturbation models by name, and make_perturbation_model(name, params, model params). The names
// and the model parameters (with their defaults and checks) are those of the background registry
// (Background/Presets.h make_background), which builds the background the model is built on:
//   lcdm   PerturbationModel_LCDM.h         (make_lcdm_perturbations)
//   jbd    PerturbationModel_BransDicke.h   (make_jbd_perturbations; bd_omega, bd_GeffG_today,
//                                            bd_density_parameter_definition)
// Adding a model: its PerturbationModel_<name>.h (and SpeciesPerturbation_ / PerturbedFieldEquation_ headers) and an
// entry in perturbation_registry().
// =====================================================================================================================

#include <FML/Background/BackgroundParameters.h>
#include <FML/Background/ComponentBackground.h>
#include <FML/Background/Presets.h>
#include <FML/Perturbations/PerturbationModel.h>
#include <FML/Perturbations/PerturbationModel_BransDicke.h>
#include <FML/Perturbations/PerturbationModel_LCDM.h>

#include <functional>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace FML::COSMOLOGY::PERTURBATIONS {

    struct PerturbationModelInfo {
        std::string name;
        /// The model on its background (built by the background registry)
        std::function<PerturbationModelPtr(BACKGROUND::ComponentBackgroundPtr)> build;
    };

    inline const std::vector<PerturbationModelInfo> & perturbation_registry() {
        static const std::vector<PerturbationModelInfo> models{
            {"lcdm",
             [](BACKGROUND::ComponentBackgroundPtr bg) {
                 return std::make_shared<const PerturbationModel_LCDM>(std::move(bg));
             }},
            {"jbd",
             [](BACKGROUND::ComponentBackgroundPtr bg) {
                 return std::make_shared<const PerturbationModel_BransDicke>(std::move(bg));
             }},
        };
        return models;
    }

    /// The model by name with its background. model_parameters: as for BACKGROUND::make_background
    inline PerturbationModelPtr make_perturbation_model(const std::string & model,
                                                        const BACKGROUND::BackgroundParameters & p,
                                                        const std::map<std::string, double> & model_parameters = {},
                                                        BACKGROUND::BackgroundRange r = {}) {
        for (const auto & m : perturbation_registry())
            if (m.name == model)
                return m.build(BACKGROUND::make_background(model, p, model_parameters, r));
        std::string known;
        for (const auto & m : perturbation_registry())
            known += " " + m.name;
        throw std::invalid_argument("[Perturbations] No perturbations for the model [" + model + "] (known:" + known +
                                    ")");
    }

} // namespace FML::COSMOLOGY::PERTURBATIONS

#endif
